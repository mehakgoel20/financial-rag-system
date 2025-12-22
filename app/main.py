import time
import json
import os
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from retrievers.hybrid_rerank import HybridReranker
from app.query_processor.intent import detect_intent
from app.query_processor.rewrite import rewrite_query
from app.memory.conversation import ConversationState
from app.cache import get_cache, set_cache
from app.guardrails import should_refuse
from app.metric import log_metric
from app.math.fact_extractor import extract_facts
from app.math.executor import execute_math

# ------------------------------------------------------------------
# Environment & logging
# ------------------------------------------------------------------

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------

app = FastAPI(
    title="Production GenAI RAG",
    description="Intent-aware, production-grade RAG system",
    version="3.1"
)

conversation_state = ConversationState()
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = HybridReranker()
    return pipeline


@app.get("/")
def root():
    return {"status": "RAG API running"}


# ------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class ChunkResponse(BaseModel):
    chunk_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    intent: str
    top_chunks: List[ChunkResponse]
    answer: Optional[str]


# ------------------------------------------------------------------
# Extractive answering (SAFE, journalized)
# ------------------------------------------------------------------

def extractive_answer(question: str, chunks: List[Dict]) -> Optional[str]:
    q = question.lower()

    for c in chunks:
        text = c["text"].lower()

        if "transactions" in q and "transactions" in text:
            return f"According to the document, {c['text']}"

        if "cards" in q and "cards" in text:
            return f"According to the document, {c['text']}"

        if "company" in q and "american express" in text:
            return "The document discusses American Express."

    return None


# ------------------------------------------------------------------
# Main RAG endpointx
# ------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    start_time = time.time()
    question = request.question

    # 1️⃣ Memory
    enriched_question = conversation_state.enrich(question)

    # 2️⃣ Intent
    intent_obj = detect_intent(enriched_question)
    intent_type = intent_obj["type"]
    intent_subtype = intent_obj.get("subtype")

    # 3️⃣ Rewrite
    rewritten_queries = rewrite_query(enriched_question, intent_type)

    # 4️⃣ Cache
    cache_key = f"{intent_type}:{rewritten_queries}"
    cached = get_cache(cache_key)
    if cached:
        log_metric(
            cache_hit=True,
            latency_ms=(time.time() - start_time) * 1000,
            used_llm=False
        )
        return cached

    # 5️⃣ Retrieve + rerank
    rag_pipeline = get_pipeline()
    retrieved = rag_pipeline.retrieve_multiple(rewritten_queries)
    reranked = rag_pipeline.rerank(enriched_question, retrieved, top_k=3)

    # 6️⃣ Arithmetic (symbolic, safe)
    if intent_type == "arithmetic":
        facts, missing = extract_facts(intent_subtype, reranked)

        if missing:
            answer = f"Missing required facts: {missing}"
        else:
            result, error = execute_math(intent_subtype, facts)
            answer = json.dumps(result, indent=2) if not error else error

        response = {
            "question": question,
            "intent": intent_type,
            "top_chunks": [],
            "answer": answer
        }

        set_cache(cache_key, response)
        log_metric(
            cache_hit=False,
            latency_ms=(time.time() - start_time) * 1000,
            used_llm=False,
            symbolic=True
        )
        return response

    # 7️⃣ Descriptive (extractive BEFORE refusal)
    answer = extractive_answer(question, reranked)

    if answer is None:
        if should_refuse(intent_type, reranked):
            answer = "Insufficient evidence to answer reliably."
        else:
            answer = "No explicit statement found, but relevant context exists."

    response = {
        "question": question,
        "intent": intent_type,
        "top_chunks": [
            {
                "chunk_id": str(c["chunk_id"]),
                "text": str(c["text"]),
                "score": float(c["score"]),
            }
            for c in reranked
        ],
        "answer": answer
    }

    set_cache(cache_key, response)
    log_metric(
        cache_hit=False,
        latency_ms=(time.time() - start_time) * 1000,
        used_llm=False,
        symbolic=False
    )

    return response
