from fastapi import FastAPI
from pydantic import BaseModel

from retrievers.hybrid_rerank import HybridReranker

app = FastAPI(
    title="Financial RAG System",
    description="Hybrid Retrieval + Reranking for FinQA-style questions",
    version="1.0"
)

@app.get("/")
def root():
    return {"status": "RAG API running"}

# ✅ Lazy-loaded pipeline
pipeline = None
# Simple in-memory cache
query_cache = {}

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = HybridReranker()
    return pipeline



class QueryRequest(BaseModel):
    question: str


class ChunkResponse(BaseModel):
    chunk_id: str
    text: str
    score: float


class QueryResponse(BaseModel):
    question: str
    top_chunks: list[ChunkResponse]


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    question = request.question

    # 🔹 Cache hit
    if question in query_cache:
        return query_cache[question]

    # 🔹 Cache miss
    rag_pipeline = get_pipeline()
    retrieved = rag_pipeline.retrieve(question)
    reranked = rag_pipeline.rerank(question, retrieved, top_k=3)

    results = [
        {
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "score": round(c["rerank_score"], 3)
        }
        for c in reranked
    ]

    response = {
        "question": question,
        "top_chunks": results
    }

    # 🔹 Store in cache
    query_cache[question] = response

    return response
