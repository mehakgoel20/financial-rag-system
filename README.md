# Financial RAG System (FinQA)

# Production-Grade Financial RAG System (FinQA)

## Overview
This project is a production-grade Retrieval-Augmented Generation (RAG) system
built on financial documents (FinQA dataset). The system is designed to be
**hallucination-safe, cost-efficient, and observable**.

Unlike naive RAG systems, this architecture:
- Avoids LLM hallucination for arithmetic questions
- Uses symbolic execution for financial calculations
- Employs Redis caching to reduce latency and cost
- Logs real system metrics (latency, cache hits, cost savings)

---

## Architecture

Query Flow:
1. User Query
2. Conversation Memory (multi-turn)
3. Intent Detection (descriptive vs arithmetic)
4. Query Rewriting
5. Redis Cache Lookup
6. Hybrid Retrieval (Dense + BM25 via Qdrant)
7. Reranking
8. Guardrails (refuse if evidence insufficient)
9. Answer Generation
   - Arithmetic → Symbolic Execution (no LLM)
   - Descriptive → Extractive Answering
10. Metrics Logging

---

## Key Design Decisions

### 1. Hallucination Prevention
- Arithmetic questions are **never answered by an LLM**
- All numeric answers are symbolically computed
- If required facts are missing → system refuses safely

### 2. Cost Control
- Redis caching prevents repeated computation
- LLM usage rate intentionally kept near zero
- Cost savings tracked per request

### 3. Observability
Each request logs:
- Latency
- Cache hit/miss
- Symbolic execution usage
- LLM usage

Metrics are aggregated offline using a metrics script.

---

## System Metrics (Local Evaluation)

Total Requests : 14
Cache Hit Rate : 50%
LLM Usage Rate : 0%
Symbolic Exec Rate : 21.43%
Avg Latency (ms) : 9273
P95 Latency (ms) : 160870

> Note: High P95 is due to cold-start effects (model loading, vector warm-up).

## Cost Estimation

Estimated LLM Cost : $0.00
Estimated Cost Saved : $0.42 (linear scaling)


At scale (100k queries/day), this architecture saves thousands of dollars per month.

---

## Tech Stack
- FastAPI
- Qdrant (Vector DB)
- Redis (Caching)
- SentenceTransformers
- Docker & Docker Compose

---

## Why This Project Matters
This project demonstrates **real-world GenAI system design**:
- Safe by default
- Cost-aware
- Observable
- Production-minded

It prioritizes correctness over flashy demos.
