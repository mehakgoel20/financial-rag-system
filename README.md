# Financial RAG System (FinQA)

## Overview
This project implements a **production-style Retrieval-Augmented Generation (RAG) system**
for answering numerical financial questions over document-heavy data.

The system is built using **hybrid retrieval (BM25 + dense embeddings)**,
**cross-encoder reranking**, a **standalone vector database (Qdrant)**,
and a **FastAPI backend**, fully **Dockerized using Docker Compose**.

The focus of this project is **retrieval quality, evidence grounding, and system design**,
rather than end-to-end LLM answer generation.

---

## Dataset
- Based on the **FinQA dataset**
- Original data consists of **financial PDFs** (annual reports, filings)
- PDFs are preprocessed into **structured JSON**, containing:
  - extracted text
  - extracted tables
  - question–answer pairs
- This project operates on the structured JSON representation, simulating
  a real-world document ingestion pipeline.

---

## Architecture

### 1. Ingestion
- Documents are chunked into text-based and table-based units

### 2. Retrieval
- **BM25** for lexical matching
- **Dense retrieval** using sentence-transformer embeddings
- Results are merged to maximize recall

### 3. Reranking
- A **cross-encoder reranker** scores retrieved chunks
- Top-k evidence is selected for precision

### 4. Vector Database
- Uses a **standalone Qdrant service**
- Deployed via Docker Compose
- Enables safe concurrent access and production-style retrieval

### 5. API Layer
- FastAPI backend exposing a `/query` endpoint
- Lazy-loaded models for stable startup
- Query-level caching to reduce recomputation

### 6. Deployment
- Fully Dockerized
- One-command startup using Docker Compose

---

## API Usage

Start the system:
```bash
docker-compose up --build


Open Swagger UI:
http://127.0.0.1:8000/docs


Example request:
{
  "question": "what is the average payment volume per transaction for american express?"
}


Evaluation:

Retrieval Recall

Recall@10 ≈ 0.40
FinQA is table-heavy and noisy, with multiple equivalent evidence rows
Retrieval recall saturates without symbolic reasoning

Evidence Quality

Retrieved chunks consistently contain the numeric operands
required to compute the correct answer
Exact-match evidence accuracy is low by design, as answers require arithmetic reasoning

Answer Accuracy

Exact numeric answer accuracy is not the focus of this project
FinQA requires symbolic program execution, which is listed as future work

Limitations

Does not execute symbolic arithmetic programs
In-memory query cache is single-node
Evaluation metrics for arithmetic QA require careful interpretation


Future Work

Symbolic execution of FinQA programs
Redis-based distributed caching
LLM-based answer generation on top of grounded evidence
Cloud deployment


Key Takeaways

Built a full end-to-end RAG system, not a demo
Emphasized retrieval quality, grounding, and system design
Designed with production constraints in mind