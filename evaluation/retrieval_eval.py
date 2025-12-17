import json
from retrievers.hybrid_rerank import HybridReranker


def evaluate_recall_at_k(k=5, max_samples=10):
    pipeline = HybridReranker()

    with open("data/processed/finqa_chunks.json") as f:
        chunks = json.load(f)

    # group chunks by question
    questions = {}
    for c in chunks:
        q = c["question"]
        questions.setdefault(q, []).append(c)

    questions = list(questions.items())[:max_samples]

    hits = 0

    for question, gold_chunks in questions:
        gold_ids = {
            c["chunk_id"] for c in gold_chunks if c["is_relevant"]
        }

        retrieved = pipeline.retrieve(question)
        reranked = pipeline.rerank(question, retrieved, top_k=k)

        retrieved_ids = {c["chunk_id"] for c in reranked}

        if gold_ids & retrieved_ids:
            hits += 1

    recall = hits / len(questions)
    print(f"Recall@{k}: {recall:.2f}")


if __name__ == "__main__":
    # evaluate_recall_at_k(k=5)
    evaluate_recall_at_k(k=10)
    # evaluate_recall_at_k(k=20)

