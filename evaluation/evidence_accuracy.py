import json
from retrievers.hybrid_rerank import HybridReranker


def evaluate_evidence_accuracy(max_samples=50):
    pipeline = HybridReranker()

    with open("data/processed/finqa_chunks.json") as f:
        chunks = json.load(f)

    # map question → gold answer
    questions = {}
    for c in chunks:
        questions[c["question"]] = str(c["answer"])

    questions = list(questions.items())[:max_samples]

    correct = 0
    total = 0

    for question, gold_answer in questions:
        retrieved = pipeline.retrieve(question)
        reranked = pipeline.rerank(question, retrieved, top_k=1)

        top_text = reranked[0]["text"]

        if gold_answer in top_text:
            correct += 1

        total += 1

        print(f"\nQ: {question}")
        print(f"Gold answer: {gold_answer}")
        print(f"Top chunk: {top_text[:120]}...")

    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ Evidence Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    evaluate_evidence_accuracy()
