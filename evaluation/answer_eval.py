# import json
# from retrievers.hybrid_rerank import HybridReranker
# from generator.llm_answer import generate_answer


# def evaluate_numeric_accuracy(max_samples=50):
#     pipeline = HybridReranker()

#     with open("data/processed/finqa_chunks.json") as f:
#         chunks = json.load(f)

#     questions = {}
#     for c in chunks:
#         q = c["question"]
#         questions.setdefault(q, c["answer"])

#     questions = list(questions.items())[:max_samples]

#     correct = 0

#     for question, gold_answer in questions:
#         retrieved = pipeline.retrieve(question)
#         reranked = pipeline.rerank(question, retrieved, top_k=2)

#         answer = generate_answer(question, reranked)

#         try:
#             if abs(float(answer) - float(gold_answer)) < 0.01:
#                 correct += 1
#         except:
#             pass

#     accuracy = correct / len(questions)
#     print(f"Answer Accuracy: {accuracy:.2f}")


# if __name__ == "__main__":
#     evaluate_numeric_accuracy()

import json
import re
from retrievers.hybrid_rerank import HybridReranker
from generator.llm_answer import generate_answer


def extract_number(text):
    """
    Extract first number from LLM output.
    """
    match = re.search(r"[-+]?\d*\.\d+|\d+", text)
    if match:
        return float(match.group())
    return None


def evaluate_answer_accuracy(max_samples=30, tolerance=0.01):
    pipeline = HybridReranker()

    with open("data/processed/finqa_chunks.json") as f:
        chunks = json.load(f)

    # unique questions with gold answers
    questions = {}
    for c in chunks:
        questions[c["question"]] = c["answer"]

    questions = list(questions.items())[:max_samples]

    correct = 0
    total = 0

    for question, gold_answer in questions:
        retrieved = pipeline.retrieve(question)
        reranked = pipeline.rerank(question, retrieved, top_k=2)

        predicted_text = generate_answer(question, reranked)
        predicted_value = extract_number(predicted_text)

        try:
            gold_value = float(gold_answer)
        except:
            continue

        if predicted_value is not None:
            if abs(predicted_value - gold_value) <= tolerance:
                correct += 1

        total += 1

        print(f"\nQ: {question}")
        print(f"Gold: {gold_value}")
        print(f"Predicted: {predicted_value}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n✅ Answer Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    evaluate_answer_accuracy()
