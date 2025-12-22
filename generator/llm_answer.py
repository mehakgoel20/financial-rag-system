from typing import List
from openai import OpenAI

client = OpenAI()


SYSTEM_PROMPT = """
You are a financial assistant.

Rules you MUST follow:
- Use ONLY the provided evidence.
- DO NOT use outside knowledge.
- DO NOT invent numbers.
- DO NOT perform calculations or arithmetic.
- Only restate values explicitly present in the evidence.

- If the answer cannot be determined from the evidence, respond exactly with:
  "Insufficient evidence to answer reliably."
"""


def build_user_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(
        [f"Evidence {i+1}: {text}" for i, text in enumerate(contexts)]
    )

    return f"""
Evidence:
{context_block}

Question:
{question}

Answer:
"""


def generate_answer(question: str, top_chunks: List[dict]) -> str:
    contexts = [chunk["text"] for chunk in top_chunks]

    if not contexts:
        return "Insufficient evidence to answer reliably."

    # Arithmetic questions should NOT be answered by LLM
    if any(word in question.lower() for word in ["average", "ratio", "difference", "change"]):
        return "Insufficient evidence to answer reliably."

    ...

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    temperature=0.0,
    max_tokens=150
)

    

    return response.choices[0].message.content.strip()
