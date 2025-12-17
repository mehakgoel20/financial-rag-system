# from typing import List
# import openai


# def build_prompt(question: str, contexts: List[str]) -> str:
#     context_block = "\n\n".join(contexts)

#     prompt = f"""
# You are a factual assistant.

# Use ONLY the information provided in the context below to answer the question.
# Do NOT use outside knowledge.
# If the answer cannot be determined from the context, say "I don't know".

# Context:
# {context_block}

# Question:
# {question}

# Answer:
# """
#     return prompt


# def generate_answer(question: str, top_chunks: List[dict]) -> str:
#     contexts = [chunk["text"] for chunk in top_chunks]

#     prompt = build_prompt(question, contexts)

#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0
#     )

#     return response["choices"][0]["message"]["content"].strip()



# generator/llm_answer.py

from typing import List
from openai import OpenAI

client = OpenAI()


def build_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(contexts)

    prompt = f"""
You are a factual assistant.

Use ONLY the information provided in the context below to answer the question.
Do NOT use outside knowledge.
If the answer cannot be determined from the context, say "I don't know".

Context:
{context_block}

Question:
{question}

Answer:
"""
    return prompt


def generate_answer(question: str, top_chunks: List[dict]) -> str:
    contexts = [chunk["text"] for chunk in top_chunks]
    prompt = build_prompt(question, contexts)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()
