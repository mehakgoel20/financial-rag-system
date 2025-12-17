# ingestion/prepare_finqa.py

import json
from typing import List, Dict


def table_to_text(table: List[List[str]]) -> List[Dict]:
    """
    Converts a FinQA table into row-wise text chunks
    """
    header = table[0]
    rows = table[1:]

    table_chunks = []
    for row_id, row in enumerate(rows):
        parts = []
        for h, v in zip(header, row):
            parts.append(f"{h} is {v}")
        table_chunks.append({
            "row_id": row_id,
            "text": " ; ".join(parts)
        })
    return table_chunks
    


def build_chunks(item: Dict) -> List[Dict]:
    """
    Converts ONE FinQA item into multiple RAG chunks
    """
    chunks = []

    question = item["qa"]["question"]
    answer = item["qa"]["answer"]
    gold_inds = item["qa"].get("gold_inds", {})
    base_id = item["id"]

    # -------- pre_text --------
    for i, text in enumerate(item.get("pre_text", [])):
        chunks.append({
            "chunk_id": f"{base_id}_pre_{i}",
            "source": "pre_text",
            "text": text,
            "question": question,
            "answer": answer,
            "is_relevant": False
        })

    # -------- post_text --------
    for i, text in enumerate(item.get("post_text", [])):
        chunks.append({
            "chunk_id": f"{base_id}_post_{i}",
            "source": "post_text",
            "text": text,
            "question": question,
            "answer": answer,
            "is_relevant": False
        })

    # -------- table --------
    table_chunks = table_to_text(item["table"])
    for t in table_chunks:
        chunk_key = f"table_{t['row_id']}"
        is_rel = chunk_key in gold_inds

        chunks.append({
            "chunk_id": f"{base_id}_{chunk_key}",
            "source": "table",
            "text": t["text"],
            "question": question,
            "answer": answer,
            "is_relevant": is_rel
        })

    return chunks


def main():
    INPUT_PATH = "data/raw/finqa.json"
    OUTPUT_PATH = "data/processed/finqa_chunks.json"

    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    all_chunks = []
    for item in data:
        all_chunks.extend(build_chunks(item))

    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"✅ Saved {len(all_chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
