# retrievers/bm25.py

import json
from rank_bm25 import BM25Okapi
from typing import List, Dict


class BM25Retriever:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

        self._load_data()
        self._build_index()

    def _load_data(self):
        with open(self.data_path, "r") as f:
            raw_chunks = json.load(f)

    # 🔥 FILTER NOISY CHUNKS
        self.chunks = [
            c for c in raw_chunks
            if len(c["text"].strip()) > 30 and len(c["text"].split()) > 5
    ]

        self.tokenized_corpus = [
            chunk["text"].lower().split()
            for chunk in self.chunks
    ]


    def _build_index(self):
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["bm25_score"] = scores[idx]
            results.append(chunk)

        return results
    def retrieve(self, query: str, top_k: int):
        return self.search(query, top_k)



if __name__ == "__main__":
    retriever = BM25Retriever("data/processed/finqa_chunks.json")

    sample_query = "average payment volume per transaction for american express"
    results = retriever.search(sample_query, top_k=5)

    print("\n🔍 BM25 Results:")
    for r in results:
        print(f"- {r['chunk_id']} | score={round(r['bm25_score'], 2)}")
        print(f"  {r['text']}\n")
