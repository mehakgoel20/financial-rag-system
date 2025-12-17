import json
from sentence_transformers import CrossEncoder
from retrievers.bm25 import BM25Retriever
from retrievers.dense import DenseRetriever

def expand_query(query: str) -> str:
    q = query.lower()

    if "average" in q:
        q += " payments volume total transactions"

    return q

class HybridReranker:
    def __init__(self):
        # Load retrievers
        self.bm25 = BM25Retriever("data/processed/finqa_chunks.json")
        self.dense = DenseRetriever("data/processed/finqa_chunks.json")

        # Cross-encoder = the judge
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def retrieve(self, query, bm25_k=10, dense_k=10):
        expanded_query = expand_query(query)

        bm25_results = self.bm25.search(expanded_query, top_k=bm25_k)
        dense_results = self.dense.search(expanded_query, top_k=dense_k)


        # Merge results (remove duplicates)
        merged = {}
        for chunk in bm25_results + dense_results:
            merged[chunk["chunk_id"]] = chunk
            # key = chunk["text"].strip().lower()
            # merged[key] = chunk

        return list(merged.values())

    def rerank(self, query, chunks, top_k=3): 
        pairs = [(query, c["text"]) for c in chunks] 
        scores = self.reranker.predict(pairs) 
        reranked = sorted( zip(scores, chunks), key=lambda x: x[0], reverse=True ) 
        return [ {**chunk, "rerank_score": score} for score, chunk in reranked[:top_k] ]

if __name__ == "__main__":
    pipeline = HybridReranker()

    query = "what is the average payment volume per transaction for american express"

    print("\n🔎 Step 1: Hybrid Retrieval")
    candidates = pipeline.retrieve(query)

    print(f"Retrieved {len(candidates)} candidates")

    print("\n🎯 Step 2: Cross-Encoder Reranking")
    final_results = pipeline.rerank(query, candidates)

    for r in final_results:
        print(f"\nScore: {round(r['rerank_score'], 4)}")
        print(f"Chunk ID: {r['chunk_id']}")
        print(f"Text: {r['text']}")
