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
    
    # 
    # def retrieve_multiple(self, queries, top_k=20):
    #     all_results = {}

    #     per_query_k = max(5, top_k // len(queries))

    #     for q in queries:
    #         bm25_hits = self.bm25.retrieve(q, top_k=per_query_k)
    #         dense_hits = self.dense.retrieve(q, top_k=per_query_k)

    #         for hit in bm25_hits + dense_hits:
    #             cid = hit["chunk_id"]
    #             if cid not in all_results:
    #                 all_results[cid] = hit
    #             else:
    #                 all_results[cid]["score"] = max(
    #                     all_results[cid]["score"], hit["score"]
    #                 )

        # return list(all_results.values())
    def retrieve_multiple(self, queries, per_query_k=5):
        all_results = {}

        for q in queries:
            bm25_hits = self.bm25.retrieve(q, top_k=per_query_k)
            dense_hits = self.dense.retrieve(q, top_k=per_query_k)

            for hit in bm25_hits:
                cid = hit["chunk_id"]
                score = hit.get("bm25_score", 0.0)

                if cid not in all_results:
                    all_results[cid] = hit
                    all_results[cid]["score"] = score
                else:
                    all_results[cid]["score"] = max(all_results[cid]["score"], score)
            for hit in dense_hits:
                cid = hit["chunk_id"]
                score = hit.get("dense_score", 0.0)

                if cid not in all_results:
                    all_results[cid] = hit
                    all_results[cid]["score"] = score
                else:
                    all_results[cid]["score"] = max(all_results[cid]["score"], score)
        return list(all_results.values())




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
