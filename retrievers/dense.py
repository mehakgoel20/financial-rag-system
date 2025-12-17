import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List, Dict


COLLECTION_NAME = "finqa_chunks"


class DenseRetriever:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = SentenceTransformer("intfloat/e5-base-v2")

        self.client = QdrantClient(
    host="qdrant",
    port=6333
)

        self._load_data()
        self._setup_collection()
        self._index_data()

    def _load_data(self):
        with open(self.data_path, "r") as f:
            raw_chunks = json.load(f)

    # 🔥 FILTER NOISY CHUNKS
        self.chunks = [
            c for c in raw_chunks
            if len(c["text"].strip()) > 30 and len(c["text"].split()) > 5
    ]


    def _setup_collection(self):
        if COLLECTION_NAME not in [
            c.name for c in self.client.get_collections().collections
        ]:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )

    def _index_data(self):
        points = []

        for idx, chunk in enumerate(self.chunks):
            text = f"query: {chunk['question']} passage: {chunk['text']}"
            vector = self.model.encode(text).tolist()

            points.append(
                PointStruct(
                    id=idx,
                    vector=vector,
                    payload=chunk
                )
            )

            # batch insert (safe for memory)
            if len(points) == 256:
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                points = []

        if points:
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

    def search(self, query: str, top_k: int = 5):
        query_vector = self.model.encode(
            f"query: {query}"
        ).tolist()

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        ).points

        return [
        {
            **hit.payload,
            "dense_score": hit.score
        }
        for hit in results
    ]



if __name__ == "__main__":
    retriever = DenseRetriever("data/processed/finqa_chunks.json")

    query = "average payment volume per transaction for american express"
    results = retriever.search(query, top_k=5)

    print("\n🔍 Dense Retrieval Results:")
    for r in results:
        print(f"- {r['chunk_id']} | score={round(r['dense_score'], 4)}")
        print(f"  {r['text']}\n")
