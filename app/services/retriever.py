from typing import List, Dict
import numpy as np

from app.services.embedder import TextEmbedder
from app.services.vector_store import FaissVectorStore


class Retriever:
    """
    Retrieves the most relevant chunks for a given query.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        vector_store: FaissVectorStore,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> List[Dict]:
        # 1️⃣ Query embedding
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        # 2️⃣ FAISS search
        scores, indices = self.vector_store.index.search(
            query_embedding, self.top_k
        )

        results = []

        # 3️⃣ Threshold filtering
        for score, idx in zip(scores[0], indices[0]):
            if score < self.score_threshold:
                continue

            metadata = self.vector_store.metadata[idx]

            results.append(
                {
                    "score": float(score),
                    "text": metadata["text"],
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                }
            )

        return results
