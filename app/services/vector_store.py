import faiss
import os
import pickle
import numpy as np
from typing import List, Dict


class FaissVectorStore:
    """
    Stores and retrieves embeddings using FAISS.
    """

    def __init__(self, index_path: str, dim: int = 384):
        self.index_path = index_path
        self.dim = dim

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        if os.path.exists(index_path):
            self._load()
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.metadata: List[Dict] = []

    def add(self, embeddings: List[List[float]], metadatas: List[Dict]):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.metadata.extend(metadatas)
        self._save()

    def search(self, query_embedding, top_k: int = 5):
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])

        return results


    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".meta", "wb") as f:
            pickle.dump(self.metadata, f)

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".meta", "rb") as f:
            self.metadata = pickle.load(f)
