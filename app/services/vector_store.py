import faiss
import os
import pickle
from typing import List, Dict


class FaissVectorStore:
    """
    Stores and retrieves embeddings using FAISS.
    """

    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

        if os.path.exists(index_path):
            self._load()

    def add(self, embeddings: List[List[float]], metadatas: List[Dict]):
        self.index.add(faiss.vector_to_array(embeddings).reshape(len(embeddings), self.dim))
        self.metadata.extend(metadatas)
        self._save()

    def search(self, query_embedding, top_k: int = 5):
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
