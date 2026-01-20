from typing import List, Dict
from app.services.embedder import TextEmbedder
from app.services.vector_store import FaissVectorStore


class EmbeddingPipeline:
    def __init__(self, embedder: TextEmbedder, vector_store: FaissVectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def run(self, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]

        metadatas = [
            {
                "text": c["text"],          
                "source": c["metadata"].get("source"),
                "page": c["metadata"].get("page"),
            }
            for c in chunks
        ]

        embeddings = self.embedder.encode(texts)
        self.vector_store.add(embeddings, metadatas)
