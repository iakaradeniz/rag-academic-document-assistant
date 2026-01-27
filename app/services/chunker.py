from typing import List, Dict


class TextChunker:
    """
    Splits page-level documents into overlapping chunks
    while preserving metadata.
    """

    def __init__(self, chunk_size: int = 800, overlap: float = 0.15):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, documents: List[Dict]) -> List[Dict]:
        chunks = []
        overlap_size = int(self.chunk_size * self.overlap)

        for doc in documents:
            text = doc["text"]
            metadata = doc["metadata"]

            start = 0
            text_length = len(text)

            while start < text_length:
                end = start + self.chunk_size
                chunk_text = text[start:end]

                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": metadata,
                    }
                )

                start = end - overlap_size

        return chunks
