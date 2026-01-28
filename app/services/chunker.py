import re
from typing import List, Dict


class TextChunker:
    """
    Smart text chunker:
    - Paragraph-aware
    - Sentence-safe
    - Overlap without breaking meaning
    """

    def __init__(
        self,
        max_chunk_size: int = 800,
        sentence_overlap: int = 1,
    ):
        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap

    def chunk(self, documents: List[Dict]) -> List[Dict]:
        chunks = []

        for doc in documents:
            text = self._clean_text(doc["text"])
            metadata = doc["metadata"]

            paragraphs = self._split_paragraphs(text)

            current_chunk = ""
            current_sentences = []

            for para in paragraphs:
                if len(para) > self.max_chunk_size:
                    sentences = self._split_sentences(para)
                else:
                    sentences = [para]

                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                        current_chunk += sentence + " "
                        current_sentences.append(sentence)
                    else:
                        chunks.append(
                            {
                                "text": current_chunk.strip(),
                                "metadata": metadata,
                            }
                        )

                        # overlap: last N sentences
                        overlap_sentences = current_sentences[-self.sentence_overlap :]
                        current_chunk = " ".join(overlap_sentences) + " " + sentence + " "
                        current_sentences = overlap_sentences + [sentence]

            if current_chunk.strip():
                chunks.append(
                    {
                        "text": current_chunk.strip(),
                        "metadata": metadata,
                    }
                )

        return chunks

    # ---------- helpers ----------

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    def _split_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
