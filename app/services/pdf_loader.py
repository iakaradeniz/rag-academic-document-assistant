from pathlib import Path
from typing import List, Dict
import pdfplumber
import re


class PDFLoader:
    """
    Loads PDF documents using pdfplumber and extracts clean text
    page by page with metadata.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

    def _clean_text(self, text: str) -> str:
        """
        Fix common PDF extraction issues:
        - broken newlines
        - extra spaces
        - hyphenated line breaks
        """
        if not text:
            return ""

        # Satır sonu tirelerini birleştir
        text = re.sub(r"-\n", "", text)

        # Satır sonlarını boşluk yap
        text = re.sub(r"\n+", " ", text)

        # Fazla boşlukları temizle
        text = re.sub(r"\s{2,}", " ", text)

        return text.strip()

    def load(self) -> List[Dict]:
        documents = []

        with pdfplumber.open(self.file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(layout=True)

                if not text:
                    continue

                clean_text = self._clean_text(text)

                if len(clean_text) < 50:
                    continue  # çok kısa / anlamsız sayfaları at

                documents.append(
                    {
                        "text": clean_text,
                        "metadata": {
                            "source": self.file_path.name,
                            "page": page_number,
                        },
                    }
                )

        return documents
