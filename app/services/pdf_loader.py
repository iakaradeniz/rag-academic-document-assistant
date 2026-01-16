from pathlib import Path
from typing import List, Dict

from PyPDF2 import PdfReader


class PDFLoader:
    """
    Loads PDF documents and extracts text page by page
    with metadata (source file, page number).
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

    def load(self) -> List[Dict]:
        reader = PdfReader(self.file_path)
        documents = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if not text:
                continue

            documents.append(
                {
                    "text": text.strip(),
                    "metadata": {
                        "source": self.file_path.name,
                        "page": page_number,
                    },
                }
            )

        return documents
