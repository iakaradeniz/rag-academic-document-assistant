from fastapi import APIRouter, UploadFile, File
from app.services.pdf_loader import PDFLoader
from app.services.chunker import TextChunker
from app.services.embedder import TextEmbedder
from app.services.vector_store import FaissVectorStore
from app.config import settings
import os
import shutil

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    # 1️⃣ PDF'yi diske kaydet
    os.makedirs("data/raw_docs", exist_ok=True)
    file_path = f"data/raw_docs/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2️⃣ PDF → text
    loader = PDFLoader(file_path)
    pages = loader.load()

    # 3️⃣ Chunking
    chunker = TextChunker()
    chunks = chunker.chunk(pages)  

    # 4️⃣ Embedding
    embedder = TextEmbedder(settings.EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)

    # 5️⃣ Vector Store
    vector_store = FaissVectorStore(
    index_path=settings.FAISS_INDEX_PATH,
    dim=384
)
    metadatas = [
        {
            "text": c["text"],
            "source": c["metadata"].get("source"),
            "page": c["metadata"].get("page"),
        }
        for c in chunks
    ]
    vector_store.add(embeddings, metadatas)

    return {
        "message": f"Indexed {len(chunks)} chunks from {file.filename}"
    }