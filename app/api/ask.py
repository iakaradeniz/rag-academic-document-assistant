from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import numpy as np
import requests

from app.services.embedder import TextEmbedder
from app.services.vector_store import FaissVectorStore
from app.services.context_cleaner import ContextCleaner
from app.config import settings

router = APIRouter(prefix="/ask", tags=["ask"])


# ===== Request / Response Schemas =====

class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    contexts: List[str]


# ===== Endpoint =====

@router.post("/", response_model=AskResponse)
async def ask_question(payload: AskRequest):
    """
    1. Embed the question
    2. Search FAISS
    3. Clean contexts
    4. Send context to Ollama (Mistral)
    5. Return answer + contexts
    """

    question = payload.question

    # 1️⃣ Embed question
    embedder = TextEmbedder(settings.EMBEDDING_MODEL_NAME)
    query_embedding = embedder.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    # 2️⃣ Load vector store & search
    vector_store = FaissVectorStore(
        index_path=settings.FAISS_INDEX_PATH,
        dim=settings.EMBEDDING_DIM,
    )

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=payload.top_k,
    )

    # 3️⃣ Extract & clean contexts (limit for prompt safety)
    raw_contexts = [r["text"] for r in results[:3]]
    contexts = [
        ContextCleaner.clean(c)[:800]
        for c in raw_contexts
    ]

    # 4️⃣ Build prompt
    context_text = "\n\n".join(
        f"- {c}" for c in contexts
    )

    prompt = f"""
Aşağıda bazı akademik doküman parçaları verilmiştir.
Sadece bu metinleri kullanarak soruyu cevapla.
Eğer cevap metinlerde yoksa açıkça "Bu bilgi dokümanlarda yok." de.

### Dokümanlar:
{context_text}

### Soru:
{question}

### Cevap:
"""

    # 5️⃣ Ollama call (Mistral)
    response = requests.post(
        f"{settings.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": settings.LLM_MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=300  # Mistral yavaş olabilir
    )

    response.raise_for_status()

    answer = response.json().get("response", "").strip()

    return AskResponse(
        question=question,
        answer=answer,
        contexts=contexts,
    )
