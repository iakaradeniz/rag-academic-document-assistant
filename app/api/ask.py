

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.services.embedder import TextEmbedder
from app.services.vector_store import FaissVectorStore
from app.services.retriever import Retriever
from app.services.llm_client import OllamaClient
from app.services.rag_service import RAGService

router = APIRouter()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    # 1️⃣ Embedder
    embedder = TextEmbedder(
        model_name=settings.EMBEDDING_MODEL_NAME
    )

    # 2️⃣ Vector store
    vector_store = FaissVectorStore(
        index_path=settings.FAISS_INDEX_PATH
    )

    # 3️⃣ Retriever
    retriever = Retriever(
        vector_store=vector_store,
        embedder=embedder,
        top_k=settings.TOP_K
    )

    # 4️⃣ LLM client
    llm_client = OllamaClient(
        model_name=settings.LLM_MODEL_NAME,
        base_url=settings.OLLAMA_BASE_URL
    )

    # 5️⃣ RAG service
    rag_service = RAGService(
        retriever=retriever,
        llm_client=llm_client
    )

    # 6️⃣ Ask
    answer = rag_service.ask(request.question)

    return AskResponse(answer=answer)
