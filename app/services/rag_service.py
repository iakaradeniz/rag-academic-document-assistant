from app.services.retriever import Retriever
from app.services.prompt_builder import PromptBuilder
from app.services.llm_client import OllamaClient


class RAGService:
    """
    Orchestrates the full RAG pipeline.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: OllamaClient,
    ):
        # Bağımlılıkları DI ile alıyoruz
        self.retriever = retriever
        self.prompt_builder = PromptBuilder()
        self.llm_client = llm_client

    def ask(self, question: str) -> str:
        # 1️⃣ En alakalı chunk'ları bul
        contexts = self.retriever.retrieve(question)

        # 2️⃣ Hiç context yoksa
        if not contexts:
            return "I could not find the answer in the provided documents."

        # 3️⃣ Prompt oluştur
        prompt = self.prompt_builder.build(
            question=question,
            contexts=contexts
        )

        # 4️⃣ LLM çağır
        answer = self.llm_client.generate(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
        )

        return answer
