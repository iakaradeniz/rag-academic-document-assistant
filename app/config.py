from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    LLM_MODEL_NAME: str = "mistral"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    EMBEDDING_DIM: int = 384

    # Embedding
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

    # Vector store
    FAISS_INDEX_PATH: str = "data/faiss_index/index.faiss"


    # Document storage
    RAW_DOCS_PATH: str = "data/raw_docs"

    # Retrieval
    TOP_K: int = 5


settings = Settings()

