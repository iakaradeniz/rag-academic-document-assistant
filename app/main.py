from fastapi import FastAPI

from app.api.ask import router as ask_router
from app.api.upload import router as upload_router

app = FastAPI(title="Academic RAG Assistant")

app.include_router(upload_router)
app.include_router(ask_router)



