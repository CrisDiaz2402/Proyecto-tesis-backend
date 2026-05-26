import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import engine
from app.db import models
from app.api.routers import documents, chat, usuarios, auth, monitor

models.Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.rag_service import precalentar
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, precalentar)
    yield


app = FastAPI(
    title="API Asistente RAG — EPN Local",
    version="4.0.0",
    lifespan=lifespan,
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(usuarios.router)
app.include_router(monitor.router)


@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status":  "ok",
        "version": "4.0.0",
        "motor":   "ollama-gpu + hf-cpu-embeddings",
    }