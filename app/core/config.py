# app/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL   = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
QDRANT_URL    = os.getenv("QDRANT_URL")
REDIS_URL     = os.getenv("REDIS_URL")

QDRANT_COLLECTION_LOCAL = os.getenv("QDRANT_COLLECTION_LOCAL")

LLM_MODEL_LOCAL   = os.getenv("LLM_MODEL_LOCAL")
EMBED_MODEL_LOCAL = os.getenv("EMBED_MODEL_LOCAL")

LLM_MODEL_CLOUD   = os.getenv("LLM_MODEL_CLOUD")

SECRET_KEY                  = os.getenv("SECRET_KEY")
ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

EMBED_DIMENSION_LOCAL = int(os.getenv("EMBED_DIMENSION_LOCAL", "384"))

DOCUMENTS_DIR_LOCAL = Path(os.getenv("DOCUMENTS_DIR_LOCAL"))
DOCUMENTS_DIR_LOCAL.mkdir(parents=True, exist_ok=True)

UMBRAL_RELEVANCIA_LOCAL = 0.15

RAG_K_LOCAL = 10

MAX_DOCUMENTOS         = 10
LIMITE_TAMANO_MB       = 5
LIMITE_TAMANO_BYTES    = LIMITE_TAMANO_MB * 1024 * 1024
EXTENSIONES_PERMITIDAS = {".pdf", ".docx", ".txt", ".md"}