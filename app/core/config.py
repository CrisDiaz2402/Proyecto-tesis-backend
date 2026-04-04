# app/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar las variables de infraestructura desde el archivo .env
load_dotenv()

# ─── 1. INFRAESTRUCTURA (Leído del .env) ───────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama3.2:3b")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "nomic-embed-text")

VECTOR_STORE_DIR   = os.getenv("VECTOR_STORE_DIR", "./vector_store")
VECTOR_STORE_CACHE = os.getenv("VECTOR_STORE_CACHE", "./vector_store_cache")

DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "./documents"))
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── SEGURIDAD JWT ─────────────────────────────────────────────────────────────
# SECRET_KEY se lee del .env. NUNCA hardcodear en el código.
SECRET_KEY                 = os.getenv("SECRET_KEY")
ALGORITHM                  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))  # 8 horas por defecto


# ─── 2. REGLAS DE NEGOCIO Y RAG (Única Fuente de Verdad - Hardcodeado) ─────────
# Estos valores dictan cómo piensa la IA y las reglas estrictas del sistema.
# No deben estar en el .env porque son parte de la lógica central de la tesis.

# Parámetros del Motor de Búsqueda y Caché
UMBRAL_RELEVANCIA = 0.20
UMBRAL_SIMILITUD  = 0.02
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200
RAG_K_RETRIEVAL   = 8
RAG_K_FINAL       = 6

# Restricciones estrictas del módulo de Documentos
MAX_DOCUMENTOS         = 10
LIMITE_TAMANO_MB       = 5
LIMITE_TAMANO_BYTES    = LIMITE_TAMANO_MB * 1024 * 1024
EXTENSIONES_PERMITIDAS = {".pdf", ".docx", ".txt", ".md"}