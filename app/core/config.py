# app/core/config.py
"""
Configuración de infraestructura y constantes globales.

IMPORTANTE — Parámetros RAG:
  Los valores de UMBRAL_RELEVANCIA_*, RAG_K_*, NUM_TOKENS_*, CACHE_THRESHOLD_*
  y UMBRAL_SIMILITUD que aparecen aquí son DEFAULTS DE FALLBACK únicamente.
  La fuente de verdad en runtime es la tabla `configuracion_rag` de la BD,
  gestionada por app/services/rag_params_service.py.
  Estos valores solo se usan si la BD no está disponible en el arranque.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── 1. INFRAESTRUCTURA ────────────────────────────────────────────────────────
DATABASE_URL   = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Modelos IA
LLM_MODEL_LOCAL   = os.getenv("LLM_MODEL_LOCAL")
EMBED_MODEL_LOCAL = os.getenv("EMBED_MODEL_LOCAL")

LLM_MODEL_CLOUD   = os.getenv("LLM_MODEL_CLOUD")
EMBED_MODEL_CLOUD = os.getenv("EMBED_MODEL_CLOUD")

# Seguridad JWT
SECRET_KEY                  = os.getenv("SECRET_KEY")
ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

# ─── 2. RUTAS DE ALMACENAMIENTO ────────────────────────────────────────────────

# Documentos físicos
DOCUMENTS_DIR_LOCAL = Path(os.getenv("DOCUMENTS_DIR_LOCAL", "./documents_local"))
DOCUMENTS_DIR_CLOUD = Path(os.getenv("DOCUMENTS_DIR_CLOUD", "./documents_cloud"))

# Bases de datos vectoriales (ChromaDB)
VECTOR_STORE_DIR_LOCAL = Path(os.getenv("VECTOR_STORE_DIR_LOCAL", "./vector_store_local"))
VECTOR_STORE_DIR_CLOUD = Path(os.getenv("VECTOR_STORE_DIR_CLOUD", "./vector_store_cloud"))

# Cachés semánticos — uno por combinación de motores activos
VECTOR_STORE_CACHE_LL = Path(os.getenv("VECTOR_STORE_CACHE_LOCAL_LOCAL", "./cache_ll"))
VECTOR_STORE_CACHE_LC = Path(os.getenv("VECTOR_STORE_CACHE_LOCAL_CLOUD", "./cache_lc"))
VECTOR_STORE_CACHE_CC = Path(os.getenv("VECTOR_STORE_CACHE_CLOUD_CLOUD", "./cache_cc"))

# Mapa de cachés: (motor_vectores, motor_llm) → directorio
_CACHE_DIR_MAP: dict[tuple[str, str], Path] = {
    ("local", "local"):  VECTOR_STORE_CACHE_LL,
    ("local", "cloud"):  VECTOR_STORE_CACHE_LC,
    ("cloud", "cloud"):  VECTOR_STORE_CACHE_CC,
}


def get_cache_dir(motor_vectores: str, motor_llm: str) -> Path:
    """Devuelve el directorio de caché correcto para la combinación de motores activa."""
    return _CACHE_DIR_MAP.get((motor_vectores, motor_llm), VECTOR_STORE_CACHE_LL)


# Crear todos los directorios necesarios automáticamente
for _folder in [
    DOCUMENTS_DIR_LOCAL, DOCUMENTS_DIR_CLOUD,
    VECTOR_STORE_DIR_LOCAL, VECTOR_STORE_DIR_CLOUD,
    VECTOR_STORE_CACHE_LL, VECTOR_STORE_CACHE_LC, VECTOR_STORE_CACHE_CC,
]:
    _folder.mkdir(parents=True, exist_ok=True)


# ─── 3–6. DEFAULTS DE FALLBACK (no usar directamente en rag_service) ──────────
# Estos valores se usan SOLO como fallback en rag_params_service.get_params()
# si la BD no está disponible. En operación normal, la BD es la fuente de verdad.

UMBRAL_RELEVANCIA_LOCAL = 0.15
UMBRAL_RELEVANCIA_CLOUD = 0.30

CACHE_THRESHOLD_LL = 0.82
CACHE_THRESHOLD_LC = 0.83
CACHE_THRESHOLD_CC = 0.88

UMBRAL_SIMILITUD = 0.02

RAG_K_LOCAL = 10
RAG_K_CLOUD = 8

NUM_TOKENS_NORMAL_LOCAL = 350
NUM_TOKENS_LISTA_LOCAL  = 750
NUM_TOKENS_NORMAL_CLOUD = 700
NUM_TOKENS_LISTA_CLOUD  = 1400

# ─── 7. RESTRICCIONES DE DOCUMENTOS ───────────────────────────────────────────
MAX_DOCUMENTOS         = 10
LIMITE_TAMANO_MB       = 5
LIMITE_TAMANO_BYTES    = LIMITE_TAMANO_MB * 1024 * 1024
EXTENSIONES_PERMITIDAS = {".pdf", ".docx", ".txt", ".md"}