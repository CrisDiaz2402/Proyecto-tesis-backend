# app/core/config.py
"""
Configuración de infraestructura y constantes globales.

IMPORTANTE — Parámetros RAG:
  Los valores de UMBRAL_RELEVANCIA_*, RAG_K_* que aparecen aquí son
  DEFAULTS DE FALLBACK únicamente.
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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Servicios de infraestructura
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
QDRANT_URL    = os.getenv("QDRANT_URL", "http://localhost:6333")
REDIS_URL     = os.getenv("REDIS_URL", "redis://localhost:6379")

# Colecciones Qdrant
QDRANT_COLLECTION_LOCAL = os.getenv("QDRANT_COLLECTION_LOCAL", "documentos_local")
QDRANT_COLLECTION_CLOUD = os.getenv("QDRANT_COLLECTION_CLOUD", "documentos_cloud")

# Modelos IA
LLM_MODEL_LOCAL   = os.getenv("LLM_MODEL_LOCAL")
EMBED_MODEL_LOCAL = os.getenv("EMBED_MODEL_LOCAL")

LLM_MODEL_CLOUD   = os.getenv("LLM_MODEL_CLOUD")
EMBED_MODEL_CLOUD = os.getenv("EMBED_MODEL_CLOUD")

# Seguridad JWT
SECRET_KEY                  = os.getenv("SECRET_KEY")
ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

# Dimensión de embeddings (debe coincidir con el modelo usado)
EMBED_DIMENSION_LOCAL = int(os.getenv("EMBED_DIMENSION_LOCAL", "384"))
EMBED_DIMENSION_CLOUD = int(os.getenv("EMBED_DIMENSION_CLOUD", "768"))

# ─── 2. RUTAS DE ALMACENAMIENTO ────────────────────────────────────────────────

# Documentos físicos
DOCUMENTS_DIR_LOCAL = Path(os.getenv("DOCUMENTS_DIR_LOCAL", "./documents_local"))
DOCUMENTS_DIR_CLOUD = Path(os.getenv("DOCUMENTS_DIR_CLOUD", "./documents_cloud"))

# Crear directorios necesarios
for _folder in [DOCUMENTS_DIR_LOCAL, DOCUMENTS_DIR_CLOUD]:
    _folder.mkdir(parents=True, exist_ok=True)


# ─── 3–6. DEFAULTS DE FALLBACK (no usar directamente en rag_service) ──────────
# Estos valores se usan SOLO como fallback en rag_params_service.get_params()
# si la BD no está disponible. En operación normal, la BD es la fuente de verdad.

UMBRAL_RELEVANCIA_LOCAL = 0.15
UMBRAL_RELEVANCIA_CLOUD = 0.30

RAG_K_LOCAL = 10
RAG_K_CLOUD = 8

# ─── 7. RESTRICCIONES DE DOCUMENTOS ───────────────────────────────────────────
MAX_DOCUMENTOS         = 10
LIMITE_TAMANO_MB       = 5
LIMITE_TAMANO_BYTES    = LIMITE_TAMANO_MB * 1024 * 1024
EXTENSIONES_PERMITIDAS = {".pdf", ".docx", ".txt", ".md"}