# app/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── 1. INFRAESTRUCTURA ────────────────────────────────────────────────────────
DATABASE_URL   = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Modelos IA
LLM_MODEL_LOCAL   = os.getenv("LLM_MODEL_LOCAL",   "llama3.1:8b")
EMBED_MODEL_LOCAL = os.getenv("EMBED_MODEL_LOCAL",  "nomic-embed-text:v1.5")

LLM_MODEL_CLOUD   = os.getenv("LLM_MODEL_CLOUD",   "gemini-2.5-flash")
EMBED_MODEL_CLOUD = os.getenv("EMBED_MODEL_CLOUD",  "gemini-embedding-001")

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
# Esto garantiza que una respuesta de Gemini nunca contamine las métricas del modo local
VECTOR_STORE_CACHE_LL = Path(os.getenv("VECTOR_STORE_CACHE_LOCAL_LOCAL", "./cache_ll"))  # local+local
VECTOR_STORE_CACHE_LC = Path(os.getenv("VECTOR_STORE_CACHE_LOCAL_CLOUD", "./cache_lc"))  # local+cloud
VECTOR_STORE_CACHE_CC = Path(os.getenv("VECTOR_STORE_CACHE_CLOUD_CLOUD", "./cache_cc"))  # cloud+cloud

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

# ─── 3. UMBRALES DE RELEVANCIA (divididos por motor de vectores) ───────────────
# nomic-embed-text:v1.5 produce scores más bajos → umbral más permisivo
# gemini-embedding-001 produce scores más altos  → umbral más estricto
UMBRAL_RELEVANCIA_LOCAL = 0.15
UMBRAL_RELEVANCIA_CLOUD = 0.30

def get_umbral_relevancia(motor_vectores: str) -> float:
    return UMBRAL_RELEVANCIA_LOCAL if motor_vectores == "local" else UMBRAL_RELEVANCIA_CLOUD

# ─── 4. UMBRALES DE CACHÉ (divididos por modo completo) ───────────────────────
# El embedding local produce scores más bajos → umbral de caché más permisivo
CACHE_THRESHOLD_LL = 0.82
CACHE_THRESHOLD_LC = 0.83
CACHE_THRESHOLD_CC = 0.88

_CACHE_THRESHOLD_MAP: dict[tuple[str, str], float] = {
    ("local", "local"):  CACHE_THRESHOLD_LL,
    ("local", "cloud"):  CACHE_THRESHOLD_LC,
    ("cloud", "cloud"):  CACHE_THRESHOLD_CC,
}

def get_cache_threshold(motor_vectores: str, motor_llm: str) -> float:
    return _CACHE_THRESHOLD_MAP.get((motor_vectores, motor_llm), 0.85)

# Umbral de similitud mínima para considerar dos preguntas "la misma" en caché
UMBRAL_SIMILITUD = 0.02

# ─── 5. PARÁMETROS K DE RECUPERACIÓN ──────────────────────────────────────────
# Con semantic chunking los fragmentos son precisos; no se necesita recuperar
# más de los que se van a usar.
RAG_K_LOCAL = 10
RAG_K_CLOUD = 8

def get_k_retrieval(motor_vectores: str) -> int:
    return RAG_K_LOCAL if motor_vectores == "local" else RAG_K_CLOUD

# ─── 6. LÍMITE DE TOKENS EN LA RESPUESTA (divididos por motor LLM) ────────────
# Cloud: más tokens porque Gemini es rápido y los truncamientos son el problema principal.
# Local: menos tokens para controlar la latencia de generación de llama3.1:8b.
NUM_TOKENS_NORMAL_LOCAL = 350
NUM_TOKENS_LISTA_LOCAL  = 750
NUM_TOKENS_NORMAL_CLOUD = 700
NUM_TOKENS_LISTA_CLOUD  = 1400

def get_num_tokens(motor_llm: str, es_lista_larga: bool) -> int:
    if motor_llm == "cloud":
        return NUM_TOKENS_LISTA_CLOUD if es_lista_larga else NUM_TOKENS_NORMAL_CLOUD
    return NUM_TOKENS_LISTA_LOCAL if es_lista_larga else NUM_TOKENS_NORMAL_LOCAL

# ─── 7. RESTRICCIONES DE DOCUMENTOS ───────────────────────────────────────────
MAX_DOCUMENTOS         = 10
LIMITE_TAMANO_MB       = 5
LIMITE_TAMANO_BYTES    = LIMITE_TAMANO_MB * 1024 * 1024
EXTENSIONES_PERMITIDAS = {".pdf", ".docx", ".txt", ".md"}