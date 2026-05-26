import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tesis_rag.db")

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

SECRET_KEY                  = os.getenv("SECRET_KEY", "cambia-esta-clave-en-produccion")
ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

DOCUMENTS_DIR    = Path(os.getenv("DOCUMENTS_DIR",    "./documents_local"))
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "./vector_store_local"))
CACHE_DIR        = Path(os.getenv("CACHE_DIR",        "./cache_ll"))

for _d in [DOCUMENTS_DIR, VECTOR_STORE_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

UMBRAL_RELEVANCIA    = float(os.getenv("UMBRAL_RELEVANCIA",    "0.15"))
RAG_K                = int(os.getenv("RAG_K",                  "10"))
BREAKPOINT_THRESHOLD = int(os.getenv("BREAKPOINT_THRESHOLD",   "75"))

NUM_TOKENS_NORMAL = int(os.getenv("NUM_TOKENS_NORMAL", "350"))
NUM_TOKENS_LISTA  = int(os.getenv("NUM_TOKENS_LISTA",  "750"))
HYDE_NUM_PREDICT  = int(os.getenv("HYDE_NUM_PREDICT",  "120"))

CACHE_L2_UMBRAL_SIMILITUD = float(os.getenv("CACHE_L2_UMBRAL_SIMILITUD", "0.02"))
CACHE_L2_THRESHOLD        = float(os.getenv("CACHE_L2_THRESHOLD",        "0.82"))

CACHE_L1_MAX_ENTRIES = int(os.getenv("CACHE_L1_MAX_ENTRIES", "500"))

LLM_TEMPERATURE    = 0
LLM_TOP_K          = int(os.getenv("LLM_TOP_K",           "10"))
LLM_TOP_P          = float(os.getenv("LLM_TOP_P",          "0.5"))
LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", "1.3"))
LLM_STOP_WORDS     = ["Consulta del usuario:", "Usuario:", "Pregunta:", "[FIN]"]

MAX_DOCUMENTOS         = 10
LIMITE_TAMANO_MB       = 5
LIMITE_TAMANO_BYTES    = LIMITE_TAMANO_MB * 1024 * 1024
EXTENSIONES_PERMITIDAS = {".pdf", ".docx", ".txt", ".md"}

PROMPT_PRINCIPAL = """Eres el Asistente Académico de la EPN. Respondes ÚNICAMENTE con el dato solicitado, sin explicaciones, sin citar el contexto, sin markdown, sin negritas.

IDIOMA: Responde SIEMPRE en español.

REGLAS:
1. Usa SOLO información del CONTEXTO. Nunca inventes datos.
2. Si la información no está en el CONTEXTO, responde exactamente: "Lo siento, esa información no existe en mi base de datos oficial."
3. NUNCA uses frases como "Según el contexto", "El documento dice", "La respuesta es".
4. NUNCA uses negritas, bullets, markdown ni formato especial.
5. Si la respuesta es un número o dato simple, escríbelo solo. Nada más.
6. Si el usuario pregunta por una materia o código que NO aparece en el CONTEXTO, usa la regla 2.

EJEMPLOS DE RESPUESTA CORRECTA:
Pregunta: ¿Cuántos semestres tiene la carrera?
Respuesta: 9 semestres.

Pregunta: ¿En qué nivel se dicta Álgebra Lineal?
Respuesta: Nivel 1, primer semestre.

Pregunta: ¿Cuántos créditos tiene Programación I?
Respuesta: 4 créditos.

Pregunta: ¿Cuáles son los prerrequisitos de Inteligencia Artificial?
Respuesta: ICCD412 y ICCD442.

CONTEXTO:
{contexto}

Pregunta del usuario: {pregunta}
Respuesta:"""

PROMPT_HYDE = (
    "Escribe una respuesta corta, factual y directa en español a esta pregunta "
    "sobre la malla curricular de la Carrera de Ciencias de la Computación de la EPN. "
    "Usa términos académicos concretos. Máximo 3 oraciones. "
    "No expliques, solo responde con datos.\n\n"
    "Pregunta: {pregunta}\nRespuesta:"
)