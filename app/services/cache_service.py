# app/services/cache_service.py
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import (
    get_cache_dir, get_cache_threshold,
    EMBED_MODEL_LOCAL, EMBED_MODEL_CLOUD, GOOGLE_API_KEY,
    UMBRAL_SIMILITUD,
)

# ─────────────────────────────────────────────────────────────────────────────
# MAPA COMPLETO DE COMBINACIONES DE CACHÉ
# cache_ll → local:local  (vectores Ollama + LLM Ollama)
# cache_lc → local:cloud  (vectores Ollama + LLM Gemini)
# cache_cc → cloud:cloud  (vectores Gemini + LLM Gemini)
# ─────────────────────────────────────────────────────────────────────────────
_TODAS_LAS_COMBIS: list[tuple[str, str]] = [
    ("local", "local"),
    ("local", "cloud"),
    ("cloud", "cloud"),
]

# Qué cachés afecta cada motor_vectores al borrar
_COMBIS_POR_MOTOR_VECTORES: dict[str, list[tuple[str, str]]] = {
    "local": [("local", "local"), ("local", "cloud")],  # cache_ll y cache_lc
    "cloud": [("cloud", "cloud")],                       # solo cache_cc
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _get_embeddings(motor_vectores: str):
    """Devuelve el modelo de embeddings según el motor de vectores activo."""
    if motor_vectores == "cloud":
        return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL_CLOUD, google_api_key=GOOGLE_API_KEY)
    return OllamaEmbeddings(model=EMBED_MODEL_LOCAL)


def _get_cache_db(motor_vectores: str, motor_llm: str) -> Chroma:
    """Instancia la BD de caché correcta para la combinación de motores activa."""
    cache_dir = get_cache_dir(motor_vectores, motor_llm)
    embeddings = _get_embeddings(motor_vectores)
    return Chroma(
        persist_directory=str(cache_dir),
        embedding_function=embeddings,
        collection_name="cache_respuestas",
    )


def _es_respuesta_cacheable(respuesta: str) -> bool:
    """
    Valida que una respuesta sea digna de ser cacheada.
    Evita propagar respuestas truncadas, errores o rechazos.
    """
    if not respuesta or len(respuesta.strip()) < 20:
        return False
    texto = respuesta.lower().strip()
    if "no existe en mi base de datos" in texto:
        return False
    if texto.startswith("lo siento") and len(respuesta) < 120:
        return False
    if respuesta.strip()[-1] not in ".!?»":
        return False
    return True


def _limpiar_un_cache(motor_vectores: str, motor_llm: str) -> str:
    """
    Borra completamente un caché específico (una combinación motor_vectores:motor_llm).
    Devuelve un string con el resultado para componer mensajes multi-caché.
    """
    cache_dir = get_cache_dir(motor_vectores, motor_llm)
    if not os.path.exists(str(cache_dir)):
        return f"caché {motor_vectores}:{motor_llm} ya estaba vacío"
    try:
        cache_db = _get_cache_db(motor_vectores, motor_llm)
        cache_db.delete_collection()
        print(f"[CACHÉ] 🗑️ Caché {motor_vectores}:{motor_llm} limpiado completamente.")
        return f"caché {motor_vectores}:{motor_llm} limpiado"
    except Exception as e:
        print(f"[CACHÉ] Error al limpiar {motor_vectores}:{motor_llm}: {e}")
        return f"error en {motor_vectores}:{motor_llm}: {e}"


def _limpiar_doc_en_un_cache(nombre_coleccion: str, motor_vectores: str, motor_llm: str) -> str:
    """
    Elimina del caché específico todas las entradas de un documento.
    Devuelve string con resultado para componer mensajes.
    """
    cache_dir = get_cache_dir(motor_vectores, motor_llm)
    if not os.path.exists(str(cache_dir)):
        return f"{motor_vectores}:{motor_llm} vacío"
    try:
        cache_db = _get_cache_db(motor_vectores, motor_llm)
        if cache_db._collection.count() == 0:
            return f"{motor_vectores}:{motor_llm} vacío"

        todos = cache_db._collection.get(include=["metadatas"])
        ids_a_eliminar = [
            todos["ids"][i]
            for i, meta in enumerate(todos["metadatas"])
            if meta.get("documento_origen") == nombre_coleccion
        ]
        if ids_a_eliminar:
            cache_db._collection.delete(ids=ids_a_eliminar)
            print(f"[CACHÉ] 🗑️ '{nombre_coleccion}' eliminado de {motor_vectores}:{motor_llm} ({len(ids_a_eliminar)} entradas).")
            return f"{motor_vectores}:{motor_llm} limpiado ({len(ids_a_eliminar)} entradas)"
        return f"{motor_vectores}:{motor_llm} sin entradas del documento"
    except Exception as e:
        return f"error en {motor_vectores}:{motor_llm}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────────────────

def buscar_en_cache(pregunta: str, motor_vectores: str, motor_llm: str) -> str | None:
    """
    Busca en el caché semántico de la combinación de motores activa.
    Devuelve la respuesta cacheada si existe y supera el umbral, o None.
    """
    cache_dir = get_cache_dir(motor_vectores, motor_llm)
    if not os.path.exists(str(cache_dir)):
        return None

    try:
        cache_db = _get_cache_db(motor_vectores, motor_llm)
        if cache_db._collection.count() == 0:
            return None

        threshold = get_cache_threshold(motor_vectores, motor_llm)
        resultados = cache_db.similarity_search_with_relevance_scores(pregunta, k=1)
        if not resultados:
            return None

        doc, score = resultados[0]
        if score >= (1 - UMBRAL_SIMILITUD):
            print(f"[CACHÉ] ✅ Hit semántico (score={score:.3f}, umbral={threshold:.2f}, modo={motor_vectores}:{motor_llm})")
            return doc.metadata.get("respuesta")

        return None
    except Exception as e:
        print(f"[CACHÉ] Error en búsqueda ({motor_vectores}:{motor_llm}): {e}")
        return None


def guardar_en_cache(
    pregunta: str,
    respuesta: str,
    documento_origen: str = "desconocido",
    motor_vectores: str = "local",
    motor_llm: str = "local",
) -> None:
    """
    Guarda una respuesta en el caché del modo activo.
    Solo cachea respuestas válidas (no truncadas, no rechazos).
    """
    if not _es_respuesta_cacheable(respuesta):
        print(f"[CACHÉ] ⚠️ Respuesta no cacheada (inválida, truncada o rechazo).")
        return

    try:
        cache_db = _get_cache_db(motor_vectores, motor_llm)
        cache_db.add_documents([
            Document(
                page_content=pregunta,
                metadata={
                    "respuesta":         respuesta,
                    "documento_origen":  documento_origen,
                    "motor_vectores":    motor_vectores,
                    "motor_llm":         motor_llm,
                },
            )
        ])
        print(f"[CACHÉ] 💾 Respuesta guardada (modo={motor_vectores}:{motor_llm})")
    except Exception as e:
        print(f"[CACHÉ] Error al guardar ({motor_vectores}:{motor_llm}): {e}")


def limpiar_cache_por_documento(
    nombre_coleccion: str,
    motor_vectores: str = "local",
) -> dict:
    """
    Elimina del caché todas las entradas asociadas a un documento específico,
    en TODOS los cachés que usan ese motor_vectores.

    - motor_vectores="local" → limpia cache_ll (local:local) y cache_lc (local:cloud)
    - motor_vectores="cloud" → limpia cache_cc (cloud:cloud)

    Esto es correcto porque un documento vectorizado en "local" puede tener
    respuestas cacheadas tanto en modo local:local como en modo local:cloud.
    """
    combis = _COMBIS_POR_MOTOR_VECTORES.get(motor_vectores, [])
    resultados = [
        _limpiar_doc_en_un_cache(nombre_coleccion, mv, ml)
        for mv, ml in combis
    ]
    return {"mensaje": f"Caché del documento '{nombre_coleccion}' procesado: {' | '.join(resultados)}"}


def limpiar_cache(
    motor_vectores: str = "local",
    motor_llm: str = "local",
    motor: str | None = None,
    scope: str = "auto",
) -> dict:
    """
    Borrado total del caché. Admite tres modos de uso:

    1. Por motor completo (shorthand desde documents.py):
         limpiar_cache(motor="local")   → limpia cache_ll y cache_lc
         limpiar_cache(motor="cloud")   → limpia cache_cc
         limpiar_cache(motor="all")     → limpia los 3 cachés

    2. Por combinación exacta (forma explícita):
         limpiar_cache(motor_vectores="local", motor_llm="cloud")  → solo cache_lc

    3. Por scope global:
         limpiar_cache(scope="all")     → limpia los 3 cachés

    Regla de expansión cuando se usa motor= shorthand:
        "local" → limpia cache_ll + cache_lc  (todo lo que usa vectores locales)
        "cloud" → limpia cache_cc             (todo lo que usa vectores cloud)
        "all"   → limpia los 3 cachés
    """
    # Scope global explícito
    if scope == "all" or motor == "all":
        resultados = [_limpiar_un_cache(mv, ml) for mv, ml in _TODAS_LAS_COMBIS]
        return {"mensaje": f"Todos los cachés limpiados: {' | '.join(resultados)}"}

    # Shorthand motor= expande por motor_vectores
    if motor is not None:
        combis = _COMBIS_POR_MOTOR_VECTORES.get(motor, [(motor, motor)])
        resultados = [_limpiar_un_cache(mv, ml) for mv, ml in combis]
        return {"mensaje": " | ".join(resultados)}

    # Forma explícita: combinación exacta
    resultado = _limpiar_un_cache(motor_vectores, motor_llm)
    return {"mensaje": resultado}