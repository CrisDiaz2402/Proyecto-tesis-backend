# app/services/cache_service.py
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import (
    get_cache_dir,
    EMBED_MODEL_LOCAL, EMBED_MODEL_CLOUD, GOOGLE_API_KEY,
)
# umbral_similitud ya NO viene de config.py — se lee dinámicamente desde la BD
from app.services.rag_params_service import get_params

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

_COMBIS_POR_MOTOR_VECTORES: dict[str, list[tuple[str, str]]] = {
    "local": [("local", "local"), ("local", "cloud")],
    "cloud": [("cloud", "cloud")],
}


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON OllamaEmbeddings para el caché semántico local
#
# Problema original: _get_embeddings() creaba OllamaEmbeddings(model=...) en
# cada llamada a buscar_en_cache() y guardar_en_cache(), añadiendo overhead de
# inicialización incluso en los cache hits (que deben ser ultrarrápidos).
#
# NOTA: No importamos el singleton de rag_service.py para evitar importación
# circular (rag_service importa cache_service). Mantenemos un singleton propio
# que coexiste sin problema — ambos apuntan al mismo proceso Ollama.
# ─────────────────────────────────────────────────────────────────────────────
_ollama_embeddings_cache: OllamaEmbeddings | None = None


def _get_ollama_embeddings_local() -> OllamaEmbeddings:
    """Retorna la instancia singleton de OllamaEmbeddings para el caché local.
    Se crea solo en la primera llamada y se reutiliza indefinidamente."""
    global _ollama_embeddings_cache
    if _ollama_embeddings_cache is None:
        _ollama_embeddings_cache = OllamaEmbeddings(model=EMBED_MODEL_LOCAL)
        print(f"[CACHÉ] ✅ OllamaEmbeddings inicializado (singleton, model={EMBED_MODEL_LOCAL})")
    return _ollama_embeddings_cache


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _get_embeddings(motor_vectores: str):
    """Retorna embeddings según el motor.
    Para local reutiliza el singleton; para cloud crea instancia nueva
    (Google no mantiene estado local, es stateless por diseño)."""
    if motor_vectores == "cloud":
        return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL_CLOUD, google_api_key=GOOGLE_API_KEY)
    return _get_ollama_embeddings_local()


def _get_cache_db(motor_vectores: str, motor_llm: str) -> Chroma:
    cache_dir  = get_cache_dir(motor_vectores, motor_llm)
    embeddings = _get_embeddings(motor_vectores)
    return Chroma(
        persist_directory=str(cache_dir),
        embedding_function=embeddings,
        collection_name="cache_respuestas",
    )


def _es_respuesta_cacheable(respuesta: str) -> bool:
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
    umbral_similitud se lee dinámicamente desde get_params() en cada llamada.
    """
    cache_dir = get_cache_dir(motor_vectores, motor_llm)
    if not os.path.exists(str(cache_dir)):
        return None

    try:
        cache_db = _get_cache_db(motor_vectores, motor_llm)
        if cache_db._collection.count() == 0:
            return None

        # Leer umbral dinámico desde la BD
        umbral_similitud = get_params().get("umbral_similitud", 0.02)

        resultados = cache_db.similarity_search_with_relevance_scores(pregunta, k=1)
        if not resultados:
            return None

        doc, score = resultados[0]
        if score >= (1 - umbral_similitud):
            print(
                f"[CACHÉ] ✅ Hit semántico "
                f"(score={score:.3f}, umbral_sim={umbral_similitud:.3f}, "
                f"modo={motor_vectores}:{motor_llm})"
            )
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
    if not _es_respuesta_cacheable(respuesta):
        print(f"[CACHÉ] ⚠️ Respuesta no cacheada (inválida, truncada o rechazo).")
        return

    try:
        cache_db = _get_cache_db(motor_vectores, motor_llm)
        cache_db.add_documents([
            Document(
                page_content=pregunta,
                metadata={
                    "respuesta":        respuesta,
                    "documento_origen": documento_origen,
                    "motor_vectores":   motor_vectores,
                    "motor_llm":        motor_llm,
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
    combis    = _COMBIS_POR_MOTOR_VECTORES.get(motor_vectores, [])
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

    1. Scope global:   limpiar_cache(scope="all")         → limpia los 3 cachés
    2. Shorthand:      limpiar_cache(motor="local")        → cache_ll + cache_lc
                       limpiar_cache(motor="cloud")        → cache_cc
                       limpiar_cache(motor="all")          → los 3 cachés
    3. Combinación:    limpiar_cache(motor_vectores="local", motor_llm="cloud") → cache_lc
    """
    if scope == "all" or motor == "all":
        resultados = [_limpiar_un_cache(mv, ml) for mv, ml in _TODAS_LAS_COMBIS]
        return {"mensaje": f"Todos los cachés limpiados: {' | '.join(resultados)}"}

    if motor is not None:
        combis     = _COMBIS_POR_MOTOR_VECTORES.get(motor, [(motor, motor)])
        resultados = [_limpiar_un_cache(mv, ml) for mv, ml in combis]
        return {"mensaje": " | ".join(resultados)}

    resultado = _limpiar_un_cache(motor_vectores, motor_llm)
    return {"mensaje": resultado}