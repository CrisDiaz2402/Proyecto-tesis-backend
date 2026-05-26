from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.core.config import (
    CACHE_DIR,
    EMBED_MODEL,
    CACHE_L2_UMBRAL_SIMILITUD,
    CACHE_L2_THRESHOLD,
)

_embeddings_cache: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_cache


def _get_cache_db() -> Chroma:
    return Chroma(
        persist_directory=str(CACHE_DIR),
        embedding_function=_get_embeddings(),
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


def buscar_en_cache(pregunta: str) -> str | None:
    if not CACHE_DIR.exists():
        return None
    try:
        cache_db = _get_cache_db()
        if cache_db._collection.count() == 0:
            return None
        resultados = cache_db.similarity_search_with_relevance_scores(pregunta, k=1)
        if not resultados:
            return None
        doc, score = resultados[0]
        if score >= CACHE_L2_THRESHOLD - CACHE_L2_UMBRAL_SIMILITUD:
            return doc.metadata.get("respuesta")
    except Exception:
        pass
    return None


def guardar_en_cache(
    pregunta: str,
    respuesta: str,
    documento_origen: str = "desconocido",
) -> None:
    if not _es_respuesta_cacheable(respuesta):
        return
    try:
        cache_db = _get_cache_db()
        doc = Document(
            page_content=pregunta,
            metadata={"respuesta": respuesta, "documento_origen": documento_origen},
        )
        cache_db.add_documents([doc])
    except Exception:
        pass


def limpiar_cache() -> dict:
    if not CACHE_DIR.exists():
        return {"mensaje": "El caché ya estaba vacío."}
    try:
        _get_cache_db().delete_collection()
        return {"mensaje": "Caché L2 limpiado correctamente."}
    except Exception as e:
        return {"mensaje": str(e)}


def limpiar_cache_por_documento(nombre_coleccion: str) -> dict:
    if not CACHE_DIR.exists():
        return {"mensaje": "Caché vacío."}
    try:
        cache_db = _get_cache_db()
        if cache_db._collection.count() == 0:
            return {"mensaje": "Caché vacío."}
        todos          = cache_db._collection.get(include=["metadatas"])
        ids_a_eliminar = [
            todos["ids"][i]
            for i, meta in enumerate(todos["metadatas"])
            if meta.get("documento_origen") == nombre_coleccion
        ]
        if ids_a_eliminar:
            cache_db._collection.delete(ids=ids_a_eliminar)
            return {"mensaje": f"'{nombre_coleccion}' eliminado del caché."}
        return {"mensaje": f"No había entradas de '{nombre_coleccion}' en el caché."}
    except Exception as e:
        return {"mensaje": str(e)}
