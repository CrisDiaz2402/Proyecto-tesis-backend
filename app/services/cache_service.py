# app/services/cache_service.py
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from app.core.config import VECTOR_STORE_CACHE, EMBED_MODEL, UMBRAL_SIMILITUD

def _get_cache_db() -> Chroma:
    """Función auxiliar para instanciar la conexión a la BD del caché sin repetir código."""
    return Chroma(
        persist_directory=VECTOR_STORE_CACHE,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        collection_name="cache_respuestas"
    )

def buscar_en_cache(pregunta: str) -> str | None:
    if not os.path.exists(VECTOR_STORE_CACHE): return None
    try:
        cache_db = _get_cache_db()
        if cache_db._collection.count() == 0: return None

        resultados = cache_db.similarity_search_with_relevance_scores(pregunta, k=1)
        if not resultados: return None

        doc, score = resultados[0]
        
        # Validamos contra el umbral global de config.py
        if score >= (1 - UMBRAL_SIMILITUD):
            return doc.metadata.get("respuesta")
            
        return None
    except Exception as e:
        print(f"[CACHÉ] Error en búsqueda: {e}")
        return None

def guardar_en_cache(pregunta: str, respuesta: str, documento_origen: str = "desconocido") -> None:
    try:
        cache_db = _get_cache_db()
        cache_db.add_documents([
            Document(
                page_content=pregunta,
                metadata={"respuesta": respuesta, "documento_origen": documento_origen}
            )
        ])
    except Exception as e:
        print(f"[CACHÉ] Error al guardar memoria: {e}")

def limpiar_cache_por_documento(nombre_coleccion: str) -> dict:
    if not os.path.exists(VECTOR_STORE_CACHE): return {"mensaje": "El caché ya estaba vacío."}
    try:
        cache_db = _get_cache_db()
        if cache_db._collection.count() == 0: return {"mensaje": "El caché ya estaba vacío."}

        todos = cache_db._collection.get(include=["metadatas"])
        ids_a_eliminar = [
            todos["ids"][i] for i, meta in enumerate(todos["metadatas"])
            if meta.get("documento_origen") == nombre_coleccion
        ]

        if ids_a_eliminar:
            cache_db._collection.delete(ids=ids_a_eliminar)
            print(f"[CACHÉ] 🗑️ Entradas asociadas a '{nombre_coleccion}' eliminadas.")
            return {"mensaje": "Caché de documento limpiado."}
            
        return {"mensaje": "No había caché para limpiar."}
    except Exception as e:
        return {"mensaje": f"Error: {e}"}

def limpiar_cache() -> dict:
    """Borrado total de la memoria caché temporal."""
    if not os.path.exists(VECTOR_STORE_CACHE): return {"mensaje": "El caché ya estaba vacío."}
    try:
        cache_db = _get_cache_db()
        cache_db.delete_collection()
        print("[CACHÉ] 🗑️ Limpieza profunda de memoria temporal completada.")
        return {"mensaje": "Caché temporal limpiado."}
    except Exception as e:
        print(f"[CACHÉ] Error al limpiar: {e}")
        return {"mensaje": str(e)}