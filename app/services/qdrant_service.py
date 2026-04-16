# app/services/qdrant_service.py
"""
Servicio de vector store basado en Qdrant.
Reemplaza pgvector para almacenar y buscar embeddings de documentos.

Colecciones:
  - documentos_local  (384 dims, sentence-transformers)
  - documentos_cloud  (768 dims, Gemini)
"""

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import (
    QDRANT_URL,
    QDRANT_COLLECTION_LOCAL,
    QDRANT_COLLECTION_CLOUD,
    EMBED_DIMENSION_LOCAL,
    EMBED_DIMENSION_CLOUD,
)

# ─── Cliente singleton ───────────────────────────────────────────────────────
_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """Devuelve el cliente Qdrant (singleton)."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
        print(f"[QDRANT] ✅ Cliente conectado a {QDRANT_URL}")
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _nombre_coleccion(motor: str) -> str:
    """Retorna el nombre de colección Qdrant según el motor."""
    if motor == "cloud":
        return QDRANT_COLLECTION_CLOUD
    return QDRANT_COLLECTION_LOCAL


def _dimension(motor: str) -> int:
    """Retorna la dimensión de embedding según el motor."""
    if motor == "cloud":
        return EMBED_DIMENSION_CLOUD
    return EMBED_DIMENSION_LOCAL


# ─────────────────────────────────────────────────────────────────────────────
# GESTIÓN DE COLECCIONES
# ─────────────────────────────────────────────────────────────────────────────

def crear_coleccion(motor: str) -> None:
    """Crea la colección en Qdrant si no existe."""
    client = _get_client()
    nombre = _nombre_coleccion(motor)
    dim = _dimension(motor)

    colecciones = [c.name for c in client.get_collections().collections]
    if nombre not in colecciones:
        client.create_collection(
            collection_name=nombre,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"[QDRANT] ✅ Colección '{nombre}' creada ({dim} dims)")
    else:
        print(f"[QDRANT] Colección '{nombre}' ya existe.")


def eliminar_coleccion_qdrant(motor: str) -> dict:
    """Elimina por completo una colección de Qdrant."""
    client = _get_client()
    nombre = _nombre_coleccion(motor)
    try:
        client.delete_collection(collection_name=nombre)
        print(f"[QDRANT] 🗑️ Colección '{nombre}' eliminada.")
        return {"mensaje": f"Colección '{nombre}' eliminada."}
    except Exception as e:
        print(f"[QDRANT] Error al eliminar colección '{nombre}': {e}")
        return {"mensaje": str(e)}


def listar_colecciones() -> list[str]:
    """Lista todas las colecciones existentes en Qdrant."""
    client = _get_client()
    return [c.name for c in client.get_collections().collections]


# ─────────────────────────────────────────────────────────────────────────────
# INSERTAR PUNTOS
# ─────────────────────────────────────────────────────────────────────────────

def insertar_puntos(
    motor: str,
    embeddings: list[list[float]],
    payloads: list[dict],
) -> int:
    """
    Inserta puntos (vectores + payload) en la colección del motor indicado.
    Crea la colección automáticamente si no existe.
    Retorna la cantidad de puntos insertados.
    """
    client = _get_client()
    nombre = _nombre_coleccion(motor)

    # Asegurar que la colección existe
    crear_coleccion(motor)

    # Obtener el siguiente ID base (offset basado en el count actual)
    info = client.get_collection(collection_name=nombre)
    base_id = info.points_count

    points = [
        PointStruct(
            id=base_id + i,
            vector=emb,
            payload=payload,
        )
        for i, (emb, payload) in enumerate(zip(embeddings, payloads))
    ]

    client.upsert(collection_name=nombre, points=points)
    print(f"[QDRANT] ✅ {len(points)} puntos insertados en '{nombre}'")
    return len(points)


# ─────────────────────────────────────────────────────────────────────────────
# ELIMINAR PUNTOS POR DOCUMENT_NAME (hot-swap)
# ─────────────────────────────────────────────────────────────────────────────

def eliminar_puntos_por_documento(nombre_documento: str, motor: str) -> dict:
    """
    Elimina todos los puntos cuyo payload contiene document_name == nombre_documento.
    Permite hot-swap de documentos sin recrear la colección.
    """
    client = _get_client()
    nombre = _nombre_coleccion(motor)

    try:
        client.delete(
            collection_name=nombre,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_name",
                        match=MatchValue(value=nombre_documento),
                    )
                ]
            ),
        )
        print(f"[QDRANT] 🗑️ Puntos de '{nombre_documento}' eliminados de '{nombre}'")
        return {"mensaje": f"Documento '{nombre_documento}' eliminado de '{nombre}'."}
    except Exception as e:
        print(f"[QDRANT] Error al eliminar puntos de '{nombre_documento}': {e}")
        return {"mensaje": str(e)}


def eliminar_todos_los_puntos(motor: str) -> dict:
    """Elimina todos los puntos de una colección (recreándola vacía)."""
    client = _get_client()
    nombre = _nombre_coleccion(motor)
    dim = _dimension(motor)

    try:
        colecciones = [c.name for c in client.get_collections().collections]
        if nombre in colecciones:
            client.delete_collection(collection_name=nombre)
        client.create_collection(
            collection_name=nombre,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"[QDRANT] 🗑️ Todos los puntos eliminados de '{nombre}' (recreada).")
        return {"mensaje": f"Todos los vectores de '{nombre}' eliminados."}
    except Exception as e:
        print(f"[QDRANT] Error al vaciar '{nombre}': {e}")
        return {"mensaje": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# BÚSQUEDA DE SIMILITUD
# ─────────────────────────────────────────────────────────────────────────────

def buscar_similares(
    query_embedding: list[float],
    motor: str,
    k: int = 10,
    umbral: float = 0.15,
    filtro_documento: Optional[str] = None,
) -> list[dict]:
    """
    Búsqueda de similitud coseno en Qdrant con filtro de umbral.
    Retorna lista de dicts con contenido, metadata, score.
    """
    client = _get_client()
    nombre = _nombre_coleccion(motor)

    query_filter = None
    if filtro_documento:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="document_name",
                    match=MatchValue(value=filtro_documento),
                )
            ]
        )

    results = client.query_points(
        collection_name=nombre,
        query=query_embedding,
        query_filter=query_filter,
        limit=k,
        score_threshold=umbral,
    )

    return [
        {
            "id": point.id,
            "coleccion": point.payload.get("document_name", "desconocido"),
            "contenido": point.payload.get("contenido", ""),
            "metadata": point.payload.get("metadata", "{}"),
            "score": round(point.score, 4),
        }
        for point in results.points
    ]
