from typing import Optional

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import (
    QDRANT_COLLECTION_LOCAL,
    EMBED_DIMENSION_LOCAL,
)
from app.core.singletons import QdrantClientSingleton

def crear_coleccion(motor: str) -> None:
    client = QdrantClientSingleton().client
    nombre = QDRANT_COLLECTION_LOCAL
    dim = EMBED_DIMENSION_LOCAL

    colecciones = [c.name for c in client.get_collections().collections]
    if nombre not in colecciones:
        client.create_collection(
            collection_name=nombre,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"[QDRANT] Colección '{nombre}' creada ({dim} dims)")


def insertar_puntos(
    motor: str,
    embeddings: list[list[float]],
    payloads: list[dict],
) -> int:
    client = QdrantClientSingleton().client
    nombre = QDRANT_COLLECTION_LOCAL

    crear_coleccion(motor)

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
    print(f"[QDRANT] {len(points)} puntos insertados en '{nombre}'")
    return len(points)

def eliminar_puntos_por_documento(nombre_documento: str, motor: str) -> dict:

    client = QdrantClientSingleton().client
    nombre = QDRANT_COLLECTION_LOCAL

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
        print(f"[QDRANT] Puntos de '{nombre_documento}' eliminados de '{nombre}'")
        return {"mensaje": f"Documento '{nombre_documento}' eliminado de '{nombre}'."}
    except Exception as e:
        print(f"[QDRANT] Error al eliminar puntos de '{nombre_documento}': {e}")
        return {"mensaje": str(e)}


def eliminar_todos_los_puntos(motor: str) -> dict:
    client = QdrantClientSingleton().client
    nombre = QDRANT_COLLECTION_LOCAL
    dim = EMBED_DIMENSION_LOCAL

    try:
        colecciones = [c.name for c in client.get_collections().collections]
        if nombre in colecciones:
            client.delete_collection(collection_name=nombre)
        client.create_collection(
            collection_name=nombre,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"[QDRANT] colección '{nombre}' recreada")
        return {"mensaje": f"Todos los vectores de '{nombre}' eliminados."}
    except Exception as e:
        print(f"[QDRANT] Error al vaciar '{nombre}': {e}")
        return {"mensaje": str(e)}

def buscar_similares(
    query_embedding: list[float],
    motor: str,
    k: int = 10,
    umbral: float = 0.15,
    filtro_documento: Optional[str] = None,
) -> list[dict]:
    client = QdrantClientSingleton().client
    nombre = QDRANT_COLLECTION_LOCAL

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
