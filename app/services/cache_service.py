# app/services/cache_service.py
"""
Caché semántico de respuestas RAG — interfaz pública.
Delega la implementación al servicio Redis (redis_cache_service).

Mantiene la misma API pública para no romper importaciones existentes.
"""

from typing import Optional

from app.services.redis_cache_service import (
    buscar_en_cache as _redis_buscar,
    guardar_en_cache as _redis_guardar,
    limpiar_cache as _redis_limpiar,
    limpiar_cache_por_documento as _redis_limpiar_doc,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _generar_embedding_para_cache(texto: str, motor_vectores: str) -> list[float]:
    """Genera embedding usando el mismo motor que el servicio RAG."""
    from app.services.rag_service import _generar_embedding
    return _generar_embedding(texto, motor_vectores)


# ─────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────────────────

def buscar_en_cache(pregunta: str, motor_vectores: str, motor_llm: str) -> Optional[str]:
    """Busca en el caché semántico (Redis)."""
    # Generar embedding para búsqueda semántica
    try:
        embedding = _generar_embedding_para_cache(pregunta, motor_vectores)
    except Exception:
        embedding = None

    return _redis_buscar(
        pregunta=pregunta,
        motor_vectores=motor_vectores,
        motor_llm=motor_llm,
        embedding=embedding,
    )


def guardar_en_cache(
    pregunta: str,
    respuesta: str,
    documento_origen: str = "desconocido",
    motor_vectores: str = "local",
    motor_llm: str = "local",
) -> None:
    """Guarda una respuesta en el caché (Redis) con TTL de 24h."""
    try:
        embedding = _generar_embedding_para_cache(pregunta, motor_vectores)
    except Exception:
        embedding = None

    _redis_guardar(
        pregunta=pregunta,
        respuesta=respuesta,
        documento_origen=documento_origen,
        motor_vectores=motor_vectores,
        motor_llm=motor_llm,
        embedding=embedding,
    )


def limpiar_cache_por_documento(
    nombre_coleccion: str,
    motor_vectores: str = "local",
) -> dict:
    """Elimina entradas de caché para un documento específico."""
    return _redis_limpiar_doc(
        nombre_coleccion=nombre_coleccion,
        motor_vectores=motor_vectores,
    )


def limpiar_cache(
    motor_vectores: str = "local",
    motor_llm: str = "local",
    motor: Optional[str] = None,
    scope: str = "auto",
) -> dict:
    """
    Borrado total del caché semántico en Redis.

    1. Scope global:   limpiar_cache(scope="all")         → limpia los 3 cachés
    2. Shorthand:      limpiar_cache(motor="local")        → cache local:local + local:cloud
                       limpiar_cache(motor="cloud")        → cache cloud:cloud
                       limpiar_cache(motor="all")          → los 3 cachés
    3. Combinación:    limpiar_cache(motor_vectores="local", motor_llm="cloud")
    """
    return _redis_limpiar(
        motor=motor,
        scope=scope,
        motor_vectores=motor_vectores,
        motor_llm=motor_llm,
    )