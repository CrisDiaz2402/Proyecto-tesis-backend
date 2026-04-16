# app/services/redis_cache_service.py
"""
Caché semántico de respuestas RAG — almacenado en Redis.
Reemplaza la implementación anterior basada en pgvector/PostgreSQL.

Estrategia:
  - Cada entrada se guarda como hash en Redis con TTL.
  - Clave: cache:{motor_vectores}:{motor_llm}:{hash_pregunta}
  - Fallback a búsqueda por iteración si RedisSearch no está disponible.
"""

import hashlib
import json
import time
from typing import Optional

import redis

from app.core.config import REDIS_URL

# ─── TTL por defecto ─────────────────────────────────────────────────────────
TTL_DEFAULT = 60 * 60 * 24        # 24 horas
TTL_EXTENDED = 60 * 60 * 24 * 7   # 7 días

# ─── Umbral de similitud para cache hit ──────────────────────────────────────
UMBRAL_SIMILITUD = 0.02

# ─── Cliente singleton ───────────────────────────────────────────────────────
_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    """Devuelve el cliente Redis (singleton, decode_responses=True)."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        print(f"[REDIS_CACHE] ✅ Conectado a {REDIS_URL}")
    return _redis_client


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _hash_pregunta(pregunta: str) -> str:
    """Genera un hash determinista de la pregunta normalizada."""
    normalizada = pregunta.strip().lower()
    return hashlib.sha256(normalizada.encode("utf-8")).hexdigest()[:32]


def _cache_key(motor_vectores: str, motor_llm: str, pregunta_hash: str) -> str:
    """Genera la clave Redis para una entrada de caché."""
    return f"cache:{motor_vectores}:{motor_llm}:{pregunta_hash}"


def _pattern_key(motor_vectores: str, motor_llm: str) -> str:
    """Patrón para buscar todas las claves de una combinación de motores."""
    return f"cache:{motor_vectores}:{motor_llm}:*"


def _es_respuesta_cacheable(respuesta: str) -> bool:
    """Valida si una respuesta merece ser cacheada."""
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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similitud coseno entre dos vectores."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────────────────

def buscar_en_cache(
    pregunta: str,
    motor_vectores: str,
    motor_llm: str,
    embedding: Optional[list[float]] = None,
) -> Optional[str]:
    """
    Busca en el caché semántico de Redis.

    Estrategia en dos fases:
      1. Búsqueda exacta por hash de la pregunta normalizada.
      2. Si se proporciona embedding, búsqueda semántica por iteración
         sobre las claves del mismo motor (fallback sin RedisSearch).
    """
    r = _get_redis()

    # ── Fase 1: Búsqueda exacta por hash ─────────────────────────────────────
    pregunta_hash = _hash_pregunta(pregunta)
    key = _cache_key(motor_vectores, motor_llm, pregunta_hash)

    data = r.hgetall(key)
    if data and "respuesta" in data:
        print(
            f"[REDIS_CACHE] ✅ Hit exacto "
            f"(modo={motor_vectores}:{motor_llm})"
        )
        return data["respuesta"]

    # ── Fase 2: Búsqueda semántica por similitud (fallback) ───────────────────
    if embedding is not None:
        pattern = _pattern_key(motor_vectores, motor_llm)
        min_score = 1 - UMBRAL_SIMILITUD
        best_score = 0.0
        best_respuesta = None

        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=100)
            for k in keys:
                cached = r.hgetall(k)
                if "embedding" not in cached or "respuesta" not in cached:
                    continue
                try:
                    cached_emb = json.loads(cached["embedding"])
                    score = _cosine_similarity(embedding, cached_emb)
                    if score >= min_score and score > best_score:
                        best_score = score
                        best_respuesta = cached["respuesta"]
                except (json.JSONDecodeError, TypeError):
                    continue
            if cursor == 0:
                break

        if best_respuesta:
            print(
                f"[REDIS_CACHE] ✅ Hit semántico "
                f"(score={best_score:.3f}, umbral_sim={UMBRAL_SIMILITUD:.3f}, "
                f"modo={motor_vectores}:{motor_llm})"
            )
            return best_respuesta

    return None


def guardar_en_cache(
    pregunta: str,
    respuesta: str,
    documento_origen: str = "desconocido",
    motor_vectores: str = "local",
    motor_llm: str = "local",
    embedding: Optional[list[float]] = None,
    ttl: int = TTL_DEFAULT,
) -> None:
    """
    Guarda una respuesta en el caché Redis con TTL.
    Si se proporciona embedding, lo almacena para búsqueda semántica.
    """
    if not _es_respuesta_cacheable(respuesta):
        print("[REDIS_CACHE] ⚠️ Respuesta no cacheada (inválida, truncada o rechazo).")
        return

    r = _get_redis()
    pregunta_hash = _hash_pregunta(pregunta)
    key = _cache_key(motor_vectores, motor_llm, pregunta_hash)

    entry = {
        "pregunta": pregunta,
        "respuesta": respuesta,
        "documento_origen": documento_origen,
        "motor_vectores": motor_vectores,
        "motor_llm": motor_llm,
        "timestamp": str(time.time()),
    }

    if embedding is not None:
        entry["embedding"] = json.dumps(embedding)

    r.hset(key, mapping=entry)
    r.expire(key, ttl)
    print(f"[REDIS_CACHE] 💾 Respuesta guardada (modo={motor_vectores}:{motor_llm}, TTL={ttl}s)")


def limpiar_cache(
    motor: Optional[str] = None,
    scope: str = "auto",
    motor_vectores: str = "local",
    motor_llm: str = "local",
) -> dict:
    """
    Borrado del caché semántico en Redis.

    Scopes:
      - scope="all" o motor="all" → limpia todos los cachés
      - motor="local" → cachés local:local + local:cloud
      - motor="cloud" → caché cloud:cloud
      - combinación específica con motor_vectores + motor_llm
    """
    r = _get_redis()

    combis_por_motor = {
        "local": [("local", "local"), ("local", "cloud")],
        "cloud": [("cloud", "cloud")],
    }
    todas = [("local", "local"), ("local", "cloud"), ("cloud", "cloud")]

    if scope == "all" or motor == "all":
        combis = todas
    elif motor is not None:
        combis = combis_por_motor.get(motor, [(motor, motor)])
    else:
        combis = [(motor_vectores, motor_llm)]

    resultados = []
    for mv, ml in combis:
        pattern = _pattern_key(mv, ml)
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=200)
            if keys:
                deleted += r.delete(*keys)
            if cursor == 0:
                break
        resultados.append(f"caché {mv}:{ml} limpiado ({deleted})")
        print(f"[REDIS_CACHE] 🗑️ Caché {mv}:{ml} limpiado ({deleted} entradas).")

    return {"mensaje": " | ".join(resultados)}


def limpiar_cache_por_documento(
    nombre_coleccion: str,
    motor_vectores: str = "local",
) -> dict:
    """
    Elimina entradas de caché que corresponden a un documento específico.
    Busca por el campo documento_origen en los hashes.
    """
    r = _get_redis()

    combis_por_motor = {
        "local": [("local", "local"), ("local", "cloud")],
        "cloud": [("cloud", "cloud")],
    }
    combis = combis_por_motor.get(motor_vectores, [(motor_vectores, motor_vectores)])

    resultados = []
    for mv, ml in combis:
        pattern = _pattern_key(mv, ml)
        deleted = 0
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=200)
            for k in keys:
                cached = r.hgetall(k)
                if cached.get("documento_origen") == nombre_coleccion:
                    r.delete(k)
                    deleted += 1
            if cursor == 0:
                break
        if deleted:
            resultados.append(f"{mv}:{ml} limpiado ({deleted} entradas)")
            print(f"[REDIS_CACHE] 🗑️ '{nombre_coleccion}' eliminado de {mv}:{ml} ({deleted} entradas).")
        else:
            resultados.append(f"{mv}:{ml} sin entradas del documento")

    return {"mensaje": f"Caché del documento '{nombre_coleccion}' procesado: {' | '.join(resultados)}"}
