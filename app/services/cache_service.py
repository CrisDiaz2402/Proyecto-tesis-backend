import hashlib
import json
import time
import redis
from typing import Optional

from app.core.singletons import RedisClientSingleton

TTL_DEFAULT = 60 * 60 * 24
TTL_EXTENDED = 60 * 60 * 24 * 7
UMBRAL_SIMILITUD_CACHE = 0.02

def _get_redis():
    return RedisClientSingleton().client


def _iterar_claves_redis(r, pattern, count=100):
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=count)
        yield from keys
        if cursor == 0:
            break


def _hash_pregunta(pregunta: str) -> str:
    normalizada = pregunta.strip().lower()
    return hashlib.sha256(normalizada.encode("utf-8")).hexdigest()[:32]


def _cache_key(motor_vectores: str, motor_llm: str, pregunta_hash: str) -> str:
    return f"cache:{motor_vectores}:{motor_llm}:{pregunta_hash}"


def _pattern_key(motor_vectores: str, motor_llm: str) -> str:
    return f"cache:{motor_vectores}:{motor_llm}:*"


def _es_respuesta_cacheable(respuesta: str) -> bool:
    if not respuesta or len(respuesta.strip()) < 20:
        return False
    texto = respuesta.lower().strip()
    if "Error interno al consultar" in respuesta:
        print("[REDIS_CACHE] ⚠️  Respuesta de error NO cacheada.")
        return False
    from app.services.nlu_config_service import get_nlu_config
    cfg = get_nlu_config()
    frases_rechazo = cfg.get("frases_rechazo", [])
    for frase in frases_rechazo:
        if frase.lower() in texto:
            return False
    if texto.startswith("lo siento") and len(respuesta) < 120:
        return False
    if respuesta.strip()[-1] not in ".!?»":
        return False
    return True


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class CacheProxy:
    _failures = 0
    _circuit_open_until = 0.0
    MAX_FAILURES = 3
    CIRCUIT_COOLDOWN = 30

    def buscar(self, pregunta: str, motor_vectores: str, motor_llm: str, **kwargs) -> Optional[str]:
        if not self._puede_conectar():
            return None
        try:
            resultado = _buscar_en_redis(pregunta, motor_vectores=motor_vectores, motor_llm=motor_llm, **kwargs)
            self._registrar_exito()
            return resultado
        except redis.ConnectionError:
            self._registrar_fallo()
            return None

    def guardar(self, pregunta: str, respuesta: str, **kwargs) -> None:
        if not self._puede_conectar():
            return
        try:
            _guardar_en_redis(pregunta, respuesta, **kwargs)
            self._registrar_exito()
        except redis.ConnectionError:
            self._registrar_fallo()

    def _puede_conectar(self) -> bool:
        if self._failures >= self.MAX_FAILURES:
            if time.time() < self._circuit_open_until:
                return False
            self._failures = 0
        return True

    def _registrar_fallo(self):
        self._failures += 1
        if self._failures >= self.MAX_FAILURES:
            self._circuit_open_until = time.time() + self.CIRCUIT_COOLDOWN
            print(f"[CACHE PROXY] ⚠️ Circuit breaker abierto por {self.CIRCUIT_COOLDOWN}s")

    def _registrar_exito(self):
        self._failures = 0


cache_proxy = CacheProxy()

def _buscar_en_redis(
    pregunta: str,
    motor_vectores: str,
    motor_llm: str,
    embedding: Optional[list[float]] = None,
) -> Optional[str]:
    r = _get_redis()

    pregunta_hash = _hash_pregunta(pregunta)
    key = _cache_key(motor_vectores, motor_llm, pregunta_hash)

    data = r.hgetall(key)
    if data and "respuesta" in data:
        print(
            f"[REDIS_CACHE] ✅ Hit exacto "
            f"(modo={motor_vectores}:{motor_llm})"
        )
        return data["respuesta"]

    if embedding is not None:
        pattern = _pattern_key(motor_vectores, motor_llm)
        min_score = 1 - UMBRAL_SIMILITUD_CACHE
        best_score = 0.0
        best_respuesta = None

        for k in _iterar_claves_redis(r, pattern):
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

        if best_respuesta:
            print(
                f"[REDIS_CACHE] ✅ Hit semántico "
                f"(score={best_score:.3f}, umbral_sim={UMBRAL_SIMILITUD_CACHE:.3f}, "
                f"modo={motor_vectores}:{motor_llm})"
            )
            return best_respuesta

    return None


def _guardar_en_redis(
    pregunta: str,
    respuesta: str,
    documento_origen: str = "desconocido",
    motor_vectores: str = "local",
    motor_llm: str = "local",
    embedding: Optional[list[float]] = None,
    ttl: int = TTL_DEFAULT,
) -> None:
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


def _limpiar_redis(
    motor: Optional[str] = None,
    scope: str = "auto",
    motor_vectores: str = "local",
    motor_llm: str = "local",
) -> dict:
    r = _get_redis()

    combis_por_motor = {
        "local": [("local", "local"), ("local", "cloud")],
    }
    todas = [("local", "local"), ("local", "cloud")]

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
        batch = []
        for key in _iterar_claves_redis(r, pattern, count=200):
            batch.append(key)
            if len(batch) >= 200:
                deleted += r.delete(*batch)
                batch = []
        if batch:
            deleted += r.delete(*batch)
        resultados.append(f"caché {mv}:{ml} limpiado ({deleted})")
        print(f"[REDIS_CACHE] 🗑️ Caché {mv}:{ml} limpiado ({deleted} entradas).")

    return {"mensaje": " | ".join(resultados)}


def _limpiar_redis_por_documento(
    nombre_coleccion: str,
    motor_vectores: str = "local",
) -> dict:
    r = _get_redis()

    combis_por_motor = {
        "local": [("local", "local"), ("local", "cloud")],
    }
    combis = combis_por_motor.get(motor_vectores, [(motor_vectores, motor_vectores)])

    resultados = []
    for mv, ml in combis:
        pattern = _pattern_key(mv, ml)
        deleted = 0
        for k in _iterar_claves_redis(r, pattern, count=200):
            cached = r.hgetall(k)
            if cached.get("documento_origen") == nombre_coleccion:
                r.delete(k)
                deleted += 1
        if deleted:
            resultados.append(f"{mv}:{ml} limpiado ({deleted} entradas)")
            print(f"[REDIS_CACHE] 🗑️ '{nombre_coleccion}' eliminado de {mv}:{ml} ({deleted} entradas).")
        else:
            resultados.append(f"{mv}:{ml} sin entradas del documento")

    return {"mensaje": f"Caché del documento '{nombre_coleccion}' procesado: {' | '.join(resultados)}"}


def _generar_embedding_para_cache(texto: str, motor_vectores: str) -> list[float]:
    from app.services.rag_service import _generar_embedding
    return _generar_embedding(texto, motor_vectores)


def buscar_en_cache(pregunta: str, motor_vectores: str, motor_llm: str) -> Optional[str]:
    try:
        embedding = _generar_embedding_para_cache(pregunta, motor_vectores)
    except Exception:
        embedding = None

    return cache_proxy.buscar(
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
    try:
        embedding = _generar_embedding_para_cache(pregunta, motor_vectores)
    except Exception:
        embedding = None

    cache_proxy.guardar(
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
    return _limpiar_redis_por_documento(
        nombre_coleccion=nombre_coleccion,
        motor_vectores=motor_vectores,
    )


def limpiar_cache(
    motor_vectores: str = "local",
    motor_llm: str = "local",
    motor: Optional[str] = None,
    scope: str = "auto",
) -> dict:
    return _limpiar_redis(
        motor=motor,
        scope=scope,
        motor_vectores=motor_vectores,
        motor_llm=motor_llm,
    )
