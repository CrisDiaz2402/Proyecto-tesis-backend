# app/api/routers/monitor.py
import time
from fastapi import APIRouter, Depends
from app.core.security import get_current_user

router = APIRouter(prefix="/api/monitor", tags=["monitor"])

# ── Estado compartido en memoria (proceso único Uvicorn) ──────────────────────
_consultas_activas: dict[str, dict] = {}   # {request_id: {pregunta, inicio, motor}}
_historial: list[dict] = []                # últimas 50 consultas completadas
_MAX_HISTORIAL = 50


def registrar_inicio(request_id: str, pregunta: str, motor: str) -> None:
    _consultas_activas[request_id] = {
        "id":       request_id,
        "pregunta": pregunta[:80] + ("…" if len(pregunta) > 80 else ""),
        "motor":    motor,
        "inicio":   time.time(),
    }


def registrar_fin(request_id: str, desde_cache: bool) -> None:
    if request_id not in _consultas_activas:
        return
    entrada = _consultas_activas.pop(request_id)
    latencia_ms = round((time.time() - entrada["inicio"]) * 1000)
    registro = {**entrada, "latencia_ms": latencia_ms, "cache": desde_cache,
                "fin": time.time()}
    _historial.insert(0, registro)
    if len(_historial) > _MAX_HISTORIAL:
        _historial.pop()


@router.get("/estado")
def estado_sistema(_: object = Depends(get_current_user)):
    """Snapshot en tiempo real del estado del sistema."""
    latencias = [r["latencia_ms"] for r in _historial if not r["cache"]]
    return {
        "activas":         list(_consultas_activas.values()),
        "total_activas":   len(_consultas_activas),
        "historial":       _historial[:20],
        "latencia_avg_ms": round(sum(latencias) / len(latencias)) if latencias else 0,
        "cache_hits":      sum(1 for r in _historial if r["cache"]),
        "total_consultas": len(_historial),
        "timestamp":       time.time(),
    }