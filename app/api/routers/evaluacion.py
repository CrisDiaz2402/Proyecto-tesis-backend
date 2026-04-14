# app/api/routers/evaluacion.py
"""
Router de evaluación RAG.

Endpoints:
  POST /api/evaluacion/ejecutar
    Evaluación clásica síncrona — compatible con versiones anteriores.

  POST /api/evaluacion/ejecutar-stream
    Evaluación con Server-Sent Events (SSE).
    Emite un evento por cada caso completado con porcentaje real de avance.
    El frontend consume este stream en lugar del endpoint clásico.

  GET  /api/evaluacion/phoenix-status
    Proxy liviano hacia Phoenix.
    El frontend NO debe contactar Phoenix directamente (bloqueo CORS).
    Este endpoint hace el ping server-to-server y devuelve {"online": bool}.

Todos los endpoints requieren JWT igual que el resto de routers de administración.
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.db import models
from app.core.security import get_current_user
from app.schemas.evaluacion_schemas import EjecucionRequest, ResultadoEvaluacion
from app.services.evaluacion_service import (
    ejecutar_evaluacion,
    ejecutar_evaluacion_stream,
    PHOENIX_URL,
)

router = APIRouter(prefix="/api/evaluacion", tags=["evaluacion"])


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1 — Evaluación clásica síncrona (sin cambios, compatibilidad)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/ejecutar", response_model=ResultadoEvaluacion)
def ejecutar(
    request: EjecucionRequest,
    _: models.Usuario = Depends(get_current_user),
):
    """
    Lanza una sesión de evaluación RAG completa de forma síncrona.
    Devuelve el reporte completo al finalizar todos los casos.
    """
    casos_dict = [caso.model_dump() for caso in request.casos]

    if not any(c.get("habilitado", True) for c in casos_dict):
        raise HTTPException(
            status_code=400,
            detail="No hay casos habilitados. Activa al menos un caso antes de lanzar la evaluación.",
        )

    try:
        resultado = ejecutar_evaluacion(
            experimento=request.experimento,
            casos=casos_dict,
        )
        return resultado
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la evaluación: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2 — Evaluación con SSE (progreso caso a caso)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/ejecutar-stream")
def ejecutar_stream(
    request: EjecucionRequest,
    _: models.Usuario = Depends(get_current_user),
):
    """
    Lanza la evaluación y emite Server-Sent Events (SSE) mientras avanza.

    Cada evento tiene el formato estándar SSE:
        data: {"tipo": "progreso", "caso_actual": N, "total_casos": T,
               "porcentaje": P, "resultado": {...}}\\n\\n
        data: {"tipo": "completado", ..., "reporte_final": {...}}\\n\\n
        data: {"tipo": "error", "mensaje_error": "..."}\\n\\n

    El cliente debe abrir la conexión con fetch + ReadableStream
    y parsear cada línea "data: <json>".
    """
    casos_dict = [caso.model_dump() for caso in request.casos]

    if not any(c.get("habilitado", True) for c in casos_dict):
        raise HTTPException(
            status_code=400,
            detail="No hay casos habilitados. Activa al menos un caso antes de lanzar la evaluación.",
        )

    def _generar_sse():
        try:
            for evento in ejecutar_evaluacion_stream(
                experimento=request.experimento,
                casos=casos_dict,
            ):
                # Formato SSE estándar: "data: <json>\n\n"
                yield f"data: {json.dumps(evento, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_evento = {"tipo": "error", "mensaje_error": str(e)}
            yield f"data: {json.dumps(error_evento, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _generar_sse(),
        media_type="text/event-stream",
        headers={
            # Evitar que proxies o el navegador guarden en caché el stream
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3 — Proxy de estado Phoenix (resuelve el CORS)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/phoenix-status")
def phoenix_status(
    _: models.Usuario = Depends(get_current_user),
):
    """
    Verifica si Phoenix está disponible haciendo un ping server-to-server.

    El frontend NUNCA debe contactar Phoenix directamente (puerto 6006)
    porque Phoenix no tiene CORS configurado y el navegador bloquea la petición.
    Este endpoint actúa como proxy: el frontend llama aquí (puerto 8000,
    que sí tiene CORS habilitado) y el backend hace el ping internamente.

    Respuesta: {"online": true} | {"online": false}
    """
    import requests as req
    try:
        resp = req.get(
            f"{PHOENIX_URL}/v1/spans",
            params={"limit": 1},
            timeout=3,
        )
        return {"online": resp.status_code == 200}
    except Exception:
        return {"online": False}