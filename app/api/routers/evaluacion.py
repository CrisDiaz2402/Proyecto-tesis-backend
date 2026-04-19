# app/api/routers/evaluacion.py

import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.db import models
from app.core.security import get_current_user
from app.schemas.schemas import EjecucionRequest, ResultadoEvaluacion
from app.services.evaluacion_service import (
    ejecutar_evaluacion,
    ejecutar_evaluacion_stream,
)

router = APIRouter(prefix="/api/evaluacion", tags=["evaluacion"])


@router.post("/ejecutar", response_model=ResultadoEvaluacion)
def ejecutar(
    request: EjecucionRequest,
    _: models.Usuario = Depends(get_current_user),
):
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


@router.post("/ejecutar-stream")
def ejecutar_stream(
    request: EjecucionRequest,
    _: models.Usuario = Depends(get_current_user),
):
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
                yield f"data: {json.dumps(evento, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_evento = {"tipo": "error", "mensaje_error": str(e)}
            yield f"data: {json.dumps(error_evento, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _generar_sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )