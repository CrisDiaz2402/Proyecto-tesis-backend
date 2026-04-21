# app/api/routers/evaluacion.py

import json
import traceback
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
        return ejecutar_evaluacion(
            experimento=request.experimento,
            casos=casos_dict,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la evaluación: {str(e)}")


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
        casos_activos = [c for c in casos_dict if c.get("habilitado", True)]
        print(
            f"[EVAL_STREAM] ▶ Iniciando experimento='{request.experimento}' "
            f"con {len(casos_activos)} casos activos."
        )
        try:
            for evento in ejecutar_evaluacion_stream(
                experimento=request.experimento,
                casos=casos_dict,
            ):
                if evento.get("tipo") == "progreso":
                    r = evento.get("resultado", {})
                    print(
                        f"[EVAL_STREAM] [{evento['caso_actual']}/{evento['total_casos']}] "
                        f"id={r.get('id')} | veredicto={r.get('veredicto')} | "
                        f"latencia={r.get('latencia_ms')}ms"
                    )
                elif evento.get("tipo") == "completado":
                    rep = evento.get("reporte_final", {})
                    print(
                        f"[EVAL_STREAM] ✅ Completado — "
                        f"score={rep.get('score_global')} | "
                        f"similitud={rep.get('similitud_promedio')} | "
                        f"duración={rep.get('duracion_total_seg')}s"
                    )
                yield f"data: {json.dumps(evento, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[EVAL_STREAM] ❌ Error inesperado:\n{traceback.format_exc()}")
            error_evento = {
                "tipo": "error",
                "mensaje_error": str(e),
                "detalle": traceback.format_exc(),
            }
            yield f"data: {json.dumps(error_evento, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _generar_sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )