from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional
from app.db import models
from app.db.deps import get_db
from app.core.security import get_current_user
from app.core.prompts import PROMPT_PRINCIPAL_DEFAULT
from app.services import rag_params_service
from app.services.rag_params_service import (
    DEFAULTS,
    PARAM_LIMITS,
    determinar_limpieza,
)

router = APIRouter(prefix="/api/rag-params", tags=["rag-params"])

class RagParamsRequest(BaseModel):
    # ── Parámetros con impacto demostrable ────────────────────────────────────
    umbral_relevancia_local:     Optional[float] = Field(None, ge=0.05, le=0.50)
    rag_k_local:                 Optional[int]   = Field(None, ge=2,    le=20)
    prompt_principal:            Optional[str]   = Field(None)

def _ejecutar_limpieza(limpieza: dict) -> list[str]:
    from app.services.cache_service import limpiar_cache
    from app.services.rag_service import eliminar_todos_los_vectores

    acciones: list[str] = []

    if limpieza["limpiar_cache_ll"]:
        limpiar_cache(motor_vectores="local", motor_llm="local")
        acciones.append("Caché L2 local:local limpiado")

    if limpieza["limpiar_cache_lc"]:
        limpiar_cache(motor_vectores="local", motor_llm="cloud")
        acciones.append("Caché L2 local:cloud limpiado")

    if limpieza["limpiar_vectores_local"]:
        eliminar_todos_los_vectores("local")
        acciones.append("Vectores LOCAL eliminados (requiere reindexar)")

    return acciones

@router.get("", summary="Obtener parámetros RAG actuales con defaults y límites")
def get_rag_params(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    try:
        config  = rag_params_service.get_params_con_db(db)
        current = rag_params_service._config_to_dict(config)
        return {
            "ok":                   True,
            "parametros_actuales":  current,
            "defaults":             DEFAULTS,
            "limites":              PARAM_LIMITS,
            "fecha_actualizacion":  config.fecha_actualizacion,
            "prompts_default_texto": {
                "prompt_principal": PROMPT_PRINCIPAL_DEFAULT,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener parámetros RAG: {str(e)}",
        )


@router.put("", summary="Actualizar parámetros RAG con limpieza automática")
def update_rag_params(
    request: RagParamsRequest,
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    new_data = {k: v for k, v in request.dict().items() if v is not None}

    if not new_data:
        raise HTTPException(
            status_code=400,
            detail="No se enviaron parámetros para actualizar. Envía al menos un campo.",
        )

    errores = rag_params_service.validar_params(new_data)
    if errores:
        raise HTTPException(
            status_code=422,
            detail={"mensaje": "Parámetros fuera de rango.", "errores": errores},
        )

    try:
        old_params, new_params = rag_params_service.actualizar_params(db, new_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al guardar parámetros en base de datos: {str(e)}",
        )

    limpieza = determinar_limpieza(old_params, new_params)

    try:
        acciones = _ejecutar_limpieza(limpieza)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Parámetros guardados correctamente, pero ocurrió un error durante "
                f"la limpieza automática: {str(e)}. "
                f"Considera limpiar cachés y vectores manualmente desde la sección de Documentos."
            ),
        )

    advertencias: list[str] = []
    if limpieza["requiere_reindexar"]:
        advertencias.append(
            "El parámetro de chunking cambió. Los vectores fueron eliminados automáticamente. "
            "Debes reprocesar (reindexar) todos los documentos desde la sección de Documentos "
            "para que el nuevo chunking tenga efecto."
        )

    params_cambiados = limpieza["params_cambiados"]
    sin_cambio = [k for k in new_data if k not in params_cambiados]

    return {
        "ok":                  True,
        "mensaje":             (
            f"{len(params_cambiados)} parámetro(s) actualizado(s). "
            f"{len(acciones)} acción(es) de limpieza ejecutada(s)."
        ),
        "params_cambiados":    params_cambiados,
        "params_sin_cambio":   sin_cambio,
        "acciones_limpieza":   acciones,
        "advertencias":        advertencias,
        "parametros_actuales": new_params,
    }


@router.post("/reset", summary="Restaurar parámetros RAG a valores por defecto")
def reset_rag_params(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):

    try:
        old_params, new_params = rag_params_service.resetear_params(db)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al resetear parámetros: {str(e)}",
        )

    limpieza = determinar_limpieza(old_params, new_params)

    try:
        acciones = _ejecutar_limpieza(limpieza)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Parámetros reseteados, pero error en limpieza automática: {str(e)}. "
                f"Considera limpiar cachés manualmente."
            ),
        )

    advertencias: list[str] = []
    if limpieza["requiere_reindexar"]:
        advertencias.append(
            "El parámetro de chunking fue modificado antes del reset. "
            "Los vectores fueron eliminados. Reprocesa los documentos."
        )

    return {
        "ok":                  True,
        "mensaje":             "Parámetros restaurados a valores por defecto.",
        "params_reseteados":   limpieza["params_cambiados"],
        "acciones_limpieza":   acciones,
        "advertencias":        advertencias,
        "parametros_actuales": new_params,
        "defaults":            DEFAULTS,
    }


@router.get("/defaults", summary="Consultar defaults y límites sin modificar nada")
def get_defaults(
    _: models.Usuario = Depends(get_current_user),
):
    return {
        "ok":      True,
        "defaults": DEFAULTS,
        "limites":  PARAM_LIMITS,
    }