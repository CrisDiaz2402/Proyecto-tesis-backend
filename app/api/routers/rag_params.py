"""
Router para gestión de parámetros RAG desde el frontend.

Endpoints:
  GET  /api/rag-params          → parámetros actuales + defaults + límites + prompts_default_texto
  PUT  /api/rag-params          → actualizar parámetros (incluye prompts) + limpieza automática
  POST /api/rag-params/reset    → restaurar defaults + limpieza automática
  GET  /api/rag-params/defaults → solo defaults y límites (sin tocar BD)
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# ESQUEMAS PYDANTIC — Validación de entrada (primera capa de validación)
# ge/le corresponden a los min/max de PARAM_LIMITS en rag_params_service.py.
# ─────────────────────────────────────────────────────────────────────────────

class RagParamsRequest(BaseModel):
    # ── Parámetros con impacto demostrable ────────────────────────────────────
    umbral_relevancia_local:     Optional[float] = Field(None, ge=0.05, le=0.50)
    umbral_relevancia_cloud:     Optional[float] = Field(None, ge=0.10, le=0.70)
    rag_k_local:                 Optional[int]   = Field(None, ge=2,    le=20)
    rag_k_cloud:                 Optional[int]   = Field(None, ge=2,    le=15)

    # ── Prompt principal ──────────────────────────────────────────────────────
    # Sin restricciones de rango — cualquier string es válido.
    # Cadena vacía "" → el servicio lo interpreta como "volver al hardcodeado".
    prompt_principal:            Optional[str]   = Field(None)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — Ejecuta las acciones de limpieza y devuelve log
# ─────────────────────────────────────────────────────────────────────────────

def _ejecutar_limpieza(limpieza: dict) -> list[str]:
    """
    Ejecuta las acciones de limpieza indicadas por determinar_limpieza().
    Importaciones lazy para evitar importación circular con cache_service/rag_service.
    Retorna lista de acciones ejecutadas.
    """
    from app.services.cache_service import limpiar_cache
    from app.services.rag_service import eliminar_todos_los_vectores_chroma

    acciones: list[str] = []

    if limpieza["limpiar_cache_ll"]:
        limpiar_cache(motor_vectores="local", motor_llm="local")
        acciones.append("Caché L2 local:local limpiado")

    if limpieza["limpiar_cache_lc"]:
        limpiar_cache(motor_vectores="local", motor_llm="cloud")
        acciones.append("Caché L2 local:cloud limpiado")

    if limpieza["limpiar_cache_cc"]:
        limpiar_cache(motor_vectores="cloud", motor_llm="cloud")
        acciones.append("Caché L2 cloud:cloud limpiado")

    if limpieza["limpiar_vectores_local"]:
        eliminar_todos_los_vectores_chroma("local")
        acciones.append("Vectores LOCAL eliminados (requiere reindexar)")

    if limpieza["limpiar_vectores_cloud"]:
        eliminar_todos_los_vectores_chroma("cloud")
        acciones.append("Vectores CLOUD eliminados (requiere reindexar)")

    return acciones


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", summary="Obtener parámetros RAG actuales con defaults y límites")
def get_rag_params(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    """
    Retorna:
    - parametros_actuales: valores guardados en BD (o defaults si es primera vez).
    - defaults: valores por defecto del sistema.
    - limites: min/max/type/label/descripcion por parámetro.
    - prompts_default_texto: textos hardcodeados de los prompts (para el frontend).
    """
    try:
        config  = rag_params_service.get_params_con_db(db)
        current = rag_params_service._config_to_dict(config)
        return {
            "ok":                   True,
            "parametros_actuales":  current,
            "defaults":             DEFAULTS,
            "limites":              PARAM_LIMITS,
            "fecha_actualizacion":  config.fecha_actualizacion,
            # ── NUEVO: textos hardcodeados para que el frontend pueda mostrar
            # el placeholder correcto y detectar si el prompt fue personalizado.
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
    """
    Actualiza uno o más parámetros RAG (incluidos los prompts) y ejecuta
    automáticamente la limpieza de vectores y/o cachés que corresponda.

    Para los prompts:
    - Texto completo → se persiste en BD y se usa en runtime.
    - Cadena vacía "" → restaura al texto hardcodeado por defecto.

    Responde con:
    - ok: bool
    - params_cambiados: lista de parámetros que efectivamente cambiaron de valor.
    - acciones_limpieza: lista de acciones ejecutadas (cachés/vectores borrados).
    - advertencias: avisos importantes (ej: requiere reindexar documentos).
    - parametros_actuales: estado final de todos los parámetros.
    """
    # Filtrar solo los campos que vienen en el request (no None).
    # Los prompts pueden venir como cadena vacía "" (reset), que sí es un valor
    # válido y no debe filtrarse — solo se filtra None (campo no enviado).
    new_data = {k: v for k, v in request.dict().items() if v is not None}

    if not new_data:
        raise HTTPException(
            status_code=400,
            detail="No se enviaron parámetros para actualizar. Envía al menos un campo.",
        )

    # Segunda capa de validación de rangos (solo para campos numéricos)
    errores = rag_params_service.validar_params(new_data)
    if errores:
        raise HTTPException(
            status_code=422,
            detail={"mensaje": "Parámetros fuera de rango.", "errores": errores},
        )

    # Guardar en BD
    try:
        old_params, new_params = rag_params_service.actualizar_params(db, new_data)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al guardar parámetros en base de datos: {str(e)}",
        )

    # Determinar y ejecutar limpieza
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

    # Construir advertencias
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
    """
    Restaura TODOS los parámetros (incluidos los prompts) a sus valores por
    defecto y ejecuta la limpieza automática correspondiente.
    """
    try:
        old_params, new_params = rag_params_service.resetear_params(db)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al resetear parámetros: {str(e)}",
        )

    # Misma lógica de limpieza que el update
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
    """
    Retorna solo los valores por defecto y los límites de validación.
    No lee ni modifica la BD.
    """
    return {
        "ok":      True,
        "defaults": DEFAULTS,
        "limites":  PARAM_LIMITS,
    }