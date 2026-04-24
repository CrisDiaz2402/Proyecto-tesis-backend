# app/api/routers/nlu_config.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import models
from app.db.deps import get_db
from app.core.defaults import DEFAULTS_NLU
from app.core.security import get_current_user
from app.schemas.schemas import NLUConfigOut, NLUConfigUpdate
from app.services import nlu_config_service

router = APIRouter(prefix="/api/nlu-config", tags=["nlu-config"])


@router.get("", response_model=NLUConfigOut, summary="Obtener configuración NLU actual")
def get_nlu_config(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    try:
        config = nlu_config_service.get_nlu_config(db)
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener config NLU: {e}")


@router.put("", response_model=NLUConfigOut, summary="Actualizar configuración NLU (parcial)")
def update_nlu_config(
    payload: NLUConfigUpdate,
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    data = payload.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(status_code=400, detail="No se enviaron campos para actualizar.")

    try:
        _old, new = nlu_config_service.actualizar_nlu_config(db, data)
        return new
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al actualizar config NLU: {e}")


@router.post("/reset", response_model=NLUConfigOut, summary="Restaurar configuración NLU a defaults")
def reset_nlu_config(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    try:
        _old, new = nlu_config_service.resetear_nlu_config(db)
        return new
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al resetear config NLU: {e}")


@router.get("/defaults", summary="Obtener valores default de configuración NLU")
def get_nlu_defaults(
    _: models.Usuario = Depends(get_current_user),
):
    return DEFAULTS_NLU
