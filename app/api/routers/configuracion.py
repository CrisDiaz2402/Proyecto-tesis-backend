# app/api/routers/configuracion.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import models
from app.db.deps import get_db
from app.core.security import get_current_user
from app.core.constants import MODO_LABELS, COMBINACIONES_INVALIDAS
from app.core.exceptions import CombinacionInvalidaError, ConfiguracionError
from app.services.config_service import obtener_configuracion, cambiar_configuracion

router = APIRouter(prefix="/api/config", tags=["configuracion"])

# ── Esquemas ──────────────────────────────────────────────────────────────────

class ConfigResponse(BaseModel):
    motor_vectores: str
    motor_llm: str

class ConfigRequest(BaseModel):
    motor_vectores: str
    motor_llm: str

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/motor", response_model=ConfigResponse)
def get_motor(
    _: models.Usuario = Depends(get_current_user)  # 🔒 Protegido con JWT
):
    """Devuelve la configuración dual de motores (vectores + LLM)."""
    try:
        config = obtener_configuracion()
        return ConfigResponse(**config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/motor")
def update_motor(
    request: ConfigRequest,
    _: models.Usuario = Depends(get_current_user)  # 🔒 Protegido con JWT
):
    """Cambia la combinación de motores del Avatar."""
    try:
        # Validar combinación antes de aplicar
        combinacion = (request.motor_vectores, request.motor_llm)
        if combinacion in COMBINACIONES_INVALIDAS:
            raise CombinacionInvalidaError(request.motor_vectores, request.motor_llm)
        
        cambiar_configuracion(
            motor_vectores=request.motor_vectores,
            motor_llm=request.motor_llm,
        )
        
        # Usar etiquetas centralizadas
        etiqueta = MODO_LABELS.get(
            combinacion, 
            f"{request.motor_vectores} / {request.motor_llm}"
        )
        
        return {"ok": True, "mensaje": f"Modo activado: {etiqueta}"}
    
    except CombinacionInvalidaError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except ConfiguracionError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno al guardar la configuración")