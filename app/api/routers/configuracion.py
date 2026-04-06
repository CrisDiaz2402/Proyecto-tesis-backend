# app/api/routers/configuracion.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models
from app.core.security import get_current_user
from app.services.config_service import obtener_configuracion, cambiar_configuracion

router = APIRouter(prefix="/api/config", tags=["configuracion"])

# ── Esquemas ──────────────────────────────────────────────────────────────────

class ConfigResponse(BaseModel):
    motor_vectores: str
    motor_llm: str

class ConfigRequest(BaseModel):
    motor_vectores: str
    motor_llm: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
        cambiar_configuracion(
            motor_vectores=request.motor_vectores,
            motor_llm=request.motor_llm,
        )
        etiquetas = {
            ("local",  "local"):  "Todo Local (Ollama + Ollama)",
            ("cloud",  "cloud"):  "Todo Nube (Gemini + Gemini)",
            ("local",  "cloud"):  "Vectores Local + LLM Nube",
            ("cloud",  "local"):  "Vectores Nube + LLM Local",
        }
        clave = (request.motor_vectores, request.motor_llm)
        etiqueta = etiquetas.get(clave, f"{request.motor_vectores} / {request.motor_llm}")
        return {"ok": True, "mensaje": f"Modo activado: {etiqueta}"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno al guardar la configuración")