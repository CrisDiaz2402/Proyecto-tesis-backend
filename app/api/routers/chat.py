# app/api/routers/chat.py
import uuid
import asyncio
import traceback
from fastapi import APIRouter, HTTPException
from app.schemas.schemas import PreguntaRequest, ChatResponse
from app.services.rag_service import consultar_base_conocimiento
from app.services.config_service import obtener_motor_activo

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/consultar", response_model=ChatResponse)
async def consultar_ia(request: PreguntaRequest):
    motor_actual = obtener_motor_activo()

    try:
        loop = asyncio.get_event_loop()
        respuesta = await loop.run_in_executor(
            None,
            lambda: consultar_base_conocimiento(request.pregunta, motor=motor_actual)
        )
        return ChatResponse(
            pregunta_original=request.pregunta,
            respuesta=respuesta
        )
    except Exception as e:
        print(f"[CHAT ERROR] Error procesando pregunta: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))