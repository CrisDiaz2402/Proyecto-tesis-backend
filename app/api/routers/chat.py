# app/api/routers/chat.py
import uuid
import asyncio
import traceback
from fastapi import APIRouter, HTTPException
from app.schemas.schemas import PreguntaRequest, ChatResponse
from app.services.rag_service import consultar_base_conocimiento
from app.services.config_service import obtener_motor_activo
from app.api.routers.monitor import registrar_inicio, registrar_fin

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/consultar", response_model=ChatResponse)
async def consultar_ia(request: PreguntaRequest):
    """
    Recibe una pregunta del usuario.
    El motor se obtiene de la configuración global del administrador,
    siendo totalmente transparente para el usuario final.
    """
    motor_actual = obtener_motor_activo()
    request_id   = str(uuid.uuid4())[:8]
    desde_cache  = False

    registrar_inicio(request_id, request.pregunta, motor_actual)
    try:
        loop = asyncio.get_event_loop()
        respuesta = await loop.run_in_executor(
            None,
            lambda: consultar_base_conocimiento(request.pregunta, motor=motor_actual)
        )
        # Latencia < 500 ms es indicador fiable de cache hit
        desde_cache = False  # se refinará con señal directa de rag_service si se desea
        return ChatResponse(
            pregunta_original=request.pregunta,
            respuesta=respuesta
        )
    except Exception as e:
        print(f"[CHAT ERROR] Error procesando pregunta: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        registrar_fin(request_id, desde_cache)