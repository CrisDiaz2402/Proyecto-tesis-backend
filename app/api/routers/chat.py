# app/api/routers/chat.py
from fastapi import APIRouter, HTTPException
from app.schemas.schemas import PreguntaRequest, ChatResponse
from app.services.rag_service import consultar_base_conocimiento
from app.services.config_service import obtener_motor_activo # <--- Nuevo servicio
import traceback

router = APIRouter(prefix="/api/chat", tags=["chat"])

@router.post("/consultar", response_model=ChatResponse)
async def consultar_ia(request: PreguntaRequest):
    """
    Recibe una pregunta de Rasa.
    El motor se obtiene de la configuración global del administrador,
    siendo totalmente transparente para el usuario final.
    """
    try:
        # 1. Obtenemos el motor activo desde el servicio de configuración
        # Este servicio leerá si el admin puso "local" o "cloud"
        motor_actual = obtener_motor_activo()
        
        # 2. Ejecutamos la consulta RAG pasando el motor configurado
        respuesta = consultar_base_conocimiento(request.pregunta, motor=motor_actual)
        
        return ChatResponse(
            pregunta_original = request.pregunta,
            respuesta         = respuesta
        )
    except Exception as e:
        print(f"[CHAT ERROR] Error procesando pregunta: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))