import asyncio
import traceback
import uuid
from fastapi import APIRouter, HTTPException

from app.schemas.schemas import PreguntaRequest, ChatResponse
from app.services.rag_service import consultar_base_conocimiento

from app.api.routers.monitor import registrar_inicio, registrar_fin

_CUDA_KEYWORDS = ("CUDA error", "runner process has terminated")

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/consultar", response_model=ChatResponse)
async def consultar_ia(request: PreguntaRequest):
    request_id = str(uuid.uuid4())
    try:
        loop = asyncio.get_event_loop()

        from app.services.config_service import obtener_motor_activo
        motor = obtener_motor_activo()

        registrar_inicio(request_id, request.pregunta, motor)

        for _intento in range(2):
            try:
                respuesta, desde_cache = await loop.run_in_executor(
                    None,
                    consultar_base_conocimiento,
                    request.pregunta,
                )
                break
            except Exception as _e:
                if any(k in str(_e) for k in _CUDA_KEYWORDS) and _intento == 0:
                    await asyncio.sleep(3)
                    continue
                raise

        registrar_fin(request_id, desde_cache=desde_cache)

        return ChatResponse(
            pregunta_original=request.pregunta,
            respuesta=respuesta,
        )
    except Exception as e:
        registrar_fin(request_id, desde_cache=False)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))