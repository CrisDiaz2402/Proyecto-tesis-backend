# app/api/routers/chat.py
from fastapi import APIRouter
from app.schemas.schemas import PreguntaRequest, ChatResponse
from app.services.rag_service import consultar_base_conocimiento

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/consultar", response_model=ChatResponse)
async def consultar_ia(request: PreguntaRequest):
    """Recibe una pregunta de Rasa y retorna la respuesta del pipeline RAG."""
    respuesta = consultar_base_conocimiento(request.pregunta)
    return ChatResponse(
        pregunta_original = request.pregunta,
        respuesta         = respuesta
    )