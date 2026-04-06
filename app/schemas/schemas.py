# app/schemas/schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DocumentoOut(BaseModel):
    id: int
    nombre_archivo: str
    subido_por: str
    fecha_subida: datetime
    
    # ─── ECOSISTEMA DUAL (Para pintar en el Frontend) ───
    procesado_local: bool
    procesado_cloud: bool
    estado_local: str
    estado_cloud: str

    class Config:
        from_attributes = True

class DocumentoUploadResponse(BaseModel):
    ok: bool
    mensaje: str
    nombre: Optional[str] = None

class AccionGlobalResponse(BaseModel):
    ok: bool
    mensaje: str

class PreguntaRequest(BaseModel):
    pregunta: str
    # NUEVO: Indica al RAG qué modelo usar ("local" o "cloud")
    motor: str = "local" 

class ChatResponse(BaseModel):
    pregunta_original: str
    respuesta: str