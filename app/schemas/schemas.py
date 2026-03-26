# app/schemas/schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class DocumentoOut(BaseModel):
    id: int
    nombre_archivo: str
    subido_por: str
    fecha_subida: datetime
    estado: str

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

class ChatResponse(BaseModel):
    pregunta_original: str
    respuesta: str