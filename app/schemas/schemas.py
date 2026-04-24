# app/schemas/schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class DocumentoOut(BaseModel):
    id: int
    nombre_archivo: str
    subido_por: str
    fecha_subida: datetime
    procesado_local: bool
    estado_local: str

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
    motor: str = "local" 

class ChatResponse(BaseModel):
    pregunta_original: str
    respuesta: str

class UsuarioBase(BaseModel):
    username: str
    rol: Optional[str] = "Admin"

class UsuarioCreate(UsuarioBase):
    password: str  

class UsuarioUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    rol: Optional[str] = None

class UsuarioOut(UsuarioBase):
    id: str    
    fecha_creacion: datetime
    
    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    rol: str

class NLUConfigOut(BaseModel):
    palabras_saludo: list[str]
    frases_despedida: list[str]
    frases_agradecimiento: list[str]
    palabras_lista_larga: list[str]
    frases_rechazo: list[str]
    mensaje_saludo: str
    mensaje_despedida: str
    mensaje_agradecimiento: str
    mensaje_fuera_de_tema: str
    mensaje_sin_resultados: str

    class Config:
        from_attributes = True


class NLUConfigUpdate(BaseModel):
    palabras_saludo: Optional[list[str]] = None
    frases_despedida: Optional[list[str]] = None
    frases_agradecimiento: Optional[list[str]] = None
    palabras_lista_larga: Optional[list[str]] = None
    frases_rechazo: Optional[list[str]] = None
    mensaje_saludo: Optional[str] = None
    mensaje_despedida: Optional[str] = None
    mensaje_agradecimiento: Optional[str] = None
    mensaje_fuera_de_tema: Optional[str] = None
    mensaje_sin_resultados: Optional[str] = None