# app/schemas/user_schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UsuarioBase(BaseModel):
    username: str
    rol: Optional[str] = "Admin"

class UsuarioCreate(UsuarioBase):
    password: str  # El frontend nos envía la contraseña en texto plano

class UsuarioUpdate(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    rol: Optional[str] = None

class UsuarioOut(UsuarioBase):
    id: str        # Ahora es un string (UUID)
    fecha_creacion: datetime
    
    # Esta configuración permite leer directamente desde el modelo de SQLAlchemy
    class Config:
        from_attributes = True