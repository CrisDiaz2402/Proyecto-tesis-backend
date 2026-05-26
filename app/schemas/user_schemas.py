from pydantic import BaseModel
from datetime import datetime
from typing import Optional

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