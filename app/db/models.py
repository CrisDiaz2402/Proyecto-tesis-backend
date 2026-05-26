import uuid
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from app.db.database import Base


class Usuario(Base):
    __tablename__ = "usuarios"

    id              = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    username        = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    rol             = Column(String, default="Admin")
    fecha_creacion  = Column(DateTime(timezone=True), server_default=func.now())


class Documento(Base):
    __tablename__ = "documentos"

    id              = Column(Integer, primary_key=True, index=True)
    nombre_archivo  = Column(String, index=True)
    subido_por      = Column(String, default="Admin")
    fecha_subida    = Column(DateTime(timezone=True), server_default=func.now())
    ruta_local      = Column(String, nullable=True)
    procesado_local = Column(Boolean, default=False)
    estado_local    = Column(String, default="No subido")
