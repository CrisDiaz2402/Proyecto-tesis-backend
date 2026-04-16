# app/db/models.py
import uuid
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text
from sqlalchemy.sql import func
from app.db.database import Base


class Usuario(Base):
    __tablename__ = "usuarios"

    id             = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    username       = Column(String, unique=True, index=True, nullable=False)
    hashed_password= Column(String, nullable=False)
    rol            = Column(String, default="Admin")
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())


class Documento(Base):
    __tablename__ = "documentos"

    id             = Column(Integer, primary_key=True, index=True)
    nombre_archivo = Column(String, index=True)
    subido_por     = Column(String, default="Admin")
    fecha_subida   = Column(DateTime(timezone=True), server_default=func.now())

    # ─── ECOSISTEMA LOCAL ────────────────────────────────────────────────────
    ruta_local      = Column(String, nullable=True)
    procesado_local = Column(Boolean, default=False)
    estado_local    = Column(String, default="No subido")

    # ─── ECOSISTEMA CLOUD ────────────────────────────────────────────────────
    ruta_cloud      = Column(String, nullable=True)
    procesado_cloud = Column(Boolean, default=False)
    estado_cloud    = Column(String, default="No subido")


class ConfiguracionRAG(Base):
    __tablename__ = "configuracion_rag"

    id = Column(Integer, primary_key=True, default=1)

    umbral_relevancia_local = Column(Float, default=0.15)
    umbral_relevancia_cloud = Column(Float, default=0.30)

    rag_k_local = Column(Integer, default=10)
    rag_k_cloud = Column(Integer, default=8)

    prompt_principal = Column(Text, nullable=True, default=None)

    fecha_actualizacion = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )