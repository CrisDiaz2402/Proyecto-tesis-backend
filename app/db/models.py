# app/db/models.py
import uuid
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.sql import func
from app.db.database import Base

class Usuario(Base):
    __tablename__ = "usuarios"

    # UUID v4 autogenerado, indexado para búsquedas ultrarrápidas
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    rol = Column(String, default="Admin")
    fecha_creacion = Column(DateTime(timezone=True), server_default=func.now())

class Documento(Base):
    __tablename__ = "documentos"

    # Identificadores base (Compartidos para ambos ecosistemas)
    id             = Column(Integer, primary_key=True, index=True)
    nombre_archivo = Column(String, index=True)
    subido_por     = Column(String, default="Admin")
    fecha_subida   = Column(DateTime(timezone=True), server_default=func.now())
    
    # ─── ECOSISTEMA LOCAL ───────────────────────────────────────────────────
    ruta_local      = Column(String, nullable=True) # Dónde está el PDF físico local
    procesado_local = Column(Boolean, default=False) # ¿Ya está vectorizado en Ollama?
    estado_local    = Column(String, default="No subido") # Texto para tu frontend

    # ─── ECOSISTEMA CLOUD ───────────────────────────────────────────────────
    ruta_cloud      = Column(String, nullable=True) # Dónde está el PDF físico cloud
    procesado_cloud = Column(Boolean, default=False) # ¿Ya está vectorizado en Google?
    estado_cloud    = Column(String, default="No subido") # Texto para tu frontend

class CacheRespuesta(Base):
    __tablename__ = "cache_respuestas"

    id               = Column(Integer, primary_key=True, index=True)
    pregunta         = Column(String, nullable=False)
    respuesta        = Column(String, nullable=False)
    similitud_score  = Column(Float,  nullable=True)
    
    # NUEVO: Para saber qué modelo de IA dio esta respuesta (ej. "local" o "cloud")
    motor            = Column(String, default="local", index=True) 
    
    fecha_creacion   = Column(DateTime(timezone=True), server_default=func.now())
    veces_consultado = Column(Integer, default=1)