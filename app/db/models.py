# app/db/models.py
from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.sql import func
from app.db.database import Base


class Documento(Base):
    __tablename__ = "documentos"

    id             = Column(Integer, primary_key=True, index=True)
    nombre_archivo = Column(String, index=True)
    ruta_archivo   = Column(String, nullable=True)   # path físico en disco
    subido_por     = Column(String, default="Admin")
    fecha_subida   = Column(DateTime(timezone=True), server_default=func.now())
    estado         = Column(String, default="Procesado en ChromaDB")


class CacheRespuesta(Base):
    __tablename__ = "cache_respuestas"

    id               = Column(Integer, primary_key=True, index=True)
    pregunta         = Column(String, nullable=False)
    respuesta        = Column(String, nullable=False)
    similitud_score  = Column(Float,  nullable=True)
    fecha_creacion   = Column(DateTime(timezone=True), server_default=func.now())
    veces_consultado = Column(Integer, default=1)