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


class CacheRespuesta(Base):
    __tablename__ = "cache_respuestas"

    id               = Column(Integer, primary_key=True, index=True)
    pregunta         = Column(String, nullable=False)
    respuesta        = Column(String, nullable=False)
    similitud_score  = Column(Float,  nullable=True)
    motor            = Column(String, default="local", index=True)
    fecha_creacion   = Column(DateTime(timezone=True), server_default=func.now())
    veces_consultado = Column(Integer, default=1)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN RAG — Tabla singleton (siempre id=1).
# Almacena todos los parámetros editables del pipeline RAG.
# La fuente de verdad en runtime es esta tabla; config.py solo provee defaults
# de fallback en caso de que la BD no esté disponible.
# ─────────────────────────────────────────────────────────────────────────────
class ConfiguracionRAG(Base):
    __tablename__ = "configuracion_rag"

    id = Column(Integer, primary_key=True, default=1)

    # ── PARÁMETROS CON IMPACTO DEMOSTRABLE ────────────────────────────────────
    # Score mínimo de similitud coseno para que un fragmento pase al contexto.
    umbral_relevancia_local = Column(Float, default=0.15)
    umbral_relevancia_cloud = Column(Float, default=0.30)

    # Número de fragmentos a recuperar de ChromaDB por colección.
    rag_k_local = Column(Integer, default=10)
    rag_k_cloud = Column(Integer, default=8)

    # ── PROMPT PRINCIPAL ──────────────────────────────────────────────────────
    # NULL = usar el prompt hardcodeado por defecto en rag_service.py.
    prompt_principal = Column(Text, nullable=True, default=None)

    # ── Auditoría ─────────────────────────────────────────────────────────────
    fecha_actualizacion = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )