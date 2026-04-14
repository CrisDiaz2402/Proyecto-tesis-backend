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

    # ── ALTO IMPACTO — Chunking semántico ─────────────────────────────────────
    # Percentil de corte del SemanticChunker (50–95).
    # A mayor valor → chunks más pequeños y precisos.
    # Cambiar este valor OBLIGA a eliminar vectores y reindexar.
    breakpoint_threshold_amount = Column(Integer, default=75)

    # ── ALTO IMPACTO — Retrieval ──────────────────────────────────────────────
    # Score mínimo de similitud coseno para que un fragmento pase al contexto.
    umbral_relevancia_local = Column(Float, default=0.15)
    umbral_relevancia_cloud = Column(Float, default=0.30)

    # Número de fragmentos a recuperar de ChromaDB por colección.
    rag_k_local = Column(Integer, default=10)
    rag_k_cloud = Column(Integer, default=8)

    # ── MEDIO IMPACTO — Tokens de respuesta ───────────────────────────────────
    num_tokens_normal_local = Column(Integer, default=350)
    num_tokens_lista_local  = Column(Integer, default=750)
    num_tokens_normal_cloud = Column(Integer, default=700)
    num_tokens_lista_cloud  = Column(Integer, default=1400)

    # ── MEDIO IMPACTO — Umbrales de caché L2 ──────────────────────────────────
    cache_threshold_ll = Column(Float, default=0.82)   # modo local:local
    cache_threshold_lc = Column(Float, default=0.83)   # modo local:cloud
    cache_threshold_cc = Column(Float, default=0.88)   # modo cloud:cloud
    umbral_similitud   = Column(Float, default=0.02)   # margen global de hit

    # ── BAJO IMPACTO — Sampling del LLM local (Ollama) ───────────────────────
    repeat_penalty   = Column(Float,   default=1.3)
    top_k_llm        = Column(Integer, default=10)
    top_p_llm        = Column(Float,   default=0.5)
    hyde_num_predict = Column(Integer, default=120)

    # ── BAJO IMPACTO — Caché L1 RAM ───────────────────────────────────────────
    max_l1_entries = Column(Integer, default=500)

    # ── PROMPTS EDITABLES ─────────────────────────────────────────────────────
    # NULL = usar el prompt hardcodeado por defecto en rag_service.py.
    # Compatibilidad: columnas nullable → filas existentes en BD no se rompen.
    prompt_principal = Column(Text, nullable=True, default=None)
    prompt_hyde      = Column(Text, nullable=True, default=None)

    # ── Auditoría ─────────────────────────────────────────────────────────────
    fecha_actualizacion = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )