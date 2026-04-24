import uuid
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, JSON
from sqlalchemy.sql import func
from app.db.database import Base
from app.core.defaults import DEFAULTS_NLU

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
    ruta_local      = Column(String, nullable=True)
    procesado_local = Column(Boolean, default=False)
    estado_local    = Column(String, default="No subido")


class ConfiguracionRAG(Base):
    __tablename__ = "configuracion_rag"

    id = Column(Integer, primary_key=True, default=1)
    umbral_relevancia_local = Column(Float, default=0.15)
    rag_k_local = Column(Integer, default=10)
    prompt_principal = Column(Text, nullable=True, default=None)
    system_prompt = Column(Text, nullable=True, default=None)
    fecha_actualizacion = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class ConfiguracionNLU(Base):
    __tablename__ = "configuracion_nlu"

    id = Column(Integer, primary_key=True, default=1)
    palabras_saludo = Column(JSON, default=DEFAULTS_NLU["palabras_saludo"])
    frases_despedida = Column(JSON, default=DEFAULTS_NLU["frases_despedida"])
    frases_agradecimiento = Column(JSON, default=DEFAULTS_NLU["frases_agradecimiento"])
    palabras_lista_larga = Column(JSON, default=DEFAULTS_NLU["palabras_lista_larga"])
    frases_rechazo = Column(JSON, default=DEFAULTS_NLU["frases_rechazo"])
    mensaje_saludo = Column(Text, default=DEFAULTS_NLU["mensaje_saludo"])
    mensaje_despedida = Column(Text, default=DEFAULTS_NLU["mensaje_despedida"])
    mensaje_agradecimiento = Column(Text, default=DEFAULTS_NLU["mensaje_agradecimiento"])
    mensaje_fuera_de_tema = Column(Text, default=DEFAULTS_NLU["mensaje_fuera_de_tema"])
    mensaje_sin_resultados = Column(Text, default=DEFAULTS_NLU["mensaje_sin_resultados"])
    fecha_actualizacion = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


class ConfiguracionMotor(Base):
    __tablename__ = "configuracion_motor"

    id = Column(Integer, primary_key=True, default=1)
    motor_vectores = Column(String, default="local", nullable=False)
    motor_llm = Column(String, default="local", nullable=False)
    fecha_actualizacion = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )