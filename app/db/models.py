# app/db/models.py
import uuid
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, JSON
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
    ruta_local      = Column(String, nullable=True)
    procesado_local = Column(Boolean, default=False)
    estado_local    = Column(String, default="No subido")


class ConfiguracionRAG(Base):
    __tablename__ = "configuracion_rag"

    id = Column(Integer, primary_key=True, default=1)

    umbral_relevancia_local = Column(Float, default=0.15)

    rag_k_local = Column(Integer, default=10)

    prompt_principal = Column(Text, nullable=True, default=None)

    fecha_actualizacion = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )


DEFAULTS_NLU = {
    "palabras_saludo": [
        "hola", "buenos días", "buenas tardes", "buenas noches",
        "buen día", "buenas", "hey", "saludos", "hi", "hello",
    ],
    "frases_despedida": [
        "adiós", "adios", "hasta luego", "chao", "chau",
        "nos vemos", "hasta pronto", "bye",
    ],
    "frases_agradecimiento": [
        "gracias", "muchas gracias", "te agradezco", "gracias por tu ayuda",
        "muy amable", "perfecto gracias",
    ],
    "palabras_lista_larga": [
        "todas las materias", "todos los niveles", "lista completa",
        "enumera todas", "todos los semestres",
        "qué materias hay en", "materias del nivel", "cuáles son todas",
        "prerrequisitos transitivos", "debería haber aprobado antes",
        "sin ningún prerrequisito", "no tienen prerrequisito",
        "qué necesito para graduarme", "requisitos para graduarme",
        "qué requisitos", "cuáles son los requisitos",
    ],
    "frases_rechazo": [
        "no encontré información",
        "no está disponible",
        "lo siento",
        "no tengo informacion",
        "no hay informacion",
        "no se encuentra",
        "no consta",
        "no puedo",
        "no dispongo",
        "no cuento con esa información",
    ],
    "mensaje_saludo": "¡Hola! Soy el Asistente Académico de la EPN. ¿En qué puedo ayudarte hoy?",
    "mensaje_despedida": "¡Hasta luego! Si tienes más consultas académicas, aquí estaré.",
    "mensaje_agradecimiento": "Con gusto. ¿Hay algo más en lo que pueda ayudarte?",
    "mensaje_fuera_de_tema": "Solo puedo ayudarte con consultas académicas de la EPN. ¿Tienes alguna pregunta sobre materias, créditos o requisitos de graduación?",
    "mensaje_sin_resultados": "No encontré información sobre eso en los documentos académicos disponibles. Intenta reformular tu pregunta.",
}


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