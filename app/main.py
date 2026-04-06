# app/main.py
import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Base de datos
from app.db.database import engine
from app.db import models

# Enrutadores
from app.api.routers import documents, chat, usuarios, auth, configuracion

# ── PHOENIX TRACING ───────────────────────────────────────────────────────────
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

px.launch_app()
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter("http://127.0.0.1:6006/v1/traces"))
)
trace_api.set_tracer_provider(tracer_provider)
LangChainInstrumentor().instrument()
# ─────────────────────────────────────────────────────────────────────────────

# Inicializar Base de Datos
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="API Asistente RAG EPN (Dual Local/Cloud)", version="3.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:80",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── ENRUTADORES ───────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(usuarios.router)
app.include_router(configuracion.router)

# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "message": "Servidor Backend RAG (Arquitectura Dual) en línea.",
        "version": "3.0.0",
        "modos_activos": ["local:local", "local:cloud", "cloud:cloud"],
    }


# ── PRECALENTAMIENTO EN STARTUP ───────────────────────────────────────────────
@app.on_event("startup")
async def precalentar_sistema():
    """
    Al iniciar el servidor, pre-carga el modelo LLM local en memoria (Ollama)
    y pre-popula el caché con las preguntas más frecuentes del sistema.

    Solo se ejecuta cuando el motor LLM activo es LOCAL, porque:
    - El precalentamiento con LLM cloud (Gemini) consume quota de API innecesariamente.
    - Ollama sí necesita el precalentamiento para evitar la latencia de carga
      inicial del modelo (~5-10 segundos en llama3.1:8b).
    - Con LLM cloud la primera consulta real es igualmente rápida sin precalentar.
    """
    from app.services.rag_service import consultar_base_conocimiento
    from app.services.config_service import obtener_motor_activo, obtener_configuracion

    motor_activo = obtener_motor_activo()
    config       = obtener_configuracion()
    motor_llm    = config.get("motor_llm", "local")

    # No precalentar si el LLM activo es cloud: preservar quota de API
    if motor_llm == "cloud":
        print(f"[STARTUP] ⏭️  Precalentamiento omitido — motor LLM activo es cloud ({motor_activo}).")
        print("[STARTUP]    El precalentamiento solo aplica al LLM local (Ollama).")
        return

    print(f"[STARTUP] 🔥 Iniciando precalentamiento del sistema RAG (motor: {motor_activo})...")

    preguntas_frecuentes = [
        "¿Cuántos créditos necesito para graduarme?",
        "¿Cuántos semestres dura la carrera?",
        "¿Cuáles son los requisitos para graduarme?",
        "¿Cuántas materias tiene la carrera?",
        "¿Cuántas horas de prácticas laborales necesito?",
    ]

    try:
        loop = asyncio.get_event_loop()
        for pregunta in preguntas_frecuentes:
            await loop.run_in_executor(
                None,
                lambda p=pregunta: consultar_base_conocimiento(p, motor=motor_activo),
            )
        print(f"[STARTUP] ✅ Precalentamiento completado — {len(preguntas_frecuentes)} preguntas procesadas.")
    except Exception as e:
        # El precalentamiento es opcional: si falla (p.ej. no hay documentos aún),
        # el servidor debe arrancar igual.
        print(f"[STARTUP] ⚠️  Precalentamiento omitido: {e}")
        print("[STARTUP]    El servidor continúa iniciando normalmente.")