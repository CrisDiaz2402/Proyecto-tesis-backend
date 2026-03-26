# app/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Base de datos
from app.db.database import engine
from app.db import models

# Enrutadores
from app.api.routers import documents, chat

# ── PHOENIX TRACING (Métricas de IA) ──────────────────────────────────────────
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

# Inicializar Base de Datos (Crea tablas si no existen)
models.Base.metadata.create_all(bind=engine)

# Inicializar FastAPI
app = FastAPI(title="API Asistente RAG EPN", version="1.0.0")

# ── CORS (Comunicación segura con Frontend Vue) ───────────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,              # Acceso por IP desde otras computadoras (Intranet)
        "http://localhost:5173",   # Acceso local para ti mismo en la compu buena
        "http://127.0.0.1:5173",   # Respaldo local
        "http://localhost:80"      # Respaldo de producción
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REGISTRO DE ENRUTADORES ───────────────────────────────────────────────────
app.include_router(documents.router)
app.include_router(chat.router)

# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def health_check():
    """Verifica que el servidor backend responda correctamente."""
    return {
        "status": "ok", 
        "message": "Servidor Backend RAG en línea."
    }