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
from app.api.routers import rag_params    # parámetros RAG editables
from app.api.routers import evaluacion    # evaluador RAG
from app.api.routers import monitor       # ← NUEVO: monitoreo de concurrencia

# ── PHOENIX TRACING ───────────────────────────────────────────────────────────
import phoenix as px
px.launch_app()
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
app.include_router(rag_params.router)
app.include_router(evaluacion.router)
app.include_router(monitor.router)    # ← NUEVO

# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status":        "ok",
        "message":       "Servidor Backend RAG (Arquitectura Dual) en línea.",
        "version":       "3.0.0",
        "modos_activos": ["local:local", "local:cloud", "cloud:cloud"],
    }


# ── PRECALENTAMIENTO EN STARTUP ───────────────────────────────────────────────
@app.on_event("startup")
async def precalentar_sistema():
    """
    Al iniciar el servidor:
      1. Migra los prompts NULL en BD al texto hardcodeado por defecto.
      2. Pre-carga el modelo LLM local en memoria (Ollama).
      3. Pre-popula el caché con las preguntas más frecuentes del sistema.
    """
    from app.services.rag_params_service import (
        _get_or_create,
        PROMPT_PRINCIPAL_DEFAULT,
        PROMPT_HYDE_DEFAULT,
    )
    from app.db.database import SessionLocal
    from app.services.rag_service import consultar_base_conocimiento
    from app.services.config_service import obtener_motor_activo, obtener_configuracion

    # ── 1. MIGRACIÓN DE PROMPTS NULL ──────────────────────────────────────────
    try:
        db = SessionLocal()
        config = _get_or_create(db)
        actualizado = False

        if not config.prompt_principal:
            config.prompt_principal = PROMPT_PRINCIPAL_DEFAULT
            actualizado = True

        if not config.prompt_hyde:
            config.prompt_hyde = PROMPT_HYDE_DEFAULT
            actualizado = True

        if actualizado:
            db.commit()
            print("[STARTUP] ✅ Prompts guardados en BD (migración desde NULL completada).")
        else:
            print("[STARTUP] ✅ Prompts ya presentes en BD, no requieren migración.")

        db.close()
    except Exception as e:
        print(f"[STARTUP] ⚠️  Error al migrar prompts en BD: {e}")
        print("[STARTUP]    El servidor continúa iniciando normalmente.")

    # ── 2 y 3. PRECALENTAMIENTO DEL SISTEMA RAG ───────────────────────────────
    motor_activo = obtener_motor_activo()
    config_motor = obtener_configuracion()
    motor_llm    = config_motor.get("motor_llm", "local")

    if motor_llm == "cloud":
        print(f"[STARTUP] ⏭️  Precalentamiento omitido — motor LLM activo es cloud ({motor_activo}).")
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
        print(f"[STARTUP] ⚠️  Precalentamiento omitido: {e}")
        print("[STARTUP]    El servidor continúa iniciando normalmente.")