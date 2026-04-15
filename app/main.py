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

# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status":        "ok",
        "message":       "Servidor Backend RAG (Arquitectura Dual) en línea.",
        "version":       "3.0.0",
        "modos_activos": ["local:local", "local:cloud", "cloud:cloud"],
    }


# ── SEED AUTOMÁTICO AL STARTUP ───────────────────────────────────────────────
@app.on_event("startup")
async def inicializar_sistema():
    """
    Seed automático al arranque:
      1. Crea el usuario admin si no existe.
      2. Crea la configuración RAG por defecto si no existe.
      3. Precalienta el modelo local (solo si motor_llm = local).
    """
    from app.db.database import SessionLocal
    from app.db import models
    from app.core.security import get_password_hash
    from app.services.rag_params_service import _get_or_create
    from app.services.config_service import obtener_configuracion

    db = SessionLocal()
    try:
        # ── 1. USUARIO ADMIN POR DEFECTO ─────────────────────────────────────
        admin_existente = db.query(models.Usuario).filter(
            models.Usuario.username == "admin"
        ).first()

        if not admin_existente:
            admin = models.Usuario(
                username="admin",
                hashed_password=get_password_hash("admin123"),
                rol="Admin",
            )
            db.add(admin)
            db.commit()
            print("[STARTUP] ✅ Usuario admin creado (usuario: admin / contraseña: admin123).")
        else:
            print("[STARTUP] ✅ Usuario admin ya existe.")

        # ── 2. CONFIGURACIÓN RAG POR DEFECTO ─────────────────────────────────
        config = _get_or_create(db)
        # _get_or_create ya maneja la creación con todos los defaults si no existe
        print(f"[STARTUP] ✅ Configuración RAG lista (id={config.id}).")

    except Exception as e:
        print(f"[STARTUP] ⚠️  Error en seed inicial: {e}")
    finally:
        db.close()

    # ── 3. PRECALENTAMIENTO DEL MODELO LOCAL ─────────────────────────────────
    try:
        from app.services.config_service import obtener_motor_activo
        from app.services.rag_service import consultar_base_conocimiento
        import asyncio

        config_motor = obtener_configuracion()
        if config_motor.get("motor_llm") == "local":
            motor_activo = obtener_motor_activo()
            print(f"[STARTUP] 🔥 Precalentando modelo local (motor: {motor_activo})...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: consultar_base_conocimiento("¿Cuántos créditos necesito para graduarme?", motor=motor_activo),
            )
            print("[STARTUP] ✅ Precalentamiento completado.")
        else:
            print("[STARTUP] ⏭️  Motor cloud activo — precalentamiento omitido.")
    except Exception as e:
        print(f"[STARTUP] ⚠️  Precalentamiento omitido: {e}")