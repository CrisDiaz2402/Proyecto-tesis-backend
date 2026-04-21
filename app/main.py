# app/main.py
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
from prometheus_fastapi_instrumentator import Instrumentator

from app.db.database import engine
from app.db import models

from app.api.routers import documents, chat, usuarios, auth, configuracion
from app.api.routers import rag_params  
from app.api.routers import evaluacion  
from app.api.routers import ws_chat  
from app.api.routers import nlu_config

models.Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.db.database import SessionLocal
    from app.db import models
    from app.core.security import get_password_hash
    from app.services.rag_params_service import _get_or_create
    from app.services.config_service import obtener_configuracion
    from app.services.nlu_config_service import _get_or_create as _get_or_create_nlu

    db = SessionLocal()
    try:
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

        config = _get_or_create(db)
        print(f"[STARTUP] ✅ Configuración RAG lista (id={config.id}).")

        nlu_config_row = _get_or_create_nlu(db)
        print(f"[STARTUP] ✅ Configuración NLU lista (id={nlu_config_row.id}).")

    except Exception as e:
        print(f"[STARTUP] ⚠️  Error en seed inicial: {e}")
    finally:
        db.close()

    try:
        from app.services.config_service import obtener_motor_activo
        from app.services.rag_service import consultar_base_conocimiento

        config_motor = obtener_configuracion()
        if config_motor.get("motor_llm") == "local":
            motor_activo = obtener_motor_activo()
            print(f"[STARTUP] 🔥 Precalentando modelo local (motor: {motor_activo})...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: consultar_base_conocimiento(
                    "¿Cuántos créditos necesito para graduarme?",
                    motor=motor_activo,
                ),
            )
            print("[STARTUP] ✅ Precalentamiento completado.")
        else:
            print("[STARTUP] ⏭️  Motor LLM cloud activo (local:cloud) — precalentamiento omitido.")
    except Exception as e:
        print(f"[STARTUP] ⚠️  Precalentamiento omitido: {e}")

    yield

    print("[SHUTDOWN] 🛑 Apagando servidor...")

class WSOriginBypassMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "websocket":
            scope["headers"] = [
                (name, value)
                for name, value in scope.get("headers", [])
                if name.lower() != b"origin"
            ]
        await self.app(scope, receive, send)

app = FastAPI(
    title="API Asistente RAG EPN (Local / Local+Cloud)",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(WSOriginBypassMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(usuarios.router)
app.include_router(configuracion.router)
app.include_router(rag_params.router)
app.include_router(evaluacion.router)
app.include_router(ws_chat.router)
app.include_router(nlu_config.router)

Instrumentator().instrument(app).expose(app)


@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status":        "ok",
        "message":       "Servidor Backend RAG (Arquitectura Dual) en línea.",
        "version":       "4.0.0",
        "modos_activos": ["local:local", "local:cloud"],
    }