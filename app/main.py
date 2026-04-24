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

from app.api.routers import documents, usuarios, auth, configuracion
from app.api.routers import rag_params  
from app.api.routers import ws_chat  
from app.api.routers import nlu_config
from app.api.routers import cache_admin

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
            print("[STARTUP] admin creado")

        config = _get_or_create(db)
        nlu_config_row = _get_or_create_nlu(db)

        import json
        from pathlib import Path
        _config_json = Path("./config_ia.json")
        if _config_json.exists():
            try:
                with open(_config_json, "r", encoding="utf-8") as f:
                    _data = json.load(f)
                if "motor_activo" in _data and "motor_vectores" not in _data:
                    _mv = _data.get("motor_activo", "local")
                    _ml = _mv
                else:
                    _mv = _data.get("motor_vectores", "local")
                    _ml = _data.get("motor_llm", "local")
                from app.db import models as _models
                _cm = db.query(_models.ConfiguracionMotor).filter_by(id=1).first()
                if _cm is None:
                    _cm = _models.ConfiguracionMotor(id=1, motor_vectores=_mv, motor_llm=_ml)
                    db.add(_cm)
                else:
                    _cm.motor_vectores = _mv
                    _cm.motor_llm = _ml
                db.commit()
                _config_json.unlink()
                print(f"[STARTUP] Migración config_ia.json → BD completada ({_mv}:{_ml})")
            except Exception as _e:
                print(f"[STARTUP] Error en migración config_ia.json: {_e}")

    except Exception as e:
        print(f"[STARTUP]   Error en seed inicial: {e}")
    finally:
        db.close()

    try:
        from app.services.config_service import obtener_motor_activo
        from app.services.rag_service import consultar_base_conocimiento

        config_motor = obtener_configuracion()
        if config_motor.get("motor_llm") == "local":
            motor_activo = obtener_motor_activo()
            print(f"[STARTUP] precalentando modelo ({motor_activo})")
            await consultar_base_conocimiento(
                "¿Cuántos créditos necesito para graduarme?",
                motor=motor_activo,
            )
        else:
            print("[STARTUP] motor cloud activo, precalentamiento omitido")
    except Exception as e:
        print(f"[STARTUP]   Precalentamiento omitido: {e}")

    yield

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
app.include_router(usuarios.router)
app.include_router(configuracion.router)
app.include_router(rag_params.router)
app.include_router(ws_chat.router)
app.include_router(nlu_config.router)
app.include_router(cache_admin.router)

Instrumentator().instrument(app).expose(app)


@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status":        "ok",
        "message":       "Servidor Backend RAG (Arquitectura Dual) en línea.",
        "version":       "4.0.0",
        "modos_activos": ["local:local", "local:cloud"],
    }