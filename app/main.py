# app/main.py
import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Base de datos
from app.db.database import engine
from app.db import models

# Enrutadores
from app.api.routers import documents, chat, usuarios, auth, configuracion
from app.api.routers import rag_params    # parámetros RAG editables
from app.api.routers import evaluacion    # evaluador RAG
from app.api.routers import ws_chat       # WebSocket chat

# Inicializar Base de Datos (solo tablas relacionales)
models.Base.metadata.create_all(bind=engine)


# ── LIFESPAN (reemplaza @app.on_event) ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
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
        print(f"[STARTUP] ✅ Configuración RAG lista (id={config.id}).")

    except Exception as e:
        print(f"[STARTUP] ⚠️  Error en seed inicial: {e}")
    finally:
        db.close()

    # ── 3. PRECALENTAMIENTO DEL MODELO LOCAL ─────────────────────────────────
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
            print("[STARTUP] ⏭️  Motor cloud activo — precalentamiento omitido.")
    except Exception as e:
        print(f"[STARTUP] ⚠️  Precalentamiento omitido: {e}")

    yield  # La aplicación se ejecuta aquí

    # Cleanup al apagar
    print("[SHUTDOWN] 🛑 Apagando servidor...")


# ── APLICACIÓN ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="API Asistente RAG EPN (Dual Local/Cloud)",
    version="4.0.0",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────────────
# Lee los orígenes permitidos desde variables de entorno para que sea fácil
# cambiarlos sin modificar código.
#
# En desarrollo puedes dejar los defaults.
# En producción define en tu .env:
#   FRONTEND_URL=http://192.168.100.43:5173  (o la IP/dominio real del frontend)
#
# NOTA IMPORTANTE PARA WEBSOCKETS:
# CORSMiddleware de FastAPI NO protege las conexiones WebSocket — el navegador
# no envía el header Origin en el handshake WS de la misma forma que en HTTP.
# El control de acceso para WS se hace validando el token JWT en el endpoint.
# Por eso aquí ponemos allow_origins generosos para desarrollo; en producción
# restringe al dominio exacto del frontend.

FRONTEND_URL  = os.getenv("FRONTEND_URL",  "http://localhost:5173")
FRONTEND_URL2 = os.getenv("FRONTEND_URL2", "http://192.168.100.43:5173")

# Construir lista de orígenes permitidos sin duplicados
_origins_base = [
    FRONTEND_URL,
    FRONTEND_URL2,
    "http://localhost:5173",
    "http://localhost:5174",   # puerto alternativo de Vite
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://localhost:80",
    "http://localhost",
    "http://192.168.100.43:5173",
    "http://192.168.100.43:5173/"
]
ALLOWED_ORIGINS = list(dict.fromkeys(o for o in _origins_base if o))  # deduplica manteniendo orden

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=".*",  # <--- EL MARTILLO: Acepta cualquier origen, burlando el bloqueo del celular
    allow_credentials=True,   # necesario para que el frontend envíe cookies / headers de auth
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],     # permite que el cliente lea headers personalizados en las respuestas
)

# ── ENRUTADORES ───────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(usuarios.router)
app.include_router(configuracion.router)
app.include_router(rag_params.router)
app.include_router(evaluacion.router)
app.include_router(ws_chat.router)


# ── HEALTH CHECK ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def health_check():
    return {
        "status":          "ok",
        "message":         "Servidor Backend RAG (Arquitectura Dual) en línea.",
        "version":         "4.0.0",
        "modos_activos":   ["local:local", "local:cloud", "cloud:cloud"],
        "allowed_origins": ALLOWED_ORIGINS,
    }