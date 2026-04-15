# app/core/constants.py
"""
Constantes centralizadas del sistema RAG.
Única fuente de verdad para etiquetas, validaciones y configuraciones.
"""

from typing import FrozenSet

# ── MOTORES Y COMBINACIONES ──────────────────────────────────────────────────────
# Usado en: configuracion.py, documents.py, rag_params.py, cache_service.py

# Motores individuales
MOTOR_LABELS = {
    "local": "Local",
    "cloud": "Nube",
    "all":   "Local + Nube",
}

# Combinaciones de motor vectorial + LLM
MODO_LABELS = {
    ("local", "local"):  "Todo Local (Ollama + Ollama)",
    ("cloud", "cloud"):  "Todo Nube (Gemini + Gemini)",
    ("local", "cloud"):  "Vectores Local + LLM Nube",
}

# Validaciones
MOTORES_VALIDOS: FrozenSet[str] = frozenset({"local", "cloud"})
COMBINACIONES_INVALIDAS: FrozenSet[tuple] = frozenset({
    ("cloud", "local"),  # Vectores nube + LLM local no soportado
})

# ── CONFIGURACION FUTURA vLLM+Qdrant ────────────────────────────────────────────
# En la migración: un solo motor, configuración simplificada

# MOTOR_QDRANT_URL = "http://localhost:6333"
# MOTOR_VLLM_URL = "http://localhost:8000"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── ESTADOS Y TIPOS ────────────────────────────────────────────────────────────

ROLES_USUARIO = {
    "admin": "Administrador",
    "user": "Usuario",  # para futuro
}

ESTADOS_DOCUMENTO = {
    "procesando": "Procesando",
    "listo": "Disponible",
    "error": "Error",
}

TIPOS_WEBSOCKET = {
    "estado": "status",
    "token": "token", 
    "final": "complete",
    "error": "error",
}
