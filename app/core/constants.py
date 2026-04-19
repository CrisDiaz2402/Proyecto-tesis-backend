# app/core/constants.py
from typing import FrozenSet
MOTOR_LABELS = {
    "local": "Local",
}

MODO_LABELS = {
    ("local", "local"):  "Todo Local (vLLM + sentence-transformers)",
    ("local", "cloud"):  "Vectores Local + LLM Nube (Gemini)",
}

MOTORES_VECTORES_VALIDOS: FrozenSet[str] = frozenset({"local"})
MOTORES_LLM_VALIDOS: FrozenSet[str] = frozenset({"local", "cloud"})
COMBINACIONES_VALIDAS: FrozenSet[tuple] = frozenset({("local", "local"), ("local", "cloud")})

TIPOS_WEBSOCKET = {
    "estado": "status",
    "token": "token", 
    "final": "complete",
    "error": "error",
}
