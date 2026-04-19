# app/services/vllm_service.py
"""
Servicio centralizado para llamadas a LLM y embeddings.
- vLLM (local) vía API OpenAI-compatible.
- Google Gemini (cloud).
- sentence-transformers para embeddings locales (CPU, sin Ollama).
"""

import os
from typing import AsyncGenerator

import httpx
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from app.core.config import (
    VLLM_BASE_URL,
    LLM_MODEL_LOCAL,
    LLM_MODEL_CLOUD,
    EMBED_MODEL_LOCAL,
    GOOGLE_API_KEY,
)

# ─── Configurar Gemini ──────────────────────────────────────────────────────
genai.configure(api_key=GOOGLE_API_KEY)

# ─── Singleton del modelo de embeddings local ────────────────────────────────
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    """Instancia el modelo de embeddings una sola vez (singleton)."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_LOCAL, device="cpu")
        print(f"[VLLM_SERVICE] ✅ Modelo de embeddings cargado: {EMBED_MODEL_LOCAL}")
    return _embed_model


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def generar_embedding_local(texto: str) -> list[float]:
    """
    Genera embedding usando sentence-transformers (CPU).
    Devuelve una lista de floats normalizados.
    """
    model = _get_embed_model()
    embedding = model.encode(texto, normalize_embeddings=True)
    return embedding.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# LLM — vLLM LOCAL (síncrono)
# ─────────────────────────────────────────────────────────────────────────────

def generar_respuesta_local(prompt: str, num_tokens: int = 512) -> str:
    """
    Llama al endpoint OpenAI-compatible de vLLM vía httpx (síncrono).
    Timeout de 120s para generaciones largas.
    """
    url = f"{VLLM_BASE_URL}/chat/completions"
    payload = {
        "model": LLM_MODEL_LOCAL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": num_tokens,
        "temperature": 0,
        "top_p": 0.9,
        "stop": ["Consulta del usuario:", "Usuario:", "Pregunta:", "[FIN]"],
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# LLM — vLLM LOCAL (streaming async para WebSocket)
# ─────────────────────────────────────────────────────────────────────────────

async def generar_respuesta_stream_local(prompt: str, num_tokens: int = 512) -> AsyncGenerator[str, None]:
    """
    Async generator que hace streaming token por token desde vLLM.
    Usa el endpoint /chat/completions con stream=true.
    """
    import json as _json

    url = f"{VLLM_BASE_URL}/chat/completions"
    payload = {
        "model": LLM_MODEL_LOCAL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": num_tokens,
        "temperature": 0,
        "top_p": 0.9,
        "stream": True,
        "stop": ["Consulta del usuario:", "Usuario:", "Pregunta:", "[FIN]"],
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (KeyError, _json.JSONDecodeError):
                    continue


# ─────────────────────────────────────────────────────────────────────────────
# LLM — GOOGLE GEMINI (cloud)
# ─────────────────────────────────────────────────────────────────────────────

def generar_respuesta_cloud(prompt: str, num_tokens: int = 1024) -> str:
    """Invoca Google Gemini directamente."""
    model = genai.GenerativeModel(
        model_name=LLM_MODEL_CLOUD,
        generation_config=genai.GenerationConfig(
            temperature=0,
            max_output_tokens=num_tokens,
        ),
    )
    response = model.generate_content(prompt)
    return response.text
