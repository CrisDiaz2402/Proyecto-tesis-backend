# app/services/providers.py
from abc import ABC, abstractmethod
import httpx
from app.core.config import (
    VLLM_BASE_URL, LLM_MODEL_LOCAL, LLM_MODEL_CLOUD,
)
from app.core.exceptions import LLMError


class LLMAdapter(ABC):
    @abstractmethod
    def completar(self, prompt: str, max_tokens: int) -> str: ...


class VLLMAdapter(LLMAdapter):
    def completar(self, prompt: str, max_tokens: int) -> str:
        url = f"{VLLM_BASE_URL}/chat/completions"
        payload = {
            "model": LLM_MODEL_LOCAL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        try:
            with httpx.Client(timeout=120.0) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            detail = ""
            try:
                detail = e.response.json().get("message", "")
            except Exception:
                pass
            print(f"[VLLM] ❌ HTTP {status} — {detail or str(e)[:120]}")
            raise LLMError(f"vLLM respondió {status}: {detail or 'prompt demasiado largo o modelo no disponible'}")
        except httpx.ConnectError:
            print("[VLLM] ❌ No se pudo conectar a vLLM en", VLLM_BASE_URL)
            raise LLMError("vLLM no está disponible. Verifica que el servidor esté corriendo.")


class GeminiAdapter(LLMAdapter):
    def completar(self, prompt: str, max_tokens: int) -> str:
        import google.generativeai as genai
        model = genai.GenerativeModel(
            model_name=LLM_MODEL_CLOUD,
            generation_config=genai.GenerationConfig(temperature=0, max_output_tokens=max_tokens),
        )
        return model.generate_content(prompt).text


class LLMProvider(ABC):
    @abstractmethod
    def generar_embedding(self, texto: str) -> list[float]: ...
    @abstractmethod
    def generar_respuesta(self, prompt: str, num_tokens: int) -> str: ...


class LocalLLMProvider(LLMProvider):
    def __init__(self):
        self._adapter = VLLMAdapter()

    def generar_embedding(self, texto: str) -> list[float]:
        from app.core.singletons import EmbedModelSingleton
        return EmbedModelSingleton().model.encode(texto, normalize_embeddings=True).tolist()

    def generar_respuesta(self, prompt: str, num_tokens: int) -> str:
        return self._adapter.completar(prompt, num_tokens)


class CloudLLMProvider(LLMProvider):
    def __init__(self):
        self._adapter = GeminiAdapter()

    def generar_embedding(self, texto: str) -> list[float]:
        raise NotImplementedError(
            "CloudLLMProvider no soporta embeddings. "
            "Use motor_vectores='local' para generar embeddings."
        )

    def generar_respuesta(self, prompt: str, num_tokens: int) -> str:
        return self._adapter.completar(prompt, num_tokens)


def crear_proveedor_llm(motor: str) -> LLMProvider:
    proveedores = {"local": LocalLLMProvider, "cloud": CloudLLMProvider}
    cls = proveedores.get(motor)
    if not cls:
        raise ValueError(f"Motor desconocido: {motor}")
    return cls()
