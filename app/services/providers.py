# app/services/providers.py
from abc import ABC, abstractmethod
import httpx
from app.core.config import (
    VLLM_BASE_URL, LLM_MODEL_LOCAL, LLM_MODEL_CLOUD,
)
from app.core.exceptions import LLMError

_VLLM_ASYNC_CLIENT: httpx.AsyncClient | None = None


def _get_vllm_async_client() -> httpx.AsyncClient:
    global _VLLM_ASYNC_CLIENT
    if _VLLM_ASYNC_CLIENT is None or _VLLM_ASYNC_CLIENT.is_closed:
        _VLLM_ASYNC_CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )
    return _VLLM_ASYNC_CLIENT


class LLMAdapter(ABC):
    @abstractmethod
    async def completar(self, prompt: str, max_tokens: int) -> str: ...

class VLLMAdapter(LLMAdapter):
    async def completar(self, prompt: str, max_tokens: int) -> str:
        url = f"{VLLM_BASE_URL}/chat/completions"
        payload = {
            "model": LLM_MODEL_LOCAL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            # stop tokens para cortar alucinaciones de turno en el modelo Qwen
            "stop": ["Consulta del usuario:", "Usuario:", "Pregunta:", "[FIN]"],
        }
        client = _get_vllm_async_client()
        try:
            r = await client.post(url, json=payload)
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
            raise LLMError(
                f"vLLM respondió {status}: "
                f"{detail or 'prompt demasiado largo o modelo no disponible'}"
            )
        except httpx.ConnectError:
            print("[VLLM] ❌ No se pudo conectar a vLLM en", VLLM_BASE_URL)
            raise LLMError("vLLM no está disponible. Verifica que el servidor esté corriendo.")
        except httpx.ReadTimeout:
            print("[VLLM] ❌ Timeout esperando respuesta de vLLM (>120 s)")
            raise LLMError("Timeout esperando respuesta del modelo. El prompt puede ser demasiado largo.")

class GeminiAdapter(LLMAdapter):
    async def completar(self, prompt: str, max_tokens: int) -> str:
        import asyncio
        import google.generativeai as genai
        loop = asyncio.get_event_loop()

        def _llamar_gemini():
            model = genai.GenerativeModel(
                model_name=LLM_MODEL_CLOUD,
                generation_config=genai.GenerationConfig(
                    temperature=0,
                    max_output_tokens=max_tokens,
                ),
            )
            return model.generate_content(prompt).text

        return await loop.run_in_executor(None, _llamar_gemini)

class LLMProvider(ABC):
    @abstractmethod
    def generar_embedding(self, texto: str) -> list[float]: ...

    @abstractmethod
    async def generar_respuesta(self, prompt: str, num_tokens: int) -> str: ...


class LocalLLMProvider(LLMProvider):
    def __init__(self):
        self._adapter = VLLMAdapter()

    def generar_embedding(self, texto: str) -> list[float]:
        from app.core.singletons import EmbedModelSingleton
        return EmbedModelSingleton().model.encode(texto, normalize_embeddings=True).tolist()

    async def generar_respuesta(self, prompt: str, num_tokens: int) -> str:
        return await self._adapter.completar(prompt, num_tokens)


class CloudLLMProvider(LLMProvider):
    def __init__(self):
        self._adapter = GeminiAdapter()

    def generar_embedding(self, texto: str) -> list[float]:
        raise NotImplementedError(
            "CloudLLMProvider no soporta embeddings. "
            "Use motor_vectores='local' para generar embeddings."
        )

    async def generar_respuesta(self, prompt: str, num_tokens: int) -> str:
        return await self._adapter.completar(prompt, num_tokens)


def crear_proveedor_llm(motor: str) -> LLMProvider:
    proveedores = {"local": LocalLLMProvider, "cloud": CloudLLMProvider}
    cls = proveedores.get(motor)
    if not cls:
        raise ValueError(f"Motor desconocido: {motor}")
    return cls()