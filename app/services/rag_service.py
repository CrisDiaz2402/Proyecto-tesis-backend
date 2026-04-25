import os
import re
import json
import time
import asyncio
from typing import AsyncGenerator

import httpx
import pymupdf4llm
import docx2txt
import google.generativeai as genai

from app.db.database import SessionLocal
from app.db import models
from app.core.config import (
    LLM_MODEL_LOCAL, LLM_MODEL_CLOUD,
    EMBED_MODEL_LOCAL,
    GOOGLE_API_KEY, EMBED_DIMENSION_LOCAL,
    VLLM_BASE_URL,
)
from app.services.rag_params_service import get_params, construir_prompt_completo
from app.core.prompts import SYSTEM_PROMPT_EDITABLE, USER_TEMPLATE
from app.core.prompts import PROMPT_ERROR_FALLBACK
from app.core.exceptions import LLMError
from app.core.singletons import EmbedModelSingleton, HttpxClientSingleton
from app.core.defaults import (
    FRASES_NORMALIZACION_VACIA as _FRASES_VACIO,
    MARCADORES_IDIOMA_INCORRECTO as _MARCADORES_NO_ESPANOL,
    DEFAULTS_NLU as _DEFAULTS_NLU,
)

_PROMPT_PRINCIPAL_DEFAULT = construir_prompt_completo(SYSTEM_PROMPT_EDITABLE)

from app.services.qdrant_service import (
    insertar_puntos,
    eliminar_puntos_por_documento,
    eliminar_todos_los_puntos,
    buscar_similares,
    crear_coleccion,
)

from app.services.providers import crear_proveedor_llm

genai.configure(api_key=GOOGLE_API_KEY)

CHUNK_SIZE_LOCAL        = 400
CHUNK_OVERLAP           = 50
UMBRAL_BUSQUEDA_QDRANT  = 0.02
NUM_TOKENS_LOCAL        = 512
NUM_TOKENS_CLOUD        = 1024
MAX_CHARS_CONTEXTO      = 2200

_LLM_SEMAPHORE  = asyncio.Semaphore(3)
_EMBED_SEMAPHORE = asyncio.Semaphore(6)


async def _generar_embedding_async(texto: str, motor_vectores: str) -> list[float]:
    async with _EMBED_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: EmbedModelSingleton().model.encode(texto, normalize_embeddings=True).tolist()
        )


async def _invocar_llm(prompt: str, motor_llm: str, num_tokens: int) -> str:
    async with _LLM_SEMAPHORE:
        proveedor = crear_proveedor_llm(motor_llm)
        return await proveedor.generar_respuesta(prompt, num_tokens)


def _split_text(texto: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    chunks: list[str] = []

    def _split_recursive(text: str, seps: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        sep = seps[0] if seps else ""
        remaining_seps = seps[1:] if len(seps) > 1 else [""]

        if sep:
            parts = text.split(sep)
        else:
            result = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                result.append(text[i:i + chunk_size])
            return result

        current_chunk = ""
        result = []

        for part in parts:
            candidate = f"{current_chunk}{sep}{part}" if current_chunk else part
            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    result.append(current_chunk)
                if len(part) > chunk_size:
                    result.extend(_split_recursive(part, remaining_seps))
                    current_chunk = ""
                else:
                    if result and chunk_overlap > 0:
                        overlap_text = result[-1][-chunk_overlap:]
                        current_chunk = overlap_text + sep + part
                        if len(current_chunk) > chunk_size:
                            current_chunk = part
                    else:
                        current_chunk = part

        if current_chunk:
            result.append(current_chunk)

        return result

    raw_chunks = _split_recursive(texto, separators)
    return [c.strip() for c in raw_chunks if c.strip()]


_MENSAJE_CANONICO_VACIO = (
    "No encontré información sobre eso en los documentos académicos disponibles."
)
_FRASE_CANONICA_BASE = "no encontré información sobre"


def _normalizar_respuesta_vacia(respuesta: str) -> str:
    resp_strip = respuesta.strip()
    if len(resp_strip) >= 200:
        return respuesta
    r_lower = resp_strip.lower()
    if _FRASE_CANONICA_BASE in r_lower:
        return respuesta
    if any(frase in r_lower for frase in _FRASES_VACIO):
        print("[RAG] respuesta normalizada a frase canónica")
        return _MENSAJE_CANONICO_VACIO
    return respuesta


def _detectar_idioma_incorrecto(respuesta: str) -> bool:
    r_lower = respuesta.lower()
    return any(marcador in r_lower for marcador in _MARCADORES_NO_ESPANOL)


PALABRAS_LISTA_LARGA = _DEFAULTS_NLU["palabras_lista_larga"]


def _es_pregunta_de_lista_larga(pregunta: str) -> bool:
    from app.services.nlu_config_service import get_nlu_config_cached
    cfg = get_nlu_config_cached()
    palabras = cfg.get("palabras_lista_larga", PALABRAS_LISTA_LARGA)
    return any(k in pregunta.lower() for k in palabras)


def _parsear_motor(motor: str) -> tuple[str, str]:
    if ":" in motor:
        partes = motor.split(":", 1)
        return partes[0], partes[1]
    return motor, motor


def _inferir_categoria(nombre_archivo: str) -> dict:
    nombre = nombre_archivo.lower()
    if any(k in nombre for k in ["malla", "curricul", "pensum", "plan_estudio"]):
        return {"categoria": "malla_curricular", "tipo": "academico"}
    if any(k in nombre for k in ["graduac", "titulac", "requisito", "egreso"]):
        return {"categoria": "requisitos_graduacion", "tipo": "academico"}
    if any(k in nombre for k in ["reglamento", "normativa", "resolucion"]):
        return {"categoria": "reglamento", "tipo": "normativo"}
    if any(k in nombre for k in ["horario", "calendario", "fechas"]):
        return {"categoria": "horarios", "tipo": "administrativo"}
    return {"categoria": "general", "tipo": "desconocido"}


def _get_num_tokens(motor_llm: str) -> int:
    return NUM_TOKENS_CLOUD if motor_llm == "cloud" else NUM_TOKENS_LOCAL


def _get_retrieval_params(motor_vectores: str) -> tuple[int, float]:
    params = get_params()
    return (
        params.get("rag_k_local", 10),
        params.get("umbral_relevancia_local", 0.05),
    )

def procesar_y_guardar_documento(filepath: str, motor: str = "local") -> dict:
    motor_vectores, _ = _parsear_motor(motor)

    proveedor = crear_proveedor_llm(motor_vectores)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    extension        = os.path.splitext(filepath)[1].lower()
    nombre_coleccion = os.path.splitext(os.path.basename(filepath))[0]
    meta_extra       = _inferir_categoria(os.path.basename(filepath))

    if extension == ".pdf":
        contenido = pymupdf4llm.to_markdown(filepath)
    elif extension == ".docx":
        contenido = docx2txt.process(filepath)
    elif extension in [".txt", ".md"]:
        with open(filepath, "r", encoding="utf-8") as f:
            contenido = f.read()
    else:
        raise ValueError(f"Formato no soportado: {extension}")

    chunk_size = CHUNK_SIZE_LOCAL
    fragmentos = _split_text(contenido, chunk_size, CHUNK_OVERLAP)

    print(f"[RAG] {len(fragmentos)} fragmentos generados ({chunk_size} chars, overlap {CHUNK_OVERLAP})")

    embedder   = proveedor
    embeddings = [embedder.generar_embedding(fragmento) for fragmento in fragmentos]
    payloads   = [
        {
            "document_name": nombre_coleccion,
            "contenido":     fragmento,
            "metadata":      json.dumps({"source": filepath, **meta_extra}),
        }
        for fragmento in fragmentos
    ]

    insertar_puntos(motor_vectores, embeddings, payloads)
    print(f"[RAG] {len(fragmentos)} chunks guardados en Qdrant ({motor_vectores})")

    return {
        "mensaje":   f"Procesado en {motor_vectores}: {len(fragmentos)} fragmentos.",
        "coleccion": nombre_coleccion,
    }


def eliminar_coleccion(nombre_coleccion: str, motor: str = "local") -> dict:
    motor_vectores, _ = _parsear_motor(motor)
    return eliminar_puntos_por_documento(nombre_coleccion, motor_vectores)


def eliminar_todos_los_vectores(motor: str = "local") -> dict:
    motor_vectores, _ = _parsear_motor(motor)
    return eliminar_todos_los_puntos(motor_vectores)


def _buscar_chunks_similares(
    query_embedding: list[float],
    motor_vectores: str,
    k: int,
    umbral: float,
) -> list[dict]:
    return buscar_similares(query_embedding, motor_vectores, k, umbral)

class PipelineRAGBuilder:
    def __init__(self):
        self._pregunta      = ""
        self._chunks        = []
        self._prompt_template = ""
        self._motor_llm     = "local"
        self._num_tokens    = 512

    def set_pregunta(self, pregunta: str) -> "PipelineRAGBuilder":
        self._pregunta = pregunta
        return self

    def set_chunks(self, chunks: list[dict]) -> "PipelineRAGBuilder":
        self._chunks = chunks
        return self

    def set_prompt_template(self, template: str) -> "PipelineRAGBuilder":
        self._prompt_template = template
        return self

    def set_motor_llm(self, motor_llm: str) -> "PipelineRAGBuilder":
        self._motor_llm = motor_llm
        return self

    def set_num_tokens(self, num_tokens: int) -> "PipelineRAGBuilder":
        self._num_tokens = num_tokens
        return self

    def build(self) -> dict:
        chunks_seleccionados = []
        chars_acumulados     = 0
        for chunk in self._chunks:
            contenido = chunk["contenido"]
            if chars_acumulados + len(contenido) > MAX_CHARS_CONTEXTO:
                break
            chunks_seleccionados.append(chunk)
            chars_acumulados += len(contenido)

        if not chunks_seleccionados and self._chunks:
            primer_chunk = dict(self._chunks[0])
            primer_chunk["contenido"] = self._chunks[0]["contenido"][:MAX_CHARS_CONTEXTO]
            chunks_seleccionados = [primer_chunk]

        print(f"[RAG] chunks usados: {len(chunks_seleccionados)}/{len(self._chunks)} ({chars_acumulados} chars)")

        contexto  = "\n\n---\n\n".join([c["contenido"] for c in chunks_seleccionados])
        prompt_ia = self._prompt_template.format(contexto=contexto, pregunta=self._pregunta)

        LIMITE_TOKENS_ENTRADA = 1200

        if len(prompt_ia) / 3.5 > LIMITE_TOKENS_ENTRADA:
            overhead = len(prompt_ia) - len(contexto)
            max_contexto = max(0, int(LIMITE_TOKENS_ENTRADA * 3.5) - overhead)
            contexto = contexto[:max_contexto]
            prompt_ia = self._prompt_template.format(contexto=contexto, pregunta=self._pregunta)

        return {
            "prompt":     prompt_ia,
            "contexto":   contexto,
            "motor_llm":  self._motor_llm,
            "num_tokens": self._num_tokens,
            "coleccion":  chunks_seleccionados[0]["coleccion"] if chunks_seleccionados else "desconocido",
        }

_PATRONES_AFIRMACION = [
    r"(.+?),?\s*¿(?:es correcto|verdad|cierto|no es así|no)\?",
    r"tengo entendido que (.+?),?\s*¿",
    r"escuché que (.+?),?\s*¿",
    r"(.+?)\s*¿(?:es así|correcto|verdad)\?",
]


def _reescribir_query_para_retrieval(pregunta: str) -> str:
    for patron in _PATRONES_AFIRMACION:
        m = re.search(patron, pregunta, re.IGNORECASE)
        if m:
            afirmacion = m.group(1).strip()
            print(f"[RAG] query reescrita: '{afirmacion}'")
            return afirmacion
    return pregunta


def _get_system_prompt_from_db() -> str:
    params = get_params()
    return params.get("prompt_principal") or SYSTEM_PROMPT_EDITABLE


def _get_prompt_template_from_db() -> str:
    params = get_params()
    prompt_editable = params.get("prompt_principal") or SYSTEM_PROMPT_EDITABLE
    return construir_prompt_completo(prompt_editable)


async def _consultar_rag_puro(
    pregunta: str,
    motor: str = "local",
    embedding_precalculado: list[float] | None = None,
) -> tuple[str, str]:
    motor_vectores, motor_llm = _parsear_motor(motor)

    query_retrieval = _reescribir_query_para_retrieval(pregunta)
    if embedding_precalculado is not None:
        query_embedding = embedding_precalculado
    else:
        query_embedding = await _generar_embedding_async(query_retrieval, motor_vectores)

    k_retrieval, umbral = _get_retrieval_params(motor_vectores)
    resultados = _buscar_chunks_similares(query_embedding, motor_vectores, k_retrieval, umbral)

    if not resultados:
        from app.services.nlu_config_service import get_nlu_config_cached
        return (
            get_nlu_config_cached().get(
                "mensaje_sin_resultados",
                "No encontré información sobre eso en los documentos académicos disponibles.",
            ),
            "desconocido",
        )

    num_tokens = _get_num_tokens(motor_llm)
    plantilla  = _get_prompt_template_from_db()

    pipeline = (
        PipelineRAGBuilder()
        .set_pregunta(pregunta)
        .set_chunks(resultados)
        .set_prompt_template(plantilla)
        .set_motor_llm(motor_llm)
        .set_num_tokens(num_tokens)
        .build()
    )

    try:
        respuesta = await _invocar_llm(
            pipeline["prompt"],
            pipeline["motor_llm"],
            pipeline["num_tokens"],
        )
    except LLMError as e:
        print(f"[RAG] LLM falló: {e}")
        return PROMPT_ERROR_FALLBACK, "error"

    respuesta = _normalizar_respuesta_vacia(respuesta)
    if _detectar_idioma_incorrecto(respuesta):
        print("[RAG] idioma incorrecto, usando fallback")
        respuesta = _MENSAJE_CANONICO_VACIO

    documento_origen = resultados[0]["coleccion"]
    return respuesta, documento_origen


async def consultar_base_conocimiento(pregunta: str, motor: str = "local") -> str:
    partes = motor.split(":", 1) if ":" in motor else (motor, motor)
    motor_vectores, motor_llm = partes[0], partes[1]

    try:
        embedding = await _generar_embedding_async(pregunta, motor_vectores)
    except Exception as e:
        print(f"[RAG] embedding fallido, sin búsqueda en caché: {e}")
        embedding = None

    from app.services.cache_service import buscar_en_cache, guardar_en_cache

    cached = buscar_en_cache(
        pregunta,
        motor_vectores=motor_vectores,
        motor_llm=motor_llm,
        embedding=embedding,
    )
    if cached:
        print(f"[CACHE] hit ({motor_vectores}:{motor_llm})")
        return cached

    t0 = time.time()
    respuesta, documento_origen = await _consultar_rag_puro(
        pregunta,
        motor,
        embedding_precalculado=embedding,
    )
    print(f"[RAG] Consulta completada en {round(time.time() - t0, 2)}s")

    guardar_en_cache(
        pregunta,
        respuesta,
        documento_origen=documento_origen,
        motor_vectores=motor_vectores,
        motor_llm=motor_llm,
        embedding=embedding,
    )
    return respuesta

async def pipeline_streaming(pregunta: str, motor: str = "local"):
    motor_vectores, motor_llm = _parsear_motor(motor)

    query_embedding = await _generar_embedding_async(pregunta, motor_vectores)
    k_retrieval, umbral = _get_retrieval_params(motor_vectores)
    resultados = _buscar_chunks_similares(query_embedding, motor_vectores, k_retrieval, umbral)

    if not resultados:
        from app.services.nlu_config_service import get_nlu_config_cached
        yield get_nlu_config_cached().get(
            "mensaje_sin_resultados",
            "No encontré información sobre eso en los documentos académicos disponibles.",
        )
        return

    plantilla = _get_prompt_template_from_db()
    system_prompt = _get_system_prompt_from_db()

    pipeline = (
        PipelineRAGBuilder()
        .set_pregunta(pregunta)
        .set_chunks(resultados)
        .set_prompt_template(plantilla)
        .set_motor_llm(motor_llm)
        .set_num_tokens(_get_num_tokens(motor_llm))
        .build()
    )

    user_content = USER_TEMPLATE.format(contexto=pipeline["contexto"], pregunta=pregunta)
    async for token in generar_respuesta_stream_local(
        system_prompt, user_content, pipeline["num_tokens"]
    ):
        yield token


async def generar_respuesta_stream_local(
    system_prompt: str,
    user_content: str,
    num_tokens: int = 512,
) -> AsyncGenerator[str, None]:
    import json as _json

    url = f"{VLLM_BASE_URL}/chat/completions"
    payload = {
        "model": LLM_MODEL_LOCAL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": num_tokens,
        "temperature": 0,
        "top_p": 0.9,
        "stream": True,
        "stop": ["Consulta del usuario:", "Usuario:", "Pregunta:", "[FIN]"],
    }

    client = HttpxClientSingleton().client
    async with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk   = _json.loads(data_str)
                delta   = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
            except (KeyError, _json.JSONDecodeError):
                continue