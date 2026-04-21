# app/services/rag_service.py
import os
import re
import json
import time
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
from app.services.rag_params_service import get_params
from app.core.prompts import PROMPT_PRINCIPAL_DEFAULT as _PROMPT_PRINCIPAL_DEFAULT
from app.core.prompts import PROMPT_ERROR_FALLBACK
from app.core.exceptions import LLMError

from app.services.qdrant_service import (
    insertar_puntos,
    eliminar_puntos_por_documento,
    eliminar_todos_los_puntos,
    buscar_similares,
    crear_coleccion,
)

from app.services.providers import crear_proveedor_llm

genai.configure(api_key=GOOGLE_API_KEY)

CHUNK_SIZE_LOCAL   = 400
CHUNK_OVERLAP      = 50
UMBRAL_BUSQUEDA_QDRANT = 0.02
NUM_TOKENS_LOCAL   = 512
NUM_TOKENS_CLOUD   = 1024
MAX_MODEL_LEN      = 2048  
MAX_CHARS_CONTEXTO = 3500   

def _generar_embedding(texto: str, motor: str) -> list[float]:
    proveedor = crear_proveedor_llm(motor)
    return proveedor.generar_embedding(texto)

def _invocar_llm(prompt: str, motor_llm: str, num_tokens: int) -> str:
    proveedor = crear_proveedor_llm(motor_llm)
    return proveedor.generar_respuesta(prompt, num_tokens)

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


_FRASES_VACIO = [
    "lo siento",
    "no existe en mi base",
    "no tengo esa información",
    "no poseo información",
    "no cuento con esa",
    "esa información no existe",
    "no hay información",
    "no se encontró información",
    "no dispongo de",
]
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
    # Solo normalizar variantes NO canónicas
    if any(frase in r_lower for frase in _FRASES_VACIO):
        print("[RAG] 🔄 Respuesta normalizada a frase canónica de vacío.")
        return _MENSAJE_CANONICO_VACIO
    return respuesta


_MARCADORES_NO_ESPANOL = [
    "você", "voce", "matérias", "seguintes", "tem como",
    "são ", "está ", "precisa ", "todas as ", "pode ",
]


def _detectar_idioma_incorrecto(respuesta: str) -> bool:
    r_lower = respuesta.lower()
    return any(marcador in r_lower for marcador in _MARCADORES_NO_ESPANOL)


PALABRAS_LISTA_LARGA = [
    "todas las materias", "todos los niveles", "lista completa",
    "enumera todas", "todos los semestres",
    "qué materias hay en", "materias del nivel", "cuáles son todas",
    "prerrequisitos transitivos", "debería haber aprobado antes",
    "sin ningún prerrequisito", "no tienen prerrequisito",
    "qué necesito para graduarme", "requisitos para graduarme",
    "qué requisitos", "cuáles son los requisitos",
]


def _es_pregunta_de_lista_larga(pregunta: str) -> bool:
    from app.services.nlu_config_service import get_nlu_config
    cfg = get_nlu_config()
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

    print(f"[RAG] 📄 Text splitting ({chunk_size} chars + {CHUNK_OVERLAP} overlap): {len(fragmentos)} fragmentos generados.")

    embedder = proveedor
    embeddings = [embedder.generar_embedding(fragmento) for fragmento in fragmentos]
    payloads = [
        {
            "document_name": nombre_coleccion,
            "contenido": fragmento,
            "metadata": json.dumps({"source": filepath, **meta_extra}),
        }
        for fragmento in fragmentos
    ]

    insertar_puntos(motor_vectores, embeddings, payloads)
    print(f"[RAG] ✅ {len(fragmentos)} chunks guardados en Qdrant ({motor_vectores})")

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
        self._pregunta = ""
        self._chunks = []
        self._prompt_template = ""
        self._motor_llm = "local"
        self._num_tokens = 512

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
        chars_acumulados = 0
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

        print(f"[RAG] 📦 Chunks usados: {len(chunks_seleccionados)}/{len(self._chunks)} "
              f"({chars_acumulados} chars de contexto)")

        contexto  = "\n\n---\n\n".join([c["contenido"] for c in chunks_seleccionados])
        prompt_ia = self._prompt_template.format(contexto=contexto, pregunta=self._pregunta)
        return {
            "prompt":    prompt_ia,
            "motor_llm": self._motor_llm,
            "num_tokens": self._num_tokens,
            "coleccion": chunks_seleccionados[0]["coleccion"] if chunks_seleccionados else "desconocido",
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
            print(f"[RAG] 🔍 Query reescrita para retrieval: '{afirmacion}'")
            return afirmacion
    return pregunta


def _consultar_rag_puro(pregunta: str, motor: str = "local") -> tuple[str, str]:

    motor_vectores, motor_llm = _parsear_motor(motor)

    query_retrieval = _reescribir_query_para_retrieval(pregunta)
    query_embedding = _generar_embedding(query_retrieval, motor_vectores)

    k_retrieval, umbral = _get_retrieval_params(motor_vectores)
    resultados = _buscar_chunks_similares(query_embedding, motor_vectores, k_retrieval, umbral)

    if not resultados:
        from app.services.nlu_config_service import get_nlu_config
        return (
            get_nlu_config().get("mensaje_sin_resultados", "No encontré información sobre eso en los documentos académicos disponibles."),
            "desconocido",
        )

    num_tokens = _get_num_tokens(motor_llm)
    params     = get_params()
    plantilla  = params.get("prompt_principal") or _PROMPT_PRINCIPAL_DEFAULT

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
        respuesta = _invocar_llm(pipeline["prompt"], pipeline["motor_llm"], pipeline["num_tokens"])
    except LLMError as e:
        print(f"[RAG] ❌ LLM falló: {e}")
        return PROMPT_ERROR_FALLBACK, "error"

    respuesta = _normalizar_respuesta_vacia(respuesta)
    if _detectar_idioma_incorrecto(respuesta):
        print("[RAG] ⚠️ Respuesta en idioma incorrecto detectada, usando fallback.")
        respuesta = _MENSAJE_CANONICO_VACIO
    documento_origen = resultados[0]["coleccion"]
    return respuesta, documento_origen

def consultar_base_conocimiento(pregunta: str, motor: str = "local") -> str:
    partes = motor.split(":", 1) if ":" in motor else (motor, motor)
    motor_vectores, motor_llm = partes[0], partes[1]

    from app.services.cache_service import buscar_en_cache, guardar_en_cache
    cached = buscar_en_cache(pregunta, motor_vectores=motor_vectores, motor_llm=motor_llm)
    if cached:
        print(f"[CACHE] ⚡ Hit ({motor_vectores}:{motor_llm})")
        return cached

    t0 = time.time()
    respuesta, documento_origen = _consultar_rag_puro(pregunta, motor)
    print(f"[RAG] ✅ Consulta completada en {round(time.time() - t0, 2)}s")

    guardar_en_cache(pregunta, respuesta, documento_origen=documento_origen, motor_vectores=motor_vectores, motor_llm=motor_llm)
    return respuesta

async def pipeline_streaming(pregunta: str, motor: str = "local"):
    motor_vectores, motor_llm = _parsear_motor(motor)

    query_embedding = _generar_embedding(pregunta, motor_vectores)
    k_retrieval, umbral = _get_retrieval_params(motor_vectores)
    resultados = _buscar_chunks_similares(query_embedding, motor_vectores, k_retrieval, umbral)

    if not resultados:
        from app.services.nlu_config_service import get_nlu_config
        yield get_nlu_config().get("mensaje_sin_resultados", "No encontré información sobre eso en los documentos académicos disponibles.")
        return

    params = get_params()
    plantilla = params.get("prompt_principal") or _PROMPT_PRINCIPAL_DEFAULT

    pipeline = (
        PipelineRAGBuilder()
        .set_pregunta(pregunta)
        .set_chunks(resultados)
        .set_prompt_template(plantilla)
        .set_motor_llm(motor_llm)
        .set_num_tokens(_get_num_tokens(motor_llm))
        .build()
    )

    async for token in generar_respuesta_stream_local(pipeline["prompt"], pipeline["num_tokens"]):
        yield token

async def generar_respuesta_stream_local(prompt: str, num_tokens: int = 512) -> AsyncGenerator[str, None]:
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