# app/services/rag_service.py
"""
Servicio RAG refactorizado — vLLM + sentence-transformers + Qdrant.
Usa llamadas directas a vLLM (local) y Google Generative AI (cloud),
con Qdrant como vector store.
"""

import os
import json
import time
import pymupdf4llm
import docx2txt
import google.generativeai as genai

from app.db.database import SessionLocal
from app.db import models
from app.services.cache_service import buscar_en_cache, guardar_en_cache
from app.core.config import (
    LLM_MODEL_LOCAL, LLM_MODEL_CLOUD,
    EMBED_MODEL_LOCAL, EMBED_MODEL_CLOUD,
    GOOGLE_API_KEY, EMBED_DIMENSION_LOCAL, EMBED_DIMENSION_CLOUD,
    VLLM_BASE_URL,
)
from app.services.rag_params_service import get_params
from app.core.prompts import PROMPT_PRINCIPAL_DEFAULT as _PROMPT_PRINCIPAL_DEFAULT

# Nuevos servicios
from app.services.vllm_service import (
    generar_embedding_local,
    generar_respuesta_local,
    generar_respuesta_cloud,
)
from app.services.qdrant_service import (
    insertar_puntos,
    eliminar_puntos_por_documento,
    eliminar_todos_los_puntos,
    buscar_similares,
    crear_coleccion,
)

# ─── Configurar Gemini ──────────────────────────────────────────────────────
genai.configure(api_key=GOOGLE_API_KEY)

# ─── PARÁMETROS FIJOS ───────────────────────────────────────────────────────
CHUNK_SIZE_LOCAL   = 400
CHUNK_SIZE_CLOUD   = 1200
CHUNK_OVERLAP      = 50
UMBRAL_SIMILITUD   = 0.02
NUM_TOKENS_LOCAL   = 512
NUM_TOKENS_CLOUD   = 1024


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDINGS — llamadas directas
# ─────────────────────────────────────────────────────────────────────────────

def _generar_embedding_local(texto: str) -> list[float]:
    """Genera embedding usando sentence-transformers (CPU, sin Ollama)."""
    return generar_embedding_local(texto)


def _generar_embedding_cloud(texto: str) -> list[float]:
    """Genera embedding usando Google Generative AI directamente."""
    result = genai.embed_content(
        model=EMBED_MODEL_CLOUD,
        content=texto,
    )
    return result["embedding"]


def _generar_embedding(texto: str, motor: str) -> list[float]:
    """Dispatcher de embeddings según motor."""
    if motor == "cloud":
        return _generar_embedding_cloud(texto)
    return _generar_embedding_local(texto)


# ─────────────────────────────────────────────────────────────────────────────
# LLM — llamadas directas
# ─────────────────────────────────────────────────────────────────────────────

def _invocar_llm_local(prompt: str, num_tokens: int) -> str:
    """Invoca vLLM vía httpx (síncrono, timeout 120s)."""
    return generar_respuesta_local(prompt, num_tokens)


def _invocar_llm_cloud(prompt: str, num_tokens: int) -> str:
    """Invoca Gemini directamente."""
    return generar_respuesta_cloud(prompt, num_tokens)


def _invocar_llm(prompt: str, motor_llm: str, num_tokens: int) -> str:
    """Dispatcher de LLM según motor."""
    if motor_llm == "cloud":
        return _invocar_llm_cloud(prompt, num_tokens)
    return _invocar_llm_local(prompt, num_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT SPLITTING — nativo, sin LangChain
# ─────────────────────────────────────────────────────────────────────────────

def _split_text(texto: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Divide texto en fragmentos con overlap, usando separadores naturales."""
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
            # Último recurso: cortar por caracteres
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
                    # Overlap: tomar final del chunk anterior
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


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

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
    return any(k in pregunta.lower() for k in PALABRAS_LISTA_LARGA)


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
    if motor_vectores == "cloud":
        return (
            params.get("rag_k_cloud", 8),
            params.get("umbral_relevancia_cloud", 0.30),
        )
    return (
        params.get("rag_k_local", 10),
        params.get("umbral_relevancia_local", 0.15),
    )


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR STORE — Qdrant
# ─────────────────────────────────────────────────────────────────────────────

def procesar_y_guardar_documento(filepath: str, motor: str = "local") -> dict:
    """Procesa documento, genera embeddings y los almacena en Qdrant."""
    motor_vectores, _ = _parsear_motor(motor)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    extension        = os.path.splitext(filepath)[1].lower()
    nombre_coleccion = os.path.splitext(os.path.basename(filepath))[0]
    meta_extra       = _inferir_categoria(os.path.basename(filepath))

    # ── Cargar documento ────────────────────────────────────────────────────
    if extension == ".pdf":
        contenido = pymupdf4llm.to_markdown(filepath)
    elif extension == ".docx":
        contenido = docx2txt.process(filepath)
    elif extension in [".txt", ".md"]:
        with open(filepath, "r", encoding="utf-8") as f:
            contenido = f.read()
    else:
        raise ValueError(f"Formato no soportado: {extension}")

    # ── Text splitting ──────────────────────────────────────────────────────
    chunk_size = CHUNK_SIZE_CLOUD if motor_vectores == "cloud" else CHUNK_SIZE_LOCAL
    fragmentos = _split_text(contenido, chunk_size, CHUNK_OVERLAP)

    print(f"[RAG] 📄 Text splitting ({chunk_size} chars + {CHUNK_OVERLAP} overlap): {len(fragmentos)} fragmentos generados.")

    # ── Generar embeddings y guardar en Qdrant ──────────────────────────────
    embeddings = [_generar_embedding(fragmento, motor_vectores) for fragmento in fragmentos]
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
    """Elimina todos los chunks de un documento en Qdrant."""
    motor_vectores, _ = _parsear_motor(motor)
    return eliminar_puntos_por_documento(nombre_coleccion, motor_vectores)


def eliminar_todos_los_vectores(motor: str = "local") -> dict:
    """Elimina todos los chunks vectoriales de un motor en Qdrant."""
    motor_vectores, _ = _parsear_motor(motor)
    return eliminar_todos_los_puntos(motor_vectores)


def _buscar_chunks_similares(
    query_embedding: list[float],
    motor_vectores: str,
    k: int,
    umbral: float,
) -> list[dict]:
    """Búsqueda de similitud coseno en Qdrant."""
    return buscar_similares(query_embedding, motor_vectores, k, umbral)


# ─────────────────────────────────────────────────────────────────────────────
# CONSULTA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def consultar_base_conocimiento(pregunta: str, motor: str = "local") -> str:
    """
    Pipeline RAG completo:
      1. Caché semántico en pgvector.
      2. Embedding de la pregunta.
      3. Búsqueda de similitud en pgvector.
      4. Generación LLM con prompt dinámico.
      5. Guardado en caché.
    """
    t_inicio = time.time()
    motor_vectores, motor_llm = _parsear_motor(motor)

    # ── 1. Caché semántico ────────────────────────────────────────────────────
    cached = buscar_en_cache(pregunta, motor_vectores=motor_vectores, motor_llm=motor_llm)
    if cached:
        print(f"[RAG] ⚡ Cache hit ({motor_vectores}:{motor_llm})")
        return cached

    # ── 2. Embedding de la pregunta ───────────────────────────────────────────
    query_embedding = _generar_embedding(pregunta, motor_vectores)

    # ── 3. Búsqueda en pgvector ───────────────────────────────────────────────
    k_retrieval, umbral = _get_retrieval_params(motor_vectores)
    resultados = _buscar_chunks_similares(query_embedding, motor_vectores, k_retrieval, umbral)

    if not resultados:
        return "Lo siento, esa información no existe en mi base de datos oficial."

    contexto = "\n\n---\n\n".join([r["contenido"] for r in resultados])
    coleccion_principal = resultados[0]["coleccion"] if resultados else "desconocido"

    # ── 4. Generación LLM ────────────────────────────────────────────────────
    num_tokens = _get_num_tokens(motor_llm)
    params     = get_params()
    plantilla  = params.get("prompt_principal") or _PROMPT_PRINCIPAL_DEFAULT
    prompt_ia  = plantilla.format(contexto=contexto, pregunta=pregunta)

    respuesta = _invocar_llm(prompt_ia, motor_llm, num_tokens)

    # ── 5. Guardado en caché ──────────────────────────────────────────────────
    guardar_en_cache(
        pregunta,
        respuesta,
        documento_origen=coleccion_principal,
        motor_vectores=motor_vectores,
        motor_llm=motor_llm,
    )

    t_total = round(time.time() - t_inicio, 2)
    print(f"[RAG] ✅ Respuesta generada ({motor_vectores}:{motor_llm}) en {t_total}s")
    return respuesta