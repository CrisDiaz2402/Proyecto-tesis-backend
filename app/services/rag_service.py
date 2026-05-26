import os
import re
import time
import hashlib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymupdf4llm
import chromadb
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from app.services.cache_service import buscar_en_cache, guardar_en_cache
from app.core.config import (
    VECTOR_STORE_DIR,
    LLM_MODEL, EMBED_MODEL,
    UMBRAL_RELEVANCIA, RAG_K, BREAKPOINT_THRESHOLD,
    NUM_TOKENS_NORMAL, NUM_TOKENS_LISTA, HYDE_NUM_PREDICT,
    CACHE_L1_MAX_ENTRIES,
    LLM_TEMPERATURE, LLM_TOP_K, LLM_TOP_P, LLM_REPEAT_PENALTY, LLM_STOP_WORDS,
    PROMPT_PRINCIPAL, PROMPT_HYDE,
)


_CACHE_L1: dict[str, str] = {}


def limpiar_cache_l1() -> None:
    _CACHE_L1.clear()
    _invalidar_singletons_llm()


def _llave_l1(pregunta: str) -> str:
    return hashlib.md5(pregunta.lower().strip().encode()).hexdigest()


def _guardar_l1(llave: str, respuesta: str) -> None:
    if len(_CACHE_L1) >= CACHE_L1_MAX_ENTRIES:
        del _CACHE_L1[next(iter(_CACHE_L1))]
    _CACHE_L1[llave] = respuesta


def guardar_en_cache_l1(llave: str, respuesta: str) -> None:
    _guardar_l1(llave, respuesta)


_embeddings_cpu: HuggingFaceEmbeddings | None = None
_ollama_llm_cache: dict[int, OllamaLLM] = {}
_chroma_client: chromadb.ClientAPI | None = None


def _invalidar_singletons_llm() -> None:
    global _ollama_llm_cache
    _ollama_llm_cache.clear()


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_cpu
    if _embeddings_cpu is None:
        _embeddings_cpu = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_cpu


def _get_llm(num_tokens: int) -> OllamaLLM:
    global _ollama_llm_cache
    if num_tokens not in _ollama_llm_cache:
        _ollama_llm_cache[num_tokens] = OllamaLLM(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            num_predict=num_tokens,
            top_k=LLM_TOP_K,
            top_p=LLM_TOP_P,
            repeat_penalty=LLM_REPEAT_PENALTY,
            stop=LLM_STOP_WORDS,
            keep_alive="60m",
            num_gpu=999,
        )
    return _ollama_llm_cache[num_tokens]


def _get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    return _chroma_client


_PALABRAS_LISTA = [
    "todas las materias", "todos los niveles", "lista completa",
    "enumera todas", "todos los semestres",
    "qué materias hay en", "materias del nivel", "cuáles son todas",
    "prerrequisitos transitivos", "debería haber aprobado antes",
    "sin ningún prerrequisito", "no tienen prerrequisito",
    "qué necesito para graduarme", "requisitos para graduarme",
    "qué requisitos", "cuáles son los requisitos",
]

_SALUDOS = {"hola", "buenas", "buenos días", "buenas tardes", "buenas noches", "hi", "hello", "hey"}
_DESPEDIDAS = {"adiós", "adios", "hasta luego", "chao", "chau", "bye", "nos vemos", "gracias", "muchas gracias"}


def _es_lista_larga(pregunta: str) -> bool:
    return any(k in pregunta.lower() for k in _PALABRAS_LISTA)


def _es_saludo(pregunta: str) -> bool:
    p = pregunta.lower().strip().rstrip("!?.,")
    return any(s in p for s in _SALUDOS)


def _es_despedida(pregunta: str) -> bool:
    p = pregunta.lower().strip().rstrip("!?.,")
    return any(d in p for d in _DESPEDIDAS)


def _num_tokens_para(pregunta: str) -> int:
    return NUM_TOKENS_LISTA if _es_lista_larga(pregunta) else NUM_TOKENS_NORMAL


def _inferir_categoria(nombre: str) -> dict:
    n = nombre.lower()
    if any(k in n for k in ["malla", "curricul", "pensum", "plan_estudio"]):
        return {"categoria": "malla_curricular", "tipo": "academico"}
    if any(k in n for k in ["graduac", "titulac", "requisito", "egreso"]):
        return {"categoria": "requisitos_graduacion", "tipo": "academico"}
    if any(k in n for k in ["reglamento", "normativa", "resolucion"]):
        return {"categoria": "reglamento", "tipo": "normativo"}
    if any(k in n for k in ["horario", "calendario", "fechas"]):
        return {"categoria": "horarios", "tipo": "administrativo"}
    return {"categoria": "general", "tipo": "desconocido"}


def _listar_colecciones() -> list[str]:
    try:
        return [
            c.name for c in _get_chroma_client().list_collections()
            if c.name != "cache_respuestas"
        ]
    except Exception:
        return []


def _buscar_en_coleccion(args: tuple) -> tuple[str, list]:
    nombre_col, embeddings, query, k, umbral = args
    try:
        db = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings,
            collection_name=nombre_col,
        )
        resultados = db.similarity_search_with_relevance_scores(query, k=k)
        filtrados  = [(doc, s) for doc, s in resultados if s >= umbral]
        for doc, s in filtrados:
            doc.metadata.update({"_coleccion": nombre_col, "_score": round(s, 4)})
        return nombre_col, filtrados
    except Exception:
        return nombre_col, []


def procesar_y_guardar_documento(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    ext              = os.path.splitext(filepath)[1].lower()
    nombre_coleccion = os.path.splitext(os.path.basename(filepath))[0]
    meta_extra       = _inferir_categoria(os.path.basename(filepath))

    if ext == ".pdf":
        documentos = [Document(
            page_content=pymupdf4llm.to_markdown(filepath),
            metadata={"source": filepath, **meta_extra},
        )]
    elif ext == ".docx":
        documentos = Docx2txtLoader(filepath).load()
        for d in documentos:
            d.metadata.update(meta_extra)
    elif ext in {".txt", ".md"}:
        documentos = TextLoader(filepath, encoding="utf-8").load()
        for d in documentos:
            d.metadata.update(meta_extra)
    else:
        raise ValueError(f"Formato no soportado: {ext}")

    embeddings = _get_embeddings()
    splitter   = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=BREAKPOINT_THRESHOLD,
    )
    fragmentos = splitter.split_documents(documentos)

    Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name=nombre_coleccion,
    )
    return {
        "mensaje":   f"Procesado: {len(fragmentos)} fragmentos.",
        "coleccion": nombre_coleccion,
    }


def eliminar_coleccion_chroma(nombre_coleccion: str) -> dict:
    try:
        db = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=_get_embeddings(),
            collection_name=nombre_coleccion,
        )
        db.delete_collection()
        return {"mensaje": f"Colección '{nombre_coleccion}' eliminada."}
    except Exception as e:
        return {"mensaje": str(e)}


def eliminar_todos_los_vectores_chroma() -> dict:
    try:
        cliente = _get_chroma_client()
        for c in cliente.list_collections():
            if c.name != "cache_respuestas":
                cliente.delete_collection(c.name)
        return {"mensaje": "Todos los vectores han sido eliminados."}
    except Exception as e:
        return {"mensaje": str(e)}


def preparar_prompt_rag(pregunta: str) -> dict:

    llave_l1 = _llave_l1(pregunta)
    if llave_l1 in _CACHE_L1:
        return {"cache_hit": True, "respuesta": _CACHE_L1[llave_l1]}

    cached = buscar_en_cache(pregunta)
    if cached:
        _guardar_l1(llave_l1, cached)
        return {"cache_hit": True, "respuesta": cached}

    if _es_saludo(pregunta):
        return {
            "cache_hit": True,
            "respuesta": "¡Hola! Soy el Asistente Académico de la EPN. Puedo ayudarte con información sobre la malla curricular de Ciencias de la Computación. ¿Qué deseas saber?",
        }

    if _es_despedida(pregunta):
        return {
            "cache_hit": True,
            "respuesta": "¡Hasta luego! Si tienes más preguntas sobre la carrera, aquí estaré.",
        }

    if not VECTOR_STORE_DIR.exists():
        return {
            "cache_hit": True,
            "respuesta": "Error: La base de datos de vectores está vacía.",
        }

    colecciones = _listar_colecciones()
    if not colecciones:
        return {
            "cache_hit": True,
            "respuesta": "Lo siento, aún no hay documentos indexados en el sistema.",
        }

    embeddings     = _get_embeddings()
    query_busqueda = pregunta
    args_lista     = [
        (col, embeddings, query_busqueda, RAG_K, UMBRAL_RELEVANCIA)
        for col in colecciones
    ]

    resultados_con_score: list[tuple] = []
    coleccion_principal: str | None   = None

    with ThreadPoolExecutor(max_workers=min(len(colecciones), 3)) as executor:
        futuros = {executor.submit(_buscar_en_coleccion, args): args[0] for args in args_lista}
        for futuro in as_completed(futuros):
            nombre_col, filtrados = futuro.result()
            if filtrados and not coleccion_principal:
                coleccion_principal = nombre_col
            resultados_con_score.extend(filtrados)

    resultados_con_score.sort(key=lambda x: x[1], reverse=True)
    resultados = [doc for doc, _ in resultados_con_score[:RAG_K]]

    if not resultados:
        return {
            "cache_hit": True,
            "respuesta": "Lo siento, esa información no existe en mi base de datos oficial.",
        }

    contexto   = "\n\n---\n\n".join(doc.page_content for doc in resultados)
    num_tokens = _num_tokens_para(pregunta)
    prompt_ia  = PROMPT_PRINCIPAL.format(contexto=contexto, pregunta=pregunta)

    return {
        "cache_hit":        False,
        "prompt":           prompt_ia,
        "coleccion_origen": coleccion_principal or "desconocido",
        "num_tokens":       num_tokens,
        "llave_l1":         llave_l1,
    }


def consultar_base_conocimiento(pregunta: str) -> tuple[str, bool]:
    resultado = preparar_prompt_rag(pregunta)

    if resultado["cache_hit"]:
        return resultado["respuesta"], True

    llm        = _get_llm(resultado["num_tokens"])
    resp_cruda = llm.invoke(resultado["prompt"])
    respuesta  = resp_cruda.content if hasattr(resp_cruda, "content") else resp_cruda
    respuesta  = re.sub(r"<think>.*?</think>", "", respuesta, flags=re.DOTALL).strip()
    respuesta  = re.sub(r"(?m)^(Okay[,.]?|Let me|First,|Wait[,.]?).*\n?", "", respuesta).strip()

    if not respuesta:
        resp_cruda2 = llm.invoke(resultado["prompt"], think=False)
        respuesta   = resp_cruda2.content if hasattr(resp_cruda2, "content") else resp_cruda2
        respuesta   = re.sub(r"<think>.*?</think>", "", respuesta, flags=re.DOTALL).strip()
        respuesta   = re.sub(r"(?m)^(Okay[,.]?|Let me|First,|Wait[,.]?).*\n?", "", respuesta).strip()

    if not respuesta:
        respuesta = "Lo siento, no pude generar una respuesta. Intenta reformular tu pregunta."

    guardar_en_cache(pregunta, respuesta, documento_origen=resultado["coleccion_origen"])
    _guardar_l1(resultado["llave_l1"], respuesta)

    return respuesta, False


def precalentar() -> None:
    try:
        _get_embeddings()
    except Exception:
        pass

    _ollama_listo = False
    for _intento in range(3):
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.status_code == 200:
                _ollama_listo = True
                break
        except Exception:
            pass
        if _intento < 2:
            time.sleep(2)

    if not _ollama_listo:
        print("[PRECALENTAR] Advertencia: Ollama no respondio tras 3 intentos. Warmup omitido.")
        return

    try:
        preguntas_warmup = [
            "¿Cuántos créditos necesito para graduarme?",
            "¿Cuántos semestres dura la carrera?",
        ]
        for p in preguntas_warmup:
            consultar_base_conocimiento(p)
    except Exception:
        pass
