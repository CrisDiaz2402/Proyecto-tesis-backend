# app/services/rag_service.py
import os
import time
import hashlib
import pymupdf4llm
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from app.services.cache_service import buscar_en_cache, guardar_en_cache
from app.core.config import (
    VECTOR_STORE_DIR_LOCAL, VECTOR_STORE_DIR_CLOUD,
    LLM_MODEL_LOCAL, LLM_MODEL_CLOUD,
    EMBED_MODEL_LOCAL, EMBED_MODEL_CLOUD, GOOGLE_API_KEY,
)
from app.services.rag_params_service import get_params

# ─── PARÁMETROS FIJOS — no editables desde UI ─────────────────────────────────
# Chunking: RecursiveCharacterTextSplitter
CHUNK_SIZE_LOCAL   = 400   # chars — compatible con VRAM 4GB
CHUNK_SIZE_CLOUD   = 1200  # chars — compatible con gemini-embedding-001
CHUNK_OVERLAP      = 50    # chars

# Caché L2: umbral fijo
CACHE_THRESHOLD    = 0.85
UMBRAL_SIMILITUD   = 0.02

# LLM local: sampling fijo
REPEAT_PENALTY     = 1.1
TOP_K_LLM          = 40
TOP_P_LLM          = 0.9

# Tokens de respuesta fijos
NUM_TOKENS_LOCAL   = 512
NUM_TOKENS_CLOUD   = 1024

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS POR DEFECTO — hardcodeados como fallback.
# Se usa cuando prompt_principal en la BD es NULL.
# Deben contener los placeholders {contexto} y {pregunta}.
# Importado desde rag_params_service para evitar duplicación.
# ─────────────────────────────────────────────────────────────────────────────
from app.core.prompts import PROMPT_PRINCIPAL_DEFAULT as _PROMPT_PRINCIPAL_DEFAULT


# ── Singletons para Ollama (local) — se eliminarán con migración a vLLM ────────
# TODO: ELIMINAR tras migración a vLLM + sentence-transformers
_ollama_embeddings: OllamaEmbeddings | None = None
_ollama_llm_cache: dict[int, OllamaLLM] = {}
_chroma_client_local = None  # chromadb.ClientAPI


def _invalidar_singletons_llm() -> None:
    """Limpia el caché de instancias OllamaLLM. TODO: ELIMINAR tras migración a vLLM."""
    global _ollama_llm_cache
    _ollama_llm_cache.clear()
    print("[RAG] 🔄 Singletons OllamaLLM invalidados (parámetros de sampling cambiados).")


def _get_ollama_embeddings() -> OllamaEmbeddings:
    """TODO: ELIMINAR tras migración a sentence-transformers."""
    global _ollama_embeddings
    if _ollama_embeddings is None:
        _ollama_embeddings = OllamaEmbeddings(model=EMBED_MODEL_LOCAL)
        print(f"[RAG] ✅ OllamaEmbeddings inicializado (singleton, model={EMBED_MODEL_LOCAL})")
    return _ollama_embeddings


def _get_ollama_llm(num_tokens: int) -> OllamaLLM:
    """TODO: ELIMINAR tras migración a vLLM."""
    global _ollama_llm_cache
    if num_tokens not in _ollama_llm_cache:
        _ollama_llm_cache[num_tokens] = OllamaLLM(
            model=LLM_MODEL_LOCAL,
            temperature=0,          # No modificable: determinismo crítico para el caché
            num_predict=num_tokens,
            top_k=TOP_K_LLM,
            top_p=TOP_P_LLM,
            repeat_penalty=REPEAT_PENALTY,
            stop=["Consulta del usuario:", "Usuario:", "Pregunta:", "[FIN]"],
        )
        print(f"[RAG] ✅ OllamaLLM inicializado (singleton, num_tokens={num_tokens})")
    return _ollama_llm_cache[num_tokens]


def _get_chroma_client_local():
    """TODO: ELIMINAR tras migración a Qdrant."""
    global _chroma_client_local
    if _chroma_client_local is None:
        import chromadb
        _chroma_client_local = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR_LOCAL))
        print(f"[RAG] ✅ ChromaDB PersistentClient inicializado (singleton, path={VECTOR_STORE_DIR_LOCAL})")
    return _chroma_client_local


# ─────────────────────────────────────────────────────────────────────────────
# PALABRAS CLAVE PARA DETECCIÓN DE PREGUNTAS DE LISTA LARGA
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


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def _parsear_motor(motor: str) -> tuple[str, str]:
    """TODO: ELIMINAR - Con vLLM+Qdrant solo hay un motor."""
    if ":" in motor:
        partes = motor.split(":", 1)
        return partes[0], partes[1]
    return motor, motor


def _get_embeddings_and_dir(motor_vectores: str):
    """TODO: ELIMINAR - Con Qdrant solo hay un directorio y embeddings."""
    if motor_vectores == "cloud":
        return (
            str(VECTOR_STORE_DIR_CLOUD),
            GoogleGenerativeAIEmbeddings(model=EMBED_MODEL_CLOUD, google_api_key=GOOGLE_API_KEY),
        )
    return str(VECTOR_STORE_DIR_LOCAL), _get_ollama_embeddings()


def _get_llm(motor_llm: str, num_tokens: int):
    """
    Retorna (llm, model_name) según el motor LLM elegido.
    Para el motor local reutiliza el singleton cacheado por num_tokens.
    Los parámetros de sampling del LLM local se leen dinámicamente desde get_params().
    """
    if motor_llm == "cloud":
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_CLOUD,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,          # No modificable: determinismo crítico para el caché
            max_tokens=num_tokens,
        )
        return llm, LLM_MODEL_CLOUD

    return _get_ollama_llm(num_tokens), LLM_MODEL_LOCAL


def _get_num_tokens(motor_llm: str, es_lista_larga: bool) -> int:
    """Devuelve cuántos tokens debe generar el LLM según motor (valores hardcodeados optimizados)."""
    # Sin distinción lista/normal - simplificación basada en evaluación de impacto
    return NUM_TOKENS_CLOUD if motor_llm == "cloud" else NUM_TOKENS_LOCAL


def _get_retrieval_params(motor_vectores: str) -> tuple[int, float]:
    """Retorna (k_retrieval, umbral_relevancia) para el motor dado."""
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


def _obtener_todas_las_colecciones(vector_dir: str) -> list[str]:
    """Lista las colecciones disponibles en ChromaDB.
    Para el motor local reutiliza el PersistentClient singleton."""
    try:
        # Reutilizar el cliente singleton si es el motor local
        if vector_dir == str(VECTOR_STORE_DIR_LOCAL):
            cliente = _get_chroma_client_local()
        else:
            import chromadb
            cliente = chromadb.PersistentClient(path=vector_dir)
        return [c.name for c in cliente.list_collections() if c.name != "cache_respuestas"]
    except Exception as e:
        print(f"[RAG] Error al listar colecciones: {e}")
        return []


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


# ─────────────────────────────────────────────────────────────────────────────
# HyDE — Hypothetical Document Embeddings (solo para motor_vectores=local)
# ─────────────────────────────────────────────────────────────────────────────

def _aplicar_hyde(pregunta: str) -> str:
    """Aplica HyDE - ELIMINADO. Esta función ya no se usa.
    Mantenida solo para compatibilidad de imports legacy.
    """
    # HyDE eliminado - retorna pregunta original
    return pregunta


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL PARALELO
# ─────────────────────────────────────────────────────────────────────────────

def _buscar_en_coleccion(args: tuple) -> tuple[str, list]:
    """Función auxiliar para búsqueda paralela en una colección de ChromaDB.
    El objeto Chroma de LangChain es un wrapper ligero — se crea por hilo
    pero usa el singleton de embeddings para no reconectar a Ollama."""
    nombre_col, vector_dir, embeddings, query, k_retrieval, umbral = args
    try:
        db = Chroma(
            persist_directory=vector_dir,
            embedding_function=embeddings,
            collection_name=nombre_col,
        )
        resultados = db.similarity_search_with_relevance_scores(query, k=k_retrieval)
        filtrados  = [(doc, score) for doc, score in resultados if score >= umbral]
        for doc, score in filtrados:
            doc.metadata.update({"_coleccion": nombre_col, "_score": round(score, 4)})
        return nombre_col, filtrados
    except Exception as e:
        print(f"[RAG] Error en colección '{nombre_col}': {e}")
        return nombre_col, []


# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO DE DOCUMENTOS
# ─────────────────────────────────────────────────────────────────────────────

def procesar_y_guardar_documento(filepath: str, motor: str = "local") -> dict:
    """
    Procesa un documento y lo guarda en el vector store del motor indicado.
    """
    motor_vectores, _ = _parsear_motor(motor)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    extension        = os.path.splitext(filepath)[1].lower()
    nombre_coleccion = os.path.splitext(os.path.basename(filepath))[0]
    meta_extra       = _inferir_categoria(os.path.basename(filepath))

    if extension == ".pdf":
        documentos = [Document(
            page_content=pymupdf4llm.to_markdown(filepath),
            metadata={"source": filepath, **meta_extra},
        )]
    elif extension == ".docx":
        documentos = Docx2txtLoader(filepath).load()
        for doc in documentos:
            doc.metadata.update(meta_extra)
    elif extension in [".txt", ".md"]:
        documentos = TextLoader(filepath, encoding="utf-8").load()
        for doc in documentos:
            doc.metadata.update(meta_extra)
    else:
        raise ValueError(f"Formato no soportado: {extension}")

    # ── Text Splitting — tamaño diferenciado por motor ─────────────────────────────────────
    vector_dir, embeddings = _get_embeddings_and_dir(motor_vectores)

    # Chunk size diferenciado: local=400 (VRAM 4GB), cloud=1200 (Gemini optimizado)
    chunk_size = CHUNK_SIZE_CLOUD if motor_vectores == "cloud" else CHUNK_SIZE_LOCAL
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    fragmentos = splitter.split_documents(documentos)

    print(f"[RAG] 📄 Text splitting ({chunk_size} chars + {CHUNK_OVERLAP} overlap): {len(fragmentos)} fragmentos generados.")

    Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=vector_dir,
        collection_name=nombre_coleccion,
    )
    return {
        "mensaje":   f"Procesado en {motor_vectores}: {len(fragmentos)} fragmentos (400 chars + 50 overlap).",
        "coleccion": nombre_coleccion,
    }


def eliminar_coleccion_chroma(nombre_coleccion: str, motor: str = "local") -> dict:
    motor_vectores, _ = _parsear_motor(motor)
    vector_dir, embeddings = _get_embeddings_and_dir(motor_vectores)
    try:
        if not os.path.exists(vector_dir):
            return {"mensaje": f"Vector store {motor_vectores} no existe."}
        db = Chroma(persist_directory=vector_dir, embedding_function=embeddings, collection_name=nombre_coleccion)
        db.delete_collection()
        print(f"[CHROMA] 🗑️ Colección '{nombre_coleccion}' eliminada de {motor_vectores}.")
        return {"mensaje": f"Colección '{nombre_coleccion}' eliminada de {motor_vectores}."}
    except Exception as e:
        print(f"[CHROMA] Error al eliminar '{nombre_coleccion}' en {motor_vectores}: {e}")
        return {"mensaje": str(e)}


def eliminar_todos_los_vectores_chroma(motor: str = "local") -> dict:
    motor_vectores, _ = _parsear_motor(motor)
    vector_dir, _     = _get_embeddings_and_dir(motor_vectores)
    try:
        # Reutilizar el cliente singleton si es el motor local
        if motor_vectores == "local":
            cliente = _get_chroma_client_local()
        else:
            import chromadb
            cliente = chromadb.PersistentClient(path=vector_dir)
        for c in cliente.list_collections():
            if c.name != "cache_respuestas":
                cliente.delete_collection(c.name)
        print(f"[CHROMA] 🗑️ Todas las colecciones eliminadas en {motor_vectores}.")
        return {"mensaje": f"Todos los vectores ({motor_vectores}) han sido eliminados."}
    except Exception as e:
        print(f"[CHROMA] Error crítico al vaciar vectores {motor_vectores}: {e}")
        return {"mensaje": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# CONSULTA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def consultar_base_conocimiento(pregunta: str, motor: str = "local") -> str:
    """
    Pipeline RAG completo:
      1. Caché L2 semántico — milisegundos, preguntas similares.
      2. Vector Search — recupera fragmentos k más similares por colección.
      3. Retrieval paralelo — todas las colecciones en simultáneo.
      4. Generación LLM    — con parámetros y prompt dinámicos desde BD.
    """
    t_inicio = time.time()
    motor_vectores, motor_llm = _parsear_motor(motor)

    # ── 1. Caché L2 semántico ─────────────────────────────────────────────────
    cached = buscar_en_cache(pregunta, motor_vectores=motor_vectores, motor_llm=motor_llm)
    if cached:
        print(f"[RAG] ⚡ L2 cache hit ({motor_vectores}:{motor_llm})")
        return cached

    # ── 3. Recuperar vector store ─────────────────────────────────────────────
    vector_dir, embeddings = _get_embeddings_and_dir(motor_vectores)
    if not os.path.exists(vector_dir):
        return f"Error: La base de datos de vectores '{motor_vectores}' está vacía."

    colecciones = _obtener_todas_las_colecciones(vector_dir)
    if not colecciones:
        return f"Lo siento, no hay documentos en la base de conocimiento '{motor_vectores}'."

    # ── 4. Búsqueda semántica directa ───────────────────────────────────
    # Sin HyDE - búsqueda semántica directa simplificada
    query_busqueda = pregunta

    # ── 5. Retrieval paralelo ─────────────────────────────────────────────────
    k_retrieval, umbral = _get_retrieval_params(motor_vectores)

    args_lista = [
        (col, vector_dir, embeddings, query_busqueda, k_retrieval, umbral)
        for col in colecciones
    ]

    resultados_con_score: list[tuple] = []
    coleccion_principal:  str | None  = None

    with ThreadPoolExecutor(max_workers=min(len(colecciones), 5)) as executor:
        futuros = {executor.submit(_buscar_en_coleccion, args): args[0] for args in args_lista}
        for futuro in as_completed(futuros):
            nombre_col, filtrados = futuro.result()
            if filtrados and not coleccion_principal:
                coleccion_principal = nombre_col
            resultados_con_score.extend(filtrados)

    resultados_con_score.sort(key=lambda x: x[1], reverse=True)
    resultados = [doc for doc, _ in resultados_con_score[:k_retrieval]]

    if not resultados:
        return "Lo siento, esa información no existe en mi base de datos oficial."

    contexto = "\n\n---\n\n".join([doc.page_content for doc in resultados])

    # ── 6. Tokens dinámicos según motor y tipo de pregunta ────────────────────
    es_lista   = _es_pregunta_de_lista_larga(pregunta)
    num_tokens = _get_num_tokens(motor_llm, es_lista)

    # ── 7. Instanciar LLM y construir prompt dinámico ─────────────────────────
    llm, model_name = _get_llm(motor_llm, num_tokens)
    embed_name      = EMBED_MODEL_CLOUD if motor_vectores == "cloud" else EMBED_MODEL_LOCAL

    # Leer prompt desde BD; si es NULL usar el hardcodeado por defecto
    params    = get_params()
    plantilla = params.get("prompt_principal") or _PROMPT_PRINCIPAL_DEFAULT
    prompt_ia = plantilla.format(contexto=contexto, pregunta=pregunta)

    t_llm           = time.time()
    respuesta_cruda = llm.invoke(prompt_ia)
    respuesta       = respuesta_cruda.content if hasattr(respuesta_cruda, "content") else respuesta_cruda

    # ── 8. Guardado en caché L2 ───────────────────────────────────────────────
    guardar_en_cache(
        pregunta,
        respuesta,
        documento_origen=coleccion_principal or "desconocido",
        motor_vectores=motor_vectores,
        motor_llm=motor_llm,
    )

    return respuesta