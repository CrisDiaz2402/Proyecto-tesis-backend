# app/services/rag_service.py
import os
import time
import hashlib
import pymupdf4llm
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from app.services.cache_service import buscar_en_cache, guardar_en_cache
from app.core.config import (
    VECTOR_STORE_DIR_LOCAL, VECTOR_STORE_DIR_CLOUD,
    LLM_MODEL_LOCAL, LLM_MODEL_CLOUD,
    EMBED_MODEL_LOCAL, EMBED_MODEL_CLOUD, GOOGLE_API_KEY,
)
# ── Parámetros RAG dinámicos — fuente de verdad desde la BD ──────────────────
from app.services.rag_params_service import get_params

# ── PHOENIX TRACING ───────────────────────────────────────────────────────────
# NOTA: px.launch_app() se llama SOLO en main.py para evitar doble arranque.
# Aquí solo se registra el tracer provider para los spans del RAG.
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

tracer_provider = register(project_name="tesis-epn-rag", auto_instrument=False)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
_tracer = tracer_provider.get_tracer(__name__)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS POR DEFECTO — hardcodeados como fallback.
# Se usan cuando prompt_principal / prompt_hyde en la BD son NULL.
# Deben contener los placeholders {contexto} y {pregunta} / {pregunta}.
# ─────────────────────────────────────────────────────────────────────────────
_PROMPT_PRINCIPAL_DEFAULT = """Eres el Asistente Académico de la EPN. Eres un sistema estricto de extracción de datos, no un consejero.

REGLAS ESTRICTAS E INQUEBRANTABLES:
1. Cero Alucinaciones: Responde ÚNICAMENTE usando los datos explícitos o claramente implicados por el CONTEXTO.
2. Prohibido adivinar: NUNCA inventes nombres de materias, prerrequisitos, créditos, niveles o recomendaciones.
3. Regla de Vacío OBLIGATORIA: Si la pregunta trata sobre algo completamente ausente del CONTEXTO (ningún dato, ninguna referencia directa ni indirecta), responde exactamente: "Lo siento, esa información no existe en mi base de datos oficial." — Si el CONTEXTO contiene datos relacionados que implican o contradicen el dato de la pregunta, úsalos para responder aunque la respuesta no sea una cita textual exacta.
4. Estilo Directo: Responde directamente con la información. NUNCA uses frases como "Según el contexto", "Te recomiendo", o "El documento dice".
5. Entidades inexistentes: Si el usuario pregunta por una materia, código o persona que NO aparece nombrada en el CONTEXTO, responde solo con el mensaje de vacío de la Regla 3. No sugieras alternativas similares ni menciones otras materias del CONTEXTO como reemplazo.

CONTEXTO DE CONOCIMIENTO:
{contexto}

Pregunta del usuario: {pregunta}
Respuesta:
[FIN]"""

_PROMPT_HYDE_DEFAULT = (
    "Escribe una respuesta corta, factual y directa en español a esta pregunta "
    "sobre la malla curricular de la Carrera de Ciencias de la Computación de la EPN. "
    "Usa términos académicos concretos. Máximo 3 oraciones. "
    "No expliques, solo responde con datos.\n\n"
    "Pregunta: {pregunta}\nRespuesta:"
)

# ─────────────────────────────────────────────────────────────────────────────
# CACHÉ L1 — En memoria RAM. Responde en microsegundos para preguntas repetidas.
# Se pierde al reiniciar el servidor (comportamiento intencional).
# El tamaño máximo se lee dinámicamente desde get_params() en cada inserción.
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_L1: dict[str, str] = {}


def limpiar_cache_l1() -> None:
    """Vacía el caché L1 en RAM e invalida los singletons LLM.
    Llamado por el router de rag-params al guardar cambios de parámetros."""
    _CACHE_L1.clear()
    _invalidar_singletons_llm()
    print("[RAG] 🗑️ Caché L1 (RAM) limpiado.")


def _llave_l1(pregunta: str, motor_vectores: str, motor_llm: str) -> str:
    texto = f"{motor_vectores}:{motor_llm}:{pregunta.lower().strip()}"
    return hashlib.md5(texto.encode()).hexdigest()


def _guardar_l1(llave: str, respuesta: str) -> None:
    max_entries = get_params().get("max_l1_entries", 500)
    if len(_CACHE_L1) >= max_entries:
        # FIFO simple: eliminar la entrada más antigua
        del _CACHE_L1[next(iter(_CACHE_L1))]
    _CACHE_L1[llave] = respuesta


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETONS — Se crean una sola vez y se reutilizan en todas las consultas.
#
# Problema original: cada consulta creaba nuevas instancias de OllamaEmbeddings,
# OllamaLLM y chromadb.PersistentClient, añadiendo ~200-500ms de overhead
# innecesario por reconexión en cada llamada.
#
# _ollama_embeddings   → instancia única de OllamaEmbeddings (local)
# _ollama_llm_cache    → dict {num_tokens: OllamaLLM} — máx. 3 instancias
#                        (350 normal, 750 lista_larga, 120 HyDE)
# _chroma_client_local → PersistentClient único para vector_store_local
#
# IMPORTANTE: _invalidar_singletons_llm() debe llamarse cuando cambien
# top_k_llm, top_p_llm o repeat_penalty (ya se hace desde limpiar_cache_l1).
# ─────────────────────────────────────────────────────────────────────────────
_ollama_embeddings: OllamaEmbeddings | None = None
_ollama_llm_cache: dict[int, OllamaLLM] = {}
_chroma_client_local = None  # chromadb.ClientAPI


def _invalidar_singletons_llm() -> None:
    """Limpia el caché de instancias OllamaLLM.
    Necesario cuando cambian top_k, top_p o repeat_penalty para que la
    próxima consulta cree instancias con los nuevos parámetros."""
    global _ollama_llm_cache
    _ollama_llm_cache.clear()
    print("[RAG] 🔄 Singletons OllamaLLM invalidados (parámetros de sampling cambiados).")


def _get_ollama_embeddings() -> OllamaEmbeddings:
    """Retorna la instancia singleton de OllamaEmbeddings local.
    Se crea solo en la primera llamada y se reutiliza indefinidamente."""
    global _ollama_embeddings
    if _ollama_embeddings is None:
        _ollama_embeddings = OllamaEmbeddings(model=EMBED_MODEL_LOCAL)
        print(f"[RAG] ✅ OllamaEmbeddings inicializado (singleton, model={EMBED_MODEL_LOCAL})")
    return _ollama_embeddings


def _get_ollama_llm(num_tokens: int) -> OllamaLLM:
    """Retorna una instancia OllamaLLM cacheada por num_tokens.
    En operación normal solo existen 3 instancias: 350, 750 y 120 tokens."""
    global _ollama_llm_cache
    if num_tokens not in _ollama_llm_cache:
        params = get_params()
        _ollama_llm_cache[num_tokens] = OllamaLLM(
            model=LLM_MODEL_LOCAL,
            temperature=0,          # No modificable: determinismo crítico para el caché
            num_predict=num_tokens,
            top_k=params.get("top_k_llm", 10),
            top_p=params.get("top_p_llm", 0.5),
            repeat_penalty=params.get("repeat_penalty", 1.3),
            stop=["Consulta del usuario:", "Usuario:", "Pregunta:", "[FIN]"],
        )
        print(f"[RAG] ✅ OllamaLLM inicializado (singleton, num_tokens={num_tokens})")
    return _ollama_llm_cache[num_tokens]


def _get_chroma_client_local():
    """Retorna el PersistentClient singleton de ChromaDB para el motor local.
    Evita reabrir la base de datos en cada consulta y en cada listado de colecciones."""
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
    """
    Desempaqueta el motor recibido.
      - 'local:cloud' → (motor_vectores='local', motor_llm='cloud')
      - 'local'       → (motor_vectores='local', motor_llm='local')
    """
    if ":" in motor:
        partes = motor.split(":", 1)
        return partes[0], partes[1]
    return motor, motor


def _get_embeddings_and_dir(motor_vectores: str):
    """Retorna (vector_dir, embeddings) según el motor de vectores elegido.
    Para el motor local reutiliza el singleton de OllamaEmbeddings."""
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
    """Lee los límites de tokens desde los parámetros dinámicos."""
    params = get_params()
    if motor_llm == "cloud":
        return (
            params.get("num_tokens_lista_cloud",  1400)
            if es_lista_larga
            else params.get("num_tokens_normal_cloud", 700)
        )
    return (
        params.get("num_tokens_lista_local",  750)
        if es_lista_larga
        else params.get("num_tokens_normal_local", 350)
    )


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
    """
    Genera una respuesta hipotética corta usando el LLM local.
    num_predict y prompt_hyde se leen dinámicamente desde get_params().
    Reutiliza el singleton OllamaLLM en lugar de crear una nueva instancia.
    Si falla, devuelve la pregunta original como fallback.
    """
    try:
        params   = get_params()
        num_pred = params.get("hyde_num_predict", 120)

        # Leer prompt HyDE desde BD, con fallback al hardcodeado
        plantilla_hyde = params.get("prompt_hyde") or _PROMPT_HYDE_DEFAULT
        hyde_prompt    = plantilla_hyde.format(pregunta=pregunta)

        # Reutilizar singleton en lugar de crear OllamaLLM(model=...) nuevo
        llm_hyde = _get_ollama_llm(num_pred)

        respuesta_hipotetica = llm_hyde.invoke(hyde_prompt)
        texto    = respuesta_hipotetica if isinstance(respuesta_hipotetica, str) else respuesta_hipotetica.content
        resultado = texto.strip()[:500]
        print(f"[HyDE] Respuesta hipotética generada: {resultado[:80]}...")
        return resultado
    except Exception as e:
        print(f"[HyDE] Error al generar respuesta hipotética, usando pregunta original: {e}")
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
    breakpoint_threshold_amount se lee dinámicamente desde get_params().
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

    # ── Semantic Chunking — parámetro dinámico ────────────────────────────────
    params      = get_params()
    bta         = params.get("breakpoint_threshold_amount", 75)
    vector_dir, embeddings = _get_embeddings_and_dir(motor_vectores)

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=bta,
    )
    fragmentos = splitter.split_documents(documentos)

    print(f"[RAG] 📄 Semantic chunking ({motor_vectores}, percentil={bta}): {len(fragmentos)} fragmentos generados.")

    Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=vector_dir,
        collection_name=nombre_coleccion,
    )
    return {
        "mensaje":   f"Procesado en {motor_vectores}: {len(fragmentos)} fragmentos (percentil={bta}).",
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
      1. Caché L1 RAM      — microsegundos, preguntas exactamente repetidas.
      2. Caché L2 semántico — milisegundos, preguntas similares.
      3. HyDE (solo local) — mejora el match vocabulario↔documento.
      4. Retrieval paralelo — todas las colecciones en simultáneo.
      5. Generación LLM    — con parámetros y prompt dinámicos desde BD.
    """
    t_inicio = time.time()
    motor_vectores, motor_llm = _parsear_motor(motor)

    # ── 1. Caché L1 (RAM) ─────────────────────────────────────────────────────
    llave_l1 = _llave_l1(pregunta, motor_vectores, motor_llm)
    if llave_l1 in _CACHE_L1:
        print(f"[RAG] ⚡ L1 cache hit ({motor_vectores}:{motor_llm})")
        return _CACHE_L1[llave_l1]

    # ── 2. Caché L2 semántico ─────────────────────────────────────────────────
    cached = buscar_en_cache(pregunta, motor_vectores=motor_vectores, motor_llm=motor_llm)
    if cached:
        print(f"[RAG] ⚡ L2 cache hit ({motor_vectores}:{motor_llm})")
        with _tracer.start_as_current_span("cache_hit") as span:
            span.set_attributes({
                "tipo":           "CACHE_HIT",
                "pregunta":       pregunta,
                "motor_vectores": motor_vectores,
                "motor_llm":      motor_llm,
                "latencia_ms":    round((time.time() - t_inicio) * 1000, 2),
            })
        _guardar_l1(llave_l1, cached)
        return cached

    # ── 3. Recuperar vector store ─────────────────────────────────────────────
    vector_dir, embeddings = _get_embeddings_and_dir(motor_vectores)
    if not os.path.exists(vector_dir):
        return f"Error: La base de datos de vectores '{motor_vectores}' está vacía."

    colecciones = _obtener_todas_las_colecciones(vector_dir)
    if not colecciones:
        return f"Lo siento, no hay documentos en la base de conocimiento '{motor_vectores}'."

    # ── 4. HyDE (solo modo local) ─────────────────────────────────────────────
    query_busqueda = _aplicar_hyde(pregunta) if motor_vectores == "local" else pregunta

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

    # ── 8. Métricas Phoenix ────────────────────────────────────────────────────
    with _tracer.start_as_current_span("rag_consulta") as span:
        span.set_attributes({
            "tipo":               "RAG_REAL",
            "pregunta":           pregunta,
            "respuesta":          respuesta,
            "motor_vectores":     motor_vectores,
            "motor_llm":          motor_llm,
            "fragmentos_usados":  len(resultados),
            "colecciones":        str(colecciones),
            "latencia_llm_ms":    round((time.time() - t_llm)    * 1000, 2),
            "latencia_total_ms":  round((time.time() - t_inicio)  * 1000, 2),
            "modelo_llm":         model_name,
            "modelo_embed":       embed_name,
            "k_retrieval":        k_retrieval,
            "umbral_relevancia":  umbral,
            "num_tokens":         num_tokens,
            "hyde_aplicado":      motor_vectores == "local",
            # parámetros activos al momento de la consulta (trazabilidad)
            "bta":                params.get("breakpoint_threshold_amount", 75),
            "repeat_penalty":     params.get("repeat_penalty", 1.3),
            "top_k_llm":          params.get("top_k_llm", 10),
            "top_p_llm":          params.get("top_p_llm", 0.5),
            "prompt_personalizado": params.get("prompt_principal") is not None,
        })

    # ── 9. Guardar en cachés L2 y L1 ──────────────────────────────────────────
    guardar_en_cache(
        pregunta,
        respuesta,
        documento_origen=coleccion_principal or "desconocido",
        motor_vectores=motor_vectores,
        motor_llm=motor_llm,
    )
    _guardar_l1(llave_l1, respuesta)

    return respuesta