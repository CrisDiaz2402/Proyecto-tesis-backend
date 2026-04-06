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
    get_umbral_relevancia, get_k_retrieval, get_num_tokens,
)

# ── PHOENIX TRACING ───────────────────────────────────────────────────────────
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

px.launch_app()
tracer_provider = register(project_name="tesis-epn-rag", auto_instrument=False)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
_tracer = tracer_provider.get_tracer(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CACHÉ L1 — En memoria RAM. Responde en microsegundos para preguntas repetidas.
# Se pierde al reiniciar el servidor (comportamiento intencional).
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_L1: dict[str, str] = {}
_MAX_L1_ENTRIES = 500


def _llave_l1(pregunta: str, motor_vectores: str, motor_llm: str) -> str:
    texto = f"{motor_vectores}:{motor_llm}:{pregunta.lower().strip()}"
    return hashlib.md5(texto.encode()).hexdigest()


def _guardar_l1(llave: str, respuesta: str) -> None:
    if len(_CACHE_L1) >= _MAX_L1_ENTRIES:
        # FIFO simple: eliminar la entrada más antigua
        del _CACHE_L1[next(iter(_CACHE_L1))]
    _CACHE_L1[llave] = respuesta


# ─────────────────────────────────────────────────────────────────────────────
# PALABRAS CLAVE PARA DETECCIÓN DE PREGUNTAS DE LISTA LARGA
# Determina si se necesitan más tokens en la respuesta.
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
    """Retorna (vector_dir, embeddings) según el motor de vectores elegido."""
    if motor_vectores == "cloud":
        return (
            str(VECTOR_STORE_DIR_CLOUD),
            GoogleGenerativeAIEmbeddings(model=EMBED_MODEL_CLOUD, google_api_key=GOOGLE_API_KEY),
        )
    return str(VECTOR_STORE_DIR_LOCAL), OllamaEmbeddings(model=EMBED_MODEL_LOCAL)


def _get_llm(motor_llm: str, num_tokens: int):
    """Retorna (llm, model_name) según el motor LLM elegido."""
    if motor_llm == "cloud":
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_CLOUD,
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
            max_tokens=num_tokens,
        )
        return llm, LLM_MODEL_CLOUD

    llm = OllamaLLM(
        model=LLM_MODEL_LOCAL,
        temperature=0,
        num_predict=num_tokens,
        top_k=10,
        top_p=0.5,
        repeat_penalty=1.3,
        stop=["Consulta del usuario:", "Usuario:", "Pregunta:", "[FIN]"],
    )
    return llm, LLM_MODEL_LOCAL


def _obtener_todas_las_colecciones(vector_dir: str) -> list[str]:
    try:
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
# Genera una respuesta hipotética corta para mejorar el match semántico.
# El embedding de esa respuesta es más cercano al fragmento del documento
# que el embedding de la pregunta original.
# ─────────────────────────────────────────────────────────────────────────────

def _aplicar_hyde(pregunta: str) -> str:
    """
    Genera una respuesta hipotética corta usando el LLM local.
    Se usa solo en modo local para compensar el gap vocabulario.
    Si falla por cualquier razón, devuelve la pregunta original como fallback.
    """
    try:
        llm_hyde = OllamaLLM(
            model=LLM_MODEL_LOCAL,
            temperature=0,
            num_predict=120,  # respuesta corta → rápido
        )
        hyde_prompt = (
            "Escribe una respuesta corta, factual y directa en español a esta pregunta "
            "sobre la malla curricular de la Carrera de Ciencias de la Computación de la EPN. "
            "Usa términos académicos concretos. Máximo 3 oraciones. "
            "No expliques, solo responde con datos.\n\n"
            f"Pregunta: {pregunta}\nRespuesta:"
        )
        respuesta_hipotetica = llm_hyde.invoke(hyde_prompt)
        texto = respuesta_hipotetica if isinstance(respuesta_hipotetica, str) else respuesta_hipotetica.content
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
    """Función auxiliar para búsqueda paralela en una colección de ChromaDB."""
    nombre_col, vector_dir, embeddings, query, k_retrieval, umbral = args
    try:
        db = Chroma(
            persist_directory=vector_dir,
            embedding_function=embeddings,
            collection_name=nombre_col,
        )
        resultados = db.similarity_search_with_relevance_scores(query, k=k_retrieval)
        filtrados = [(doc, score) for doc, score in resultados if score >= umbral]
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
    Usa SemanticChunker para dividir el documento respetando los límites
    semánticos naturales del texto, evitando cortes arbitrarios.
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

    # ── Semantic Chunking ─────────────────────────────────────────────────────
    # Divide el documento solo donde cambia el tema (baja similitud coseno entre
    # oraciones consecutivas), preservando secciones completas como chunks.
    vector_dir, embeddings = _get_embeddings_and_dir(motor_vectores)

    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=75,
    )
    fragmentos = splitter.split_documents(documentos)

    print(f"[RAG] 📄 Semantic chunking ({motor_vectores}): {len(fragmentos)} fragmentos generados.")

    Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=vector_dir,
        collection_name=nombre_coleccion,
    )
    return {
        "mensaje":   f"Procesado en {motor_vectores}: {len(fragmentos)} fragmentos.",
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
    vector_dir, _ = _get_embeddings_and_dir(motor_vectores)
    try:
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
    Pipeline RAG completo con las siguientes optimizaciones:
      1. Caché L1 en RAM (microsegundos para preguntas exactamente repetidas)
      2. Caché L2 semántico por modo (milisegundos para preguntas similares)
      3. HyDE en modo local (mejora el match semántico vocabulario→documento)
      4. Retrieval paralelo (todas las colecciones en simultáneo)
      5. Tokens de respuesta divididos por motor LLM
    """
    t_inicio = time.time()
    motor_vectores, motor_llm = _parsear_motor(motor)

    # ── 1. Caché L1 (RAM) — respuesta instantánea para preguntas repetidas exactas ──
    llave_l1 = _llave_l1(pregunta, motor_vectores, motor_llm)
    if llave_l1 in _CACHE_L1:
        print(f"[RAG] ⚡ L1 cache hit ({motor_vectores}:{motor_llm})")
        return _CACHE_L1[llave_l1]

    # ── 2. Caché L2 semántico ──────────────────────────────────────────────────
    cached = buscar_en_cache(pregunta, motor_vectores=motor_vectores, motor_llm=motor_llm)
    if cached:
        print(f"[RAG] ⚡ L2 cache hit ({motor_vectores}:{motor_llm})")
        with _tracer.start_as_current_span("cache_hit") as span:
            span.set_attributes({
                "tipo": "CACHE_HIT",
                "pregunta": pregunta,
                "motor_vectores": motor_vectores,
                "motor_llm": motor_llm,
                "latencia_ms": round((time.time() - t_inicio) * 1000, 2),
            })
        _guardar_l1(llave_l1, cached)
        return cached

    # ── 3. Recuperar vector store ──────────────────────────────────────────────
    vector_dir, embeddings = _get_embeddings_and_dir(motor_vectores)
    if not os.path.exists(vector_dir):
        return f"Error: La base de datos de vectores '{motor_vectores}' está vacía."

    colecciones = _obtener_todas_las_colecciones(vector_dir)
    if not colecciones:
        return f"Lo siento, no hay documentos en la base de conocimiento '{motor_vectores}'."

    # ── 4. HyDE (solo modo local) ──────────────────────────────────────────────
    # Genera una respuesta hipotética para mejorar el match vocabulario↔documento.
    if motor_vectores == "local":
        query_busqueda = _aplicar_hyde(pregunta)
    else:
        query_busqueda = pregunta

    # ── 5. Retrieval paralelo ──────────────────────────────────────────────────
    k_retrieval = get_k_retrieval(motor_vectores)
    umbral      = get_umbral_relevancia(motor_vectores)

    args_lista = [
        (col, vector_dir, embeddings, query_busqueda, k_retrieval, umbral)
        for col in colecciones
    ]

    resultados_con_score: list[tuple] = []
    coleccion_principal: str | None   = None

    with ThreadPoolExecutor(max_workers=min(len(colecciones), 5)) as executor:
        futuros = {executor.submit(_buscar_en_coleccion, args): args[0] for args in args_lista}
        for futuro in as_completed(futuros):
            nombre_col, filtrados = futuro.result()
            if filtrados and not coleccion_principal:
                coleccion_principal = nombre_col
            resultados_con_score.extend(filtrados)

    # Ordenar por score descendente y tomar los mejores K
    resultados_con_score.sort(key=lambda x: x[1], reverse=True)
    resultados = [doc for doc, _ in resultados_con_score[:k_retrieval]]

    if not resultados:
        return "Lo siento, esa información no existe en mi base de datos oficial."

    contexto = "\n\n---\n\n".join([doc.page_content for doc in resultados])

    # ── 6. Determinar límite de tokens según motor y tipo de pregunta ──────────
    es_lista = _es_pregunta_de_lista_larga(pregunta)
    num_tokens = get_num_tokens(motor_llm, es_lista)

    # ── 7. Instanciar LLM y ejecutar prompt ───────────────────────────────────
    llm, model_name = _get_llm(motor_llm, num_tokens)
    embed_name      = EMBED_MODEL_CLOUD if motor_vectores == "cloud" else EMBED_MODEL_LOCAL

    prompt_ia = f"""Eres el Asistente Académico de la EPN. Eres un sistema estricto de extracción de datos, no un consejero.

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

    t_llm           = time.time()
    respuesta_cruda = llm.invoke(prompt_ia)
    respuesta       = respuesta_cruda.content if hasattr(respuesta_cruda, "content") else respuesta_cruda

    # ── 8. Métricas Phoenix ────────────────────────────────────────────────────
    with _tracer.start_as_current_span("rag_consulta") as span:
        span.set_attributes({
            "tipo":              "RAG_REAL",
            "pregunta":          pregunta,
            "respuesta":         respuesta,
            "motor_vectores":    motor_vectores,
            "motor_llm":         motor_llm,
            "fragmentos_usados": len(resultados),
            "colecciones":       str(colecciones),
            "latencia_llm_ms":   round((time.time() - t_llm)   * 1000, 2),
            "latencia_total_ms": round((time.time() - t_inicio) * 1000, 2),
            "modelo_llm":        model_name,
            "modelo_embed":      embed_name,
            "k_retrieval":       k_retrieval,
            "umbral_relevancia": umbral,
            "num_tokens":        num_tokens,
            "hyde_aplicado":     motor_vectores == "local",
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