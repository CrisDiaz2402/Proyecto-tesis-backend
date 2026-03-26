# app/services/rag_service.py
import os
import time
import pymupdf4llm
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from app.services.cache_service import buscar_en_cache, guardar_en_cache
from app.core.config import (
    VECTOR_STORE_DIR, LLM_MODEL, EMBED_MODEL, 
    UMBRAL_RELEVANCIA, CHUNK_SIZE, CHUNK_OVERLAP, RAG_K_RETRIEVAL, RAG_K_FINAL
)

# ── PHOENIX TRACING ───────────────────────────────────────────────────────────
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from phoenix.otel import register

px.launch_app()

tracer_provider = register(project_name="tesis-epn-rag", auto_instrument=False)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
_tracer = tracer_provider.get_tracer(__name__)

# ── CONSTANTES DE LÓGICA DE NEGOCIO ───────────────────────────────────────────
PALABRAS_CLAVE_TOTALES = [
    "horas", "total", "autónomo", "autonomo", "práctico", "practico",
    "experimental", "aprendizaje", "contacto", "docente", "resumen",
    "cuántas horas", "cuantas horas", "graduación", "graduarme",
    "graduarse", "requisitos adicionales", "inglés", "ingles",
    "deportes", "clubes", "emprendimiento", "ecología", "ecologia",
    "comunicación", "comunicacion", "proyectos", "titulación", "titulacion"
]

QUERY_TOTALES = "TABLAS RESUMEN Organización Aprendizaje Autónomo Práctico-Experimental total horas 2800 1104 2000 5904 6480"
QUERY_GRADUACION = "Requisitos Adicionales Graduación inglés B1 deportes clubes emprendimiento ecología comunicación proyectos"

PALABRAS_LISTA_LARGA = [
    "todas las materias", "todos los niveles", "lista completa",
    "enumera todas", "todos los semestres", "todos los niveles",
    "qué materias hay en", "materias del nivel", "cuáles son todas"
]

# ── FUNCIONES INTERNAS ────────────────────────────────────────────────────────
def _inferir_categoria(nombre_archivo: str) -> dict:
    nombre = nombre_archivo.lower()
    if any(k in nombre for k in ["malla", "curricul", "pensum", "plan_estudio"]): return {"categoria": "malla_curricular", "tipo": "academico"}
    if any(k in nombre for k in ["graduac", "titulac", "requisito", "egreso"]): return {"categoria": "requisitos_graduacion", "tipo": "academico"}
    if any(k in nombre for k in ["reglamento", "normativa", "resolucion"]):     return {"categoria": "reglamento", "tipo": "normativo"}
    if any(k in nombre for k in ["horario", "calendario", "fechas"]):           return {"categoria": "horarios", "tipo": "administrativo"}
    return {"categoria": "general", "tipo": "desconocido"}

def _obtener_todas_las_colecciones() -> list[str]:
    try:
        import chromadb
        cliente = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        return [c.name for c in cliente.list_collections() if c.name != "cache_respuestas"]
    except Exception as e:
        print(f"[RAG] Error al listar colecciones: {e}")
        return []

def _necesita_totales(pregunta: str) -> bool:
    return any(k in pregunta.lower() for k in PALABRAS_CLAVE_TOTALES)

def _necesita_graduacion(pregunta: str) -> bool:
    return any(k in pregunta.lower() for k in ["graduación", "graduarme", "graduarse", "requisitos adicionales", "inglés", "ingles", "deportes", "clubes", "emprendimiento", "ecología", "ecologia", "comunicación", "comunicacion", "proyectos"])

def _es_pregunta_de_lista_larga(pregunta: str) -> bool:
    return any(k in pregunta.lower() for k in PALABRAS_LISTA_LARGA)

def _agregar_sin_duplicados(base: list, extras: list) -> list:
    vistos = {doc.page_content for doc in base}
    for doc in extras:
        if doc.page_content not in vistos:
            base.append(doc)
            vistos.add(doc.page_content)
    return base

# ── ENDPOINTS DE SERVICIO ─────────────────────────────────────────────────────
def procesar_y_guardar_documento(filepath: str) -> dict:
    if not os.path.exists(filepath): raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

    extension = os.path.splitext(filepath)[1].lower()
    nombre_coleccion = os.path.splitext(os.path.basename(filepath))[0]
    meta_extra = _inferir_categoria(os.path.basename(filepath))

    if extension == ".pdf":
        documentos = [Document(page_content=pymupdf4llm.to_markdown(filepath), metadata={"source": filepath, **meta_extra})]
    elif extension == ".docx":
        documentos = Docx2txtLoader(filepath).load()
        for doc in documentos: doc.metadata.update(meta_extra)
    elif extension in [".txt", ".md"]:
        documentos = TextLoader(filepath, encoding="utf-8").load()
        for doc in documentos: doc.metadata.update(meta_extra)
    else:
        raise ValueError(f"Formato no soportado: {extension}")

    splitter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    fragmentos = splitter.split_documents(documentos)

    Chroma.from_documents(
        documents=fragmentos,
        embedding=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=VECTOR_STORE_DIR,
        collection_name=nombre_coleccion
    )
    return {"mensaje": f"Procesado: {len(fragmentos)} fragmentos.", "coleccion": nombre_coleccion}

def eliminar_coleccion_chroma(nombre_coleccion: str) -> dict:
    try:
        if not os.path.exists(VECTOR_STORE_DIR): return {"mensaje": "Vector store no existe."}
        db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=OllamaEmbeddings(model=EMBED_MODEL), collection_name=nombre_coleccion)
        db.delete_collection()
        print(f"[CHROMA] 🗑️ Colección '{nombre_coleccion}' eliminada.")
        return {"mensaje": f"Colección '{nombre_coleccion}' eliminada."}
    except Exception as e:
        print(f"[CHROMA] Error al eliminar '{nombre_coleccion}': {e}")
        return {"mensaje": str(e)}

def eliminar_todos_los_vectores_chroma() -> dict:
    try:
        import chromadb
        cliente = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        for c in cliente.list_collections():
            if c.name != "cache_respuestas": cliente.delete_collection(c.name)
        print("[CHROMA] 🗑️ Todas las colecciones eliminadas.")
        return {"mensaje": "Todos los vectores han sido eliminados."}
    except Exception as e:
        print(f"[CHROMA] Error crítico al vaciar vectores: {e}")
        return {"mensaje": str(e)}

def consultar_base_conocimiento(pregunta: str) -> str:
    t_inicio = time.time()
    cached = buscar_en_cache(pregunta)
    
    if cached:
        print("[RAG] ⚡ Servido desde caché")
        with _tracer.start_as_current_span("cache_hit") as span:
            span.set_attributes({"tipo": "CACHE_HIT", "pregunta": pregunta, "latencia_ms": round((time.time() - t_inicio) * 1000, 2)})
        return cached

    if not os.path.exists(VECTOR_STORE_DIR): return "Error: La base de datos está vacía."
    
    colecciones = _obtener_todas_las_colecciones()
    if not colecciones: return "Lo siento, no hay documentos en la base de conocimiento."

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    resultados_con_score = []
    coleccion_principal = None

    for nombre_col in colecciones:
        try:
            db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings, collection_name=nombre_col)
            resultados_col = db.similarity_search_with_relevance_scores(pregunta, k=RAG_K_RETRIEVAL)
            
            filtrados = [(doc, score) for doc, score in resultados_col if score >= UMBRAL_RELEVANCIA]
            for doc, score in filtrados:
                doc.metadata.update({"_coleccion": nombre_col, "_score": round(score, 4)})
            
            resultados_con_score.extend(filtrados)
            if filtrados and not coleccion_principal: coleccion_principal = nombre_col
        except Exception:
            continue

    resultados_con_score.sort(key=lambda x: x[1], reverse=True)
    resultados = [doc for doc, score in resultados_con_score[:RAG_K_FINAL]]

    # ── EXTRACCIONES ADICIONALES DE CONTEXTO ──
    if _necesita_totales(pregunta):
        for nombre_col in colecciones:
            try:
                extras = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings, collection_name=nombre_col).similarity_search(QUERY_TOTALES, k=3)
                resultados = _agregar_sin_duplicados(resultados, extras)
            except Exception: pass

    if _necesita_graduacion(pregunta):
        for nombre_col in colecciones:
            try:
                extras = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings, collection_name=nombre_col).similarity_search(QUERY_GRADUACION, k=3)
                resultados = _agregar_sin_duplicados(resultados, extras)
            except Exception: pass

    if not resultados:
        return "Lo siento, esa información no existe en mi base de datos oficial."

    contexto = "\n\n---\n\n".join([doc.page_content for doc in resultados])
    num_tokens = 700 if _es_pregunta_de_lista_larga(pregunta) else 400

    llm = OllamaLLM(
        model=LLM_MODEL, temperature=0, num_predict=num_tokens,
        top_k=10, top_p=0.5, repeat_penalty=1.2,
        stop=["Consulta del usuario:", "Usuario:", "Pregunta:", "[FIN]"]
    )

    prompt_ia = f"""Eres el Asistente Académico de la EPN. Eres un sistema estricto de extracción de datos, no un consejero.

REGLAS ESTRICTAS E INQUEBRANTABLES:
1. Cero Alucinaciones: Responde ÚNICAMENTE usando los datos explícitos del CONTEXTO.
2. Prohibido adivinar: NUNCA inventes nombres de materias, prerrequisitos, créditos, niveles o recomendaciones. Si el usuario pregunta por una materia específica (ej. Inteligencia Artificial) y no ves su nombre exacto en el CONTEXTO, no intentes adivinar dónde podría estar.
3. Regla de Vacío OBLIGATORIA: Si la respuesta a la pregunta NO está claramente escrita en el CONTEXTO, tienes PROHIBIDO intentar ayudar o suponer. Debes responder exactamente con esta frase y nada más: "Lo siento, esa información no existe en mi base de datos oficial."
4. Estilo Directo: Responde directamente con la información. NUNCA uses frases como "Según el contexto", "Te recomiendo", o "El documento dice".

CONTEXTO DE CONOCIMIENTO:
{contexto}

Pregunta del usuario: {pregunta}
Respuesta:
[FIN]"""

    t_llm = time.time()
    respuesta = llm.invoke(prompt_ia)
    
    with _tracer.start_as_current_span("rag_consulta") as span:
        span.set_attributes({
            "tipo": "RAG_REAL", "pregunta": pregunta, "respuesta": respuesta,
            "fragmentos_usados": len(resultados), "colecciones": str(colecciones),
            "latencia_llm_ms": round((time.time() - t_llm) * 1000, 2),
            "latencia_total_ms": round((time.time() - t_inicio) * 1000, 2),
            "modelo_llm": LLM_MODEL, "modelo_embed": EMBED_MODEL,
            "umbral_relevancia": UMBRAL_RELEVANCIA, "num_tokens": num_tokens
        })

    guardar_en_cache(pregunta, respuesta, documento_origen=coleccion_principal or "desconocido")
    return respuesta