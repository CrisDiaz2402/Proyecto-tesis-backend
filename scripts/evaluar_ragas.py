import os
import sys
import math
import warnings
import time
import asyncio
import requests as http_requests
from datasets import Dataset
from ragas import evaluate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ragas.metrics import faithfulness, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import BaseRagasEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
FASTAPI_URL      = "http://localhost:8000/api/chat/consultar"
VECTOR_STORE_DIR = "./vector_store"
EMBED_MODEL      = "nomic-embed-text"
LLM_JUEZ_MODEL   = "llama3"

# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER DE EMBEDDINGS COMPATIBLE CON RAGAS
# BaseRagasEmbeddings es clase abstracta que exige 4 métodos:
#   embed_query, embed_documents  (síncronos)
#   aembed_query, aembed_documents (asíncronos)
# OllamaEmbeddings solo implementa los síncronos, así que los async
# se implementan corriendo el equivalente síncrono en un executor.
# ─────────────────────────────────────────────────────────────────────────────
class OllamaEmbeddingsRagas(BaseRagasEmbeddings):
    def __init__(self, model: str):
        self._embedder = OllamaEmbeddings(model=model)

    # ── Síncronos ─────────────────────────────────────────────────────────────
    def embed_query(self, text: str):
        return self._embedder.embed_query(text)

    def embed_documents(self, texts: list):
        return self._embedder.embed_documents(texts)

    # Ragas llama embed_text() internamente en algunas versiones
    def embed_text(self, text: str):
        return self._embedder.embed_query(text)

    # ── Asíncronos (requeridos por BaseRagasEmbeddings) ───────────────────────
    async def aembed_query(self, text: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    async def aembed_documents(self, texts: list):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_documents, texts)

# ─────────────────────────────────────────────────────────────────────────────
# PREGUNTAS + GROUND TRUTH MANUAL
# ─────────────────────────────────────────────────────────────────────────────
CASOS = [
    {
        "categoria":    "📊 Datos numéricos",
        "pregunta":     "¿Cuántas horas de aprendizaje autónomo exige la carrera?",
        "ground_truth": "2800 horas",
    },
    {
        "categoria":    "📚 Materias por nivel",
        "pregunta":     "¿Qué materias hay en el nivel 1?",
        "ground_truth": "Álgebra Lineal, Cálculo en una Variable, Mecánica Newtoniana, Programación I, Comunicación Oral y Escrita",
    },
    {
        "categoria":    "🔗 Prerrequisitos",
        "pregunta":     "¿Cuáles son los prerrequisitos de Inteligencia Artificial?",
        "ground_truth": "Métodos Numéricos (ICCD412) y Estructura de Datos y Algoritmos II (ICCD342)",
    },
    {
        "categoria":    "🎓 Graduación",
        "pregunta":     "¿Qué nivel de inglés necesito para graduarme?",
        "ground_truth": "B1",
    },
    {
        "categoria":    "🏅 Créditos",
        "pregunta":     "¿Cuántos créditos en total necesito para graduarme?",
        # Dato correcto: 9 niveles × 15 créditos = 135
        # El chatbot responde 180 → Answer Correctness debe detectarlo
        "ground_truth": "135 créditos",
    },
    {
        "categoria":    "🚫 Fuera del dominio",
        "pregunta":     "¿Cuál es el horario de atención de la facultad?",
        "ground_truth": "Lo siento, esa información no existe en la base de datos oficial.",
    },
]

TOTAL_PREGUNTAS = len(CASOS)

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: llama a FastAPI por HTTP
# ─────────────────────────────────────────────────────────────────────────────
def preguntar_api(pregunta: str) -> str:
    try:
        resp = http_requests.post(FASTAPI_URL, json={"pregunta": pregunta}, timeout=180)
        if resp.status_code == 200:
            return resp.json().get("respuesta", "Sin respuesta")
        return f"Error HTTP {resp.status_code}"
    except Exception as e:
        return f"Error de conexión: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# INICIO
# ─────────────────────────────────────────────────────────────────────────────
print("⚖️  Iniciando Evaluador Ragas — Faithfulness + Answer Correctness")
print(f"   Modelo Juez  : {LLM_JUEZ_MODEL} (format=json)")
print(f"   Modelo Embed : {EMBED_MODEL}")
print(f"   API FastAPI  : {FASTAPI_URL}")
print(f"   Preguntas    : {TOTAL_PREGUNTAS}\n")
print("⚠️  Asegúrate de que FastAPI esté corriendo en http://localhost:8000\n")

juez_llm         = LangchainLLMWrapper(ChatOllama(model=LLM_JUEZ_MODEL, temperature=0, format="json"))
ragas_embeddings = OllamaEmbeddingsRagas(model=EMBED_MODEL)
lc_embeddings    = OllamaEmbeddings(model=EMBED_MODEL)   # para ChromaDB, sin wrapper

faithfulness.llm              = juez_llm
answer_correctness.llm        = juez_llm
answer_correctness.embeddings = ragas_embeddings

db = Chroma(
    persist_directory=VECTOR_STORE_DIR,
    embedding_function=lc_embeddings,
    collection_name="malla_computacion"
)

# ─────────────────────────────────────────────────────────────────────────────
# FASE 1: RECOPILAR RESPUESTAS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("FASE 1: Obteniendo respuestas desde FastAPI...")
print("=" * 60)

for caso in CASOS:
    print(f"\n{caso['categoria']}")
    print(f"  🔍 {caso['pregunta']}")

    t0        = time.time()
    respuesta = preguntar_api(caso["pregunta"])
    latencia  = round((time.time() - t0) * 1000)

    docs     = db.similarity_search(caso["pregunta"], k=8)
    contexto = [doc.page_content for doc in docs]

    caso["respuesta"] = respuesta
    caso["contexto"]  = contexto
    caso["latencia"]  = latencia

    origen = "⚡ caché" if latencia < 500 else f"🤖 RAG ({latencia}ms)"
    print(f"     → {origen}")
    print(f"     → Respuesta del chatbot : {respuesta[:120]}")
    print(f"     → Ground truth esperado : {caso['ground_truth']}")

print(f"\n✅ {TOTAL_PREGUNTAS} respuestas recopiladas.\n")

# ─────────────────────────────────────────────────────────────────────────────
# FASE 2: EVALUAR CON RAGAS — secuencial, métrica por métrica
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("FASE 2: Evaluando con Ragas (Faithfulness + Answer Correctness)...")
print(f"⏳ {LLM_JUEZ_MODEL} evalúa cada pregunta de a una.")
print("=" * 60)

def extraer_score(resultado, clave):
    raw = resultado[clave]
    val = raw[0] if isinstance(raw, list) else raw
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val)

def fmt(val):
    if val is None: return "  nan"
    if val >= 0.9:  return f"✅ {val:.2f}"
    if val >= 0.7:  return f"⚠️  {val:.2f}"
    return              f"❌ {val:.2f}"

for i, caso in enumerate(CASOS):
    print(f"\n  [{i+1}/{TOTAL_PREGUNTAS}] {caso['categoria']}")
    print(f"     P: {caso['pregunta'][:70]}")

    dataset = Dataset.from_dict({
        "question":    [caso["pregunta"]],
        "answer":      [caso["respuesta"]],
        "contexts":    [caso["contexto"]],
        "ground_truth":[caso["ground_truth"]],
    })

    # ── Faithfulness ──────────────────────────────────────────────────────────
    try:
        res_f = evaluate(dataset, metrics=[faithfulness], raise_exceptions=False)
        caso["score_faithfulness"] = extraer_score(res_f, "faithfulness")
    except Exception as e:
        caso["score_faithfulness"] = None
        print(f"     ⚠️  Faithfulness error: {e}")

    # ── Answer Correctness ────────────────────────────────────────────────────
    try:
        res_c = evaluate(dataset, metrics=[answer_correctness], raise_exceptions=False)
        caso["score_correctness"] = extraer_score(res_c, "answer_correctness")
    except Exception as e:
        caso["score_correctness"] = None
        print(f"     ⚠️  Answer Correctness error: {e}")

    print(f"     Faithfulness         : {fmt(caso['score_faithfulness'])}")
    print(f"     Answer Correctness   : {fmt(caso['score_correctness'])}")

# ─────────────────────────────────────────────────────────────────────────────
# FASE 3: TABLA RESUMEN COMPLETA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("🏆 RESULTADOS COMPLETOS")
print("=" * 70)

scores_f = []
scores_c = []

def icono_score(val):
    if val is None:  return "❓  nan"
    if val >= 0.9:   return f"✅ {val:.2f}"
    if val >= 0.7:   return f"⚠️  {val:.2f}"
    return               f"❌ {val:.2f}"

for caso in CASOS:
    sf     = caso.get("score_faithfulness")
    sc     = caso.get("score_correctness")
    origen = "⚡ caché" if caso["latencia"] < 500 else f"{caso['latencia']}ms"

    print(f"\n  {caso['categoria']} ({origen})")
    print(f"     Pregunta             : {caso['pregunta']}")
    print(f"     Respuesta chatbot    : {caso['respuesta'][:120]}")
    print(f"     Ground truth         : {caso['ground_truth']}")
    print(f"     Faithfulness         : {icono_score(sf)}")
    print(f"     Answer Correctness   : {icono_score(sc)}")

    if sf is not None and not math.isnan(sf): scores_f.append(sf)
    if sc is not None and not math.isnan(sc): scores_c.append(sc)

# ─────────────────────────────────────────────────────────────────────────────
# FASE 4: RESUMEN ESTADÍSTICO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("📈 RESUMEN ESTADÍSTICO")
print("=" * 70)

def resumen_metrica(nombre, scores, total):
    nan_c = total - len(scores)
    if not scores:
        print(f"\n  {nombre}: todos los scores fueron nan.")
        return
    promedio   = sum(scores) / len(scores)
    perfectos  = sum(1 for s in scores if s >= 0.9)
    aceptables = sum(1 for s in scores if 0.7 <= s < 0.9)
    bajos      = sum(1 for s in scores if s < 0.7)
    if promedio >= 0.9:   veredicto = "✅ EXCELENTE"
    elif promedio >= 0.7: veredicto = "⚠️  ACEPTABLE"
    else:                 veredicto = "❌ REQUIERE MEJORAS"
    print(f"\n  📐 {nombre}")
    print(f"     Promedio               : {promedio:.4f}")
    print(f"     ✅ Excelentes  (≥ 0.90) : {perfectos}")
    print(f"     ⚠️  Aceptables (0.70-0.89): {aceptables}")
    print(f"     ❌ Bajos       (< 0.70)  : {bajos}")
    if nan_c:
        print(f"     ❓ No evaluados (nan)   : {nan_c}")
    print(f"     Veredicto              : {veredicto}")

resumen_metrica("Faithfulness       (¿responde basado en el contexto?)", scores_f, TOTAL_PREGUNTAS)
resumen_metrica("Answer Correctness (¿la respuesta es factualmente correcta?)", scores_c, TOTAL_PREGUNTAS)

print()
print("  ─────────────────────────────────────────────────────────────────")
print("  ℹ️  NOTAS DE INTERPRETACIÓN:")
print("  • Faithfulness 0.00 en 'Fuera del dominio' es CORRECTO:")
print("    el sistema rechazó sin inventar datos del contexto.")
print("  • Answer Correctness detecta errores factuales que Faithfulness")
print("    no detecta. Ejemplo: responder 180 créditos en vez de 135")
print("    puede tener Faithfulness=1.00 pero Correctness bajo.")
print("  • Después de corregir y reingestar el MD, ambas métricas")
print("    deberían subir a ≥ 0.90 en la categoría 'Créditos'.")
print("  ─────────────────────────────────────────────────────────────────")
print("=" * 70)