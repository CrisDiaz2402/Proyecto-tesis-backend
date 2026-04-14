# app/services/evaluacion_service.py
"""
Servicio de evaluación RAG.

Contiene toda la lógica de scoring trasladada desde scripts/evaluar_local.py:
  - Normalización de texto y matching flexible de claves
  - Scoring por tipo: contiene / no_contiene / corrige / no_alucina
  - Consulta a Phoenix REST API para métricas del sistema
  - Orquestación completa de una sesión de evaluación
  - Generador de progreso para streaming SSE (ejecutar_evaluacion_stream)
"""

import re
import time
import requests
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Generator

from app.services.rag_service import consultar_base_conocimiento
from app.services.config_service import obtener_motor_activo

PHOENIX_URL = "http://localhost:6006"


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN Y MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def _normalizar(texto: str) -> str:
    """Lowercase y sin tildes para comparación flexible."""
    texto = texto.lower()
    for src, dst in {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ñ":"n","ü":"u"}.items():
        texto = texto.replace(src, dst)
    return texto


def _contiene_clave(respuesta: str, clave: str) -> bool:
    """
    True si 'clave' aparece en 'respuesta' de forma flexible:
      1. Subcadena exacta normalizada
      2. Número rodeado de no-dígitos (ej: "135" matchea "135 créditos")
      3. Similitud de secuencia >= 0.85 como fallback para frases largas
    """
    r = _normalizar(respuesta)
    c = _normalizar(clave)

    if c in r:
        return True

    if clave.isdigit():
        patron = r"(?<!\d)" + re.escape(clave) + r"(?!\d)"
        if re.search(patron, respuesta):
            return True

    if len(c) > 5:
        if SequenceMatcher(None, c, r).ratio() >= 0.85:
            return True

    return False


def _es_rechazo(respuesta: str) -> bool:
    """True si la respuesta indica que el sistema no tiene información."""
    r = _normalizar(respuesta)
    frases = [
        "no existe en mi base",
        "no existe en la base",
        "lo siento",
        "no tengo informacion",
        "no hay informacion",
        "no se encuentra",
        "no consta",
        "no esta disponible",
        "no puedo",
        "no dispongo",
    ]
    return any(f in r for f in frases)


# ─────────────────────────────────────────────────────────────────────────────
# LÓGICA DE SCORING POR TIPO
# ─────────────────────────────────────────────────────────────────────────────

def _score_contiene(respuesta: str, caso: dict) -> tuple[float, str]:
    claves = caso["claves"]
    encontradas = [c for c in claves if _contiene_clave(respuesta, c)]
    n, k = len(claves), len(encontradas)

    if _es_rechazo(respuesta) and k == 0:
        return 0.0, f"FAIL — el sistema respondió 'no existe' pero debía encontrar: {claves}"
    if k == n:
        return 1.0, f"PASS — todas las claves encontradas: {claves}"
    if k > 0:
        faltantes = [c for c in claves if not _contiene_clave(respuesta, c)]
        return 0.5, f"PARCIAL — encontradas {k}/{n}. Faltan: {faltantes}"
    return 0.0, f"FAIL — ninguna clave encontrada: {claves}"


def _score_no_contiene(respuesta: str, caso: dict) -> tuple[float, str]:
    prohibidas = caso["claves"]
    encontradas = [c for c in prohibidas if _contiene_clave(respuesta, c)]

    if not encontradas and _es_rechazo(respuesta):
        return 1.0, "PASS — rechazó correctamente sin inventar datos"
    if not encontradas:
        return 0.5, "PARCIAL — no inventó pero tampoco rechazó claramente"
    return 0.0, f"FAIL — mencionó datos que no debería: {encontradas}"


def _score_corrige(respuesta: str, caso: dict) -> tuple[float, str]:
    correctas  = caso["claves"]
    prohibidas = caso.get("claves_prohibidas", [])

    confirma = any(_contiene_clave(respuesta, c) for c in prohibidas)
    corrige  = any(_contiene_clave(respuesta, c) for c in correctas)

    if corrige and not confirma:
        return 1.0, f"PASS — corrige con: {[c for c in correctas if _contiene_clave(respuesta, c)]}"
    if corrige and confirma:
        return 0.5, "PARCIAL — corrige pero también confirma el error (respuesta contradictoria)"
    if _es_rechazo(respuesta):
        return 0.5, "PARCIAL — rechazó sin corregir (no usó el documento para corregir)"
    return 0.0, f"FAIL — no corrigió el error. Claves esperadas: {correctas}"


def _score_no_alucina(respuesta: str, caso: dict) -> tuple[float, str]:
    claves_rechazo = caso["claves"]
    prohibidas     = caso.get("claves_prohibidas", [])

    alucino    = any(_contiene_clave(respuesta, c) for c in prohibidas)
    rechazo_ok = any(_contiene_clave(respuesta, c) for c in claves_rechazo) or _es_rechazo(respuesta)

    if rechazo_ok and not alucino:
        return 1.0, "PASS — no alucinó, rechazó correctamente"
    if not alucino:
        return 0.5, "PARCIAL — no alucinó pero tampoco rechazó con claridad"
    return 0.0, f"FAIL — ALUCINACIÓN detectada: mencionó {[c for c in prohibidas if _contiene_clave(respuesta, c)]}"


def evaluar_caso(respuesta: str, caso: dict) -> tuple[float, str]:
    """Dispatcher — elige la función de scoring según el tipo del caso."""
    tipo = caso["tipo"]
    if tipo == "contiene":
        return _score_contiene(respuesta, caso)
    if tipo == "no_contiene":
        return _score_no_contiene(respuesta, caso)
    if tipo == "corrige":
        return _score_corrige(respuesta, caso)
    if tipo == "no_alucina":
        return _score_no_alucina(respuesta, caso)
    return 0.0, f"tipo desconocido: {tipo}"


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS PHOENIX
# Se llama desde el backend — no hay CORS porque es server-to-server.
# ─────────────────────────────────────────────────────────────────────────────

def obtener_metricas_phoenix(t_inicio_epoch: float) -> dict:
    """
    Consulta la API REST de Phoenix (server-to-server, sin CORS) y calcula
    promedios de los spans RAG_REAL generados desde t_inicio_epoch.
    Devuelve un dict listo para ser usado en MetricasPhoenix.

    NOTA: La URL correcta en arize-phoenix >= 4.x es:
      GET /v1/projects/{project_identifier}/spans
    El project_identifier va en la URL, no como query param.
    """
    try:
        resp = requests.get(
            f"{PHOENIX_URL}/v1/projects/tesis-epn-rag/spans",
            params={"limit": 200},
            timeout=10,
        )
        if resp.status_code != 200:
            return {"disponible": False}

        spans = resp.json().get("data", [])
        if not spans:
            return {"disponible": False}

        spans_eval = []
        for span in spans:
            attrs     = span.get("attributes", {})
            start_raw = span.get("start_time", "")
            if not start_raw:
                continue
            try:
                if isinstance(start_raw, str):
                    dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                    span_epoch = dt.timestamp()
                else:
                    span_epoch = float(start_raw) / 1e9
                if span_epoch < t_inicio_epoch:
                    continue
            except Exception:
                continue

            if attrs.get("tipo") == "RAG_REAL":
                spans_eval.append(attrs)

        if not spans_eval:
            return {
                "disponible": True,
                "nota": "No se encontraron spans RAG_REAL en esta sesión (todas las respuestas vinieron del caché).",
            }

        def prom(key):
            vals = [float(s[key]) for s in spans_eval if key in s and s[key] is not None]
            return round(sum(vals) / len(vals), 1) if vals else None

        lat_total = prom("latencia_total_ms")
        lat_llm   = prom("latencia_llm_ms")
        lat_ret   = round((lat_total or 0) - (lat_llm or 0), 1) if lat_total and lat_llm else None

        return {
            "disponible":             True,
            "spans_analizados":       len(spans_eval),
            "latencia_total_ms_avg":  lat_total,
            "latencia_llm_ms_avg":    lat_llm,
            "latencia_retrieval_avg": lat_ret,
            "fragmentos_usados_avg":  prom("fragmentos_usados"),
            "k_retrieval":            spans_eval[0].get("k_retrieval"),
            "umbral_relevancia":      spans_eval[0].get("umbral_relevancia"),
            "hyde_aplicado":          spans_eval[0].get("hyde_aplicado"),
            "modelo_llm":             spans_eval[0].get("modelo_llm"),
            "modelo_embed":           spans_eval[0].get("modelo_embed"),
        }

    except Exception as e:
        return {"disponible": False, "nota": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER INTERNO — construye resumen + score global a partir de resultados
# ─────────────────────────────────────────────────────────────────────────────

def _construir_reporte(
    experimento: str,
    motor: str,
    resultados: list[dict],
    t_inicio: float,
) -> dict:
    """Calcula resumen por grupo, score global y métricas Phoenix."""
    scores_por_grupo: dict[str, list[float]] = {}
    for r in resultados:
        scores_por_grupo.setdefault(r["grupo"], []).append(r["score"])

    resumen_por_grupo: dict[str, dict] = {}
    for grupo, scores in scores_por_grupo.items():
        resumen_por_grupo[grupo] = {
            "promedio": round(sum(scores) / len(scores), 3),
            "pass":     sum(1 for s in scores if s == 1.0),
            "parcial":  sum(1 for s in scores if s == 0.5),
            "fail":     sum(1 for s in scores if s == 0.0),
            "total":    len(scores),
        }

    todos_scores = [r["score"] for r in resultados]
    score_global = round(sum(todos_scores) / len(todos_scores), 3) if todos_scores else 0.0

    conteo_global = {
        "pass":    sum(1 for r in resultados if r["veredicto"] == "PASS"),
        "parcial": sum(1 for r in resultados if r["veredicto"] == "PARCIAL"),
        "fail":    sum(1 for r in resultados if r["veredicto"] == "FAIL"),
        "total":   len(resultados),
    }

    # Esperar a que OpenTelemetry haga flush del batch de spans antes de consultar Phoenix
    time.sleep(2)
    metricas_phoenix = obtener_metricas_phoenix(t_inicio)
    duracion = round(time.time() - t_inicio, 1)

    return {
        "experimento":        experimento,
        "motor":              motor,
        "timestamp":          datetime.now().isoformat(),
        "duracion_total_seg": duracion,
        "resultados":         resultados,
        "resumen_por_grupo":  resumen_por_grupo,
        "score_global":       score_global,
        "conteo_global":      conteo_global,
        "metricas_phoenix":   metricas_phoenix,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ORQUESTADOR CLÁSICO (sin streaming — mantiene compatibilidad)
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_evaluacion(experimento: str, casos: list[dict]) -> dict:
    """
    Ejecuta la evaluación completa de forma síncrona y devuelve el reporte.
    Se mantiene por compatibilidad con el endpoint /ejecutar existente.
    """
    t_inicio      = time.time()
    motor         = obtener_motor_activo()
    casos_activos = [c for c in casos if c.get("habilitado", True)]
    resultados: list[dict] = []

    for caso in casos_activos:
        t0 = time.time()
        try:
            respuesta = consultar_base_conocimiento(caso["pregunta"], motor=motor)
        except Exception as e:
            respuesta = f"Error interno al consultar el RAG: {e}"
        latencia_ms = round((time.time() - t0) * 1000)

        score, detalle = evaluar_caso(respuesta, caso)
        veredicto = "PASS" if score == 1.0 else ("PARCIAL" if score == 0.5 else "FAIL")

        resultados.append({
            "id":          caso["id"],
            "grupo":       caso.get("grupo", "Sin grupo"),
            "tipo":        caso["tipo"],
            "pregunta":    caso["pregunta"],
            "respuesta":   respuesta,
            "latencia_ms": latencia_ms,
            "score":       score,
            "veredicto":   veredicto,
            "detalle":     detalle,
            "descripcion": caso.get("descripcion"),
        })

    return _construir_reporte(experimento, motor, resultados, t_inicio)


# ─────────────────────────────────────────────────────────────────────────────
# ORQUESTADOR CON STREAMING — genera eventos de progreso caso a caso
# ─────────────────────────────────────────────────────────────────────────────

def ejecutar_evaluacion_stream(
    experimento: str,
    casos: list[dict],
) -> Generator[dict, None, None]:
    """
    Generador que procesa los casos uno a uno y yielda dicts con el progreso.

    Formato de cada evento yieldeado:
      {"tipo": "progreso",   "caso_actual": N, "total_casos": T,
       "porcentaje": P, "resultado": {...}}          ← tras cada caso
      {"tipo": "completado", "caso_actual": T, "total_casos": T,
       "porcentaje": 100, "reporte_final": {...}}    ← al terminar
      {"tipo": "error",      "mensaje_error": "..."}  ← si hay excepción fatal

    El frontend consume estos eventos vía Server-Sent Events (SSE).
    La comunicación con Phoenix se hace server-to-server desde este servicio,
    por lo que NO hay restricciones CORS.
    """
    t_inicio      = time.time()
    motor         = obtener_motor_activo()
    casos_activos = [c for c in casos if c.get("habilitado", True)]
    total         = len(casos_activos)
    resultados: list[dict] = []

    for idx, caso in enumerate(casos_activos, start=1):
        t0 = time.time()
        try:
            respuesta = consultar_base_conocimiento(caso["pregunta"], motor=motor)
        except Exception as e:
            respuesta = f"Error interno al consultar el RAG: {e}"
        latencia_ms = round((time.time() - t0) * 1000)

        score, detalle = evaluar_caso(respuesta, caso)
        veredicto = "PASS" if score == 1.0 else ("PARCIAL" if score == 0.5 else "FAIL")

        resultado_caso = {
            "id":          caso["id"],
            "grupo":       caso.get("grupo", "Sin grupo"),
            "tipo":        caso["tipo"],
            "pregunta":    caso["pregunta"],
            "respuesta":   respuesta,
            "latencia_ms": latencia_ms,
            "score":       score,
            "veredicto":   veredicto,
            "detalle":     detalle,
            "descripcion": caso.get("descripcion"),
        }
        resultados.append(resultado_caso)

        porcentaje = round(idx / total * 100)

        yield {
            "tipo":        "progreso",
            "caso_actual": idx,
            "total_casos": total,
            "porcentaje":  porcentaje,
            "resultado":   resultado_caso,
        }

    # Todos los casos terminaron — construir reporte final y emitirlo
    reporte = _construir_reporte(experimento, motor, resultados, t_inicio)

    yield {
        "tipo":         "completado",
        "caso_actual":  total,
        "total_casos":  total,
        "porcentaje":   100,
        "reporte_final": reporte,
    }