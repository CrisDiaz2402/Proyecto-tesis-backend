# app/services/evaluacion_service.py
"""
Servicio de evaluación RAG.

Contiene toda la lógica de scoring:
  - Normalización de texto y matching flexible de claves
  - Scoring por tipo: contiene / no_contiene / corrige / no_alucina
  - Orquestación completa de una sesión de evaluación
  - Generador de progreso para streaming SSE (ejecutar_evaluacion_stream)
"""

import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Generator

from app.services.rag_service import consultar_base_conocimiento
from app.services.config_service import obtener_motor_activo


def _normalizar(texto: str) -> str:
    """Lowercase y sin tildes para comparación flexible."""
    texto = texto.lower()
    for src, dst in {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ñ":"n","ü":"u"}.items():
        texto = texto.replace(src, dst)
    return texto


def _contiene_clave(respuesta: str, clave: str) -> bool:
    """True si 'clave' aparece en 'respuesta' de forma flexible."""
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
# MÉTRICAS DE EVALUACIÓN
def _construir_reporte(
    experimento: str,
    motor: str,
    resultados: list[dict],
    t_inicio: float,
) -> dict:
    """Calcula resumen por grupo y score global."""
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
    }


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


def ejecutar_evaluacion_stream(
    experimento: str,
    casos: list[dict],
) -> Generator[dict, None, None]:
    """Generador que procesa los casos uno a uno y yielda dicts con el progreso."""
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