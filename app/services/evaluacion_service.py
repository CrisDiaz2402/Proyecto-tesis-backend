# app/services/evaluacion_service.py
import re
import time
import traceback
import numpy as np
from datetime import datetime
from typing import Generator

from app.services.rag_service import consultar_base_conocimiento
from app.services.config_service import obtener_motor_activo


_FRASES_NEGATIVAS_EVAL = [
    "no encontré", "no hay información", "no tengo datos",
    "no existe", "no se encuentra",
]


def _score_semantico(
    respuesta: str,
    respuesta_esperada: str,
    umbral: float = 0.80,
) -> tuple[float, str]:
    try:
        # ── Regla 1: respuestas negativas equivalentes ───────────────────────
        r_lower  = respuesta.strip().lower()
        e_lower  = respuesta_esperada.strip().lower()
        es_neg_r = any(f in r_lower for f in _FRASES_NEGATIVAS_EVAL)
        es_neg_e = any(f in e_lower for f in _FRASES_NEGATIVAS_EVAL)
        if es_neg_r and es_neg_e:
            return 1.0, "PASS — ambas respuestas indican ausencia de información"

        from app.core.singletons import EmbedModelSingleton
        model = EmbedModelSingleton().model

        emb_resp = model.encode(respuesta.strip(), normalize_embeddings=True)
        emb_esp  = model.encode(respuesta_esperada.strip(), normalize_embeddings=True)

        similitud = round(float(np.dot(emb_resp, emb_esp)), 4)

        # ── Regla 2: bonus numérico ──────────────────────────────────────────
        nums_r = set(re.findall(r'\b\d+\b', respuesta))
        nums_e = set(re.findall(r'\b\d+\b', respuesta_esperada))
        if nums_r and nums_r == nums_e:
            similitud = min(1.0, round(similitud + 0.15, 4))

        umbral_parcial = round(umbral * 0.75, 3)

        if similitud >= umbral:
            return 1.0, f"PASS — similitud: {similitud} ≥ umbral {umbral}"
        elif similitud >= umbral_parcial:
            return 0.5, f"PARCIAL — similitud: {similitud} (entre {umbral_parcial} y {umbral})"
        else:
            return 0.0, f"FAIL — similitud: {similitud} < {umbral_parcial}"

    except Exception as e:
        return 0.0, f"ERROR en scoring semántico: {e}"


def _construir_reporte(
    experimento: str,
    motor: str,
    resultados: list[dict],
    t_inicio: float,
) -> dict:
    scores_por_grupo: dict[str, list[float]] = {}
    similitudes_por_grupo: dict[str, list[float]] = {}

    for r in resultados:
        scores_por_grupo.setdefault(r["grupo"], []).append(r["score"])
        m = re.search(r"similitud:\s*([\d.]+)", r.get("detalle", ""))
        if m:
            similitudes_por_grupo.setdefault(r["grupo"], []).append(float(m.group(1)))

    resumen_por_grupo: dict[str, dict] = {}
    for grupo, scores in scores_por_grupo.items():
        sims = similitudes_por_grupo.get(grupo, [])
        resumen_por_grupo[grupo] = {
            "promedio":           round(sum(scores) / len(scores), 3),
            "similitud_promedio": round(sum(sims) / len(sims), 3) if sims else 0.0,
            "pass":               sum(1 for s in scores if s == 1.0),
            "parcial":            sum(1 for s in scores if s == 0.5),
            "fail":               sum(1 for s in scores if s == 0.0),
            "total":              len(scores),
        }

    todos_scores = [r["score"] for r in resultados]
    score_global = round(sum(todos_scores) / len(todos_scores), 3) if todos_scores else 0.0

    todas_sims = [v for vals in similitudes_por_grupo.values() for v in vals]
    similitud_promedio_global = round(sum(todas_sims) / len(todas_sims), 3) if todas_sims else 0.0

    latencias = [r["latencia_ms"] for r in resultados]
    latencia_promedio_ms = round(sum(latencias) / len(latencias)) if latencias else 0

    conteo_global = {
        "pass":    sum(1 for r in resultados if r["veredicto"] == "PASS"),
        "parcial": sum(1 for r in resultados if r["veredicto"] == "PARCIAL"),
        "fail":    sum(1 for r in resultados if r["veredicto"] == "FAIL"),
        "total":   len(resultados),
    }

    duracion = round(time.time() - t_inicio, 1)

    return {
        "experimento":          experimento,
        "motor":                motor,
        "timestamp":            datetime.now().isoformat(),
        "duracion_total_seg":   duracion,
        "resultados":           resultados,
        "resumen_por_grupo":    resumen_por_grupo,
        "score_global":         score_global,
        "similitud_promedio":   similitud_promedio_global,
        "latencia_promedio_ms": latencia_promedio_ms,
        "conteo_global":        conteo_global,
    }


def ejecutar_evaluacion(experimento: str, casos: list[dict]) -> dict:
    t_inicio      = time.time()
    motor         = obtener_motor_activo()
    casos_activos = [c for c in casos if c.get("habilitado", True)]
    resultados: list[dict] = []

    for caso in casos_activos:
        t0 = time.time()
        try:
            respuesta = consultar_base_conocimiento(caso["pregunta"], motor=motor)
        except Exception as e:
            respuesta = f"Error interno al consultar el sistema: {e}"
        latencia_ms = round((time.time() - t0) * 1000)

        umbral = caso.get("umbral_similitud", 0.80)
        score, detalle = _score_semantico(respuesta, caso["respuesta_esperada"], umbral)
        veredicto = "PASS" if score == 1.0 else ("PARCIAL" if score == 0.5 else "FAIL")

        resultados.append({
            "id":          caso["id"],
            "grupo":       caso.get("grupo", "General"),
            "tipo":        "semantico",
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
            respuesta = f"Error interno al consultar el sistema: {e}"
        latencia_ms = round((time.time() - t0) * 1000)

        time.sleep(0.3)  # A) delay para evitar saturación de buffers en vLLM bajo --enforce-eager

        umbral = caso.get("umbral_similitud", 0.80)

        # B) Detectar respuestas inválidas o truncadas antes del scoring
        if (
            not respuesta
            or len(respuesta.strip()) < 3
            or respuesta.startswith("Error interno al consultar")
        ):
            print(f"[EVAL_STREAM] ⚠️ Caso id={caso['id']} — respuesta inválida, saltando scoring.")
            score, detalle, veredicto = 0.0, "Respuesta inválida o truncada del LLM", "ERROR"
        else:
            # C) _score_semantico con su propio try/except independiente
            try:
                score, detalle = _score_semantico(respuesta, caso["respuesta_esperada"], umbral)
                veredicto = "PASS" if score == 1.0 else ("PARCIAL" if score == 0.5 else "FAIL")
            except Exception as e:
                print(
                    f"[EVAL_STREAM] ❌ Fallo en scoring (id={caso['id']}):\n{traceback.format_exc()}"
                )
                score, detalle, veredicto = 0.0, f"Fallo en scoring: {str(e)}", "ERROR"

        resultado_caso = {
            "id":          caso["id"],
            "grupo":       caso.get("grupo", "General"),
            "tipo":        "semantico",
            "pregunta":    caso["pregunta"],
            "respuesta":   respuesta,
            "latencia_ms": latencia_ms,
            "score":       score,
            "veredicto":   veredicto,
            "detalle":     detalle,
            "descripcion": caso.get("descripcion"),
        }
        resultados.append(resultado_caso)

        yield {
            "tipo":        "progreso",
            "caso_actual": idx,
            "total_casos": total,
            "porcentaje":  round(idx / total * 100),
            "resultado":   resultado_caso,
        }

    reporte = _construir_reporte(experimento, motor, resultados, t_inicio)
    yield {
        "tipo":          "completado",
        "caso_actual":   total,
        "total_casos":   total,
        "porcentaje":    100,
        "reporte_final": reporte,
    }
