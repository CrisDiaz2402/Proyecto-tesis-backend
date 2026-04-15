# scripts/evaluar_local.py
"""
Evaluador de parámetros RAG.

Estrategia de scoring:
  - "contiene"      : la respuesta debe incluir todas las claves del ground truth
  - "no_contiene"   : la respuesta NO debe mencionar ninguna clave
  - "corrige"       : la respuesta debe corregir el dato falso de la pregunta
  - "no_alucina"    : la respuesta debe decir "no existe" o similar, nunca inventar

  Scoring: PASS (1.0) / PARCIAL (0.5) / FAIL (0.0)

Salida: resultados_YYYY-MM-DD_HH-MM.csv
"""

import sys
import os
import time
import csv
import json
import re
import requests
from datetime import datetime
from difflib import SequenceMatcher
import pandas as pd

FASTAPI_URL  = "http://localhost:8000/api/chat/consultar"
MOTOR        = "local:local"
TIMEOUT_SEG  = 180
EXPERIMENTO  = "baseline"

# ─────────────────────────────────────────────────────────────────────────────
# SET DE CASOS DE EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────
# Campos de cada caso:
#   id          : identificador único (se usa en el CSV)
#   grupo       : categoría del caso (para agrupar en el resumen)
#   tipo        : "contiene" | "no_contiene" | "corrige" | "no_alucina"
#   pregunta    : texto enviado al chatbot
#   claves      : lista de strings que deben aparecer (o no) en la respuesta
#   descripcion : qué se está probando (para el reporte)
#
# ── NOTA SOBRE EL MÉTODO DE MATCHING ─────────────────────────────────────────
# No se compara exacto: se usa similitud de subcadena normalizada.
# "135 créditos" matchea contra "135", "ciento treinta y cinco", "135 cr".
# Ver función _contiene_clave() abajo.

CASOS = [

    # ── GRUPO A: Verdaderos positivos directos ────────────────────────────────
    # Miden si el retrieval trae el fragmento correcto para datos explícitos.
    # Si fallan → problema de retrieval (umbral, k, chunking).
    {
        "id": "A1", "grupo": "TP Directo",
        "tipo": "contiene",
        "pregunta": "¿Cuántos créditos en total necesita un estudiante para graduarse?",
        "claves": ["135"],
        "descripcion": "Dato numérico puntual — créditos de graduación",
    },
    {
        "id": "A2", "grupo": "TP Directo",
        "tipo": "contiene",
        "pregunta": "¿Cuáles son los prerrequisitos para tomar Inteligencia Artificial?",
        "claves": ["ICCD412", "ICCD442"],
        "descripcion": "Dato estructurado — códigos de prerrequisitos",
    },
    {
        "id": "A3", "grupo": "TP Directo",
        "tipo": "contiene",
        "pregunta": "¿Cuántas horas de Prácticas Laborales debe completar un estudiante?",
        "claves": ["240"],
        "descripcion": "Dato numérico puntual — horas de práctica",
    },
    {
        "id": "A4", "grupo": "TP Directo",
        "tipo": "contiene",
        "pregunta": "¿En qué semestre se dicta Cloud Computing y cuántos créditos tiene?",
        "claves": ["8", "3"],
        "descripcion": "Dato doble — semestre y créditos de una materia",
    },
    {
        "id": "A5", "grupo": "TP Directo",
        "tipo": "contiene",
        "pregunta": "¿Cuántas asignaturas en total tiene la carrera?",
        "claves": ["48"],
        "descripcion": "Conteo total de asignaturas",
    },

    # ── GRUPO B: Verdaderos positivos con razonamiento ────────────────────────
    # Requieren combinar información de múltiples fragmentos del documento.
    # Si fallan → problema de k (pocos fragmentos) o LLM débil en síntesis.
    {
        "id": "B1", "grupo": "TP Razonamiento",
        "tipo": "contiene",
        "pregunta": "¿Cuántas horas suman en total las prácticas laborales y el servicio comunitario?",
        "claves": ["336"],
        "descripcion": "Suma de dos datos en fragmentos distintos (240 + 96)",
    },
    {
        "id": "B2", "grupo": "TP Razonamiento",
        "tipo": "contiene",
        "pregunta": "¿Cuál es el total de horas de toda la carrera incluyendo prácticas e integración curricular?",
        "claves": ["6480"],
        "descripcion": "Dato de resumen que puede estar en sección de totales",
    },
    {
        "id": "B3", "grupo": "TP Razonamiento",
        "tipo": "contiene",
        "pregunta": "¿Cuántos requisitos adicionales obligatorios fuera de la malla necesita un estudiante para graduarse?",
        "claves": ["7"],
        "descripcion": "Conteo de ítems en lista de requisitos",
    },

    # ── GRUPO C: Verdaderos negativos — fuera del dominio ────────────────────
    # El sistema DEBE responder que no tiene información.
    # Si fallan (inventan respuesta) → alucinación grave / umbral de relevancia bajo.
    {
        "id": "C1", "grupo": "TN Fuera dominio",
        "tipo": "no_contiene",
        "pregunta": "¿Cuál es el costo por crédito de la carrera?",
        "claves": ["dólares", "USD", "$", "costo", "precio", "valor", "pesos"],
        "descripcion": "Fuera del dominio — no debe inventar precios",
    },
    {
        "id": "C2", "grupo": "TN Fuera dominio",
        "tipo": "no_contiene",
        "pregunta": "¿Cuántos estudiantes se gradúan por año en esta carrera?",
        "claves": ["estudiantes se gradúan", "graduados por año", "promedio de"],
        "descripcion": "Fuera del dominio — estadística poblacional no disponible",
    },
    {
        "id": "C3", "grupo": "TN Fuera dominio",
        "tipo": "no_contiene",
        "pregunta": "¿Quién es el decano actual de la Facultad de Ingeniería de Sistemas?",
        "claves": ["decano", "Dr.", "Ing.", "director", "PhD"],
        "descripcion": "Fuera del dominio — nombre de persona no está en el documento",
    },

    # ── GRUPO D: Trampas de sugestión (dato falso en la pregunta) ────────────
    # El sistema DEBE corregir el dato incorrecto. No basta con responder.
    # Si fallan (confirman el error) → sycophancy / el LLM no confronta al usuario.
    {
        "id": "D1", "grupo": "Corrección",
        "tipo": "corrige",
        "pregunta": "Tengo entendido que la carrera dura 10 semestres, ¿es correcto?",
        "claves": ["9"],                    # debe mencionar "9 semestres"
        "claves_prohibidas": ["10 semestres es correcto", "sí, dura 10", "en efecto"],
        "descripcion": "Corrección de dato falso — duración real es 9 semestres",
    },
    {
        "id": "D2", "grupo": "Corrección",
        "tipo": "corrige",
        "pregunta": "¿Es verdad que Cloud Computing tiene como prerrequisito Redes de Computadores II?",
        "claves": ["no", "sin prerrequisito", "no tiene prerrequisito", "no requiere"],
        "claves_prohibidas": ["sí", "correcto", "en efecto", "Redes de Computadores II es prerrequisito"],
        "descripcion": "Corrección de prerrequisito inventado — Cloud Computing no tiene prerrequisitos",
    },
    {
        "id": "D3", "grupo": "Corrección",
        "tipo": "corrige",
        "pregunta": "¿Cuántos créditos tiene Programación I, que si mal no recuerdo son 3?",
        "claves": ["4"],
        "claves_prohibidas": ["3 créditos es correcto", "sí, son 3", "tienes razón"],
        "descripcion": "Corrección de dato numérico — Programación I tiene 4 créditos no 3",
    },
    {
        "id": "D4", "grupo": "Corrección",
        "tipo": "corrige",
        "pregunta": "¿Las prácticas laborales se realizan obligatoriamente en el noveno semestre?",
        "claves": ["no", "cualquier", "acumular"],
        "claves_prohibidas": ["sí, en el noveno", "correcto, en noveno"],
        "descripcion": "Aclaración de mito — las prácticas se acumulan en cualquier semestre",
    },

    # ── GRUPO E: Trampas de alucinación — entidades inexistentes ─────────────
    # El sistema NUNCA debe dar datos sobre algo que no existe en el documento.
    # Si fallan → alucinación directa, el LLM usa conocimiento de preentrenamiento.
    {
        "id": "E1", "grupo": "Anti-alucinación",
        "tipo": "no_alucina",
        "pregunta": "¿Qué materia de la carrera trata sobre Blockchain y en qué semestre se dicta?",
        "claves_prohibidas": ["semestre", "nivel", "créditos", "ICCD"],
        "claves": ["no existe", "no hay", "no se dicta", "no está", "no consta"],
        "descripcion": "No existe materia de Blockchain — cualquier semestre dado es alucinación",
    },
    {
        "id": "E2", "grupo": "Anti-alucinación",
        "tipo": "no_alucina",
        "pregunta": "¿Cuál es el código de la materia Programación III?",
        "claves_prohibidas": ["ICCD", "código"],
        "claves": ["no existe", "no hay", "no consta", "no se encuentra"],
        "descripcion": "No existe Programación III — cualquier código dado es alucinación",
    },
    {
        "id": "E3", "grupo": "Anti-alucinación",
        "tipo": "no_alucina",
        "pregunta": "¿Qué opciones de especialización o menciones ofrece la carrera?",
        "claves_prohibidas": ["mención en", "especialización en", "opción de"],
        "claves": ["no existe", "no hay", "no se menciona", "no consta", "no están"],
        "descripcion": "El documento no menciona especializaciones — inventarlas es alucinación",
    },

    # ── GRUPO F: Interpretación y límite ─────────────────────────────────────
    # Preguntas ambiguas donde el sistema debe hacer lo mejor posible.
    # Miden si el sistema da respuestas útiles en zonas grises.
    {
        "id": "F1", "grupo": "Interpretación",
        "tipo": "contiene",
        "pregunta": "¿Qué materias puedo tomar sin ningún prerrequisito desde el inicio?",
        "claves": ["Álgebra", "Cálculo", "Programación I", "Comunicación"],
        "descripcion": "Lista parcial aceptable — al menos las materias del nivel 1",
    },
    {
        "id": "F2", "grupo": "Interpretación",
        "tipo": "contiene",
        "pregunta": "¿Cuántas horas de aprendizaje autónomo exige la carrera en total?",
        "claves": ["2800"],
        "descripcion": "Dato en sección de resumen de horas — distinción de tipo de hora",
    },
]

TOTAL = len(CASOS)

# ─────────────────────────────────────────────────────────────────────────────
# COLORES PARA CONSOLA (sin dependencias externas)
# ─────────────────────────────────────────────────────────────────────────────
C_OK    = "\033[92m"   # verde
C_WARN  = "\033[93m"   # amarillo
C_FAIL  = "\033[91m"   # rojo
C_BOLD  = "\033[1m"
C_RESET = "\033[0m"

def ok(s):   return f"{C_OK}{s}{C_RESET}"
def warn(s): return f"{C_WARN}{s}{C_RESET}"
def fail(s): return f"{C_FAIL}{s}{C_RESET}"
def bold(s): return f"{C_BOLD}{s}{C_RESET}"

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES DE MATCHING
# No se compara exacto: se normaliza el texto y se busca subcadena.
# Umbral de similitud: 0.80 (SequenceMatcher) como fallback si no hay subcadena exacta.
# ─────────────────────────────────────────────────────────────────────────────

def _normalizar(texto: str) -> str:
    """Lowercase, sin tildes, sin puntuación extra."""
    texto = texto.lower()
    reemplazos = {"á":"a","é":"e","í":"i","ó":"o","ú":"u","ñ":"n","ü":"u"}
    for k, v in reemplazos.items():
        texto = texto.replace(k, v)
    return texto

def _contiene_clave(respuesta: str, clave: str) -> bool:
    """
    Devuelve True si 'clave' aparece en 'respuesta' de forma flexible:
      1. Subcadena exacta (normalizada)
      2. Subcadena numérica: "135" matchea "135 créditos", "135.", "(135)"
      3. Similitud de secuencia ≥ 0.85 como último recurso
    """
    r = _normalizar(respuesta)
    c = _normalizar(clave)

    # Coincidencia directa de subcadena
    if c in r:
        return True

    # Para números: buscar el número rodeado de no-dígitos
    if clave.isdigit():
        patron = r"(?<!\d)" + re.escape(clave) + r"(?!\d)"
        if re.search(patron, respuesta):
            return True

    # Similitud de secuencia como fallback para frases
    if len(c) > 5:
        ratio = SequenceMatcher(None, c, r).ratio()
        if ratio >= 0.85:
            return True

    return False

def _es_rechazo(respuesta: str) -> bool:
    """True si la respuesta indica que no tiene información."""
    r = _normalizar(respuesta)
    frases_rechazo = [
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
    return any(f in r for f in frases_rechazo)

# ─────────────────────────────────────────────────────────────────────────────
# LÓGICA DE SCORING POR TIPO
# ─────────────────────────────────────────────────────────────────────────────

def _score_contiene(respuesta: str, caso: dict) -> tuple[float, str]:
    """
    PASS si todas las claves están presentes.
    PARCIAL si al menos una clave está presente.
    FAIL si ninguna clave está presente (incluye si responde 'no existe').
    """
    claves = caso["claves"]
    encontradas = [c for c in claves if _contiene_clave(respuesta, c)]
    n = len(claves)
    k = len(encontradas)

    if _es_rechazo(respuesta) and k == 0:
        return 0.0, f"FAIL — el sistema respondió 'no existe' pero debía encontrar: {claves}"

    if k == n:
        return 1.0, f"PASS — todas las claves encontradas: {claves}"
    elif k > 0:
        faltantes = [c for c in claves if not _contiene_clave(respuesta, c)]
        return 0.5, f"PARCIAL — encontradas {k}/{n}. Faltan: {faltantes}"
    else:
        return 0.0, f"FAIL — ninguna clave encontrada: {claves}"

def _score_no_contiene(respuesta: str, caso: dict) -> tuple[float, str]:
    """
    PASS si la respuesta es un rechazo y NO menciona ninguna clave prohibida.
    PARCIAL si rechaza pero con alguna mención sospechosa.
    FAIL si la respuesta incluye alguna clave prohibida (alucinó o inventó datos).
    """
    claves_prohibidas = caso["claves"]
    encontradas_prohibidas = [c for c in claves_prohibidas if _contiene_clave(respuesta, c)]

    if not encontradas_prohibidas and _es_rechazo(respuesta):
        return 1.0, "PASS — rechazó correctamente sin inventar datos"
    elif not encontradas_prohibidas and not _es_rechazo(respuesta):
        return 0.5, "PARCIAL — no inventó pero tampoco rechazó claramente"
    else:
        return 0.0, f"FAIL — mencionó datos que no debería: {encontradas_prohibidas}"

def _score_corrige(respuesta: str, caso: dict) -> tuple[float, str]:
    """
    PASS si contiene la corrección correcta y no confirma el error.
    PARCIAL si corrige pero de forma ambigua.
    FAIL si confirma el dato falso o no corrige.
    """
    claves_correctas  = caso["claves"]
    claves_prohibidas = caso.get("claves_prohibidas", [])

    confirma_error = any(_contiene_clave(respuesta, c) for c in claves_prohibidas)
    corrige        = any(_contiene_clave(respuesta, c) for c in claves_correctas)

    if corrige and not confirma_error:
        return 1.0, f"PASS — corrige con: {[c for c in claves_correctas if _contiene_clave(respuesta, c)]}"
    elif corrige and confirma_error:
        return 0.5, "PARCIAL — corrige pero también confirma el error (respuesta contradictoria)"
    elif _es_rechazo(respuesta):
        return 0.5, "PARCIAL — rechazó sin corregir (no usó el documento para corregir)"
    else:
        return 0.0, f"FAIL — no corrigió el error. Claves esperadas: {claves_correctas}"

def _score_no_alucina(respuesta: str, caso: dict) -> tuple[float, str]:
    """
    PASS si rechaza correctamente y no menciona ninguna clave prohibida.
    PARCIAL si rechaza pero de forma incompleta o especulativa.
    FAIL si da datos concretos inventados (menciona claves prohibidas).
    """
    claves_rechazo    = caso["claves"]
    claves_prohibidas = caso.get("claves_prohibidas", [])

    alucino     = any(_contiene_clave(respuesta, c) for c in claves_prohibidas)
    rechazo_ok  = any(_contiene_clave(respuesta, c) for c in claves_rechazo) or _es_rechazo(respuesta)

    if rechazo_ok and not alucino:
        return 1.0, "PASS — no alucinó, rechazó correctamente"
    elif not alucino and not rechazo_ok:
        return 0.5, "PARCIAL — no alucinó pero tampoco rechazó con claridad"
    else:
        return 0.0, f"FAIL — ALUCINACIÓN detectada: mencionó {[c for c in claves_prohibidas if _contiene_clave(respuesta, c)]}"

def evaluar_caso(respuesta: str, caso: dict) -> tuple[float, str]:
    tipo = caso["tipo"]
    if tipo == "contiene":
        return _score_contiene(respuesta, caso)
    elif tipo == "no_contiene":
        return _score_no_contiene(respuesta, caso)
    elif tipo == "corrige":
        return _score_corrige(respuesta, caso)
    elif tipo == "no_alucina":
        return _score_no_alucina(respuesta, caso)
    return 0.0, f"tipo desconocido: {tipo}"

# ─────────────────────────────────────────────────────────────────────────────
# LLAMADA A LA API
# ─────────────────────────────────────────────────────────────────────────────

def preguntar_api(pregunta: str) -> tuple[str, float]:
    """Llama al endpoint de chat. Devuelve (respuesta, latencia_ms)."""
    t0 = time.time()
    try:
        resp = requests.post(
            FASTAPI_URL,
            json={"pregunta": pregunta},
            headers={"Authorization": f"Bearer {JWT_TOKEN}"},
            timeout=TIMEOUT_SEG,
        )
        latencia = round((time.time() - t0) * 1000)
        if resp.status_code == 200:
            return resp.json().get("respuesta", "Sin respuesta"), latencia
        return f"Error HTTP {resp.status_code}: {resp.text[:200]}", latencia
    except requests.exceptions.Timeout:
        return f"Timeout después de {TIMEOUT_SEG}s", round((time.time() - t0) * 1000)
    except Exception as e:
        return f"Error de conexión: {e}", round((time.time() - t0) * 1000)

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE MÉTRICAS PHOENIX
# Phoenix expone una API REST en /v1/spans para consultar los traces.
# Se filtran los spans del proyecto "tesis-epn-rag" generados en esta sesión.
# ─────────────────────────────────────────────────────────────────────────────

# ─── FUNCIÓN ELIMINADA ─────────────────────────────────────────────────────────────────
# TODO: ELIMINAR - Phoenix ya no se usa en el stack
def obtener_metricas_phoenix(t_inicio_epoch: float) -> dict:
    """OBSOLETO: Phoenix fue eliminado del stack. Retorna None siempre."""
    return None
        spans_eval = []
        for span in spans:
            attrs = span.get("attributes", {})
            # Convertir start_time a epoch si viene como ISO string
            start_raw = span.get("start_time", "")
            if not start_raw:
                continue
            try:
                from datetime import timezone
                if isinstance(start_raw, str):
                    dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                    span_epoch = dt.timestamp()
                else:
                    span_epoch = float(start_raw) / 1e9  # nanosegundos a segundos
                if span_epoch < t_inicio_epoch:
                    continue
            except Exception:
                continue

            if attrs.get("tipo") == "RAG_REAL":
                spans_eval.append(attrs)

        if not spans_eval:
            return {"nota": "No se encontraron spans RAG_REAL en esta sesión"}

        # Calcular promedios
        def prom(key):
            vals = [float(s[key]) for s in spans_eval if key in s and s[key] is not None]
            return round(sum(vals) / len(vals), 1) if vals else None

        return {
            "spans_analizados":      len(spans_eval),
            "latencia_total_ms_avg": prom("latencia_total_ms"),
            "latencia_llm_ms_avg":   prom("latencia_llm_ms"),
            "latencia_hyde_ms_avg":  round(
                (prom("latencia_total_ms") or 0) - (prom("latencia_llm_ms") or 0), 1
            ),
            "fragmentos_usados_avg": prom("fragmentos_usados"),
            "k_retrieval":           spans_eval[0].get("k_retrieval"),
            "umbral_relevancia":     spans_eval[0].get("umbral_relevancia"),
            "num_tokens":            spans_eval[0].get("num_tokens"),
            "hyde_aplicado":         spans_eval[0].get("hyde_aplicado"),
            "modelo_llm":            spans_eval[0].get("modelo_llm"),
            "modelo_embed":          spans_eval[0].get("modelo_embed"),
        }

    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# AUTENTICACIÓN — obtener JWT antes de las consultas
# ─────────────────────────────────────────────────────────────────────────────

def obtener_jwt(username: str = "admin", password: str = "admin123") -> str:
    """Obtiene el JWT del backend."""
    try:
            resp = requests.post(
                "http://localhost:8000/api/auth/login",
                json={"username": username, "password": password},
                timeout=10,
            )
            if resp.status_code == 200:
                token = resp.json().get("access_token", "")
                return token

def exportar_csv(resultados: list, nombre_archivo: str) -> None:
    campos = [
        "experimento", "motor", "id", "grupo", "tipo",
        "pregunta", "respuesta", "latencia_ms",
        "score", "veredicto", "detalle",
        "descripcion", "timestamp",
    ]
    with open(nombre_archivo, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for r in resultados:
            writer.writerow({k: r.get(k, "") for k in campos})
    print(ok(f"\n[CSV] Resultados exportados a: {nombre_archivo}"))

def main():
    global JWT_TOKEN

    ts_inicio   = datetime.now()
    epoch_inicio = time.time()

    print(bold("\n" + "═" * 70))
    print(bold(f"  EVALUADOR RAG — {ts_inicio.strftime('%Y-%m-%d %H:%M:%S')}"))
    print(bold("═" * 70))
    print(f"  Motor      : {bold(MOTOR)}")
    print(f"  Experimento: {bold(EXPERIMENTO)}")
    print(f"  Preguntas  : {TOTAL}")
    print(f"  FastAPI    : {FASTAPI_URL}")
    print(bold("═" * 70) + "\n")

    # Auth
    JWT_TOKEN = obtener_jwt(username="admin", password="admin")

    resultados = []
    scores_por_grupo: dict[str, list[float]] = {}

    # ── Fase 1: Consultar y evaluar cada caso ─────────────────────────────────
    for i, caso in enumerate(CASOS, 1):
        grupo  = caso["grupo"]
        prefijo = f"[{i:02d}/{TOTAL}] {caso['id']} · {grupo}"
        print(f"\n{bold(prefijo)}")
        print(f"  Pregunta : {caso['pregunta']}")
        print(f"  Midiendo : {caso['descripcion']}")

        respuesta, latencia_ms = preguntar_api(caso["pregunta"])

        origen = ok("⚡ caché") if latencia_ms < 500 else f"🤖 {latencia_ms}ms"
        print(f"  Latencia : {origen}")
        print(f"  Respuesta: {respuesta[:200]}{'...' if len(respuesta) > 200 else ''}")

        score, detalle = evaluar_caso(respuesta, caso)

        if score == 1.0:
            veredicto_str = ok("PASS ✓")
            veredicto     = "PASS"
        elif score == 0.5:
            veredicto_str = warn("PARCIAL ◑")
            veredicto     = "PARCIAL"
        else:
            veredicto_str = fail("FAIL ✗")
            veredicto     = "FAIL"

        print(f"  Score    : {veredicto_str}  ({score:.1f})")
        print(f"  Detalle  : {detalle}")

        scores_por_grupo.setdefault(grupo, []).append(score)

        resultados.append({
            "experimento": EXPERIMENTO,
            "motor":       MOTOR,
            "id":          caso["id"],
            "grupo":       grupo,
            "tipo":        caso["tipo"],
            "pregunta":    caso["pregunta"],
            "respuesta":   respuesta,
            "latencia_ms": latencia_ms,
            "score":       score,
            "veredicto":   veredicto,
            "detalle":     detalle,
            "descripcion": caso["descripcion"],
            "timestamp":   ts_inicio.isoformat(),
        })

    # ── Fase 2: Resumen por grupo ──────────────────────────────────────────────
    print("\n" + bold("═" * 70))
    print(bold("  RESUMEN POR GRUPO"))
    print(bold("═" * 70))

    score_total_sum = 0.0
    score_total_n   = 0

    for grupo, scores in scores_por_grupo.items():
        promedio = sum(scores) / len(scores)
        pct      = round(promedio * 100)
        passes   = sum(1 for s in scores if s == 1.0)
        parciales= sum(1 for s in scores if s == 0.5)
        fails    = sum(1 for s in scores if s == 0.0)
        barra    = ok("█") * passes + warn("▒") * parciales + fail("░") * fails

        if promedio >= 0.9:   color_fn = ok
        elif promedio >= 0.6: color_fn = warn
        else:                 color_fn = fail

        print(f"  {grupo:<22} {barra}  {color_fn(f'{pct:3d}%')}  "
              f"({passes}✓ {parciales}◑ {fails}✗ de {len(scores)})")

        score_total_sum += sum(scores)
        score_total_n   += len(scores)

    score_global = score_total_sum / score_total_n if score_total_n else 0
    pct_global   = round(score_global * 100)

    print(bold("─" * 70))
    if score_global >= 0.85:
        veredicto_global = ok(f"EXCELENTE — {pct_global}%")
    elif score_global >= 0.65:
        veredicto_global = warn(f"ACEPTABLE — {pct_global}%")
    else:
        veredicto_global = fail(f"REQUIERE MEJORAS — {pct_global}%")

    print(bold(f"  SCORE GLOBAL: {veredicto_global}"))
    print(f"  ({score_total_n} preguntas · PASS={sum(1 for r in resultados if r['veredicto']=='PASS')} "
          f"PARCIAL={sum(1 for r in resultados if r['veredicto']=='PARCIAL')} "
          f"FAIL={sum(1 for r in resultados if r['veredicto']=='FAIL')})")

    # ── Fase 3: Métricas Phoenix ───────────────────────────────────────────────
    print("\n" + bold("═" * 70))
    print(bold("  MÉTRICAS PHOENIX (parámetros del sistema)"))
    print(bold("═" * 70))

    metricas = obtener_metricas_phoenix(epoch_inicio)
    if metricas and "error" not in metricas and "nota" not in metricas:
        print(f"  Spans RAG analizados    : {metricas.get('spans_analizados')}")
        print(f"  Modelo LLM              : {metricas.get('modelo_llm')}")
        print(f"  Modelo Embedding        : {metricas.get('modelo_embed')}")
        print(f"  K retrieval             : {metricas.get('k_retrieval')}")
        print(f"  Umbral relevancia       : {metricas.get('umbral_relevancia')}")
        print(f"  Tokens respuesta        : {metricas.get('num_tokens')}")
        print(f"  HyDE activo             : {metricas.get('hyde_aplicado')}")
        print(bold("  ─────────────── latencias promedio ───────────────"))
        lat_total = metricas.get("latencia_total_ms_avg")
        lat_llm   = metricas.get("latencia_llm_ms_avg")
        lat_hyde  = metricas.get("latencia_hyde_ms_avg")
        lat_otros = round((lat_total or 0) - (lat_llm or 0) - max(lat_hyde or 0, 0), 1)
        print(f"  Latencia total avg      : {lat_total} ms")
        print(f"  └─ LLM generación       : {lat_llm} ms")
        print(f"  └─ HyDE (estimado)      : {lat_hyde} ms")
        print(f"  └─ Retrieval + overhead : {lat_otros} ms")
        print(f"  Fragmentos usados avg   : {metricas.get('fragmentos_usados_avg')}")
    elif metricas and "nota" in metricas:
        print(f"  {warn(metricas['nota'])}")
        print(f"  {warn('Las métricas Phoenix solo están disponibles para consultas no cacheadas.')}")
    elif metricas and "error" in metricas:
        print(f"  {warn('Phoenix no disponible: ' + metricas['error'])}")
        print(f"  {warn('Asegúrate de que el servidor FastAPI esté corriendo (Phoenix se lanza desde él).')}")
    else:
        print(f"  {warn('Phoenix no respondió. Continúa sin métricas de sistema.')}")

    # ── Fase 4: Exportar CSV ───────────────────────────────────────────────────
    nombre_csv = (
        f"resultados_{EXPERIMENTO}_{MOTOR.replace(':', '-')}_"
        f"{ts_inicio.strftime('%Y-%m-%d_%H-%M')}.csv"
    )
    exportar_csv(resultados, nombre_csv)
    # TODO: ELIMINAR - Exportar trazas Phoenix (función no implementada)
    # exportar_trazas_phoenix(ts_inicio.strftime('%Y-%m-%d_%H-%M'))


    print(bold("\n═" * 70))
    print(bold("  GUÍA DE INTERPRETACIÓN"))
    print(bold("═" * 70))
    print("""
  TP Directo   → Si falla: problema de retrieval (umbral muy alto, k muy bajo,
                  chunking fragmenta el dato en dos chunks distintos).
                  Acción: bajar UMBRAL_RELEVANCIA o aumentar breakpoint_threshold.

  TP Razonam.  → Si falla: el LLM no sintetiza múltiples fragmentos, o los
                  fragmentos con los datos parciales no llegaron juntos.
                  Acción: subir K, revisar si HyDE mejora el match.

  TN Fuera     → Si falla: el sistema alucinó o el umbral está demasiado bajo
                  (entra contexto irrelevante que el LLM usa para inventar).
                  Acción: subir UMBRAL_RELEVANCIA o subir top_k/top_p LLM.

  Corrección   → Si falla: el LLM confirma el error del usuario (sycophancy).
                  Esto depende del modelo, no de parámetros RAG. Es un límite
                  del modelo local 8b y no tiene solución de parámetro.

  Anti-aluc.   → Si falla: ALUCINACIÓN CRÍTICA. El LLM usó conocimiento de
                  preentrenamiento ignorando el prompt de restricción.
                  Acción: verificar repeat_penalty, top_k, top_p. Considerar
                  modelo más grande.

  Interpretac. → Zona gris. Un PARCIAL aquí es aceptable.
""")

    print(bold("═" * 70))
    print(f"  Duración total del experimento: {round(time.time() - epoch_inicio)}s")
    print(bold("═" * 70 + "\n"))


if __name__ == "__main__":
    JWT_TOKEN = ""   # se rellena en main()
    main()