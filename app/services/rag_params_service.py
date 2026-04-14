# app/services/rag_params_service.py
"""
Servicio de gestión de parámetros RAG.

Responsabilidades:
  - Definir DEFAULTS y PARAM_LIMITS (fuente de verdad para validación y UI).
  - Leer/escribir ConfiguracionRAG en BD (tabla singleton id=1).
  - Validar rangos de parámetros numéricos.
  - Determinar qué cachés/vectores deben limpiarse según qué cambió.
  - Exponer get_params() para uso sin sesión explícita (cache_service, rag_service).

NO importa cache_service ni rag_service — esas dependencias se resuelven
con imports lazy dentro del router para evitar importaciones circulares.
"""

from __future__ import annotations

from typing import Any, Optional
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models


# ─────────────────────────────────────────────────────────────────────────────
# TEXTOS POR DEFECTO DE LOS PROMPTS
#
# Son idénticos a _PROMPT_PRINCIPAL_DEFAULT y _PROMPT_HYDE_DEFAULT de
# rag_service.py. Se definen aquí (y no se importan desde rag_service) para
# evitar importaciones circulares.
#
# Estos textos se usan en dos situaciones:
#   1. Crear la fila id=1 por primera vez (_get_or_create).
#   2. Migrar filas existentes que tengan NULL (startup de main.py).
#   3. Resetear los prompts a su valor original (resetear_params).
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_PRINCIPAL_DEFAULT = """Eres el Asistente Académico de la EPN. Eres un sistema estricto de extracción de datos, no un consejero.

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

PROMPT_HYDE_DEFAULT = (
    "Escribe una respuesta corta, factual y directa en español a esta pregunta "
    "sobre la malla curricular de la Carrera de Ciencias de la Computación de la EPN. "
    "Usa términos académicos concretos. Máximo 3 oraciones. "
    "No expliques, solo responde con datos.\n\n"
    "Pregunta: {pregunta}\nRespuesta:"
)


# ─────────────────────────────────────────────────────────────────────────────
# VALORES POR DEFECTO — espejo de los Column(default=...) en models.py
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS: dict[str, Any] = {
    # Alto impacto — Chunking
    "breakpoint_threshold_amount": 75,

    # Alto impacto — Retrieval
    "umbral_relevancia_local": 0.15,
    "umbral_relevancia_cloud": 0.30,
    "rag_k_local":             10,
    "rag_k_cloud":             8,

    # Medio impacto — Tokens
    "num_tokens_normal_local": 350,
    "num_tokens_lista_local":  750,
    "num_tokens_normal_cloud": 700,
    "num_tokens_lista_cloud":  1400,

    # Medio impacto — Caché L2
    "cache_threshold_ll": 0.82,
    "cache_threshold_lc": 0.83,
    "cache_threshold_cc": 0.88,
    "umbral_similitud":   0.02,

    # Bajo impacto — LLM local
    "repeat_penalty":    1.3,
    "top_k_llm":         10,
    "top_p_llm":         0.5,
    "hyde_num_predict":  120,

    # Bajo impacto — Caché L1
    "max_l1_entries": 500,

    # Prompts editables — se guardan en BD con su texto completo.
    # NULL en BD significa que rag_service usará el fallback hardcodeado,
    # pero desde este deploy siempre se inicializan con el texto real.
    "prompt_principal": PROMPT_PRINCIPAL_DEFAULT,
    "prompt_hyde":      PROMPT_HYDE_DEFAULT,
}


# ─────────────────────────────────────────────────────────────────────────────
# LÍMITES DE VALIDACIÓN — se devuelven al frontend para renderizar controles
# ─────────────────────────────────────────────────────────────────────────────
PARAM_LIMITS: dict[str, dict[str, Any]] = {
    "breakpoint_threshold_amount": {
        "min": 50, "max": 95, "type": "int",
        "default": 75,
        "label": "Umbral de corte semántico (%)",
        "descripcion": (
            "Percentil para el SemanticChunker. Valores altos → chunks más pequeños. "
            "Cambiar este valor elimina los vectores y obliga a reindexar."
        ),
    },
    "umbral_relevancia_local": {
        "min": 0.05, "max": 0.50, "type": "float",
        "default": 0.15,
        "label": "Umbral de relevancia local",
        "descripcion": "Score coseno mínimo para que un fragmento local entre al contexto.",
    },
    "umbral_relevancia_cloud": {
        "min": 0.10, "max": 0.70, "type": "float",
        "default": 0.30,
        "label": "Umbral de relevancia cloud",
        "descripcion": "Score coseno mínimo para que un fragmento cloud entre al contexto.",
    },
    "rag_k_local": {
        "min": 2, "max": 20, "type": "int",
        "default": 10,
        "label": "K local (fragmentos a recuperar)",
        "descripcion": "Número de fragmentos a recuperar de ChromaDB local.",
    },
    "rag_k_cloud": {
        "min": 2, "max": 15, "type": "int",
        "default": 8,
        "label": "K cloud (fragmentos a recuperar)",
        "descripcion": "Número de fragmentos a recuperar de ChromaDB cloud.",
    },
    "num_tokens_normal_local": {
        "min": 100, "max": 800, "type": "int",
        "default": 350,
        "label": "Tokens respuesta normal (local)",
        "descripcion": "Máximo de tokens para respuestas normales con LLM local.",
    },
    "num_tokens_lista_local": {
        "min": 200, "max": 1500, "type": "int",
        "default": 750,
        "label": "Tokens respuesta lista (local)",
        "descripcion": "Máximo de tokens para respuestas tipo lista con LLM local.",
    },
    "num_tokens_normal_cloud": {
        "min": 200, "max": 1500, "type": "int",
        "default": 700,
        "label": "Tokens respuesta normal (cloud)",
        "descripcion": "Máximo de tokens para respuestas normales con LLM cloud.",
    },
    "num_tokens_lista_cloud": {
        "min": 400, "max": 3000, "type": "int",
        "default": 1400,
        "label": "Tokens respuesta lista (cloud)",
        "descripcion": "Máximo de tokens para respuestas tipo lista con LLM cloud.",
    },
    "cache_threshold_ll": {
        "min": 0.60, "max": 0.99, "type": "float",
        "default": 0.82,
        "label": "Umbral caché L2 local:local",
        "descripcion": "Similitud mínima para hit de caché en modo local:local.",
    },
    "cache_threshold_lc": {
        "min": 0.60, "max": 0.99, "type": "float",
        "default": 0.83,
        "label": "Umbral caché L2 local:cloud",
        "descripcion": "Similitud mínima para hit de caché en modo local:cloud.",
    },
    "cache_threshold_cc": {
        "min": 0.70, "max": 0.99, "type": "float",
        "default": 0.88,
        "label": "Umbral caché L2 cloud:cloud",
        "descripcion": "Similitud mínima para hit de caché en modo cloud:cloud.",
    },
    "umbral_similitud": {
        "min": 0.01, "max": 0.20, "type": "float",
        "default": 0.02,
        "label": "Margen global de similitud",
        "descripcion": "Margen de tolerancia para hit de caché L2.",
    },
    "repeat_penalty": {
        "min": 1.0, "max": 1.8, "type": "float",
        "default": 1.3,
        "label": "Penalización de repetición (LLM local)",
        "descripcion": "Penaliza tokens repetidos en el LLM local (Ollama).",
    },
    "top_k_llm": {
        "min": 1, "max": 100, "type": "int",
        "default": 10,
        "label": "Top-K (LLM local)",
        "descripcion": "Candidatos de vocabulario por paso en el LLM local.",
    },
    "top_p_llm": {
        "min": 0.1, "max": 1.0, "type": "float",
        "default": 0.5,
        "label": "Top-P (LLM local)",
        "descripcion": "Nucleus sampling para el LLM local.",
    },
    "hyde_num_predict": {
        "min": 40, "max": 300, "type": "int",
        "default": 120,
        "label": "Tokens HyDE (num_predict)",
        "descripcion": "Tokens máximos para la respuesta hipotética HyDE.",
    },
    "max_l1_entries": {
        "min": 50, "max": 2000, "type": "int",
        "default": 500,
        "label": "Entradas máximas caché L1",
        "descripcion": "Tamaño máximo del caché L1 en RAM.",
    },
}

# Campos que NO son numéricos y deben excluirse de la validación de rangos
_CAMPOS_TEXTO = {"prompt_principal", "prompt_hyde"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _config_to_dict(config: models.ConfiguracionRAG) -> dict[str, Any]:
    """Convierte una fila ConfiguracionRAG a dict plano (incluye prompts)."""
    return {
        "breakpoint_threshold_amount": config.breakpoint_threshold_amount,
        "umbral_relevancia_local":     config.umbral_relevancia_local,
        "umbral_relevancia_cloud":     config.umbral_relevancia_cloud,
        "rag_k_local":                 config.rag_k_local,
        "rag_k_cloud":                 config.rag_k_cloud,
        "num_tokens_normal_local":     config.num_tokens_normal_local,
        "num_tokens_lista_local":      config.num_tokens_lista_local,
        "num_tokens_normal_cloud":     config.num_tokens_normal_cloud,
        "num_tokens_lista_cloud":      config.num_tokens_lista_cloud,
        "cache_threshold_ll":          config.cache_threshold_ll,
        "cache_threshold_lc":          config.cache_threshold_lc,
        "cache_threshold_cc":          config.cache_threshold_cc,
        "umbral_similitud":            config.umbral_similitud,
        "repeat_penalty":              config.repeat_penalty,
        "top_k_llm":                   config.top_k_llm,
        "top_p_llm":                   config.top_p_llm,
        "hyde_num_predict":            config.hyde_num_predict,
        "max_l1_entries":              config.max_l1_entries,
        "prompt_principal":            config.prompt_principal,
        "prompt_hyde":                 config.prompt_hyde,
    }


def _get_or_create(db: Session) -> models.ConfiguracionRAG:
    """
    Devuelve la fila singleton (id=1). Si no existe la crea con los defaults,
    incluyendo el texto completo de los prompts (nunca NULL en filas nuevas).
    """
    config = db.query(models.ConfiguracionRAG).filter_by(id=1).first()
    if config is None:
        config = models.ConfiguracionRAG(id=1, **{
            k: v for k, v in DEFAULTS.items()
        })
        db.add(config)
        db.commit()
        db.refresh(config)
    return config


# ─────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────────────────

def get_params_con_db(db: Session) -> models.ConfiguracionRAG:
    """Devuelve el objeto ORM. Útil cuando el llamador ya tiene la sesión."""
    return _get_or_create(db)


def get_params() -> dict[str, Any]:
    """
    Devuelve los parámetros actuales como dict, abriendo su propia sesión.
    Usar desde cache_service y rag_service (no tienen sesión inyectada).
    """
    db = SessionLocal()
    try:
        config = _get_or_create(db)
        return _config_to_dict(config)
    finally:
        db.close()


def validar_params(data: dict[str, Any]) -> dict[str, str]:
    """
    Valida rangos de los parámetros numéricos recibidos.
    Ignora los campos de texto (prompts).
    Devuelve dict {campo: mensaje_error}; vacío si todo es válido.
    """
    errores: dict[str, str] = {}
    for campo, valor in data.items():
        if campo in _CAMPOS_TEXTO:
            continue
        if campo not in PARAM_LIMITS:
            continue
        limites = PARAM_LIMITS[campo]
        if valor < limites["min"] or valor > limites["max"]:
            errores[campo] = (
                f"{limites['label']}: valor {valor} fuera del rango "
                f"[{limites['min']}, {limites['max']}]."
            )
    return errores


def actualizar_params(
    db: Session,
    new_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Persiste los campos de new_data en BD.
    Para prompts: cadena vacía "" → restaura al texto por defecto (no NULL).
    Devuelve (old_params_dict, new_params_dict).
    """
    config    = _get_or_create(db)
    old_state = _config_to_dict(config)

    for campo, valor in new_data.items():
        if not hasattr(config, campo):
            continue
        # Cadena vacía en prompts → volver al texto hardcodeado (no a NULL)
        if campo in _CAMPOS_TEXTO and isinstance(valor, str) and valor.strip() == "":
            setattr(config, campo, DEFAULTS[campo])
        else:
            setattr(config, campo, valor)

    db.commit()
    db.refresh(config)
    new_state = _config_to_dict(config)
    return old_state, new_state


def resetear_params(db: Session) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Restaura TODOS los parámetros (incluidos los prompts) a sus defaults.
    Los prompts se resetean al texto completo hardcodeado, nunca a NULL.
    Devuelve (old_params_dict, new_params_dict).
    """
    config    = _get_or_create(db)
    old_state = _config_to_dict(config)

    for campo, valor in DEFAULTS.items():
        if hasattr(config, campo):
            setattr(config, campo, valor)

    db.commit()
    db.refresh(config)
    new_state = _config_to_dict(config)
    return old_state, new_state


def determinar_limpieza(
    old: dict[str, Any],
    new: dict[str, Any],
) -> dict[str, Any]:
    """
    Compara old vs new y decide qué debe limpiarse.

    Reglas:
      - breakpoint_threshold_amount cambió → eliminar TODOS los vectores + L1 + L2
      - cache_threshold_ll cambió → limpiar caché L2 local:local
      - cache_threshold_lc cambió → limpiar caché L2 local:cloud
      - cache_threshold_cc cambió → limpiar caché L2 cloud:cloud
      - umbral_similitud cambió   → limpiar todos los cachés L2
      - umbral_relevancia_* o rag_k_* cambiaron → limpiar L1
      - max_l1_entries cambió     → limpiar L1
      - prompts cambiaron         → limpiar L1 + todos los cachés L2
        (respuestas cacheadas con el prompt viejo ya no son válidas)

    Devuelve dict con flags booleanos y lista params_cambiados.
    """
    params_cambiados: list[str] = [
        k for k in new
        if k in old and old[k] != new[k]
    ]
    cambios = set(params_cambiados)

    chunking_cambio   = "breakpoint_threshold_amount" in cambios
    prompts_cambiaron = bool(cambios & _CAMPOS_TEXTO)

    # Caché L2 por combinación
    limpiar_ll = (
        chunking_cambio
        or prompts_cambiaron
        or "cache_threshold_ll" in cambios
        or "umbral_similitud"   in cambios
    )
    limpiar_lc = (
        chunking_cambio
        or prompts_cambiaron
        or "cache_threshold_lc" in cambios
        or "umbral_similitud"   in cambios
    )
    limpiar_cc = (
        chunking_cambio
        or prompts_cambiaron
        or "cache_threshold_cc" in cambios
        or "umbral_similitud"   in cambios
    )

    # Caché L1
    limpiar_l1 = (
        chunking_cambio
        or prompts_cambiaron
        or limpiar_ll or limpiar_lc or limpiar_cc
        or bool(cambios & {"umbral_relevancia_local", "umbral_relevancia_cloud",
                           "rag_k_local", "rag_k_cloud", "max_l1_entries"})
    )

    return {
        "params_cambiados":     params_cambiados,
        "limpiar_l1":           limpiar_l1,
        "limpiar_cache_ll":     limpiar_ll,
        "limpiar_cache_lc":     limpiar_lc,
        "limpiar_cache_cc":     limpiar_cc,
        "limpiar_vectores_local":  chunking_cambio,
        "limpiar_vectores_cloud":  chunking_cambio,
        "requiere_reindexar":      chunking_cambio,
    }