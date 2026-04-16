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
from app.core.prompts import PROMPT_PRINCIPAL_DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES — Defaults y límites - Prompts centralizados en core/prompts.py
# ─────────────────────────────────────────────────────────────────────────────
# VALORES POR DEFECTO — espejo de los Column(default=...) en models.py
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS: dict[str, Any] = {
    # Parámetros con impacto demostrable en calidad del retrieval
    "umbral_relevancia_local": 0.15,
    "umbral_relevancia_cloud": 0.30,
    "rag_k_local":             10,
    "rag_k_cloud":             8,

    # Prompt principal — máximo impacto en calidad de respuesta
    "prompt_principal": PROMPT_PRINCIPAL_DEFAULT,
}


# ─────────────────────────────────────────────────────────────────────────────
# LÍMITES DE VALIDACIÓN — se devuelven al frontend para renderizar controles
# ─────────────────────────────────────────────────────────────────────────────
PARAM_LIMITS: dict[str, dict[str, Any]] = {
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
        "descripcion": "Número de fragmentos a recuperar de pgvector local.",
    },
    "rag_k_cloud": {
        "min": 2, "max": 15, "type": "int",
        "default": 8,
        "label": "K cloud (fragmentos a recuperar)",
        "descripcion": "Número de fragmentos a recuperar de pgvector cloud.",
    },
}

# Campos que NO son numéricos y deben excluirse de la validación de rangos
_CAMPOS_TEXTO = {"prompt_principal"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _config_to_dict(config: models.ConfiguracionRAG) -> dict[str, Any]:
    """Convierte una fila ConfiguracionRAG a dict plano (solo parámetros con impacto demostrable)."""
    return {
        "umbral_relevancia_local":     config.umbral_relevancia_local,
        "umbral_relevancia_cloud":     config.umbral_relevancia_cloud,
        "rag_k_local":                 config.rag_k_local,
        "rag_k_cloud":                 config.rag_k_cloud,
        "prompt_principal":            config.prompt_principal,
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

    Con los parámetros hardcodeados, solo cleaning necesario es cuando cambia el prompt_principal,
    ya que los umbrales de caché y similarity están hardcodeados en cache_service.py.

    Devuelve dict con flags booleanos y lista params_cambiados.
    """
    params_cambiados: list[str] = [
        k for k in new
        if k in old and old[k] != new[k]
    ]
    cambios = set(params_cambiados)

    prompt_cambio = bool(cambios & _CAMPOS_TEXTO)

    # Solo limpiamos caché L2 si cambió el prompt (impacta respuestas)
    # Los parámetros de threshold están hardcodeados, no generan limpieza
    limpiar_ll = prompt_cambio
    limpiar_lc = prompt_cambio
    limpiar_cc = prompt_cambio

    return {
        "params_cambiados":         params_cambiados,
        "limpiar_cache_ll":         limpiar_ll,
        "limpiar_cache_lc":         limpiar_lc,
        "limpiar_cache_cc":         limpiar_cc,
        "limpiar_vectores_local":   False,
        "limpiar_vectores_cloud":   False,
        "requiere_reindexar":       False,
    }