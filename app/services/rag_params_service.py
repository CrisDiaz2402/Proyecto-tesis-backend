from __future__ import annotations

import time
from typing import Any, Optional
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models
from app.core.prompts import SYSTEM_PROMPT_FIJO, USER_TEMPLATE

DEFAULTS: dict[str, Any] = {
    "umbral_relevancia_local": 0.05,
    "rag_k_local":             10,
    "prompt_principal":        USER_TEMPLATE + "\nRespuesta:",
    "system_prompt":           SYSTEM_PROMPT_FIJO,
}

PARAM_LIMITS: dict[str, dict[str, Any]] = {
    "umbral_relevancia_local": {
        "min": 0.05, "max": 0.50, "type": "float",
        "default": 0.05,
        "label": "Umbral de relevancia local",
        "descripcion": "Score coseno mínimo para que un fragmento local entre al contexto.",
    },
    "rag_k_local": {
        "min": 2, "max": 20, "type": "int",
        "default": 10,
        "label": "K local (fragmentos a recuperar)",
        "descripcion": "Número de fragmentos a recuperar de Qdrant (colección local).",
    },
}

_CAMPOS_TEXTO = {"prompt_principal", "system_prompt"}

_RAG_PARAMS_CACHE_TTL = 30.0

_rag_params_cache: dict[str, Any] = {"data": None, "ts": 0.0}


def _invalidar_cache_rag_params() -> None:
    _rag_params_cache["ts"] = 0.0


def _config_to_dict(config: models.ConfiguracionRAG) -> dict[str, Any]:
    return {
        "umbral_relevancia_local": config.umbral_relevancia_local,
        "rag_k_local":             config.rag_k_local,
        "prompt_principal":        config.prompt_principal,
        "system_prompt":           config.system_prompt,
    }


def _get_or_create(db: Session) -> models.ConfiguracionRAG:
    config = db.query(models.ConfiguracionRAG).filter_by(id=1).first()
    if config is None:
        config = models.ConfiguracionRAG(id=1, **{
            k: v for k, v in DEFAULTS.items()
        })
        db.add(config)
        db.commit()
        db.refresh(config)
    return config


def get_params_con_db(db: Session) -> models.ConfiguracionRAG:
    return _get_or_create(db)


def get_params() -> dict[str, Any]:
    now = time.time()
    if _rag_params_cache["data"] is not None and (now - _rag_params_cache["ts"]) < _RAG_PARAMS_CACHE_TTL:
        return _rag_params_cache["data"]

    db = SessionLocal()
    try:
        config = _get_or_create(db)
        data   = _config_to_dict(config)
    finally:
        db.close()

    _rag_params_cache["data"] = data
    _rag_params_cache["ts"]   = now
    return data


def validar_params(data: dict[str, Any]) -> dict[str, str]:
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
    config    = _get_or_create(db)
    old_state = _config_to_dict(config)

    for campo, valor in new_data.items():
        if not hasattr(config, campo):
            continue
        if campo in _CAMPOS_TEXTO and isinstance(valor, str) and valor.strip() == "":
            setattr(config, campo, DEFAULTS[campo])
        else:
            setattr(config, campo, valor)

    db.commit()
    db.refresh(config)
    new_state = _config_to_dict(config)

    _invalidar_cache_rag_params()
    return old_state, new_state


def resetear_params(db: Session) -> tuple[dict[str, Any], dict[str, Any]]:
    config    = _get_or_create(db)
    old_state = _config_to_dict(config)

    for campo, valor in DEFAULTS.items():
        if hasattr(config, campo):
            setattr(config, campo, valor)

    db.commit()
    db.refresh(config)
    new_state = _config_to_dict(config)

    _invalidar_cache_rag_params()
    return old_state, new_state


def determinar_limpieza(
    old: dict[str, Any],
    new: dict[str, Any],
) -> dict[str, Any]:
    params_cambiados: list[str] = [
        k for k in new
        if k in old and old[k] != new[k]
    ]
    cambios = set(params_cambiados)

    prompt_cambio = bool(cambios & _CAMPOS_TEXTO)
    limpiar_ll    = prompt_cambio
    limpiar_lc    = prompt_cambio

    return {
        "params_cambiados":       params_cambiados,
        "limpiar_cache_ll":       limpiar_ll,
        "limpiar_cache_lc":       limpiar_lc,
        "limpiar_vectores_local": False,
        "requiere_reindexar":     False,
    }