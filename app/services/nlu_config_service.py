# app/services/nlu_config_service.py

from __future__ import annotations

import time
from typing import Any
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models
from app.db.models import DEFAULTS_NLU

_CAMPOS_NLU = list(DEFAULTS_NLU.keys())

_NLU_CACHE_TTL = 60.0  # segundos

_nlu_cache: dict[str, Any] = {"data": None, "ts": 0.0}


def _invalidar_cache_nlu() -> None:
    """Fuerza lectura desde BD en la próxima llamada a get_nlu_config_cached."""
    _nlu_cache["ts"] = 0.0


def get_nlu_config_cached() -> dict[str, Any]:
    """Versión cacheada (sin SessionLocal por request).  Usar en el path caliente."""
    now = time.time()
    if _nlu_cache["data"] is not None and (now - _nlu_cache["ts"]) < _NLU_CACHE_TTL:
        return _nlu_cache["data"]

    db = SessionLocal()
    try:
        config = _get_or_create(db)
        data   = _config_to_dict(config)
    finally:
        db.close()

    _nlu_cache["data"] = data
    _nlu_cache["ts"]   = now
    return data


def _config_to_dict(config: models.ConfiguracionNLU) -> dict[str, Any]:
    return {campo: getattr(config, campo) for campo in _CAMPOS_NLU}


def _get_or_create(db: Session) -> models.ConfiguracionNLU:
    config = db.query(models.ConfiguracionNLU).filter_by(id=1).first()
    if config is None:
        config = models.ConfiguracionNLU(id=1, **DEFAULTS_NLU)
        db.add(config)
        db.commit()
        db.refresh(config)
    return config

def get_nlu_config_con_db(db: Session) -> models.ConfiguracionNLU:
    return _get_or_create(db)


def get_nlu_config(db: Session | None = None) -> dict[str, Any]:
    if db is not None:
        config = _get_or_create(db)
        return _config_to_dict(config)
    return get_nlu_config_cached()


def actualizar_nlu_config(
    db: Session,
    new_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config    = _get_or_create(db)
    old_state = _config_to_dict(config)

    for campo, valor in new_data.items():
        if campo not in _CAMPOS_NLU:
            continue
        if not hasattr(config, campo):
            continue
        setattr(config, campo, valor)

    db.commit()
    db.refresh(config)
    new_state = _config_to_dict(config)

    _invalidar_cache_nlu()
    return old_state, new_state


def resetear_nlu_config(db: Session) -> tuple[dict[str, Any], dict[str, Any]]:
    config    = _get_or_create(db)
    old_state = _config_to_dict(config)

    for campo, valor in DEFAULTS_NLU.items():
        if hasattr(config, campo):
            setattr(config, campo, valor)

    db.commit()
    db.refresh(config)
    new_state = _config_to_dict(config)

    _invalidar_cache_nlu()
    return old_state, new_state