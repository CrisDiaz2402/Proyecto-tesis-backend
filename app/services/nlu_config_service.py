# app/services/nlu_config_service.py

from __future__ import annotations

from typing import Any
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models
from app.db.models import DEFAULTS_NLU

_CAMPOS_NLU = list(DEFAULTS_NLU.keys())


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
    db_local = SessionLocal()
    try:
        config = _get_or_create(db_local)
        return _config_to_dict(config)
    finally:
        db_local.close()


def actualizar_nlu_config(
    db: Session,
    new_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _get_or_create(db)
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
    return old_state, new_state


def resetear_nlu_config(db: Session) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _get_or_create(db)
    old_state = _config_to_dict(config)

    for campo, valor in DEFAULTS_NLU.items():
        if hasattr(config, campo):
            setattr(config, campo, valor)

    db.commit()
    db.refresh(config)
    new_state = _config_to_dict(config)
    return old_state, new_state
