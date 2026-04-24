# app/services/config_service.py
import time
from app.core.event_bus import event_bus
from app.core.constants import COMBINACIONES_VALIDAS, MOTORES_VECTORES_VALIDOS, MOTORES_LLM_VALIDOS

_CONFIG_TTL = 10.0
_config_cache: dict = {"data": None, "ts": 0.0}


def _invalidar_cache() -> None:
    _config_cache["ts"] = 0.0


def _get_or_create_motor(db):
    from app.db import models
    config = db.query(models.ConfiguracionMotor).filter_by(id=1).first()
    if config is None:
        config = models.ConfiguracionMotor(id=1, motor_vectores="local", motor_llm="local")
        db.add(config)
        db.commit()
        db.refresh(config)
    return config


def obtener_configuracion() -> dict:
    if _config_cache["data"] is not None and (time.time() - _config_cache["ts"]) < _CONFIG_TTL:
        return _config_cache["data"]

    from app.db.database import SessionLocal
    db = SessionLocal()
    try:
        config = _get_or_create_motor(db)
        result = {
            "motor_vectores": config.motor_vectores,
            "motor_llm": config.motor_llm,
        }
    except Exception:
        result = {"motor_vectores": "local", "motor_llm": "local"}
    finally:
        db.close()

    _config_cache["data"] = result
    _config_cache["ts"] = time.time()
    return result


def obtener_motor_activo() -> str:
    config = obtener_configuracion()
    return f"{config['motor_vectores']}:{config['motor_llm']}"


def cambiar_configuracion(motor_vectores: str, motor_llm: str) -> bool:
    if motor_vectores not in MOTORES_VECTORES_VALIDOS:
        raise ValueError(f"motor_vectores debe ser 'local', recibido: '{motor_vectores}'")
    if motor_llm not in MOTORES_LLM_VALIDOS:
        raise ValueError(f"motor_llm debe ser 'local' o 'cloud', recibido: '{motor_llm}'")
    if (motor_vectores, motor_llm) not in COMBINACIONES_VALIDAS:
        raise ValueError(
            f"La combinación '{motor_vectores}:{motor_llm}' no es válida. "
            "Combinaciones válidas: local:local, local:cloud."
        )

    from app.db.database import SessionLocal
    db = SessionLocal()
    try:
        config = _get_or_create_motor(db)
        config.motor_vectores = motor_vectores
        config.motor_llm = motor_llm
        db.commit()
        print(f"[CONFIG] Motor cambiado a {motor_vectores}:{motor_llm}")

        _invalidar_cache()

        event_bus.publicar("motor_cambiado", {
            "motor_vectores": motor_vectores,
            "motor_llm": motor_llm,
        })
        return True
    except Exception as e:
        print(f"[CONFIG] Error al guardar configuración: {e}")
        db.rollback()
        return False
    finally:
        db.close()


def cambiar_motor_activo(nuevo_motor: str) -> bool:
    return cambiar_configuracion(motor_vectores=nuevo_motor, motor_llm=nuevo_motor)
