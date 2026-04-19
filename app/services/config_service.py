import json
import os
from pathlib import Path

from app.core.event_bus import event_bus
from app.core.constants import COMBINACIONES_VALIDAS, MOTORES_VECTORES_VALIDOS, MOTORES_LLM_VALIDOS

CONFIG_FILE = Path("./config_ia.json")


def _crear_config_si_no_existe():
    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"motor_vectores": "local", "motor_llm": "local"}, f)


def _migrar_config_si_es_vieja(data: dict) -> dict:
    if "motor_activo" in data and "motor_vectores" not in data:
        motor = data.get("motor_activo", "local")
        return {"motor_vectores": motor, "motor_llm": motor}
    return data


def obtener_configuracion() -> dict:
    _crear_config_si_no_existe()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            data = _migrar_config_si_es_vieja(data)
            return {
                "motor_vectores": data.get("motor_vectores", "local"),
                "motor_llm":      data.get("motor_llm",      "local"),
            }
    except Exception:
        return {"motor_vectores": "local", "motor_llm": "local"}


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

    _crear_config_si_no_existe()
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"motor_vectores": motor_vectores, "motor_llm": motor_llm}, f, indent=2)
        print(f"[CONFIG] ✅ Motor cambiado a {motor_vectores}:{motor_llm}")
        event_bus.publicar("motor_cambiado", {
            "motor_vectores": motor_vectores,
            "motor_llm": motor_llm,
        })
        return True
    except Exception as e:
        print(f"[CONFIG] Error al guardar configuración: {e}")
        return False


def cambiar_motor_activo(nuevo_motor: str) -> bool:
    return cambiar_configuracion(motor_vectores=nuevo_motor, motor_llm=nuevo_motor)