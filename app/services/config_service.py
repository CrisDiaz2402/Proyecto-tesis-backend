import json
import os
from pathlib import Path

CONFIG_FILE = Path("./config_ia.json")

MOTORES_VALIDOS = ["local", "cloud"]

COMBINACIONES_INVALIDAS: set[tuple[str, str]] = {("cloud", "local")}


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
    if motor_vectores not in MOTORES_VALIDOS:
        raise ValueError(f"motor_vectores debe ser 'local' o 'cloud', recibido: '{motor_vectores}'")
    if motor_llm not in MOTORES_VALIDOS:
        raise ValueError(f"motor_llm debe ser 'local' o 'cloud', recibido: '{motor_llm}'")
    if (motor_vectores, motor_llm) in COMBINACIONES_INVALIDAS:
        raise ValueError(
            f"La combinación '{motor_vectores}:{motor_llm}' está deshabilitada. "
            "Los chunks del motor cloud son incompatibles con el LLM local y generan alucinaciones. "
            "Combinaciones válidas: local:local, local:cloud, cloud:cloud."
        )

    _crear_config_si_no_existe()
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"motor_vectores": motor_vectores, "motor_llm": motor_llm}, f, indent=2)
        return True
    except Exception:
        return False
