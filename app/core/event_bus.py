from typing import Callable, Dict, List


class EventBus:
    _listeners: Dict[str, List[Callable]] = {}

    @classmethod
    def suscribir(cls, evento: str, callback: Callable) -> None:
        cls._listeners.setdefault(evento, []).append(callback)

    @classmethod
    def publicar(cls, evento: str, datos: dict) -> None:
        for listener in cls._listeners.get(evento, []):
            try:
                listener(datos)
            except Exception as e:
                print(f"[EVENT_BUS] Error en listener de '{evento}': {e}")


event_bus = EventBus()
