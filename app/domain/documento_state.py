from abc import ABC, abstractmethod


class EstadoDocumento(ABC):
    @abstractmethod
    def procesar(self, documento, motor: str) -> None: ...
    @abstractmethod
    def eliminar(self, documento, motor: str) -> None: ...
    @abstractmethod
    def nombre(self) -> str: ...


def _aplicar_estado_procesando(doc, motor: str):
    doc.procesado_local = False
    doc.estado_local = "Procesando"


def _aplicar_estado_no_subido(doc, motor: str):
    doc.procesado_local = False
    doc.estado_local = "No subido"


class EstadoNoSubido(EstadoDocumento):
    def procesar(self, doc, motor: str):
        _aplicar_estado_procesando(doc, motor)

    def eliminar(self, doc, motor: str):
        raise ValueError("No hay nada que eliminar en estado 'No subido'")

    def nombre(self): return "No subido"


class EstadoProcesado(EstadoDocumento):
    def procesar(self, doc, motor: str):
        _aplicar_estado_procesando(doc, motor)

    def eliminar(self, doc, motor: str):
        _aplicar_estado_no_subido(doc, motor)

    def nombre(self): return "Procesado"


class EstadoError(EstadoDocumento):
    def procesar(self, doc, motor: str):
        _aplicar_estado_procesando(doc, motor)

    def eliminar(self, doc, motor: str):
        _aplicar_estado_no_subido(doc, motor)

    def nombre(self): return "Error"


def get_estado(doc, motor: str) -> EstadoDocumento:
    procesado = doc.procesado_local
    estado_str = doc.estado_local
    if procesado:
        return EstadoProcesado()
    if "Error" in (estado_str or ""):
        return EstadoError()
    return EstadoNoSubido()
