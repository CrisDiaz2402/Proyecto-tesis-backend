from abc import ABC, abstractmethod


class ValidadorDocumento(ABC):
    def __init__(self):
        self._siguiente: "ValidadorDocumento" = None

    def set_siguiente(self, validador: "ValidadorDocumento") -> "ValidadorDocumento":
        self._siguiente = validador
        return validador

    @abstractmethod
    def validar(self, contexto: dict) -> None: ...

    def _pasar_siguiente(self, contexto: dict) -> None:
        if self._siguiente:
            self._siguiente.validar(contexto)


class ValidadorLimiteDocumentos(ValidadorDocumento):
    def validar(self, ctx: dict) -> None:
        from app.core.config import MAX_DOCUMENTOS
        if ctx["repository"].count_total() >= MAX_DOCUMENTOS:
            from fastapi import HTTPException
            raise HTTPException(400, f"Límite máximo de {MAX_DOCUMENTOS} documentos alcanzado.")
        self._pasar_siguiente(ctx)


class ValidadorExtension(ValidadorDocumento):
    def validar(self, ctx: dict) -> None:
        from app.core.config import EXTENSIONES_PERMITIDAS
        ext = "." + ctx["file"].filename.split(".")[-1].lower()
        if ext not in EXTENSIONES_PERMITIDAS:
            from fastapi import HTTPException
            raise HTTPException(400, f"Extensión '{ext}' no permitida.")
        self._pasar_siguiente(ctx)


class ValidadorTamano(ValidadorDocumento):
    def validar(self, ctx: dict) -> None:
        from app.core.config import LIMITE_TAMANO_BYTES, LIMITE_TAMANO_MB
        if ctx["file"].size and ctx["file"].size > LIMITE_TAMANO_BYTES:
            from fastapi import HTTPException
            raise HTTPException(400, f"El archivo supera el límite de {LIMITE_TAMANO_MB} MB.")
        self._pasar_siguiente(ctx)


class ValidadorMotor(ValidadorDocumento):
    def validar(self, ctx: dict) -> None:
        from app.core.constants import MOTORES_VECTORES_VALIDOS
        if ctx["motor"] not in MOTORES_VECTORES_VALIDOS:
            from fastapi import HTTPException
            raise HTTPException(400, f"Motor '{ctx['motor']}' no válido. Solo se soporta 'local'.")
        self._pasar_siguiente(ctx)


def construir_cadena_validacion() -> ValidadorDocumento:
    v1 = ValidadorLimiteDocumentos()
    v2 = ValidadorExtension()
    v3 = ValidadorTamano()
    v4 = ValidadorMotor()
    v1.set_siguiente(v2).set_siguiente(v3).set_siguiente(v4)
    return v1
