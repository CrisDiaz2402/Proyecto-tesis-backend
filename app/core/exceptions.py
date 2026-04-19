# app/core/exceptions.py
from typing import Optional, Dict, Any

class RAGError(Exception):
    """Base para todos los errores del sistema RAG."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class DocumentoError(RAGError):
    pass


class DocumentoNoEncontradoError(DocumentoError):
    def __init__(self, documento_id: int):
        super().__init__(f"Documento con ID {documento_id} no encontrado.")
        self.documento_id = documento_id

class LimiteDocumentosError(DocumentoError):    
    def __init__(self, maximo: int):
        super().__init__(f"Límite máximo de {maximo} documentos alcanzado.")
        self.maximo = maximo


class ExtensionNoPermitidaError(DocumentoError):
    def __init__(self, extension: str, permitidas: list):
        super().__init__(f"Extensión '.{extension}' no permitida. Permitidas: {', '.join(permitidas)}.")
        self.extension = extension
        self.permitidas = permitidas

class TamanoExcedidoError(DocumentoError):    
    def __init__(self, tamano_mb: float, limite_mb: int):
        super().__init__(f"Archivo de {tamano_mb:.1f}MB excede límite de {limite_mb}MB.")
        self.tamano_mb = tamano_mb
        self.limite_mb = limite_mb

class VectorizacionError(RAGError):
    pass

class LLMError(RAGError):
    pass

class GeminiError(LLMError):
    pass


class TimeoutError(LLMError):    
    def __init__(self, timeout_seconds: int):
        super().__init__(f"Timeout de {timeout_seconds}s excedido en generación LLM.")
        self.timeout_seconds = timeout_seconds

class ConfiguracionError(RAGError):
    pass


class MotorInvalidoError(ConfiguracionError):    
    def __init__(self, motor: str, validos: list):
        super().__init__(f"Motor '{motor}' inválido. Válidos: {', '.join(validos)}.")
        self.motor = motor
        self.validos = validos


class CombinacionInvalidaError(ConfiguracionError):    
    def __init__(self, vectores: str, llm: str):
        super().__init__(f"Combinación {vectores}:{llm} no soportada.")
        self.vectores = vectores
        self.llm = llm

class AuthError(RAGError):
    pass
