# app/core/exceptions.py
"""
Excepciones tipadas del sistema RAG.
Permite manejo granular de errores en lugar de Exception genérica.
"""

from typing import Optional, Dict, Any


# ── BASE EXCEPTION ───────────────────────────────────────────────────────────

class RAGError(Exception):
    """Base para todos los errores del sistema RAG."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# ── ERRORES DE DOCUMENTOS ───────────────────────────────────────────────────────

class DocumentoError(RAGError):
    """Errores relacionados con documentos."""
    pass


class DocumentoNoEncontradoError(DocumentoError):
    """Documento no existe en base de datos."""
    
    def __init__(self, documento_id: int):
        super().__init__(f"Documento con ID {documento_id} no encontrado.")
        self.documento_id = documento_id


class LimiteDocumentosError(DocumentoError):
    """Límite de documentos alcanzado."""
    
    def __init__(self, maximo: int):
        super().__init__(f"Límite máximo de {maximo} documentos alcanzado.")
        self.maximo = maximo


class ExtensionNoPermitidaError(DocumentoError):
    """Extensión de archivo no válida."""
    
    def __init__(self, extension: str, permitidas: list):
        super().__init__(f"Extensión '.{extension}' no permitida. Permitidas: {', '.join(permitidas)}.")
        self.extension = extension
        self.permitidas = permitidas


class TamanoExcedidoError(DocumentoError):
    """Archivo excede el tamaño máximo."""
    
    def __init__(self, tamano_mb: float, limite_mb: int):
        super().__init__(f"Archivo de {tamano_mb:.1f}MB excede límite de {limite_mb}MB.")
        self.tamano_mb = tamano_mb
        self.limite_mb = limite_mb


# ── ERRORES DE VECTORIZACIÓN ──────────────────────────────────────────────────────

class VectorizacionError(RAGError):
    """Errores en el proceso de vectorización/embeddings."""
    pass


class ChromaDBError(VectorizacionError):
    """Errores específicos de ChromaDB (legacy)."""
    pass


class QdrantError(VectorizacionError):
    """Errores específicos de Qdrant (futuro)."""
    pass


# ── ERRORES DE LLM ───────────────────────────────────────────────────────────────

class LLMError(RAGError):
    """Errores de generación de lenguaje."""
    pass


class OllamaError(LLMError):
    """Errores específicos de Ollama (legacy)."""
    pass


class VLLMError(LLMError):
    """Errores específicos de vLLM (futuro)."""
    pass


class TimeoutError(LLMError):
    """Timeout en generación LLM."""
    
    def __init__(self, timeout_seconds: int):
        super().__init__(f"Timeout de {timeout_seconds}s excedido en generación LLM.")
        self.timeout_seconds = timeout_seconds


# ── ERRORES DE CONFIGURACIÓN ───────────────────────────────────────────────────────

class ConfiguracionError(RAGError):
    """Errores de configuración del sistema."""
    pass


class MotorInvalidoError(ConfiguracionError):
    """Motor especificado no es válido."""
    
    def __init__(self, motor: str, validos: list):
        super().__init__(f"Motor '{motor}' inválido. Válidos: {', '.join(validos)}.")
        self.motor = motor
        self.validos = validos


class CombinacionInvalidaError(ConfiguracionError):
    """Combinación de motores no soportada."""
    
    def __init__(self, vectores: str, llm: str):
        super().__init__(f"Combinación {vectores}:{llm} no soportada.")
        self.vectores = vectores
        self.llm = llm


# ── ERRORES DE AUTENTICACIÓN ───────────────────────────────────────────────────────

class AuthError(RAGError):
    """Errores de autenticación y autorización."""
    pass


class TokenInvalidoError(AuthError):
    """Token JWT inválido o expirado."""
    pass


class CredencialesInvalidasError(AuthError):
    """Credenciales de usuario incorrectas."""
    pass


class PermisosInsuficientesError(AuthError):
    """Usuario no tiene los permisos necesarios."""
    pass
