# app/core/exceptions.py
from typing import Optional, Dict, Any

class RAGError(Exception):
    
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

class LLMError(RAGError):
    pass

class ConfiguracionError(RAGError):
    pass


class CombinacionInvalidaError(ConfiguracionError):    
    def __init__(self, vectores: str, llm: str):
        super().__init__(f"Combinación {vectores}:{llm} no soportada.")
        self.vectores = vectores
        self.llm = llm
