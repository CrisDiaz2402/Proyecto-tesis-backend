# app/services/documento_service.py
"""
Service layer para lógica de negocio de documentos.
Separa la lógica HTTP (routers) de la lógica de negocio y acceso a datos.
"""

from pathlib import Path
from typing import Dict, Any, List
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.constants import MOTOR_LABELS, MOTORES_VALIDOS
from app.core.config import (
    DOCUMENTS_DIR_LOCAL, DOCUMENTS_DIR_CLOUD, MAX_DOCUMENTOS, 
    LIMITE_TAMANO_MB, LIMITE_TAMANO_BYTES, EXTENSIONES_PERMITIDAS
)
from app.core.exceptions import (
    DocumentoError, LimiteDocumentosError, ExtensionNoPermitidaError, 
    TamanoExcedidoError, MotorInvalidoError
)
from app.db import models
from app.repositories.documento_repository import DocumentoRepository
from app.services.rag_service import procesar_y_guardar_documento
from app.services.cache_service import limpiar_cache_por_documento


class DocumentoService:
    """Service para lógica de negocio de documentos."""
    
    def __init__(self, db: Session):
        self.db = db
        self.repository = DocumentoRepository(db)
    
    def listar_documentos(self) -> List[models.Documento]:
        """Retorna lista de todos los documentos."""
        return self.repository.get_all()
    
    async def procesar_subida(
        self, 
        file: UploadFile, 
        motor: str, 
        usuario: models.Usuario
    ) -> Dict[str, Any]:
        """
        Procesa subida completa de documento:
        1. Validaciones de negocio
        2. Guardado físico
        3. Vectorización RAG
        4. Registro en BD
        
        Returns: Dict con resultado del procesamiento
        Raises: DocumentoError para errores de negocio
        """
        # 1. Validaciones de negocio
        self._validar_limites_sistema()
        self._validar_archivo(file)
        self._validar_motor(motor)
        
        # 2. Determinar rutas según motor
        ruta_destino = self._get_ruta_destino(file.filename, motor)
        
        # 3. Guardar archivo físico
        await self._guardar_archivo_fisico(file, ruta_destino)
        
        # 4. Procesar RAG (vectorización)
        resultado_rag = procesar_y_guardar_documento(
            str(ruta_destino), 
            motor=motor
        )
        
        # 5. Crear registro en BD
        documento_data = {
            "nombre_archivo": file.filename,
            "ruta_local": str(ruta_destino) if motor in ["local", "all"] else None,
            "ruta_cloud": str(ruta_destino) if motor in ["cloud", "all"] else None,
            "procesado_local": motor in ["local", "all"],
            "procesado_cloud": motor in ["cloud", "all"],
            "tamano_bytes": file.size,
        }
        
        documento = self.repository.create(**documento_data)
        
        return {
            "documento": documento,
            "mensaje": f"Documento procesado exitosamente en {MOTOR_LABELS[motor]}.",
            "rag_info": resultado_rag,
            "motor": motor,
        }
    
    def eliminar_documento(self, documento_id: int, motor: str) -> Dict[str, Any]:
        """
        Elimina documento completamente:
        1. Archivo físico
        2. Vectores de ChromaDB/Qdrant
        3. Cache asociado
        4. Registro de BD
        """
        self._validar_motor(motor)
        
        # Obtener documento (lanza excepción si no existe)
        documento = self.repository.get_by_id(documento_id)
        
        # Eliminar archivos físicos según motor
        archivos_eliminados = self._eliminar_archivos_fisicos(documento, motor)
        
        # Limpiar vectores RAG
        # TODO: usar vector_service cuando se migre a Qdrant
        from app.services.rag_service import eliminar_coleccion_chroma
        eliminar_coleccion_chroma(documento.nombre_archivo, motor)
        
        # Limpiar cache
        limpiar_cache_por_documento(documento.nombre_archivo, motor)
        
        # Actualizar o eliminar registro BD
        if motor == "all":
            self.repository.delete(documento)
            mensaje = f"Documento '{documento.nombre_archivo}' eliminado completamente."
        else:
            # Solo marcar como no procesado en ese motor
            update_data = {}
            if motor == "local":
                update_data = {"procesado_local": False, "ruta_local": None}
            elif motor == "cloud":
                update_data = {"procesado_cloud": False, "ruta_cloud": None}
            
            self.repository.update(documento, **update_data)
            mensaje = f"Documento eliminado de {MOTOR_LABELS[motor]}."
        
        return {
            "mensaje": mensaje,
            "archivos_eliminados": archivos_eliminados,
            "motor": motor,
        }
    
    # ─────────────────────────────────────────────────────────────────────────────
    # MÉTODOS PRIVADOS DE VALIDACIÓN
    # ─────────────────────────────────────────────────────────────────────────────
    
    def _validar_limites_sistema(self):
        """Valida que no se exceda el límite de documentos."""
        total_docs = self.repository.count_total()
        if total_docs >= MAX_DOCUMENTOS:
            raise LimiteDocumentosError(MAX_DOCUMENTOS)
    
    def _validar_archivo(self, file: UploadFile):
        """Valida extensión y tamaño del archivo."""
        # Validar extensión
        extension = file.filename.split('.')[-1].lower()
        if extension not in EXTENSIONES_PERMITIDAS:
            raise ExtensionNoPermitidaError(extension, EXTENSIONES_PERMITIDAS)
        
        # Validar tamaño
        if file.size > LIMITE_TAMANO_BYTES:
            tamano_mb = file.size / (1024 * 1024)
            raise TamanoExcedidoError(tamano_mb, LIMITE_TAMANO_MB)
    
    def _validar_motor(self, motor: str):
        """Valida que el motor sea válido."""
        valid_motors = list(MOTORES_VALIDOS) + ["all"]
        if motor not in valid_motors:
            raise MotorInvalidoError(motor, valid_motors)
    
    def _get_ruta_destino(self, filename: str, motor: str) -> Path:
        """Determina ruta de destino según el motor."""
        if motor == "cloud":
            return DOCUMENTS_DIR_CLOUD / filename
        else:
            return DOCUMENTS_DIR_LOCAL / filename
    
    async def _guardar_archivo_fisico(self, file: UploadFile, ruta: Path):
        """Guarda archivo en el sistema de archivos."""
        try:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            
            with open(ruta, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            print(f"[DOCS] 💾 Archivo físico guardado: {ruta}")
        
        except Exception as e:
            raise DocumentoError(f"Error al guardar archivo físico: {str(e)}")
    
    def _eliminar_archivos_fisicos(self, documento: models.Documento, motor: str) -> List[str]:
        """Elimina archivos físicos según motor especificado."""
        eliminados = []
        
        if motor in ["cloud", "all"] and documento.ruta_cloud:
            path_cloud = Path(documento.ruta_cloud)
            if path_cloud.exists():
                path_cloud.unlink()
                eliminados.append(str(path_cloud))
        
        if motor in ["local", "all"] and documento.ruta_local:
            path_local = Path(documento.ruta_local)
            if path_local.exists():
                path_local.unlink()
                eliminados.append(str(path_local))
        
        return eliminados
