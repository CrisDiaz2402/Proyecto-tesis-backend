from pathlib import Path
from typing import Dict, Any, List
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.core.constants import MOTOR_LABELS, MOTORES_VECTORES_VALIDOS
from app.core.config import (
    DOCUMENTS_DIR_LOCAL, MAX_DOCUMENTOS, 
    LIMITE_TAMANO_MB, LIMITE_TAMANO_BYTES, EXTENSIONES_PERMITIDAS
)
from app.core.exceptions import DocumentoError
from app.db import models
from app.repositories.documento_repository import DocumentoRepository
from app.services.rag_service import procesar_y_guardar_documento
from app.services.cache_service import limpiar_cache, limpiar_cache_por_documento


class DocumentoService:
    
    def __init__(self, db: Session):
        self.db = db
        self.repository = DocumentoRepository(db)
    
    def listar_documentos(self) -> List[models.Documento]:
        return self.repository.get_all()
    
    async def procesar_subida(
        self, 
        file: UploadFile, 
        motor: str, 
        usuario: models.Usuario
    ) -> Dict[str, Any]:
        self._validar_limites_sistema()
        self._validar_archivo(file)
        self._validar_motor(motor)
        
        ruta_destino = self._get_ruta_destino(file.filename, motor)
        
        await self._guardar_archivo_fisico(file, ruta_destino)
        
        try:
            limpiar_cache(motor=motor)
            procesar_y_guardar_documento(str(ruta_destino), motor=motor)
        except Exception as e:
            ruta_destino.unlink(missing_ok=True)
            raise DocumentoError(f"Error al procesar RAG ({motor}): {e}")
        
        existente = self.repository.get_by_nombre(file.filename)
        if existente:
            update_data = {
                "procesado_local": True,
                "estado_local": "Procesado Localmente",
                "ruta_local": str(ruta_destino),
            }
            self.repository.update(existente, **update_data)
            documento = existente
        else:
            documento_data = {
                "nombre_archivo": file.filename,
                "subido_por": usuario.username,
                "ruta_local": str(ruta_destino),
                "procesado_local": True,
                "estado_local": "Procesado Localmente",
            }
            documento = self.repository.create(**documento_data)
        
        return {
            "documento": documento,
            "mensaje": f"Documento '{file.filename}' procesado y añadido a la base de conocimiento.",
            "motor": motor,
        }
    
    def eliminar_documento(self, documento_id: int, motor: str) -> Dict[str, Any]:
        self._validar_motor(motor)
        
        documento = self.repository.get_by_id(documento_id)

        documento.procesado_local = False
        documento.estado_local = "No subido"
        
        archivos_eliminados = self._eliminar_archivos_fisicos(documento, motor)
        
        from app.services.rag_service import eliminar_coleccion
        eliminar_coleccion(documento.nombre_archivo, motor)
        
        limpiar_cache_por_documento(documento.nombre_archivo, motor)
        
        nombre = documento.nombre_archivo
        self.repository.delete(documento)
        mensaje = f"Documento '{nombre}' eliminado correctamente del sistema."
        
        return {
            "mensaje": mensaje,
            "archivos_eliminados": archivos_eliminados,
            "motor": motor,
        }
    
    def _validar_limites_sistema(self):
        total_docs = self.repository.count_total()
        if total_docs >= MAX_DOCUMENTOS:
            raise HTTPException(status_code=400, detail=f"No se pueden agregar más documentos. El límite máximo es de {MAX_DOCUMENTOS} archivos.")
    
    def _validar_archivo(self, file: UploadFile):
        extension = "." + file.filename.split('.')[-1].lower()
        if extension not in EXTENSIONES_PERMITIDAS:
            raise HTTPException(status_code=400, detail=f"El tipo de archivo '{extension}' no está permitido. Usa PDF, DOCX, TXT o MD.")
        
        if file.size and file.size > LIMITE_TAMANO_BYTES:
            raise HTTPException(status_code=400, detail=f"El archivo es demasiado grande. El tamaño máximo permitido es {LIMITE_TAMANO_MB} MB.")
    
    def _validar_motor(self, motor: str):
        if motor != "local":
            raise HTTPException(status_code=400, detail=f"Motor '{motor}' no válido. Solo se soporta 'local'.")
    
    def _get_ruta_destino(self, filename: str, motor: str) -> Path:
        return DOCUMENTS_DIR_LOCAL / filename
    
    async def _guardar_archivo_fisico(self, file: UploadFile, ruta: Path):
        try:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            
            with open(ruta, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            print(f"[DOCS] archivo guardado: {ruta}")
        
        except Exception as e:
            raise DocumentoError(f"Error al guardar archivo físico: {str(e)}")
    
    def _eliminar_archivos_fisicos(self, documento: models.Documento, motor: str) -> List[str]:
        eliminados = []
        
        if documento.ruta_local:
            path_local = Path(documento.ruta_local)
            if path_local.exists():
                path_local.unlink()
                eliminados.append(str(path_local))
        
        return eliminados
