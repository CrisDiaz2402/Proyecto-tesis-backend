# app/repositories/documento_repository.py
"""
Repositorio para operaciones CRUD de documentos.
Implementa Repository Pattern para desacoplar queries de SQLAlchemy.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.db import models
from app.core.exceptions import DocumentoNoEncontradoError


class DocumentoRepository:
    """Repository para gestión de documentos en base de datos."""
    
    def __init__(self, db: Session):
        self.db = db

    def get_all(self) -> List[models.Documento]:
        """Retorna todos los documentos ordenados por fecha de subida descendente."""
        return self.db.query(models.Documento).order_by(
            models.Documento.fecha_subida.desc()
        ).all()

    def get_by_id(self, documento_id: int) -> models.Documento:
        """Busca documento por ID. Lanza excepción si no existe."""
        documento = self.db.query(models.Documento).filter(
            models.Documento.id == documento_id
        ).first()
        
        if not documento:
            raise DocumentoNoEncontradoError(documento_id)
        
        return documento

    def get_by_nombre(self, nombre: str) -> Optional[models.Documento]:
        """Busca documento por nombre de archivo."""
        return self.db.query(models.Documento).filter(
            models.Documento.nombre_archivo == nombre
        ).first()

    def get_by_motor(self, motor: str) -> List[models.Documento]:
        """Retorna documentos que tienen un motor específico procesado."""
        if motor == "local":
            return self.db.query(models.Documento).filter(
                models.Documento.procesado_local == True
            ).all()
        elif motor == "cloud":
            return self.db.query(models.Documento).filter(
                models.Documento.procesado_cloud == True
            ).all()
        else:
            return self.get_all()

    def count_total(self) -> int:
        """Cuenta total de documentos en el sistema."""
        return self.db.query(models.Documento).count()

    def create(self, **kwargs) -> models.Documento:
        """Crea un nuevo documento en la base de datos."""
        nuevo_doc = models.Documento(**kwargs)
        self.db.add(nuevo_doc)
        self.db.commit()
        self.db.refresh(nuevo_doc)
        return nuevo_doc

    def update(self, documento: models.Documento, **kwargs) -> models.Documento:
        """Actualiza campos del documento existente."""
        for field, value in kwargs.items():
            if hasattr(documento, field):
                setattr(documento, field, value)
        
        self.db.commit()
        self.db.refresh(documento)
        return documento

    def delete(self, documento: models.Documento) -> bool:
        """Elimina documento de la base de datos."""
        self.db.delete(documento)
        self.db.commit()
        return True

    def mark_processed(self, documento_id: int, motor: str) -> models.Documento:
        """Marca un documento como procesado para un motor específico."""
        documento = self.get_by_id(documento_id)
        
        if motor == "local":
            documento.procesado_local = True
        elif motor == "cloud":
            documento.procesado_cloud = True
        
        self.db.commit()
        self.db.refresh(documento)
        return documento
