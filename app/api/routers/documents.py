# app/api/routers/documents.py
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db import models
from app.db.deps import get_db
from app.schemas.schemas import DocumentoOut, DocumentoUploadResponse, AccionGlobalResponse
from app.services.rag_service import procesar_y_guardar_documento, eliminar_coleccion, eliminar_todos_los_vectores
from app.services.cache_service import limpiar_cache, limpiar_cache_por_documento
from app.core.security import get_current_user
from app.services.qdrant_service import eliminar_todos_los_puntos

from app.core.config import (
    DOCUMENTS_DIR_LOCAL,
    MAX_DOCUMENTOS, LIMITE_TAMANO_MB,
    LIMITE_TAMANO_BYTES, EXTENSIONES_PERMITIDAS
)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("", response_model=list[DocumentoOut])
def listar_documentos(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    return db.query(models.Documento).order_by(models.Documento.fecha_subida.desc()).all()


@router.post("/upload", response_model=DocumentoUploadResponse)
async def subir_documento(
    file: UploadFile = File(...),
    motor: str = Form("local"),
    db: Session = Depends(get_db),
    usuario_actual: models.Usuario = Depends(get_current_user),
):
    from app.services.documento_service import DocumentoService
    service = DocumentoService(db)

    try:
        resultado = await service.procesar_subida(file, motor, usuario_actual)
        return DocumentoUploadResponse(ok=True, mensaje=resultado["mensaje"], nombre=file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{documento_id}/download")
def descargar_documento(
    documento_id: int,
    motor: str = Query("local"),
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    doc = db.query(models.Documento).filter(models.Documento.id == documento_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado en BD.")

    ruta_str = doc.ruta_local
    if not ruta_str:
        raise HTTPException(status_code=404, detail="El documento no ha sido subido.")

    ruta_fisica = Path(ruta_str)
    if not ruta_fisica.exists():
        raise HTTPException(status_code=404, detail="El archivo físico no existe en el servidor.")

    return FileResponse(path=ruta_fisica, filename=doc.nombre_archivo, media_type='application/octet-stream')

@router.delete("/cache/all", response_model=AccionGlobalResponse)
def limpiar_solo_cache(
    motor: str = Query("local"),    
    _: models.Usuario = Depends(get_current_user),
):
    if motor not in ("local",):
        raise HTTPException(status_code=400, detail="Solo se soporta motor='local'.")
    try:
        resultado = limpiar_cache(motor=motor)
        return AccionGlobalResponse(ok=True, mensaje=f"Caché [Local (ll + lc)] limpiado. {resultado['mensaje']}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/vectors/all", response_model=AccionGlobalResponse)
def limpiar_vectores_y_cache(
    motor: str = Query("local"),        # El único valor válido es "local"
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    if motor not in ("local",):
        raise HTTPException(status_code=400, detail="Solo se soporta motor='local'.")
    try:
        limpiar_cache(motor=motor)
        eliminar_todos_los_puntos(motor=motor)
        for doc in db.query(models.Documento).all():
            doc.procesado_local = False
            doc.estado_local = "No subido"
        db.commit()

        return AccionGlobalResponse(
            ok=True,
            mensaje="Vectores y caché de [Local] eliminados. Estado de documentos actualizado."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/all", response_model=AccionGlobalResponse)
def procesar_todos_los_documentos(
    motor: str = Query("local"),
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    if motor not in ("local",):
        raise HTTPException(status_code=400, detail="Solo se soporta motor='local'.")
    documentos = db.query(models.Documento).all()
    if not documentos:
        return AccionGlobalResponse(ok=True, mensaje="No hay documentos para procesar.")

    try:
        limpiar_cache(motor=motor)
        eliminar_todos_los_vectores(motor=motor)
        procesados = 0
        for doc in documentos:
            ruta_str = doc.ruta_local
            if ruta_str:
                ruta = Path(ruta_str)
                if ruta.exists():
                    procesar_y_guardar_documento(str(ruta), motor=motor)
                    doc.procesado_local = True
                    doc.estado_local = "Procesado Localmente"
                    procesados += 1
        db.commit()
        return AccionGlobalResponse(
            ok=True,
            mensaje=f"Sincronización masiva ({motor}) completada: {procesados} documento(s) procesados."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/all/confirm", response_model=AccionGlobalResponse)
def eliminar_todos_los_documentos(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    try:
        db.query(models.Documento).delete()
        db.commit()

        if DOCUMENTS_DIR_LOCAL.exists():
            for archivo in DOCUMENTS_DIR_LOCAL.iterdir():
                if archivo.is_file():
                    archivo.unlink()

        limpiar_cache(motor="all")
        eliminar_todos_los_vectores(motor="local")

        return AccionGlobalResponse(ok=True, mensaje="Sistema completamente formateado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{documento_id}")
def eliminar_documento(
    documento_id: int,
    motor: str = Query("local"),     
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    from app.services.documento_service import DocumentoService
    service = DocumentoService(db)

    try:
        resultado = service.eliminar_documento(documento_id, motor)
        return {"ok": True, "mensaje": resultado["mensaje"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))