import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models
from app.schemas.schemas import DocumentoOut, DocumentoUploadResponse, AccionGlobalResponse
from app.services.rag_service import (
    procesar_y_guardar_documento,
    eliminar_coleccion_chroma,
    eliminar_todos_los_vectores_chroma,
)
from app.services.cache_service import limpiar_cache, limpiar_cache_por_documento
from app.core.security import get_current_user
from app.core.config import (
    DOCUMENTS_DIR,
    MAX_DOCUMENTOS,
    LIMITE_TAMANO_MB,
    EXTENSIONES_PERMITIDAS,
)

LIMITE_TAMANO_BYTES = int(LIMITE_TAMANO_MB * 1024 * 1024)

router = APIRouter(prefix="/api/documents", tags=["documents"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("", response_model=list[DocumentoOut])
def listar_documentos(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    return db.query(models.Documento).order_by(models.Documento.fecha_subida.desc()).all()


@router.post("/upload", response_model=DocumentoUploadResponse)
async def subir_documento(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    usuario_actual: models.Usuario = Depends(get_current_user),
):
    total_docs = db.query(models.Documento).count()
    ya_existe  = db.query(models.Documento).filter(
        models.Documento.nombre_archivo == file.filename
    ).first()
    if total_docs >= MAX_DOCUMENTOS and not ya_existe:
        raise HTTPException(
            status_code=400,
            detail=f"Limite maximo alcanzado. Solo se permiten {MAX_DOCUMENTOS} documentos.",
        )

    if file.size and file.size > LIMITE_TAMANO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"El archivo supera el limite de {LIMITE_TAMANO_MB} MB.",
        )

    extension = Path(file.filename).suffix.lower()
    if extension not in EXTENSIONES_PERMITIDAS:
        raise HTTPException(
            status_code=400,
            detail=f"Extension no permitida. Usa: {', '.join(EXTENSIONES_PERMITIDAS)}",
        )

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_destino = DOCUMENTS_DIR / file.filename

    try:
        with ruta_destino.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo: {e}")

    try:
        limpiar_cache()
        procesar_y_guardar_documento(str(ruta_destino))
    except Exception as e:
        ruta_destino.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar RAG: {e}")

    try:
        if ya_existe:
            ya_existe.procesado_local = True
            ya_existe.estado_local    = "Procesado Localmente"
            ya_existe.ruta_local      = str(ruta_destino)
        else:
            nuevo = models.Documento(
                nombre_archivo=file.filename,
                subido_por=usuario_actual.username,
                procesado_local=True,
                estado_local="Procesado Localmente",
                ruta_local=str(ruta_destino),
            )
            db.add(nuevo)
        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en base de datos: {e}")

    return DocumentoUploadResponse(ok=True, mensaje="Procesado correctamente.", nombre=file.filename)


@router.get("/{documento_id}/download")
def descargar_documento(
    documento_id: int,
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    doc = db.query(models.Documento).filter(models.Documento.id == documento_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")
    if not doc.ruta_local:
        raise HTTPException(status_code=404, detail="Ruta del documento no disponible.")
    ruta_fisica = Path(doc.ruta_local)
    if not ruta_fisica.exists():
        raise HTTPException(status_code=404, detail="El archivo fisico no existe en el servidor.")
    return FileResponse(path=ruta_fisica, filename=doc.nombre_archivo, media_type="application/octet-stream")


@router.delete("/cache/all", response_model=AccionGlobalResponse)
def limpiar_solo_cache(
    _: models.Usuario = Depends(get_current_user),
):
    try:
        resultado = limpiar_cache()
        return AccionGlobalResponse(ok=True, mensaje=resultado.get("mensaje", "Hecho."))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vectors/all", response_model=AccionGlobalResponse)
def limpiar_vectores_y_cache(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    try:
        limpiar_cache()
        eliminar_todos_los_vectores_chroma()
        for doc in db.query(models.Documento).all():
            doc.procesado_local = False
            doc.estado_local    = "No subido"
        db.commit()
        return AccionGlobalResponse(ok=True, mensaje="Vectores y cache eliminados.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/all", response_model=AccionGlobalResponse)
def procesar_todos_los_documentos(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    documentos = db.query(models.Documento).all()
    if not documentos:
        return AccionGlobalResponse(ok=True, mensaje="No hay documentos para procesar.")
    try:
        limpiar_cache()
        eliminar_todos_los_vectores_chroma()
        procesados = 0
        for doc in documentos:
            if doc.ruta_local:
                ruta = Path(doc.ruta_local)
                if ruta.exists():
                    procesar_y_guardar_documento(str(ruta))
                    doc.procesado_local = True
                    doc.estado_local    = "Procesado Localmente"
                    procesados += 1
        db.commit()
        return AccionGlobalResponse(ok=True, mensaje=f"Sincronizacion completa: {procesados} documentos procesados.")
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
        if DOCUMENTS_DIR.exists():
            for archivo in DOCUMENTS_DIR.iterdir():
                if archivo.is_file():
                    archivo.unlink()
        limpiar_cache()
        eliminar_todos_los_vectores_chroma()
        return AccionGlobalResponse(ok=True, mensaje="Sistema completamente formateado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{documento_id}")
def eliminar_documento(
    documento_id: int,
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    doc = db.query(models.Documento).filter(models.Documento.id == documento_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")

    nombre_archivo   = doc.nombre_archivo
    nombre_coleccion = Path(nombre_archivo).stem

    if doc.ruta_local:
        Path(doc.ruta_local).unlink(missing_ok=True)

    eliminar_coleccion_chroma(nombre_coleccion)
    limpiar_cache_por_documento(nombre_coleccion)

    db.delete(doc)
    db.commit()
    return {"ok": True, "mensaje": f"Documento '{nombre_archivo}' eliminado."}
