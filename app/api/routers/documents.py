# app/api/routers/documents.py
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models
from app.schemas.schemas import DocumentoOut, DocumentoUploadResponse, AccionGlobalResponse
from app.services.rag_service import procesar_y_guardar_documento, eliminar_coleccion_chroma, eliminar_todos_los_vectores_chroma
from app.services.cache_service import limpiar_cache, limpiar_cache_por_documento

# Importamos TODAS las reglas de negocio desde la fuente de verdad
from app.core.config import (
    DOCUMENTS_DIR, MAX_DOCUMENTOS, LIMITE_TAMANO_MB, 
    LIMITE_TAMANO_BYTES, EXTENSIONES_PERMITIDAS
)

router = APIRouter(prefix="/api/documents", tags=["documents"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("", response_model=list[DocumentoOut])
def listar_documentos(db: Session = Depends(get_db)):
    """Retorna todos los documentos registrados."""
    return db.query(models.Documento).order_by(models.Documento.fecha_subida.desc()).all()

@router.post("/upload", response_model=DocumentoUploadResponse)
async def subir_documento(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Sube un archivo, valida límites y lo procesa en la base vectorial."""
    
    total_docs = db.query(models.Documento).count()
    if total_docs >= MAX_DOCUMENTOS and not db.query(models.Documento).filter(models.Documento.nombre_archivo == file.filename).first():
        raise HTTPException(status_code=400, detail=f"Límite máximo alcanzado. Solo se permiten {MAX_DOCUMENTOS} documentos.")

    if file.size and file.size > LIMITE_TAMANO_BYTES:
        raise HTTPException(status_code=400, detail=f"El archivo supera el límite de {LIMITE_TAMANO_MB} MB.")

    extension = Path(file.filename).suffix.lower()
    if extension not in EXTENSIONES_PERMITIDAS:
        raise HTTPException(status_code=400, detail=f"Extensión no permitida. Usa: {', '.join(EXTENSIONES_PERMITIDAS)}")

    ruta_destino = DOCUMENTS_DIR / file.filename
    try:
        with ruta_destino.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"[DOCS] 💾 Archivo físico guardado: {ruta_destino}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo físico: {e}")

    try:
        limpiar_cache()
        procesar_y_guardar_documento(str(ruta_destino))
        print(f"[DOCS] ✅ Documento vectorizado: {file.filename}")
    except Exception as e:
        ruta_destino.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar RAG: {e}")

    try:
        existente = db.query(models.Documento).filter(models.Documento.nombre_archivo == file.filename).first()
        if existente:
            existente.estado = "Procesado en ChromaDB"
            existente.ruta_archivo = str(ruta_destino)
        else:
            nuevo = models.Documento(
                nombre_archivo=file.filename,
                ruta_archivo=str(ruta_destino),
                subido_por="Admin",
                estado="Procesado en ChromaDB"
            )
            db.add(nuevo)
        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en base de datos: {e}")

    return DocumentoUploadResponse(ok=True, mensaje="Documento procesado correctamente.", nombre=file.filename)

@router.get("/{documento_id}/download")
def descargar_documento(documento_id: int, db: Session = Depends(get_db)):
    """Descarga el archivo físico original."""
    doc = db.query(models.Documento).filter(models.Documento.id == documento_id).first()
    if not doc or not doc.ruta_archivo:
        raise HTTPException(status_code=404, detail="Documento no encontrado en BD.")
    
    ruta_fisica = Path(doc.ruta_archivo)
    if not ruta_fisica.exists():
        raise HTTPException(status_code=404, detail="El archivo físico no existe en el servidor.")
        
    return FileResponse(path=ruta_fisica, filename=doc.nombre_archivo, media_type='application/octet-stream')

@router.delete("/cache/all", response_model=AccionGlobalResponse)
def limpiar_solo_cache():
    try:
        limpiar_cache()
        return AccionGlobalResponse(ok=True, mensaje="Caché semántico limpiado correctamente.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/vectors/all", response_model=AccionGlobalResponse)
def limpiar_vectores_y_cache():
    try:
        limpiar_cache()
        eliminar_todos_los_vectores_chroma()
        return AccionGlobalResponse(ok=True, mensaje="Vectores y caché eliminados por completo.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/all", response_model=AccionGlobalResponse)
def procesar_todos_los_documentos(db: Session = Depends(get_db)):
    documentos = db.query(models.Documento).all()
    if not documentos:
        return AccionGlobalResponse(ok=True, mensaje="No hay documentos para procesar.")
    
    try:
        limpiar_cache()
        eliminar_todos_los_vectores_chroma()
        procesados = 0
        for doc in documentos:
            ruta = DOCUMENTS_DIR / doc.nombre_archivo
            if ruta.exists():
                procesar_y_guardar_documento(str(ruta))
                doc.estado = "Procesado en ChromaDB"
                procesados += 1
        db.commit()
        return AccionGlobalResponse(ok=True, mensaje=f"Sincronización masiva completada: {procesados} documentos procesados.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/all/confirm", response_model=AccionGlobalResponse)
def eliminar_todos_los_documentos(db: Session = Depends(get_db)):
    """Botón rojo: Elimina TODO el contenido del sistema."""
    try:
        db.query(models.Documento).delete()
        db.commit()
        
        if DOCUMENTS_DIR.exists():
            for archivo in DOCUMENTS_DIR.iterdir():
                if archivo.is_file():
                    archivo.unlink()
        
        limpiar_cache()
        eliminar_todos_los_vectores_chroma()
        return AccionGlobalResponse(ok=True, mensaje="Sistema formateado. Todos los datos fueron destruidos.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{documento_id}")
def eliminar_documento(documento_id: int, db: Session = Depends(get_db)):
    """Elimina un documento de PostgreSQL, disco, vectores y caché."""
    doc = db.query(models.Documento).filter(models.Documento.id == documento_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")

    nombre_archivo = doc.nombre_archivo
    nombre_coleccion = Path(nombre_archivo).stem

    if doc.ruta_archivo:
        ruta = Path(doc.ruta_archivo)
        ruta.unlink(missing_ok=True)
        print(f"[DOCS] 🗑️ Archivo físico eliminado: {ruta}")

    db.delete(doc)
    db.commit()
    
    eliminar_coleccion_chroma(nombre_coleccion)
    limpiar_cache_por_documento(nombre_coleccion)
    print(f"[DOCS] 🗑️ Registro y conocimiento eliminados: '{nombre_archivo}'")

    return {"ok": True, "mensaje": f"Documento '{nombre_archivo}' eliminado correctamente."}