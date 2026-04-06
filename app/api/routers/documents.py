# app/api/routers/documents.py
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.database import SessionLocal
from app.db import models
from app.schemas.schemas import DocumentoOut, DocumentoUploadResponse, AccionGlobalResponse
from app.services.rag_service import procesar_y_guardar_documento, eliminar_coleccion_chroma, eliminar_todos_los_vectores_chroma
from app.services.cache_service import limpiar_cache, limpiar_cache_por_documento
from app.core.security import get_current_user

from app.core.config import (
    DOCUMENTS_DIR_LOCAL, DOCUMENTS_DIR_CLOUD,
    MAX_DOCUMENTOS, LIMITE_TAMANO_MB,
    LIMITE_TAMANO_BYTES, EXTENSIONES_PERMITIDAS
)

router = APIRouter(prefix="/api/documents", tags=["documents"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: lista de motores a procesar según el parámetro recibido
# motor = "local" | "cloud" | "all"
# ─────────────────────────────────────────────────────────────────────────────
def _motores_a_procesar(motor: str) -> list[str]:
    if motor == "all":
        return ["local", "cloud"]
    return [motor]


@router.get("", response_model=list[DocumentoOut])
def listar_documentos(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    """Retorna todos los documentos registrados con sus estados duales."""
    return db.query(models.Documento).order_by(models.Documento.fecha_subida.desc()).all()


@router.post("/upload", response_model=DocumentoUploadResponse)
async def subir_documento(
    file: UploadFile = File(...),
    motor: str = Form("local"),
    db: Session = Depends(get_db),
    usuario_actual: models.Usuario = Depends(get_current_user),
):
    """Sube un archivo, valida límites y lo procesa en la base vectorial elegida."""

    total_docs = db.query(models.Documento).count()
    if total_docs >= MAX_DOCUMENTOS and not db.query(models.Documento).filter(
        models.Documento.nombre_archivo == file.filename
    ).first():
        raise HTTPException(
            status_code=400,
            detail=f"Límite máximo alcanzado. Solo se permiten {MAX_DOCUMENTOS} documentos."
        )

    if file.size and file.size > LIMITE_TAMANO_BYTES:
        raise HTTPException(status_code=400, detail=f"El archivo supera el límite de {LIMITE_TAMANO_MB} MB.")

    extension = Path(file.filename).suffix.lower()
    if extension not in EXTENSIONES_PERMITIDAS:
        raise HTTPException(
            status_code=400,
            detail=f"Extensión no permitida. Usa: {', '.join(EXTENSIONES_PERMITIDAS)}"
        )

    dir_destino = DOCUMENTS_DIR_CLOUD if motor == "cloud" else DOCUMENTS_DIR_LOCAL
    ruta_destino = dir_destino / file.filename

    try:
        with ruta_destino.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        print(f"[DOCS] 💾 Archivo físico guardado ({motor}): {ruta_destino}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar archivo físico: {e}")

    try:
        # Al subir un documento nuevo, invalidar el caché del motor correspondiente
        # (limpia cache_ll+cache_lc si motor=local, cache_cc si motor=cloud)
        limpiar_cache(motor=motor)
        procesar_y_guardar_documento(str(ruta_destino), motor=motor)
        print(f"[DOCS] ✅ Documento vectorizado en {motor}: {file.filename}")
    except Exception as e:
        ruta_destino.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar RAG ({motor}): {e}")

    try:
        existente = db.query(models.Documento).filter(
            models.Documento.nombre_archivo == file.filename
        ).first()

        if existente:
            if motor == "cloud":
                existente.procesado_cloud = True
                existente.estado_cloud = "Procesado en Nube"
                existente.ruta_cloud = str(ruta_destino)
            else:
                existente.procesado_local = True
                existente.estado_local = "Procesado Localmente"
                existente.ruta_local = str(ruta_destino)
        else:
            nuevo = models.Documento(nombre_archivo=file.filename, subido_por=usuario_actual.username)
            if motor == "cloud":
                nuevo.procesado_cloud = True
                nuevo.estado_cloud = "Procesado en Nube"
                nuevo.ruta_cloud = str(ruta_destino)
            else:
                nuevo.procesado_local = True
                nuevo.estado_local = "Procesado Localmente"
                nuevo.ruta_local = str(ruta_destino)
            db.add(nuevo)

        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en base de datos: {e}")

    return DocumentoUploadResponse(ok=True, mensaje=f"Procesado correctamente en {motor}.", nombre=file.filename)


@router.get("/{documento_id}/download")
def descargar_documento(
    documento_id: int,
    motor: str = Query("local"),
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    """Descarga el archivo físico original del motor especificado."""
    doc = db.query(models.Documento).filter(models.Documento.id == documento_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado en BD.")

    ruta_str = doc.ruta_cloud if motor == "cloud" else doc.ruta_local
    if not ruta_str:
        raise HTTPException(status_code=404, detail=f"El documento no ha sido subido al ecosistema {motor}.")

    ruta_fisica = Path(ruta_str)
    if not ruta_fisica.exists():
        raise HTTPException(status_code=404, detail="El archivo físico no existe en el servidor.")

    return FileResponse(path=ruta_fisica, filename=doc.nombre_archivo, media_type='application/octet-stream')


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: Limpiar solo caché
#
# motor = "local"  → limpia cache_ll (local:local) + cache_lc (local:cloud)
# motor = "cloud"  → limpia cache_cc (cloud:cloud)
# motor = "all"    → limpia los 3 cachés
#
# La lógica de expansión vive en cache_service.limpiar_cache().
# ─────────────────────────────────────────────────────────────────────────────
@router.delete("/cache/all", response_model=AccionGlobalResponse)
def limpiar_solo_cache(
    motor: str = Query("local"),        # "local" | "cloud" | "all"
    _: models.Usuario = Depends(get_current_user),
):
    """
    Elimina el caché semántico del motor indicado sin tocar vectores ni documentos.

    - local → borra cache_ll y cache_lc (ambos usan vectores Ollama)
    - cloud → borra cache_cc
    - all   → borra los 3 cachés completos
    """
    try:
        resultado = limpiar_cache(motor=motor)
        etiqueta = {"local": "Local (ll + lc)", "cloud": "Nube (cc)", "all": "Todos (ll + lc + cc)"}.get(motor, motor)
        return AccionGlobalResponse(ok=True, mensaje=f"Caché [{etiqueta}] limpiado. {resultado['mensaje']}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: Limpiar vectores + caché
#
# Borra los vectores de ChromaDB y el caché asociado.
# Además actualiza el estado en la BD (los documentos quedan como "No subido"
# en el motor afectado) para que el frontend refleje la realidad.
#
# motor = "local"  → vectores local + cache_ll + cache_lc, BD actualizada
# motor = "cloud"  → vectores cloud + cache_cc, BD actualizada
# motor = "all"    → todo lo anterior en ambos motores
# ─────────────────────────────────────────────────────────────────────────────
@router.delete("/vectors/all", response_model=AccionGlobalResponse)
def limpiar_vectores_y_cache(
    motor: str = Query("local"),        # "local" | "cloud" | "all"
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    """
    Elimina vectores de ChromaDB y el caché asociado.
    Actualiza el estado de los documentos en BD para reflejar que
    ya no están vectorizados en el/los motor(es) afectado(s).
    """
    try:
        motores = _motores_a_procesar(motor)

        for m in motores:
            # 1. Limpiar caché del motor (incluyendo combis mixtas)
            limpiar_cache(motor=m)
            # 2. Limpiar vectores de ChromaDB
            eliminar_todos_los_vectores_chroma(motor=m)
            # 3. ⚠️ Actualizar BD — sin esto el frontend muestra estado incorrecto
            docs = db.query(models.Documento).all()
            for doc in docs:
                if m == "local":
                    doc.procesado_local = False
                    doc.estado_local = "No subido"
                else:
                    doc.procesado_cloud = False
                    doc.estado_cloud = "No subido"

        db.commit()
        etiqueta = {"local": "Local", "cloud": "Nube", "all": "Local + Nube"}.get(motor, motor)
        return AccionGlobalResponse(
            ok=True,
            mensaje=f"Vectores y caché de [{etiqueta}] eliminados. Estado de documentos actualizado."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process/all", response_model=AccionGlobalResponse)
def procesar_todos_los_documentos(
    motor: str = Query("local"),
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    """Re-vectoriza todos los documentos físicos del motor indicado."""
    documentos = db.query(models.Documento).all()
    if not documentos:
        return AccionGlobalResponse(ok=True, mensaje="No hay documentos para procesar.")

    try:
        limpiar_cache(motor=motor)
        eliminar_todos_los_vectores_chroma(motor=motor)
        procesados = 0
        for doc in documentos:
            ruta_str = doc.ruta_cloud if motor == "cloud" else doc.ruta_local
            if ruta_str:
                ruta = Path(ruta_str)
                if ruta.exists():
                    procesar_y_guardar_documento(str(ruta), motor=motor)
                    if motor == "cloud":
                        doc.procesado_cloud = True
                        doc.estado_cloud = "Procesado en Nube"
                    else:
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
    """Botón rojo: Elimina TODO el contenido de AMBOS ecosistemas."""
    try:
        db.query(models.Documento).delete()
        db.commit()

        # Limpiar ambas carpetas físicas
        for dir_path in [DOCUMENTS_DIR_LOCAL, DOCUMENTS_DIR_CLOUD]:
            if dir_path.exists():
                for archivo in dir_path.iterdir():
                    if archivo.is_file():
                        archivo.unlink()

        # Limpiar los 3 cachés y vectores de ambos motores
        limpiar_cache(motor="all")
        eliminar_todos_los_vectores_chroma(motor="local")
        eliminar_todos_los_vectores_chroma(motor="cloud")

        return AccionGlobalResponse(ok=True, mensaje="Sistema completamente formateado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: Eliminar documento individual por motor
#
# Elimina el archivo físico, los vectores de su colección en ChromaDB,
# y el caché de TODAS las combis que usen ese motor_vectores.
#
# Ejemplo: eliminar doc en "local" borra:
#   - archivo en documents_local/
#   - colección ChromaDB en vector_store_local/
#   - entradas del doc en cache_ll (local:local) y cache_lc (local:cloud)
#
# Si el doc ya no existe en NINGÚN motor → se borra el registro de la BD.
# ─────────────────────────────────────────────────────────────────────────────
@router.delete("/{documento_id}")
def eliminar_documento(
    documento_id: int,
    motor: str = Query("local"),        # "local" | "cloud"
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),
):
    """
    Elimina una versión específica (Local o Cloud) del documento.
    Borra: archivo físico + colección de vectores + caché asociado (todas las combis del motor).
    Si ya no queda en ningún motor, elimina el registro de la BD.
    """
    doc = db.query(models.Documento).filter(models.Documento.id == documento_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")

    nombre_archivo   = doc.nombre_archivo
    nombre_coleccion = Path(nombre_archivo).stem

    # ── 1. Eliminar archivo físico y actualizar estado en BD ──────────────────
    if motor == "cloud":
        if doc.ruta_cloud:
            Path(doc.ruta_cloud).unlink(missing_ok=True)
            print(f"[DOCS] 🗑️ Archivo físico CLOUD eliminado: {doc.ruta_cloud}")
        doc.ruta_cloud      = None
        doc.procesado_cloud = False
        doc.estado_cloud    = "No subido"
    else:
        if doc.ruta_local:
            Path(doc.ruta_local).unlink(missing_ok=True)
            print(f"[DOCS] 🗑️ Archivo físico LOCAL eliminado: {doc.ruta_local}")
        doc.ruta_local      = None
        doc.procesado_local = False
        doc.estado_local    = "No subido"

    # ── 2. Eliminar colección de vectores del motor ───────────────────────────
    eliminar_coleccion_chroma(nombre_coleccion, motor=motor)

    # ── 3. Limpiar caché en TODAS las combis que usan ese motor_vectores ───────
    # motor_vectores es igual al motor del documento (local→local, cloud→cloud).
    # limpiar_cache_por_documento limpia cache_ll+cache_lc si motor_vectores=local,
    # o solo cache_cc si motor_vectores=cloud.
    limpiar_cache_por_documento(nombre_coleccion, motor_vectores=motor)

    # ── 4. Si ya no existe en ningún motor, eliminar registro de la BD ─────────
    if not doc.procesado_local and not doc.procesado_cloud:
        db.delete(doc)
        mensaje_final = f"Documento '{nombre_archivo}' eliminado completamente de todos los sistemas."
    else:
        motor_label = "Nube" if motor == "cloud" else "Local"
        mensaje_final = f"Versión {motor_label} del documento '{nombre_archivo}' eliminada."

    db.commit()
    return {"ok": True, "mensaje": mensaje_final}