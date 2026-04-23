# app/api/routers/usuarios.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.db import models
from app.db.deps import get_db
from app.schemas.schemas import UsuarioCreate, UsuarioUpdate, UsuarioOut
from app.core.security import get_password_hash, get_current_user

router = APIRouter(prefix="/api/usuarios", tags=["usuarios"])

@router.post("/", response_model=UsuarioOut, status_code=status.HTTP_201_CREATED)
def crear_usuario(
    usuario: UsuarioCreate,
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),  
):
    usuario_existente = db.query(models.Usuario).filter(
        models.Usuario.username == usuario.username
    ).first()
    if usuario_existente:
        raise HTTPException(status_code=400, detail="El nombre de usuario o correo ya está registrado.")

    hashed_pw = get_password_hash(usuario.password)
    nuevo_usuario = models.Usuario(
        username=usuario.username,
        hashed_password=hashed_pw,
        rol=usuario.rol
    )
    db.add(nuevo_usuario)
    db.commit()
    db.refresh(nuevo_usuario)
    return nuevo_usuario

@router.get("/", response_model=List[UsuarioOut])
def listar_usuarios(
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),   
):
    return db.query(models.Usuario).order_by(models.Usuario.fecha_creacion.desc()).all()

@router.put("/{usuario_id}", response_model=UsuarioOut)
def actualizar_usuario(
    usuario_id: str,
    usuario_update: UsuarioUpdate,
    db: Session = Depends(get_db),
    _: models.Usuario = Depends(get_current_user),  
):
    db_user = db.query(models.Usuario).filter(models.Usuario.id == usuario_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")

    if usuario_update.username:
        existe = db.query(models.Usuario).filter(
            models.Usuario.username == usuario_update.username,
            models.Usuario.id != usuario_id
        ).first()
        if existe:
            raise HTTPException(status_code=400, detail="El nombre de usuario ya está en uso.")
        db_user.username = usuario_update.username

    if usuario_update.password:
        db_user.hashed_password = get_password_hash(usuario_update.password)

    if usuario_update.rol:
        db_user.rol = usuario_update.rol

    db.commit()
    db.refresh(db_user)
    return db_user

@router.delete("/{usuario_id}")
def eliminar_usuario(
    usuario_id: str,
    db: Session = Depends(get_db),
    usuario_actual: models.Usuario = Depends(get_current_user),  
):
    if usuario_actual.id == usuario_id:
        raise HTTPException(status_code=400, detail="No puedes eliminarte a ti mismo.")

    db_user = db.query(models.Usuario).filter(models.Usuario.id == usuario_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")

    db.delete(db_user)
    db.commit()
    return {"ok": True, "mensaje": "Usuario eliminado correctamente."}