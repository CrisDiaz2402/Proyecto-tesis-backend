# app/api/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import models
from app.db.deps import get_db
from app.schemas.auth_schemas import LoginRequest, TokenResponse
from app.core.security import verify_password, create_access_token

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/login", response_model=TokenResponse)
def login(credentials: LoginRequest, db: Session = Depends(get_db)):
    """Autentica al usuario y retorna un JWT si las credenciales son correctas."""

    # 1. Buscar usuario en la base de datos
    usuario = db.query(models.Usuario).filter(
        models.Usuario.username == credentials.username
    ).first()

    # 2. Verificar existencia y contraseña (misma respuesta para ambos casos → evita enumeración)
    if not usuario or not verify_password(credentials.password, usuario.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Generar el token con el id y username del usuario como payload
    access_token = create_access_token(
        data={"sub": usuario.id, "username": usuario.username, "rol": usuario.rol}
    )

    return TokenResponse(
        access_token=access_token,
        username=usuario.username,
        rol=usuario.rol,
    )