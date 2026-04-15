# app/core/security.py
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from app.db import models
from app.db.deps import get_db

# ── Esquema de extracción del token desde el header Authorization: Bearer ─────
bearer_scheme = HTTPBearer()

# ─────────────────────────────────────────────────────────────────────────────
# BCRYPT — Hashing y verificación de contraseñas
# ─────────────────────────────────────────────────────────────────────────────

def get_password_hash(password: str) -> str:
    """Encripta la contraseña usando Bcrypt con un salt aleatorio."""
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password=pwd_bytes, salt=salt)
    return hashed_password.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica si la contraseña ingresada coincide con la encriptada."""
    password_byte_enc = plain_password.encode('utf-8')
    hashed_password_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password=password_byte_enc, hashed_password=hashed_password_bytes)

# ─────────────────────────────────────────────────────────────────────────────
# JWT — Creación y validación de tokens
# ─────────────────────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Genera un JWT firmado con los datos del usuario.
    El token expira en ACCESS_TOKEN_EXPIRE_MINUTES minutos por defecto.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCIA — Protege rutas que requieren autenticación
# ─────────────────────────────────────────────────────────────────────────────


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> models.Usuario:
    """
    Dependencia reutilizable para proteger cualquier endpoint.
    Extrae el JWT del header Authorization, lo valida y devuelve el usuario de la BD.

    Uso en un router:
        from app.core.security import get_current_user
        @router.get("/ruta-protegida")
        def ruta(usuario_actual = Depends(get_current_user)):
            ...
    """
    credenciales_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No autenticado o token inválido.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credenciales_exception
    except JWTError:
        raise credenciales_exception

    usuario = db.query(models.Usuario).filter(models.Usuario.id == user_id).first()
    if usuario is None:
        raise credenciales_exception

    return usuario