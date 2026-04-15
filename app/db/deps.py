# app/db/deps.py
"""
Dependencias de base de datos centralizadas.
Elimina la duplicación de get_db() en múltiples routers.
"""

from app.db.database import SessionLocal


def get_db():
    """
    Generador de sesiones de base de datos para FastAPI dependency injection.
    Se importa desde aquí en lugar de duplicar en cada router.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()