# app/api/routers/cache_admin.py
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.security import get_current_user
from app.db import models
from app.core.singletons import RedisClientSingleton
from app.services.cache_service import (
    _hash_pregunta,
    _cache_key,
    _pattern_key,
    _iterar_claves_redis,
)

router = APIRouter(prefix="/api/cache-admin", tags=["cache-admin"])


def _get_redis():
    return RedisClientSingleton().client


class _EntradaResumen(BaseModel):
    clave: str
    pregunta: str
    respuesta_preview: str
    documento_origen: str
    timestamp: str


class _EntradaDetalle(BaseModel):
    clave: str
    pregunta: str
    respuesta: str
    documento_origen: str
    timestamp: str
    motor_vectores: str
    motor_llm: str


class _PatchBody(BaseModel):
    nueva_respuesta: str
    corregida_por: str


@router.get("/entradas", response_model=list[_EntradaResumen])
async def listar_entradas(
    motor_llm: str = Query("local"),
    _: models.Usuario = Depends(get_current_user),
):
    r = _get_redis()
    pattern = _pattern_key("local", motor_llm)
    resultado = []
    for key in _iterar_claves_redis(r, pattern):
        data = r.hgetall(key)
        if not data:
            continue
        resultado.append(_EntradaResumen(
            clave=key,
            pregunta=data.get("pregunta", ""),
            respuesta_preview=data.get("respuesta", "")[:200],
            documento_origen=data.get("documento_origen", ""),
            timestamp=data.get("timestamp", ""),
        ))
    return resultado


@router.get("/entradas/buscar", response_model=list[_EntradaResumen])
async def buscar_entradas(
    q: str = Query(...),
    _: models.Usuario = Depends(get_current_user),
):
    r = _get_redis()
    q_lower = q.lower()
    resultado = []
    for motor_llm in ("local", "cloud"):
        pattern = _pattern_key("local", motor_llm)
        for key in _iterar_claves_redis(r, pattern):
            data = r.hgetall(key)
            if not data:
                continue
            if q_lower in data.get("pregunta", "").lower():
                resultado.append(_EntradaResumen(
                    clave=key,
                    pregunta=data.get("pregunta", ""),
                    respuesta_preview=data.get("respuesta", "")[:200],
                    documento_origen=data.get("documento_origen", ""),
                    timestamp=data.get("timestamp", ""),
                ))
    return resultado


@router.get("/entradas/{cache_key:path}", response_model=_EntradaDetalle)
async def obtener_entrada(
    cache_key: str,
    _: models.Usuario = Depends(get_current_user),
):
    r = _get_redis()
    data = r.hgetall(cache_key)
    if not data:
        raise HTTPException(status_code=404, detail="Clave no encontrada en caché.")
    return _EntradaDetalle(
        clave=cache_key,
        pregunta=data.get("pregunta", ""),
        respuesta=data.get("respuesta", ""),
        documento_origen=data.get("documento_origen", ""),
        timestamp=data.get("timestamp", ""),
        motor_vectores=data.get("motor_vectores", ""),
        motor_llm=data.get("motor_llm", ""),
    )


@router.patch("/entradas/{cache_key:path}", response_model=_EntradaDetalle)
async def corregir_entrada(
    cache_key: str,
    body: _PatchBody,
    _: models.Usuario = Depends(get_current_user),
):
    r = _get_redis()
    data = r.hgetall(cache_key)
    if not data:
        raise HTTPException(status_code=404, detail="Clave no encontrada en caché.")
    r.hset(cache_key, mapping={
        "respuesta":         body.nueva_respuesta,
        "corregida_por":     body.corregida_por,
        "fecha_correccion":  str(time.time()),
        "estado":            "corregida_manualmente",
    })
    data_actualizada = r.hgetall(cache_key)
    return _EntradaDetalle(
        clave=cache_key,
        pregunta=data_actualizada.get("pregunta", ""),
        respuesta=data_actualizada.get("respuesta", ""),
        documento_origen=data_actualizada.get("documento_origen", ""),
        timestamp=data_actualizada.get("timestamp", ""),
        motor_vectores=data_actualizada.get("motor_vectores", ""),
        motor_llm=data_actualizada.get("motor_llm", ""),
    )


@router.delete("/entradas/{cache_key:path}")
async def eliminar_entrada(
    cache_key: str,
    _: models.Usuario = Depends(get_current_user),
):
    r = _get_redis()
    deleted = r.delete(cache_key)
    return {"ok": deleted > 0}
