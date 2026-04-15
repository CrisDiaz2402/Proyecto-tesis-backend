# app/schemas/evaluacion_schemas.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# REQUEST — Lo que el frontend envía al endpoint
# ─────────────────────────────────────────────────────────────────────────────

class CasoEvaluacion(BaseModel):
    id:                str
    grupo:             str
    tipo:              str                        # contiene | no_contiene | corrige | no_alucina
    pregunta:          str
    claves:            list[str]
    claves_prohibidas: list[str]  = []
    descripcion:       Optional[str] = None
    habilitado:        bool       = True


class EjecucionRequest(BaseModel):
    experimento: str = "baseline"
    casos:       list[CasoEvaluacion]


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE — Lo que el backend devuelve al frontend
# ─────────────────────────────────────────────────────────────────────────────

class ResultadoCaso(BaseModel):
    id:          str
    grupo:       str
    tipo:        str
    pregunta:    str
    respuesta:   str
    latencia_ms: int
    score:       float                            # 0.0 | 0.5 | 1.0
    veredicto:   str                              # PASS | PARCIAL | FAIL
    detalle:     str
    descripcion: Optional[str] = None


class ResumenGrupo(BaseModel):
    promedio: float
    pass_:    int
    parcial:  int
    fail:     int
    total:    int

    class Config:
        populate_by_name = True


class ConteoGlobal(BaseModel):
    pass_:   int
    parcial: int
    fail:    int
    total:   int

    class Config:
        populate_by_name = True


class ResultadoEvaluacion(BaseModel):
    experimento:        str
    motor:              str
    timestamp:          str
    duracion_total_seg: float
    resultados:         list[ResultadoCaso]
    resumen_por_grupo:  dict[str, dict]
    score_global:       float
    conteo_global:      dict


# ─────────────────────────────────────────────────────────────────────────────
# SSE — Evento de progreso emitido caso a caso durante el streaming
# ─────────────────────────────────────────────────────────────────────────────

class ProgresoEvaluacion(BaseModel):
    """
    Evento SSE emitido por /api/evaluacion/ejecutar-stream tras procesar cada caso.

    Campos:
      tipo          → "progreso" | "completado" | "error"
      caso_actual   → número del caso que acaba de terminar (1-based)
      total_casos   → total de casos habilitados en la sesión
      porcentaje    → entero 0-100 calculado como round(caso_actual/total_casos*100)
      resultado     → ResultadoCaso completo del caso recién evaluado (solo en "progreso")
      reporte_final → ResultadoEvaluacion completo (solo en "completado")
      mensaje_error → descripción del error (solo en "error")
    """
    tipo:           str                              # "progreso" | "completado" | "error"
    caso_actual:    int       = 0
    total_casos:    int       = 0
    porcentaje:     int       = 0
    resultado:      Optional[ResultadoCaso]      = None
    reporte_final:  Optional[ResultadoEvaluacion] = None
    mensaje_error:  Optional[str]                = None