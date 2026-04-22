# app/api/routers/ws_chat.py
import json
import time
import time as _time
import asyncio
import uuid
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.websockets import WebSocketState

from app.core.security import validate_token_ws
from app.core.constants import TIPOS_WEBSOCKET
from app.core.exceptions import AuthError, RAGError
from app.db.database import SessionLocal
from app.services.rag_service import consultar_base_conocimiento, pipeline_streaming, generar_respuesta_stream_local
from app.services.config_service import obtener_motor_activo
from app.services.intent_service import detectar_intencion
from app.services.nlu_config_service import get_nlu_config
from app.core.event_bus import event_bus

router = APIRouter(prefix="/ws", tags=["websocket"])

def _on_motor_cambiado(data: dict):
    print(f"[WS] Observer: motor cambiado a {data}")

event_bus.suscribir("motor_cambiado", _on_motor_cambiado)

class ConexionInfo:
    def __init__(self, client_id: str, username: str, websocket: WebSocket, ip: str = ""):
        self.client_id = client_id
        self.username = username
        self.websocket = websocket
        self.ip = ip
        self.conectado_en = time.time()

    def to_dict(self) -> dict:
        return {
            "client_id": self.client_id,
            "username": self.username,
            "conectado_en": self.conectado_en,
            "ip": self.ip,
        }


class ConsultaActiva:
    def __init__(self, client_id: str, pregunta: str, motor: str):
        self.id = str(uuid.uuid4())[:8]
        self.client_id = client_id
        self.pregunta = pregunta
        self.motor = motor
        self.inicio = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "client_id": self.client_id,
            "pregunta": self.pregunta,
            "motor": self.motor,
            "inicio": self.inicio,
        }


class ConsultaHistorial:
    def __init__(self, consulta: ConsultaActiva, latencia_ms: int, cache: bool):
        self.id = consulta.id
        self.client_id = consulta.client_id
        self.pregunta = consulta.pregunta
        self.motor = consulta.motor
        self.latencia_ms = latencia_ms
        self.cache = cache
        self.fin = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "client_id": self.client_id,
            "pregunta": self.pregunta,
            "motor": self.motor,
            "latencia_ms": self.latencia_ms,
            "cache": self.cache,
            "fin": self.fin,
        }

class ConnectionManager:

    def __init__(self):
        self._conexiones: Dict[str, ConexionInfo] = {}
        self._consultas_activas: Dict[str, ConsultaActiva] = {}
        self._historial: List[ConsultaHistorial] = []
        self._max_historial = 50
        self._total_consultas = 0
        self._cache_hits = 0
        self._latencias: List[int] = []
        self._max_latencias = 100
        self._monitor_connections: List[WebSocket] = []

    async def conectar_usuario(self, websocket: WebSocket, client_id: str,
                               username: str, ip: str = ""):
        await websocket.accept()
        self._conexiones[client_id] = ConexionInfo(client_id, username, websocket, ip)
        print(f"[WS] {client_id} conectado. Total: {len(self._conexiones)}")
        await self._broadcast_monitor()

    def desconectar_usuario(self, client_id: str):
        self._conexiones.pop(client_id, None)
        huerfanas = [qid for qid, q in self._consultas_activas.items()
                     if q.client_id == client_id]
        for qid in huerfanas:
            self._consultas_activas.pop(qid, None)
        print(f"[WS] {client_id} desconectado. Total: {len(self._conexiones)}")
        asyncio.create_task(self._broadcast_monitor())

    async def send_to_user(self, client_id: str, message: Dict[str, Any]):
        info = self._conexiones.get(client_id)
        if info and info.websocket.client_state == WebSocketState.CONNECTED:
            await info.websocket.send_json(message)

    def iniciar_consulta(self, client_id: str, pregunta: str, motor: str) -> ConsultaActiva:
        c = ConsultaActiva(client_id, pregunta, motor)
        self._consultas_activas[c.id] = c
        asyncio.create_task(self._broadcast_monitor())
        return c

    def finalizar_consulta(self, query_id: str, latencia_ms: int, cache: bool):
        consulta = self._consultas_activas.pop(query_id, None)
        if consulta:
            self._total_consultas += 1
            if cache:
                self._cache_hits += 1
            self._latencias.append(latencia_ms)
            if len(self._latencias) > self._max_latencias:
                self._latencias.pop(0)
            registro = ConsultaHistorial(consulta, latencia_ms, cache)
            self._historial.insert(0, registro)
            if len(self._historial) > self._max_historial:
                self._historial.pop()
        asyncio.create_task(self._broadcast_monitor())

    async def conectar_monitor(self, websocket: WebSocket):
        await websocket.accept()
        self._monitor_connections.append(websocket)
        await self._send_monitor_state(websocket)

    def desconectar_monitor(self, websocket: WebSocket):
        if websocket in self._monitor_connections:
            self._monitor_connections.remove(websocket)

    async def _send_monitor_state(self, ws: WebSocket):
        if ws.client_state != WebSocketState.CONNECTED:
            return
        avg = int(sum(self._latencias) / len(self._latencias)) if self._latencias else 0
        payload = {
            "tipo": "estado_completo",
            "data": {
                "conexiones": [c.to_dict() for c in self._conexiones.values()],
                "total_conexiones": len(self._conexiones),
                "activas": [q.to_dict() for q in self._consultas_activas.values()],
                "total_activas": len(self._consultas_activas),
                "historial": [h.to_dict() for h in self._historial],
                "latencia_avg_ms": avg,
                "cache_hits": self._cache_hits,
                "total_consultas": self._total_consultas,
                "timestamp": time.time(),
            },
        }
        try:
            await ws.send_json(payload)
        except Exception:
            self.desconectar_monitor(ws)

    async def _broadcast_monitor(self):
        muertos = []
        for ws in list(self._monitor_connections):
            try:
                await self._send_monitor_state(ws)
            except Exception:
                muertos.append(ws)
        for ws in muertos:
            self.desconectar_monitor(ws)

manager = ConnectionManager()


def _check_rate_limit(client_id: str, max_por_minuto: int = 60) -> bool:
    """
    Límite de consultas por minuto por cliente.
    En producción puedes bajar este valor a 15.
    """
    try:
        from app.core.singletons import RedisClientSingleton
        r = RedisClientSingleton().client
        key = f"rate_limit:ws:{client_id}"
        ahora = _time.time()
        ventana = 60

        r.zremrangebyscore(key, 0, ahora - ventana)
        if r.zcard(key) >= max_por_minuto:
            return False
        r.zadd(key, {str(ahora): ahora})
        r.expire(key, ventana + 5)
        return True
    except Exception:
        return True  # fail-open: si Redis falla, no bloquear al usuario


# ── AVATAR: 100% público, sin autenticación ───────────────────────────────────
@router.websocket("/chat")
async def chat_websocket(
    websocket: WebSocket,
):
    """
    Endpoint WebSocket del avatar RAG EPN.
    Completamente público — no requiere token ni autenticación.
    Cada conexión recibe un client_id único anónimo para tracking interno.
    """
    anon_id = str(uuid.uuid4())[:8]
    client_id = f"anon_{anon_id}"
    username = "Visitante"

    ip = ""
    if websocket.client:
        ip = str(websocket.client.host)

    await manager.conectar_usuario(websocket, client_id, username, ip)

    try:
        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["estado"],
            "mensaje": f"Conectado al Avatar RAG EPN (motor: {obtener_motor_activo()}). Realiza tu consulta académica.",
            "timestamp": time.time(),
        })

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("tipo") == "pregunta":
                await _procesar_pregunta(client_id, message.get("pregunta", ""))

            elif message.get("tipo") == "ping":
                await manager.send_to_user(client_id, {
                    "tipo": "pong",
                    "timestamp": time.time(),
                })

    except WebSocketDisconnect:
        manager.desconectar_usuario(client_id)
    except Exception as e:
        print(f"[WS ERROR] {client_id}: {e}")
        try:
            await manager.send_to_user(client_id, {
                "tipo": TIPOS_WEBSOCKET["error"],
                "mensaje": "Error interno del servidor",
            })
        except Exception:
            pass
        manager.desconectar_usuario(client_id)


# ── MONITOR: requiere token de admin (sin cambios) ────────────────────────────
@router.websocket("/monitor")
async def monitor_websocket(
    websocket: WebSocket,
    token: str = Query(..., description="JWT token de admin"),
):
    db = SessionLocal()
    try:
        usuario = validate_token_ws(token, db)

    except (HTTPException, Exception):
        await websocket.close(code=4401, reason="Token inválido")
        return
    finally:
        db.close()

    await manager.conectar_monitor(websocket)

    try:
        while True:
            await asyncio.sleep(1)
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            await manager._broadcast_monitor()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS MONITOR ERROR] {e}")
    finally:
        manager.desconectar_monitor(websocket)


async def _procesar_pregunta(client_id: str, pregunta: str):

    if not pregunta.strip():
        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["error"],
            "mensaje": "Pregunta vacía. Por favor escribe tu consulta académica.",
        })
        return

    if not _check_rate_limit(client_id, max_por_minuto=60):
        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["error"],
            "mensaje": "Demasiadas consultas seguidas. Espera un momento antes de continuar.",
        })
        return

    nlu_cfg = get_nlu_config()
    intencion_result = detectar_intencion(
        pregunta,
        palabras_saludo=nlu_cfg["palabras_saludo"],
        frases_despedida=nlu_cfg["frases_despedida"],
        frases_agradecimiento=nlu_cfg["frases_agradecimiento"],
    )
    intencion = intencion_result["intencion"]

    _MENSAJES_INTENCION = {
        "saludo": nlu_cfg["mensaje_saludo"],
        "despedida": nlu_cfg["mensaje_despedida"],
        "agradecimiento": nlu_cfg["mensaje_agradecimiento"],
        "fuera_de_tema": nlu_cfg["mensaje_fuera_de_tema"],
    }

    if intencion in _MENSAJES_INTENCION:
        await manager.send_to_user(client_id, {
            "tipo": "respuesta",
            "pregunta": pregunta,
            "respuesta": _MENSAJES_INTENCION[intencion],
            "motor": "nlu",
            "intencion": intencion,
            "timestamp": time.time(),
        })
        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["final"],
            "mensaje": "Consulta completada.",
        })
        return

    # ── Pipeline RAG (intención = consulta_academica) ───────────────────
    motor_actual = obtener_motor_activo()
    consulta = manager.iniciar_consulta(client_id, pregunta, motor_actual)
    query_id = consulta.id
    t_inicio = time.time()

    try:
        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["estado"],
            "mensaje": "Buscando en base de conocimiento...",
        })

        motor_vectores, motor_llm = (
            motor_actual.split(":", 1) if ":" in motor_actual
            else (motor_actual, motor_actual)
        )

        if motor_llm == "local":
            from app.services.rag_service import (
                _generar_embedding, _buscar_chunks_similares,
                _get_retrieval_params, _get_num_tokens,
            )
            from app.services.rag_params_service import get_params
            from app.core.prompts import PROMPT_PRINCIPAL_DEFAULT
            from app.services.cache_service import buscar_en_cache, guardar_en_cache

            cached = buscar_en_cache(pregunta, motor_vectores=motor_vectores, motor_llm=motor_llm)
            if cached:
                latencia = int((time.time() - t_inicio) * 1000)
                await manager.send_to_user(client_id, {
                    "tipo": "respuesta",
                    "pregunta": pregunta,
                    "respuesta": cached,
                    "motor": motor_actual,
                    "cached": True,
                    "timestamp": time.time(),
                })
                manager.finalizar_consulta(query_id, latencia, cache=True)
            else:
                loop = asyncio.get_event_loop()
                query_embedding = await loop.run_in_executor(
                    None, lambda: _generar_embedding(pregunta, motor_vectores)
                )
                k_retrieval, umbral = _get_retrieval_params(motor_vectores)
                resultados = _buscar_chunks_similares(
                    query_embedding, motor_vectores, k_retrieval, umbral
                )

                if not resultados:
                    latencia = int((time.time() - t_inicio) * 1000)
                    await manager.send_to_user(client_id, {
                        "tipo": "respuesta",
                        "pregunta": pregunta,
                        "respuesta": nlu_cfg["mensaje_sin_resultados"],
                        "motor": motor_actual,
                        "timestamp": time.time(),
                    })
                    manager.finalizar_consulta(query_id, latencia, cache=False)
                else:
                    contexto = "\n\n---\n\n".join([r["contenido"] for r in resultados])
                    params = get_params()
                    plantilla = params.get("prompt_principal") or PROMPT_PRINCIPAL_DEFAULT
                    prompt_ia = plantilla.format(contexto=contexto, pregunta=pregunta)
                    num_tokens = _get_num_tokens(motor_llm)

                    await manager.send_to_user(client_id, {
                        "tipo": TIPOS_WEBSOCKET["estado"],
                        "mensaje": f"Generando respuesta (motor: {motor_actual})...",
                    })

                    respuesta_completa = ""
                    async for token in generar_respuesta_stream_local(prompt_ia, num_tokens):
                        respuesta_completa += token
                        await manager.send_to_user(client_id, {
                            "tipo": TIPOS_WEBSOCKET["token"],
                            "token": token,
                        })

                    coleccion = resultados[0]["coleccion"] if resultados else "desconocido"
                    guardar_en_cache(
                        pregunta, respuesta_completa,
                        documento_origen=coleccion,
                        motor_vectores=motor_vectores,
                        motor_llm=motor_llm,
                    )

                    latencia = int((time.time() - t_inicio) * 1000)
                    await manager.send_to_user(client_id, {
                        "tipo": "respuesta",
                        "pregunta": pregunta,
                        "respuesta": respuesta_completa,
                        "motor": motor_actual,
                        "timestamp": time.time(),
                    })
                    manager.finalizar_consulta(query_id, latencia, cache=False)

        else:
            await manager.send_to_user(client_id, {
                "tipo": TIPOS_WEBSOCKET["estado"],
                "mensaje": f"Generando respuesta (motor: {motor_actual})...",
            })

            loop = asyncio.get_event_loop()
            respuesta = await loop.run_in_executor(
                None,
                lambda: consultar_base_conocimiento(pregunta, motor=motor_actual),
            )

            latencia = int((time.time() - t_inicio) * 1000)
            await manager.send_to_user(client_id, {
                "tipo": "respuesta",
                "pregunta": pregunta,
                "respuesta": respuesta,
                "motor": motor_actual,
                "timestamp": time.time(),
            })
            manager.finalizar_consulta(query_id, latencia, cache=False)

        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["final"],
            "mensaje": "Consulta completada.",
        })

    except RAGError as e:
        latencia = int((time.time() - t_inicio) * 1000)
        manager.finalizar_consulta(query_id, latencia, cache=False)
        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["error"],
            "mensaje": f"Error RAG: {e.message}",
            "details": e.details,
        })

    except Exception as e:
        latencia = int((time.time() - t_inicio) * 1000)
        manager.finalizar_consulta(query_id, latencia, cache=False)
        print(f"[WS RAG ERROR] {client_id}: {e}")
        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["error"],
            "mensaje": "Error procesando consulta. Inténtalo nuevamente.",
        })