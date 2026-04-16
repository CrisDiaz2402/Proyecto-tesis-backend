# app/api/routers/ws_chat.py
"""
WebSocket endpoints:
  - /ws/chat    → chat RAG en tiempo real (usuarios)
  - /ws/monitor → push de estado al panel admin cada segundo
"""

import json
import time
import asyncio
import uuid
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.websockets import WebSocketState

from app.core.security import validate_token_ws
from app.core.constants import TIPOS_WEBSOCKET
from app.core.exceptions import AuthError, RAGError
from app.db.database import SessionLocal
from app.services.rag_service import consultar_base_conocimiento
from app.services.config_service import obtener_motor_activo
from app.services.vllm_service import generar_respuesta_stream_local

router = APIRouter(prefix="/ws", tags=["websocket"])


# ─────────────────────────────────────────────────────────────────────────────
# MODELOS DE DATOS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

class ConexionInfo:
    """Representa una sesión WebSocket abierta."""
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
    """Representa una consulta RAG en curso."""
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
    """Registro de una consulta ya finalizada."""
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


# ─────────────────────────────────────────────────────────────────────────────
# MANAGER CENTRAL
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """
    Gestiona:
    - Conexiones WS de usuarios (chat)
    - Consultas activas (en procesamiento)
    - Historial de consultas
    - Conexiones WS del monitor (admin)
    """

    def __init__(self):
        # Sesiones de chat activas: client_id → ConexionInfo
        self._conexiones: Dict[str, ConexionInfo] = {}
        # Consultas en proceso: query_id → ConsultaActiva
        self._consultas_activas: Dict[str, ConsultaActiva] = {}
        # Historial (últimas 50)
        self._historial: List[ConsultaHistorial] = []
        self._max_historial = 50
        # Contadores
        self._total_consultas = 0
        self._cache_hits = 0
        self._latencias: List[int] = []
        self._max_latencias = 100
        # Conexiones del monitor admin
        self._monitor_connections: List[WebSocket] = []

    # ── Conexiones de chat ────────────────────────────────────────────────────

    async def conectar_usuario(self, websocket: WebSocket, client_id: str,
                               username: str, ip: str = ""):
        await websocket.accept()
        self._conexiones[client_id] = ConexionInfo(client_id, username, websocket, ip)
        print(f"[WS] {client_id} conectado. Total: {len(self._conexiones)}")
        await self._broadcast_monitor()

    def desconectar_usuario(self, client_id: str):
        self._conexiones.pop(client_id, None)
        # Limpiar cualquier consulta activa del usuario que pueda haber quedado
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

    # ── Consultas activas ─────────────────────────────────────────────────────

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

    # ── Monitor admin ─────────────────────────────────────────────────────────

    async def conectar_monitor(self, websocket: WebSocket):
        await websocket.accept()
        self._monitor_connections.append(websocket)
        # Enviar estado inmediato al conectar
        await self._send_monitor_state(websocket)

    def desconectar_monitor(self, websocket: WebSocket):
        if websocket in self._monitor_connections:
            self._monitor_connections.remove(websocket)

    async def _send_monitor_state(self, ws: WebSocket):
        """Envía snapshot completo del estado a un websocket de monitor."""
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
        """Envía estado actualizado a todos los admins conectados al monitor."""
        muertos = []
        for ws in list(self._monitor_connections):
            try:
                await self._send_monitor_state(ws)
            except Exception:
                muertos.append(ws)
        for ws in muertos:
            self.desconectar_monitor(ws)


# Instancia global (singleton del módulo)
manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: /ws/chat
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/chat")
async def chat_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None, description="JWT token (opcional — el avatar es público)"),
):
    """
    WebSocket de chat público para el avatar.
    - Sin token: se conecta como visitante anónimo (acceso libre al avatar).
    - Con token válido: se identifica al usuario autenticado.
    - Con token inválido/corrupto: se rechaza la conexión (evita tokens manipulados).
    """
    client_id: str
    username: str

    if token:
        # Hay token → intentar autenticar; si falla, rechazar (no aceptar tokens corruptos)
        db = SessionLocal()
        try:
            usuario = validate_token_ws(token, db)
            client_id = f"user_{usuario.id}"
            username = getattr(usuario, "username", client_id)
        except (HTTPException, Exception):
            await websocket.close(code=4401, reason="Token inválido")
            return
        finally:
            db.close()
    else:
        # Sin token → visitante anónimo (flujo normal del avatar público)
        anon_id = str(uuid.uuid4())[:8]
        client_id = f"anon_{anon_id}"
        username = "Visitante"

    # Extraer IP del cliente
    ip = ""
    if websocket.client:
        ip = str(websocket.client.host)

    await manager.conectar_usuario(websocket, client_id, username, ip)

    try:
        # Mensaje de bienvenida
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
    except AuthError:
        await websocket.close(code=4401, reason="Token inválido")
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


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT: /ws/monitor
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/monitor")
async def monitor_websocket(
    websocket: WebSocket,
    token: str = Query(..., description="JWT token de admin"),
):
    """
    WebSocket del panel de administración.
    Recibe push automático cada vez que cambia el estado del sistema.
    También hace push periódico cada 1s para actualizar tiempos transcurridos.
    """
    db = SessionLocal()
    try:
        usuario = validate_token_ws(token, db)
        # Opcional: verificar que sea admin
        # if getattr(usuario, 'rol', '') != 'Admin':
        #     await websocket.close(code=4403, reason="No autorizado")
        #     return
    except (HTTPException, Exception):
        await websocket.close(code=4401, reason="Token inválido")
        return
    finally:
        db.close()

    await manager.conectar_monitor(websocket)

    try:
        # Push periódico cada 1s para que los timers en el frontend se actualicen
        while True:
            await asyncio.sleep(1)
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            await manager._send_monitor_state(websocket)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS MONITOR ERROR] {e}")
    finally:
        manager.desconectar_monitor(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO DE PREGUNTAS
# ─────────────────────────────────────────────────────────────────────────────

async def _procesar_pregunta(client_id: str, pregunta: str):
    """Procesa una pregunta RAG y envía respuesta vía WebSocket con tracking completo."""

    if not pregunta.strip():
        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["error"],
            "mensaje": "Pregunta vacía. Por favor escribe tu consulta académica.",
        })
        return

    motor_actual = obtener_motor_activo()
    consulta = manager.iniciar_consulta(client_id, pregunta, motor_actual)
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
            # ── Streaming token a token con vLLM ─────────────────────────────
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
                manager.finalizar_consulta(consulta.id, latencia, cache=True)
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
                        "respuesta": "Lo siento, esa información no existe en mi base de datos oficial.",
                        "motor": motor_actual,
                        "timestamp": time.time(),
                    })
                    manager.finalizar_consulta(consulta.id, latencia, cache=False)
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
                    manager.finalizar_consulta(consulta.id, latencia, cache=False)

        else:
            # ── Cloud: RAG síncrono en thread pool ───────────────────────────
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
            manager.finalizar_consulta(consulta.id, latencia, cache=False)

        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["final"],
            "mensaje": "Consulta completada.",
        })

    except RAGError as e:
        latencia = int((time.time() - t_inicio) * 1000)
        manager.finalizar_consulta(consulta.id, latencia, cache=False)
        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["error"],
            "mensaje": f"Error RAG: {e.message}",
            "details": e.details,
        })

    except Exception as e:
        latencia = int((time.time() - t_inicio) * 1000)
        manager.finalizar_consulta(consulta.id, latencia, cache=False)
        print(f"[WS RAG ERROR] {client_id}: {e}")
        await manager.send_to_user(client_id, {
            "tipo": TIPOS_WEBSOCKET["error"],
            "mensaje": "Error procesando consulta. Inténtalo nuevamente.",
        })