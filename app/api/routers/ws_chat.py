# app/api/routers/ws_chat.py
"""
WebSocket endpoint para chat en tiempo real.
PREPARADO para migración vLLM + Qdrant - actualmente funciona con stack legacy.
"""

import json
import asyncio
from typing import Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.websockets import WebSocketState

from app.core.security import validate_token_ws  # TODO: implementar
from app.core.constants import TIPOS_WEBSOCKET
from app.core.exceptions import AuthError, RAGError
from app.services.rag_service import consultar_base_conocimiento  # Legacy
from app.services.config_service import obtener_motor_activo

router = APIRouter(prefix="/ws", tags=["websocket"])


# ─────────────────────────────────────────────────────────────────────────────
# GESTIÓN DE CONEXIONES WEBSOCKET
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """Gestor de conexiones WebSocket activas."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Acepta nueva conexión WebSocket."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"[WS] Cliente {client_id} conectado. Total: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        """Remueve conexión del gestor."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"[WS] Cliente {client_id} desconectado. Total: {len(self.active_connections)}")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Envía mensaje a cliente específico."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)


manager = ConnectionManager()


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.websocket("/chat")
async def chat_websocket(
    websocket: WebSocket, 
    token: str = Query(..., description="JWT token para autenticación")
):
    """
    WebSocket para chat RAG en tiempo real.
    
    FLUJO:
    1. Validar JWT token
    2. Aceptar conexión WebSocket
    3. Escuchar mensajes del cliente
    4. Para cada pregunta:
       a) Enviar status "procesando"
       b) Ejecutar RAG (actualmente síncrono, futuro: async streaming)
       c) Enviar respuesta completa
       d) Enviar status "completado"
    
    FUTURO vLLM Stack:
    - Streaming real token por token
    - Async nativo (sin run_in_executor)
    - Múltiples modelos simultáneos
    - Cache Redis distribuido
    """
    client_id = f"user_{id(websocket)}"  # TODO: usar user ID real del JWT
    
    try:
        # TODO: Validar JWT token
        # usuario = validate_token_ws(token)
        # client_id = f"user_{usuario.id}"
        
        await manager.connect(websocket, client_id)
        
        # Enviar mensaje de bienvenida
        await manager.send_message(client_id, {
            "tipo": TIPOS_WEBSOCKET["estado"],
            "mensaje": "Conectado al Avatar RAG EPN. Realiza tu consulta académica.",
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Loop principal de mensajes
        while True:
            # Recibir mensaje del cliente
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("tipo") == "pregunta":
                await procesar_pregunta_ws(client_id, message.get("pregunta", ""))
            
            elif message.get("tipo") == "ping":
                await manager.send_message(client_id, {"tipo": "pong", "timestamp": asyncio.get_event_loop().time()})
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except AuthError as e:
        await websocket.close(code=4401, reason="Token inválido")
    except Exception as e:
        print(f"[WS ERROR] {client_id}: {e}")
        await manager.send_message(client_id, {
            "tipo": TIPOS_WEBSOCKET["error"],
            "mensaje": "Error interno del servidor",
            "details": str(e) if __debug__ else None
        })
        manager.disconnect(client_id)


async def procesar_pregunta_ws(client_id: str, pregunta: str):
    """
    Procesa una pregunta RAG y envía respuesta via WebSocket.
    
    LEGACY: Usa consultar_base_conocimiento síncrona.
    FUTURO: Async streaming con vLLM.
    """
    if not pregunta.strip():
        await manager.send_message(client_id, {
            "tipo": TIPOS_WEBSOCKET["error"],
            "mensaje": "Pregunta vacía. Por favor escribe tu consulta académica."
        })
        return
    
    try:
        # Status: procesando
        await manager.send_message(client_id, {
            "tipo": TIPOS_WEBSOCKET["estado"],
            "mensaje": "Buscando en base de conocimiento...",
        })
        
        # Obtener configuración actual
        motor_actual = obtener_motor_activo()
        
        # Status: generando (RAG activo)
        await manager.send_message(client_id, {
            "tipo": TIPOS_WEBSOCKET["estado"],
            "mensaje": f"Generando respuesta (motor: {motor_actual})...",
        })
        
        # LEGACY: Ejecutar RAG síncrona en thread pool
        loop = asyncio.get_event_loop()
        respuesta = await loop.run_in_executor(
            None,
            lambda: consultar_base_conocimiento(pregunta, motor=motor_actual)
        )
        
        # TODO vLLM: Streaming token por token
        # async for token in llm_service.generar_stream(chunks, pregunta):
        #     await manager.send_message(client_id, {
        #         "tipo": TIPOS_WEBSOCKET["token"],
        #         "token": token
        #     })
        
        # Enviar respuesta completa
        await manager.send_message(client_id, {
            "tipo": "respuesta",
            "pregunta": pregunta,
            "respuesta": respuesta,
            "motor": motor_actual,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Status: completado
        await manager.send_message(client_id, {
            "tipo": TIPOS_WEBSOCKET["final"],
            "mensaje": "Consulta completada."
        })
    
    except RAGError as e:
        await manager.send_message(client_id, {
            "tipo": TIPOS_WEBSOCKET["error"],
            "mensaje": f"Error RAG: {e.message}",
            "details": e.details
        })
    
    except Exception as e:
        print(f"[WS RAG ERROR] {client_id}: {e}")
        await manager.send_message(client_id, {
            "tipo": TIPOS_WEBSOCKET["error"],
            "mensaje": "Error procesando consulta. Inténtalo nuevamente."
        })


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Validación de token WebSocket (TODO: implementar)
# ─────────────────────────────────────────────────────────────────────────────

def validate_token_ws(token: str):  # -> models.Usuario:
    """
    TODO: Implementar validación JWT para WebSocket.
    
    Diferente a HTTP porque WebSocket no puede usar Depends().
    Debe validar manualmente el token antes de aceptar conexión.
    """
    # from app.core.security import decode_jwt_token
    # return decode_jwt_token(token)
    pass
