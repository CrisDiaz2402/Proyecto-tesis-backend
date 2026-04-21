#!/bin/bash
# ============================================================
# update.sh — Actualizar la tesis en Docker
# ============================================================
# Reconstruye solo las imágenes que cambiaron (backend o frontend)
# SIN borrar la base de datos, qdrant, ni documentos.
#
# USO (desde la raíz del proyecto, donde está docker-compose.yml):
#   ./scripts/update.sh           → actualiza todo
#   ./scripts/update.sh backend   → solo backend
#   ./scripts/update.sh frontend  → solo frontend
# ============================================================

set -e

SERVICIO=${1:-""}  # Vacío = actualizar todo

echo "==================================================="
echo "  ACTUALIZANDO TESIS EN DOCKER"
echo "==================================================="

# ── Verificar que estamos en el directorio correcto ───────
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ ERROR: Ejecuta este script desde el directorio raíz del proyecto"
    echo "   (donde está el docker-compose.yml)"
    exit 1
fi

# ── Función para actualizar un servicio ──────────────────
actualizar_servicio() {
    local svc=$1
    echo ""
    echo "→ Reconstruyendo '$svc'..."
    
    # Reconstruir imagen (usa caché de Docker, solo reconstruye lo que cambió)
    docker compose build $svc
    
    # Reiniciar el servicio con la nueva imagen
    docker compose up -d --no-deps $svc
    
    echo "  ✓ '$svc' actualizado"
}

# ── Actualizar según argumento ────────────────────────────
if [ -z "$SERVICIO" ]; then
    echo "Actualizando: backend + frontend"
    actualizar_servicio "backend"
    actualizar_servicio "frontend"
    
elif [ "$SERVICIO" == "backend" ]; then
    actualizar_servicio "backend"
    
elif [ "$SERVICIO" == "frontend" ]; then
    actualizar_servicio "frontend"
    
elif [ "$SERVICIO" == "vllm" ]; then
    echo "→ Actualizando vLLM (puede tardar si hay nueva versión de imagen)..."
    docker compose pull vllm
    docker compose up -d --no-deps vllm
    echo "  ✓ vLLM actualizado"
    
else
    echo "❌ Servicio desconocido: $SERVICIO"
    echo "   Opciones válidas: backend, frontend, vllm, (vacío=todo)"
    exit 1
fi

# ── Mostrar estado final ──────────────────────────────────
echo ""
echo "==================================================="
echo "  ESTADO ACTUAL DE LOS SERVICIOS"
echo "==================================================="
docker compose ps

echo ""
echo "  Comandos útiles:"
echo "  docker compose logs -f backend    → ver logs del backend"
echo "  docker compose logs -f vllm       → ver logs de vLLM"
echo "  docker compose restart backend    → reiniciar un servicio"
