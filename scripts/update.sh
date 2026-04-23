#!/bin/bash

set -e

SERVICIO=${1:-""}  

echo "==================================================="
echo "  ACTUALIZANDO TESIS EN DOCKER"
echo "==================================================="

if [ ! -f "docker-compose.yml" ]; then
    echo "❌ ERROR: Ejecuta este script desde el directorio raíz del proyecto"
    echo "   (donde está el docker-compose.yml)"
    exit 1
fi

actualizar_servicio() {
    local svc=$1
    echo ""
    echo "→ Reconstruyendo '$svc'..."
    
    docker compose build $svc
    
    docker compose up -d --no-deps $svc
    
    echo "  ✓ '$svc' actualizado"
}

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
