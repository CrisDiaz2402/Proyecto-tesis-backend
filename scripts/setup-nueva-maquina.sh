#!/bin/bash
# ============================================================
# setup-nueva-maquina.sh
# ============================================================
# Ejecutar en una máquina Ubuntu 22.04 / 24.04 FRESCA con GPU NVIDIA
# Instala: Docker, Docker Compose, NVIDIA Driver, NVIDIA Container Toolkit
#
# USO:
#   chmod +x setup-nueva-maquina.sh
#   sudo ./setup-nueva-maquina.sh
# ============================================================

set -e  # Salir si cualquier comando falla

echo "==================================================="
echo "  SETUP MÁQUINA PARA TESIS — Ubuntu 22/24 + GPU"
echo "==================================================="

# ── PASO 1: Actualizar sistema ────────────────────────────
echo ""
echo "[1/6] Actualizando sistema..."
apt-get update && apt-get upgrade -y
apt-get install -y curl wget gnupg lsb-release ca-certificates git

# ── PASO 2: Instalar Docker ───────────────────────────────
echo ""
echo "[2/6] Instalando Docker..."

# Remover versiones viejas si existen
apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

# Agregar repositorio oficial de Docker
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Agregar usuario actual al grupo docker (sin necesidad de sudo)
usermod -aG docker $SUDO_USER

# Verificar instalación
docker --version
docker compose version

echo "  ✓ Docker instalado correctamente"

# ── PASO 3: Instalar NVIDIA Driver ───────────────────────
echo ""
echo "[3/6] Instalando NVIDIA Driver..."

# Verificar si ya hay driver instalado
if nvidia-smi &>/dev/null; then
    echo "  · Driver NVIDIA ya instalado:"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    echo "  Instalando driver NVIDIA recomendado..."
    
    # Ubuntu 24.04
    apt-get install -y ubuntu-drivers-common
    ubuntu-drivers install
    
    echo "  ✓ Driver NVIDIA instalado"
    echo "  ⚠️  NECESITAS REINICIAR para activar el driver."
    echo "     Después del reinicio, vuelve a ejecutar este script"
    echo "     o continúa manualmente desde el paso 4."
    
    read -p "¿Reiniciar ahora? (s/n): " reiniciar
    if [[ "$reiniciar" == "s" ]]; then
        reboot
    fi
fi

# ── PASO 4: Instalar NVIDIA Container Toolkit ────────────
echo ""
echo "[4/6] Instalando NVIDIA Container Toolkit..."

# Repositorio oficial de NVIDIA
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y nvidia-container-toolkit

# Configurar Docker para usar NVIDIA
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "  ✓ NVIDIA Container Toolkit instalado"

# ── PASO 5: Verificar GPU accesible desde Docker ─────────
echo ""
echo "[5/6] Verificando GPU en Docker..."

docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo "  ✓ GPU accesible desde Docker"

# ── PASO 6: Clonar o copiar el proyecto ──────────────────
echo ""
echo "[6/6] Configuración final..."
echo ""
echo "==================================================="
echo "  ✅ MÁQUINA LISTA PARA CORRER LA TESIS"
echo "==================================================="
echo ""
echo "  PRÓXIMOS PASOS:"
echo ""
echo "  1. Cierra sesión y vuelve a entrar (para que docker sin sudo funcione)"
echo "     o ejecuta: newgrp docker"
echo ""
echo "  2. Copia o clona tu proyecto en esta máquina:"
echo "     git clone <tu-repo> ~/tesis"
echo "     cd ~/tesis"
echo ""
echo "  3. Configura las variables de entorno:"
echo "     cp Proyecto-tesis-backend/.env.docker.example Proyecto-tesis-backend/.env.docker"
echo "     nano Proyecto-tesis-backend/.env.docker  # Ajusta contraseñas y API keys"
echo ""
echo "  4. Levanta todo:"
echo "     docker compose up -d"
echo ""
echo "  5. La primera vez, vLLM descargará el modelo Qwen (~2GB)."
echo "     Puedes ver el progreso con:"
echo "     docker compose logs -f vllm"
echo ""
