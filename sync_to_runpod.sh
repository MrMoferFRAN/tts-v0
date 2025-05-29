#!/bin/bash

# Script para sincronizar archivos con RunPod
# Uso: ./sync_to_runpod.sh [POD_IP]

if [ "$#" -ne 1 ]; then
    echo "❌ Uso: $0 [POD_IP]"
    echo "💡 Ejemplo: $0 123.45.67.89"
    exit 1
fi

POD_IP=$1

echo "🚀 SINCRONIZANDO PROYECTO CON RUNPOD"
echo "===================================="
echo "🎯 Pod IP: $POD_IP"
echo

# 1. Verificar conexión
echo "🔍 Verificando conexión SSH..."
if ! ssh -o ConnectTimeout=10 root@$POD_IP "echo 'Conexión exitosa'"; then
    echo "❌ No se puede conectar al pod"
    echo "💡 Verifica que el pod esté corriendo y el IP sea correcto"
    exit 1
fi

# 2. Crear directorios base
echo "📁 Creando directorios en RunPod..."
ssh root@$POD_IP "mkdir -p /workspace/sync"

# 3. Sincronizar archivos del proyecto
echo "📤 Subiendo archivos del proyecto..."

# Subir script de setup
echo "  📝 Setup script..."
scp setup_csm_runpod.sh root@$POD_IP:/workspace/

# Subir README
echo "  📖 README..."
scp README.md root@$POD_IP:/workspace/

# Subir scripts adicionales (si existen)
if [ -d "scripts" ]; then
    echo "  📜 Scripts personalizados..."
    scp -r scripts/ root@$POD_IP:/workspace/sync/
fi

# Subir configuraciones (si existen)
if [ -d "configs" ]; then
    echo "  ⚙️ Configuraciones..."
    scp -r configs/ root@$POD_IP:/workspace/sync/
fi

# 4. Hacer ejecutable el script de setup
echo "🔧 Configurando permisos..."
ssh root@$POD_IP "chmod +x /workspace/setup_csm_runpod.sh"

# 5. Mostrar próximos pasos
echo
echo "✅ SINCRONIZACIÓN COMPLETADA"
echo "============================"
echo
echo "🎯 Próximos pasos:"
echo "1️⃣  Conectar al pod:"
echo "    ssh root@$POD_IP"
echo
echo "2️⃣  Ejecutar setup:"
echo "    cd /workspace"
echo "    ./setup_csm_runpod.sh"
echo
echo "3️⃣  Iniciar entorno:"
echo "    ./start.sh"
echo
echo "4️⃣  Acceder a Jupyter:"
echo "    http://$POD_IP:8888"
echo
echo "🎭 ¡Listo para entrenar Elise con CUDA!" 