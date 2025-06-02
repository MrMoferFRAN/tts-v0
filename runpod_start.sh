#!/bin/bash
# 🚀 RUNPOD AUTO-START SCRIPT - CSM VOICE CLONING
# Configurado para Container Image: runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04
# Variable de entorno requerida: HF_TOKEN

set -e  # Exit on any error

echo "🎯 RUNPOD CSM VOICE CLONING - AUTO-START"
echo "============================================================"

# 1. Verificar variable de entorno crítica
echo "🔑 1. Verificando variables de entorno..."
if [ -z "$HF_TOKEN" ]; then
    echo "❌ ERROR: HF_TOKEN no configurado"
    echo "💡 Configurar en RunPod Environment Variables:"
    echo "   HF_TOKEN = tu_token_de_huggingface"
    echo "🛑 Abortando arranque..."
    exit 1
else
    echo "✅ HF_TOKEN configurado correctamente"
fi

# 2. Navegar al workspace
echo "📁 2. Navegando al workspace..."
cd /workspace

# 3. Verificar si ya existe el repositorio
echo "🔍 3. Verificando repositorio..."
if [ -d "runttspod" ]; then
    echo "📦 Repositorio ya existe, actualizando..."
    cd runttspod
    git pull origin main || echo "⚠️ No se pudo actualizar (usando versión local)"
else
    echo "📥 Clonando repositorio desde GitHub..."
    git clone https://github.com/MrMoferFRAN/runttspod.git
    cd runttspod
fi

# 4. Verificar que estamos en el directorio correcto
echo "📂 4. Verificando estructura del proyecto..."
if [ ! -f "startup.sh" ]; then
    echo "❌ ERROR: archivo startup.sh no encontrado"
    echo "🔍 Contenido del directorio actual:"
    ls -la
    exit 1
fi

echo "✅ Estructura del proyecto verificada"

# 5. Hacer ejecutable el script de startup
echo "🔧 5. Configurando permisos..."
chmod +x startup.sh

# 6. Mostrar información del sistema
echo "🖥️ 6. Información del sistema..."
echo "📦 Container Image: runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04"
echo "🐍 Python: $(python --version)"
echo "🔥 PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "🎮 CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if command -v nvidia-smi &> /dev/null; then
    echo "🎯 GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
fi

# 7. Crear directorio de logs si no existe
echo "📝 7. Configurando logging..."
mkdir -p logs

# 8. Ejecutar startup principal con logging
echo "🚀 8. Iniciando CSM Voice Cloning System..."
echo "============================================================"
echo "📋 INFORMACIÓN DE ACCESO:"
echo "   🌐 API URL: http://[POD_IP]:7860"
echo "   📖 Docs: http://[POD_IP]:7860/docs"
echo "   🔍 Health: http://[POD_IP]:7860/health"
echo "   📢 Voices: http://[POD_IP]:7860/voices"
echo "============================================================"
echo "🔄 Ejecutando startup.sh..."
echo "============================================================"

# Ejecutar con tee para mostrar output y guardarlo en log
./startup.sh 2>&1 | tee logs/startup_$(date +%Y%m%d_%H%M%S).log 