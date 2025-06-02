#!/bin/bash
# 🚀 RUNPOD CSM VOICE CLONING STARTUP - VERSIÓN ROBUSTA
# Configurado para: runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04
# Sistema: CSM-1B nativo de Transformers 4.52.4+

set -e  # Exit on any error

echo "🎯 RUNPOD CSM VOICE CLONING - STARTUP ROBUSTO"
echo "============================================================"

# 1. Environment Verification
echo "🔍 1. Verificando entorno del sistema..."
cd /workspace/runttspod

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
echo "✅ GPU verification complete"

# 2. Setup environment variables
echo "🔑 2. Configurando variables de entorno..."
# Usar variable de entorno, no hardcodear
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ HF_TOKEN no configurado como variable de entorno"
    echo "💡 Configurar en RunPod: HF_TOKEN=tu_token_aqui"
fi
export NO_TORCH_COMPILE=1
export PYTHONPATH="/workspace/runttspod:$PYTHONPATH"
echo 'export NO_TORCH_COMPILE=1' >> ~/.bashrc
echo 'export PYTHONPATH="/workspace/runttspod:$PYTHONPATH"' >> ~/.bashrc
echo "✅ Variables de entorno configuradas"

# 3. INSTALAR DEPENDENCIAS CRÍTICAS PRIMERO
echo "🔧 3. INSTALANDO DEPENDENCIAS CRÍTICAS..."
pip install --no-cache-dir \
    "transformers>=4.52.1" \
    "accelerate>=0.20.0" \
    fastapi \
    uvicorn \
    python-multipart \
    aiofiles \
    --upgrade

echo "✅ Dependencias críticas instaladas"

# 4. Verificar modelo CSM-1B
echo "🔍 4. Verificando modelo CSM-1B..."
if [ -d "./models/sesame-csm-1b" ]; then
    model_size=$(du -h models/sesame-csm-1b/model.safetensors | cut -f1)
    echo "✅ Modelo CSM-1B encontrado: $model_size"
else
    echo "❌ Modelo CSM-1B no encontrado"
    echo "🔄 Descargando modelo CSM-1B..."
    
    mkdir -p models
    cd models
    
    # Install git-lfs if not installed
    if ! command -v git-lfs &> /dev/null; then
        echo "📦 Instalando git-lfs..."
        apt update && apt install -y git-lfs
        git lfs install
    fi
    
    # Download model
    git clone https://huggingface.co/sesame/csm-1b sesame-csm-1b
    cd ..
    
    if [ -f "./models/sesame-csm-1b/model.safetensors" ]; then
        echo "✅ Modelo CSM-1B descargado exitosamente"
    else
        echo "❌ Error descargando modelo CSM-1B"
        exit 1
    fi
fi

# 5. Verificar dataset Elise (opcional)
echo "🔍 5. Verificando dataset Elise..."
if [ -d "./datasets/csm-1b-elise" ]; then
    echo "✅ Dataset Elise CSM ya existe"
else
    echo "⚠️ Dataset Elise no encontrado (opcional)"
fi

# 6. VERIFICAR DEPENDENCIAS PYTHON
echo "🔧 6. VERIFICANDO DEPENDENCIAS PYTHON..."

# Verificar Python packages críticos
echo "📦 Verificando dependencias críticas..."
python -c "
import sys
missing = []

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError:
    missing.append('torch>=2.0.0')

try:
    import transformers
    print(f'✅ Transformers: {transformers.__version__}')
    # Verificar que sea una versión que soporte CSM
    if hasattr(transformers, 'CsmForConditionalGeneration'):
        print('✅ CSM support available')
    else:
        print('❌ CSM support not available, need Transformers >= 4.52.1')
        missing.append('transformers>=4.52.1')
except ImportError:
    missing.append('transformers>=4.52.1')

try:
    import fastapi
    print(f'✅ FastAPI: {fastapi.__version__}')
except ImportError:
    missing.append('fastapi')

try:
    import uvicorn
    print(f'✅ Uvicorn available')
except ImportError:
    missing.append('uvicorn')

try:
    import torchaudio
    print(f'✅ TorchAudio: {torchaudio.__version__}')
except ImportError:
    missing.append('torchaudio')

if missing:
    print(f'❌ Missing packages: {missing}')
    sys.exit(1)
else:
    print('✅ All critical dependencies available')
"

if [ $? -ne 0 ]; then
    echo "🔧 Instalando dependencias faltantes..."
    
    # Instalar Transformers actualizado
    pip install transformers>=4.52.1 --upgrade
    
    # Instalar dependencias de API
    pip install fastapi uvicorn python-multipart aiofiles
    
    # Verificar instalación
    python -c "
from transformers import CsmForConditionalGeneration, AutoProcessor
print('✅ CSM imports working correctly')
"
fi

# 6. Configurar estructura de directorios
echo "📁 6. Configurando estructura de directorios..."
mkdir -p outputs temp logs voices
echo "✅ Directorios creados"

# 7. Verificar archivo de voz de referencia
echo "🔍 7. Verificando archivo de voz de referencia..."
reference_voice="voices/Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3"
if [ -f "$reference_voice" ]; then
    echo "✅ Archivo de referencia encontrado: $reference_voice"
else
    echo "⚠️ Archivo de referencia no encontrado: $reference_voice"
    echo "💡 El sistema funcionará, pero sin perfil de voz predefinido"
fi

# 8. Test rápido del sistema
echo "🔧 8. Probando sistema CSM..."
python -c "
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor

print('🔍 Testing CSM system...')
try:
    model_path = './models/sesame-csm-1b'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f'📥 Loading processor from {model_path}...')
    processor = AutoProcessor.from_pretrained(model_path)
    
    print(f'📥 Loading model on {device}...')
    model = CsmForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    )
    
    print('✅ CSM system test successful!')
    
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_properties(0)
        memory_gb = gpu_info.total_memory / 1024**3
        print(f'🖥️ GPU: {gpu_info.name} ({memory_gb:.1f} GB)')
    
except Exception as e:
    print(f'❌ CSM system test failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Sistema CSM no funcionó correctamente"
    exit 1
fi

# 9. Información del sistema configurado
echo "📊 9. Información del sistema configurado..."
echo "============================================================"
echo "🎤 CSM VOICE CLONING SYSTEM - READY"
echo "============================================================"
echo "📦 Sistema: CSM-1B nativo de Transformers"
echo "🤖 Modelo: models/sesame-csm-1b ($(du -h models/sesame-csm-1b/model.safetensors | cut -f1))"
echo "🎭 Voces: $(ls voices/ 2>/dev/null | wc -l) perfiles disponibles"
echo "🔧 API: FastAPI + Uvicorn"
echo "🚀 Puerto: 7860"
echo "============================================================"

# 10. Iniciar API
echo "🚀 10. Iniciando CSM Voice Cloning API..."
echo "============================================================"
echo "🌐 ACCESO A LA API:"
echo "   • URL Principal: http://0.0.0.0:7860"
echo "   • Documentación: http://0.0.0.0:7860/docs"
echo "   • Health Check: http://0.0.0.0:7860/health"
echo "   • Voice Profiles: http://0.0.0.0:7860/voices"
echo "============================================================"
echo "🎯 COMANDOS DE PRUEBA:"
echo "   # Health check:"
echo "   curl http://localhost:7860/health"
echo ""
echo "   # Listar voces:"
echo "   curl http://localhost:7860/voices"
echo ""
echo "   # Clonar voz:"
echo "   curl -X POST 'http://localhost:7860/clone-voice' \\"
echo "        -F 'text=Hola mundo' \\"
echo "        -F 'temperature=0.7'"
echo "============================================================"
echo "🛑 Presiona Ctrl+C para detener el servidor"
echo "============================================================"

# Ejecutar API
python quick_start.py 