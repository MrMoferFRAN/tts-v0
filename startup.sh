#!/bin/bash
# 🚀 RUNPOD CSM VOICE CLONING STARTUP - VERSIÓN ROBUSTA
# Configurado para: runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04
# Sistema: CSM-1B nativo de Transformers 4.52.4+
# Incluye: Dependencias de audio (libsndfile, ffmpeg, soundfile, librosa) para backends robustos

set -e  # Exit on any error

echo "🎯 RUNPOD CSM VOICE CLONING - STARTUP ROBUSTO"
echo "============================================================"

# 1. Environment Verification
echo "🔍 1. Verificando entorno del sistema..."
cd /workspace/tts-v0

# Check GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
echo "✅ GPU verification complete"

# 2. Setup environment variables
echo "🔑 2. Configurando variables de entorno..."
# Manejar RunPod Secrets y variables de entorno
if [ -n "$RUNPOD_SECRET_HF_TOKEN" ]; then
    export HF_TOKEN="$RUNPOD_SECRET_HF_TOKEN"
    echo "✅ HF_TOKEN configurado desde RunPod Secret"
elif [ -n "$HF_TOKEN" ]; then
    echo "✅ HF_TOKEN configurado desde variable de entorno"
else
    echo "❌ ERROR: HF_TOKEN no configurado"
    echo "💡 Configurar en RunPod usando Secrets: RUNPOD_SECRET_HF_TOKEN"
    echo "💡 O como variable de entorno: HF_TOKEN"
    exit 1
fi

# Configurar autenticación de Hugging Face
echo "🔐 Configurando autenticación de Hugging Face..."
mkdir -p ~/.cache/huggingface
echo "$HF_TOKEN" > ~/.cache/huggingface/token

# Configurar git credentials para Hugging Face
git config --global credential.helper store
echo "https://MrMoferFRAN:$HF_TOKEN@huggingface.co" > ~/.git-credentials

# También configurar usando huggingface-hub
pip install --no-cache-dir huggingface-hub --upgrade
python -c "from huggingface_hub import login; login('$HF_TOKEN')" 2>/dev/null || echo "⚠️ huggingface-hub login failed, using git credentials"

export NO_TORCH_COMPILE=1
export PYTHONPATH="/workspace/tts-v0:$PYTHONPATH"
echo 'export NO_TORCH_COMPILE=1' >> ~/.bashrc
echo 'export PYTHONPATH="/workspace/tts-v0:$PYTHONPATH"' >> ~/.bashrc
echo "✅ Variables de entorno y autenticación configuradas"

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

# 3.5. INSTALAR DEPENDENCIAS DE AUDIO (CRÍTICO)
echo "🔊 3.5. INSTALANDO DEPENDENCIAS DE AUDIO..."
echo "📦 Instalando librerías de sistema para audio..."

# Instalar librerías de sistema necesarias para torchaudio backends
apt-get update && apt-get install -y \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    --no-install-recommends

echo "📦 Instalando soundfile y librosa para manejo robusto de archivos de audio..."
pip install --no-cache-dir soundfile librosa

# Verificar que los backends de audio estén disponibles
echo "🔍 Verificando backends de audio..."
python -c "
import torchaudio
backends = torchaudio.list_audio_backends()
print(f'✅ TorchAudio backends disponibles: {backends}')

try:
    import soundfile as sf
    print('✅ SoundFile disponible')
except ImportError:
    print('❌ SoundFile no disponible')
    exit(1)

try:
    import librosa
    print('✅ Librosa disponible')
except ImportError:
    print('❌ Librosa no disponible')
    exit(1)

if not backends:
    print('❌ No hay backends de audio disponibles para torchaudio')
    print('⚠️  Esto podría causar errores al guardar archivos de audio')
    exit(1)
else:
    print('✅ Backends de audio configurados correctamente')
"

if [ $? -ne 0 ]; then
    echo "❌ Error configurando dependencias de audio"
    exit 1
fi

echo "✅ Dependencias de audio instaladas y verificadas"

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
    
    # Verificar backends de audio
    backends = torchaudio.list_audio_backends()
    print(f'✅ TorchAudio backends: {backends}')
    if not backends:
        print('⚠️  Sin backends de audio - puede causar problemas')
except ImportError:
    missing.append('torchaudio')

try:
    import soundfile as sf
    print(f'✅ SoundFile: disponible')
except ImportError:
    missing.append('soundfile')

try:
    import librosa
    print(f'✅ Librosa: disponible')
except ImportError:
    missing.append('librosa')

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
    
    # Instalar dependencias de API y audio
    pip install fastapi uvicorn python-multipart aiofiles soundfile librosa
    
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
echo "🔍 7. Verificando archivos de voz de referencia..."
reference_voice_old="voices/fran-fem/Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo. Y ¿cómo lo tomaron?.wav"
reference_voice_new="voices/fran-fem/fran_fem_sample.wav"

if [ -f "$reference_voice_new" ]; then
    echo "✅ Archivo de referencia encontrado: $reference_voice_new"
elif [ -f "$reference_voice_old" ]; then
    echo "⚠️ Archivo con nombre problemático encontrado, renombrando..."
    cd voices/fran-fem && mv *.wav fran_fem_sample.wav && cd ../..
    echo "✅ Archivo renombrado a: $reference_voice_new"
    
    # Actualizar profiles.json si existe
    if [ -f "voices/fran-fem/profiles.json" ]; then
        echo "🔧 Actualizando profiles.json..."
        python -c "
import json
from datetime import datetime

try:
    with open('voices/fran-fem/profiles.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Actualizar path del audio
    if 'profiles' in data and len(data['profiles']) > 0:
        data['profiles'][0]['name'] = 'fran_fem_sample'
        data['profiles'][0]['audio_path'] = '/workspace/tts-v0/voices/fran-fem/fran_fem_sample.wav'
        data['updated_at'] = datetime.now().isoformat()
        
        with open('voices/fran-fem/profiles.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print('✅ profiles.json actualizado')
    else:
        print('⚠️ profiles.json no tiene el formato esperado')
except Exception as e:
    print(f'❌ Error actualizando profiles.json: {e}')
"
    fi
else
    echo "⚠️ Archivo de referencia no encontrado"
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
    
    # Test torch.compiler compatibility
    if not hasattr(torch.compiler, 'is_compiling'):
        print('⚠️  torch.compiler compatibility patch needed')
    else:
        print('✅ torch.compiler compatible')
    
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
echo "🔧 API: FastAPI + Uvicorn (voice_api_complete.py)"
echo "🚀 Puerto: 7860"
echo "============================================================"

# 10. Iniciar API
echo "🚀 10. Iniciando CSM Voice Cloning API..."

# Ejecutar API completa
python voice_api_complete.py 
