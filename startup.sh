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

# 4. Descargar modelo CSM-1B Turbo INT8
echo "🔍 4. Descargando modelo CSM-1B Turbo INT8..."
TURBO_DIR="./models/csm-1b-turbo"

# Verificar si ya existe el modelo turbo completo
if [ -f "$TURBO_DIR/model.safetensors" ] && [ -f "$TURBO_DIR/transformers-00001-of-00002.safetensors" ] && [ -f "$TURBO_DIR/transformers-00002-of-00002.safetensors" ]; then
    model_size=$(du -sh "$TURBO_DIR/model.safetensors" | cut -f1)
    echo "✅ Modelo CSM-1B Turbo completo encontrado: $model_size"
else
    echo "🔄 Descargando modelo CSM-1B Turbo desde lunahr/csm-1b-safetensors-quants..."
    
    # Crear directorio models si no existe
    mkdir -p "$TURBO_DIR"
    
    # 4.1. Descargar modelo uint8 cuantizado
    echo "📥 4.1. Descargando modelo uint8 cuantizado..."
    python - <<'PY'
import os
from huggingface_hub import hf_hub_download

print("📥 Descargando model_uint8.safetensors...")
print("🔗 Repo: lunahr/csm-1b-safetensors-quants")
print("📁 Destino: models/csm-1b-turbo")

try:
    downloaded_file = hf_hub_download(
        repo_id="lunahr/csm-1b-safetensors-quants",
        filename="model_uint8.safetensors",
        local_dir="models/csm-1b-turbo",
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN")
    )
    print("✅ Modelo turbo descargado exitosamente")
except Exception as e:
    print(f"❌ Error durante la descarga: {e}")
    exit(1)
PY

    if [ $? -ne 0 ]; then
        echo "❌ Error descargando modelo turbo"
        exit 1
    fi
    
    # 4.2. Copiar como model.safetensors
    echo "🔍 4.2. Copiando modelo como model.safetensors..."
    if [ -f "$TURBO_DIR/model_uint8.safetensors" ]; then
        cp "$TURBO_DIR/model_uint8.safetensors" "$TURBO_DIR/model.safetensors"
        echo "✅ model_uint8.safetensors copiado como model.safetensors"
    else
        echo "❌ No se encontró model_uint8.safetensors"
        exit 1
    fi
    
    # 4.3. Descargar archivos de configuración desde sesame/csm-1b
    echo "📥 4.3. Descargando archivos de configuración..."
    python - <<'PY'
import os
from huggingface_hub import hf_hub_download

print('📥 Descargando archivos de configuración CSM...')
print('🔗 Repo: sesame/csm-1b')
print('📁 Destino: models/csm-1b-turbo')

config_files = [
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'preprocessor_config.json',
    'special_tokens_map.json',
    'generation_config.json',
    'chat_template.jinja'
]

for filename in config_files:
    try:
        print(f'📥 Descargando {filename}...')
        downloaded_file = hf_hub_download(
            repo_id='sesame/csm-1b',
            filename=filename,
            local_dir='models/csm-1b-turbo',
            token=os.environ.get('HF_TOKEN')
        )
        print(f'✅ {filename} descargado')
    except Exception as e:
        print(f'❌ Error descargando {filename}: {e}')
        exit(1)

print('✅ Archivos de configuración descargados')
PY

    if [ $? -ne 0 ]; then
        echo "❌ Error descargando archivos de configuración"
        exit 1
    fi
    
    # 4.4. Descargar índice de transformers
    echo "📥 4.4. Descargando índice de transformers..."
    python - <<'PY'
import os
from huggingface_hub import hf_hub_download

print('📥 Descargando transformers.safetensors.index.json...')
downloaded_file = hf_hub_download(
    repo_id='sesame/csm-1b',
    filename='transformers.safetensors.index.json',
    local_dir='models/csm-1b-turbo',
    token=os.environ.get('HF_TOKEN')
)
print('✅ transformers.safetensors.index.json descargado')
PY

    if [ $? -ne 0 ]; then
        echo "❌ Error descargando índice de transformers"
        exit 1
    fi
    
    # 4.5. Descargar archivos transformers
    echo "📥 4.5. Descargando archivos transformers..."
    python - <<'PY'
import os
from huggingface_hub import hf_hub_download

print('📥 Descargando archivos transformers...')
print('🔗 Repo: sesame/csm-1b')
print('📁 Destino: models/csm-1b-turbo')

transformer_files = [
    'transformers-00001-of-00002.safetensors',
    'transformers-00002-of-00002.safetensors'
]

for filename in transformer_files:
    try:
        print(f'📥 Descargando {filename}...')
        downloaded_file = hf_hub_download(
            repo_id='sesame/csm-1b',
            filename=filename,
            local_dir='models/csm-1b-turbo',
            token=os.environ.get('HF_TOKEN')
        )
        print(f'✅ {filename} descargado')
    except Exception as e:
        print(f'❌ Error descargando {filename}: {e}')
        exit(1)

print('✅ Archivos transformers descargados')
PY

    if [ $? -ne 0 ]; then
        echo "❌ Error descargando archivos transformers"
        exit 1
    fi
fi

# Mostrar información del modelo turbo
if [ -f "$TURBO_DIR/model.safetensors" ]; then
    model_size=$(du -sh "$TURBO_DIR/model.safetensors" | cut -f1)
    echo "📦 Tamaño del modelo turbo: $model_size"
fi

# Verificar que todos los archivos estén presentes
echo "🔍 Verificando archivos del modelo turbo..."
required_files=(
    "model.safetensors"
    "config.json"
    "tokenizer.json"
    "transformers.safetensors.index.json"
    "transformers-00001-of-00002.safetensors"
    "transformers-00002-of-00002.safetensors"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$TURBO_DIR/$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✅ Todos los archivos del modelo turbo están presentes"
else
    echo "❌ Archivos faltantes: ${missing_files[*]}"
    exit 1
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

# 7. Configurar estructura de directorios
echo "📁 7. Configurando estructura de directorios..."
mkdir -p outputs temp logs voices
echo "✅ Directorios creados"

# 8. Verificar archivo de voz de referencia
echo "🔍 8. Verificando archivos de voz de referencia..."
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

# 9. Test robusto del sistema CSM Turbo
echo "🔧 9. Probando sistema CSM Turbo..."
python -c "
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor
import os

print('🔍 Testing CSM Turbo system...')
try:
    model_path = './models/csm-1b-turbo'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Verificar que el archivo model.safetensors existe
    model_file = os.path.join(model_path, 'model.safetensors')
    if os.path.exists(model_file):
        size_mb = os.path.getsize(model_file) / (1024*1024)
        print(f'✅ model.safetensors: {size_mb:.1f} MB')
    else:
        print(f'❌ model.safetensors: NO ENCONTRADO')
        raise FileNotFoundError('Archivo crítico faltante: model.safetensors')
    
    print(f'📥 Loading model from {model_path} on {device}...')
    model = CsmForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    )
    
    print('✅ CSM Turbo system test successful!')
    
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
    print(f'❌ CSM Turbo system test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Sistema CSM Turbo no funcionó correctamente"
    echo "🔍 Información de debugging:"
    echo "📁 Contenido del directorio del modelo:"
    ls -la "$TURBO_DIR/" || echo "Directorio no accesible"
    exit 1
fi

# 10. Información del sistema configurado
echo "📊 10. Información del sistema configurado..."
echo "============================================================"
echo "🎤 CSM VOICE CLONING SYSTEM - READY"
echo "============================================================"
echo "📦 Sistema: CSM-1B Turbo INT8"
echo "🤖 Modelo: models/csm-1b-turbo ($(du -sh models/csm-1b-turbo/model.safetensors | cut -f1))"
echo "🎭 Voces: $(ls voices/ 2>/dev/null | wc -l) perfiles disponibles"
echo "🔧 API: FastAPI + Uvicorn (voice_api_complete.py)"
echo "🚀 Puerto: 7860"
echo "✅ Modelo turbo verificado:"
ls -la "$TURBO_DIR"/model.safetensors
echo "============================================================"

# 11. Iniciar API
echo "🚀 11. Iniciando CSM Voice Cloning API..."

# Ejecutar API completa
python voice_api_complete.py
