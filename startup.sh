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

# Verificar si ya existe el modelo turbo
if [ -f "$TURBO_DIR/model.safetensors" ]; then
    model_size=$(du -sh "$TURBO_DIR/model.safetensors" | cut -f1)
    echo "✅ Modelo CSM-1B Turbo encontrado: $model_size"
else
    echo "🔄 Descargando modelo CSM-1B Turbo desde lunahr/csm-1b-safetensors-quants..."
    
    # Crear directorio models si no existe
    mkdir -p "$TURBO_DIR"
    
    # Descargar solo el archivo model_uint8.safetensors
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
fi

# 4.1.b AJUSTAR NOMBRE DEL PESO INT8 ────────────────────────────────────────
echo "🔍 4.1.b Ajustando nombre del peso INT8…"

if [ -f "$TURBO_DIR/model_uint8.safetensors" ] && [ ! -f "$TURBO_DIR/model.safetensors" ]; then
    # Copiar model_uint8.safetensors como model.safetensors
    cp "$TURBO_DIR/model_uint8.safetensors" "$TURBO_DIR/model.safetensors"
    echo "✅ model_uint8.safetensors copiado como model.safetensors"
elif [ -f "$TURBO_DIR/model.safetensors" ]; then
    echo "✅ model.safetensors ya presente"
else
    echo "❌ No se encontró model_uint8.safetensors"
    exit 1
fi

# Mostrar información del modelo turbo
if [ -f "$TURBO_DIR/model.safetensors" ]; then
    model_size=$(du -sh "$TURBO_DIR/model.safetensors" | cut -f1)
    echo "📦 Tamaño del modelo turbo: $model_size"
fi

# 4.2  DESCARGAR ARCHIVOS AUXILIARES DEL MODELO BASE ─────────────────────────
echo "🔍 4.2 Descargando archivos auxiliares de sesame/csm-1b ..."

BASE_REPO="sesame/csm-1b"
AUX_FILES=(
  "config.json"
  "generation_config.json"
  "tokenizer_config.json"
  "tokenizer.json"                 # si el modelo usa tokenizador JSON
  "spiece.model"                   # o bien vocab.json/merges.txt si es BPE
  "vocab.json"
  "merges.txt"
  "special_tokens_map.json"
  "preprocessor_config.json"
)

for file in "${AUX_FILES[@]}"; do
  # omite archivos inexistentes en el repo (HF lanza error 404)
  if [ ! -f "$TURBO_DIR/$file" ]; then
    echo "📥  Descargando $file ..."
    python - <<PY
import os, sys
from huggingface_hub import hf_hub_download
try:
    from huggingface_hub import HfHubHTTPError          # >=0.24
except ImportError:
    from huggingface_hub.utils import HfHubHTTPError    # <=0.23

repo      = "$BASE_REPO"
filename  = "$file"
dest_dir  = "$TURBO_DIR"
token     = os.environ.get("HF_TOKEN")

try:
    hf_hub_download(
        repo_id = repo,
        filename = filename,
        local_dir = dest_dir,
        local_dir_use_symlinks = False,
        token = token
    )
    print(f"✅  $file descargado")
except HfHubHTTPError as err:
    if err.response_code == 404:
        print(f"⚠️  $file no existe en el repo, se omite")
    else:
        print(f"❌  Error descargando $file: {err}")
        sys.exit(1)
PY
  else
    echo "✅  $file ya presente"
  fi
done

# 4.3  AÑADIR PLANTILLA DE CHAT (chat_template) SI FALTA ─────────────────────–
python - <<'PY'
import json, pathlib, textwrap, datetime

cfg_path = pathlib.Path("./models/csm-1b-turbo/tokenizer_config.json")
if cfg_path.exists():
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "chat_template" not in cfg:
        cfg["chat_template"] = textwrap.dedent("""
        {% for m in messages %}[{{ m.role }}]{% for c in m.content %}
        {{ (c.text if c.type == 'text' else '<AUDIO>') | trim }}{% endfor %}
        {% endfor %}""").strip()

        cfg["updated_at"] = datetime.datetime.now().isoformat()
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print("✅  chat_template añadido a tokenizer_config.json")
    else:
        print("✅  tokenizer_config.json ya contiene chat_template")
else:
    print("⚠️  tokenizer_config.json no encontrado; omitiendo parche")
PY
echo "✅ Archivos auxiliares preparados"

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
