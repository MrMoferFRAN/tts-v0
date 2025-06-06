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

# 4. Verificar / descargar modelo CSM-1B (MÉTODO ROBUSTO CON HUGGINGFACE_HUB)
echo "🔍 4. Verificando modelo CSM-1B..."
MODEL_DIR="./models/sesame-csm-1b"

# Verificar si ya existe el modelo completo
if [ -f "$MODEL_DIR/config.json" ] && ls "$MODEL_DIR"/transformers-*-of-*.safetensors 1>/dev/null 2>&1; then
    model_size=$(du -sh "$MODEL_DIR" | cut -f1)
    echo "✅ Modelo CSM-1B encontrado: $model_size"
    echo "📋 Archivos safetensors encontrados:"
    ls -la "$MODEL_DIR"/transformers-*-of-*.safetensors
else
    echo "🔄 Descargando modelo CSM-1B con huggingface_hub (método robusto)..."
    
    # Asegurar que huggingface_hub esté actualizado
    pip install --no-cache-dir huggingface_hub --upgrade
    
    # Crear directorio models si no existe
    mkdir -p models
    
    # Descargar usando huggingface_hub (más robusto que git-lfs)
    python - <<'PY'
import os
from huggingface_hub import snapshot_download

print("📥 Iniciando descarga del modelo CSM-1B...")
print("🔗 Repo: sesame/csm-1b")
print("📁 Destino: models/sesame-csm-1b")

try:
    snapshot_download(
        repo_id="sesame/csm-1b",
        local_dir="models/sesame-csm-1b",
        local_dir_use_symlinks=False,  # copia real, sin symlinks → evita problemas en contenedores
        token=os.environ.get("HF_TOKEN"),
        resume_download=True  # continúa descarga si se interrumpió
    )
    print("✅ Descarga completada exitosamente")
except Exception as e:
    print(f"❌ Error durante la descarga: {e}")
    exit(1)
PY

    if [ $? -ne 0 ]; then
        echo "❌ Error descargando modelo con huggingface_hub"
        exit 1
    fi
fi

# Verificación exhaustiva de archivos críticos
echo "🔍 Verificando integridad del modelo..."

# Verificar archivos safetensors específicos
if ! ls "$MODEL_DIR"/transformers-*-of-*.safetensors 1>/dev/null 2>&1; then
    echo "❌ No se han descargado los archivos safetensors"
    echo "📋 Archivos esperados:"
    echo "   - transformers-00001-of-00002.safetensors"
    echo "   - transformers-00002-of-00002.safetensors"
    echo "📁 Contenido actual del directorio:"
    ls -la "$MODEL_DIR"/ || echo "Directorio no existe"
    exit 1
fi

# Verificar archivos específicos mencionados en el error
required_files=(
    "$MODEL_DIR/transformers-00001-of-00002.safetensors"
    "$MODEL_DIR/transformers-00002-of-00002.safetensors"
    "$MODEL_DIR/config.json"
    "$MODEL_DIR/tokenizer.json"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ Archivos faltantes:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

echo "✅ Todos los archivos críticos del modelo están presentes:"
echo "📋 Archivos safetensors verificados:"
ls -la "$MODEL_DIR"/transformers-*-of-*.safetensors

# Mostrar tamaño total del modelo
model_size=$(du -sh "$MODEL_DIR" | cut -f1)
echo "📦 Tamaño total del modelo: $model_size"

# 4.1 DESCARGAR VARIANTE TURBO INT8 ──────────────────────────────────────────
echo "🔍 4.1 Verificando modelo TURBO INT8..."
TURBO_DIR="./models/csm-1b-turbo"

# Si el safetensor INT8 no existe, lo descargamos
if [ ! -f "$TURBO_DIR/model_int8.safetensors" ]; then
    echo "🔄 Descargando INT8 turbo (lunahr/csm-1b-safetensors-quants)…"
    
    pip install --no-cache-dir huggingface_hub --upgrade
    
    python - <<'PY'
import os, sys
from huggingface_hub import snapshot_download

try:
    snapshot_download(
        repo_id="lunahr/csm-1b-safetensors-quants",
        local_dir="models/csm-1b-turbo",
        local_dir_use_symlinks=False,
        allow_patterns=[
            "model_int8.safetensors",        # pesos INT8 (≈1.55 GB)
            "config.json",
            "tokenizer.json", 
            "generation_config.json",
            "special_tokens_map.json",
            "*.txt"
        ],
        resume_download=True,
        token=os.environ.get("HF_TOKEN"),
    )
    print("✅ INT8 turbo descargado con éxito")
except Exception as e:
    print(f"❌ Error descargando INT8 turbo: {e}")
    sys.exit(1)
PY

    if [ $? -ne 0 ]; then
        echo "❌ Error descargando modelo INT8 turbo"
        exit 1
    fi
else
    echo "✅ INT8 turbo ya presente"
fi

# Verificación rápida
if [ -f "$TURBO_DIR/model_int8.safetensors" ] && [ -f "$TURBO_DIR/config.json" ]; then
    size_int8=$(du -sh "$TURBO_DIR/model_int8.safetensors" | cut -f1)
    echo "📦 INT8 turbo listo (${size_int8}) → $TURBO_DIR"
else
    echo "❌ Faltan archivos críticos del INT8 turbo"
    echo "📋 Archivos esperados:"
    echo "   - $TURBO_DIR/model_int8.safetensors"
    echo "   - $TURBO_DIR/config.json"
    echo "📁 Contenido actual:"
    ls -la "$TURBO_DIR/" 2>/dev/null || echo "Directorio no existe"
    exit 1
fi

# ---------------------------------------------------------------------------
# 4.2 COMPLETAR METADATOS DEL TURBO INT8
# ---------------------------------------------------------------------------
echo "🔍 4.2 Completando metadatos del turbo INT8…"
TURBO_DIR="./models/csm-1b-turbo"
BASE_DIR="./models/sesame-csm-1b"  # ya lo tienes completo

# 1) Si el repositorio INT8 contiene los ficheros, descárgalos
python - <<'PY'
import os, sys
from huggingface_hub import snapshot_download

need = {
    "tokenizer.json",
    "tokenizer_config.json", 
    "special_tokens_map.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.json",
    "generation_config.json",
}

have = set(os.listdir("models/csm-1b-turbo"))
missing = need - have

if missing:
    print(f"🔄 Faltan {len(missing)} archivos en turbo → intentando descargarlos…")
    try:
        snapshot_download(
            repo_id="lunahr/csm-1b-safetensors-quants",
            local_dir="models/csm-1b-turbo",
            allow_patterns=list(missing),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=os.environ.get("HF_TOKEN"),
        )
        print("✅ Metadatos adicionales descargados")
    except Exception as e:
        print(f"⚠️ No se pudieron descargar algunos metadatos: {e}")
        print("💡 Se intentará copiar desde el modelo base")
else:
    print("✅ Todos los metadatos ya están presentes")
PY

# 2) Si aún faltan, cópialos del modelo "normal"
for f in tokenizer.json tokenizer_config.json special_tokens_map.json \
         processor_config.json preprocessor_config.json chat_template.json \
         generation_config.json; do
    if [ ! -f "$TURBO_DIR/$f" ] && [ -f "$BASE_DIR/$f" ]; then
        echo "📥 Copiando $f desde modelo base"
        cp "$BASE_DIR/$f" "$TURBO_DIR/"
    fi
done

# 3) Verificación final
missing_critical=()
for f in tokenizer.json processor_config.json config.json; do
    if [ ! -f "$TURBO_DIR/$f" ]; then
        missing_critical+=("$f")
    fi
done

if [ ${#missing_critical[@]} -gt 0 ]; then
    echo "❌ Faltan archivos críticos para AutoProcessor:"
    for f in "${missing_critical[@]}"; do
        echo "   - $f"
    done
    echo "📁 Contenido actual de $TURBO_DIR:"
    ls -la "$TURBO_DIR/"
    exit 1
fi

echo "✅ Metadatos del turbo completos"
echo "📋 Archivos de configuración verificados:"
ls -la "$TURBO_DIR"/*.json | sed 's/^/   /'

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

# 9. Test robusto del sistema CSM
echo "🔧 9. Probando sistema CSM..."
python -c "
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor
import os

print('🔍 Testing CSM system...')

# Test modelo principal
try:
    model_path = './models/sesame-csm-1b'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Verificar que los archivos específicos existen
    safetensor_files = [
        'transformers-00001-of-00002.safetensors',
        'transformers-00002-of-00002.safetensors'
    ]
    
    print('🔍 Verificando archivos safetensors del modelo principal...')
    for file in safetensor_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f'✅ {file}: {size_mb:.1f} MB')
        else:
            print(f'❌ {file}: NO ENCONTRADO')
            raise FileNotFoundError(f'Archivo crítico faltante: {file}')
    
    print(f'📥 Loading processor from {model_path}...')
    processor = AutoProcessor.from_pretrained(model_path)
    
    print(f'📥 Loading model on {device}...')
    model = CsmForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32
    )
    
    print('✅ Modelo principal CSM test successful!')
    del model  # Liberar memoria para el test del turbo
    
except Exception as e:
    print(f'❌ CSM main model test failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)

# Test modelo turbo INT8 si existe
turbo_path = './models/csm-1b-turbo'
if os.path.exists(os.path.join(turbo_path, 'model_int8.safetensors')):
    print('🔍 Testing modelo turbo INT8...')
    try:
        print(f'📥 Loading turbo processor from {turbo_path}...')
        turbo_processor = AutoProcessor.from_pretrained(turbo_path)
        
        print(f'📥 Loading turbo model (INT8) on {device}...')
        turbo_model = CsmForConditionalGeneration.from_pretrained(
            turbo_path,
            device_map=device,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        )
        
        print('✅ Modelo turbo INT8 test successful!')
        del turbo_model  # Liberar memoria
        
    except Exception as e:
        print(f'⚠️ Turbo INT8 model test failed (optional): {e}')
        print('💡 El modelo principal sigue funcionando')
else:
    print('ℹ️ Modelo turbo INT8 no disponible (opcional)')

# GPU info
if torch.cuda.is_available():
    gpu_info = torch.cuda.get_device_properties(0)
    memory_gb = gpu_info.total_memory / 1024**3
    print(f'🖥️ GPU: {gpu_info.name} ({memory_gb:.1f} GB)')

# Test torch.compiler compatibility
if not hasattr(torch.compiler, 'is_compiling'):
    print('⚠️  torch.compiler compatibility patch needed')
else:
    print('✅ torch.compiler compatible')

print('✅ Sistema CSM completamente funcional!')
"

if [ $? -ne 0 ]; then
    echo "❌ Sistema CSM no funcionó correctamente"
    echo "🔍 Información de debugging:"
    echo "📁 Contenido del directorio del modelo:"
    ls -la "$MODEL_DIR/" || echo "Directorio no accesible"
    exit 1
fi

# 10. Información del sistema configurado
echo "📊 10. Información del sistema configurado..."
echo "============================================================"
echo "🎤 CSM VOICE CLONING SYSTEM - READY"
echo "============================================================"
echo "📦 Sistema: CSM-1B nativo de Transformers"
echo "🤖 Modelo Principal: models/sesame-csm-1b ($(du -sh models/sesame-csm-1b | cut -f1))"
if [ -f "./models/csm-1b-turbo/model_int8.safetensors" ]; then
    echo "⚡ Modelo Turbo INT8: models/csm-1b-turbo ($(du -sh models/csm-1b-turbo/model_int8.safetensors | cut -f1))"
else
    echo "⚠️ Modelo Turbo INT8: No disponible"
fi
echo "🎭 Voces: $(ls voices/ 2>/dev/null | wc -l) perfiles disponibles"
echo "🔧 API: FastAPI + Uvicorn (voice_api_complete.py)"
echo "🚀 Puerto: 7860"
echo "✅ Archivos safetensors verificados:"
echo "   📁 Modelo Principal:"
ls -la "$MODEL_DIR"/transformers-*-of-*.safetensors | sed 's/^/      /'
if [ -f "./models/csm-1b-turbo/model_int8.safetensors" ]; then
    echo "   ⚡ Modelo Turbo INT8:"
    ls -la "./models/csm-1b-turbo/model_int8.safetensors" | sed 's/^/      /'
fi
echo "============================================================"

# 11. Iniciar API
echo "🚀 11. Iniciando CSM Voice Cloning API..."

# Ejecutar API completa
python voice_api_complete.py
