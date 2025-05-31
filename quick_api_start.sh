#!/bin/bash
# 🚀 QUICK API START - Voice Cloning System (cuando ya está instalado)

echo "🎯 QUICK START - VOICE CLONING API"
echo "============================================"

cd /workspace/runttspod

# Verificar que voice_cloning está instalado
echo "🔍 Verificando instalación voice_cloning..."
python -c "
import sys
sys.path.insert(0, '/workspace/runttspod')
sys.path.insert(0, '/workspace/runttspod/voice_cloning')

try:
    from voice_cloning import VoiceCloner
    print('✅ Voice Cloning System OK')
except Exception as e:
    print(f'❌ Error: {e}')
    exit(1)

try:
    from pathlib import Path
    model_path = Path('./models/sesame-csm-1b')
    if model_path.exists():
        print('✅ Modelo CSM-1B encontrado')
    else:
        print('❌ Modelo CSM-1B no encontrado')
        exit(1)
except Exception as e:
    print(f'❌ Error verificando modelo: {e}')
    exit(1)
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "❌ Voice Cloning no está instalado. Ejecuta ./startup.sh primero"
    exit 1
fi

# Configurar entorno
export PYTHONPATH="/workspace/runttspod:/workspace/runttspod/voice_cloning:$PYTHONPATH"
export NO_TORCH_COMPILE=1
export HF_TOKEN=|==>REMOVED
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Verificar GPU
echo "🔍 Verificando GPU..."
python -c "import torch; print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}'); print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB' if torch.cuda.is_available() else '')"

# Verificar perfiles de voz
echo "🔍 Verificando perfiles de voz..."
if [ -f "voices/Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3" ]; then
    echo "✅ Perfil de voz 'voices' disponible"
else
    echo "⚠️ Perfil de voz de referencia no encontrado"
fi

# Iniciar API
echo "🚀 Iniciando Voice Cloning API en puerto 7860..."
echo "============================================"
echo "🌐 ACCESO:"
echo "   • URL: http://0.0.0.0:7860"
echo "   • Docs: http://0.0.0.0:7860/docs"
echo "   • Health: http://0.0.0.0:7860/health"
echo "   • Voices: http://0.0.0.0:7860/voices"
echo "============================================"
echo "🎯 PRUEBA RÁPIDA:"
echo "   curl http://localhost:7860/voices"
echo "============================================"
echo "🛑 Presiona Ctrl+C para detener"
echo "============================================"

python quick_start.py 