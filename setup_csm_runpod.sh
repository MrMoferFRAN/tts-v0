#!/bin/bash

echo "🎯 SETUP CSM TTS + ELISE EN RUNPOD"
echo "================================"

# 1. Actualizar sistema
echo "📦 Actualizando sistema..."
apt update && apt upgrade -y
apt install -y git wget curl htop nvtop tree ffmpeg

# 2. Verificar CUDA
echo "🔍 Verificando CUDA..."
nvidia-smi
nvcc --version

# 3. Crear estructura de directorios
echo "📁 Creando estructura de directorios..."
mkdir -p /workspace/{csm-tts,datasets,models,outputs,scripts}
cd /workspace/csm-tts

# 4. Instalar dependencias básicas
echo "🔧 Instalando dependencias Python..."
pip install --upgrade pip
pip install jupyter numpy scipy matplotlib tqdm
pip install librosa soundfile audiofile audresample
pip install huggingface-hub transformers datasets
pip install wandb tensorboard

# 5. Clonar repositorio CSM original (CUDA)
echo "📥 Clonando CSM original..."
git clone https://github.com/p0p4k/csm.git
cd csm

# 6. Instalar CSM requirements
echo "🔧 Instalando CSM requirements..."
pip install -r requirements.txt
pip install -e .

# 7. Descargar modelo base CSM
echo "📥 Descargando modelo base CSM..."
cd /workspace/models
wget -O csm-1b.safetensors "https://huggingface.co/p0p4k/csm/resolve/main/model.safetensors"

# 8. Descargar dataset Elise
echo "📥 Descargando dataset Elise..."
cd /workspace/datasets
git clone https://huggingface.co/datasets/MrDragonFox/Elise
cd Elise
# Extraer audios y metadatos

# 9. Crear scripts de entrenamiento
echo "📝 Creando scripts..."
cat > /workspace/scripts/train_elise.py << 'EOF'
#!/usr/bin/env python3
"""
Script de entrenamiento Elise con CSM CUDA
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
import wandb
import os

def main():
    print("🎭 Iniciando entrenamiento de Elise...")
    
    # Configurar device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Usando device: {device}")
    
    # Configurar WandB para tracking
    wandb.init(project="elise-csm-tts", name="elise-finetune")
    
    # TODO: Implementar lógica de entrenamiento
    
if __name__ == "__main__":
    main()
EOF

# 10. Configurar Jupyter
echo "🪐 Configurando Jupyter..."
jupyter lab --generate-config
cat >> ~/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.allow_root = True
EOF

# 11. Crear script de inicio
cat > /workspace/start.sh << 'EOF'
#!/bin/bash
echo "🎯 Iniciando entorno CSM TTS..."
cd /workspace/csm-tts
export CUDA_VISIBLE_DEVICES=0
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
echo "🪐 Jupyter Lab iniciado en puerto 8888"
echo "🎭 Listo para entrenar Elise!"
EOF

chmod +x /workspace/start.sh

echo "✅ Setup completado!"
echo "🎯 Para iniciar: cd /workspace && ./start.sh"
echo "🪐 Jupyter estará en: http://[POD_IP]:8888" 