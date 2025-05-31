#!/bin/bash
# Autonomous Startup Script for RunPod TTS Server

# 1. Environment Verification
echo "🔍 Verifying system environment..."
python3 system_checks/verify_environment.py

# 2. Install dependencies
echo "📦 Installing required packages..."
pip install --upgrade pip
pip install -r csm-tts/csm/requirements.txt
pip install -r requirements_api.txt
pip install torch-sparse==0.6.18 \
  -f https://data.pyg.org/whl/torch-2.1.1+cu121.html # Fix for torch sparse module

export NO_TORCH_COMPILE=1

# 3. Install system executables
echo "🛠️  Installing audio tools..."
apt-get update
apt-get install -y ffmpeg sox libsndfile1-dev

# 4. Model verification and download
# echo "🔍 Verificando modelos..."

# # Verificar modelo base CSM
# CSM_MODEL_PATH="/workspace/models/csm-1b.safetensors"
# if [ ! -f "$CSM_MODEL_PATH" ]; then
#     echo "📥 Descargando modelo base CSM..."
#     mkdir -p /workspace/models
#     cd /workspace/models
#     wget -O csm-1b.safetensors "https://huggingface.co/p0p4k/csm/resolve/main/model.safetensors"
#     if [ $? -eq 0 ]; then
#         echo "✅ Modelo CSM descargado correctamente"
#     else
#         echo "❌ Error descargando modelo CSM"
#         exit 1
#     fi
# else
#     echo "✅ Modelo CSM ya existe: $CSM_MODEL_PATH"
# fi

# # Verificar dataset Elise
# ELISE_DATASET_PATH="/workspace/datasets/Elise"
# if [ ! -d "$ELISE_DATASET_PATH" ]; then
#     echo "📥 Descargando dataset Elise..."
#     mkdir -p /workspace/datasets
#     cd /workspace/datasets
#     git clone https://huggingface.co/datasets/MrDragonFox/Elise
#     if [ $? -eq 0 ]; then
#         echo "✅ Dataset Elise descargado correctamente"
#     else
#         echo "❌ Error descargando dataset Elise"
#         exit 1
#     fi
# else
#     echo "✅ Dataset Elise ya existe: $ELISE_DATASET_PATH"
# fi


# 5. Start services with monitoring
echo "🚀 Starting services with monitoring..."
python service_manager.py 