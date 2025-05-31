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

# 4. Start services with monitoring
echo "🚀 Starting services with monitoring..."
python service_manager.py 