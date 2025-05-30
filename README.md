# 🎯 RunPod Setup: CSM TTS + Elise Finetuning (CUDA)

**Plan B: Usar RunPod con CUDA para finetuning real de CSM + Elise**

## 🚀 **ESTADO ACTUAL DEL POD ACTIVADO**

### ✅ **Especificaciones de Hardware Actual:**
- **GPU**: NVIDIA A100 80GB PCIe (**80GB VRAM** 🔥)
- **CPU**: AMD EPYC 7763 64-Core Processor (31 vCPU)
- **RAM**: 944GB total (831GB disponible)
- **Almacenamiento**: 294TB total (135TB disponible)
- **CUDA**: 12.7 (Driver 565.57.01)
- **PyTorch**: 2.1.1+cu121
- **Python**: 3.10.12

### 🎯 **Capacidad de Entrenamiento:**
Con **80GB de VRAM**, este pod puede manejar:
- ✅ **Full finetuning** de CSM (sin limitaciones)
- ✅ **Batch sizes grandes** (16-32+ dependiendo del modelo)
- ✅ **Modelos grandes** sin técnicas de optimización
- ✅ **Entrenamiento simultáneo** de múltiples experimentos
- ✅ **Zero problemas de memoria** con Elise dataset

## 📋 Especificaciones de Hardware Recomendadas (REFERENCIA)

### 🎯 **Hardware Mínimo Recomendado:**
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) o superior
- **VRAM**: Mínimo 16GB, **recomendado 24GB+**
- **RAM**: 32GB+ (para datasets grandes)
- **Almacenamiento**: 100GB+ SSD

### 🔥 **Hardware Óptimo para Elise:**
- **GPU**: NVIDIA A100 40GB/80GB o H100
- **VRAM**: 40GB+ para full finetuning
- **RAM**: 64GB+ 
- **Almacenamiento**: 200GB+ NVMe SSD

### 💰 **Opciones de Costo-Beneficio:**
1. **RTX 4090** (24GB) - ~$0.69/hora - Para modelos pequeños/medianos
2. **RTX A6000** (48GB) - ~$1.20/hora - Balance perfecto
3. **A100 40GB** - ~$2.50/hora - Para modelos grandes
4. **H100** - ~$4.00/hora - Máximo rendimiento

## 🚀 Configuración en RunPod

### 1. **Reservar Máquina**

**Especificaciones Recomendadas:**
```
GPU: RTX A6000 (48GB VRAM) o A100
RAM: 64GB+
Storage: 200GB+ SSD
Region: US-West o EU (menor latencia)
Type: On-Demand (para estabilidad)
```

### 2. **Container/Imagen Recomendada**

**Imagen Base Oficial:**
```bash
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

**Características:**
- Ubuntu 22.04 LTS
- Python 3.11
- PyTorch 2.4.0
- CUDA 12.4.1 + cuDNN
- Jupyter Lab pre-instalado
- SSH access habilitado

**Alternativas de Imagen:**
```bash
# Para máxima compatibilidad
runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Para último PyTorch
runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu
```

### 3. **Network Volume**
```
Tamaño: 100GB mínimo, 200GB recomendado
Tipo: NVMe SSD
Propósito: 
- Datasets de entrenamiento
- Checkpoints del modelo
- Resultados de generación
- Backup de configuraciones
```

## 🔧 Script de Setup Inicial

Guarda este script como `setup_csm_runpod.sh`:

```bash
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
```

## 📊 Configuración de Entrenamiento

### **Parámetros Recomendados para Elise:**

```python
# Configuración de entrenamiento optimizada
training_config = {
    "model_name": "csm-1b-elise",
    "base_model": "/workspace/models/csm-1b.safetensors",
    "dataset_path": "/workspace/datasets/Elise",
    
    # Hyperparámetros
    "learning_rate": 5e-5,
    "batch_size": 8,  # Ajustar según VRAM
    "gradient_accumulation_steps": 4,
    "max_epochs": 10,
    "warmup_steps": 500,
    
    # Técnicas de optimización
    "use_lora": True,  # Para reducir VRAM
    "lora_rank": 16,
    "use_gradient_checkpointing": True,
    "fp16": True,  # Mixed precision
    
    # Monitoreo
    "eval_steps": 100,
    "save_steps": 500,
    "logging_steps": 50,
    "wandb_project": "elise-csm-tts"
}
```

### **Estimación de Recursos:**

| Técnica | VRAM Necesaria | Tiempo Estimado | Costo/Hora |
|---------|----------------|-----------------|------------|
| Full Finetuning | 40GB+ | 24-48h | $2.50-4.00 |
| LoRA | 16GB+ | 12-24h | $0.69-1.20 |
| QLoRA | 12GB+ | 16-32h | $0.69 |

## 🔥 Workflow de Entrenamiento

### **Fase 1: Preparación (30 min)**
```bash
# 1. Conectar a RunPod
ssh root@[POD_IP]

# 2. Ejecutar setup
cd /workspace
chmod +x setup_csm_runpod.sh
./setup_csm_runpod.sh

# 3. Verificar instalación
python -c "import torch; print(torch.cuda.is_available())"
python -c "import csm; print('CSM OK')"
```

### **Fase 2: Preparación de Datos (1-2h)**
```bash
# 1. Procesar dataset Elise
cd /workspace/scripts
python process_elise_dataset.py

# 2. Crear splits train/val/test
python create_data_splits.py

# 3. Verificar calidad de datos
python validate_dataset.py
```

### **Fase 3: Entrenamiento (8-48h)**
```bash
# 1. Iniciar entrenamiento con LoRA
python train_elise.py \
    --config configs/elise_lora.yaml \
    --output_dir /workspace/outputs/elise-lora \
    --wandb_project elise-csm-tts

# 2. Monitorear progreso
tensorboard --logdir /workspace/outputs/elise-lora/logs
```

### **Fase 4: Evaluación (1h)**
```bash
# 1. Generar muestras de test
python generate_samples.py \
    --model_path /workspace/outputs/elise-lora/final \
    --test_texts test_emotions.txt

# 2. Evaluar calidad emocional
python evaluate_emotions.py \
    --generated_dir /workspace/outputs/samples \
    --reference_dir /workspace/datasets/Elise/test
```

## 📁 Estructura de Archivos

```
/workspace/
├── csm-tts/              # Repositorio CSM principal
│   ├── csm/              # Código fuente CSM
│   ├── configs/          # Configuraciones de entrenamiento
│   └── scripts/          # Scripts auxiliares
├── datasets/             # Datasets de entrenamiento
│   ├── Elise/           # Dataset Elise original
│   ├── processed/       # Datos procesados
│   └── splits/          # Train/val/test splits
├── models/              # Modelos y checkpoints
│   ├── csm-1b.safetensors # Modelo base
│   ├── elise-checkpoints/ # Checkpoints durante entrenamiento
│   └── final/           # Modelo final entrenado
├── outputs/             # Resultados de generación
│   ├── samples/         # Muestras generadas
│   ├── logs/            # Logs de entrenamiento
│   └── evaluations/     # Métricas de evaluación
└── scripts/             # Scripts personalizados
    ├── train_elise.py   # Script principal de entrenamiento
    ├── generate_samples.py # Generación de muestras
    └── evaluate_emotions.py # Evaluación de emociones
```

## 🔍 Troubleshooting

### **Problemas Comunes:**

1. **CUDA Out of Memory:**
   ```bash
   # Reducir batch_size
   batch_size = 4  # o menor
   
   # Usar gradient checkpointing
   use_gradient_checkpointing = True
   
   # Cambiar a fp16
   fp16 = True
   ```

2. **Dataset Corrupto:**
   ```bash
   # Verificar archivos de audio
   python scripts/validate_audio_files.py
   
   # Re-procesar dataset
   python scripts/clean_dataset.py
   ```

3. **Convergencia Lenta:**
   ```bash
   # Ajustar learning rate
   learning_rate = 1e-4  # o 1e-5
   
   # Usar warmup
   warmup_steps = 1000
   ```

## 💰 Optimización de Costos

### **Estrategias para Reducir Costos:**

1. **Usar Spot Instances** (50% descuento)
   - Riesgo: Posible interrupción
   - Mitigation: Guardar checkpoints frecuentes

2. **Técnicas de Memory Efficient Training:**
   - LoRA/QLoRA en lugar de full finetuning
   - Gradient checkpointing
   - Mixed precision (fp16)

3. **Programar Entrenamiento:**
   - Entrenar durante horas de menor demanda
   - Usar regiones con menor costo

4. **Monitoreo de Uso:**
   ```bash
   # Script para parar automáticamente
   python monitor_training.py --max_hours 24 --auto_stop
   ```

## 🎯 Siguiente Paso

Una vez creado este README:

1. **Reservar RunPod** con especificaciones recomendadas
2. **Crear nuevo directorio** del proyecto
3. **Sincronizar** código con el pod
4. **Ejecutar setup** y comenzar entrenamiento

¿Quieres que prepare el **código de entrenamiento específico** para Elise con CSM CUDA? 