# 🔧 **GUÍA DE INSTALACIÓN VOICE CLONING - RUNPOD OPTIMIZADA**

## 📋 **ENTORNO DE REFERENCIA**
```bash
# Base Container: runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04
# GPU: NVIDIA A100 80GB PCIe (85.1 GB VRAM)
# CUDA: 12.1 + CuDNN 90100
# Python: 3.10
# Sistema: voice_cloning + CSM-1B
```

## 🚀 **INSTALACIÓN PASO A PASO (CORREGIDA)**

### **1. INSTALACIÓN AUTOMÁTICA (RECOMENDADO)**
```bash
cd /workspace/runttspod
./startup.sh
```

**Este script hace:**
- ✅ Descarga modelo CSM-1B (5.8GB) 
- ✅ Descarga dataset Elise
- ✅ Instala dependencias voice_cloning
- ✅ Configura estructura de directorios
- ✅ Inicia API en puerto 7860

### **2. VERIFICAR DEPENDENCIAS**
```bash
# Dependencias voice_cloning
pip install -r voice_cloning/requirements.txt

# Dependencias de la API
pip install -r requirements_api.txt

# Dependencias adicionales
pip install peft>=0.4.0 fastapi uvicorn python-multipart aiofiles
```

### **3. ESTRUCTURA CORRECTA**
```
/workspace/runttspod/
├── voice_cloning/          # 🎤 Sistema principal de clonación
│   ├── __init__.py
│   ├── voice_clone.py     # Clase VoiceCloner
│   ├── models.py          # Configuración CSM-1B
│   ├── generator.py       # Generador compatible
│   └── requirements.txt   # Dependencias base
├── models/
│   └── sesame-csm-1b/     # 🤖 Modelo CSM-1B (5.8GB)
├── datasets/
│   └── csm-1b-elise/      # 🎭 Dataset para fine-tuning
├── voices/                # 📢 Perfiles de voz
├── outputs/               # 💾 Audio generado
├── quick_start.py         # 🚀 API optimizada
└── startup.sh             # 🔧 Instalación automática
```

## ✅ **VERIFICAR INSTALACIÓN**

### **Test 1: Importaciones**
```python
from voice_cloning import VoiceCloner
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print("Voice Cloning: OK")
```

### **Test 2: Modelo**
```python
from pathlib import Path
model_path = Path("./models/sesame-csm-1b")
print(f"Modelo existe: {model_path.exists()}")

if model_path.exists():
    model_file = model_path / "model.safetensors"
    print(f"Archivo principal: {model_file.exists()}")
```

### **Test 3: API**
```bash
# Iniciar API
./quick_api_start.sh

# En otra terminal, probar
curl http://localhost:7860/health
curl http://localhost:7860/voices
```

## 🎯 **USO DEL SISTEMA**

### **Método 1: Via API REST**
```bash
# Listar voces disponibles
curl http://localhost:7860/voices

# Clonar voz
curl -X POST 'http://localhost:7860/clone-voice' \
     -F 'text=Hola, esto es una prueba' \
     -F 'voice_name=voices' \
     -F 'temperature=0.7'
```

### **Método 2: Via Python**
```python
from voice_cloning import VoiceCloner

# Inicializar
cloner = VoiceCloner(model_path="./models/sesame-csm-1b")

# Clonar desde archivo
output = cloner.clone_voice_from_file(
    reference_audio="voices/reference.mp3",
    reference_transcript="Transcripción del audio",
    target_text="Texto a sintetizar",
    output_path="output.wav"
)
```

### **Método 3: Generación por lotes**
```python
textos = [
    "Primera frase",
    "Segunda frase", 
    "Tercera frase"
]

outputs = cloner.batch_generate(
    text_list=textos,
    context_text="Transcripción de referencia",
    context_audio_path="audio_referencia.mp3",
    output_dir="outputs/batch"
)
```

## 🔧 **OPTIMIZACIONES DISPONIBLES**

### **csm-tts Optimizaciones**
El proyecto incluye scripts optimizados en `csm-tts/`:
- `test_csm_optimized.py` - Uso intensivo de recursos A100
- `test_csm_simple.py` - Pruebas básicas
- Monitoreo de recursos en tiempo real
- Procesamiento por lotes optimizado

### **Configuración Producción**
```python
# En quick_start.py
server.setup_optimization(
    gpu_optimization=True,
    max_cache_mb=6144,        # 6GB cache
    adaptive_chunking=True,
    max_concurrent=3,         # 3 requests simultáneos
    production=True,          # Modo producción
    gpu_memory_fraction=0.85  # 85% de VRAM
)
```

## 🚨 **DIFERENCIAS CLAVE vs CSM GITHUB**

### **❌ NO Usar CSM de GitHub:**
```bash
# INCORRECTO - No hacer esto:
git clone https://github.com/p0p4k/csm.git
pip install -e models/csm/
```

### **✅ Usar voice_cloning local:**
```bash
# CORRECTO - Sistema actual:
from voice_cloning import VoiceCloner
# Usa: ./models/sesame-csm-1b/ (descargado automáticamente)
```

### **Dependencias Correctas:**
- **voice_cloning/requirements.txt** - Sistema principal
- **requirements_api.txt** - API REST
- **NO usar** models/csm/requirements.txt

## 🎉 **SISTEMA LISTO**

Después del `startup.sh` exitoso:
- ✅ **Voice Cloning API**: http://0.0.0.0:7860
- ✅ **Documentación**: http://0.0.0.0:7860/docs
- ✅ **Modelo CSM-1B**: Cargado y optimizado
- ✅ **Dataset Elise**: Disponible para fine-tuning
- ✅ **GPU A100**: Configurada para máximo rendimiento

**🚀 El sistema está optimizado para voice_cloning, no para CSM de GitHub!** 