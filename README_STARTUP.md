# 🚀 **RUNPOD CSM TTS - GUÍA DE INICIO**

## 📋 **SCRIPTS DISPONIBLES**

### **1. 🔧 `startup.sh` - INSTALACIÓN COMPLETA**
**Para primera vez o instalación desde cero:**

```bash
cd /workspacetts-v0
./startup.sh
```

**Lo que hace:**
- ✅ Verifica entorno y GPU
- ✅ Descarga modelo CSM (5.8GB) si no existe
- ✅ Descarga dataset Elise si no existe  
- ✅ Instala dependencias optimizadas
- ✅ Configura CSM en modo editable
- ✅ Crea estructura de directorios
- ✅ Inicia automáticamente la API en puerto 7860

### **2. ⚡ `quick_api_start.sh` - INICIO RÁPIDO**
**Para cuando ya tienes todo instalado:**

```bash
cd /workspacetts-v0
./quick_api_start.sh
```

**Lo que hace:**
- ✅ Verifica que CSM esté instalado
- ✅ Configura variables de entorno
- ✅ Inicia directamente la API

## 🌐 **ACCESO A LA API**

Una vez iniciada cualquiera de las opciones:

- **🎤 Voice Cloning API**: http://localhost:7860
- **📚 Documentación**: http://localhost:7860/docs
- **🎭 Perfiles de voz**: http://localhost:7860/voices

## 📊 **ESTADO DEL SISTEMA**

### **✅ VERIFICADO Y FUNCIONANDO:**
```bash
# Entorno: runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04
# GPU: NVIDIA A100 80GB PCIe (85.1 GB VRAM)
# PyTorch: 2.4.0+cu121 (Compatible CUDA 12.1)
# CSM: Instalado en modo editable
# Modelo: /workspacetts-v0/models/csm-1b.safetensors (5.8 GB)
# Dataset: /workspacetts-v0/datasets/csm-1b-elise/
```

## 🎯 **COMANDOS DE VERIFICACIÓN**

```bash
# Verificar instalación completa
python test_csm_installation.py

# Probar CSM directamente
cd models/csm
python -c "from generator import load_csm_1b; print('✅ CSM OK')"

# Ver logs de la API
tail -f /tmp/voice_api.log  # (si está configurado)
```

## 🛠️ **TROUBLESHOOTING**

### **Problema: API no inicia**
```bash
# Verificar que CSM esté instalado
python test_csm_installation.py

# Reinstalar si es necesario
./startup.sh
```

### **Problema: Error de CUDA**
```bash
# Verificar GPU
nvidia-smi

# Verificar PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### **Problema: Modelo no encontrado**
```bash
# Verificar modelo
ls -la models/csm-1b.safetensors

# Redownload si es necesario
rm models/csm-1b.safetensors
./startup.sh
```

## 🎤 **EJEMPLOS DE USO DE LA API**

### **1. Clonar voz con texto:**
```bash
curl -X POST 'http://localhost:7860/clone-voice' \
     -F 'text=Hola, esto es una prueba de clonación de voz' \
     -F 'voice_name=voices' \
     -F 'temperature=0.7'
```

### **2. Listar voces disponibles:**
```bash
curl http://localhost:7860/voices
```

### **3. Subir nueva voz:**
```bash
curl -X POST 'http://localhost:7860/upload-voice' \
     -F 'name=mi_voz' \
     -F 'audio=@mi_audio.wav' \
     -F 'transcription=Texto de referencia'
```

## 📁 **ESTRUCTURA DEL PROYECTO**

```
/workspacetts-v0/
├── startup.sh                 # 🔧 Instalación completa
├── quick_api_start.sh         # ⚡ Inicio rápido
├── quick_start.py             # 🎤 API de Voice Cloning
├── test_csm_installation.py   # ✅ Verificación
├── models/
│   ├── csm/                   # Código fuente CSM
│   ├── csm-1b.safetensors    # Modelo (5.8 GB)
│   └── sesame-csm-1b/        # Symlinks compatibilidad
├── datasets/
│   └── csm-1b-elise/         # Dataset Elise
├── outputs/                   # Archivos generados
└── voices/                    # Perfiles de voz
```

## 🎉 **READY TO USE!**

El sistema está optimizado para el entorno RunPod y listo para:
- ✅ **Voice Cloning en tiempo real**
- ✅ **Entrenamiento con dataset Elise**
- ✅ **API REST completa**
- ✅ **Máximo aprovechamiento de A100 80GB** 