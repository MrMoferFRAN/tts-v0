# 🚀 CONFIGURACIÓN RUNPOD - CSM VOICE CLONING

## 📋 Configuración del Container

### **Container Image**
```
runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04
```

### **RunPod Secrets (Recomendado - MÁS SEGURO)**
| Secret Name | Valor | Descripción |
|-------------|-------|-------------|
| `RUNPOD_SECRET_HF_TOKEN` | `hf_xxxxxxxxxxxxxxxxxxxxxxxxx` | Token de Hugging Face (REQUERIDO) |

**Cómo configurar RunPod Secrets:**
1. En RunPod, ve a **Settings → Secrets**
2. Crear nuevo Secret: `RUNPOD_SECRET_HF_TOKEN`
3. Valor: Tu token de Hugging Face
4. Al crear el pod, selecciona este Secret

### **Environment Variables (Alternativa)**
| Variable | Valor | Descripción |
|----------|-------|-------------|
| `HF_TOKEN` | `hf_xxxxxxxxxxxxxxxxxxxxxxxxx` | Token de Hugging Face (REQUERIDO) |

### **Container Start Command (recomendado)**
```bash
bash -c "cd /workspace && (git clone https://github.com/MrMoferFRANtts-v0.git || (cd runttspod && git pull origin main)) && cd runttspod && chmod +x startup.sh && ./startup.sh"
```

## 🎯 Especificaciones GPU Recomendadas

| GPU | VRAM | Costo/Hora | Rendimiento |
|-----|------|------------|-------------|
| **RTX 4090** | 24GB | ~$0.69/hr | ⭐⭐⭐⭐⭐ |
| **A100 80GB** | 80GB | ~$2.50/hr | ⭐⭐⭐⭐⭐ |
| **RTX 3090** | 24GB | ~$0.59/hr | ⭐⭐⭐⭐ |

## 🔧 Configuración de Discos

### **Container Disk**
- **Mínimo**: 50GB
- **Recomendado**: 100GB
- Almacena temporalmente: modelo CSM-1B (~6GB), dependencias, outputs

### **Volume Disk (Opcional)**
- **Uso**: Persistir modelos descargados y outputs
- **Mount Path**: `/workspace/persistent`
- **Tamaño**: 50GB+

## 🚀 Proceso de Arranque Automático

### **1. Clone automático del repositorio**
```bash
git clone https://github.com/MrMoferFRANtts-v0.git
```

### **2. Instalación automática de dependencias**
- transformers>=4.52.1
- accelerate>=0.20.0
- fastapi, uvicorn
- python-multipart, aiofiles

### **3. Descarga automática del modelo CSM-1B**
- Modelo: `sesame/csm-1b` (~6GB)
- Ubicación: `/workspacetts-v0/models/sesame-csm-1b/`

### **4. Inicio automático de la API**
- Puerto: `7860`
- Acceso: `http://[POD_IP]:7860`

## 🌐 Endpoints de la API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/` | GET | Página principal con documentación |
| `/health` | GET | Estado del sistema |
| `/voices` | GET | Lista de perfiles de voz |
| `/clone-voice` | POST | Clonación de voz |
| `/upload-voice` | POST | Subir nuevo perfil de voz |
| `/docs` | GET | Documentación interactiva Swagger |

## 🎯 Comandos de Prueba

### **Health Check**
```bash
curl http://[POD_IP]:7860/health
```

### **Listar Voces**
```bash
curl http://[POD_IP]:7860/voices
```

### **Clonar Voz**
```bash
curl -X POST 'http://[POD_IP]:7860/clone-voice' \
     -F 'text=Hola mundo desde RunPod' \
     -F 'temperature=0.7'
```

### **Con archivo de voz**
```bash
curl -X POST 'http://[POD_IP]:7860/clone-voice' \
     -F 'text=Este es un texto de prueba' \
     -F 'context_text=Transcripción del audio de referencia' \
     -F 'context_audio=@mi_voz.wav' \
     -F 'temperature=0.8'
```

## ⚡ Optimizaciones de Rendimiento

### **GPU Memory**
- Usa torch.float16 para menor uso de VRAM
- Gradient checkpointing habilitado
- Batch size optimizado automáticamente

### **Caching**
- Modelo se carga una vez al iniciar
- Audio profiles se mantienen en memoria
- Outputs se cachean en `/workspacetts-v0/outputs/`

## 🐛 Troubleshooting

### **Error: "HF_TOKEN no configurado"**
- **Solución**: Configurar variable de entorno en RunPod
- **Ubicación**: Environment Variables → `HF_TOKEN`

### **Error: "Model not found"**
- **Causa**: Fallo en descarga del modelo
- **Solución**: Reiniciar el pod (descarga automática)

### **Error: "CUDA Out of Memory"**
- **Causa**: GPU insuficiente
- **Solución**: Usar GPU con más VRAM (RTX 4090 24GB mínimo)

### **API no responde en puerto 7860**
- **Verificar**: `curl http://localhost:7860/health`
- **Log**: Revisar `/workspacetts-v0/logs/startup_[fecha].log`

## 💰 Estimación de Costos

### **Setup Inicial** (~15-30 min)
- Clone repo: 2 min
- Instalar deps: 5 min  
- Descargar modelo: 10 min
- **Costo**: ~$0.35 (RTX 4090)

### **Uso Continuo**
- Generación de voz: ~2-10 segundos
- **Costo por hora**: $0.69 (RTX 4090) / $2.50 (A100)

## 🎭 Perfiles de Voz

### **Por defecto**
El sistema incluye 1 perfil de voz de ejemplo en `/voices/`

### **Agregar nuevos perfiles**
```bash
# Método 1: API
curl -X POST 'http://[POD_IP]:7860/upload-voice' \
     -F 'name=mi_voz' \
     -F 'transcript=Texto de la grabación' \
     -F 'audio_file=@mi_audio.wav'

# Método 2: Manual
# Copiar archivos a /workspacetts-v0/voices/
# - mi_voz.wav (audio)
# - mi_voz.txt (transcripción)
```

## 📊 Monitoreo

### **Logs del sistema**
```bash
tail -f /workspacetts-v0/logs/startup_*.log
tail -f /workspacetts-v0/logs/voice_api.log
```

### **Uso de GPU**
```bash
watch nvidia-smi
```

### **Estado de la API**
```bash
curl http://localhost:7860/health | jq
```

## 📊 Configuración Recomendada RunPod

```
Container Image: runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04
RunPod Secret: RUNPOD_SECRET_HF_TOKEN=tu_token_aqui (RECOMENDADO)
Environment Variables: HF_TOKEN=tu_token_aqui (alternativa)
Container Start Command: bash -c "cd /workspace && (git clone https://github.com/MrMoferFRANtts-v0.git || (cd runttspod && git pull origin main)) && cd runttspod && chmod +x startup.sh && ./startup.sh"
Container Disk: 100GB
GPU: RTX 4090 24GB (mínimo)
``` 