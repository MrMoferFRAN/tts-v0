# 🎤 Voice Cloning API Complete - CSM-1B Turbo

API completa de clonación de voz con gestión avanzada de perfiles y **modo turbo** para inferencia ultrarrápida.

## 🚀 Nuevas Características - Modo Turbo

### ✨ Características Principales

- **🎯 Dual Model Support**: Modelo normal (float32) y turbo (int8 cuantizado)
- **⚡ Modo Turbo**: Inferencia ultrarrápida con modelo cuantizado int8
- **📁 Gestión por Carpetas**: Cada voz tiene su propia carpeta con múltiples muestras
- **📤 Upload Inteligente**: Validación automática 3-9s, WAV 24kHz mono normalizado
- **🎯 Clonación Precisa**: Selección específica de muestras para mejor calidad
- **📊 Análisis Completo**: Estadísticas detalladas y métricas de calidad

### 🏗️ Estructura de Modelos

```
models/
├── sesame-csm-1b/           # Modelo normal (float32)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
└── csm-1b-turbo/            # Modelo turbo (int8 cuantizado)
    ├── config.json
    ├── model_int8.safetensors  # ← Modelo cuantizado principal
    ├── tokenizer.json
    └── ...
```

## 🚀 Instalación y Configuración

### 1. Descargar Modelo Turbo

```bash
# El modelo normal ya está disponible
# Descargar modelo turbo cuantizado
mkdir -p models/csm-1b-turbo
cd models/csm-1b-turbo
wget https://huggingface.co/lunahr/csm-1b-safetensors-quants/resolve/main/model_int8.safetensors

# Copiar archivos de configuración del modelo normal
cp -r ../sesame-csm-1b/* ./
```

### 2. Iniciar API

```bash
python voice_api_complete.py
```

La API cargará automáticamente ambos modelos al iniciar:
- ✅ Modelo normal: Mayor calidad, más lento
- 🚀 Modelo turbo: Velocidad ultrarrápida, calidad comparable

## 📋 API Endpoints

### 🔍 Health Check Mejorado

```bash
GET /health
```

**Respuesta:**
```json
{
  "status": "healthy",
  "normal_model": {
    "loaded": true,
    "processor_loaded": true,
    "path": "./models/sesame-csm-1b"
  },
  "turbo_model": {
    "loaded": true,
    "processor_loaded": true,
    "path": "./models/csm-1b-turbo",
    "available": true
  },
  "gpu_available": true,
  "gpu_info": {...},
  "voice_collections": 2,
  "total_voice_samples": 5
}
```

### 🎤 Clonación de Voz con Modo Turbo

```bash
POST /clone
```

**Parámetros:**
- `text` (requerido): Texto a sintetizar
- `voice_id` (opcional): ID de la colección de voz
- `sample_name` (opcional): Nombre específico de muestra
- `temperature` (opcional): Temperatura de sampling (default: 0.8)
- `max_tokens` (opcional): Máximo tokens a generar (default: 512)
- **`turbo` (nuevo)**: Usar modo turbo (default: false)
- `output_format` (opcional): Formato de salida (default: wav)

## 🧪 Ejemplos de Uso

### 1. Modo Normal (Mayor Calidad)

```bash
curl -X POST 'http://localhost:7860/clone' \
  -F 'text=Hola, soy una voz clonada con alta calidad' \
  -F 'voice_id=fran-fem' \
  -F 'turbo=false'
```

### 2. Modo Turbo (Ultrarrápido)

```bash
curl -X POST 'http://localhost:7860/clone' \
  -F 'text=Hola, soy una voz clonada ultra rápida' \
  -F 'voice_id=fran-fem' \
  -F 'turbo=true'
```

### 3. Comparación de Velocidad

```python
import requests
import time

# Modo normal
start = time.time()
response_normal = requests.post('http://localhost:7860/clone', data={
    'text': 'Texto de prueba',
    'voice_id': 'mi-voz',
    'turbo': 'false'
})
time_normal = time.time() - start

# Modo turbo
start = time.time()
response_turbo = requests.post('http://localhost:7860/clone', data={
    'text': 'Texto de prueba',
    'voice_id': 'mi-voz',
    'turbo': 'true'
})
time_turbo = time.time() - start

speedup = time_normal / time_turbo
print(f"Speedup: {speedup:.2f}x más rápido")
```

## 🧪 Script de Prueba

Incluimos un script de prueba que compara automáticamente ambos modelos:

```bash
python test_turbo.py
```

**Funciones del script:**
- ✅ Verifica estado de ambos modelos
- 🎵 Genera audio con ambos modelos
- ⏱️ Mide y compara velocidades
- 📊 Calcula speedup automáticamente
- 💾 Guarda archivos de prueba

## 📊 Benchmarks Esperados

| Modo | Velocidad | Calidad | Uso RAM | Uso VRAM |
|------|-----------|---------|---------|----------|
| Normal | 1.0x | Alta | ~8GB | ~6GB |
| Turbo | **2-3x** | Comparable | ~6GB | ~4GB |

## 🔧 Configuración Avanzada

### Variables de Entorno

```bash
export NO_TORCH_COMPILE=1           # Desactivar compilación torch
export HF_TOKEN=your_token_here     # Token Hugging Face
export CUDA_VISIBLE_DEVICES=0       # GPU específica
```

### Parámetros de Modelo

```python
# En voice_api_complete.py
manager = CSMVoiceManager(
    model_path="./models/sesame-csm-1b",     # Modelo normal
    turbo_model_path="./models/csm-1b-turbo", # Modelo turbo
    voices_dir="./voices"                     # Directorio de voces
)
```

## 🚀 Optimizaciones Implementadas

### Modelo Turbo (int8)
- **Cuantización int8**: Reduce tamaño del modelo ~50%
- **Memoria optimizada**: Menor uso de VRAM
- **Compatibilidad**: Misma API, misma calidad
- **Speed boost**: 2-3x más rápido

### Carga Dual
- **Precarga**: Ambos modelos se cargan al inicio
- **Selección dinámica**: Cambio instantáneo entre modelos
- **Fallback**: Si turbo falla, usa modelo normal automáticamente

## 🎯 Casos de Uso Recomendados

### Modo Normal
- ✅ Producción de alta calidad
- ✅ Voces para contenido final
- ✅ Cuando la calidad es prioritaria

### Modo Turbo
- ✅ Prototipado rápido
- ✅ Aplicaciones en tiempo real
- ✅ Pruebas y desarrollo
- ✅ APIs con alta concurrencia

## 🔍 Troubleshooting

### Error: Turbo model not available
```bash
# Verificar que el modelo turbo esté descargado
ls -la models/csm-1b-turbo/model_int8.safetensors

# Si no existe, descargar:
cd models/csm-1b-turbo
wget https://huggingface.co/lunahr/csm-1b-safetensors-quants/resolve/main/model_int8.safetensors
```

### Memoria insuficiente
```python
# Usar modo turbo para reducir uso de memoria
response = requests.post('/clone', data={
    'text': 'Mi texto',
    'turbo': 'true'  # Menor uso de VRAM
})
```

## 📈 Roadmap

- [ ] **Modelos adicionales**: Soporte para más variantes cuantizadas
- [ ] **Auto-selection**: Selección automática según carga del sistema
- [ ] **Streaming**: Generación de audio en tiempo real
- [ ] **Batch processing**: Procesamiento de múltiples textos
- [ ] **Model caching**: Cache inteligente para cambio de modelos

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature
3. Prueba con ambos modelos (normal y turbo)
4. Envía un pull request

## 📄 Licencia

Este proyecto utiliza el modelo CSM-1B bajo su licencia correspondiente.

---

**🚀 ¡Disfruta de la velocidad ultrarrápida del modo turbo!** 