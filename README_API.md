# 🎤 Advanced Voice Cloning API

Una API robusta de clonación de voz con streaming, monitoreo de rendimiento y optimización avanzada usando el modelo CSM-1B.

## 🚀 Características Principales

### ✨ Funcionalidades Core
- **Clonación de Voz**: Clona voces usando audio de referencia y transcripción
- **Streaming en Tiempo Real**: Generación de audio por chunks para baja latencia
- **Procesamiento por Lotes**: Múltiples textos en una sola solicitud
- **TTS Simple**: Síntesis de voz sin clonación

### 🔧 Optimizaciones Avanzadas
- **Optimización de GPU**: Gestión automática de memoria y configuración
- **Chunking Adaptativo**: Tamaño de chunks optimizado según recursos del sistema
- **Cache Inteligente**: LRU cache para audio de referencia
- **Monitoreo en Tiempo Real**: Métricas detalladas de rendimiento
- **Limpieza de Audio**: Eliminación automática de silencios excesivos

### 📊 Monitoreo y Performance
- **Métricas Detalladas**: Tiempo de procesamiento, factor de tiempo real, tokens/segundo
- **Estadísticas del Sistema**: CPU, RAM, memoria GPU en tiempo real
- **Perfilado de Performance**: Análisis granular de cada operación
- **Optimización Automática**: Ajuste dinámico según carga del sistema

## 📋 Requisitos del Sistema

### Mínimos
- **RAM**: 8GB (recomendado 16GB+)
- **GPU**: 6GB VRAM (recomendado 8GB+)
- **Almacenamiento**: 10GB libres
- **Python**: 3.8+

### Recomendados
- **RAM**: 32GB
- **GPU**: RTX 3080/4080 o superior (12GB+ VRAM)
- **CPU**: 8+ núcleos
- **SSD**: Para almacenamiento temporal

## 🛠️ Instalación

### 1. Clonar y Configurar
```bash
git clone <repository>
cd voice-cloning-api

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 2. Instalar Dependencias
```bash
pip install -r requirements_api.txt
```

### 3. Descargar Modelo CSM-1B
```bash
# Crear directorio del modelo
mkdir -p models/sesame-csm-1b

# Descargar modelo (ajustar según disponibilidad)
# wget https://huggingface.co/sesame/csm-1b/... 
# O seguir instrucciones del repositorio oficial
```

### 4. Verificar Instalación
```bash
python start_voice_api.py --check-only
```

## 🚀 Uso Rápido

### Iniciar el Servidor
```bash
# Inicio básico
python start_voice_api.py

# Con configuración personalizada
python start_voice_api.py \
    --host 0.0.0.0 \
    --port 8000 \
    --cache-size 4096 \
    --log-level DEBUG
```

### Parámetros de Inicio
- `--host`: Dirección IP (default: 0.0.0.0)
- `--port`: Puerto (default: 8000)
- `--workers`: Número de workers (default: 1)
- `--cache-size`: Tamaño de cache en MB (default: 2048)
- `--no-gpu`: Deshabilitar optimización GPU
- `--no-adaptive`: Deshabilitar chunking adaptativo
- `--reload`: Habilitar auto-reload para desarrollo

## 📡 Endpoints de la API

### 🏥 Health Check
```bash
GET /health
```
Verifica el estado del servidor y métricas del sistema.

### 🎭 Clonación de Voz
```bash
POST /clone-voice
```

**Parámetros:**
```json
{
    "text": "Texto a sintetizar",
    "reference_text": "Transcripción del audio de referencia",
    "speaker_id": "0",
    "temperature": 0.7,
    "chunk_size": null,
    "remove_silence": true,
    "streaming": false,
    "max_silence_duration": 0.5,
    "use_optimization": true
}
```

**Archivo:**
- `reference_audio`: Archivo de audio de referencia (opcional)

### 🌊 Streaming
```bash
POST /clone-voice-stream
```
Mismos parámetros que `/clone-voice` pero con `streaming: true`.

### 📦 Procesamiento por Lotes
```bash
POST /batch-clone-voice
```

**Parámetros:**
```json
{
    "texts": ["Texto 1", "Texto 2", "Texto 3"],
    "reference_text": "Transcripción de referencia",
    "speaker_id": "0",
    "temperature": 0.7,
    "chunk_size": null,
    "remove_silence": true,
    "max_silence_duration": 0.5,
    "use_optimization": true
}
```

### 📊 Métricas y Optimización

#### Estadísticas de Performance
```bash
GET /performance-stats
```

#### Configuración de Optimización
```bash
GET /optimization-config
POST /optimize-settings
```

#### Limpiar Cache
```bash
POST /clear-cache
```

#### Recomendación de Chunk Size
```bash
GET /chunk-size-recommendation?text=...&streaming=false
```

## 💻 Ejemplo de Cliente

### Cliente Python Básico
```python
import aiohttp
import asyncio

async def clone_voice_example():
    async with aiohttp.ClientSession() as session:
        # Datos de la solicitud
        data = {
            "text": "Hola, esto es una prueba de clonación de voz",
            "reference_text": "Texto del audio de referencia",
            "temperature": 0.7,
            "remove_silence": True
        }
        
        # Archivo de audio de referencia
        files = {'reference_audio': open('reference.wav', 'rb')}
        
        # Realizar solicitud
        async with session.post(
            'http://localhost:8000/clone-voice',
            data=data,
            files=files
        ) as response:
            result = await response.json()
            
            if result['success']:
                print(f"Audio generado: {result['audio_url']}")
                print(f"Tiempo de procesamiento: {result['performance_metrics']['processing_time']:.2f}s")
            else:
                print(f"Error: {result['error']}")

# Ejecutar
asyncio.run(clone_voice_example())
```

### Cliente de Demo Completo
```bash
python voice_cloning_client.py
```

### cURL Examples
```bash
# Health check
curl http://localhost:8000/health

# Clonación básica
curl -X POST "http://localhost:8000/clone-voice" \
     -F "text=Hola mundo" \
     -F "temperature=0.7" \
     -F "remove_silence=true"

# Con audio de referencia
curl -X POST "http://localhost:8000/clone-voice" \
     -F "text=Texto a clonar" \
     -F "reference_text=Transcripción de referencia" \
     -F "reference_audio=@reference.wav" \
     -F "temperature=0.8"
```

## ⚡ Optimización de Performance

### Configuración Automática
La API incluye optimización automática que ajusta:
- **Chunk Size**: Basado en carga del sistema y longitud del texto
- **Memoria GPU**: Gestión automática de memoria
- **Cache**: LRU cache para audio de referencia
- **Garbage Collection**: Limpieza automática cuando es necesario

### Configuración Manual
```python
# Actualizar configuración de optimización
response = requests.post('http://localhost:8000/optimize-settings', json={
    "max_cache_size_mb": 4096,
    "adaptive_chunking": True,
    "enable_gpu_optimization": True
})
```

### Mejores Prácticas

#### Para Máximo Rendimiento
1. **Use GPU**: Asegúrese de tener una GPU compatible
2. **Memoria Suficiente**: 16GB+ RAM, 8GB+ VRAM
3. **Cache Grande**: Configure cache según memoria disponible
4. **Chunking Adaptativo**: Mantenga habilitado para optimización automática

#### Para Streaming
1. **Chunks Pequeños**: Use chunk_size 50-75 para baja latencia
2. **Red Rápida**: Conexión estable para streaming fluido
3. **Buffer**: Implemente buffering en el cliente

#### Para Calidad
1. **Audio de Referencia**: Use audio claro y de buena calidad
2. **Transcripción Exacta**: Asegúrese de que la transcripción sea precisa
3. **Temperatura**: 0.7-0.8 para balance calidad/variabilidad

## 📈 Monitoreo y Métricas

### Métricas Principales
- **Realtime Factor**: `processing_time / audio_duration`
- **Tokens/Second**: Velocidad de procesamiento de texto
- **Memory Usage**: RAM y GPU en tiempo real
- **Cache Hit Ratio**: Eficiencia del cache

### Logging
```bash
# Ver logs en tiempo real
tail -f voice_api.log

# Filtrar errores
grep "ERROR" voice_api.log
```

### Dashboard de Métricas
```bash
# Obtener métricas completas
curl http://localhost:8000/performance-stats | jq
```

## 🔧 Troubleshooting

### Problemas Comunes

#### Error de Memoria GPU
```bash
# Reducir cache size
python start_voice_api.py --cache-size 1024

# Deshabilitar optimización GPU
python start_voice_api.py --no-gpu
```

#### Performance Lento
1. Verificar que GPU esté siendo utilizada
2. Aumentar chunk size para textos largos
3. Revisar memoria disponible
4. Considerar usar CPU si GPU es limitada

#### Errores de Modelo
1. Verificar que el modelo esté descargado completamente
2. Comprobar permisos de archivos
3. Validar integridad del modelo

### Comandos de Diagnóstico
```bash
# Check del sistema
python start_voice_api.py --check-only

# Métricas en tiempo real
watch -n 5 curl -s http://localhost:8000/performance-stats

# Limpiar cache si hay problemas de memoria
curl -X POST http://localhost:8000/clear-cache
```

## 🤝 Contribuir

1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia especificada en el archivo LICENSE.

## 🙏 Reconocimientos

- Modelo CSM-1B por el equipo de Sesame
- FastAPI por el framework web
- PyTorch por el backend de ML
- Librosa por procesamiento de audio

---

**🎤 ¡Disfruta clonando voces con performance optimizada!** 