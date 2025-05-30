# 🚀 Voice Cloning API - Quick Start Guide

Configuración optimizada para máximo rendimiento en puerto **7860** con perfil de voz "**voices**" preconfigurado.

## ⚡ Inicio Rápido

### 1. Iniciar el Servidor (Configuración Optimizada)
```bash
# Inicio optimizado con configuración automática
python quick_start.py

# O con el script estándar pero optimizado
python start_voice_api.py --production --cache-size 6144 --max-concurrent 3
```

### 2. Verificar que Todo Funciona
```bash
# Verificar estado de la API
python voice_commands.py status

# Listar voces disponibles
python voice_commands.py voices
```

### 3. Prueba Básica
```bash
# Clonar voz con el perfil preconfigurado "voices"
python voice_commands.py clone "Hola, esta es una prueba de clonación de voz"

# Con configuración personalizada
python voice_commands.py clone "Texto a sintetizar" --voice voices --temperature 0.8
```

## 🎭 Tu Configuración Actual

### Audio de Referencia
- **Archivo**: `Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3`
- **Transcripción**: "Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo."
- **Perfil**: `voices` (configurado automáticamente)

### Configuración del Servidor
- **Puerto**: 7860
- **Cache**: 6GB (optimizado para rendimiento)
- **Requests concurrentes**: 3
- **Chunking adaptativo**: Habilitado
- **Optimización GPU**: Habilitada
- **Modo producción**: Activado

## 📝 Comandos Esenciales

### Usar Voice Cloning
```bash
# Básico con perfil "voices"
python voice_commands.py clone "Tu texto aquí"

# Con streaming (tiempo real)
python voice_commands.py clone "Tu texto aquí" --stream

# Con temperatura personalizada
python voice_commands.py clone "Tu texto aquí" --temperature 0.8

# Guardar en archivo específico
python voice_commands.py clone "Tu texto aquí" --output mi_audio.wav
```

### Gestionar Voces
```bash
# Ver voces disponibles
python voice_commands.py voices

# Agregar nueva voz (ej: "fran")
python voice_commands.py add fran "ruta/a/audio_fran.mp3" "Transcripción del audio de Fran"

# Usar la nueva voz
python voice_commands.py clone "Texto con voz de Fran" --voice fran
```

### Monitoreo
```bash
# Estado del sistema
python voice_commands.py status

# Prueba completa
python test_voices_api.py

# Prueba rápida
python test_voices_api.py --quick
```

## 🌐 URLs Importantes

- **API Base**: http://localhost:7860
- **Health Check**: http://localhost:7860/health
- **Documentación API**: http://localhost:7860/docs
- **Voice Profiles**: http://localhost:7860/voices
- **Performance Stats**: http://localhost:7860/performance-stats

## 🔥 Ejemplos Avanzados

### Clonación con cURL
```bash
# Usando el perfil "voices"
curl -X POST "http://localhost:7860/clone-voice" \
     -F "text=Hola mundo desde la API optimizada" \
     -F "voice_name=voices" \
     -F "temperature=0.7" \
     -F "remove_silence=true"

# Streaming
curl -X POST "https://i251u0bgdxqzvq-7860.proxy.runpod.net/clone-voice-stream" \
     -F "text=Prueba de streaming" \
     -F "voice_name=voices" \
     -F "streaming=true" \
     --output stream_test.wav
```

### Con Python (Async)
```python
import asyncio
import aiohttp

async def clone_voice():
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('text', 'Tu texto aquí')
        data.add_field('voice_name', 'voices')
        data.add_field('temperature', '0.7')
        
        async with session.post('http://localhost:7860/clone-voice', data=data) as resp:
            result = await resp.json()
            print(f"Audio generado: {result['audio_url']}")

asyncio.run(clone_voice())
```

## 🎯 Optimizaciones Aplicadas

### GPU
- ✅ Memoria GPU optimizada (85% del total)
- ✅ Precision mixta habilitada
- ✅ Cache de memoria GPU
- ✅ Garbage collection automático

### Procesamiento
- ✅ Chunking adaptativo basado en carga del sistema
- ✅ Cache LRU para audio de referencia (6GB)
- ✅ Paralelización de audio preprocessing
- ✅ Eliminación de silencios optimizada

### Red
- ✅ Streaming chunkeado para baja latencia
- ✅ Compresión automática de respuestas
- ✅ Máximo 3 requests concurrentes
- ✅ Timeout optimizados

## 📊 Métricas de Performance

La API reporta estas métricas en tiempo real:

- **Realtime Factor**: Tiempo de procesamiento vs duración del audio
- **Tokens/Second**: Velocidad de procesamiento de texto
- **Memory Usage**: RAM y GPU en tiempo real
- **Cache Hit Ratio**: Eficiencia del cache de audio
- **System Load**: Carga del sistema para chunking adaptativo

### Ver Métricas
```bash
# Métricas del sistema
curl http://localhost:7860/performance-stats | jq

# Recomendación de chunk size
curl "http://localhost:7860/chunk-size-recommendation?text=Tu%20texto&streaming=false"
```

## 🔧 Troubleshooting

### Problemas Comunes

**API no responde**
```bash
# Verificar que el puerto 7860 esté libre
lsof -i :7860

# Reiniciar con logs detallados
python quick_start.py
```

**Audio de referencia no encontrado**
```bash
# Verificar que el archivo existe
ls -la "Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3"

# Re-configurar voice profiles
python voice_commands.py add voices "tu_audio.mp3" "transcripción"
```

**Performance lento**
```bash
# Verificar GPU
python -c "import torch; print(torch.cuda.is_available())"

# Limpiar cache si está lleno
curl -X POST http://localhost:7860/clear-cache

# Ajustar configuración
python start_voice_api.py --cache-size 2048 --max-concurrent 1
```

## 🚀 Próximos Pasos

### Agregar Más Voces
```bash
# Ejemplo: agregar voz "fran"
python voice_commands.py add fran "audio_fran.mp3" "Transcripción exacta del audio"

# Usar la nueva voz
python voice_commands.py clone "Hola desde Fran" --voice fran
```

### Integración en Aplicaciones
- Ver `voice_cloning_client.py` para ejemplos de integración
- Usar `/docs` para explorar la API interactivamente
- Implementar retry logic para requests concurrentes

### Optimizaciones Adicionales
- Aumentar cache size si tienes más RAM: `--cache-size 8192`
- Ajustar concurrent requests según tu GPU: `--max-concurrent 4`
- Usar chunking manual para textos muy largos

---

**🎤 ¡Disfruta de la clonación de voz optimizada en el puerto 7860!**

Para soporte adicional, revisa los logs en `voice_api.log` o ejecuta las pruebas completas con `python test_voices_api.py`. 