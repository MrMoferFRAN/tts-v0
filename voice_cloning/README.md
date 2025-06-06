# Voice Cloning con Sesame CSM-1B

Este módulo proporciona una implementación estructurada para clonación de voz usando el modelo Sesame CSM-1B, basado en el repositorio [isaiahbjork/csm-voice-cloning](https://github.com/isaiahbjork/csm-voice-cloning).

## Características

- ✅ **Uso de modelo local**: Utiliza el modelo CSM-1B ya descargado
- ✅ **Interfaz estructurada**: Clases y métodos organizados para fácil uso
- ✅ **Múltiples modos de generación**: Individual, por lotes, y con parámetros personalizados
- ✅ **Watermarking**: Funcionalidades para marcar y detectar audio generado
- ✅ **Compatibilidad**: Mantiene compatibilidad con el repositorio original

## Estructura del Módulo

```
voice_cloning/
├── __init__.py          # Inicialización del módulo
├── models.py            # Carga y configuración del modelo CSM-1B
├── voice_clone.py       # Clase principal VoiceCloner
├── generator.py         # Generador compatible con repo original
├── watermarking.py      # Utilidades de watermarking
├── example_usage.py     # Ejemplos de uso
├── requirements.txt     # Dependencias
└── README.md           # Este archivo
```

## Instalación

### 1. Instalar dependencias

```bash
pip install -r voice_cloning/requirements.txt
```

### 2. Verificar el modelo

Asegúrate de que el modelo CSM-1B esté en `./models/sesame-csm-1b/`

## Uso Básico

### Ejemplo Simple

```python
from voice_cloning import VoiceCloner

# Inicializar el clonador de voz
cloner = VoiceCloner(model_path="./models/sesame-csm-1b")

# Clonar voz desde archivo de referencia
output_path = cloner.clone_voice_from_file(
    reference_audio="Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3",
    reference_transcript="Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo.",
    target_text="Hola, ¿cómo estás hoy?",
    output_path="mi_voz_clonada.wav"
)

print(f"Audio generado: {output_path}")
```

### Generación por Lotes

```python
textos = [
    "Primera frase a sintetizar",
    "Segunda frase con mi voz clonada",
    "Tercera frase de ejemplo"
]

outputs = cloner.batch_generate(
    text_list=textos,
    context_text="Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo.",
    context_audio_path="Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3",
    output_dir="outputs/batch"
)
```

### Uso Avanzado con Parámetros Personalizados

```python
output = cloner.generate_speech(
    context_text="Transcripción del audio de referencia",
    target_text="Texto a sintetizar",
    context_audio_path="audio_referencia.mp3",
    output_path="output_personalizado.wav",
    temperature=0.8,           # Creatividad del modelo
    max_new_tokens=1024        # Longitud máxima de generación
)
```

## Ejecutar el Ejemplo

### Método 1: Usar el ejemplo incluido

```bash
cd /workspacetts-v0
python voice_cloning/example_usage.py
```

### Método 2: Usar el generador compatible

```bash
cd /workspacetts-v0
python voice_cloning/generator.py
```

### Método 3: Uso directo en Python

```python
# Desde el directorio raíz del proyecto
from voice_cloning.generator import VoiceGenerator

generator = VoiceGenerator()
result = generator.generate(
    context_audio_path="Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3",
    context_text="Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo.",
    text="Tu texto aquí",
    output_filename="resultado.wav"
)
```

## Watermarking

### Aplicar Marca de Agua

```python
from voice_cloning.watermarking import apply_watermark

watermarked_file = apply_watermark(
    audio_path="audio_generado.wav",
    output_path="audio_con_marca.wav",
    watermark_text="Generado por CSM Voice Cloning",
    method="metadata"  # o "spectral" o "temporal"
)
```

### Detectar Marca de Agua

```python
from voice_cloning.watermarking import detect_watermark

resultado = detect_watermark(
    audio_path="audio_con_marca.wav",
    expected_watermark="Generado por CSM Voice Cloning"
)
print(resultado)
```

## Configuración del Modelo

### Personalizar Parámetros

```python
from voice_cloning.models import CSMModelConfig

config = CSMModelConfig(
    max_seq_len=4096,      # Longitud máxima de secuencia
    temperature=0.7        # Temperatura de generación
)

cloner = VoiceCloner(
    model_path="./models/sesame-csm-1b",
    max_seq_len=4096
)
```

### Información del Modelo

```python
from voice_cloning.models import get_model_info

info = get_model_info("./models/sesame-csm-1b")
print(info)
```

## Audio de Referencia

El módulo está configurado para usar el audio de referencia:
- **Archivo**: `Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3`
- **Transcripción**: `"Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo."`

Asegúrate de que este archivo esté en el directorio raíz del proyecto.

## Solución de Problemas

### Error: Model not found
```bash
# Verificar que el modelo existe
ls -la models/sesame-csm-1b/
```

### Error: CUDA out of memory
- Reducir `max_seq_len`
- Usar un audio de referencia más corto
- Ejecutar en CPU (más lento pero funcional)

### Error: Tensor dimension mismatch
- Ajustar `max_seq_len` en la configuración
- Verificar compatibilidad del audio de entrada

## Limitaciones Actuales

⚠️ **Nota importante**: Esta implementación actual incluye:
- Carga y uso del modelo CSM-1B
- Procesamiento de prompts y generación de texto
- Creación de audio placeholder como demostración

Para obtener audio real clonado, se necesitaría:
- Integración con un vocoder específico para CSM-1B
- Procesamiento adicional de las salidas del modelo
- Conversión de embeddings a forma de onda de audio

## Próximos Pasos

1. **Integrar vocoder real**: Conectar con el sistema de síntesis de audio de CSM-1B
2. **Mejorar calidad**: Optimizar prompts y parámetros de generación
3. **Soporte multi-idioma**: Extender para diferentes idiomas
4. **Interfaz web**: Crear una interfaz web para uso fácil

## Referencias

- [Repositorio original](https://github.com/isaiahbjork/csm-voice-cloning)
- [Modelo Sesame CSM-1B](https://huggingface.co/sesame-csm-1b)
- Basado en la arquitectura Llama 3.2

## Licencia

Este módulo está basado en el trabajo de [isaiahbjork/csm-voice-cloning](https://github.com/isaiahbjork/csm-voice-cloning) y sigue la misma licencia Apache 2.0. 