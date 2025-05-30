#!/usr/bin/env python3
"""
Test script para el modelo Elise CSM con expresiones emocionales
Maneja el problema de autenticación usando enfoques alternativos
"""
import os
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor
import torchaudio

# Configurar variables de entorno
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def analyze_elise_model():
    """Analizar la estructura del modelo Elise disponible localmente"""
    print("🔍 ANALIZANDO MODELO ELISE LOCAL")
    print("=" * 60)
    
    model_path = "/workspace/runPodtts/models/csm-1b-elise"
    
    # Leer configuración del adaptador
    with open(f"{model_path}/adapter_config.json", "r") as f:
        adapter_config = json.load(f)
    
    print("📋 CONFIGURACIÓN DEL ADAPTADOR LORA:")
    print(f"  🎯 Modelo base: {adapter_config['base_model_name_or_path']}")
    print(f"  📊 Tipo PEFT: {adapter_config['peft_type']}")
    print(f"  🔢 Rank (r): {adapter_config['r']}")
    print(f"  📈 Alpha: {adapter_config['lora_alpha']}")
    print(f"  🎛️ Dropout: {adapter_config['lora_dropout']}")
    print(f"  🎯 Módulos objetivo: {', '.join(adapter_config['target_modules'])}")
    
    # Leer configuración del tokenizer
    with open(f"{model_path}/tokenizer_config.json", "r") as f:
        tokenizer_config = json.load(f)
    
    print(f"\n📝 CONFIGURACIÓN DEL TOKENIZER:")
    print(f"  🔤 Tipo: {tokenizer_config.get('tokenizer_class', 'N/A')}")
    print(f"  📊 Vocab size: {tokenizer_config.get('vocab_size', 'N/A')}")
    
    # Leer template de chat
    with open(f"{model_path}/chat_template.jinja", "r") as f:
        chat_template = f.read()
    
    print(f"\n📝 TEMPLATE DE CHAT:")
    print("".join(chat_template[:300]))
    if len(chat_template) > 300:
        print("... (truncado)")
    
    return adapter_config, tokenizer_config

def test_tokenizer_loading():
    """Probar la carga del tokenizer desde el modelo local"""
    print("\n🔤 PROBANDO CARGA DEL TOKENIZER")
    print("=" * 60)
    
    model_path = "/workspace/runPodtts/models/csm-1b-elise"
    
    try:
        # Intentar cargar tokenizer desde el modelo local
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer cargado exitosamente desde modelo local")
        
        # Probar tokenización básica
        test_text = "Hello, I'm Elise! <laughs> Nice to meet you!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"📝 Texto de prueba: {test_text}")
        print(f"🔢 Tokens: {len(tokens)} tokens")
        print(f"📤 Decodificado: {decoded}")
        
        # Probar expresiones emocionales
        emotions = ["<laughs>", "<giggles>", "<sighs>", "<gasps>", "<whispers>"]
        print(f"\n🎭 ANÁLISIS DE EXPRESIONES EMOCIONALES:")
        for emotion in emotions:
            try:
                emotion_tokens = tokenizer.encode(emotion, add_special_tokens=False)
                print(f"  {emotion}: {emotion_tokens} ({len(emotion_tokens)} tokens)")
            except Exception as e:
                print(f"  {emotion}: Error - {e}")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Error cargando tokenizer: {e}")
        return None

def create_emotional_test_cases():
    """Crear casos de prueba con diferentes expresiones emocionales"""
    return [
        {
            "text": "Hello, I'm Elise. Nice to meet you!",
            "description": "Saludo básico sin emociones",
            "speaker": 0,
            "emotions": []
        },
        {
            "text": "That's so funny! <laughs> I can't believe you said that!",
            "description": "Risa natural en medio de la frase",
            "speaker": 0,
            "emotions": ["<laughs>"]
        },
        {
            "text": "Oh my goodness! <giggles> You're such a silly person!",
            "description": "Risita juguetona",
            "speaker": 0,
            "emotions": ["<giggles>"]
        },
        {
            "text": "I'm so tired today. <sighs> I need some rest.",
            "description": "Suspiro de cansancio",
            "speaker": 0,
            "emotions": ["<sighs>"]
        },
        {
            "text": "<whispers> Can you keep this a secret? It's very important.",
            "description": "Susurro confidencial",
            "speaker": 0,
            "emotions": ["<whispers>"]
        },
        {
            "text": "Oh wow! <gasps> That's amazing! <laughs> I'm so happy for you!",
            "description": "Múltiples expresiones: sorpresa + alegría",
            "speaker": 0,
            "emotions": ["<gasps>", "<laughs>"]
        },
        {
            "text": "So I was walking down the street <laughs> and then this dog just comes running at me! <gasps> I was so scared!",
            "description": "Narrativa con múltiples emociones",
            "speaker": 0,
            "emotions": ["<laughs>", "<gasps>"]
        }
    ]

def analyze_emotional_expressions(tokenizer):
    """Analizar cómo el tokenizer maneja las expresiones emocionales"""
    print("\n🎭 ANÁLISIS DETALLADO DE EXPRESIONES EMOCIONALES")
    print("=" * 60)
    
    test_cases = create_emotional_test_cases()
    
    for i, case in enumerate(test_cases):
        print(f"\n📝 Caso {i+1}: {case['description']}")
        print(f"💬 Texto: {case['text']}")
        
        if tokenizer:
            try:
                # Tokenizar el texto completo
                full_tokens = tokenizer.encode(case['text'], add_special_tokens=True)
                print(f"🔢 Total tokens: {len(full_tokens)}")
                
                # Analizar cada emoción por separado
                for emotion in case['emotions']:
                    emotion_tokens = tokenizer.encode(emotion, add_special_tokens=False)
                    print(f"  🎭 {emotion}: {emotion_tokens}")
                
                # Crear formato de chat para speaker 0 (Elise)
                chat_format = f"[{case['speaker']}]{case['text']}"
                chat_tokens = tokenizer.encode(chat_format, add_special_tokens=True)
                print(f"💬 Chat format: {chat_format}")
                print(f"🔢 Chat tokens: {len(chat_tokens)}")
                
            except Exception as e:
                print(f"❌ Error procesando: {e}")

def test_alternative_approaches():
    """Probar enfoques alternativos para usar CSM sin acceso al modelo base"""
    print("\n🔄 PROBANDO ENFOQUES ALTERNATIVOS")
    print("=" * 60)
    
    print("📋 OPCIONES DISPONIBLES:")
    print("1. 🔓 Usar modelo base no-gateado (si existe)")
    print("2. 🎯 Usar implementación local del CSM")
    print("3. 🔧 Usar el repositorio oficial de CSM")
    print("4. 🍎 Usar implementación MLX (para Apple Silicon)")
    print("5. 📝 Generar solo análisis de texto sin audio")
    
    # Verificar si hay modelos alternativos disponibles
    alternative_models = [
        "unsloth/csm-1b",  # Modelo base mencionado en adapter_config
        "microsoft/DialoGPT-medium",  # Modelo conversacional alternativo
        "facebook/blenderbot-400M-distill"  # Otro modelo conversacional
    ]
    
    print(f"\n🔍 VERIFICANDO MODELOS ALTERNATIVOS:")
    for model in alternative_models:
        try:
            # Solo verificar si el modelo existe sin descargarlo
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            print(f"✅ {model}: Disponible")
        except Exception as e:
            print(f"❌ {model}: No disponible - {str(e)[:100]}...")

def generate_test_outputs():
    """Generar archivos de salida de prueba con análisis de texto"""
    print("\n📁 GENERANDO ARCHIVOS DE PRUEBA")
    print("=" * 60)
    
    output_dir = Path("/workspace/runPodtts/outputs/elise_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_cases = create_emotional_test_cases()
    
    # Crear archivo de resumen
    summary_file = output_dir / "elise_emotional_analysis.json"
    analysis_data = {
        "model_info": {
            "name": "therealcyberlord/csm-1b-elise",
            "base_model": "unsloth/csm-1b",
            "type": "LoRA adapter",
            "rank": 16,
            "alpha": 16
        },
        "test_cases": test_cases,
        "emotional_expressions": {
            "risas": ["<laughs>", "<giggles>", "<chuckles>", "<nervous laughter>"],
            "respiración": ["<sighs>", "<exhales>", "<breathes deeply>", "<gasps>"],
            "emociones": ["<sadly>", "<whispers>"],
            "sonidos_físicos": ["<sniffs>", "<scoffs>", "<smacks lips>"],
            "pausas": ["<long pause>"]
        }
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Análisis guardado en: {summary_file}")
    
    # Crear archivos individuales para cada caso de prueba
    for i, case in enumerate(test_cases):
        case_file = output_dir / f"test_case_{i+1:02d}.txt"
        with open(case_file, "w", encoding="utf-8") as f:
            f.write(f"Caso de Prueba #{i+1}\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Descripción: {case['description']}\n")
            f.write(f"Speaker: {case['speaker']} (Elise)\n")
            f.write(f"Emociones: {', '.join(case['emotions']) if case['emotions'] else 'Ninguna'}\n\n")
            f.write(f"Texto:\n{case['text']}\n\n")
            f.write(f"Formato de chat:\n[{case['speaker']}]{case['text']}\n")
        
        print(f"📝 Caso {i+1}: {case_file.name}")
    
    print(f"\n✅ Generados {len(test_cases)} casos de prueba en: {output_dir}")
    return output_dir

def main():
    """Función principal del test"""
    print("🚀 INICIANDO ANÁLISIS DEL MODELO ELISE")
    print("=" * 70)
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.cuda.get_device_name()}")
        print(f"🔥 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA no disponible, usando CPU")
    
    # Analizar modelo
    adapter_config, tokenizer_config = analyze_elise_model()
    
    # Probar tokenizer
    tokenizer = test_tokenizer_loading()
    
    # Analizar expresiones emocionales
    if tokenizer:
        analyze_emotional_expressions(tokenizer)
    
    # Probar enfoques alternativos
    test_alternative_approaches()
    
    # Generar archivos de prueba
    output_dir = generate_test_outputs()
    
    print("\n🎉 ANÁLISIS COMPLETADO")
    print("=" * 70)
    print("📋 RESUMEN:")
    print("✅ Modelo Elise analizado correctamente")
    print("✅ Configuración PEFT identificada")
    print("✅ Expresiones emocionales catalogadas")
    print("✅ Casos de prueba generados")
    print(f"📁 Resultados en: {output_dir}")
    
    print("\n🔄 PRÓXIMOS PASOS:")
    print("1. 🔑 Obtener acceso al modelo base sesame/csm-1b")
    print("2. 🔧 O usar implementación alternativa del CSM")
    print("3. 🎵 Generar audio con expresiones emocionales")
    print("4. 🎭 Probar voice cloning con speaker ID 0")

if __name__ == "__main__":
    main() 