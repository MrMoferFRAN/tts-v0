#!/usr/bin/env python3
"""
Script para probar el modelo CSM-1B finetuneado de Elise con expresiones emocionales
"""
import os
import torch
import json
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoProcessor
import torchaudio
from pathlib import Path

# Configurar variables de entorno
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def analyze_elise_model():
    """Analizar la composición y estructura del modelo Elise"""
    print("🔍 ANALIZANDO MODELO ELISE")
    print("=" * 50)
    
    # Cargar configuración del adaptador
    with open("/workspace/runPodtts/models/csm-1b-elise/adapter_config.json", "r") as f:
        adapter_config = json.load(f)
    
    print("📋 CONFIGURACIÓN DEL ADAPTADOR LORA:")
    print(f"  🎯 Modelo base: {adapter_config['base_model_name_or_path']}")
    print(f"  📊 Tipo PEFT: {adapter_config['peft_type']}")
    print(f"  🔢 Rank (r): {adapter_config['r']}")
    print(f"  📈 Alpha: {adapter_config['lora_alpha']}")
    print(f"  🎛️ Dropout: {adapter_config['lora_dropout']}")
    print(f"  🎯 Módulos objetivo: {', '.join(adapter_config['target_modules'])}")
    
    # Leer template de chat
    with open("/workspace/runPodtts/models/csm-1b-elise/chat_template.jinja", "r") as f:
        chat_template = f.read()
    
    print(f"\n📝 TEMPLATE DE CHAT:")
    print("".join(chat_template[:500]))
    if len(chat_template) > 500:
        print("... (truncado)")
    
    return adapter_config

def get_emotional_expressions():
    """Definir las expresiones emocionales soportadas por Elise"""
    return {
        "risas": ["<laughs>", "<giggles>", "<chuckles>", "<nervous laughter>", "<laughs nervously>"],
        "respiración": ["<sighs>", "<exhales>", "<breathes deeply>", "<gasps>"],
        "sonidos_vocales": ["<moans>", "<yawning>", "<clears throat>", "<coughs>"],
        "emociones": ["<sadly>", "<whispers>"],
        "sonidos_físicos": ["<sniffs>", "<scoffs>", "<smacks lips>", "<clicks tongue>"],
        "pausas": ["<long pause>"],
        "otros": ["<trails off>", "<stutters>"]
    }

def create_emotional_test_cases():
    """Crear casos de prueba con diferentes expresiones emocionales"""
    emotions = get_emotional_expressions()
    
    test_cases = [
        # Casos básicos sin emociones
        {
            "text": "Hello, I'm Elise. Nice to meet you!",
            "description": "Saludo básico sin emociones",
            "speaker": 0
        },
        
        # Casos con risas
        {
            "text": "That's so funny! <laughs> I can't believe you said that!",
            "description": "Risa natural en medio de la frase",
            "speaker": 0
        },
        {
            "text": "Oh my goodness! <giggles> You're such a silly person!",
            "description": "Risita juguetona",
            "speaker": 0
        },
        
        # Casos con suspiros
        {
            "text": "I'm so tired today. <sighs> I need some rest.",
            "description": "Suspiro de cansancio",
            "speaker": 0
        },
        {
            "text": "<sighs> Well, I guess we have to do this again.",
            "description": "Suspiro al inicio (resignación)",
            "speaker": 0
        },
        
        # Casos con múltiples emociones
        {
            "text": "Oh wow! <gasps> That's amazing! <laughs> I'm so happy for you!",
            "description": "Múltiples expresiones: sorpresa + alegría",
            "speaker": 0
        },
        
        # Casos con susurros
        {
            "text": "<whispers> Can you keep this a secret? It's very important.",
            "description": "Susurro confidencial",
            "speaker": 0
        },
        
        # Casos con tristeza
        {
            "text": "<sadly> I miss you so much. When will I see you again?",
            "description": "Expresión triste",
            "speaker": 0
        },
        
        # Casos complejos conversacionales
        {
            "text": "So I was walking down the street <laughs> and then this dog just comes running at me! <gasps> I was so scared!",
            "description": "Narrativa con múltiples emociones",
            "speaker": 0
        },
        
        # Casos con pausas
        {
            "text": "I was thinking about what you said <long pause> and I think you're right.",
            "description": "Pausa reflexiva",
            "speaker": 0
        }
    ]
    
    return test_cases

def load_base_model_for_analysis():
    """Intentar cargar modelo base para análisis (sin fine-tuning)"""
    try:
        # Usar transformers directamente para análisis
        from transformers import AutoTokenizer
        
        print("📥 Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("sesame/csm-1b")
        
        # Analizar tokens especiales
        special_tokens = tokenizer.special_tokens_map
        print("🔤 TOKENS ESPECIALES:")
        for key, value in special_tokens.items():
            print(f"  {key}: {value}")
        
        # Analizar algunos tokens emocionales
        emotions = ["<laughs>", "<giggles>", "<sighs>", "<gasps>"]
        print("\n🎭 ANÁLISIS DE TOKENS EMOCIONALES:")
        for emotion in emotions:
            try:
                token_ids = tokenizer.encode(emotion, add_special_tokens=False)
                print(f"  {emotion}: {token_ids}")
            except Exception as e:
                print(f"  {emotion}: Error - {e}")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ Error cargando modelo base: {e}")
        return None

def generate_test_audio_samples(tokenizer=None):
    """Generar archivos de audio de prueba con diferentes expresiones"""
    print("\n🎵 GENERANDO MUESTRAS DE AUDIO DE PRUEBA")
    print("=" * 50)
    
    # Crear directorio de salida
    output_dir = Path("/workspace/runPodtts/outputs/elise_emotional_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_cases = create_emotional_test_cases()
    
    # Por ahora, crear archivos de texto con las muestras
    # En una implementación completa, aquí generaríamos audio real
    for i, test_case in enumerate(test_cases):
        # Guardar el texto del caso de prueba
        text_file = output_dir / f"test_{i:02d}_{test_case['description'].replace(' ', '_').replace(',', '')}.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(f"Descripción: {test_case['description']}\n")
            f.write(f"Speaker: {test_case['speaker']}\n")
            f.write(f"Texto: {test_case['text']}\n")
        
        print(f"📝 Test {i+1:2d}: {test_case['description']}")
        print(f"    📄 Archivo: {text_file.name}")
        print(f"    💬 Texto: {test_case['text']}")
        
        # Analizar tokens si tenemos tokenizer
        if tokenizer:
            try:
                tokens = tokenizer.encode(test_case['text'], add_special_tokens=True)
                print(f"    🔤 Tokens: {len(tokens)} tokens")
            except:
                pass
        print()
    
    print(f"✅ Generados {len(test_cases)} casos de prueba en: {output_dir}")
    return test_cases

def analyze_emotional_expressions():
    """Analizar las expresiones emocionales disponibles"""
    print("\n🎭 ANÁLISIS DE EXPRESIONES EMOCIONALES")
    print("=" * 50)
    
    emotions = get_emotional_expressions()
    
    total_expressions = sum(len(exprs) for exprs in emotions.values())
    print(f"📊 Total de expresiones disponibles: {total_expressions}")
    print()
    
    for category, expressions in emotions.items():
        print(f"🎪 {category.upper()} ({len(expressions)} expresiones):")
        for expr in expressions:
            print(f"    • {expr}")
        print()
    
    # Estadísticas del dataset original (basado en la información web)
    print("📈 ESTADÍSTICAS DEL DATASET ORIGINAL:")
    original_stats = {
        "giggles": 76,
        "laughs": 336, 
        "long pause": 2,
        "chuckles": 20,
        "whispers": 2,
        "sighs": 156,
        "gasps": 4,
        "moans": 8,
        "nervously": 6,
        "sadly": 2,
        "sniffs": 6,
        "scoffs": 8
    }
    
    for emotion, count in original_stats.items():
        print(f"    • {emotion}: {count} instancias")

def create_voice_cloning_guide():
    """Crear guía para usar Elise como voice cloning"""
    print("\n🎤 GUÍA DE VOICE CLONING CON ELISE")
    print("=" * 50)
    
    guide = """
    PASOS PARA USAR ELISE COMO VOICE CLONING:
    
    1. 📚 PREPARAR CONTEXTO EMOCIONAL:
       - Usar ejemplos de audio con las emociones deseadas
       - Incluir expresiones como <laughs>, <sighs>, etc.
       - Mantener consistencia en el estilo emocional
    
    2. 🎯 DEFINIR SPEAKER ID:
       - Usar speaker=0 para la voz de Elise
       - Mantener consistencia en el speaker ID
    
    3. 💬 ESTRUCTURAR EL TEXTO:
       - Incluir expresiones emocionales en formato <expresión>
       - Ejemplo: "I'm so happy! <laughs> This is amazing!"
       - Usar pausas: <long pause> para efectos dramáticos
    
    4. 🔄 USAR CONTEXTO CONVERSACIONAL:
       - Proveer segmentos previos de la conversación
       - Mantener el flujo emocional consistente
       - Usar la misma voz pero con diferentes estados emocionales
    
    5. ⚙️ PARÁMETROS RECOMENDADOS:
       - Temperature: 0.7-0.8 para naturalidad
       - Max audio length: 10000ms para frases normales
       - Usar contexto de 2-3 segmentos previos
    """
    
    print(guide)
    
    # Guardar guía en archivo
    guide_file = Path("/workspace/runPodtts/outputs/elise_voice_cloning_guide.txt")
    with open(guide_file, "w", encoding="utf-8") as f:
        f.write(guide)
    
    print(f"📖 Guía guardada en: {guide_file}")

def main():
    """Función principal"""
    print("🚀 ANÁLISIS COMPLETO DEL MODELO ELISE CSM-1B")
    print("=" * 60)
    
    # Análizar configuración del modelo
    adapter_config = analyze_elise_model()
    
    # Cargar tokenizer para análisis
    tokenizer = load_base_model_for_analysis()
    
    # Analizar expresiones emocionales
    analyze_emotional_expressions()
    
    # Generar casos de prueba
    test_cases = generate_test_audio_samples(tokenizer)
    
    # Crear guía de voice cloning
    create_voice_cloning_guide()
    
    print("\n🎉 ANÁLISIS COMPLETADO")
    print("=" * 60)
    print("📁 Archivos generados en: /workspace/runPodtts/outputs/")
    print("📋 Para usar el modelo:")
    print("   1. Revisa los casos de prueba generados")
    print("   2. Consulta la guía de voice cloning")
    print("   3. Usa las expresiones emocionales listadas")
    print("   4. Mantén consistencia en speaker ID y estilo")
    
    return {
        "adapter_config": adapter_config,
        "test_cases": test_cases,
        "emotional_expressions": get_emotional_expressions()
    }

if __name__ == "__main__":
    main() 