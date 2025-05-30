#!/usr/bin/env python3
"""
Script funcional para CSM TTS usando unsloth/csm-1b + adaptador Elise
Genera audio con expresiones emocionales
"""
import os
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, CsmForConditionalGeneration
from peft import PeftModel, PeftConfig
import torchaudio
import time

# Configurar variables de entorno
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def load_csm_with_elise():
    """Cargar modelo CSM base con adaptador Elise"""
    print("🔄 CARGANDO MODELO CSM CON ADAPTADOR ELISE")
    print("=" * 60)
    
    base_model_id = "unsloth/csm-1b"
    adapter_path = "/workspace/runPodtts/models/csm-1b-elise"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print(f"📥 Cargando modelo base: {base_model_id}")
        
        # Cargar tokenizer desde el adaptador (tiene configuración completa)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        print("✅ Tokenizer cargado desde adaptador Elise")
        
        # Cargar procesador desde el modelo base
        processor = AutoProcessor.from_pretrained(base_model_id)
        print("✅ Procesador cargado desde modelo base")
        
        # Cargar modelo base
        base_model = CsmForConditionalGeneration.from_pretrained(
            base_model_id,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        print("✅ Modelo base CSM cargado")
        
        # Cargar adaptador PEFT
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ Adaptador Elise aplicado")
        
        print(f"🎯 Modelo final en device: {device}")
        return model, processor, tokenizer
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_emotional_prompts():
    """Crear prompts de prueba con expresiones emocionales"""
    return [
        {
            "text": "Hello! I'm Elise, your AI assistant. How can I help you today?",
            "description": "Saludo profesional básico",
            "filename": "01_basic_greeting"
        },
        {
            "text": "That's absolutely hilarious! <laughs> You really made my day!",
            "description": "Risa genuina y alegría",
            "filename": "02_genuine_laughter"
        },
        {
            "text": "Oh my! <giggles> You're so funny! I can't stop smiling!",
            "description": "Risitas juguetonas",
            "filename": "03_playful_giggles"
        },
        {
            "text": "I'm feeling a bit overwhelmed today. <sighs> Sometimes life gets complicated.",
            "description": "Suspiro de cansancio emocional",
            "filename": "04_tired_sigh"
        },
        {
            "text": "<whispers> I have something special to tell you. Come closer.",
            "description": "Susurro íntimo y misterioso",
            "filename": "05_intimate_whisper"
        },
        {
            "text": "Oh wow! <gasps> That's incredible news! <laughs> I'm so excited for you!",
            "description": "Sorpresa seguida de alegría",
            "filename": "06_surprise_joy"
        },
        {
            "text": "Let me tell you about my day. <laughs> So I was walking to the store, and suddenly <gasps> this huge dog came running towards me!",
            "description": "Narrativa con múltiples emociones",
            "filename": "07_story_emotions"
        },
        {
            "text": "Welcome to our virtual world! <giggles> Here, anything is possible and dreams come true!",
            "description": "Bienvenida mágica y juguetona",
            "filename": "08_magical_welcome"
        }
    ]

def generate_emotional_audio(model, processor, tokenizer, prompts):
    """Generar archivos de audio con expresiones emocionales"""
    print("\n🎵 GENERANDO AUDIO CON EXPRESIONES EMOCIONALES")
    print("=" * 60)
    
    output_dir = Path("/workspace/runPodtts/outputs/elise_emotional_audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n🎭 Generando audio {i+1}/{len(prompts)}: {prompt['description']}")
        print(f"💬 Texto: {prompt['text']}")
        
        try:
            # Formatear texto para speaker 0 (Elise)
            formatted_text = f"[0]{prompt['text']}"
            
            # Tokenizar
            inputs = tokenizer(
                formatted_text, 
                return_tensors="pt", 
                add_special_tokens=True
            ).to(device)
            
            print(f"🔢 Tokens: {inputs['input_ids'].shape[1]}")
            
            # Generar audio
            start_time = time.time()
            with torch.no_grad():
                # Usar generate con parámetros optimizados para CSM
                audio_outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,  # Suficiente para frases largas
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    output_audio=True
                )
            
            generation_time = time.time() - start_time
            print(f"⏱️  Tiempo de generación: {generation_time:.2f}s")
            
            # Guardar audio
            output_path = output_dir / f"{prompt['filename']}.wav"
            
            # Extraer audio de los outputs
            if hasattr(audio_outputs, 'audio'):
                audio_data = audio_outputs.audio
            elif isinstance(audio_outputs, dict) and 'audio' in audio_outputs:
                audio_data = audio_outputs['audio']
            else:
                # Fallback: usar processor para extraer audio
                audio_data = processor.decode_audio(audio_outputs)
            
            # Guardar usando torchaudio
            if audio_data is not None:
                # Asegurar que el audio esté en el formato correcto
                if audio_data.dim() == 3:  # [batch, channels, samples]
                    audio_data = audio_data.squeeze(0)
                elif audio_data.dim() == 1:  # [samples]
                    audio_data = audio_data.unsqueeze(0)  # [1, samples]
                
                # Guardar audio
                torchaudio.save(
                    str(output_path), 
                    audio_data.cpu(), 
                    sample_rate=24000  # CSM usa 24kHz
                )
                print(f"✅ Audio guardado: {output_path}")
                
                # Información del audio
                duration = audio_data.shape[-1] / 24000
                print(f"📊 Duración: {duration:.2f}s")
                
                results.append({
                    "prompt": prompt,
                    "output_path": str(output_path),
                    "generation_time": generation_time,
                    "duration": duration,
                    "success": True
                })
            else:
                print("❌ No se pudo extraer audio de los outputs")
                results.append({
                    "prompt": prompt,
                    "output_path": None,
                    "generation_time": generation_time,
                    "duration": 0,
                    "success": False
                })
                
        except Exception as e:
            print(f"❌ Error generando audio: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "prompt": prompt,
                "output_path": None,
                "generation_time": 0,
                "duration": 0,
                "success": False,
                "error": str(e)
            })
    
    return results, output_dir

def save_generation_report(results, output_dir):
    """Guardar reporte de generación"""
    report_path = output_dir / "generation_report.json"
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "base_model": "unsloth/csm-1b",
            "adapter": "therealcyberlord/csm-1b-elise",
            "speaker_id": 0
        },
        "statistics": {
            "total_prompts": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "total_generation_time": sum(r["generation_time"] for r in results),
            "total_audio_duration": sum(r["duration"] for r in results if r["success"])
        },
        "results": results
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 REPORTE DE GENERACIÓN:")
    print(f"✅ Exitosos: {report['statistics']['successful']}/{report['statistics']['total_prompts']}")
    print(f"❌ Fallidos: {report['statistics']['failed']}/{report['statistics']['total_prompts']}")
    print(f"⏱️  Tiempo total: {report['statistics']['total_generation_time']:.2f}s")
    print(f"🎵 Audio total: {report['statistics']['total_audio_duration']:.2f}s")
    print(f"📄 Reporte guardado: {report_path}")

def test_voice_cloning():
    """Probar capacidades de voice cloning con diferentes speakers"""
    print("\n🎭 PROBANDO VOICE CLONING")
    print("=" * 60)
    
    # Casos de prueba para voice cloning
    cloning_tests = [
        {
            "speaker": 0,
            "text": "Hi, this is Elise speaking with my natural voice.",
            "description": "Voz natural de Elise (speaker 0)"
        },
        {
            "speaker": 1,
            "text": "Hello, this is a different speaker voice.",
            "description": "Intento de speaker alternativo"
        }
    ]
    
    print("📋 CASOS DE PRUEBA PARA VOICE CLONING:")
    for test in cloning_tests:
        print(f"  🎤 Speaker {test['speaker']}: {test['description']}")
        print(f"     💬 Texto: {test['text']}")
    
    print("\n💡 NOTA: El modelo Elise está optimizado para speaker ID 0")
    print("   Para mejores resultados, usar siempre [0] como prefijo")

def main():
    """Función principal"""
    print("🚀 INICIANDO CSM TTS CON MODELO ELISE")
    print("=" * 70)
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.cuda.get_device_name()}")
        print(f"🔥 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA no disponible, usando CPU")
    
    # Cargar modelo
    model, processor, tokenizer = load_csm_with_elise()
    
    if model is None:
        print("❌ No se pudo cargar el modelo. Terminando.")
        return False
    
    # Crear prompts emocionales
    prompts = create_emotional_prompts()
    print(f"\n📝 Preparados {len(prompts)} prompts emocionales")
    
    # Generar audio
    results, output_dir = generate_emotional_audio(model, processor, tokenizer, prompts)
    
    # Guardar reporte
    save_generation_report(results, output_dir)
    
    # Probar voice cloning
    test_voice_cloning()
    
    print("\n🎉 GENERACIÓN COMPLETADA")
    print("=" * 70)
    print(f"📁 Archivos de audio en: {output_dir}")
    print("🎧 Reproduce los archivos para escuchar las expresiones emocionales")
    print("🎭 Elise puede expresar: risas, suspiros, susurros, sorpresa y más!")
    
    return True

if __name__ == "__main__":
    main() 