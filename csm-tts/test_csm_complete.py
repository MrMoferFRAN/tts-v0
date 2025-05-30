#!/usr/bin/env python3
"""
Script completo para CSM TTS usando sesame/csm-1b + adaptador Elise
Genera audio con expresiones emocionales usando el modelo base descargado
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

def load_csm_with_elise_local():
    """Cargar modelo CSM local con adaptador Elise"""
    print("🔄 CARGANDO MODELO CSM COMPLETO CON ADAPTADOR ELISE")
    print("=" * 70)
    
    base_model_path = "/workspace/runPodtts/models/sesame-csm-1b"
    adapter_path = "/workspace/runPodtts/models/csm-1b-elise"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print(f"📥 Cargando modelo base desde: {base_model_path}")
        
        # Cargar tokenizer y procesador desde el modelo base local
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        print("✅ Tokenizer cargado desde modelo base local")
        
        processor = AutoProcessor.from_pretrained(base_model_path)
        print("✅ Procesador cargado desde modelo base local")
        
        # Cargar modelo base con configuración optimizada para A100
        print("🔥 Cargando modelo CSM en GPU...")
        base_model = CsmForConditionalGeneration.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_safetensors=True
        )
        print("✅ Modelo base CSM cargado")
        
        # Cargar adaptador PEFT Elise
        print("🎭 Aplicando adaptador Elise...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ Adaptador Elise aplicado exitosamente")
        
        # Información del modelo
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 INFORMACIÓN DEL MODELO:")
        print(f"🔢 Parámetros totales: {total_params:,}")
        print(f"🎯 Parámetros entrenables: {trainable_params:,}")
        print(f"📈 % Entrenables: {100 * trainable_params / total_params:.2f}%")
        print(f"🎯 Device final: {next(model.parameters()).device}")
        
        return model, processor, tokenizer
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_comprehensive_emotional_prompts():
    """Crear un conjunto completo de prompts emocionales para prueba"""
    return [
        # Saludos y presentaciones
        {
            "text": "Hello! I'm Elise, your emotional AI companion. How are you feeling today?",
            "description": "Saludo cálido y empático",
            "filename": "01_warm_greeting",
            "category": "greeting"
        },
        
        # Expresiones de alegría
        {
            "text": "That's absolutely wonderful! <laughs> I'm so happy to hear that news!",
            "description": "Alegría genuina con risa",
            "filename": "02_joyful_laughter",
            "category": "joy"
        },
        {
            "text": "Oh my goodness! <giggles> You always know how to make me smile!",
            "description": "Risita divertida y cariñosa",
            "filename": "03_affectionate_giggles",
            "category": "joy"
        },
        
        # Expresiones de tristeza y cansancio
        {
            "text": "I understand how you feel. <sighs> Sometimes life can be really challenging.",
            "description": "Empatía con suspiro comprensivo",
            "filename": "04_empathetic_sigh",
            "category": "empathy"
        },
        {
            "text": "<sadly> I'm sorry to hear you're going through a difficult time. I'm here for you.",
            "description": "Expresión triste pero solidaria",
            "filename": "05_compassionate_sadness",
            "category": "comfort"
        },
        
        # Expresiones de sorpresa
        {
            "text": "Wait, what?! <gasps> That's incredible! Tell me more!",
            "description": "Sorpresa y entusiasmo",
            "filename": "06_surprised_gasp",
            "category": "surprise"
        },
        {
            "text": "Oh wow! <gasps> I never expected that! <laughs> This is amazing!",
            "description": "Sorpresa que se convierte en alegría",
            "filename": "07_surprise_to_joy",
            "category": "surprise"
        },
        
        # Susurros y confidencias
        {
            "text": "<whispers> Can I tell you a secret? I think you're absolutely wonderful.",
            "description": "Susurro íntimo y cariñoso",
            "filename": "08_intimate_whisper",
            "category": "intimate"
        },
        {
            "text": "<whispers> Come closer, I have something important to share with you.",
            "description": "Susurro misterioso y atractivo",
            "filename": "09_mysterious_whisper",
            "category": "intimate"
        },
        
        # Narrativas con múltiples emociones
        {
            "text": "Let me tell you about my day! <laughs> So I was learning something new, and suddenly <gasps> everything just clicked! <giggles> It was such an amazing moment!",
            "description": "Historia con transiciones emocionales",
            "filename": "10_emotional_story",
            "category": "narrative"
        },
        
        # Expresiones de apoyo
        {
            "text": "You know what? <sighs> Even when things get tough, you always find a way to keep going. That's really inspiring.",
            "description": "Apoyo reflexivo y motivador",
            "filename": "11_supportive_reflection",
            "category": "support"
        },
        
        # Expresiones juguetonas
        {
            "text": "Welcome to our magical conversation! <giggles> Here, we can explore any topic and have the most wonderful discussions!",
            "description": "Bienvenida juguetona y mágica",
            "filename": "12_playful_welcome",
            "category": "playful"
        }
    ]

def generate_emotional_audio_complete(model, processor, tokenizer, prompts):
    """Generar archivos de audio completos con expresiones emocionales"""
    print("\n🎵 GENERANDO AUDIO COMPLETO CON EXPRESIONES EMOCIONALES")
    print("=" * 70)
    
    output_dir = Path("/workspace/runPodtts/outputs/elise_complete_audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    results = []
    
    print(f"🎯 Generando en device: {device}")
    print(f"🔥 Modelo dtype: {next(model.parameters()).dtype}")
    
    for i, prompt in enumerate(prompts):
        print(f"\n🎭 [{i+1:2d}/{len(prompts)}] {prompt['description']}")
        print(f"📁 Categoría: {prompt['category']}")
        print(f"💬 Texto: {prompt['text']}")
        
        try:
            # Formatear texto para speaker 0 (Elise)
            formatted_text = f"[0]{prompt['text']}"
            
            # Tokenizar con configuración específica para CSM
            inputs = tokenizer(
                formatted_text, 
                return_tensors="pt", 
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            print(f"🔢 Tokens de entrada: {inputs['input_ids'].shape[1]}")
            
            # Generar audio con parámetros optimizados
            start_time = time.time()
            
            with torch.no_grad():
                # Configuración optimizada para CSM + Elise
                generation_config = {
                    "max_new_tokens": 750,  # Más tokens para frases complejas
                    "do_sample": True,
                    "temperature": 0.8,     # Ligeramente más creativo para emociones
                    "top_p": 0.9,
                    "top_k": 50,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    "use_cache": True
                }
                
                # Generar tokens y audio
                outputs = model.generate(
                    **inputs,
                    **generation_config
                )
            
            generation_time = time.time() - start_time
            print(f"⏱️  Tiempo de generación: {generation_time:.2f}s")
            print(f"🔢 Tokens generados: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
            
            # Procesar outputs para extraer audio
            try:
                # Intentar extraer audio usando el procesador
                audio_data = processor.decode_audio(outputs)
                
                if audio_data is not None:
                    output_path = output_dir / f"{prompt['filename']}.wav"
                    
                    # Procesar formato del audio
                    if isinstance(audio_data, torch.Tensor):
                        if audio_data.dim() == 3:  # [batch, channels, samples]
                            audio_data = audio_data.squeeze(0)
                        elif audio_data.dim() == 1:  # [samples]
                            audio_data = audio_data.unsqueeze(0)  # [1, samples]
                    
                    # Guardar audio
                    sample_rate = 24000  # CSM usa 24kHz
                    torchaudio.save(
                        str(output_path), 
                        audio_data.cpu().float(), 
                        sample_rate=sample_rate
                    )
                    
                    duration = audio_data.shape[-1] / sample_rate
                    print(f"✅ Audio guardado: {output_path}")
                    print(f"📊 Duración: {duration:.2f}s")
                    print(f"🎵 Sample rate: {sample_rate}Hz")
                    
                    results.append({
                        "prompt": prompt,
                        "output_path": str(output_path),
                        "generation_time": generation_time,
                        "duration": duration,
                        "sample_rate": sample_rate,
                        "success": True
                    })
                else:
                    print("❌ No se pudo extraer audio del modelo")
                    results.append({
                        "prompt": prompt,
                        "output_path": None,
                        "generation_time": generation_time,
                        "duration": 0,
                        "success": False,
                        "error": "Audio extraction failed"
                    })
                    
            except Exception as audio_error:
                print(f"❌ Error procesando audio: {audio_error}")
                results.append({
                    "prompt": prompt,
                    "output_path": None,
                    "generation_time": generation_time,
                    "duration": 0,
                    "success": False,
                    "error": str(audio_error)
                })
                
        except Exception as e:
            print(f"❌ Error en generación: {e}")
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

def save_comprehensive_report(results, output_dir):
    """Guardar reporte completo de generación"""
    report_path = output_dir / "comprehensive_generation_report.json"
    
    # Calcular estadísticas por categoría
    categories = {}
    for result in results:
        cat = result["prompt"]["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "successful": 0, "failed": 0, "total_duration": 0}
        
        categories[cat]["total"] += 1
        if result["success"]:
            categories[cat]["successful"] += 1
            categories[cat]["total_duration"] += result["duration"]
        else:
            categories[cat]["failed"] += 1
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "base_model": "sesame/csm-1b",
            "adapter": "therealcyberlord/csm-1b-elise",
            "speaker_id": 0,
            "device": str(next(iter(results))["prompt"].get("device", "cuda")),
            "dtype": "float16"
        },
        "global_statistics": {
            "total_prompts": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "success_rate": f"{100 * sum(1 for r in results if r['success']) / len(results):.1f}%",
            "total_generation_time": sum(r["generation_time"] for r in results),
            "total_audio_duration": sum(r["duration"] for r in results if r["success"]),
            "avg_generation_time": sum(r["generation_time"] for r in results) / len(results),
            "categories": list(categories.keys())
        },
        "category_statistics": categories,
        "detailed_results": results
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 REPORTE COMPLETO DE GENERACIÓN:")
    print(f"✅ Exitosos: {report['global_statistics']['successful']}/{report['global_statistics']['total_prompts']} ({report['global_statistics']['success_rate']})")
    print(f"❌ Fallidos: {report['global_statistics']['failed']}/{report['global_statistics']['total_prompts']}")
    print(f"⏱️  Tiempo total: {report['global_statistics']['total_generation_time']:.2f}s")
    print(f"🎵 Audio total: {report['global_statistics']['total_audio_duration']:.2f}s")
    print(f"📈 Tiempo promedio: {report['global_statistics']['avg_generation_time']:.2f}s por muestra")
    
    print(f"\n📋 ESTADÍSTICAS POR CATEGORÍA:")
    for cat, stats in categories.items():
        success_rate = 100 * stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  🎭 {cat}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%) - {stats['total_duration']:.1f}s")
    
    print(f"\n📄 Reporte detallado guardado: {report_path}")

def main():
    """Función principal"""
    print("🚀 INICIANDO CSM TTS COMPLETO CON MODELO ELISE")
    print("=" * 80)
    
    # Verificar CUDA y recursos
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.cuda.get_device_name()}")
        print(f"🔥 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"💾 VRAM libre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA no disponible, usando CPU")
    
    # Cargar modelo completo
    model, processor, tokenizer = load_csm_with_elise_local()
    
    if model is None:
        print("❌ No se pudo cargar el modelo. Terminando.")
        return False
    
    # Crear prompts emocionales completos
    prompts = create_comprehensive_emotional_prompts()
    print(f"\n📝 Preparados {len(prompts)} prompts emocionales en {len(set(p['category'] for p in prompts))} categorías")
    
    # Mostrar categorías
    categories = {}
    for p in prompts:
        cat = p['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("🎭 Categorías emocionales:")
    for cat, count in categories.items():
        print(f"   • {cat}: {count} prompts")
    
    # Generar audio completo
    print(f"\n🎵 Iniciando generación de audio...")
    results, output_dir = generate_emotional_audio_complete(model, processor, tokenizer, prompts)
    
    # Guardar reporte completo
    save_comprehensive_report(results, output_dir)
    
    print("\n🎉 GENERACIÓN COMPLETA FINALIZADA")
    print("=" * 80)
    print(f"📁 Archivos de audio en: {output_dir}")
    print("🎧 Reproduce los archivos para escuchar a Elise con expresiones emocionales")
    print("🎭 Elise puede expresar: alegría, tristeza, sorpresa, susurros, narrativas y más!")
    print("\n💡 PRÓXIMOS PASOS:")
    print("   • Escucha los archivos generados")
    print("   • Experimenta con nuevos prompts emocionales")
    print("   • Ajusta parámetros de generación según preferencias")
    print("   • Usa Elise para proyectos de voice cloning emocional")
    
    return True

if __name__ == "__main__":
    main() 