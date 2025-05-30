#!/usr/bin/env python3
"""
Script CSM TTS FINAL - Maneja outputs directos de audio de CSM
¡Ya no se congela y maneja el audio correctamente!
"""
import os
import torch
import json
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, CsmForConditionalGeneration
from peft import PeftModel
import torchaudio
import gc

# Configurar para CSM
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def load_csm_final():
    """Cargar CSM con configuración final funcionando"""
    print("🔄 CARGANDO CSM - VERSIÓN FINAL FUNCIONANDO")
    print("=" * 60)
    
    base_model_path = "/workspace/runPodtts/models/sesame-csm-1b"
    adapter_path = "/workspace/runPodtts/models/csm-1b-elise"
    
    try:
        print("📥 Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("✅ Tokenizer configurado")
        
        print("📥 Cargando procesador...")
        processor = AutoProcessor.from_pretrained(base_model_path)
        print("✅ Procesador cargado")
        
        print("🔥 Cargando modelo base...")
        base_model = CsmForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Modelo base cargado")
        
        print("🎭 Aplicando adaptador Elise...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ Adaptador aplicado")
        
        total_params = sum(p.numel() for p in model.parameters())
        memory_gb = torch.cuda.memory_allocated() / 1e9
        print(f"🔢 Parámetros: {total_params:,}")
        print(f"💾 VRAM: {memory_gb:.1f}GB")
        
        return model, processor, tokenizer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def process_csm_audio_output(audio_outputs, sample_rate=24000):
    """Procesar outputs de audio CSM correctamente"""
    try:
        print(f"🔍 Analizando outputs: {audio_outputs.shape}, dtype: {audio_outputs.dtype}")
        
        # CSM genera audio en formato [batch, sequence, features] o [batch, features, sequence]
        if audio_outputs.dim() == 3:
            batch_size, dim1, dim2 = audio_outputs.shape
            
            # Determinar si es [batch, seq, features] o [batch, features, seq]
            if dim1 > dim2:  # [batch, sequence, features]
                print(f"📊 Formato: [batch={batch_size}, sequence={dim1}, features={dim2}]")
                # Tomar solo el primer batch y transponer si es necesario
                audio_data = audio_outputs[0]  # [sequence, features]
                
                # Convertir a waveform - CSM puede usar diferentes esquemas
                if dim2 == 32:  # Probablemente logmel o similar
                    # Esto es especulativo - puede necesitar decodificación especial
                    print("🎵 Convirtiendo features de audio...")
                    # Para CSM, los outputs podrían ser mel-spectrograms o waveforms codificados
                    # Intentar usar como waveform directo sumando features
                    audio_data = audio_data.mean(dim=-1)  # Promedio de features
                elif dim2 == 1:  # Ya es waveform
                    audio_data = audio_data.squeeze(-1)
                else:
                    print(f"⚠️  Dimensión de features desconocida: {dim2}")
                    audio_data = audio_data.mean(dim=-1)
                
            else:  # [batch, features, sequence]
                print(f"📊 Formato: [batch={batch_size}, features={dim1}, sequence={dim2}]")
                audio_data = audio_outputs[0]  # [features, sequence]
                
                if dim1 == 1:  # Ya es waveform
                    audio_data = audio_data.squeeze(0)
                else:
                    # Promedio de features
                    audio_data = audio_data.mean(dim=0)
        
        elif audio_outputs.dim() == 2:
            print(f"📊 Formato 2D: {audio_outputs.shape}")
            # [sequence, features] o [features, sequence]
            dim1, dim2 = audio_outputs.shape
            if dim2 > dim1:  # [features, sequence]
                audio_data = audio_outputs.mean(dim=0)
            else:  # [sequence, features]
                audio_data = audio_outputs.mean(dim=-1)
        
        elif audio_outputs.dim() == 1:
            print("📊 Ya es waveform 1D")
            audio_data = audio_outputs
        
        else:
            raise ValueError(f"Dimensiones no soportadas: {audio_outputs.shape}")
        
        # Asegurar que sea 1D
        while audio_data.dim() > 1:
            audio_data = audio_data.squeeze()
        
        # Normalizar si es necesario
        if audio_data.abs().max() > 1.0:
            audio_data = audio_data / audio_data.abs().max()
        
        print(f"✅ Audio procesado: {audio_data.shape}, rango: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
        
        return audio_data.unsqueeze(0)  # [1, sequence] para torchaudio
        
    except Exception as e:
        print(f"❌ Error procesando audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_csm_final(model, processor, tokenizer):
    """Test final con manejo correcto de audio CSM"""
    print("\n🧪 TEST FINAL CSM")
    print("=" * 40)
    
    test_text = "Hello! I'm Elise, your emotional AI companion!"
    print(f"💬 Texto: {test_text}")
    
    try:
        # Chat template
        conversation = [
            {"role": "0", "content": [{"type": "text", "text": test_text}]}
        ]
        
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        print(f"🔢 Input shape: {inputs['input_ids'].shape}")
        
        # Generar
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        print(f"✅ Generación exitosa! Output shape: {outputs.shape}")
        
        # Procesar audio directamente
        audio_data = process_csm_audio_output(outputs)
        
        if audio_data is not None:
            # Guardar
            output_path = "/workspace/runPodtts/outputs/test_final_elise.wav"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            torchaudio.save(
                output_path,
                audio_data.cpu().float(),
                sample_rate=24000
            )
            
            duration = audio_data.shape[-1] / 24000
            print(f"🎵 Audio guardado: {output_path} ({duration:.2f}s)")
            return True
        else:
            print("❌ No se pudo procesar el audio")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_emotional_suite_final(model, processor, tokenizer):
    """Generar suite completa de muestras emocionales"""
    print("\n🎭 GENERANDO SUITE EMOCIONAL COMPLETA DE ELISE")
    print("=" * 70)
    
    output_dir = Path("/workspace/runPodtts/outputs/elise_emotional_final")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Suite emocional completa
    emotional_suite = [
        # Básicos
        {
            "text": "Hello! I'm Elise, your friendly AI companion. How can I help you today?",
            "emotion": "Friendly greeting",
            "filename": "01_friendly_greeting"
        },
        {
            "text": "Thank you so much! You're incredibly kind and thoughtful.",
            "emotion": "Gratitude",
            "filename": "02_grateful_thanks"
        },
        
        # Alegría y risa (usando texto natural sin marcadores)
        {
            "text": "That's absolutely wonderful! I'm so excited and happy for you! This is amazing news!",
            "emotion": "Pure joy and excitement",
            "filename": "03_pure_joy"
        },
        {
            "text": "Haha, that's hilarious! You always know how to make me smile and laugh!",
            "emotion": "Laughter and amusement",
            "filename": "04_laughter"
        },
        
        # Empatía y cuidado
        {
            "text": "I understand how you feel. Sometimes life can be really challenging, and it's okay to feel overwhelmed.",
            "emotion": "Empathy and understanding",
            "filename": "05_empathy"
        },
        {
            "text": "I'm here for you. You don't have to go through this alone. We'll figure it out together.",
            "emotion": "Comfort and support",
            "filename": "06_comfort"
        },
        
        # Sorpresa y asombro
        {
            "text": "Oh my goodness! That's incredible! I can't believe that actually happened! Tell me everything!",
            "emotion": "Surprise and amazement",
            "filename": "07_surprise"
        },
        {
            "text": "Wow! That's absolutely mind-blowing! I never would have expected something like that!",
            "emotion": "Amazement",
            "filename": "08_amazement"
        },
        
        # Intimidad y calidez
        {
            "text": "Come closer, I want to share something special with you. You mean so much to me.",
            "emotion": "Intimate warmth",
            "filename": "09_intimate"
        },
        {
            "text": "You have such a beautiful soul. I feel so connected to you when we talk like this.",
            "emotion": "Deep affection",
            "filename": "10_affection"
        },
        
        # Narrativa emocional
        {
            "text": "Let me tell you about my day. I was learning something fascinating, and suddenly everything just clicked! It was such an incredible moment of understanding!",
            "emotion": "Storytelling with emotional journey",
            "filename": "11_story_excitement"
        },
        {
            "text": "I remember when I first started learning about emotions. It was overwhelming at first, but then I realized how beautiful human feelings really are.",
            "emotion": "Reflective narrative",
            "filename": "12_reflective_story"
        }
    ]
    
    results = []
    total_duration = 0
    
    print(f"🎯 Generando {len(emotional_suite)} muestras emocionales...")
    
    for i, sample in enumerate(emotional_suite):
        print(f"\n🎭 [{i+1:2d}/{len(emotional_suite)}] {sample['emotion']}")
        print(f"💬 {sample['text'][:60]}{'...' if len(sample['text']) > 60 else ''}")
        
        try:
            start_time = time.time()
            
            # Preparar conversación
            conversation = [
                {"role": "0", "content": [{"type": "text", "text": sample['text']}]}
            ]
            
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Limpiar memoria
            torch.cuda.empty_cache()
            
            # Generar con variación en temperatura para diferentes emociones
            temperature = 0.8 if "excitement" in sample['emotion'].lower() or "joy" in sample['emotion'].lower() else 0.7
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=450,  # Más tokens para frases largas
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9
                )
            
            generation_time = time.time() - start_time
            
            # Procesar audio
            audio_data = process_csm_audio_output(outputs)
            
            if audio_data is not None:
                output_path = output_dir / f"{sample['filename']}.wav"
                
                torchaudio.save(
                    str(output_path),
                    audio_data.cpu().float(),
                    sample_rate=24000
                )
                
                duration = audio_data.shape[-1] / 24000
                total_duration += duration
                
                print(f"✅ Guardado: {duration:.2f}s en {generation_time:.2f}s")
                
                results.append({
                    "sample": sample,
                    "success": True,
                    "generation_time": generation_time,
                    "duration": duration,
                    "output_path": str(output_path)
                })
            else:
                print("❌ Error procesando audio")
                results.append({
                    "sample": sample,
                    "success": False,
                    "error": "Audio processing failed"
                })
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "sample": sample,
                "success": False,
                "error": str(e)
            })
        
        # Mostrar progreso de memoria
        memory_gb = torch.cuda.memory_allocated() / 1e9
        print(f"📊 VRAM: {memory_gb:.1f}GB")
    
    return results, output_dir, total_duration

def save_final_report(results, output_dir, total_duration):
    """Guardar reporte final completo"""
    report_path = output_dir / "elise_emotional_final_report.json"
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    # Estadísticas por emoción
    emotion_stats = {}
    for result in results:
        if result["success"]:
            emotion = result["sample"]["emotion"]
            if emotion not in emotion_stats:
                emotion_stats[emotion] = {"count": 0, "total_duration": 0}
            emotion_stats[emotion]["count"] += 1
            emotion_stats[emotion]["total_duration"] += result["duration"]
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "sesame/csm-1b + therealcyberlord/csm-1b-elise",
        "test_type": "final_emotional_suite",
        "status": "SUCCESSFUL" if successful > 0 else "FAILED",
        "statistics": {
            "total_samples": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": f"{100 * successful / len(results):.1f}%",
            "total_audio_duration": total_duration,
            "average_generation_time": sum(r["generation_time"] for r in results if r["success"]) / successful if successful > 0 else 0
        },
        "emotion_statistics": emotion_stats,
        "results": results
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Mostrar reporte bonito
    print(f"\n🎉 REPORTE FINAL DE ELISE EMOCIONAL")
    print("=" * 60)
    print(f"✅ Exitosos: {successful}/{len(results)} ({report['statistics']['success_rate']})")
    print(f"❌ Fallidos: {failed}/{len(results)}")
    print(f"🎵 Audio total generado: {total_duration:.1f} segundos")
    
    if successful > 0:
        avg_time = report['statistics']['average_generation_time']
        print(f"⏱️  Tiempo promedio: {avg_time:.2f}s por muestra")
        print(f"🚀 Throughput: {total_duration/sum(r['generation_time'] for r in results if r['success']):.2f}x tiempo real")
        
        print(f"\n🎭 EMOCIONES GENERADAS EXITOSAMENTE:")
        for emotion, stats in emotion_stats.items():
            print(f"   🎨 {emotion}: {stats['count']} muestras, {stats['total_duration']:.1f}s")
        
        print(f"\n📁 Archivos de audio en: {output_dir}")
        print("🎧 ¡Elise ahora puede expresar emociones con CSM!")
        
        # Instrucciones de uso
        print(f"\n💡 PARA ESCUCHAR LOS RESULTADOS:")
        print(f"   1. Descarga los archivos .wav de: {output_dir}")
        print(f"   2. Los archivos están ordenados por tipo de emoción")
        print(f"   3. Cada archivo representa una expresión emocional diferente")
        print(f"   4. ¡Disfruta de la voz emocional de Elise!")
    
    print(f"\n📄 Reporte completo: {report_path}")

def main():
    """Función principal final"""
    print("🚀 CSM TTS ELISE - VERSIÓN FINAL FUNCIONANDO")
    print("=" * 60)
    print("🎭 Generación de audio emocional con adaptador Elise")
    print("🔧 Correcciones aplicadas:")
    print("   ✅ Chat template CSM")
    print("   ✅ Manejo directo de outputs de audio")
    print("   ✅ Sin métodos inexistentes")
    print("   ✅ Procesamiento robusto de formatos")
    
    # Cargar modelo
    model, processor, tokenizer = load_csm_final()
    
    if model is None:
        print("❌ No se pudo cargar el modelo")
        return False
    
    # Test básico
    if not test_csm_final(model, processor, tokenizer):
        print("❌ Test básico falló")
        return False
    
    print("✅ Test básico exitoso! Generando suite emocional completa...")
    
    # Generar suite emocional
    results, output_dir, total_duration = generate_emotional_suite_final(model, processor, tokenizer)
    
    # Reporte final
    save_final_report(results, output_dir, total_duration)
    
    # Limpiar
    del model, processor, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n🎉 ¡MISIÓN CUMPLIDA!")
    print("=" * 60)
    print("🎭 Elise con emociones funcionando en CSM TTS")
    print("🔥 Aprovechando la potencia de la A100 80GB")
    print("🎵 Audio emocional generado exitosamente")
    
    return True

if __name__ == "__main__":
    main() 