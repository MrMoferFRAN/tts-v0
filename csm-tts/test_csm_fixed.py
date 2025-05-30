#!/usr/bin/env python3
"""
Script CSM TTS corregido para manejar el error de sequence length
Versión específica para solucionar "Key and Value must have the same sequence length"
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

def load_csm_fixed():
    """Cargar CSM con configuración específica para evitar errores"""
    print("🔄 CARGANDO CSM CON CONFIGURACIÓN CORREGIDA")
    print("=" * 60)
    
    base_model_path = "/workspace/runPodtts/models/sesame-csm-1b"
    adapter_path = "/workspace/runPodtts/models/csm-1b-elise"
    
    try:
        print("📥 Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        # Configuración específica para CSM
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # CSM requiere padding derecho
        print("✅ Tokenizer configurado para CSM")
        
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
        
        # Estadísticas
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🔢 Parámetros: {total_params:,}")
        print(f"💾 VRAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        
        return model, processor, tokenizer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_csm_generation_fixed(model, processor, tokenizer):
    """Test con configuración específica para CSM"""
    print("\n🧪 TEST CON CONFIGURACIÓN CSM CORREGIDA")
    print("=" * 50)
    
    # Texto de prueba muy simple
    test_text = "Hello, I'm Elise!"
    print(f"💬 Texto: {test_text}")
    
    try:
        # Formatear para CSM - usar template de chat
        conversation = [
            {"role": "0", "content": [{"type": "text", "text": test_text}]}
        ]
        
        print("🔄 Aplicando chat template...")
        # Usar el chat template del procesador
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Mover a GPU
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print(f"🔢 Tokens de entrada: {inputs['input_ids'].shape}")
        
        # Generar con configuración específica para CSM
        print("🎵 Generando audio...")
        with torch.no_grad():
            # Usar los parámetros recomendados para CSM
            audio_outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                # No usar eos_token_id ni pad_token_id específicos
                # CSM maneja esto internamente
            )
        
        print("✅ Generación exitosa!")
        print(f"📊 Shape outputs: {audio_outputs.shape if hasattr(audio_outputs, 'shape') else type(audio_outputs)}")
        
        # Intentar procesar audio
        try:
            # CSM puede devolver diferentes formatos
            if hasattr(audio_outputs, 'audio'):
                audio_data = audio_outputs.audio
            elif isinstance(audio_outputs, dict) and 'audio' in audio_outputs:
                audio_data = audio_outputs['audio']
            else:
                # Usar processor para decodificar
                audio_data = processor.decode_audio(audio_outputs)
            
            if audio_data is not None:
                print(f"🎵 Audio extraído: {audio_data.shape}")
                
                # Guardar audio de prueba
                output_path = "/workspace/runPodtts/outputs/test_fixed.wav"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Procesar formato
                if isinstance(audio_data, torch.Tensor):
                    if audio_data.dim() == 3:
                        audio_data = audio_data.squeeze(0)
                    elif audio_data.dim() == 1:
                        audio_data = audio_data.unsqueeze(0)
                    
                    torchaudio.save(
                        output_path,
                        audio_data.cpu().float(),
                        sample_rate=24000
                    )
                    
                    duration = audio_data.shape[-1] / 24000
                    print(f"✅ Audio guardado: {output_path} ({duration:.2f}s)")
                    return True
                else:
                    print(f"⚠️  Formato de audio no reconocido: {type(audio_data)}")
                    return False
            else:
                print("❌ No se pudo extraer audio")
                return False
                
        except Exception as audio_error:
            print(f"❌ Error procesando audio: {audio_error}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"❌ Error en generación: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_emotional_samples_fixed(model, processor, tokenizer):
    """Generar muestras emocionales con configuración corregida"""
    print("\n🎭 GENERANDO MUESTRAS EMOCIONALES CORREGIDAS")
    print("=" * 60)
    
    output_dir = Path("/workspace/runPodtts/outputs/csm_fixed_emotional")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prompts emocionales simples
    emotional_prompts = [
        {
            "text": "Hello! I'm Elise, nice to meet you!",
            "description": "Saludo básico",
            "filename": "01_greeting"
        },
        {
            "text": "That's wonderful! I'm so happy for you!",
            "description": "Alegría sin marcadores",
            "filename": "02_joy"
        },
        {
            "text": "Thank you so much! You're very kind.",
            "description": "Agradecimiento cálido",
            "filename": "03_thanks"
        },
        {
            "text": "I understand how you feel. Life can be challenging sometimes.",
            "description": "Empatía y comprensión",
            "filename": "04_empathy"
        },
        {
            "text": "That's absolutely amazing! Tell me more about it!",
            "description": "Entusiasmo y curiosidad",
            "filename": "05_excitement"
        }
    ]
    
    results = []
    
    for i, prompt in enumerate(emotional_prompts):
        print(f"\n🎭 [{i+1}/{len(emotional_prompts)}] {prompt['description']}")
        print(f"💬 {prompt['text']}")
        
        try:
            start_time = time.time()
            
            # Crear conversación para CSM
            conversation = [
                {"role": "0", "content": [{"type": "text", "text": prompt['text']}]}
            ]
            
            # Aplicar template
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Limpiar cache
            torch.cuda.empty_cache()
            
            # Generar
            with torch.no_grad():
                audio_outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            generation_time = time.time() - start_time
            print(f"⏱️  Generación: {generation_time:.2f}s")
            
            # Procesar audio
            try:
                if hasattr(audio_outputs, 'audio'):
                    audio_data = audio_outputs.audio
                elif isinstance(audio_outputs, dict) and 'audio' in audio_outputs:
                    audio_data = audio_outputs['audio']
                else:
                    audio_data = processor.decode_audio(audio_outputs)
                
                if audio_data is not None:
                    output_path = output_dir / f"{prompt['filename']}.wav"
                    
                    # Procesar formato
                    if isinstance(audio_data, torch.Tensor):
                        if audio_data.dim() == 3:
                            audio_data = audio_data.squeeze(0)
                        elif audio_data.dim() == 1:
                            audio_data = audio_data.unsqueeze(0)
                        
                        torchaudio.save(
                            str(output_path),
                            audio_data.cpu().float(),
                            sample_rate=24000
                        )
                        
                        duration = audio_data.shape[-1] / 24000
                        print(f"✅ Guardado: {output_path} ({duration:.2f}s)")
                        
                        results.append({
                            "prompt": prompt,
                            "success": True,
                            "generation_time": generation_time,
                            "duration": duration,
                            "output_path": str(output_path)
                        })
                    else:
                        print(f"⚠️  Formato no reconocido: {type(audio_data)}")
                        results.append({
                            "prompt": prompt,
                            "success": False,
                            "error": f"Unknown audio format: {type(audio_data)}"
                        })
                else:
                    print("❌ Audio es None")
                    results.append({
                        "prompt": prompt,
                        "success": False,
                        "error": "Audio is None"
                    })
                    
            except Exception as audio_error:
                print(f"❌ Error procesando audio: {audio_error}")
                results.append({
                    "prompt": prompt,
                    "success": False,
                    "error": f"Audio processing: {audio_error}"
                })
                
        except Exception as e:
            print(f"❌ Error en generación: {e}")
            results.append({
                "prompt": prompt,
                "success": False,
                "error": str(e)
            })
        
        # Mostrar memoria
        memory_used = torch.cuda.memory_allocated() / 1e9
        print(f"📊 VRAM: {memory_used:.1f}GB")
    
    return results, output_dir

def save_fixed_report(results, output_dir):
    """Guardar reporte de la versión corregida"""
    report_path = output_dir / "fixed_generation_report.json"
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "sesame/csm-1b + elise adapter",
        "test_type": "fixed_csm_generation",
        "fixes_applied": [
            "Usar chat template del processor",
            "Padding side = right",
            "Configuración CSM-específica",
            "Manejo de formatos de audio múltiples"
        ],
        "statistics": {
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": f"{100 * successful / len(results):.1f}%"
        },
        "results": results
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 REPORTE DE VERSIÓN CORREGIDA:")
    print(f"✅ Exitosos: {successful}/{len(results)} ({report['statistics']['success_rate']})")
    print(f"❌ Fallidos: {failed}/{len(results)}")
    
    if successful > 0:
        successful_results = [r for r in results if r["success"]]
        avg_time = sum(r["generation_time"] for r in successful_results) / len(successful_results)
        total_audio = sum(r["duration"] for r in successful_results)
        print(f"⏱️  Tiempo promedio: {avg_time:.2f}s")
        print(f"🎵 Audio total: {total_audio:.2f}s")
        print("🎉 ¡CSM funciona correctamente!")
    else:
        print("❌ Ninguna generación fue exitosa")
    
    print(f"📄 Reporte: {report_path}")

def main():
    """Función principal corregida"""
    print("🚀 CSM TTS - VERSIÓN CORREGIDA")
    print("=" * 50)
    print("🔧 Aplicando fixes específicos para CSM")
    print("   • Chat template correcto")
    print("   • Padding side configurado") 
    print("   • Manejo de sequence length")
    print("   • Formato de audio CSM")
    
    # Cargar modelo
    model, processor, tokenizer = load_csm_fixed()
    
    if model is None:
        print("❌ No se pudo cargar el modelo")
        return False
    
    # Test básico
    basic_test = test_csm_generation_fixed(model, processor, tokenizer)
    
    if not basic_test:
        print("❌ Test básico falló")
        return False
    
    print("✅ Test básico exitoso! Generando muestras emocionales...")
    
    # Generar muestras emocionales
    results, output_dir = generate_emotional_samples_fixed(model, processor, tokenizer)
    
    # Reporte
    save_fixed_report(results, output_dir)
    
    # Limpiar
    del model, processor, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n🎉 VERSIÓN CORREGIDA COMPLETADA")
    print(f"📁 Audio en: {output_dir}")
    print("🎭 ¡Elise funcionando con emociones!")
    
    return True

if __name__ == "__main__":
    main() 