#!/usr/bin/env python3
"""
Script robusto para CSM TTS que evita congelamiento y maneja limitaciones de CSM
Versión con timeouts y parámetros seguros
"""
import os
import torch
import json
import psutil
import signal
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, CsmForConditionalGeneration
from peft import PeftModel
import torchaudio
from tqdm import tqdm
import gc

# Configurar variables de entorno
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operación timeout")

def load_csm_simple():
    """Cargar modelo CSM de forma simple y robusta"""
    print("🔄 CARGANDO MODELO CSM DE FORMA ROBUSTA")
    print("=" * 60)
    
    base_model_path = "/workspace/runPodtts/models/sesame-csm-1b"
    adapter_path = "/workspace/runPodtts/models/csm-1b-elise"
    
    try:
        print("📥 Cargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer cargado")
        
        print("📥 Cargando procesador...")
        processor = AutoProcessor.from_pretrained(base_model_path)
        print("✅ Procesador cargado")
        
        print("🔥 Cargando modelo base...")
        base_model = CsmForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True
        )
        print("✅ Modelo base cargado")
        
        print("🎭 Aplicando adaptador Elise...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ Adaptador aplicado")
        
        # Estadísticas del modelo
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🔢 Parámetros: {total_params:,}")
        print(f"💾 VRAM usada: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        
        return model, processor, tokenizer
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

def test_single_generation(model, processor, tokenizer):
    """Test de una sola generación para verificar funcionamiento"""
    print("\n🧪 TEST DE GENERACIÓN SIMPLE")
    print("=" * 50)
    
    test_text = "[0]Hello, I'm Elise!"
    print(f"💬 Texto: {test_text}")
    
    try:
        # Configurar timeout de 60 segundos
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        # Tokenizar
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            padding=True
        ).to("cuda")
        
        print(f"🔢 Tokens: {inputs['input_ids'].shape[1]}")
        
        # Generar con parámetros muy conservadores
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=200,  # Muy conservador
                do_sample=False,     # Determinístico
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False      # Evitar problemas de cache
            )
        
        signal.alarm(0)  # Cancelar timeout
        
        print(f"✅ Generación exitosa! Tokens generados: {outputs.shape[1] - inputs['input_ids'].shape[1]}")
        
        # Intentar extraer audio
        try:
            audio_data = processor.decode_audio(outputs)
            if audio_data is not None:
                print(f"🎵 Audio extraído: {audio_data.shape}")
                return True
            else:
                print("⚠️  Audio es None")
                return False
        except Exception as audio_error:
            print(f"⚠️  Error extrayendo audio: {audio_error}")
            return False
            
    except TimeoutError:
        print("❌ TIMEOUT: La generación se colgó")
        signal.alarm(0)
        return False
    except Exception as e:
        print(f"❌ Error en generación: {e}")
        signal.alarm(0)
        return False

def generate_simple_samples(model, processor, tokenizer):
    """Generar muestras simples con manejo robusto de errores"""
    print("\n🎵 GENERANDO MUESTRAS SIMPLES")
    print("=" * 50)
    
    output_dir = Path("/workspace/runPodtts/outputs/csm_robust_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prompts muy simples para empezar
    simple_prompts = [
        {
            "text": "[0]Hello!",
            "description": "Saludo muy simple",
            "filename": "01_hello"
        },
        {
            "text": "[0]Thank you!",
            "description": "Agradecimiento simple",
            "filename": "02_thanks"
        },
        {
            "text": "[0]How are you?",
            "description": "Pregunta simple",
            "filename": "03_question"
        },
        {
            "text": "[0]I'm happy!",
            "description": "Emoción básica",
            "filename": "04_happy"
        },
        {
            "text": "[0]That's wonderful! <laughs>",
            "description": "Con risa simple",
            "filename": "05_laugh"
        }
    ]
    
    results = []
    
    for i, prompt in enumerate(simple_prompts):
        print(f"\n🎭 [{i+1}/{len(simple_prompts)}] {prompt['description']}")
        print(f"💬 {prompt['text']}")
        
        try:
            # Timeout por muestra
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(45)  # 45 segundos por muestra
            
            start_time = time.time()
            
            # Tokenizar
            inputs = tokenizer(
                prompt['text'],
                return_tensors="pt",
                add_special_tokens=True,
                max_length=64,  # Muy corto
                truncation=True,
                padding=True
            ).to("cuda")
            
            # Limpiar cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Generar con parámetros ultra-conservadores
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=150,  # Muy conservador
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            
            generation_time = time.time() - start_time
            signal.alarm(0)
            
            print(f"⏱️  Generación: {generation_time:.2f}s")
            
            # Procesar audio
            try:
                audio_data = processor.decode_audio(outputs)
                
                if audio_data is not None:
                    output_path = output_dir / f"{prompt['filename']}.wav"
                    
                    # Procesar formato
                    if audio_data.dim() == 3:
                        audio_data = audio_data.squeeze(0)
                    elif audio_data.dim() == 1:
                        audio_data = audio_data.unsqueeze(0)
                    
                    # Guardar
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
                
        except TimeoutError:
            print("❌ TIMEOUT en esta muestra")
            signal.alarm(0)
            results.append({
                "prompt": prompt,
                "success": False,
                "error": "Generation timeout"
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            signal.alarm(0)
            results.append({
                "prompt": prompt,
                "success": False,
                "error": str(e)
            })
        
        # Mostrar stats de memoria
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9
            print(f"📊 VRAM: {memory_used:.1f}GB")
    
    return results, output_dir

def save_robust_report(results, output_dir):
    """Guardar reporte de resultados"""
    report_path = output_dir / "robust_test_report.json"
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "sesame/csm-1b + elise adapter",
        "test_type": "robust_simple_generation",
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
    
    print(f"\n📊 REPORTE FINAL:")
    print(f"✅ Exitosos: {successful}/{len(results)} ({report['statistics']['success_rate']})")
    print(f"❌ Fallidos: {failed}/{len(results)}")
    
    if successful > 0:
        print("🎉 ¡Al menos algunas generaciones funcionaron!")
        successful_results = [r for r in results if r["success"]]
        avg_time = sum(r["generation_time"] for r in successful_results) / len(successful_results)
        total_audio = sum(r["duration"] for r in successful_results)
        print(f"⏱️  Tiempo promedio: {avg_time:.2f}s")
        print(f"🎵 Audio total: {total_audio:.2f}s")
    
    print(f"📄 Reporte: {report_path}")

def main():
    """Función principal robusta"""
    print("🚀 CSM TTS - PRUEBA ROBUSTA ANTI-CONGELAMIENTO")
    print("=" * 70)
    
    # Stats iniciales
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name()}")
        print(f"🔥 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    print(f"💻 CPU: {psutil.cpu_count()} cores")
    print(f"🧠 RAM: {psutil.virtual_memory().total / 1e9:.1f}GB")
    
    # Cargar modelo
    print("\n⏳ Cargando modelo...")
    model, processor, tokenizer = load_csm_simple()
    
    if model is None:
        print("❌ No se pudo cargar el modelo")
        return False
    
    # Test simple primero
    print("\n🧪 Probando generación básica...")
    basic_test = test_single_generation(model, processor, tokenizer)
    
    if not basic_test:
        print("❌ El test básico falló. El modelo tiene problemas.")
        return False
    
    print("✅ Test básico exitoso! Procediendo con muestras...")
    
    # Generar muestras
    results, output_dir = generate_simple_samples(model, processor, tokenizer)
    
    # Reporte final
    save_robust_report(results, output_dir)
    
    # Limpiar
    del model, processor, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print("\n🎉 PRUEBA ROBUSTA COMPLETADA")
    print(f"📁 Archivos en: {output_dir}")
    print("💡 Si esto funcionó, podemos intentar versiones más complejas")
    
    return True

if __name__ == "__main__":
    main() 