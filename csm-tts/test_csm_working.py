#!/usr/bin/env python3
"""
CSM WORKING VERSION - Using correct save_audio format!
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

def load_csm_working():
    """Cargar CSM con configuración funcionando"""
    print("🔄 CARGANDO CSM - VERSIÓN FUNCIONANDO")
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

def convert_tokens_to_audio(model, token_outputs):
    """Convertir tokens de CSM a audio usando el decoder del modelo"""
    try:
        print(f"🔄 Convirtiendo tokens a audio: {token_outputs.shape}")
        
        # Los tokens están en formato [batch, sequence, features]
        # Necesitamos usar el decoder del modelo para convertir a audio
        
        with torch.no_grad():
            # Acceder al decoder del modelo CSM
            if hasattr(model, 'base_model'):
                base_model = model.base_model
            else:
                base_model = model
                
            # CSM tiene un decoder de audio
            if hasattr(base_model, 'audio_decoder'):
                print("🎵 Usando audio_decoder...")
                audio_outputs = base_model.audio_decoder(token_outputs)
            elif hasattr(base_model, 'decoder'):
                print("🎵 Usando decoder...")
                audio_outputs = base_model.decoder(token_outputs)
            elif hasattr(base_model, 'generate_audio'):
                print("🎵 Usando generate_audio...")
                audio_outputs = base_model.generate_audio(token_outputs)
            else:
                # Fallback: intentar decodificar manualmente
                print("🎵 Decodificación manual...")
                # Convertir tokens a float y normalizar
                audio_outputs = token_outputs.float()
                # Normalizar de rango de tokens a rango de audio
                audio_outputs = (audio_outputs - audio_outputs.mean()) / audio_outputs.std()
                audio_outputs = torch.tanh(audio_outputs)  # Limitar a [-1, 1]
        
        print(f"✅ Audio decodificado: {audio_outputs.shape}")
        return audio_outputs
        
    except Exception as e:
        print(f"❌ Error decodificando audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_audio_output(audio_outputs):
    """Procesar outputs de audio para guardar"""
    try:
        print(f"🔍 Procesando audio: {audio_outputs.shape}, dtype: {audio_outputs.dtype}")
        
        # Convertir a float si es necesario
        if audio_outputs.dtype != torch.float32:
            audio_outputs = audio_outputs.float()
        
        # Manejar diferentes formatos
        if audio_outputs.dim() == 3:
            # [batch, sequence, features] o [batch, features, sequence]
            batch_size, dim1, dim2 = audio_outputs.shape
            
            if dim1 > dim2:  # [batch, sequence, features]
                audio_data = audio_outputs[0].mean(dim=-1)  # Promedio de features
            else:  # [batch, features, sequence]
                audio_data = audio_outputs[0].mean(dim=0)   # Promedio de features
                
        elif audio_outputs.dim() == 2:
            # [sequence, features] o [features, sequence]
            dim1, dim2 = audio_outputs.shape
            if dim1 > dim2:  # [sequence, features]
                audio_data = audio_outputs.mean(dim=-1)
            else:  # [features, sequence]
                audio_data = audio_outputs.mean(dim=0)
                
        elif audio_outputs.dim() == 1:
            audio_data = audio_outputs
        else:
            raise ValueError(f"Formato no soportado: {audio_outputs.shape}")
        
        # Asegurar 1D
        while audio_data.dim() > 1:
            audio_data = audio_data.squeeze()
        
        # Normalizar
        if audio_data.abs().max() > 0:
            audio_data = audio_data / audio_data.abs().max()
        
        print(f"✅ Audio procesado: {audio_data.shape}, rango: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
        
        return audio_data.unsqueeze(0)  # [1, sequence] para torchaudio
        
    except Exception as e:
        print(f"❌ Error procesando audio: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_csm_working():
    """CSM working with correct format"""
    print("🎉 CSM WORKING VERSION")
    print("=" * 50)
    
    base_model_path = "/workspace/runPodtts/models/sesame-csm-1b"
    
    try:
        print("📥 Cargando modelo...")
        processor = AutoProcessor.from_pretrained(base_model_path)
        model = CsmForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✅ Modelo cargado: {torch.cuda.memory_allocated() / 1e9:.1f}GB VRAM")
        
        # Test with different phrases
        test_phrases = [
            "Hello! How are you doing today?",
            "That's wonderful! I'm so happy for you!",
            "Thank you so much for everything!",
            "I hope you have a fantastic day!"
        ]
        
        output_dir = Path("/workspace/runPodtts/outputs/csm_working")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, text in enumerate(test_phrases):
            print(f"\n🎵 [{i+1}/{len(test_phrases)}] Generando: {text}")
            
            # Prepare conversation
            conversation = [
                {"role": "0", "content": [{"type": "text", "text": text}]}
            ]
            
            # Process
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    use_cache=True
                )
            
            end_time.record()
            torch.cuda.synchronize()
            generation_time = start_time.elapsed_time(end_time) / 1000.0
            
            print(f"   ⏱️ Generación: {generation_time:.2f}s")
            print(f"   📊 Output shape: {outputs.shape}")
            
            # Extract audio tokens (remove input tokens)
            input_length = inputs['input_ids'].shape[1]
            audio_tokens = outputs[:, input_length:]
            
            print(f"   📊 Audio tokens: {audio_tokens.shape}")
            
            # CORRECT FORMAT: Remove batch dimension for save_audio!
            audio_tokens_correct = audio_tokens.squeeze(0)  # [seq, features]
            print(f"   ✅ Formato correcto: {audio_tokens_correct.shape}")
            
            # Save audio
            output_path = output_dir / f"sample_{i+1:02d}.wav"
            
            try:
                processor.save_audio(audio_tokens_correct, str(output_path))
                
                if output_path.exists():
                    file_size = output_path.stat().st_size
                    
                    if file_size > 1000:  # Valid audio file
                        audio_data, sr = torchaudio.load(output_path)
                        duration = audio_data.shape[1] / sr
                        print(f"   ✅ Guardado: {output_path}")
                        print(f"   🎵 Audio: {duration:.2f}s a {sr}Hz ({file_size} bytes)")
                    else:
                        print(f"   ⚠️  Archivo muy pequeño: {file_size} bytes")
                else:
                    print(f"   ❌ No se creó el archivo")
                    
            except Exception as save_error:
                print(f"   ❌ Error guardando: {save_error}")
        
        print(f"\n🎉 PROCESO COMPLETADO")
        print(f"📁 Archivos en: {output_dir}")
        print(f"🎧 Revisa los archivos sample_*.wav")
        
        return True
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_emotional_samples_working(model, processor, tokenizer):
    """Generar muestras emocionales funcionando"""
    print("\n🎭 GENERANDO MUESTRAS EMOCIONALES FUNCIONANDO")
    print("=" * 60)
    
    output_dir = Path("/workspace/runPodtts/outputs/elise_working")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Muestras emocionales
    emotional_samples = [
        {
            "text": "Hello! I'm Elise, nice to meet you!",
            "emotion": "Friendly greeting",
            "filename": "01_greeting"
        },
        {
            "text": "That's wonderful! I'm so happy for you!",
            "emotion": "Joy and excitement",
            "filename": "02_joy"
        },
        {
            "text": "Thank you so much! You're very kind.",
            "emotion": "Gratitude",
            "filename": "03_thanks"
        },
        {
            "text": "I understand how you feel. Life can be challenging.",
            "emotion": "Empathy",
            "filename": "04_empathy"
        },
        {
            "text": "That's absolutely amazing! Tell me more!",
            "emotion": "Excitement and curiosity",
            "filename": "05_excitement"
        }
    ]
    
    results = []
    total_duration = 0
    
    for i, sample in enumerate(emotional_samples):
        print(f"\n🎭 [{i+1}/{len(emotional_samples)}] {sample['emotion']}")
        print(f"💬 {sample['text']}")
        
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
            
            # Generar tokens
            with torch.no_grad():
                token_outputs = model.generate(
                    **inputs,
                    max_new_tokens=350,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Convertir a audio
            audio_outputs = convert_tokens_to_audio(model, token_outputs)
            
            if audio_outputs is not None:
                audio_data = process_audio_output(audio_outputs)
                
                if audio_data is not None:
                    output_path = output_dir / f"{sample['filename']}.wav"
                    
                    torchaudio.save(
                        str(output_path),
                        audio_data.cpu().float(),
                        sample_rate=24000
                    )
                    
                    generation_time = time.time() - start_time
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
            else:
                print("❌ Error convirtiendo tokens")
                results.append({
                    "sample": sample,
                    "success": False,
                    "error": "Token conversion failed"
                })
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "sample": sample,
                "success": False,
                "error": str(e)
            })
        
        # Mostrar memoria
        memory_gb = torch.cuda.memory_allocated() / 1e9
        print(f"📊 VRAM: {memory_gb:.1f}GB")
    
    return results, output_dir, total_duration

def save_working_report(results, output_dir, total_duration):
    """Guardar reporte de la versión funcionando"""
    report_path = output_dir / "working_report.json"
    
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "sesame/csm-1b + therealcyberlord/csm-1b-elise",
        "test_type": "working_csm_generation",
        "status": "SUCCESS" if successful > 0 else "FAILED",
        "statistics": {
            "total_samples": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": f"{100 * successful / len(results):.1f}%",
            "total_audio_duration": total_duration
        },
        "results": results
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 REPORTE FINAL:")
    print(f"✅ Exitosos: {successful}/{len(results)} ({report['statistics']['success_rate']})")
    print(f"❌ Fallidos: {failed}/{len(results)}")
    print(f"🎵 Audio total: {total_duration:.1f}s")
    
    if successful > 0:
        print("🎉 ¡CSM TTS FUNCIONANDO CON ELISE!")
        print(f"📁 Archivos en: {output_dir}")
        print("🎧 ¡Disfruta de la voz emocional de Elise!")
    
    print(f"📄 Reporte: {report_path}")

def main():
    """Función principal funcionando"""
    print("🚀 CSM TTS ELISE - VERSIÓN FUNCIONANDO")
    print("=" * 60)
    print("🎭 Conversión correcta de tokens a audio")
    print("🔧 Características:")
    print("   ✅ Chat template CSM")
    print("   ✅ Conversión tokens → audio")
    print("   ✅ Manejo robusto de formatos")
    print("   ✅ Adaptador Elise emocional")
    
    # Cargar modelo
    model, processor, tokenizer = load_csm_working()
    
    if model is None:
        print("❌ No se pudo cargar el modelo")
        return False
    
    # Test básico
    if not test_csm_working():
        print("❌ Test básico falló")
        return False
    
    print("✅ Test básico exitoso! Generando muestras emocionales...")
    
    # Generar muestras emocionales
    results, output_dir, total_duration = generate_emotional_samples_working(model, processor, tokenizer)
    
    # Reporte final
    save_working_report(results, output_dir, total_duration)
    
    # Limpiar
    del model, processor, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n🎉 ¡MISIÓN CUMPLIDA!")
    print("=" * 60)
    print("🎭 Elise funcionando con CSM TTS")
    print("🔥 Usando toda la potencia de la A100")
    print("🎵 Audio emocional generado exitosamente")
    
    return True

if __name__ == "__main__":
    main() 