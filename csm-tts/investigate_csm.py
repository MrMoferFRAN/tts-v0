#!/usr/bin/env python3
"""
Investigar la estructura real de CSM y encontrar el método correcto para generar audio
Optimizado para A100 80GB
"""
import os
import torch
import json
import time
import psutil
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, CsmForConditionalGeneration
from peft import PeftModel
import torchaudio
import gc

# Configurar para máximo rendimiento en A100
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "32"

def investigate_csm_model():
    """Investigar la estructura del modelo CSM para encontrar la forma correcta de generar audio"""
    print("🔍 INVESTIGANDO ESTRUCTURA DE CSM")
    print("=" * 60)
    
    base_model_path = "/workspace/runPodtts/models/sesame-csm-1b"
    adapter_path = "/workspace/runPodtts/models/csm-1b-elise"
    
    try:
        print("📥 Cargando componentes...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        processor = AutoProcessor.from_pretrained(base_model_path)
        
        # Cargar con máxima VRAM
        print("🔥 Cargando modelo para investigación...")
        base_model = CsmForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "75GB"},  # Usar más VRAM
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        memory_gb = torch.cuda.memory_allocated() / 1e9
        print(f"💾 VRAM usada: {memory_gb:.1f}GB")
        
        print("\n🔍 ESTRUCTURA DEL MODELO:")
        print("=" * 40)
        
        # Investigar estructura
        if hasattr(model, 'base_model'):
            base = model.base_model
        else:
            base = model
            
        print(f"📋 Tipo de modelo: {type(base)}")
        print(f"📋 Atributos del modelo:")
        
        attrs = [attr for attr in dir(base) if not attr.startswith('_')]
        important_attrs = [attr for attr in attrs if any(keyword in attr.lower() 
                          for keyword in ['audio', 'decode', 'generate', 'forward', 'config'])]
        
        for attr in important_attrs:
            print(f"   🔸 {attr}")
            
        # Revisar configuración
        if hasattr(base, 'config'):
            config = base.config
            print(f"\n📋 CONFIG DEL MODELO:")
            config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
            for key, value in config_dict.items():
                if any(keyword in key.lower() for keyword in ['audio', 'sample', 'rate', 'vocab', 'dim']):
                    print(f"   🔸 {key}: {value}")
        
        # Revisar processor
        print(f"\n📋 PROCESSOR:")
        print(f"   🔸 Tipo: {type(processor)}")
        processor_attrs = [attr for attr in dir(processor) if not attr.startswith('_')]
        audio_attrs = [attr for attr in processor_attrs if 'audio' in attr.lower() or 'decode' in attr.lower()]
        for attr in audio_attrs:
            print(f"   🔸 {attr}")
            
        return model, processor, tokenizer, base
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def test_csm_methods(model, processor, tokenizer, base_model):
    """Probar diferentes métodos para generar audio con CSM"""
    print("\n🧪 PROBANDO MÉTODOS DE GENERACIÓN")
    print("=" * 50)
    
    test_text = "Hello! I'm Elise!"
    
    try:
        # Preparar input con chat template
        conversation = [{"role": "0", "content": [{"type": "text", "text": test_text}]}]
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print(f"🔢 Input preparado: {inputs['input_ids'].shape}")
        
        # MÉTODO 1: Generate básico (lo que ya probamos)
        print("\n🔸 MÉTODO 1: Generate básico")
        start_time = time.time()
        with torch.no_grad():
            outputs1 = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        time1 = time.time() - start_time
        print(f"   ⏱️ Tiempo: {time1:.2f}s")
        print(f"   📊 Shape: {outputs1.shape}, dtype: {outputs1.dtype}")
        
        # MÉTODO 2: Usar forward pass directo
        print("\n🔸 MÉTODO 2: Forward pass directo")
        try:
            start_time = time.time()
            with torch.no_grad():
                outputs2 = model(**inputs)
            time2 = time.time() - start_time
            print(f"   ⏱️ Tiempo: {time2:.2f}s")
            print(f"   📊 Outputs keys: {outputs2.keys() if hasattr(outputs2, 'keys') else type(outputs2)}")
            
            # Ver si hay audio en los outputs
            if hasattr(outputs2, 'audio'):
                print(f"   🎵 Audio encontrado: {outputs2.audio.shape}")
            elif hasattr(outputs2, 'logits'):
                print(f"   🔢 Logits: {outputs2.logits.shape}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
        # MÉTODO 3: Buscar métodos específicos de audio
        print("\n🔸 MÉTODO 3: Métodos específicos de audio")
        audio_methods = ['generate_audio', 'decode_audio', 'synthesize', 'tts']
        
        for method_name in audio_methods:
            if hasattr(base_model, method_name):
                print(f"   ✅ Encontrado: {method_name}")
                try:
                    method = getattr(base_model, method_name)
                    print(f"      📋 Tipo: {type(method)}")
                    if callable(method):
                        print(f"      🔧 Es callable")
                except Exception as e:
                    print(f"      ❌ Error accediendo: {e}")
            else:
                print(f"   ❌ No encontrado: {method_name}")
                
        # MÉTODO 4: Buscar en processor
        print("\n🔸 MÉTODO 4: Métodos del processor")
        if hasattr(processor, 'decode'):
            print("   ✅ Processor tiene decode")
            try:
                start_time = time.time()
                decoded = processor.decode(outputs1[0])
                time4 = time.time() - start_time
                print(f"   ⏱️ Tiempo decode: {time4:.2f}s")
                print(f"   📊 Decoded type: {type(decoded)}")
                if hasattr(decoded, 'shape'):
                    print(f"   📊 Decoded shape: {decoded.shape}")
            except Exception as e:
                print(f"   ❌ Error decode: {e}")
                
        # MÉTODO 5: Investigar la salida de generate más profundo
        print("\n🔸 MÉTODO 5: Análisis profundo de generate output")
        print(f"   📊 Output shape: {outputs1.shape}")
        print(f"   📊 Output dtype: {outputs1.dtype}")
        print(f"   📊 Output device: {outputs1.device}")
        print(f"   📊 Min/Max valores: {outputs1.min().item()}/{outputs1.max().item()}")
        
        # Probar si es simplemente reshape/cast el problema
        print("\n🔸 MÉTODO 6: Conversión directa experimental")
        try:
            # Tomar solo los nuevos tokens (sin el input)
            input_len = inputs['input_ids'].shape[1]
            new_tokens = outputs1[:, input_len:]
            print(f"   📊 Nuevos tokens: {new_tokens.shape}")
            
            # Convertir a audio experimental
            if new_tokens.dim() == 3:
                # [batch, seq, features] -> tratar como espectrograma
                audio_attempt = new_tokens[0].float()  # [seq, features]
                
                # Probar diferentes conversiones
                print("   🔄 Probando conversiones...")
                
                # Opción A: Promedio de features
                audio_a = audio_attempt.mean(dim=-1)
                print(f"      📊 Opción A (mean): {audio_a.shape}, rango: [{audio_a.min():.3f}, {audio_a.max():.3f}]")
                
                # Opción B: Primera feature
                audio_b = audio_attempt[:, 0] if audio_attempt.shape[1] > 0 else audio_attempt.flatten()
                print(f"      📊 Opción B (first): {audio_b.shape}, rango: [{audio_b.min():.3f}, {audio_b.max():.3f}]")
                
                # Opción C: Normalizar como waveform
                audio_c = audio_attempt.flatten()
                audio_c = (audio_c - audio_c.mean()) / (audio_c.std() + 1e-8)
                audio_c = torch.tanh(audio_c)  # Limitar a [-1, 1]
                print(f"      📊 Opción C (norm): {audio_c.shape}, rango: [{audio_c.min():.3f}, {audio_c.max():.3f}]")
                
                # Guardar las 3 opciones para escuchar
                output_dir = Path("/workspace/runPodtts/outputs/csm_investigation")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                for i, (audio, name) in enumerate([(audio_a, "mean"), (audio_b, "first"), (audio_c, "norm")]):
                    if len(audio) > 100:  # Solo si tiene suficientes samples
                        # Asegurar formato correcto
                        if audio.dim() == 1:
                            audio = audio.unsqueeze(0)
                        
                        output_path = output_dir / f"test_{name}.wav"
                        torchaudio.save(
                            str(output_path),
                            audio.cpu().float(),
                            sample_rate=24000
                        )
                        duration = audio.shape[-1] / 24000
                        print(f"      💾 Guardado {name}: {output_path} ({duration:.2f}s)")
                
        except Exception as e:
            print(f"   ❌ Error conversión experimental: {e}")
            import traceback
            traceback.print_exc()
            
        return outputs1
        
    except Exception as e:
        print(f"❌ Error en tests: {e}")
        import traceback
        traceback.print_exc()
        return None

def optimize_for_a100(model, processor, tokenizer):
    """Optimizar el modelo para usar más recursos de la A100"""
    print("\n🚀 OPTIMIZANDO PARA A100")
    print("=" * 40)
    
    # Configurar para batch processing
    test_texts = [
        "Hello! I'm Elise!",
        "How are you today?", 
        "That's wonderful news!",
        "I'm so happy for you!",
        "Thank you very much!"
    ]
    
    print(f"📊 Procesando {len(test_texts)} textos en batch...")
    
    try:
        # Preparar batch
        conversations = []
        for text in test_texts:
            conversations.append([{"role": "0", "content": [{"type": "text", "text": text}]}])
        
        # Procesar batch completo
        start_time = time.time()
        
        batch_inputs = []
        for conv in conversations:
            inputs = processor.apply_chat_template(
                conv, tokenize=True, return_dict=True, return_tensors="pt"
            )
            batch_inputs.append(inputs)
        
        # Combinar en batch (si es posible)
        print("🔄 Generando batch optimizado...")
        
        memory_before = torch.cuda.memory_allocated() / 1e9
        
        batch_outputs = []
        for i, inputs in enumerate(batch_inputs):
            inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1  # Optimizar para velocidad
                )
            batch_outputs.append(outputs)
            
            if i == 0:
                memory_after = torch.cuda.memory_allocated() / 1e9
                print(f"💾 VRAM después de primera generación: {memory_after:.1f}GB (diff: +{memory_after-memory_before:.1f}GB)")
        
        total_time = time.time() - start_time
        print(f"⏱️ Tiempo total batch: {total_time:.2f}s")
        print(f"⚡ Promedio por texto: {total_time/len(test_texts):.2f}s")
        
        # Mostrar stats finales
        final_memory = torch.cuda.memory_allocated() / 1e9
        print(f"💾 VRAM final: {final_memory:.1f}GB")
        
        return batch_outputs
        
    except Exception as e:
        print(f"❌ Error optimización: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Función principal de investigación"""
    print("🔍 INVESTIGACIÓN COMPLETA DE CSM TTS")
    print("=" * 70)
    print("🎯 Objetivos:")
    print("   1. Encontrar método correcto para audio")
    print("   2. Optimizar uso de A100 80GB") 
    print("   3. Acelerar generación")
    print("   4. Solucionar audios vacíos")
    
    # Mostrar recursos disponibles
    print(f"\n📊 RECURSOS DISPONIBLES:")
    print(f"   🔥 GPU: {torch.cuda.get_device_name()}")
    print(f"   💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"   💻 CPU: {psutil.cpu_count()} cores")
    print(f"   🧠 RAM: {psutil.virtual_memory().total / 1e9:.1f}GB")
    
    # Investigar modelo
    model, processor, tokenizer, base_model = investigate_csm_model()
    
    if model is None:
        print("❌ No se pudo cargar el modelo")
        return False
    
    # Probar métodos
    outputs = test_csm_methods(model, processor, tokenizer, base_model)
    
    if outputs is not None:
        print("✅ Al menos un método funcionó")
        
        # Optimizar para A100
        batch_outputs = optimize_for_a100(model, processor, tokenizer)
        
        if batch_outputs:
            print("✅ Optimización A100 exitosa")
        
    else:
        print("❌ Ningún método funcionó")
    
    # Limpiar
    del model, processor, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"\n🎉 INVESTIGACIÓN COMPLETADA")
    print("🔍 Revisa los archivos de audio generados en:")
    print("   📁 /workspacetts-v0/outputs/csm_investigation/")
    print("💡 Los archivos test_*.wav te dirán qué método funciona")
    
    return True

if __name__ == "__main__":
    main() 