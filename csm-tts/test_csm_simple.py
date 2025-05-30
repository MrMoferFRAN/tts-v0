#!/usr/bin/env python3
"""
Simple CSM test - only text, no audio context
"""
import os
import torch
import torchaudio
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, CsmForConditionalGeneration

# Set environment
os.environ["NO_TORCH_COMPILE"] = "1"

def test_csm_text_only():
    """Test CSM with text only - simplest possible approach"""
    print("🚀 CSM SIMPLE TEXT-ONLY TEST")
    print("=" * 50)
    
    base_model_path = "/workspace/runPodtts/models/sesame-csm-1b"
    
    try:
        print("📥 Cargando modelo...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        processor = AutoProcessor.from_pretrained(base_model_path)
        model = CsmForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✅ Modelo cargado: {torch.cuda.memory_allocated() / 1e9:.1f}GB VRAM")
        
        # Simple text-only conversation
        conversation = [
            {"role": "0", "content": [{"type": "text", "text": "Hello! How are you doing today?"}]}
        ]
        
        print("🔄 Procesando texto...")
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move to GPU
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        print(f"📊 Input shape: {inputs['input_ids'].shape}")
        
        print("🎵 Generando audio...")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                use_cache=True
            )
        
        end_time.record()
        torch.cuda.synchronize()
        generation_time = start_time.elapsed_time(end_time) / 1000.0
        
        print(f"✅ Generación: {outputs.shape} en {generation_time:.2f}s")
        
        # Try to save audio
        output_path = "/workspace/runPodtts/outputs/csm_simple_test.wav"
        
        try:
            print("💾 Guardando con processor.save_audio...")
            processor.save_audio(outputs, output_path, sample_rate=24000)
            
            if Path(output_path).exists():
                file_size = Path(output_path).stat().st_size
                print(f"✅ Audio guardado: {output_path} ({file_size} bytes)")
                
                # Verify
                audio_data, sr = torchaudio.load(output_path)
                duration = audio_data.shape[1] / sr
                print(f"🎵 Verificación: {duration:.2f}s a {sr}Hz")
                
                return True
            else:
                print("❌ No se creó el archivo")
                return False
                
        except Exception as save_error:
            print(f"❌ Error guardando: {save_error}")
            
            # Try alternative: save only new tokens
            print("🔄 Probando solo tokens nuevos...")
            try:
                input_length = inputs['input_ids'].shape[1]
                audio_tokens = outputs[:, input_length:]
                print(f"📊 Audio tokens: {audio_tokens.shape}")
                
                alt_path = "/workspace/runPodtts/outputs/csm_simple_alternative.wav"
                processor.save_audio(audio_tokens, alt_path, sample_rate=24000)
                
                if Path(alt_path).exists():
                    file_size = Path(alt_path).stat().st_size
                    print(f"✅ Audio alternativo: {alt_path} ({file_size} bytes)")
                    return True
                else:
                    print("❌ Alternativo falló")
                    return False
                    
            except Exception as alt_error:
                print(f"❌ Error alternativo: {alt_error}")
                return False
    
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🎯 CSM SIMPLE TEST")
    print("Solo texto, sin contexto de audio")
    print("=" * 40)
    
    success = test_csm_text_only()
    
    if success:
        print("\n🎉 ¡ÉXITO!")
        print("✅ CSM base funciona con texto simple")
        print("✅ Audio generado correctamente")
    else:
        print("\n❌ FALLÓ")
        print("❌ Problemas con CSM base")
    
    torch.cuda.empty_cache()
    return success

if __name__ == "__main__":
    main() 