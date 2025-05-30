#!/usr/bin/env python3
"""
Investigate CSM processor.save_audio format requirements
"""
import os
import torch
import torchaudio
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, CsmForConditionalGeneration

os.environ["NO_TORCH_COMPILE"] = "1"

def investigate_save_audio_format():
    """Investigate what format save_audio expects"""
    print("🔍 INVESTIGANDO FORMATO DE save_audio")
    print("=" * 60)
    
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
        
        print(f"✅ Modelo cargado")
        
        # Check processor methods
        print("\n🔍 MÉTODOS DEL PROCESSOR:")
        for attr in dir(processor):
            if 'audio' in attr.lower() or 'save' in attr.lower():
                print(f"   🔸 {attr}")
        
        # Check save_audio method signature
        if hasattr(processor, 'save_audio'):
            print(f"\n📋 save_audio method: {processor.save_audio}")
            print(f"📋 save_audio __doc__: {processor.save_audio.__doc__}")
        
        # Generate some tokens first
        conversation = [
            {"role": "0", "content": [{"type": "text", "text": "Hello!"}]}
        ]
        
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print(f"\n🎵 Generando tokens para análisis...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        
        print(f"📊 Outputs shape: {outputs.shape}")
        
        # Analyze what save_audio expects
        input_length = inputs['input_ids'].shape[1]
        audio_tokens = outputs[:, input_length:]
        
        print(f"📊 Audio tokens shape: {audio_tokens.shape}")
        print(f"📊 Audio tokens dtype: {audio_tokens.dtype}")
        print(f"📊 Audio tokens device: {audio_tokens.device}")
        print(f"📊 Audio tokens min/max: {audio_tokens.min()}/{audio_tokens.max()}")
        
        # Try different formats
        print(f"\n🧪 PROBANDO DIFERENTES FORMATOS:")
        
        # Format 1: Original 3D tensor
        print(f"1️⃣ Formato original [1, seq, features]: {audio_tokens.shape}")
        try:
            test_path_1 = "/workspace/runPodtts/outputs/test_format_1.wav"
            processor.save_audio(audio_tokens, test_path_1)
            print(f"   ✅ Éxito: {test_path_1}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Format 2: Remove batch dimension
        print(f"2️⃣ Sin batch dimension [seq, features]: {audio_tokens.squeeze(0).shape}")
        try:
            test_path_2 = "/workspace/runPodtts/outputs/test_format_2.wav"
            processor.save_audio(audio_tokens.squeeze(0), test_path_2)
            print(f"   ✅ Éxito: {test_path_2}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Format 3: Flatten completely
        print(f"3️⃣ Completamente flat [total_samples]: {audio_tokens.flatten().shape}")
        try:
            test_path_3 = "/workspace/runPodtts/outputs/test_format_3.wav"
            processor.save_audio(audio_tokens.flatten(), test_path_3)
            print(f"   ✅ Éxito: {test_path_3}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Format 4: Try full output tensor
        print(f"4️⃣ Tensor completo [1, total_seq, features]: {outputs.shape}")
        try:
            test_path_4 = "/workspace/runPodtts/outputs/test_format_4.wav"
            processor.save_audio(outputs, test_path_4)
            print(f"   ✅ Éxito: {test_path_4}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Format 5: Convert to numpy
        print(f"5️⃣ Como numpy array: {audio_tokens.cpu().numpy().shape}")
        try:
            test_path_5 = "/workspace/runPodtts/outputs/test_format_5.wav"
            processor.save_audio(audio_tokens.cpu().numpy(), test_path_5)
            print(f"   ✅ Éxito: {test_path_5}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Format 6: Try different reshaping
        reshaped = audio_tokens.reshape(-1, audio_tokens.shape[-1])
        print(f"6️⃣ Reshape [total_seq, features]: {reshaped.shape}")
        try:
            test_path_6 = "/workspace/runPodtts/outputs/test_format_6.wav"
            processor.save_audio(reshaped, test_path_6)
            print(f"   ✅ Éxito: {test_path_6}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Check which files were created successfully
        print(f"\n📁 ARCHIVOS CREADOS EXITOSAMENTE:")
        output_dir = Path("/workspace/runPodtts/outputs")
        for i in range(1, 7):
            test_path = output_dir / f"test_format_{i}.wav"
            if test_path.exists():
                size = test_path.stat().st_size
                try:
                    audio, sr = torchaudio.load(test_path)
                    duration = audio.shape[1] / sr
                    print(f"   ✅ Format {i}: {test_path} ({size} bytes, {duration:.2f}s)")
                except Exception as load_error:
                    print(f"   ⚠️  Format {i}: {test_path} ({size} bytes, error loading: {load_error})")
            else:
                print(f"   ❌ Format {i}: No creado")
        
        return True
        
    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 INVESTIGACIÓN DE FORMATO save_audio")
    print("=" * 50)
    
    success = investigate_save_audio_format()
    
    if success:
        print("\n🎉 INVESTIGACIÓN COMPLETADA")
        print("🔍 Revisa los archivos test_format_*.wav")
        print("📊 Los que funcionaron nos dicen el formato correcto")
    else:
        print("\n❌ INVESTIGACIÓN FALLÓ")
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 