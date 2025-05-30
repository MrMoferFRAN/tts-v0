#!/usr/bin/env python3
"""
Test CSM base model using official API
Based on the suggested approach
"""
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, CsmForConditionalGeneration

# Set environment
os.environ["NO_TORCH_COMPILE"] = "1"

def create_dummy_audio_files():
    """Create dummy audio files for testing"""
    print("🎵 Creando archivos de audio de prueba...")
    
    output_dir = Path("/workspace/runPodtts/outputs/dummy_audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create simple sine wave audio files
    sample_rate = 24000
    duration = 2.0  # 2 seconds
    
    audio_files = []
    
    for i in range(4):
        # Generate different frequency sine waves
        freq = 440 + i * 110  # A4, B4, C#5, D#5
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * torch.sin(2 * torch.pi * freq * t)
        
        # Add some envelope to make it more natural
        envelope = torch.exp(-t * 2)  # Exponential decay
        audio = audio * envelope
        
        audio_path = output_dir / f"utterance_{i}.wav"
        torchaudio.save(str(audio_path), audio.unsqueeze(0), sample_rate)
        audio_files.append(str(audio_path))
        
        print(f"   📁 {audio_path}")
    
    return audio_files

def load_csm_base():
    """Load base CSM model"""
    print("🔥 Cargando modelo CSM base...")
    
    base_model_path = "/workspace/runPodtts/models/sesame-csm-1b"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        processor = AutoProcessor.from_pretrained(base_model_path)
        model = CsmForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✅ Modelo cargado: {torch.cuda.memory_allocated() / 1e9:.1f}GB VRAM")
        return model, processor, tokenizer
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

def load_audio(audio_path, target_sample_rate=24000):
    """Load and resample audio"""
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def test_csm_base_api():
    """Test CSM using base API approach"""
    print("🧪 TESTING CSM BASE API")
    print("=" * 50)
    
    # Load model
    model, processor, tokenizer = load_csm_base()
    if model is None:
        return False
    
    # Create dummy audio files
    audio_paths = create_dummy_audio_files()
    
    # Define conversation
    speakers = [0, 1, 0, 0]
    transcripts = [
        "Hey how are you doing.",
        "Pretty good, pretty good.", 
        "I'm great.",
        "So happy to be speaking to you.",
    ]
    
    print(f"\n📝 Preparando conversación con {len(transcripts)} utterances...")
    
    # Load audio context
    context_audios = []
    for audio_path in audio_paths:
        audio = load_audio(audio_path, target_sample_rate=24000)
        context_audios.append(audio)
        print(f"   🎵 {audio_path}: {audio.shape} ({audio.shape[0]/24000:.2f}s)")
    
    # Prepare conversation context
    conversation = []
    for i, (transcript, speaker) in enumerate(zip(transcripts, speakers)):
        conversation.append({
            "role": str(speaker),
            "content": [
                {"type": "text", "text": transcript},
                {"type": "audio", "audio": audio_paths[i]}  # Use file path instead of tensor
            ]
        })
    
    print(f"\n🎯 Generando respuesta...")
    target_text = "Me too, this is some cool stuff huh?"
    target_speaker = 1
    
    # Add target response to conversation
    conversation.append({
        "role": str(target_speaker),
        "content": [{"type": "text", "text": target_text}]
    })
    
    try:
        # Process with CSM
        print("🔄 Aplicando chat template...")
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move to GPU
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        print(f"📊 Input shape: {inputs['input_ids'].shape}")
        
        # Generate
        print("🎵 Generando audio...")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                use_cache=True
            )
        
        end_time.record()
        torch.cuda.synchronize()
        generation_time = start_time.elapsed_time(end_time) / 1000.0
        
        print(f"✅ Generación completa: {outputs.shape} en {generation_time:.2f}s")
        
        # Try to save audio using processor
        output_path = "/workspace/runPodtts/outputs/csm_base_test.wav"
        
        try:
            print("💾 Guardando audio con processor.save_audio...")
            processor.save_audio(outputs, output_path, sample_rate=24000)
            
            if Path(output_path).exists():
                file_size = Path(output_path).stat().st_size
                print(f"✅ Audio guardado: {output_path} ({file_size} bytes)")
                
                # Load and check
                audio_check, sr = torchaudio.load(output_path)
                duration = audio_check.shape[1] / sr
                print(f"🎵 Verificación: {duration:.2f}s audio a {sr}Hz")
                return True
            else:
                print("❌ Archivo no fue creado")
                return False
                
        except Exception as save_error:
            print(f"❌ Error guardando: {save_error}")
            
            # Try alternative approach
            print("🔄 Intentando método alternativo...")
            try:
                # Extract only new tokens (audio tokens)
                input_length = inputs['input_ids'].shape[1]
                audio_tokens = outputs[:, input_length:]
                
                print(f"📊 Audio tokens: {audio_tokens.shape}")
                
                # Save with alternative approach
                alt_output_path = "/workspace/runPodtts/outputs/csm_base_alternative.wav"
                processor.save_audio(audio_tokens, alt_output_path, sample_rate=24000)
                
                if Path(alt_output_path).exists():
                    file_size = Path(alt_output_path).stat().st_size
                    print(f"✅ Audio alternativo guardado: {alt_output_path} ({file_size} bytes)")
                    return True
                else:
                    print("❌ Método alternativo también falló")
                    return False
                    
            except Exception as alt_error:
                print(f"❌ Error método alternativo: {alt_error}")
                return False
    
    except Exception as e:
        print(f"❌ Error en generación: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 CSM BASE MODEL TEST")
    print("=" * 60)
    print("🎯 Objetivo: Probar modelo CSM base sin adaptadores")
    print("📝 Usando API oficial sugerida")
    
    success = test_csm_base_api()
    
    if success:
        print("\n🎉 ¡ÉXITO!")
        print("✅ Modelo base CSM funcionando")
        print("✅ Audio generado correctamente")
        print("📁 Archivos en: /workspace/runPodtts/outputs/")
    else:
        print("\n❌ FALLÓ")
        print("❌ Modelo base tiene problemas")
    
    # Cleanup
    torch.cuda.empty_cache()
    
    return success

if __name__ == "__main__":
    main() 