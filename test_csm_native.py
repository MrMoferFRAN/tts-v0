#!/usr/bin/env python3
"""
Test script for CSM-1B using native Transformers API
"""
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor
import numpy as np
import torchaudio
from pathlib import Path

def test_csm_native():
    """Test CSM using native Transformers API"""
    print("🎤 Testing CSM-1B with native Transformers API")
    print("=" * 60)
    
    model_id = "./models/sesame-csm-1b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        print(f"🔍 Loading model from: {model_id}")
        print(f"🖥️ Device: {device}")
        
        # Load the model and processor
        print("📥 Loading processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        
        print("📥 Loading model...")
        model = CsmForConditionalGeneration.from_pretrained(
            model_id, 
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        print("✅ Model and processor loaded successfully")
        
        # Test basic generation
        print("\n🎵 Testing basic generation...")
        text = "[0]Hello from CSM. This is a test of voice cloning."
        inputs = processor(text, add_special_tokens=True).to(device)
        
        print(f"📝 Text: {text}")
        print("🔄 Generating audio...")
        
        with torch.no_grad():
            audio = model.generate(**inputs, output_audio=True, max_new_tokens=512)
        
        # Save the audio
        output_path = "output_basic_test.wav"
        processor.save_audio(audio, output_path)
        
        print(f"✅ Audio generated and saved to: {output_path}")
        
        # Test with context (voice cloning)
        print("\n🎭 Testing voice cloning with context...")
        
        # Check if we have reference audio
        reference_audio = "voices/Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3"
        if Path(reference_audio).exists():
            print(f"📄 Found reference audio: {reference_audio}")
            
            # Load reference audio
            reference_waveform, sr = torchaudio.load(reference_audio)
            
            # Resample to 24kHz if needed
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                reference_waveform = resampler(reference_waveform)
            
            # Convert to mono if stereo
            if reference_waveform.shape[0] > 1:
                reference_waveform = reference_waveform.mean(dim=0, keepdim=True)
            
            # Create conversation with context
            conversation = [
                {
                    "role": "0",
                    "content": [
                        {"type": "text", "text": "Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo."},
                        {"type": "audio", "path": reference_waveform.squeeze().numpy()}
                    ]
                },
                {
                    "role": "0",
                    "content": [{"type": "text", "text": "Hola, esto es una prueba de clonación de voz con CSM."}]
                }
            ]
            
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
            ).to(device)
            
            print("🔄 Generating cloned voice...")
            with torch.no_grad():
                cloned_audio = model.generate(**inputs, output_audio=True, max_new_tokens=512)
            
            output_path_cloned = "output_cloned_test.wav"
            processor.save_audio(cloned_audio, output_path_cloned)
            
            print(f"✅ Cloned audio generated and saved to: {output_path_cloned}")
        else:
            print(f"⚠️ Reference audio not found: {reference_audio}")
            print("💡 Skipping voice cloning test")
        
        print("\n🎉 All tests completed successfully!")
        print("=" * 60)
        print("🔧 Model Information:")
        print(f"   • Model ID: {model_id}")
        print(f"   • Device: {device}")
        print(f"   • Model type: {type(model).__name__}")
        print(f"   • Processor type: {type(processor).__name__}")
        if torch.cuda.is_available():
            print(f"   • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"   • GPU Usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_csm_native()
    if success:
        print("\n🚀 CSM-1B is ready for use!")
    else:
        print("\n❌ CSM-1B test failed")
        exit(1) 