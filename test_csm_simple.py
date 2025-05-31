#!/usr/bin/env python3
"""
Simple CSM test without problematic dependencies
"""
import sys
import os

def test_basic_imports():
    """Test basic imports first"""
    print("🔍 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except Exception as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import torchaudio
        print(f"✅ TorchAudio: {torchaudio.__version__}")
    except Exception as e:
        print(f"❌ TorchAudio import failed: {e}")
        return False
    
    return True

def test_model_files():
    """Test model files exist"""
    print("\n🔍 Testing model files...")
    
    from pathlib import Path
    
    model_path = Path("./models/sesame-csm-1b")
    if not model_path.exists():
        print(f"❌ Model directory not found: {model_path}")
        return False
    
    print(f"✅ Model directory found: {model_path}")
    
    model_file = model_path / "model.safetensors"
    if not model_file.exists():
        print(f"❌ Model file not found: {model_file}")
        return False
    
    size_gb = model_file.stat().st_size / (1024**3)
    print(f"✅ Model file found: {size_gb:.1f} GB")
    
    config_file = model_path / "config.json"
    if config_file.exists():
        print(f"✅ Config file found")
    else:
        print(f"⚠️ Config file not found")
    
    return True

def test_csm_import():
    """Test CSM specific imports"""
    print("\n🔍 Testing CSM imports...")
    
    try:
        # Try importing without the problematic voice_cloning
        os.environ['PYTHONPATH'] = '/workspace/runttspod'
        
        from transformers import AutoProcessor
        print("✅ AutoProcessor imported")
        
        from transformers import CsmForConditionalGeneration
        print("✅ CsmForConditionalGeneration imported")
        
        return True
        
    except Exception as e:
        print(f"❌ CSM imports failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test actual model loading"""
    print("\n🔍 Testing model loading...")
    
    try:
        import torch
        from transformers import AutoProcessor, CsmForConditionalGeneration
        
        model_id = "./models/sesame-csm-1b"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"📥 Loading from: {model_id}")
        print(f"🖥️ Device: {device}")
        
        # Load processor first
        print("📥 Loading processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        print("✅ Processor loaded")
        
        # Load model
        print("📥 Loading model...")
        model = CsmForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        print("✅ Model loaded successfully")
        
        # Test basic generation
        print("🎵 Testing basic generation...")
        text = "[0]Hello from CSM."
        inputs = processor(text, add_special_tokens=True).to(device)
        
        print("🔄 Generating...")
        with torch.no_grad():
            audio = model.generate(**inputs, output_audio=True, max_new_tokens=256)
        
        print("✅ Generation successful")
        
        # Save audio
        output_path = "test_output.wav"
        processor.save_audio(audio, output_path)
        print(f"✅ Audio saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading/generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🎤 CSM-1B SYSTEM TEST")
    print("=" * 50)
    
    success = True
    
    # Test 1: Basic imports
    if not test_basic_imports():
        success = False
    
    # Test 2: Model files
    if not test_model_files():
        success = False
    
    # Test 3: CSM imports
    if not test_csm_import():
        success = False
        
    # Test 4: Model loading and generation
    if success and not test_model_loading():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("🚀 CSM-1B is ready for use!")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Please check the errors above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 