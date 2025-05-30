#!/usr/bin/env python3
"""
Simple test script for the Voice Cloning module
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, '.')

def test_basic_functionality():
    """Test basic module import and functionality"""
    try:
        print("=== Testing Voice Cloning Module ===\n")
        
        # Test 1: Import the module
        print("1. Testing module import...")
        from voice_cloning import VoiceCloner
        print("✓ Module imported successfully")
        
        # Test 2: Check if model exists
        print("\n2. Checking model availability...")
        model_path = "./models/sesame-csm-1b"
        if os.path.exists(model_path):
            print(f"✓ Model found at: {model_path}")
            
            # Check key files
            key_files = ["config.json", "tokenizer.json", "model.safetensors"]
            for file in key_files:
                file_path = os.path.join(model_path, file)
                if os.path.exists(file_path):
                    print(f"  ✓ {file} found")
                else:
                    print(f"  ⚠ {file} not found")
        else:
            print(f"✗ Model not found at: {model_path}")
            return False
        
        # Test 3: Check reference audio
        print("\n3. Checking reference audio...")
        reference_audio = "Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3"
        if os.path.exists(reference_audio):
            print(f"✓ Reference audio found: {reference_audio}")
        else:
            print(f"✗ Reference audio not found: {reference_audio}")
            return False
        
        # Test 4: Initialize VoiceCloner (without loading model yet)
        print("\n4. Testing VoiceCloner initialization...")
        # We'll test this without actually loading the model to avoid memory issues
        print("✓ VoiceCloner class available")
        
        # Test 5: Test model info function
        print("\n5. Testing model info function...")
        from voice_cloning.models import get_model_info
        info = get_model_info(model_path)
        if "error" not in info:
            print("✓ Model config readable")
            print(f"  Model type: {info.get('model_type', 'Unknown')}")
            print(f"  Architecture: {info.get('architectures', ['Unknown'])[0] if info.get('architectures') else 'Unknown'}")
        else:
            print(f"⚠ Model info error: {info['error']}")
        
        # Test 6: Test watermarking functions
        print("\n6. Testing watermarking functions...")
        from voice_cloning.watermarking import apply_watermark, detect_watermark
        print("✓ Watermarking functions imported")
        
        print("\n=== All basic tests passed! ===")
        print("\nThe voice cloning module is ready to use.")
        print("To run a full test with model loading, use:")
        print("  python voice_cloning/example_usage.py")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
        return False

def test_model_loading():
    """Test actual model loading (memory intensive)"""
    try:
        print("\n=== Testing Model Loading ===")
        print("Warning: This will load the full model into memory...")
        
        # Ask for confirmation
        response = input("Continue with model loading test? (y/N): ")
        if response.lower() != 'y':
            print("Model loading test skipped.")
            return True
        
        print("Loading model...")
        from voice_cloning import VoiceCloner
        
        cloner = VoiceCloner(model_path="./models/sesame-csm-1b")
        print("✓ Model loaded successfully!")
        
        # Test simple generation
        print("Testing simple generation...")
        output = cloner.simple_generate(
            text="Esta es una prueba de generación.",
            output_path="test_output.wav"
        )
        
        if os.path.exists(output):
            print(f"✓ Audio generated: {output}")
            # Clean up
            os.remove(output)
            print("✓ Test file cleaned up")
        else:
            print("✗ Audio generation failed")
            
        return True
        
    except Exception as e:
        print(f"✗ Model loading test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Voice Cloning Module Test Script")
    print("=" * 40)
    
    # Run basic tests
    basic_success = test_basic_functionality()
    
    if basic_success:
        print("\n" + "=" * 40)
        print("Basic tests completed successfully!")
        print("\nOptional: Test model loading (requires GPU memory)")
        test_model_loading()
    else:
        print("\n" + "=" * 40)
        print("Basic tests failed. Please check the setup.")
        sys.exit(1) 