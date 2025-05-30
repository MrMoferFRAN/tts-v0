"""
Example usage of the Voice Cloning module with CSM-1B
"""

import os
from voice_cloning import VoiceCloner

def main():
    """
    Main example function demonstrating voice cloning
    """
    print("=== CSM-1B Voice Cloning Example ===")
    
    # Reference audio and transcript
    reference_audio = "Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3"
    reference_transcript = "Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo."
    
    # Text to synthesize with the cloned voice
    target_texts = [
        "Hola, ¿cómo estás hoy?",
        "Me alegra mucho poder ayudarte con este proyecto.",
        "La tecnología de clonación de voz está avanzando rápidamente.",
        "Espero que estos resultados sean satisfactorios para ti."
    ]
    
    # Check if reference audio exists
    if not os.path.exists(reference_audio):
        print(f"Error: Reference audio file not found: {reference_audio}")
        print("Please make sure the audio file is in the current directory.")
        return
    
    try:
        # Initialize the voice cloner
        print("Initializing Voice Cloner...")
        cloner = VoiceCloner(
            model_path="./models/sesame-csm-1b",
            max_length=2048
        )
        
        # Create outputs directory
        os.makedirs("outputs", exist_ok=True)
        
        # Test simple generation without context first
        print(f"\n=== Simple TTS Test ===")
        simple_output = cloner.simple_generate(
            text="Hola, esta es una prueba simple de síntesis de voz.",
            output_path="outputs/simple_tts.wav"
        )
        print(f"Simple TTS saved to: {simple_output}")
        
        # Generate single voice clone with context
        print(f"\n=== Single Voice Clone with Context ===")
        output_path = cloner.clone_voice_from_file(
            reference_audio=reference_audio,
            reference_transcript=reference_transcript,
            target_text=target_texts[0],
            output_path="outputs/single_clone.wav"
        )
        print(f"Single clone saved to: {output_path}")
        
        # Generate batch voice clones
        print(f"\n=== Batch Voice Clone ===")
        output_paths = cloner.batch_generate(
            text_list=target_texts,
            context_text=reference_transcript,
            context_audio_path=reference_audio,
            output_dir="outputs/batch"
        )
        
        print(f"Batch clones generated:")
        for i, path in enumerate(output_paths):
            print(f"  {i+1}. {path}")
        
        # Test individual generation with different parameters
        print(f"\n=== Custom Parameters Test ===")
        custom_output = cloner.generate_speech(
            context_text=reference_transcript,
            target_text="Esta es una prueba con parámetros personalizados.",
            context_audio_path=reference_audio,
            output_path="outputs/custom_params.wav",
            temperature=0.8
        )
        print(f"Custom generation saved to: {custom_output}")
        
        print(f"\n=== Voice Cloning Complete ===")
        print(f"All generated files are in the 'outputs' directory.")
        
    except Exception as e:
        print(f"Error during voice cloning: {str(e)}")
        print("Please check that:")
        print("1. The model is properly downloaded in ./models/sesame-csm-1b")
        print("2. You have sufficient GPU memory")
        print("3. All required dependencies are installed")
        print("4. The CSM model files are complete and valid")

def test_watermarking():
    """
    Test watermarking functionality
    """
    print("\n=== Testing Watermarking ===")
    
    from voice_cloning.watermarking import apply_watermark, detect_watermark
    
    # Test with a generated file (if it exists)
    test_file = "outputs/single_clone.wav"
    if os.path.exists(test_file):
        # Apply watermark
        watermarked_file = "outputs/watermarked_example.wav"
        apply_watermark(
            audio_path=test_file,
            output_path=watermarked_file,
            watermark_text="CSM Voice Clone - Example",
            method="metadata"
        )
        
        # Detect watermark
        detection_result = detect_watermark(
            audio_path=watermarked_file,
            expected_watermark="CSM Voice Clone - Example"
        )
        print(f"Watermark detection result: {detection_result}")
    else:
        print("No generated file found for watermarking test.")

def test_generator_compatibility():
    """
    Test compatibility with the original repository structure
    """
    print("\n=== Testing Generator Compatibility ===")
    
    try:
        from voice_cloning.generator import VoiceGenerator
        
        generator = VoiceGenerator()
        
        result = generator.generate(
            context_audio_path="Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3",
            context_text="Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo.",
            text="Prueba de compatibilidad con el generador original.",
            output_filename="outputs/generator_test.wav"
        )
        
        print(f"Generator compatibility test completed: {result}")
        
    except Exception as e:
        print(f"Generator compatibility test failed: {str(e)}")

if __name__ == "__main__":
    main()
    test_watermarking()
    test_generator_compatibility() 