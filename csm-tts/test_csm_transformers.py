#!/usr/bin/env python3
"""
Test script para CSM usando la implementación nativa de Hugging Face Transformers
"""
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor
import os

# Configurar variable de entorno
os.environ["NO_TORCH_COMPILE"] = "1"

def test_csm_basic():
    """Test básico del modelo CSM con generación simple"""
    print("🎯 Iniciando test de CSM con Transformers...")
    
    model_id = "sesame/csm-1b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔥 Usando device: {device}")
    
    try:
        # Cargar modelo y procesador
        print("📥 Cargando modelo y procesador...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = CsmForConditionalGeneration.from_pretrained(
            model_id, 
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        print("✅ Modelo cargado exitosamente!")
        
        # Generar audio simple sin contexto
        print("🎭 Generando audio: 'Hello from Sesame CSM in Spanish: Hola desde Sesame!'")
        text = "[0]Hello from Sesame CSM in Spanish: Hola desde Sesame!"
        inputs = processor(text, add_special_tokens=True).to(device)
        
        # Generar audio
        with torch.no_grad():
            audio = model.generate(**inputs, output_audio=True, max_new_tokens=250)
        
        # Guardar audio
        output_path = "/workspace/runPodtts/outputs/test_basic_csm.wav"
        processor.save_audio(audio, output_path)
        print(f"🎵 Audio guardado en: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en test básico: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csm_conversation():
    """Test con conversación usando formato de chat"""
    print("\n🎭 Test de conversación...")
    
    model_id = "sesame/csm-1b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = CsmForConditionalGeneration.from_pretrained(
            model_id, 
            device_map=device,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Crear conversación de ejemplo
        conversation = [
            {"role": "0", "content": [{"type": "text", "text": "Hello! How are you today?"}]},
            {"role": "1", "content": [{"type": "text", "text": "I'm doing great, thank you!"}]},
            {"role": "0", "content": [{"type": "text", "text": "That's wonderful to hear!"}]},
        ]
        
        print("💬 Generando conversación...")
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(device)
        
        # Generar audio
        with torch.no_grad():
            audio = model.generate(**inputs, output_audio=True, max_new_tokens=300)
        
        # Guardar audio
        output_path = "/workspace/runPodtts/outputs/test_conversation_csm.wav"
        processor.save_audio(audio, output_path)
        print(f"🎵 Conversación guardada en: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en test de conversación: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 INICIANDO TESTS DE CSM TTS")
    print("=" * 50)
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.cuda.get_device_name()}")
        print(f"🔥 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA no disponible, usando CPU")
    
    # Crear directorio de salida
    os.makedirs("/workspace/runPodtts/outputs", exist_ok=True)
    
    # Ejecutar tests
    success_basic = test_csm_basic()
    success_conversation = test_csm_conversation()
    
    print("\n📊 RESUMEN DE TESTS:")
    print(f"Test básico: {'✅ EXITOSO' if success_basic else '❌ FALLIDO'}")
    print(f"Test conversación: {'✅ EXITOSO' if success_conversation else '❌ FALLIDO'}")
    
    if success_basic and success_conversation:
        print("\n🎉 ¡Todos los tests pasaron! CSM está funcionando correctamente.")
        return True
    else:
        print("\n⚠️  Algunos tests fallaron. Revisar los errores arriba.")
        return False

if __name__ == "__main__":
    main() 