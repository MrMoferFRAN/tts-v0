#!/usr/bin/env python3
# Ejemplo de uso de CSM
import sys
import os
sys.path.append('/workspace/runttspod/models/csm')

try:
    from generator import load_csm_1b
    import torchaudio
    
    print("🔄 Cargando modelo CSM...")
    generator = load_csm_1b(device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("🎤 Generando audio...")
    audio = generator.generate(
        text="Hola, soy CSM, un modelo de síntesis de voz conversacional.",
        speaker=0,
        context=[],
        max_audio_length_ms=10_000,
    )
    
    print("💾 Guardando audio...")
    torchaudio.save("ejemplo_csm.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
    print("✅ Audio generado: ejemplo_csm.wav")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("🔧 Verifica que todas las dependencias estén instaladas correctamente")
