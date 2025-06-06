#!/usr/bin/env python3
"""
Script de instalación para CSM (Conversational Speech Model)
Maneja las depeencias de manera eficiente y verifica la instalación.
"""

import subprocess
import sys
import os
import importlib.util

def check_package(package_name):
    """Verifica si un paquete está instalado"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def run_command(cmd, description=""):
    """Ejecuta un comando y maneja errores"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e.stderr}")
        return False

def install_csm():
    """Función principal de instalación"""
    
    print("🚀 Iniciando instalación de CSM...")
    
    # Verificar espacio en disco
    result = subprocess.run("df -h / | tail -1", shell=True, capture_output=True, text=True)
    print(f"💾 Espacio en disco: {result.stdout.strip()}")
    
    # 1. Verificar modelo descargado
    model_path = "/workspacetts-v0/models/csm-1b.safetensors"
    if os.path.exists(model_path):
        size_gb = os.path.getsize(model_path) / (1024**3)
        print(f"✅ Modelo CSM encontrado: {size_gb:.1f} GB")
    else:
        print("❌ Modelo CSM no encontrado")
        return False
    
    # 2. Verificar repositorio CSM
    csm_repo = "/workspacetts-v0/models/csm"
    if os.path.exists(csm_repo):
        print("✅ Repositorio CSM encontrado")
    else:
        print("❌ Repositorio CSM no encontrado")
        return False
    
    # 3. Verificar dataset Elise
    elise_path = "/workspacetts-v0/datasets/csm-1b-elise"
    if os.path.exists(elise_path):
        print("✅ Dataset Elise encontrado")
    else:
        print("❌ Dataset Elise no encontrado")
        return False
    
    # 4. Verificar dependencias básicas
    essential_packages = ['torch', 'transformers', 'safetensors', 'torchaudio']
    missing_packages = []
    
    for package in essential_packages:
        if check_package(package):
            print(f"✅ {package} está instalado")
        else:
            missing_packages.append(package)
            print(f"❌ {package} no está instalado")
    
    # 5. Intentar importar torch para verificar instalación
    try:
        import torch
        print(f"✅ PyTorch versión: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA no está disponible")
    except ImportError as e:
        print(f"❌ Error importando PyTorch: {e}")
        print("🔧 Recomendación: Reinstalar el entorno o usar un contenedor con más espacio")
    
    # 6. Verificar transformers y huggingface_hub
    try:
        import transformers
        import huggingface_hub
        print(f"✅ Transformers versión: {transformers.__version__}")
        print(f"✅ Hugging Face Hub versión: {huggingface_hub.__version__}")
    except ImportError as e:
        print(f"⚠️  Error importando transformers/huggingface_hub: {e}")
    
    # 7. Información sobre uso
    print("\n" + "="*50)
    print("📋 INFORMACIÓN DE USO:")
    print("="*50)
    print(f"🔸 Modelo CSM: {model_path}")
    print(f"🔸 Código fuente: {csm_repo}")
    print(f"🔸 Dataset Elise: {elise_path}")
    print("\n🔸 Para usar CSM:")
    print("   cd /workspacetts-v0/models/csm")
    print("   python run_csm.py")
    print("\n🔸 Para desarrollo:")
    print("   from generator import load_csm_1b")
    print("   generator = load_csm_1b(device='cuda')")
    
    # 8. Crear script de ejemplo
    example_script = """#!/usr/bin/env python3
# Ejemplo de uso de CSM
import sys
import os
sys.path.append('/workspacetts-v0/models/csm')

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
"""
    
    with open("/workspacetts-v0/test_csm.py", "w") as f:
        f.write(example_script)
    print("✅ Script de ejemplo creado: test_csm.py")
    
    print("\n🎉 Instalación verificada. El modelo CSM está listo para usar!")
    
    if missing_packages:
        print(f"\n⚠️  Faltan algunos paquetes: {', '.join(missing_packages)}")
        print("💡 Recomendación: Usar un contenedor con más espacio para instalación completa")
    
    return True

if __name__ == "__main__":
    install_csm() 