#!/usr/bin/env python3
"""
RTX 5090 Compatibility Checker
Verifica si PyTorch es compatible con RTX 5090 y sugiere actualizaciones
"""

import torch
import subprocess
import sys

def check_rtx5090_compatibility():
    """Verifica compatibilidad RTX 5090 y sugiere soluciones"""
    
    print("🔍 RTX 5090 Compatibility Checker")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA no está disponible")
        return False
    
    try:
        device_props = torch.cuda.get_device_properties(0)
        gpu_name = device_props.name
        compute_capability = f"{device_props.major}.{device_props.minor}"
        
        print(f"🖥️ GPU Detectada: {gpu_name}")
        print(f"🔧 Compute Capability: {compute_capability}")
        
        # Check for RTX 5090
        if "RTX 5090" in gpu_name or device_props.major >= 12:
            print("🚨 RTX 5090 detectada!")
            
            # Check PyTorch version
            pytorch_version = torch.__version__
            print(f"🐍 PyTorch Version: {pytorch_version}")
            
            # RTX 5090 requires PyTorch 2.5+ with CUDA 12.4+
            major_version = int(pytorch_version.split('.')[0])
            minor_version = int(pytorch_version.split('.')[1])
            
            if major_version < 2 or (major_version == 2 and minor_version < 5):
                print("❌ PyTorch version incompatible with RTX 5090")
                print("⚠️ RTX 5090 requires PyTorch 2.5+ with CUDA 12.4+")
                print()
                print("🔧 SOLUCIONES RECOMENDADAS:")
                print("=" * 30)
                
                print("1. 🚀 Actualizar PyTorch (RECOMENDADO)")
                print("   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124")
                print()
                
                print("2. 💻 Usar modo CPU (TEMPORAL)")
                print("   export CUDA_VISIBLE_DEVICES=''")
                print("   python voice_api_complete.py")
                print()
                
                print("3. 🔄 Usar PyTorch Nightly (EXPERIMENTAL)")
                print("   pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
                print()
                
                print("4. 🐳 Usar Container con PyTorch actualizado")
                print("   docker run --gpus all -it pytorch/pytorch:2.5.0-devel-cuda12.4-cudnn9-runtime")
                print()
                
                return False
            else:
                print("✅ PyTorch version compatible!")
                return True
        else:
            print("✅ GPU compatible con PyTorch actual")
            return True
            
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")
        return False

def check_cuda_support():
    """Verifica el soporte CUDA disponible"""
    print("\n🔍 CUDA Support Check")
    print("=" * 30)
    
    try:
        # Get CUDA version
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            cuda_version = "Unknown"
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    cuda_version = line.split('release ')[1].split(',')[0]
                    break
            print(f"🔧 CUDA Version: {cuda_version}")
        else:
            print("⚠️ nvcc not found - CUDA toolkit may not be installed")
    except FileNotFoundError:
        print("⚠️ nvcc not found - CUDA toolkit not installed")
    
    # Check PyTorch CUDA support
    print(f"🐍 PyTorch CUDA Version: {torch.version.cuda}")
    
    # Check supported compute capabilities
    if hasattr(torch.cuda, 'get_arch_list'):
        archs = torch.cuda.get_arch_list()
        print(f"📊 Supported Compute Capabilities: {archs}")
        
        if 'sm_120' in archs or 'compute_120' in archs:
            print("✅ RTX 5090 (sm_120) support detected!")
        else:
            print("❌ RTX 5090 (sm_120) support NOT detected")

if __name__ == "__main__":
    compatible = check_rtx5090_compatibility()
    check_cuda_support()
    
    if not compatible:
        print("\n🚨 ACCIÓN REQUERIDA:")
        print("Tu RTX 5090 no es compatible con la versión actual de PyTorch.")
        print("Por favor, actualiza PyTorch o usa modo CPU como se indica arriba.")
        sys.exit(1)
    else:
        print("\n✅ Sistema compatible - puedes ejecutar voice_api_complete.py")
        sys.exit(0) 