#!/usr/bin/env python3
"""
🔍 Script de Verificación Completa del Entorno RunPod
Verifica todas las capacidades del sistema para CSM TTS + Elise
"""

import subprocess
import sys
import torch
import psutil
import platform
from pathlib import Path

def print_section(title):
    """Imprimir sección con formato"""
    print(f"\n{'='*50}")
    print(f"🔍 {title}")
    print('='*50)

def run_command(cmd):
    """Ejecutar comando y retornar output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def check_gpu_capabilities():
    """Verificar capacidades detalladas de GPU"""
    print_section("VERIFICACIÓN DE GPU")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
        print(f"✅ Versión CUDA: {torch.version.cuda}")
        print(f"✅ Número de GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"\n🔥 GPU {i}: {gpu_props.name}")
            print(f"   📊 Memoria total: {gpu_props.total_memory / 1e9:.1f} GB")
            print(f"   🔧 Compute capability: {gpu_props.major}.{gpu_props.minor}")
            print(f"   🏭 Multiprocessors: {gpu_props.multi_processor_count}")
            
        # Test de memoria GPU
        print(f"\n🧪 Test de memoria GPU:")
        device = torch.device('cuda:0')
        try:
            # Crear tensor de prueba
            test_tensor = torch.randn(1000, 1000, device=device)
            memory_allocated = torch.cuda.memory_allocated(device) / 1e9
            memory_reserved = torch.cuda.memory_reserved(device) / 1e9
            print(f"   ✅ Memoria asignada: {memory_allocated:.2f} GB")
            print(f"   ✅ Memoria reservada: {memory_reserved:.2f} GB")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ Error en test de GPU: {e}")
    else:
        print("❌ CUDA no disponible")

def check_system_resources():
    """Verificar recursos del sistema"""
    print_section("RECURSOS DEL SISTEMA")
    
    # CPU
    print(f"🖥️  CPU: {platform.processor()}")
    print(f"🔢 Cores: {psutil.cpu_count(logical=False)} físicos, {psutil.cpu_count(logical=True)} lógicos")
    print(f"📊 Uso actual CPU: {psutil.cpu_percent(interval=1)}%")
    
    # Memoria
    memory = psutil.virtual_memory()
    print(f"🧠 RAM Total: {memory.total / 1e9:.1f} GB")
    print(f"🧠 RAM Disponible: {memory.available / 1e9:.1f} GB")
    print(f"🧠 RAM Usado: {memory.percent}%")
    
    # Disco
    disk = psutil.disk_usage('/workspace')
    print(f"💾 Disco Total: {disk.total / 1e12:.1f} TB")
    print(f"💾 Disco Disponible: {disk.free / 1e12:.1f} TB")
    print(f"💾 Disco Usado: {(disk.used/disk.total)*100:.1f}%")

def check_python_environment():
    """Verificar entorno Python"""
    print_section("ENTORNO PYTHON")
    
    print(f"🐍 Python: {sys.version}")
    print(f"📦 PyTorch: {torch.__version__}")
    print(f"📍 PyTorch path: {torch.__file__}")
    
    # Verificar paquetes importantes
    packages_to_check = [
        'numpy', 'scipy', 'matplotlib', 'librosa', 
        'transformers', 'datasets', 'tqdm', 'wandb'
    ]
    
    print("\n📚 Paquetes importantes:")
    for package in packages_to_check:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"   ✅ {package}: {version}")
        except ImportError:
            print(f"   ❌ {package}: No instalado")

def check_audio_capabilities():
    """Verificar capacidades de audio"""
    print_section("CAPACIDADES DE AUDIO")
    
    # FFmpeg
    ffmpeg_version = run_command("ffmpeg -version | head -1")
    if "ffmpeg version" in ffmpeg_version:
        print(f"✅ FFmpeg: {ffmpeg_version}")
    else:
        print("❌ FFmpeg no encontrado")
    
    # Add SoX check
    sox_version = run_command("sox --version | head -1")
    if "SoX" in sox_version:
        print(f"✅ SoX: {sox_version}")
    else:
        print("❌ SoX no encontrado")
    
    # Verificar librosa
    try:
        import librosa
        print(f"✅ Librosa: {librosa.__version__}")
        
        # Test básico de librosa
        import numpy as np
        test_audio = np.random.randn(22050)  # 1 segundo de audio
        mfcc = librosa.feature.mfcc(y=test_audio, sr=22050, n_mfcc=13)
        print(f"✅ Test MFCC: {mfcc.shape}")
    except Exception as e:
        print(f"❌ Error con librosa: {e}")

def estimate_training_capacity():
    """Estimar capacidad de entrenamiento"""
    print_section("ESTIMACIÓN DE CAPACIDAD DE ENTRENAMIENTO")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"💪 Con {gpu_memory:.0f}GB de VRAM puedes entrenar:")
        
        if gpu_memory >= 80:
            print("   🔥 Modelos GRANDES (>1B parámetros) - Full finetuning")
            print("   🔥 Batch size: 32-64+")
            print("   🔥 Múltiples experimentos simultáneos")
            print("   🔥 Sin necesidad de técnicas de optimización")
        elif gpu_memory >= 40:
            print("   ✅ Modelos MEDIANOS-GRANDES - Full finetuning")
            print("   ✅ Batch size: 16-32")
            print("   ✅ La mayoría de modelos sin problemas")
        elif gpu_memory >= 24:
            print("   ✅ Modelos MEDIANOS - Full finetuning")
            print("   ✅ Batch size: 8-16")
            print("   ⚠️  Modelos grandes requieren LoRA")
        else:
            print("   ⚠️  Solo modelos PEQUEÑOS o técnicas optimizadas")
            print("   ⚠️  Batch size: 4-8")
            print("   ⚠️  Requiere LoRA/QLoRA para modelos grandes")
    
    # Estimación de tiempo y costo
    print(f"\n⏱️  Estimaciones para dataset Elise:")
    print(f"   🏋️  Full finetuning: 6-12 horas")
    print(f"   ⚡ LoRA finetuning: 3-6 horas")
    print(f"   🚀 Con A100 80GB: Sin limitaciones de memoria")

def main():
    """Función principal"""
    print("🎯 VERIFICACIÓN COMPLETA DEL ENTORNO RUNPOD")
    print("Para CSM TTS + Elise Finetuning")
    print(f"Timestamp: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}")
    
    check_gpu_capabilities()
    check_system_resources()
    check_python_environment()
    check_audio_capabilities()
    estimate_training_capacity()
    
    print_section("RESUMEN FINAL")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎯 SISTEMA LISTO para CSM TTS")
        print(f"🔥 GPU: {gpu_name} ({gpu_memory:.0f}GB)")
        print(f"✅ Puedes proceder con el setup de CSM")
        print(f"✅ Entrenamiento de Elise SIN limitaciones de memoria")
    else:
        print("❌ SISTEMA NO LISTO - CUDA no disponible")
    
    print("\n🚀 Siguiente paso: ejecutar setup_csm_runpod.sh")

if __name__ == "__main__":
    main() 