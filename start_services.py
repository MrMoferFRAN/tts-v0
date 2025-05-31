#!/usr/bin/env python3
"""
Service Manager Optimizado para CSM TTS
"""
import subprocess
import time
import psutil
import signal
import sys
import os
from pathlib import Path

def check_gpu():
    """Verificar disponibilidad de GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU: {gpu_name}")
            return True
        else:
            print("⚠️ GPU no disponible, usando CPU")
            return False
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")
        return False

def start_voice_api():
    """Iniciar API de Voice Cloning"""
    print("🎤 Iniciando Voice Cloning API...")
    
    # Cambiar al directorio correcto
    os.chdir("/workspace/runttspod")
    
    # Verificar que quick_start.py existe
    if not Path("quick_start.py").exists():
        print("❌ quick_start.py no encontrado")
        return None
    
    try:
        # Configurar entorno
        env = os.environ.copy()
        env["PYTHONPATH"] = "/workspace/runttspod:/workspace/runttspod/models/csm"
        env["NO_TORCH_COMPILE"] = "1"
        
        # Iniciar proceso
        proc = subprocess.Popen(
            ["python", "quick_start.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        print(f"✅ Voice API iniciada (PID: {proc.pid})")
        print("🌐 Acceso: http://0.0.0.0:7860")
        print("📚 Docs: http://0.0.0.0:7860/docs")
        
        return proc
        
    except Exception as e:
        print(f"❌ Error iniciando Voice API: {e}")
        return None

def monitor_services(processes):
    """Monitorear servicios activos"""
    print("\n🔍 Monitoreando servicios... (Ctrl+C para salir)")
    print("=" * 50)
    
    try:
        while True:
            all_running = True
            for name, proc in processes.items():
                if proc and proc.poll() is None:
                    status = "🟢 RUNNING"
                else:
                    status = "🔴 STOPPED"
                    all_running = False
                
                print(f"{name}: {status}")
            
            if not all_running:
                print("\n⚠️ Algunos servicios se detuvieron")
                break
                
            print("-" * 30)
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo servicios...")
        for name, proc in processes.items():
            if proc and proc.poll() is None:
                proc.terminate()
                print(f"🛑 Detenido: {name}")

def main():
    """Función principal"""
    print("🚀 CSM TTS SERVICE MANAGER")
    print("=" * 40)
    
    # Verificar GPU
    check_gpu()
    
    # Iniciar servicios
    services = {}
    
    # Voice Cloning API
    voice_proc = start_voice_api()
    if voice_proc:
        services["Voice Cloning API"] = voice_proc
    
    if not services:
        print("❌ No se pudo iniciar ningún servicio")
        sys.exit(1)
    
    # Monitorear
    monitor_services(services)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    main()
