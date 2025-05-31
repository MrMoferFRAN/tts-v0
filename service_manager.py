import subprocess
import time
import psutil
import signal
import sys

SERVICES = [
    {
        "name": "CSM-TTS API", 
        "command": "PYTHONPATH=/workspace/runPodtts python -m csm_tts.api"
    },
    {
        "name": "Voice Cloning", 
        "command": "PYTHONPATH=/workspace/runPodtts python -m voice_cloning.service"
    },
    # Add other services from READMEs here
]

def start_services():
    print("🚀 Starting all services...")
    processes = []
    for service in SERVICES:
        try:
            proc = subprocess.Popen(service["command"], shell=True)
            processes.append({"name": service["name"], "process": proc})
            print(f"✅ Started {service['name']} (PID: {proc.pid})")
        except Exception as e:
            print(f"❌ Failed to start {service['name']}: {str(e)}")
    return processes

def monitor_services(processes):
    print("\n🔍 Monitoring services... Press Ctrl+C to exit")
    try:
        while True:
            for proc_info in processes:
                if psutil.pid_exists(proc_info["process"].pid):
                    status = "RUNNING"
                else:
                    status = "STOPPED"
                print(f"{proc_info['name']}: {status}")
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        for proc_info in processes:
            if psutil.pid_exists(proc_info["process"].pid):
                proc_info["process"].terminate()
                print(f"🛑 Stopped {proc_info['name']}")

def signal_handler(sig, frame):
    print('\n🛑 Received shutdown signal!')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    running_services = start_services()
    monitor_services(running_services) 