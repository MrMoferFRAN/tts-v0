#!/usr/bin/env python3
"""
Script de prueba para comparar rendimiento entre modelo normal y turbo
"""

import time
import requests
import json

# Configuración
API_BASE = "http://localhost:7860"
TEST_TEXT = "Hola, soy una voz clonada usando inteligencia artificial."

def test_health():
    """Verificar estado de la API y modelos"""
    print("🔍 Verificando estado de la API...")
    
    response = requests.get(f"{API_BASE}/health")
    if response.status_code == 200:
        data = response.json()
        print("✅ API funcionando correctamente")
        print(f"📊 Modelo normal: {'✅ Cargado' if data['normal_model']['loaded'] else '❌ No cargado'}")
        print(f"🚀 Modelo turbo: {'✅ Disponible' if data['turbo_model']['available'] else '❌ No disponible'}")
        print(f"🖥️ GPU: {'✅ Disponible' if data['gpu_available'] else '❌ No disponible'}")
        
        if data['gpu_available'] and 'gpu_info' in data and data['gpu_info']:
            gpu = data['gpu_info']
            print(f"🖥️ GPU Info: {gpu.get('name', 'Unknown')} ({gpu.get('memory_gb', 0):.1f} GB)")
        
        return data
    else:
        print(f"❌ Error en health check: {response.status_code}")
        return None

def test_clone_voice(text, turbo=False, voice_id=None):
    """Probar clonación de voz"""
    mode = "turbo" if turbo else "normal"
    print(f"\n🎤 Probando clonación de voz ({mode})...")
    print(f"📝 Texto: {text}")
    
    # Preparar datos
    data = {
        'text': text,
        'turbo': str(turbo).lower()
    }
    
    if voice_id:
        data['voice_id'] = voice_id
        print(f"🎯 Usando voz: {voice_id}")
    
    # Medir tiempo
    start_time = time.time()
    
    try:
        response = requests.post(f"{API_BASE}/clone", data=data)
        
        if response.status_code == 200:
            end_time = time.time()
            duration = end_time - start_time
            
            # Guardar archivo de audio
            filename = f"test_audio_{mode}.wav"
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Generación exitosa ({mode})")
            print(f"⏱️ Tiempo: {duration:.2f} segundos")
            print(f"💾 Audio guardado: {filename}")
            print(f"📊 Tamaño: {len(response.content)} bytes")
            
            return duration
            
        else:
            print(f"❌ Error en clonación: {response.status_code}")
            print(f"📄 Respuesta: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return None

def main():
    """Función principal de prueba"""
    print("🚀 Iniciando pruebas de Voice Cloning API con modo turbo")
    print("=" * 60)
    
    # Verificar estado
    health_data = test_health()
    if not health_data:
        print("❌ No se pudo conectar a la API")
        return
    
    # Verificar si hay voces disponibles
    print("\n📢 Verificando voces disponibles...")
    try:
        response = requests.get(f"{API_BASE}/voices")
        if response.status_code == 200:
            voices_data = response.json()
            print(f"✅ Voces encontradas: {voices_data['total_collections']}")
            
            if voices_data['total_collections'] > 0:
                # Usar la primera voz disponible
                first_voice = list(voices_data['voice_collections'].keys())[0]
                print(f"🎯 Usando voz: {first_voice}")
                
                # Probar con voz específica
                print("\n" + "=" * 40)
                print("🧪 PRUEBA CON VOZ ESPECÍFICA")
                print("=" * 40)
                
                # Modo normal
                time_normal = test_clone_voice(TEST_TEXT, turbo=False, voice_id=first_voice)
                
                # Modo turbo
                time_turbo = test_clone_voice(TEST_TEXT, turbo=True, voice_id=first_voice)
                
                # Comparar velocidades
                if time_normal and time_turbo:
                    speedup = time_normal / time_turbo
                    print(f"\n🏁 COMPARACIÓN DE VELOCIDAD:")
                    print(f"⚡ Modo normal: {time_normal:.2f}s")
                    print(f"🚀 Modo turbo: {time_turbo:.2f}s")
                    print(f"📈 Speedup: {speedup:.2f}x más rápido")
                    
                    if speedup > 1.2:
                        print("✅ Modo turbo es significativamente más rápido")
                    elif speedup > 1.0:
                        print("⚠️ Modo turbo es ligeramente más rápido")
                    else:
                        print("❌ Modo turbo no es más rápido")
            
        else:
            print("⚠️ No se pudieron obtener las voces, probando sin voz específica")
            
    except Exception as e:
        print(f"⚠️ Error obteniendo voces: {e}")
    
    # Probar sin voz específica
    print("\n" + "=" * 40)
    print("🧪 PRUEBA SIN VOZ ESPECÍFICA")
    print("=" * 40)
    
    # Modo normal
    time_normal = test_clone_voice(TEST_TEXT, turbo=False)
    
    # Modo turbo
    time_turbo = test_clone_voice(TEST_TEXT, turbo=True)
    
    # Comparar velocidades
    if time_normal and time_turbo:
        speedup = time_normal / time_turbo
        print(f"\n🏁 COMPARACIÓN DE VELOCIDAD (sin voz específica):")
        print(f"⚡ Modo normal: {time_normal:.2f}s")
        print(f"🚀 Modo turbo: {time_turbo:.2f}s")
        print(f"📈 Speedup: {speedup:.2f}x más rápido")
        
        if speedup > 1.2:
            print("✅ Modo turbo es significativamente más rápido")
        elif speedup > 1.0:
            print("⚠️ Modo turbo es ligeramente más rápido")
        else:
            print("❌ Modo turbo no es más rápido")
    
    print("\n✅ Pruebas completadas")
    print("🎵 Revisa los archivos test_audio_normal.wav y test_audio_turbo.wav")

if __name__ == "__main__":
    main() 