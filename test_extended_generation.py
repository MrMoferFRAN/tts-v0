#!/usr/bin/env python3
"""
Script de prueba para capacidades de generación extendida
Demuestra la generación de audio de hasta 3 minutos
"""

import time
import requests
import json

# Configuración
API_BASE = "http://localhost:7860"

def test_duration_scaling():
    """Probar diferentes duraciones para ver la escalabilidad"""
    print("🎯 PRUEBAS DE ESCALABILIDAD DE DURACIÓN")
    print("=" * 50)
    
    test_cases = [
        {"duration": 10, "description": "10 segundos - Corto"},
        {"duration": 30, "description": "30 segundos - Medio"},
        {"duration": 60, "description": "60 segundos - 1 minuto"},
        {"duration": 90, "description": "90 segundos - 1.5 minutos"},
        {"duration": 120, "description": "120 segundos - 2 minutos"},
        {"duration": 180, "description": "180 segundos - 3 minutos MÁXIMO"}
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\n🔍 Probando {test['description']}...")
        
        text = f"Esta es una prueba de generación de audio para {test['description']}. " \
               f"El objetivo es generar exactamente {test['duration']} segundos de audio continuo " \
               f"de alta calidad usando el modelo turbo optimizado. Este texto está diseñado " \
               f"para ser lo suficientemente largo y natural para el tiempo objetivo especificado."
        
        start_time = time.time()
        
        try:
            response = requests.post(f"{API_BASE}/clone_extended", data={
                'text': text,
                'target_duration': test['duration'],
                'turbo': 'true'
            })
            
            if response.status_code == 200:
                end_time = time.time()
                generation_time = end_time - start_time
                
                # Obtener información de headers
                actual_duration = float(response.headers.get('X-Audio-Duration', 0))
                tokens_used = response.headers.get('X-Tokens-Used', 'N/A')
                
                # Guardar archivo
                filename = f"test_{test['duration']}s.wav"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                
                result = {
                    'target': test['duration'],
                    'actual': actual_duration,
                    'generation_time': generation_time,
                    'tokens': tokens_used,
                    'file_size': len(response.content),
                    'efficiency': actual_duration / generation_time if generation_time > 0 else 0,
                    'accuracy': (actual_duration / test['duration']) * 100 if test['duration'] > 0 else 0
                }
                
                results.append(result)
                
                print(f"✅ Éxito!")
                print(f"   📊 Objetivo: {test['duration']}s")
                print(f"   📊 Real: {actual_duration:.2f}s")
                print(f"   ⏱️ Tiempo gen.: {generation_time:.2f}s")
                print(f"   🎯 Precisión: {result['accuracy']:.1f}%")
                print(f"   🚀 Eficiencia: {result['efficiency']:.2f}x")
                print(f"   💾 Archivo: {filename} ({len(response.content)} bytes)")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"📄 Respuesta: {response.text}")
                
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
    
    return results

def test_token_comparison():
    """Comparar endpoint normal vs extendido"""
    print("\n🔍 COMPARACIÓN: /clone vs /clone_extended")
    print("=" * 50)
    
    text = "Esta es una prueba comparativa entre el endpoint normal y el extendido " \
           "para evaluar las diferencias en duración y calidad del audio generado " \
           "usando los mismos parámetros base pero diferentes enfoques de optimización."
    
    # Endpoint normal con max_tokens alto
    print("\n🎤 Probando /clone con max_tokens=15000...")
    start = time.time()
    try:
        response_normal = requests.post(f"{API_BASE}/clone", data={
            'text': text,
            'max_tokens': 15000,
            'turbo': 'true'
        })
        time_normal = time.time() - start
        
        if response_normal.status_code == 200:
            with open("test_normal_endpoint.wav", 'wb') as f:
                f.write(response_normal.content)
            print(f"✅ Normal: {time_normal:.2f}s de generación, {len(response_normal.content)} bytes")
        else:
            print(f"❌ Error normal: {response_normal.status_code}")
            
    except Exception as e:
        print(f"❌ Error normal: {e}")
    
    # Endpoint extendido
    print("\n🎯 Probando /clone_extended con target_duration=60s...")
    start = time.time()
    try:
        response_extended = requests.post(f"{API_BASE}/clone_extended", data={
            'text': text,
            'target_duration': 60,
            'turbo': 'true'
        })
        time_extended = time.time() - start
        
        if response_extended.status_code == 200:
            actual_duration = float(response_extended.headers.get('X-Audio-Duration', 0))
            tokens_used = response_extended.headers.get('X-Tokens-Used', 'N/A')
            
            with open("test_extended_endpoint.wav", 'wb') as f:
                f.write(response_extended.content)
            print(f"✅ Extendido: {time_extended:.2f}s de generación, {len(response_extended.content)} bytes")
            print(f"   📊 Duración real: {actual_duration:.2f}s")
            print(f"   🎯 Tokens: {tokens_used}")
        else:
            print(f"❌ Error extendido: {response_extended.status_code}")
            
    except Exception as e:
        print(f"❌ Error extendido: {e}")

def generate_summary(results):
    """Generar resumen de resultados"""
    print("\n📊 RESUMEN DE RESULTADOS")
    print("=" * 50)
    
    if not results:
        print("❌ No hay resultados para mostrar")
        return
    
    print("| Objetivo | Real    | Precisión | Eficiencia | Tiempo Gen. |")
    print("|----------|---------|-----------|------------|-------------|")
    
    total_accuracy = 0
    total_efficiency = 0
    count = 0
    
    for r in results:
        if r:
            print(f"| {r['target']:8}s | {r['actual']:7.2f}s | {r['accuracy']:9.1f}% | {r['efficiency']:10.2f}x | {r['generation_time']:11.2f}s |")
            total_accuracy += r['accuracy']
            total_efficiency += r['efficiency']
            count += 1
    
    if count > 0:
        avg_accuracy = total_accuracy / count
        avg_efficiency = total_efficiency / count
        
        print("|----------|---------|-----------|------------|-------------|")
        print(f"| PROMEDIO | -       | {avg_accuracy:9.1f}% | {avg_efficiency:10.2f}x | -           |")
        
        print(f"\n🎯 CONCLUSIONES:")
        print(f"   • Precisión promedio: {avg_accuracy:.1f}%")
        print(f"   • Eficiencia promedio: {avg_efficiency:.2f}x (audio_duration/generation_time)")
        
        max_duration = max(r['actual'] for r in results if r)
        print(f"   • Máxima duración alcanzada: {max_duration:.2f} segundos")
        
        if max_duration >= 60:
            print("   ✅ Capacidad de generar 1+ minuto confirmada")
        if max_duration >= 120:
            print("   ✅ Capacidad de generar 2+ minutos confirmada")
        if max_duration >= 150:
            print("   ✅ Cerca de los 3 minutos objetivo")

def main():
    """Función principal"""
    print("🚀 PRUEBAS COMPLETAS DE GENERACIÓN EXTENDIDA")
    print("🎤 Voice Cloning API - Capacidades de hasta 3 minutos")
    print("=" * 60)
    
    # Verificar estado de la API
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ API funcionando correctamente")
            if data.get('turbo_model', {}).get('available'):
                print("🚀 Modelo turbo disponible")
            else:
                print("⚠️ Modelo turbo no disponible")
        else:
            print(f"❌ API no responde: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ No se puede conectar a la API: {e}")
        return
    
    # Ejecutar pruebas
    results = test_duration_scaling()
    test_token_comparison()
    generate_summary(results)
    
    print(f"\n✅ Pruebas completadas")
    print(f"🎵 Revisa los archivos test_*.wav generados")

if __name__ == "__main__":
    main() 