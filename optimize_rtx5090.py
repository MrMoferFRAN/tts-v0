#!/usr/bin/env python3
"""
RTX 5090 Optimization Script
Configura el sistema para máxima compatibilidad con RTX 5090
"""

import os
import torch
import warnings

def optimize_rtx5090():
    """Optimizaciones específicas para RTX 5090"""
    
    print("🚨 RTX 5090 Optimization Mode")
    print("=" * 40)
    
    # Suprimir warnings específicos de RTX 5090
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
    
    # Variables de entorno para estabilidad RTX 5090
    rtx5090_env = {
        # Memoria y estabilidad
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:256,expandable_segments:False',
        'CUDA_LAUNCH_BLOCKING': '1',  # Sincronización para estabilidad
        'TORCH_USE_CUDA_DSA': '0',    # Desactivar device-side assertions
        
        # Compatibilidad
        'NO_TORCH_COMPILE': '1',
        'PYTORCH_ENABLE_MPS_FALLBACK': '1',
        'TOKENIZERS_PARALLELISM': 'false',
        
        # Conservador para RTX 5090
        'TORCH_CUDNN_DETERMINISTIC': '1',
        'TORCH_CUDNN_BENCHMARK': '0',
    }
    
    print("🔧 Aplicando configuraciones RTX 5090...")
    for key, value in rtx5090_env.items():
        os.environ[key] = value
        print(f"   ✅ {key}={value}")
    
    # Configurar PyTorch para RTX 5090
    if torch.cuda.is_available():
        try:
            device_props = torch.cuda.get_device_properties(0)
            
            if 'RTX 5090' in device_props.name or device_props.major >= 12:
                print(f"🖥️ GPU detectada: {device_props.name}")
                print(f"🔧 Compute Capability: {device_props.major}.{device_props.minor}")
                
                # Configuraciones conservadoras para RTX 5090
                torch.backends.cuda.matmul.allow_tf32 = False  # Más conservador
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                print("✅ Configuraciones RTX 5090 aplicadas")
                print("⚠️ Modo conservador: TF32 desactivado para máxima estabilidad")
                
                return True
            else:
                print("ℹ️ No es RTX 5090, usando configuraciones estándar")
                return False
                
        except Exception as e:
            print(f"❌ Error configurando RTX 5090: {e}")
            return False
    else:
        print("❌ CUDA no disponible")
        return False

def test_rtx5090_compatibility():
    """Prueba la compatibilidad RTX 5090"""
    print("\n🧪 Testing RTX 5090 Compatibility")
    print("=" * 35)
    
    try:
        # Test básico de tensor
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
        z = x + y
        
        print(f"✅ Basic tensor ops: {z.cpu().numpy()}")
        
        # Test de memoria
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        memory_cached = torch.cuda.memory_reserved() / 1024**2
        
        print(f"📊 Memory allocated: {memory_allocated:.1f} MB")
        print(f"📊 Memory cached: {memory_cached:.1f} MB")
        
        # Test modelo pequeño
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('./models/csm-1b-turbo')
        
        inputs = tokenizer("Test RTX 5090", return_tensors="pt").to('cuda')
        print(f"✅ Tokenizer test: {inputs['input_ids'].shape}")
        
        # Limpiar memoria
        del x, y, z, inputs
        torch.cuda.empty_cache()
        
        print("✅ RTX 5090 compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ RTX 5090 compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    # Optimizar para RTX 5090
    is_rtx5090 = optimize_rtx5090()
    
    if is_rtx5090:
        # Ejecutar test de compatibilidad
        if test_rtx5090_compatibility():
            print("\n🚀 RTX 5090 está listo para voice cloning!")
            print("💡 Recomendaciones:")
            print("   • Usar modo conservador para máxima estabilidad")
            print("   • Considerar actualizar a PyTorch 2.5+ para mejor rendimiento")
            print("   • Monitorear temperatura GPU durante uso intensivo")
        else:
            print("\n⚠️ RTX 5090 tiene problemas de compatibilidad")
            print("🔧 Soluciones:")
            print("   1. Usar modo CPU: export CUDA_VISIBLE_DEVICES=''")
            print("   2. Actualizar PyTorch: pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu124")
    else:
        print("\n✅ GPU estándar detectada - configuración normal") 