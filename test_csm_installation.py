#!/usr/bin/env python3
"""
Script de verificación de la instalación de CSM
Verifica que todas las dependencias están correctas sin cargar el modelo completo.
"""

import sys
import os

def test_imports():
    """Prueba las importaciones básicas"""
    print("🔍 Verificando importaciones...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"❌ Error importando PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Error importando Transformers: {e}")
        return False
    
    try:
        import torchaudio
        print(f"✅ TorchAudio: {torchaudio.__version__}")
    except ImportError as e:
        print(f"❌ Error importando TorchAudio: {e}")
        return False
    
    try:
        import safetensors
        print(f"✅ SafeTensors disponible")
    except ImportError as e:
        print(f"❌ Error importando SafeTensors: {e}")
        return False
    
    return True

def test_csm_module():
    """Prueba el módulo CSM"""
    print("\n🔍 Verificando módulo CSM...")
    
    # Cambiar al directorio CSM
    csm_dir = "/workspacetts-v0/models/csm"
    if not os.path.exists(csm_dir):
        print(f"❌ Directorio CSM no encontrado: {csm_dir}")
        return False
    
    # Verificar archivos principales
    required_files = ["generator.py", "models.py", "run_csm.py"]
    for file in required_files:
        file_path = os.path.join(csm_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file} encontrado")
        else:
            print(f"❌ {file} no encontrado")
            return False
    
    # Intentar importar desde CSM
    sys.path.insert(0, csm_dir)
    try:
        from generator import load_csm_1b
        print("✅ Generator importado exitosamente")
    except ImportError as e:
        print(f"❌ Error importando generator: {e}")
        return False
    
    return True

def test_model_file():
    """Verifica el archivo del modelo"""
    print("\n🔍 Verificando archivo del modelo...")
    
    model_path = "/workspacetts-v0/models/csm-1b.safetensors"
    if os.path.exists(model_path):
        size_gb = os.path.getsize(model_path) / (1024**3)
        print(f"✅ Modelo CSM encontrado: {size_gb:.1f} GB")
        return True
    else:
        print(f"❌ Modelo CSM no encontrado: {model_path}")
        return False

def test_dataset():
    """Verifica el dataset Elise"""
    print("\n🔍 Verificando dataset Elise...")
    
    dataset_path = "/workspacetts-v0/datasets/csm-1b-elise"
    if os.path.exists(dataset_path):
        print(f"✅ Dataset Elise encontrado: {dataset_path}")
        return True
    else:
        print(f"❌ Dataset Elise no encontrado: {dataset_path}")
        return False

def main():
    """Función principal"""
    print("🚀 Verificación de Instalación CSM")
    print("=" * 50)
    
    tests = [
        ("Importaciones básicas", test_imports),
        ("Módulo CSM", test_csm_module), 
        ("Archivo del modelo", test_model_file),
        ("Dataset Elise", test_dataset)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Ejecutando: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE VERIFICACIÓN:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ¡INSTALACIÓN COMPLETA Y FUNCIONAL!")
        print("💡 Para usar CSM:")
        print("   cd /workspacetts-v0/models/csm")
        print("   python run_csm.py")
    else:
        print("⚠️  INSTALACIÓN INCOMPLETA")
        print("🔧 Revisa los errores arriba")
    
    return all_passed

if __name__ == "__main__":
    main() 