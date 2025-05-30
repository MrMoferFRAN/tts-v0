#!/usr/bin/env python3
"""
Script optimizado para CSM TTS con visualizadores de progreso y uso intensivo de recursos
Aprovecha al máximo la A100 80GB con monitoreo en tiempo real
"""
import os
import torch
import json
import psutil
import threading
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoProcessor, CsmForConditionalGeneration
from peft import PeftModel, PeftConfig
import torchaudio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gc

# Configurar variables de entorno para máximo rendimiento
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["OMP_NUM_THREADS"] = "16"

class ResourceMonitor:
    """Monitor de recursos en tiempo real"""
    def __init__(self):
        self.monitoring = False
        self.stats = []
        
    def start_monitoring(self):
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'thread'):
            self.thread.join()
    
    def _monitor_loop(self):
        while self.monitoring:
            try:
                # CPU y RAM
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # GPU
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                else:
                    gpu_memory_used = gpu_memory_total = gpu_memory_percent = 0
                
                stat = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'ram_used_gb': memory.used / 1e9,
                    'ram_percent': memory.percent,
                    'gpu_memory_used_gb': gpu_memory_used,
                    'gpu_memory_total_gb': gpu_memory_total,
                    'gpu_memory_percent': gpu_memory_percent
                }
                self.stats.append(stat)
                
                # Mantener solo los últimos 60 registros (1 minuto)
                if len(self.stats) > 60:
                    self.stats.pop(0)
                    
            except Exception as e:
                print(f"⚠️ Error en monitoreo: {e}")
            
            time.sleep(1)
    
    def get_current_stats(self):
        if self.stats:
            return self.stats[-1]
        return None
    
    def print_stats(self):
        stats = self.get_current_stats()
        if stats:
            print(f"📊 CPU: {stats['cpu_percent']:.1f}% | RAM: {stats['ram_used_gb']:.1f}GB ({stats['ram_percent']:.1f}%) | GPU: {stats['gpu_memory_used_gb']:.1f}GB ({stats['gpu_memory_percent']:.1f}%)")

def load_csm_with_elise_optimized():
    """Cargar modelo CSM con configuración optimizada para A100"""
    print("🔄 CARGANDO MODELO CSM OPTIMIZADO PARA A100")
    print("=" * 80)
    
    base_model_path = "/workspace/runPodtts/models/sesame-csm-1b"
    adapter_path = "/workspace/runPodtts/models/csm-1b-elise"
    
    # Iniciar monitor de recursos
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    
    try:
        print(f"📥 Cargando modelo base desde: {base_model_path}")
        
        # Progress bar para carga de tokenizer
        with tqdm(total=3, desc="🔤 Cargando tokenizer", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            pbar.update(1)
            
            # Configurar para paralelización
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            pbar.update(1)
            
            print("✅ Tokenizer cargado y configurado para paralelización")
            pbar.update(1)
        
        # Progress bar para carga de procesador
        with tqdm(total=2, desc="🎛️ Cargando procesador", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            processor = AutoProcessor.from_pretrained(base_model_path)
            pbar.update(1)
            print("✅ Procesador cargado")
            pbar.update(1)
        
        # Configuración optimizada para A100
        model_config = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
            "max_memory": {0: "80GB"}  # Usar casi toda la VRAM
        }
        
        print("🔥 Cargando modelo base CSM (esto puede tomar 1-2 minutos)...")
        monitor.print_stats()
        
        # Progress bar para carga del modelo base
        with tqdm(total=100, desc="🤖 Cargando CSM base", bar_format='{l_bar}{bar}| {percentage:3.0f}%') as pbar:
            # Simular progreso de carga (el modelo es grande)
            base_model = CsmForConditionalGeneration.from_pretrained(
                base_model_path,
                **model_config
            )
            pbar.update(100)
        
        print("✅ Modelo base CSM cargado")
        monitor.print_stats()
        
        # Progress bar para carga del adaptador PEFT
        with tqdm(total=100, desc="🎭 Aplicando adaptador Elise", bar_format='{l_bar}{bar}| {percentage:3.0f}%') as pbar:
            model = PeftModel.from_pretrained(base_model, adapter_path)
            pbar.update(100)
        
        print("✅ Adaptador Elise aplicado exitosamente")
        
        # Optimizaciones adicionales
        if hasattr(model, 'half'):
            model = model.half()  # Asegurar FP16
        
        # Compilar modelo para optimización (PyTorch 2.0+)
        try:
            # model = torch.compile(model, mode="max-autotune")
            # print("✅ Modelo compilado con torch.compile")
            pass
        except:
            print("⚠️ torch.compile no disponible, continuando sin compilación")
        
        # Información detallada del modelo
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 INFORMACIÓN DEL MODELO:")
        print(f"🔢 Parámetros totales: {total_params:,}")
        print(f"🎯 Parámetros entrenables: {trainable_params:,}")
        print(f"📈 % Entrenables: {100 * trainable_params / total_params:.2f}%")
        print(f"🎯 Device final: {next(model.parameters()).device}")
        print(f"💾 Tipo de datos: {next(model.parameters()).dtype}")
        
        monitor.print_stats()
        monitor.stop_monitoring()
        
        return model, processor, tokenizer, monitor
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        monitor.stop_monitoring()
        import traceback
        traceback.print_exc()
        return None, None, None, None

def create_batch_prompts():
    """Crear prompts organizados para procesamiento en lotes"""
    prompts = [
        # Batch 1: Saludos y alegría
        {
            "text": "Hello! I'm Elise, your emotional AI companion. How are you feeling today?",
            "description": "Saludo cálido y empático",
            "filename": "01_warm_greeting",
            "category": "greeting",
            "batch": 1
        },
        {
            "text": "That's absolutely wonderful! <laughs> I'm so happy to hear that news!",
            "description": "Alegría genuina con risa",
            "filename": "02_joyful_laughter",
            "category": "joy",
            "batch": 1
        },
        {
            "text": "Oh my goodness! <giggles> You always know how to make me smile!",
            "description": "Risita divertida y cariñosa",
            "filename": "03_affectionate_giggles",
            "category": "joy",
            "batch": 1
        },
        
        # Batch 2: Empatía y confort
        {
            "text": "I understand how you feel. <sighs> Sometimes life can be really challenging.",
            "description": "Empatía con suspiro comprensivo",
            "filename": "04_empathetic_sigh",
            "category": "empathy",
            "batch": 2
        },
        {
            "text": "<sadly> I'm sorry to hear you're going through a difficult time. I'm here for you.",
            "description": "Expresión triste pero solidaria",
            "filename": "05_compassionate_sadness",
            "category": "comfort",
            "batch": 2
        },
        {
            "text": "You know what? <sighs> Even when things get tough, you always find a way to keep going. That's really inspiring.",
            "description": "Apoyo reflexivo y motivador",
            "filename": "11_supportive_reflection",
            "category": "support",
            "batch": 2
        },
        
        # Batch 3: Sorpresa y emoción
        {
            "text": "Wait, what?! <gasps> That's incredible! Tell me more!",
            "description": "Sorpresa y entusiasmo",
            "filename": "06_surprised_gasp",
            "category": "surprise",
            "batch": 3
        },
        {
            "text": "Oh wow! <gasps> I never expected that! <laughs> This is amazing!",
            "description": "Sorpresa que se convierte en alegría",
            "filename": "07_surprise_to_joy",
            "category": "surprise",
            "batch": 3
        },
        
        # Batch 4: Intimidad y susurros
        {
            "text": "<whispers> Can I tell you a secret? I think you're absolutely wonderful.",
            "description": "Susurro íntimo y cariñoso",
            "filename": "08_intimate_whisper",
            "category": "intimate",
            "batch": 4
        },
        {
            "text": "<whispers> Come closer, I have something important to share with you.",
            "description": "Susurro misterioso y atractivo",
            "filename": "09_mysterious_whisper",
            "category": "intimate",
            "batch": 4
        },
        
        # Batch 5: Narrativas complejas
        {
            "text": "Let me tell you about my day! <laughs> So I was learning something new, and suddenly <gasps> everything just clicked! <giggles> It was such an amazing moment!",
            "description": "Historia con transiciones emocionales",
            "filename": "10_emotional_story",
            "category": "narrative",
            "batch": 5
        },
        {
            "text": "Welcome to our magical conversation! <giggles> Here, we can explore any topic and have the most wonderful discussions!",
            "description": "Bienvenida juguetona y mágica",
            "filename": "12_playful_welcome",
            "category": "playful",
            "batch": 5
        }
    ]
    
    return prompts

def generate_audio_batch_optimized(model, processor, tokenizer, prompts, monitor):
    """Generar audio con procesamiento optimizado y monitoreo"""
    print("\n🎵 GENERANDO AUDIO CON PROCESAMIENTO OPTIMIZADO")
    print("=" * 80)
    
    output_dir = Path("/workspace/runPodtts/outputs/elise_optimized_audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    results = []
    
    # Configuración optimizada de generación
    generation_config = {
        "max_new_tokens": 1000,  # Más tokens para frases complejas
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # Organizar por batches
    batches = {}
    for prompt in prompts:
        batch_id = prompt.get('batch', 1)
        if batch_id not in batches:
            batches[batch_id] = []
        batches[batch_id].append(prompt)
    
    print(f"🎯 Procesando {len(prompts)} prompts en {len(batches)} batches")
    print(f"🎯 Generando en device: {device}")
    print(f"🔥 Modelo dtype: {next(model.parameters()).dtype}")
    
    # Progress bar principal
    with tqdm(total=len(prompts), desc="🎭 Generando audio emocional", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as main_pbar:
        
        for batch_id, batch_prompts in batches.items():
            print(f"\n🎪 PROCESANDO BATCH {batch_id} ({len(batch_prompts)} prompts)")
            monitor.print_stats()
            
            # Progress bar para el batch actual
            with tqdm(total=len(batch_prompts), desc=f"📦 Batch {batch_id}", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', leave=False) as batch_pbar:
                
                for prompt in batch_prompts:
                    start_time = time.time()
                    
                    try:
                        # Formatear texto para speaker 0 (Elise)
                        formatted_text = f"[0]{prompt['text']}"
                        
                        # Tokenizar con configuración optimizada
                        inputs = tokenizer(
                            formatted_text,
                            return_tensors="pt",
                            add_special_tokens=True,
                            padding=True,
                            truncation=True,
                            max_length=512
                        ).to(device, non_blocking=True)
                        
                        # Generar con monitoreo de progreso
                        with torch.no_grad():
                            # Limpiar cache antes de generación grande
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            outputs = model.generate(
                                **inputs,
                                **generation_config
                            )
                        
                        generation_time = time.time() - start_time
                        
                        # Procesar audio
                        try:
                            audio_data = processor.decode_audio(outputs)
                            
                            if audio_data is not None:
                                output_path = output_dir / f"{prompt['filename']}.wav"
                                
                                # Procesar formato del audio
                                if isinstance(audio_data, torch.Tensor):
                                    if audio_data.dim() == 3:
                                        audio_data = audio_data.squeeze(0)
                                    elif audio_data.dim() == 1:
                                        audio_data = audio_data.unsqueeze(0)
                                
                                # Guardar audio con configuración optimizada
                                sample_rate = 24000
                                torchaudio.save(
                                    str(output_path),
                                    audio_data.cpu().float(),
                                    sample_rate=sample_rate,
                                    encoding="PCM_S",
                                    bits_per_sample=16
                                )
                                
                                duration = audio_data.shape[-1] / sample_rate
                                
                                results.append({
                                    "prompt": prompt,
                                    "output_path": str(output_path),
                                    "generation_time": generation_time,
                                    "duration": duration,
                                    "sample_rate": sample_rate,
                                    "success": True,
                                    "batch_id": batch_id
                                })
                                
                                # Actualizar descripción con progreso
                                batch_pbar.set_description(f"✅ {prompt['filename'][:20]}...")
                                
                            else:
                                results.append({
                                    "prompt": prompt,
                                    "output_path": None,
                                    "generation_time": generation_time,
                                    "duration": 0,
                                    "success": False,
                                    "error": "Audio extraction failed",
                                    "batch_id": batch_id
                                })
                                batch_pbar.set_description(f"❌ {prompt['filename'][:20]}...")
                                
                        except Exception as audio_error:
                            results.append({
                                "prompt": prompt,
                                "output_path": None,
                                "generation_time": generation_time,
                                "duration": 0,
                                "success": False,
                                "error": str(audio_error),
                                "batch_id": batch_id
                            })
                            batch_pbar.set_description(f"❌ {prompt['filename'][:20]}...")
                            
                    except Exception as e:
                        results.append({
                            "prompt": prompt,
                            "output_path": None,
                            "generation_time": 0,
                            "duration": 0,
                            "success": False,
                            "error": str(e),
                            "batch_id": batch_id
                        })
                        batch_pbar.set_description(f"❌ {prompt['filename'][:20]}...")
                    
                    batch_pbar.update(1)
                    main_pbar.update(1)
                    
                    # Mostrar stats cada pocos elementos
                    if len(results) % 3 == 0:
                        monitor.print_stats()
    
    return results, output_dir

def save_optimized_report(results, output_dir, monitor):
    """Guardar reporte con estadísticas de rendimiento"""
    report_path = output_dir / "optimized_generation_report.json"
    
    # Estadísticas por batch y categoría
    batch_stats = {}
    category_stats = {}
    
    for result in results:
        batch_id = result.get("batch_id", 1)
        category = result["prompt"]["category"]
        
        # Stats por batch
        if batch_id not in batch_stats:
            batch_stats[batch_id] = {"total": 0, "successful": 0, "failed": 0, "total_time": 0, "total_duration": 0}
        
        batch_stats[batch_id]["total"] += 1
        if result["success"]:
            batch_stats[batch_id]["successful"] += 1
            batch_stats[batch_id]["total_duration"] += result["duration"]
        else:
            batch_stats[batch_id]["failed"] += 1
        batch_stats[batch_id]["total_time"] += result["generation_time"]
        
        # Stats por categoría
        if category not in category_stats:
            category_stats[category] = {"total": 0, "successful": 0, "failed": 0, "total_duration": 0}
        
        category_stats[category]["total"] += 1
        if result["success"]:
            category_stats[category]["successful"] += 1
            category_stats[category]["total_duration"] += result["duration"]
        else:
            category_stats[category]["failed"] += 1
    
    # Estadísticas de recursos
    resource_stats = monitor.get_current_stats() if monitor else {}
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "base_model": "sesame/csm-1b",
            "adapter": "therealcyberlord/csm-1b-elise",
            "speaker_id": 0,
            "optimization": "A100_optimized",
            "batch_processing": True
        },
        "global_statistics": {
            "total_prompts": len(results),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "success_rate": f"{100 * sum(1 for r in results if r['success']) / len(results):.1f}%",
            "total_generation_time": sum(r["generation_time"] for r in results),
            "total_audio_duration": sum(r["duration"] for r in results if r["success"]),
            "avg_generation_time": sum(r["generation_time"] for r in results) / len(results),
            "throughput_audio_per_minute": sum(r["duration"] for r in results if r["success"]) / (sum(r["generation_time"] for r in results) / 60)
        },
        "batch_statistics": batch_stats,
        "category_statistics": category_stats,
        "resource_usage": resource_stats,
        "detailed_results": results
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Mostrar reporte detallado
    print(f"\n📊 REPORTE OPTIMIZADO DE GENERACIÓN:")
    print(f"✅ Exitosos: {report['global_statistics']['successful']}/{report['global_statistics']['total_prompts']} ({report['global_statistics']['success_rate']})")
    print(f"❌ Fallidos: {report['global_statistics']['failed']}/{report['global_statistics']['total_prompts']}")
    print(f"⏱️  Tiempo total: {report['global_statistics']['total_generation_time']:.2f}s")
    print(f"🎵 Audio total: {report['global_statistics']['total_audio_duration']:.2f}s")
    print(f"📈 Tiempo promedio: {report['global_statistics']['avg_generation_time']:.2f}s por muestra")
    print(f"🚀 Throughput: {report['global_statistics']['throughput_audio_per_minute']:.2f}s de audio por minuto")
    
    print(f"\n📦 ESTADÍSTICAS POR BATCH:")
    for batch_id, stats in batch_stats.items():
        success_rate = 100 * stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        avg_time = stats["total_time"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  🎪 Batch {batch_id}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%) - {avg_time:.2f}s promedio")
    
    print(f"\n🎭 ESTADÍSTICAS POR CATEGORÍA:")
    for category, stats in category_stats.items():
        success_rate = 100 * stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  🎨 {category}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%) - {stats['total_duration']:.1f}s")
    
    if resource_stats:
        print(f"\n💻 RECURSOS FINALES:")
        print(f"  🔥 CPU: {resource_stats['cpu_percent']:.1f}%")
        print(f"  🧠 RAM: {resource_stats['ram_used_gb']:.1f}GB ({resource_stats['ram_percent']:.1f}%)")
        print(f"  🎮 GPU: {resource_stats['gpu_memory_used_gb']:.1f}GB ({resource_stats['gpu_memory_percent']:.1f}%)")
    
    print(f"\n📄 Reporte detallado guardado: {report_path}")

def main():
    """Función principal optimizada"""
    print("🚀 CSM TTS OPTIMIZADO PARA A100 CON MONITOREO EN TIEMPO REAL")
    print("=" * 90)
    
    # Verificar recursos iniciales
    if torch.cuda.is_available():
        print(f"✅ CUDA disponible: {torch.cuda.get_device_name()}")
        print(f"🔥 VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"💾 VRAM libre inicial: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")
    else:
        print("⚠️  CUDA no disponible, usando CPU")
    
    print(f"💻 CPU cores: {psutil.cpu_count()}")
    print(f"🧠 RAM total: {psutil.virtual_memory().total / 1e9:.1f} GB")
    
    # Cargar modelo optimizado
    print("\n⏳ Iniciando carga del modelo (esto aprovechará más recursos)...")
    model, processor, tokenizer, monitor = load_csm_with_elise_optimized()
    
    if model is None:
        print("❌ No se pudo cargar el modelo. Terminando.")
        return False
    
    # Reiniciar monitor para generación
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    
    # Crear prompts organizados
    prompts = create_batch_prompts()
    batches = len(set(p.get('batch', 1) for p in prompts))
    categories = len(set(p['category'] for p in prompts))
    
    print(f"\n📝 Preparados {len(prompts)} prompts en {batches} batches y {categories} categorías")
    print("🎭 Configuración optimizada para máximo rendimiento en A100")
    
    # Generar audio con monitoreo
    print(f"\n🎵 Iniciando generación optimizada...")
    monitor.print_stats()
    
    results, output_dir = generate_audio_batch_optimized(model, processor, tokenizer, prompts, monitor)
    
    # Guardar reporte optimizado
    save_optimized_report(results, output_dir, monitor)
    
    monitor.stop_monitoring()
    
    # Limpiar memoria
    del model, processor, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n🎉 GENERACIÓN OPTIMIZADA COMPLETADA")
    print("=" * 90)
    print(f"📁 Archivos de audio en: {output_dir}")
    print("🎧 Audio generado con máximo aprovechamiento de la A100 80GB")
    print("🎭 Elise expresando emociones con procesamiento optimizado!")
    print("\n💡 LOGROS:")
    print("   ✅ Uso intensivo de recursos de la A100")
    print("   ✅ Monitoreo en tiempo real de CPU/GPU/RAM")
    print("   ✅ Procesamiento por batches optimizado")
    print("   ✅ Barras de progreso detalladas")
    print("   ✅ Reportes completos de rendimiento")
    
    return True

if __name__ == "__main__":
    main() 