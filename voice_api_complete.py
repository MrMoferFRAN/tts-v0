#!/usr/bin/env python3
"""
Voice Cloning API Completa - CSM-1B
API robusta con estructura de carpetas organizadas por voz
"""

import os
import sys
import logging
import traceback
import json
import hashlib
import re
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import tempfile
import shutil
from contextlib import asynccontextmanager

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import aiofiles
from transformers import CsmForConditionalGeneration, AutoProcessor
import numpy as np
from pydantic import BaseModel

# Production dependencies
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    print("⚠️ SlowAPI not available, rate limiting disabled. Install with: pip install slowapi")

# Redis for rate limiting (optional)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Generation defaults for CSM model
GENERATION_DEFAULTS = {
    "temperature": 0.7,     # creatividad moderada
    "max_tokens": 4096,     # ≈ 1 min de audio (24 kHz)
    "do_sample": True,      # sampling estocástico activado
    "output_audio": True    # el modelo debe devolver audio WAV
}

# Production Configuration - Environment Variables
MAX_CONCURRENT_INFERENCE = int(os.getenv("MAX_CONCURRENT_INFERENCE", "3"))
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", "120"))  # seconds
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
REDIS_URL = os.getenv("REDIS_URL", None)  # redis://localhost:6379
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Concurrency control
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)

# Rate limiter setup
if SLOWAPI_AVAILABLE:
    if REDIS_AVAILABLE and REDIS_URL:
        # Use Redis for distributed rate limiting
        redis_client = redis.from_url(REDIS_URL)
        limiter = Limiter(key_func=get_remote_address, storage_uri=REDIS_URL)
    else:
        # Use in-memory rate limiting
        limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None

# CUDA Compatibility Configuration for RTX 4090 and RTX 6000 Ada
def setup_cuda_compatibility():
    """Setup CUDA environment optimized for RTX 4090 and RTX 6000 Ada"""
    
    # Essential CUDA environment variables
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
    os.environ.setdefault('TORCH_USE_CUDA_DSA', '0')
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    
    # Memory management for large models
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512,expandable_segments:True')
    
    # Optimize for RTX series
    os.environ.setdefault('TORCH_CUDNN_V8_API_ENABLED', '1')
    os.environ.setdefault('NO_TORCH_COMPILE', '1')
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    if torch.cuda.is_available():
        try:
            device_props = torch.cuda.get_device_properties(0)
            compute_capability = f"{device_props.major}.{device_props.minor}"
            
            print(f"🖥️ GPU: {device_props.name}")
            print(f"🔧 Compute Capability: {compute_capability}")
            print(f"💾 Memory: {device_props.total_memory / 1024**3:.1f} GB")
            
            # Optimize for RTX 4090 and 6000 Ada
            if device_props.major >= 9:  # RTX 4090 series with sm_90 support
                print("🚀 RTX 4090+ series detected - applying advanced optimizations")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            elif device_props.major >= 8:  # RTX 6000 Ada series
                print("⚡ RTX 6000 Ada series detected - enabling optimizations")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
            return True
            
        except Exception as e:
            print(f"⚠️ GPU detection failed: {e}")
            print("🔄 Falling back to CPU mode")
            return False
    else:
        print("💻 No CUDA available, using CPU mode")
        return False

# Setup CUDA compatibility before importing other modules
cuda_available = setup_cuda_compatibility()

# Fix for torch.compiler compatibility issues
if not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler.is_compiling = lambda: False

# Production Logging Configuration
def setup_logging():
    """Setup production-ready logging with proper formatting and rotation"""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure root logger
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    
    # Format for production with more details
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # File handler with rotation
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            'logs/voice_api.log',
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        handlers = [console_handler, file_handler]
    except Exception:
        # Fallback to simple file handler
        handlers = [console_handler]
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Configuración del entorno adicional
os.environ.setdefault('HF_TOKEN', '|==>REMOVED')

# Modelos Pydantic para respuestas
class VoiceProfile(BaseModel):
    name: str
    audio_path: str
    transcription: str
    language: str = "es"
    quality_score: float = 1.0
    duration: float
    sample_rate: int
    created_at: str

class VoiceCollection(BaseModel):
    voice_id: str
    profiles: List[VoiceProfile]
    total_samples: int
    average_duration: float
    created_at: str
    updated_at: str

class CSMVoiceManager:
    """Gestor completo de voces para CSM-1B con soporte turbo"""
    
    def __init__(
        self, 
        model_path: str = "./models/sesame-csm-1b", 
        turbo_model_path: str = "./models/csm-1b-turbo",
        voices_dir: str = "./voices"
    ):
        self.model_path = model_path
        self.turbo_model_path = turbo_model_path
        self.voices_dir = Path(voices_dir)
        
        # Use global cuda_available variable to determine device
        global cuda_available
        self.device = "cuda" if (torch.cuda.is_available() and cuda_available) else "cpu"
        
        # Modelos y procesadores
        self.model = None
        self.processor = None
        self.turbo_model = None
        self.turbo_processor = None
        
        self.voice_collections = {}
        
        logger.info(f"🎤 Initializing CSM Voice Manager - TURBO ONLY MODE")
        logger.info(f"🚀 Turbo model path: {turbo_model_path}")
        logger.info(f"📁 Normal model path: {model_path} (NOT LOADED)")
        logger.info(f"📁 Voices directory: {voices_dir}")
        logger.info(f"🖥️ Device: {self.device}")
        
        if torch.cuda.is_available() and self.device == "cuda":
            logger.info("✅ CUDA optimizations enabled for RTX 4090/6000 Ada")
        
        # Crear directorios necesarios
        self.voices_dir.mkdir(exist_ok=True)
        Path("outputs").mkdir(exist_ok=True)
        Path("temp").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Verificar SOLO modelo turbo
        if not Path(turbo_model_path).exists():
            raise FileNotFoundError(f"Turbo model directory not found: {turbo_model_path}")
        
        logger.info(f"✅ Turbo model directory found: {turbo_model_path}")
        logger.info(f"📁 Normal model directory IGNORED: {model_path}")
        
        # Cargar modelos y voces
        self._load_models()
        self._load_voice_collections()
    
    def _load_models(self):
        """Carga SOLO el modelo turbo con compatibilidad multi-GPU"""
        try:
            # Verificar que el modelo turbo esté disponible
            if not Path(self.turbo_model_path).exists():
                raise FileNotFoundError(f"Turbo model directory not found: {self.turbo_model_path}")
            
            logger.info("🚀 Loading ONLY turbo CSM processor...")
            self.processor = AutoProcessor.from_pretrained(self.turbo_model_path)
            self.turbo_processor = self.processor  # Usar el mismo processor
            
            logger.info("🚀 Loading ONLY turbo CSM model (GPU-optimized)...")
            
            # GPU-specific loading configuration
            model_kwargs = {
                "use_safetensors": True,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    # Get GPU info for optimization
                    device_props = torch.cuda.get_device_properties(0)
                    compute_capability = f"{device_props.major}.{device_props.minor}"
                    
                    # Optimize loading based on GPU architecture
                    if device_props.major >= 9:  # RTX 4090 series with sm_90 support
                        logger.info("🚀 RTX 4090+ series detected - using advanced loading")
                        model_kwargs.update({
                            "device_map": "auto",
                            "torch_dtype": torch.float16,
                        })
                    elif device_props.major >= 8:  # RTX 6000 Ada
                        logger.info("⚡ RTX 6000 Ada detected - using optimized loading")
                        model_kwargs.update({
                            "device_map": self.device,
                            "torch_dtype": torch.float16,
                        })
                    else:  # Older GPUs
                        logger.info("🔧 Legacy GPU detected - using compatible loading")
                        model_kwargs.update({
                            "device_map": self.device,
                            "torch_dtype": torch.float32,
                        })
                    
                    # Remove None values
                    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
                    
                except Exception as gpu_error:
                    logger.warning(f"⚠️ GPU detection failed: {gpu_error}")
                    logger.info("🔄 Falling back to CPU mode")
                    # Force CPU fallback
                    self.device = "cpu"
                    model_kwargs.update({
                        "torch_dtype": torch.float32,
                        "device_map": "cpu"
                    })
                
            else:
                # CPU mode (either forced or no CUDA)
                logger.info("💻 Using CPU mode for model loading")
                model_kwargs.update({
                    "torch_dtype": torch.float32,
                    "device_map": "cpu"
                })
            
            try:
                # First attempt with optimized settings
                self.model = CsmForConditionalGeneration.from_pretrained(
                    self.turbo_model_path,
                    **model_kwargs
                )
                logger.info("✅ Model loaded with GPU-optimized settings")
                
            except RuntimeError as cuda_error:
                if "CUDA" in str(cuda_error) and "kernel" in str(cuda_error):
                    logger.warning(f"⚠️ CUDA kernel error detected: {cuda_error}")
                    logger.info("🔄 Falling back to compatibility mode...")
                    
                    # Fallback to most compatible settings
                    fallback_kwargs = {
                        "device_map": self.device if torch.cuda.is_available() else "cpu",
                        "torch_dtype": torch.float32,  # Most compatible
                        "use_safetensors": True,
                        "low_cpu_mem_usage": True,
                        "trust_remote_code": True
                    }
                    
                    self.model = CsmForConditionalGeneration.from_pretrained(
                        self.turbo_model_path,
                        **fallback_kwargs
                    )
                    logger.info("✅ Model loaded in compatibility mode")
                else:
                    raise cuda_error
            
            # El modelo turbo ES el modelo principal
            self.turbo_model = self.model
            
            logger.info("🚀 Applied memory optimizations (low_cpu_mem_usage)")
            logger.info("✅ Turbo CSM model loaded successfully as primary model")
            
            if torch.cuda.is_available() and self.device == "cuda":
                try:
                    gpu_info = torch.cuda.get_device_properties(0)
                    memory_gb = gpu_info.total_memory / 1024**3
                    logger.info(f"🖥️ GPU: {gpu_info.name} ({memory_gb:.1f} GB)")
                    logger.info(f"🖥️ GPU Memory Used: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
                    logger.info(f"🔧 Compute Capability: {gpu_info.major}.{gpu_info.minor}")
                    logger.info(f"🎯 Model dtype: {next(self.model.parameters()).dtype}")
                except Exception as gpu_error:
                    logger.warning(f"⚠️ Could not get GPU info: {gpu_error}")
                    logger.info(f"🎯 Model dtype: {next(self.model.parameters()).dtype}")
            else:
                logger.info("💻 Model loaded on CPU")
                logger.info(f"🎯 Model dtype: {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load turbo model: {e}")
            raise
    
    def _load_voice_collections(self):
        """Carga todas las colecciones de voces desde el directorio voices/"""
        self.voice_collections = {}
        
        for voice_dir in self.voices_dir.iterdir():
            if voice_dir.is_dir():
                try:
                    collection = self._load_voice_collection(voice_dir.name)
                    if collection:
                        self.voice_collections[voice_dir.name] = collection
                        logger.info(f"✅ Loaded voice collection: {voice_dir.name} ({len(collection.profiles)} samples)")
                except Exception as e:
                    logger.error(f"❌ Failed to load voice collection {voice_dir.name}: {e}")
        
        logger.info(f"📢 Loaded {len(self.voice_collections)} voice collections")
    
    def _load_voice_collection(self, voice_id: str) -> Optional[VoiceCollection]:
        """Carga una colección de voz específica"""
        voice_path = self.voices_dir / voice_id
        profiles_file = voice_path / "profiles.json"
        
        if not voice_path.exists() or not profiles_file.exists():
            return None
        
        try:
            with open(profiles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            profiles = []
            
            # Cargar perfiles existentes
            if 'voices' in data and isinstance(data['voices'], dict):
                # Formato legacy - un solo perfil
                profile_data = data['voices']
                profile = VoiceProfile(
                    name=profile_data.get('name', voice_id),
                    audio_path=profile_data['audio_path'],
                    transcription=profile_data['transcription'],
                    language=profile_data.get('language', 'es'),
                    quality_score=profile_data.get('quality_score', 1.0),
                    duration=profile_data['duration'],
                    sample_rate=profile_data['sample_rate'],
                    created_at=profile_data.get('created_at', datetime.now().isoformat())
                )
                profiles.append(profile)
            elif 'profiles' in data:
                # Formato nuevo - múltiples perfiles
                for profile_data in data['profiles']:
                    profile = VoiceProfile(**profile_data)
                    profiles.append(profile)
            
            # Calcular estadísticas
            total_samples = len(profiles)
            average_duration = sum(p.duration for p in profiles) / total_samples if total_samples > 0 else 0
            
            collection = VoiceCollection(
                voice_id=voice_id,
                profiles=profiles,
                total_samples=total_samples,
                average_duration=average_duration,
                created_at=data.get('created_at', datetime.now().isoformat()),
                updated_at=data.get('updated_at', datetime.now().isoformat())
            )
            
            return collection
            
        except Exception as e:
            logger.error(f"❌ Failed to load voice collection {voice_id}: {e}")
            return None
    
    def _save_voice_collection(self, voice_id: str, collection: VoiceCollection):
        """Guarda una colección de voz"""
        voice_path = self.voices_dir / voice_id
        voice_path.mkdir(exist_ok=True)
        
        profiles_file = voice_path / "profiles.json"
        
        data = {
            "voice_id": collection.voice_id,
            "profiles": [profile.dict() for profile in collection.profiles],
            "total_samples": collection.total_samples,
            "average_duration": collection.average_duration,
            "created_at": collection.created_at,
            "updated_at": datetime.now().isoformat()
        }
        
        with open(profiles_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def upload_voice_sample(
        self, 
        voice_id: str, 
        audio_file: UploadFile, 
        transcription: str = None,
        language: str = "es"
    ) -> VoiceProfile:
        """Sube una muestra de audio para una voz con normalización y validación"""
        
        # Usar el nombre del archivo como transcripción si no se proporciona
        if not transcription:
            transcription = Path(audio_file.filename).stem.replace('_', ' ').replace('-', ' ')
        
        # Crear directorio de la voz
        voice_path = self.voices_dir / voice_id
        voice_path.mkdir(exist_ok=True)
        
        # Generar nombre de archivo normalizado (siempre WAV)
        safe_name = "".join(c for c in transcription if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name[:100]  # Limitar longitud
        
        # Siempre usar extensión .wav
        audio_filename = f"{safe_name}.wav"
        audio_path = voice_path / audio_filename
        temp_path = Path("temp") / f"upload_{audio_file.filename}"
        
        try:
            # Guardar archivo temporal
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await audio_file.read()
                await f.write(content)
            
            # Cargar y analizar audio
            try:
                waveform, sample_rate = torchaudio.load(temp_path)
                original_duration = waveform.shape[1] / sample_rate
                
                logger.info(f"📊 Original audio: {original_duration:.2f}s, {sample_rate}Hz, {waveform.shape[0]} channels")
                
                # VALIDAR DURACIÓN (3-9 segundos)
                if original_duration < 3.0:
                    raise ValueError(f"Audio demasiado corto: {original_duration:.2f}s. Mínimo requerido: 3.0s")
                elif original_duration > 9.0:
                    raise ValueError(f"Audio demasiado largo: {original_duration:.2f}s. Máximo permitido: 9.0s")
                
                # NORMALIZAR AUDIO
                # 1. Convertir a mono si es estéreo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                    logger.info("🔄 Converted to mono")
                
                # 2. Resample a 24kHz si es necesario
                if sample_rate != 24000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                    waveform = resampler(waveform)
                    logger.info(f"🔄 Resampled from {sample_rate}Hz to 24000Hz")
                    sample_rate = 24000
                
                # 3. Normalizar amplitud (RMS normalization)
                rms = torch.sqrt(torch.mean(waveform**2))
                if rms > 0:
                    target_rms = 0.1  # Nivel de normalización
                    waveform = waveform * (target_rms / rms)
                    logger.info(f"🔄 Normalized RMS from {rms:.4f} to {target_rms:.4f}")
                
                # 4. Aplicar fade in/out suave para evitar clicks
                fade_samples = int(0.01 * sample_rate)  # 10ms fade
                if waveform.shape[1] > fade_samples * 2:
                    # Fade in
                    fade_in = torch.linspace(0, 1, fade_samples)
                    waveform[0, :fade_samples] *= fade_in
                    # Fade out
                    fade_out = torch.linspace(1, 0, fade_samples)
                    waveform[0, -fade_samples:] *= fade_out
                    logger.info("🔄 Applied fade in/out")
                
                # 5. Recalcular duración final
                duration = waveform.shape[1] / sample_rate
                
                # Guardar archivo normalizado en formato WAV 24kHz
                torchaudio.save(audio_path, waveform, sample_rate)
                logger.info(f"✅ Saved normalized audio: {duration:.2f}s, 24000Hz, mono")
                
            except Exception as e:
                logger.error(f"❌ Failed to process audio: {e}")
                raise ValueError(f"Error procesando audio: {str(e)}")
                
        finally:
            # Limpiar archivo temporal
            if temp_path.exists():
                temp_path.unlink()
        
        # Crear perfil
        profile = VoiceProfile(
            name=safe_name,
            audio_path=str(audio_path),
            transcription=transcription,
            language=language,
            quality_score=1.0,
            duration=duration,
            sample_rate=sample_rate,
            created_at=datetime.now().isoformat()
        )
        
        # Cargar o crear colección
        collection = self.voice_collections.get(voice_id)
        if not collection:
            collection = VoiceCollection(
                voice_id=voice_id,
                profiles=[],
                total_samples=0,
                average_duration=0.0,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
        
        # Agregar perfil a la colección
        collection.profiles.append(profile)
        collection.total_samples = len(collection.profiles)
        collection.average_duration = sum(p.duration for p in collection.profiles) / collection.total_samples
        collection.updated_at = datetime.now().isoformat()
        
        # Guardar colección
        self._save_voice_collection(voice_id, collection)
        self.voice_collections[voice_id] = collection
        
        logger.info(f"✅ Added voice sample to {voice_id}: {safe_name}")
        return profile
    
    def _split_sentences(self, text: str, max_chars: int = 220) -> List[str]:
        """
        Divide el texto en oraciones correctas para CSM usando puntuación natural.
        220 caracteres ≈ ~18 tokens de entrada → genera ~0-1s de audio.
        """
        import textwrap
        
        # 1) Separar en oraciones básicas usando puntuación
        sentences = re.split(r'(?<=[.!?¿¡])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 2) Asegurar que no exceden max_chars
        final_sentences = []
        for sentence in sentences:
            if len(sentence) <= max_chars:
                final_sentences.append(sentence)
            else:
                # Fragmentar por comas o espacios si es muy larga
                chunks = textwrap.wrap(sentence, width=max_chars, break_long_words=False)
                final_sentences.extend(chunks)
        
        logger.info(f"📝 Split text into {len(final_sentences)} sentences (avg: {len(text)/len(final_sentences):.0f} chars each)")
        return final_sentences
    
    def _enhance_prosody_markers(self, sentences: List[str]) -> str:
        """
        Combina oraciones con marcadores de prosodia mejorados para que CSM
        pueda interpretar mejor las pausas y entonación.
        """
        # Unir oraciones preservando la estructura prosódica
        enhanced_text = ""
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            
            # Agregar la oración
            enhanced_text += sentence
            
            # Agregar pausas apropiadas entre oraciones
            if i < len(sentences) - 1:  # No agregar pausa después de la última oración
                # Si la oración termina con puntuación fuerte, agregar pausa larga
                if sentence.endswith(('.', '!', '?', '¿', '¡')):
                    enhanced_text += "  "  # Doble espacio para pausa más larga
                else:
                    enhanced_text += " "   # Espacio simple para pausa corta
        
        return enhanced_text.strip()
    
    def _build_conversation(self, reference_profile: VoiceProfile, sentences: List[str]) -> List[Dict]:
        """
        Construye la conversación correcta para CSM: 
        - Primer mensaje: referencia con audio + transcripción
        - Segundo mensaje: todas las oraciones concatenadas (solo text)
        
        CSM requiere que cada mensaje (excepto el último) tenga text + audio.
        """
        try:
            # Cargar audio de referencia → np.float32 mono 24 kHz
            waveform, sample_rate = torchaudio.load(reference_profile.audio_path)
            
            # Resample a 24kHz si es necesario
            if sample_rate != 24000:
                resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                waveform = resampler(waveform)
            
            # Convertir a mono si es estéreo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Convertir a numpy float32
            ref_audio = waveform.squeeze().cpu().numpy().astype(np.float32)
            
            # Combinar oraciones con marcadores de prosodia mejorados
            # CSM puede interpretar espacios y puntuación para pausas naturales
            full_text = self._enhance_prosody_markers(sentences)
            
            # Construir conversación CSM compatible:
            # 1. Mensaje de referencia (text + audio)
            # 2. Mensaje objetivo (solo text) - último mensaje puede ser solo text
            conversation = [
                {
                    "role": "0",
                    "content": [
                        {"type": "text", "text": reference_profile.transcription},
                        {"type": "audio", "path": ref_audio}
                    ]
                },
                {
                    "role": "0",
                    "content": [{"type": "text", "text": full_text}]
                }
            ]
            
            logger.info(f"🗣️ Built CSM conversation: reference + {len(sentences)} sentences combined")
            logger.info(f"📝 Target text length: {len(full_text)} chars")
            return conversation
            
        except Exception as e:
            logger.error(f"❌ Failed to build conversation: {e}")
            raise
    
    def clone_voice_with_sentences(
        self, 
        text: str, 
        voice_id: str = None,
        sample_name: str = None,
        enable_sentence_splitting: bool = True,
        max_chars: int = 220,
        temperature: float = GENERATION_DEFAULTS["temperature"],
        max_tokens: int = GENERATION_DEFAULTS["max_tokens"],
        turbo: bool = True
    ) -> np.ndarray:
        """
        Clona voz con división inteligente en oraciones usando conversación CSM correcta.
        Una sola llamada al modelo con toda la conversación para máxima eficiencia.
        """
        try:
            # Validar que se requiere voice_id para referencia
            if not voice_id or voice_id not in self.voice_collections:
                logger.warning("⚠️ No voice_id provided or not found, falling back to normal generation")
                return self.clone_voice(
                    text=text,
                    voice_id=voice_id,
                    sample_name=sample_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    turbo=turbo
                )
            
            collection = self.voice_collections[voice_id]
            if not collection.profiles:
                logger.warning("⚠️ No voice profiles found, falling back to normal generation")
                return self.clone_voice(
                    text=text,
                    voice_id=voice_id,
                    sample_name=sample_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    turbo=turbo
                )
            
            # Si la división está deshabilitada o el texto es corto, usar método normal
            if not enable_sentence_splitting or len(text) <= max_chars:
                logger.info("🎯 Using single-pass generation (text too short or splitting disabled)")
                return self.clone_voice(
                    text=text, 
                    voice_id=voice_id, 
                    sample_name=sample_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    turbo=turbo
                )
            
            # 1) Dividir texto en oraciones
            sentences = self._split_sentences(text, max_chars)
            
            if len(sentences) <= 1:
                logger.info("🎯 Single sentence after splitting, using normal generation")
                return self.clone_voice(
                    text=sentences[0] if sentences else text, 
                    voice_id=voice_id, 
                    sample_name=sample_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    turbo=turbo
                )
            
            # 2) Obtener perfil de referencia
            reference_profile = None
            if sample_name:
                reference_profile = next((p for p in collection.profiles if p.name == sample_name), None)
            
            if not reference_profile:
                reference_profile = collection.profiles[0]  # Usar el primero disponible
            
            # 3) Construir conversación completa
            conversation = self._build_conversation(reference_profile, sentences)
            
            # 4) Procesar con CSM en una sola llamada
            logger.info(f"🎭 Generating with prosody-aware text: {len(sentences)} sentences → enhanced punctuation")
            
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
            ).to(self.device)
            
            # Asegurar que los tipos de tensor coincidan con el modelo
            model_dtype = next(self.model.parameters()).dtype
            for key, value in inputs.items():
                if hasattr(value, 'dtype') and value.dtype.is_floating_point:
                    if value.dtype != model_dtype:
                        inputs[key] = value.to(dtype=model_dtype)
            
            # 5) Generar audio con una sola llamada
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Ajustar max_tokens basado en número de oraciones
                adjusted_tokens = min(max_tokens, len(sentences) * 400)  # ~400 tokens ≈ 1s audio
                
                outputs = self.model.generate(
                    **inputs, 
                    output_audio=GENERATION_DEFAULTS["output_audio"],
                    max_new_tokens=adjusted_tokens,
                    temperature=temperature,
                    do_sample=GENERATION_DEFAULTS["do_sample"]
                )
            
            # 6) Extraer y procesar audio
            if hasattr(outputs, 'audio_values'):
                audio = outputs.audio_values
            elif isinstance(outputs, dict) and 'audio_values' in outputs:
                audio = outputs['audio_values']
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 1:
                audio = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                audio = outputs
            
            # Convertir a numpy
            if isinstance(audio, torch.Tensor):
                audio = audio.float().cpu().numpy()
            elif isinstance(audio, list):
                if len(audio) > 0:
                    audio = audio[0]
                    if isinstance(audio, torch.Tensor):
                        audio = audio.float().cpu().numpy()
                    else:
                        audio = np.array(audio, dtype=np.float32)
                else:
                    logger.warning("⚠️ Model returned empty audio, generating silence")
                    audio = np.zeros(24000, dtype=np.float32)
            else:
                audio = np.array(audio, dtype=np.float32)
            
            # Procesar audio final
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            duration = len(audio) / 24000
            
            logger.info(f"✅ Prosody-enhanced generation complete:")
            logger.info(f"   📊 Original sentences: {len(sentences)}, Duration: {duration:.2f}s")
            logger.info(f"   🎯 Reference: {voice_id}/{reference_profile.name}")
            logger.info(f"   🎭 Enhanced punctuation preserved for natural prosody")
            logger.info(f"   🚀 Single CSM conversation - optimal efficiency")
            
            return audio
            
        except Exception as e:
            logger.error(f"❌ Sentence-split voice cloning failed: {e}")
            logger.info("🔄 Falling back to single-pass generation")
            # Fallback a generación normal
            return self.clone_voice(
                text=text, 
                voice_id=voice_id, 
                sample_name=sample_name,
                temperature=temperature,
                max_tokens=max_tokens,
                turbo=turbo
            )
    
    def clone_voice(
        self, 
        text: str, 
        voice_id: str = None,
        sample_name: str = None,
        temperature: float = GENERATION_DEFAULTS["temperature"],
        max_tokens: int = GENERATION_DEFAULTS["max_tokens"],
        turbo: bool = False
    ) -> np.ndarray:
        """Clona una voz usando una muestra específica con opción turbo"""
        try:
            # Solo tenemos modelo turbo disponible
            model = self.model  # El modelo turbo es el único modelo
            processor = self.processor
            
            if turbo:
                logger.info("🚀 Using turbo model (optimized, half precision)")
            else:
                logger.info("🚀 Using turbo model as default (no normal model available)")
            
            conversation = []
            
            # Buscar muestra de referencia
            if voice_id and voice_id in self.voice_collections:
                collection = self.voice_collections[voice_id]
                
                # Buscar muestra específica o usar la primera
                target_profile = None
                if sample_name:
                    target_profile = next((p for p in collection.profiles if p.name == sample_name), None)
                
                if not target_profile and collection.profiles:
                    target_profile = collection.profiles[0]  # Usar la primera muestra
                
                if target_profile:
                    # Cargar audio de referencia
                    try:
                        waveform, sample_rate = torchaudio.load(target_profile.audio_path)
                        
                        # Resample a 24kHz si es necesario
                        if sample_rate != 24000:
                            resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                            waveform = resampler(waveform)
                        
                        # Convertir a mono si es estéreo
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                        
                        # Convert audio to match model dtype
                        model_dtype = next(model.parameters()).dtype
                        if model_dtype == torch.float16:
                            audio_data = waveform.squeeze().numpy().astype(np.float16)
                        else:
                            audio_data = waveform.squeeze().numpy().astype(np.float32)
                        
                        conversation.append({
                            "role": "0",
                            "content": [
                                {"type": "text", "text": target_profile.transcription},
                                {"type": "audio", "path": audio_data}
                            ]
                        })
                        
                        model_type = "turbo" if turbo and self.turbo_model is not None else "normal"
                        logger.info(f"🎯 Using voice reference: {voice_id}/{target_profile.name} ({model_type})")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to load reference audio: {e}")
            
            # Agregar texto a sintetizar
            conversation.append({
                "role": "0",
                "content": [{"type": "text", "text": text}]
            })
            
            # Procesar entrada
            if conversation:
                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_dict=True,
                ).to(self.device)
            else:
                # Sin contexto, usar formato simple
                formatted_text = f"[0]{text}"
                inputs = processor(formatted_text, add_special_tokens=True).to(self.device)
            
            # Ensure tensor types match the model dtype
            model_dtype = next(model.parameters()).dtype
            for key, value in inputs.items():
                if hasattr(value, 'dtype') and value.dtype.is_floating_point:
                    if value.dtype != model_dtype:
                        inputs[key] = value.to(dtype=model_dtype)
                        logger.debug(f"🔄 Converted {key} from {value.dtype} to {model_dtype}")
            
            # Generación con manejo robusto de errores CUDA
            try:
                # Standard CUDA generation
                with torch.no_grad():
                    # Clear CUDA cache before generation for stability
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    outputs = model.generate(
                        **inputs, 
                        output_audio=GENERATION_DEFAULTS["output_audio"],
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=GENERATION_DEFAULTS["do_sample"]
                    )
                
            except RuntimeError as cuda_error:
                if "CUDA" in str(cuda_error):
                    logger.warning(f"⚠️ CUDA error during generation: {cuda_error}")
                    
                    # Standard CUDA error recovery
                    logger.info("🔄 Attempting CUDA recovery...")
                    
                    # Clear cache and retry with lower memory usage
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Retry with more conservative settings
                    try:
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs, 
                                output_audio=GENERATION_DEFAULTS["output_audio"],
                                max_new_tokens=min(max_tokens, 2048),  # Reduce tokens
                                temperature=temperature,
                                do_sample=GENERATION_DEFAULTS["do_sample"],
                                use_cache=False  # Reduce memory usage
                            )
                        logger.info("✅ Generation successful after CUDA recovery")
                    except Exception as retry_error:
                        logger.error(f"❌ CUDA recovery failed: {retry_error}")
                        raise RuntimeError(f"CUDA generation failed: {cuda_error}. Recovery attempt also failed: {retry_error}")
                else:
                    raise cuda_error
            
            # Extraer y procesar audio
            if hasattr(outputs, 'audio_values'):
                audio = outputs.audio_values
            elif isinstance(outputs, dict) and 'audio_values' in outputs:
                audio = outputs['audio_values']
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 1:
                audio = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                audio = outputs
            
            # Convertir a numpy
            if isinstance(audio, torch.Tensor):
                audio = audio.float().cpu().numpy()
            elif isinstance(audio, list):
                if len(audio) > 0:
                    audio = audio[0]
                    if isinstance(audio, torch.Tensor):
                        audio = audio.float().cpu().numpy()
                    else:
                        audio = np.array(audio, dtype=np.float32)
                else:
                    logger.warning("⚠️ Model returned empty audio, generating silence")
                    audio = np.zeros(24000, dtype=np.float32)
            else:
                audio = np.array(audio, dtype=np.float32)
            
            # Procesar audio final
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            logger.info(f"✅ Generated audio shape: {audio.shape}, dtype: {audio.dtype}")
            return audio
            
        except Exception as e:
            logger.error(f"❌ Voice cloning failed: {e}")
            raise

# Production Helper Functions
async def safe_inference_call(func, *args, **kwargs):
    """
    Wrapper seguro para llamadas de inferencia con:
    - Control de concurrencia (semáforo)
    - Timeout global
    - Ejecución en thread pool
    - Logging detallado
    """
    client_info = kwargs.pop('client_info', 'unknown')
    
    async with inference_semaphore:
        try:
            logger.info(f"🚀 Starting inference for {client_info} | Semaphore: {inference_semaphore._value}/{MAX_CONCURRENT_INFERENCE}")
            
            # Execute heavy computation in thread pool to not block event loop
            result = await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=INFERENCE_TIMEOUT
            )
            
            logger.info(f"✅ Inference completed for {client_info}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"⏰ Inference timeout ({INFERENCE_TIMEOUT}s) for {client_info}")
            raise HTTPException(
                status_code=408, 
                detail=f"Inference timeout after {INFERENCE_TIMEOUT} seconds. Try with shorter text or lower max_tokens."
            )
        except Exception as e:
            logger.exception(f"❌ Inference failed for {client_info}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal inference error: {str(e)}"
            )

def get_client_info(request: Request) -> str:
    """Extract client information for logging"""
    client_ip = getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    user_agent = request.headers.get('user-agent', 'unknown')[:50]
    return f"{client_ip} | {user_agent}"

# Application state manager
class AppState:
    def __init__(self):
        self.voice_manager: Optional[CSMVoiceManager] = None
        self.startup_time: Optional[datetime] = None
        self.is_ready: bool = False

app_state = AppState()

# Production Lifecycle Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager for production"""
    # Startup
    logger.info("🚀 Starting Voice Cloning API - Production Mode")
    logger.info(f"📊 Configuration: max_concurrent={MAX_CONCURRENT_INFERENCE}, timeout={INFERENCE_TIMEOUT}s, rate_limit={RATE_LIMIT_PER_MINUTE}/min")
    
    try:
        app_state.startup_time = datetime.now()
        
        # Initialize voice manager in startup to avoid cold starts
        logger.info("🔄 Initializing CSM Voice Manager...")
        app_state.voice_manager = CSMVoiceManager()
        
        # Warm-up: load basic components
        logger.info("🔥 Warming up model components...")
        collections_count = len(app_state.voice_manager.voice_collections)
        logger.info(f"✅ Voice manager ready: {collections_count} voice collections loaded")
        
        app_state.is_ready = True
        logger.info("✅ Application startup complete - Ready for production traffic")
        
        yield  # Application runs here
        
    except Exception as e:
        logger.exception(f"❌ Startup failed: {e}")
        app_state.is_ready = False
        raise
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Voice Cloning API")
        app_state.is_ready = False
        
        # Clean GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cleared")

# FastAPI Production Configuration
app = FastAPI(
    title="🎤 Voice Cloning API - Production Ready",
    description="Production-grade voice cloning API with CSM-1B, featuring concurrency control, rate limiting, and optimized inference",
    version="4.0.0-prod",
    lifespan=lifespan
)

# Production Middleware Stack
# 1. Trusted Host Protection
if ALLOWED_HOSTS != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

# 2. CORS (should be early in middleware stack)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. GZip Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 4. Rate Limiting
if SLOWAPI_AVAILABLE and limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    logger.info(f"✅ Rate limiting enabled: {RATE_LIMIT_PER_MINUTE} requests per {RATE_LIMIT_WINDOW}s per IP")
else:
    logger.warning("⚠️ Rate limiting disabled - install slowapi for production use")

# Health check and utility endpoints

@app.get("/", response_class=HTMLResponse)
async def home():
    """Página principal mejorada"""
    return """
    <html>
        <head>
            <title>🎤 Voice Cloning API Complete - CSM-1B Turbo</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
                .container { max-width: 1000px; margin: 0 auto; padding: 40px 20px; }
                .header { background: rgba(255,255,255,0.95); padding: 30px; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
                h1 { color: #333; text-align: center; margin: 0; font-size: 2.5em; }
                .subtitle { text-align: center; color: #666; margin-top: 10px; font-size: 1.2em; }
                .section { background: rgba(255,255,255,0.95); margin: 20px 0; padding: 25px; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
                .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 8px; font-family: 'Consolas', monospace; border-left: 4px solid #007bff; }
                .method { background: #007bff; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 10px; }
                .method.post { background: #28a745; }
                .method.get { background: #17a2b8; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .status { padding: 15px; background: linear-gradient(45deg, #d4edda, #c3e6cb); border-radius: 8px; color: #155724; text-align: center; font-weight: bold; }
                .features { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px; }
                .feature { background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #eee; }
                .feature h3 { color: #333; margin-top: 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎤 Voice Cloning API - Production Ready</h1>
                    <div class="subtitle">CSM-1B Turbo • Rate Limited • Concurrency Controlled • Production Grade</div>
                </div>
                
                <div class="section">
                    <div class="status">
                        ✅ Production API Ready • Concurrency Control • Rate Limiting • Auto-scaling Ready
                    </div>
                </div>
                
                <div class="section">
                    <h2>🚀 Características Principales</h2>
                    <div class="features">
                        <div class="feature">
                            <h3>📁 Gestión por Carpetas</h3>
                            <p>Cada voz tiene su propia carpeta con múltiples muestras</p>
                        </div>
                        <div class="feature">
                            <h3>📤 Upload Inteligente</h3>
                            <p>Validación automática: 3-9s, WAV 24kHz mono normalizado</p>
                        </div>
                        <div class="feature">
                            <h3>🎯 Clonación Precisa</h3>
                            <p>Selección específica de muestras para mejor calidad</p>
                        </div>
                        <div class="feature">
                            <h3>🚀 Production Ready</h3>
                            <p>Rate limiting, concurrency control, timeouts, health checks</p>
                        </div>
                        <div class="feature">
                            <h3>📊 Análisis Completo</h3>
                            <p>Estadísticas detalladas y métricas de calidad</p>
                        </div>
                        <div class="feature">
                            <h3>⏱️ Generación Extendida</h3>
                            <p>Hasta 3 minutos de audio continuo de alta calidad</p>
                        </div>
                        <div class="feature">
                            <h3>🎭 División Inteligente</h3>
                            <p>Mejora la prosodia dividiendo texto en oraciones con mínima latencia</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📋 Endpoints de la API</h2>
                    <div class="endpoint"><span class="method get">GET</span>/health - Estado del sistema</div>
                    <div class="endpoint"><span class="method get">GET</span>/voices - Listar todas las colecciones de voces</div>
                    <div class="endpoint"><span class="method get">GET</span>/voices/{voice_id} - Detalles de una voz específica</div>
                    <div class="endpoint"><span class="method post">POST</span>/voices/{voice_id}/upload - Subir muestra de audio</div>
                    <div class="endpoint"><span class="method post">POST</span>/clone - Clonar voz con texto (con división en oraciones opcional)</div>
                    <div class="endpoint"><span class="method post">POST</span>/clone_extended - Generación extendida (hasta 3 min)</div>
                    <div class="endpoint"><span class="method post">POST</span>/clone_with_prosody - Clonación con prosodia mejorada (división inteligente)</div>
                    <div class="endpoint"><span class="method get">GET</span>/docs - Documentación interactiva</div>
                </div>
                
                <div class="section">
                    <h2>🔗 Enlaces Rápidos</h2>
                    <p><a href="/docs">📖 Documentación Interactiva (Swagger UI)</a></p>
                    <p><a href="/health">🔍 Health Check</a></p>
                    <p><a href="/voices">📢 Ver Todas las Voces</a></p>
                </div>
                
                <div class="section">
                    <h2>💡 Ejemplo de Uso</h2>
                    <div class="endpoint">
                        # Subir muestra de voz (3-9s, será normalizado a WAV 24kHz mono)<br>
                        curl -X POST 'http://localhost:7860/voices/fran-fem/upload' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'audio_file=@audio.wav' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'transcription=Hola mundo'
                    </div>
                    <div class="endpoint">
                        # Clonar voz (modo normal)<br>
                        curl -X POST 'http://localhost:7860/clone' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'text=Texto a sintetizar' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'voice_id=fran-fem'
                    </div>
                    <div class="endpoint">
                        # Clonar voz (modo turbo - más rápido)<br>
                        curl -X POST 'http://localhost:7860/clone' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'text=Texto a sintetizar' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'voice_id=fran-fem' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'turbo=true'
                    </div>
                    <div class="endpoint">
                        # Generar audio largo (hasta 3 minutos)<br>
                        curl -X POST 'http://localhost:7860/clone_extended' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'text=Texto muy largo para generar audio extendido...' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'target_duration=120' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'voice_id=fran-fem'
                    </div>
                    <div class="endpoint">
                        # Clonación con prosodia mejorada (división inteligente)<br>
                        curl -X POST 'http://localhost:7860/clone_with_prosody' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'text=Este es un texto largo. Tiene varias oraciones! ¿Mejorará la prosodia?' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'voice_id=fran-fem' \\<br>
                        &nbsp;&nbsp;&nbsp;&nbsp;-F 'max_chunk_size=100'
                    </div>
                </div>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Production health check - Simple and fast for K8s/LB"""
    if not app_state.is_ready or not app_state.voice_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Basic health indicators
        voice_count = len(app_state.voice_manager.voice_collections)
        gpu_available = torch.cuda.is_available()
        uptime = (datetime.now() - app_state.startup_time).total_seconds() if app_state.startup_time else 0
        
        return {
            "status": "healthy",
            "uptime_seconds": int(uptime),
            "voice_collections": voice_count,
            "gpu_available": gpu_available,
            "concurrent_slots": f"{inference_semaphore._value}/{MAX_CONCURRENT_INFERENCE}",
            "version": "4.0.0-prod"
        }
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check for monitoring/debugging"""
    if not app_state.is_ready or not app_state.voice_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
        
    try:
        manager = app_state.voice_manager
        gpu_available = torch.cuda.is_available()
        
        # Detailed statistics
        total_voices = len(manager.voice_collections)
        total_samples = sum(len(collection.profiles) for collection in manager.voice_collections.values())
        uptime = (datetime.now() - app_state.startup_time).total_seconds() if app_state.startup_time else 0
        
        gpu_info = {}
        if gpu_available:
            try:
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_info = {
                    "name": gpu_props.name,
                    "memory_total_gb": round(gpu_props.total_memory / 1024**3, 1),
                    "memory_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 1),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}"
                }
            except Exception:
                gpu_info = {"error": "Could not get GPU info"}
        
        return {
            "status": "healthy",
            "uptime_seconds": int(uptime),
            "startup_time": app_state.startup_time.isoformat() if app_state.startup_time else None,
            "model": {
                "loaded": manager.model is not None,
                "device": manager.device,
                "path": manager.turbo_model_path
            },
            "gpu_info": gpu_info,
            "voice_collections": total_voices,
            "total_voice_samples": total_samples,
            "concurrency": {
                "max_concurrent": MAX_CONCURRENT_INFERENCE,
                "available_slots": inference_semaphore._value,
                "inference_timeout": INFERENCE_TIMEOUT
            },
            "rate_limiting": {
                "enabled": SLOWAPI_AVAILABLE and limiter is not None,
                "per_minute": RATE_LIMIT_PER_MINUTE if SLOWAPI_AVAILABLE else None
            }
        }
    except Exception as e:
        logger.exception("Detailed health check failed")
        raise HTTPException(status_code=500, detail=f"Detailed health check failed: {str(e)}")

@app.get("/voices")
async def list_voice_collections():
    """Lista todas las colecciones de voces"""
    if not app_state.is_ready or not app_state.voice_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        manager = app_state.voice_manager
        
        collections_summary = {}
        for voice_id, collection in manager.voice_collections.items():
            collections_summary[voice_id] = {
                "total_samples": collection.total_samples,
                "average_duration": round(collection.average_duration, 2),
                "created_at": collection.created_at,
                "updated_at": collection.updated_at,
                "samples": [
                    {
                        "name": profile.name,
                        "transcription": profile.transcription,
                        "duration": round(profile.duration, 2),
                        "language": profile.language
                    }
                    for profile in collection.profiles
                ]
            }
        
        return {
            "voice_collections": collections_summary,
            "total_collections": len(collections_summary),
            "total_samples": sum(c["total_samples"] for c in collections_summary.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")

@app.get("/voices/{voice_id}")
async def get_voice_collection(voice_id: str):
    """Obtiene detalles de una colección de voz específica"""
    if not app_state.is_ready or not app_state.voice_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        manager = app_state.voice_manager
        
        if voice_id not in manager.voice_collections:
            raise HTTPException(status_code=404, detail=f"Voice collection '{voice_id}' not found")
        
        collection = manager.voice_collections[voice_id]
        return collection.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get voice collection: {str(e)}")

@app.post("/voices/{voice_id}/upload")
async def upload_voice_sample(
    request: Request,
    voice_id: str,
    audio_file: UploadFile = File(..., description="Audio file"),
    transcription: Optional[str] = Form(None, description="Audio transcription (optional - uses filename if not provided)"),
    language: str = Form("es", description="Language code")
):
    """Sube una muestra de audio para una voz específica"""
    if not app_state.is_ready or not app_state.voice_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    client_info = get_client_info(request)
    logger.info(f"📤 Upload request for voice '{voice_id}' from {client_info}")
    
    try:
        manager = app_state.voice_manager
        
        # Validar archivo de audio
        valid_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        file_extension = Path(audio_file.filename).suffix.lower()
        
        if not (audio_file.content_type and audio_file.content_type.startswith('audio/')) and file_extension not in valid_extensions:
            raise HTTPException(status_code=400, detail=f"File must be an audio file. Supported formats: {', '.join(valid_extensions)}")
        
        # Información sobre requisitos
        logger.info(f"📤 Processing upload for voice '{voice_id}': {audio_file.filename}")
        logger.info("📋 Requirements: 3-9 seconds duration, will be normalized to WAV 24kHz mono")
        
        # Subir muestra
        profile = await manager.upload_voice_sample(
            voice_id=voice_id,
            audio_file=audio_file,
            transcription=transcription,
            language=language
        )
        
        return {
            "message": f"Voice sample uploaded successfully to '{voice_id}'",
            "profile": profile.dict(),
            "collection_stats": {
                "total_samples": manager.voice_collections[voice_id].total_samples,
                "average_duration": round(manager.voice_collections[voice_id].average_duration, 2)
            }
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        # Errores de validación específicos (duración, formato, etc.)
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Voice upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice upload failed: {str(e)}")

@app.post("/clone")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute") if limiter else lambda x: x
async def clone_voice_endpoint(
    request: Request,
    text: str = Form(..., description="Text to synthesize"),
    voice_id: Optional[str] = Form(None, description="Voice collection ID"),
    sample_name: Optional[str] = Form(None, description="Specific sample name (optional)"),
    temperature: float = Form(GENERATION_DEFAULTS["temperature"], description="Sampling temperature"),
    max_tokens: int = Form(GENERATION_DEFAULTS["max_tokens"], description="Maximum tokens to generate (higher = longer audio, max ~25000 for 3min)"),
    turbo: bool = Form(False, description="Use turbo mode (optimized model for faster inference)"),
    enable_sentence_splitting: bool = Form(True, description="Enable smart sentence splitting for better prosody"),
    max_chunk_size: int = Form(200, description="Maximum characters per chunk when using sentence splitting"),
    output_format: str = Form("wav", description="Output format (wav)")
):
    """Clona una voz con el texto especificado - Production ready with concurrency control"""
    if not app_state.is_ready or not app_state.voice_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    client_info = get_client_info(request)
    logger.info(f"🎤 Clone request from {client_info}: {len(text)} chars, voice_id={voice_id}")
    
    try:
        manager = app_state.voice_manager
        
        # Validar voice_id si se proporciona
        if voice_id and voice_id not in manager.voice_collections:
            raise HTTPException(status_code=404, detail=f"Voice collection '{voice_id}' not found")
        
        # Validar max_tokens para evitar generaciones extremadamente largas
        if max_tokens > 25000:
            raise HTTPException(status_code=400, detail="max_tokens cannot exceed 25000 (approximately 3 minutes of audio)")
        
        if max_tokens < 64:
            raise HTTPException(status_code=400, detail="max_tokens must be at least 64 for meaningful audio generation")
        
        # Validar max_chunk_size
        if max_chunk_size < 50:
            raise HTTPException(status_code=400, detail="max_chunk_size must be at least 50 characters")
        if max_chunk_size > 1000:
            raise HTTPException(status_code=400, detail="max_chunk_size cannot exceed 1000 characters")
        
        # Generar audio con o sin división en oraciones (con control de concurrencia)
        if enable_sentence_splitting and len(text) > max_chunk_size:
            logger.info(f"🎵 Using sentence splitting (max {max_chunk_size} chars per sentence)")
            audio = await safe_inference_call(
                manager.clone_voice_with_sentences,
                text=text,
                voice_id=voice_id,
                sample_name=sample_name,
                enable_sentence_splitting=enable_sentence_splitting,
                max_chars=max_chunk_size,
                temperature=temperature,
                max_tokens=max_tokens,
                turbo=turbo,
                client_info=client_info
            )
        else:
            logger.info("🎯 Using single-pass generation")
            audio = await safe_inference_call(
                manager.clone_voice,
                text=text,
                voice_id=voice_id,
                sample_name=sample_name,
                temperature=temperature,
                max_tokens=max_tokens,
                turbo=turbo,
                client_info=client_info
            )
        
        # Crear nombre de archivo único
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        voice_suffix = f"_{voice_id}" if voice_id else "_default"
        sample_suffix = f"_{sample_name}" if sample_name else ""
        filename = f"cloned{voice_suffix}{sample_suffix}_{text_hash}.{output_format}"
        
        output_path = Path("outputs") / filename
        
        # Guardar audio
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
            audio_tensor = torch.from_numpy(audio)
        else:
            audio_tensor = audio.float()
        
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Use soundfile instead of torchaudio for better compatibility
        audio_numpy = audio_tensor.squeeze().numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
        import soundfile as sf
        sf.write(output_path, audio_numpy, 24000)
        
        logger.info(f"✅ Generated audio: {output_path}")
        
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")

@app.post("/clone_extended")
@limiter.limit(f"{max(RATE_LIMIT_PER_MINUTE//2, 1)}/minute") if limiter else lambda x: x
async def clone_voice_extended(
    request: Request,
    text: str = Form(..., description="Text to synthesize (can be very long)"),
    voice_id: Optional[str] = Form(None, description="Voice collection ID"),
    sample_name: Optional[str] = Form(None, description="Specific sample name (optional)"),
    target_duration: int = Form(60, description="Target duration in seconds (60-180)"),
    temperature: float = Form(GENERATION_DEFAULTS["temperature"], description="Sampling temperature"),
    turbo: bool = Form(True, description="Use turbo mode for faster generation"),
    enable_sentence_splitting: bool = Form(True, description="Enable smart sentence splitting for better prosody"),
    max_chunk_size: int = Form(150, description="Maximum characters per chunk when using sentence splitting"),
    output_format: str = Form("wav", description="Output format (wav)")
):
    """Genera audio extendido dividiendo el texto en segmentos para mayor duración"""
    if not app_state.is_ready or not app_state.voice_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    client_info = get_client_info(request)
    logger.info(f"📏 Extended clone request from {client_info}: {len(text)} chars, target={target_duration}s")
    
    try:
        manager = app_state.voice_manager
        
        # Validar voice_id si se proporciona
        if voice_id and voice_id not in manager.voice_collections:
            raise HTTPException(status_code=404, detail=f"Voice collection '{voice_id}' not found")
        
        # Validar duración objetivo
        if target_duration < 10:
            raise HTTPException(status_code=400, detail="target_duration must be at least 10 seconds")
        if target_duration > 180:
            raise HTTPException(status_code=400, detail="target_duration cannot exceed 180 seconds (3 minutes)")
        
        # Estimar tokens necesarios basado en duración objetivo
        estimated_tokens = min(int(target_duration * 400), 25000)  # ~400 tokens por segundo
        
        logger.info(f"🎯 Extended generation: target={target_duration}s, estimated_tokens={estimated_tokens}")
        
        # Generar audio usando tokens estimados con división inteligente (con control de concurrencia)
        if enable_sentence_splitting and len(text) > max_chunk_size:
            logger.info(f"🎵 Extended generation with sentence splitting (max {max_chunk_size} chars per sentence)")
            audio = await safe_inference_call(
                manager.clone_voice_with_sentences,
                text=text,
                voice_id=voice_id,
                sample_name=sample_name,
                enable_sentence_splitting=enable_sentence_splitting,
                max_chars=max_chunk_size,
                temperature=temperature,
                max_tokens=estimated_tokens,
                turbo=turbo,
                client_info=client_info
            )
        else:
            logger.info("🎯 Extended generation with single-pass")
            audio = await safe_inference_call(
                manager.clone_voice,
                text=text,
                voice_id=voice_id,
                sample_name=sample_name,
                temperature=temperature,
                max_tokens=estimated_tokens,
                turbo=turbo,
                client_info=client_info
            )
        
        # Calcular duración real
        actual_duration = len(audio) / 24000
        
        # Crear nombre de archivo único
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        voice_suffix = f"_{voice_id}" if voice_id else "_default"
        sample_suffix = f"_{sample_name}" if sample_name else ""
        filename = f"extended{voice_suffix}{sample_suffix}_{target_duration}s_{text_hash}.{output_format}"
        
        output_path = Path("outputs") / filename
        
        # Guardar audio
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
            audio_tensor = torch.from_numpy(audio)
        else:
            audio_tensor = audio.float()
        
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        audio_numpy = audio_tensor.squeeze().numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
        import soundfile as sf
        sf.write(output_path, audio_numpy, 24000)
        
        logger.info(f"✅ Extended audio generated: {output_path}")
        logger.info(f"📊 Target: {target_duration}s, Actual: {actual_duration:.2f}s, Tokens: {estimated_tokens}")
        
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=filename,
            headers={
                "X-Audio-Duration": str(actual_duration),
                "X-Target-Duration": str(target_duration),
                "X-Tokens-Used": str(estimated_tokens)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Extended voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extended voice cloning failed: {str(e)}")

@app.post("/clone_with_prosody")
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE}/minute") if limiter else lambda x: x
async def clone_voice_with_prosody(
    request: Request,
    text: str = Form(..., description="Text to synthesize with enhanced prosody"),
    voice_id: Optional[str] = Form(None, description="Voice collection ID"),
    sample_name: Optional[str] = Form(None, description="Specific sample name (optional)"),
    temperature: float = Form(GENERATION_DEFAULTS["temperature"], description="Sampling temperature"),
    max_tokens: int = Form(GENERATION_DEFAULTS["max_tokens"], description="Maximum tokens to generate"),
    turbo: bool = Form(True, description="Use turbo mode for faster generation"),
    max_chunk_size: int = Form(150, description="Maximum characters per chunk (50-300)"),
    crossfade_duration: float = Form(0.05, description="Crossfade duration between segments in seconds"),
    use_parallel_processing: bool = Form(True, description="Enable parallel chunk processing when beneficial"),
    output_format: str = Form("wav", description="Output format (wav)")
):
    """
    Endpoint especializado para clonación de voz con división inteligente en oraciones.
    Optimizado para mejor prosodia con impacto mínimo en latencia.
    """
    if not app_state.is_ready or not app_state.voice_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    client_info = get_client_info(request)
    logger.info(f"🎭 Prosody clone request from {client_info}: {len(text)} chars, chunk_size={max_chunk_size}")
    
    try:
        manager = app_state.voice_manager
        
        # Validar voice_id si se proporciona
        if voice_id and voice_id not in manager.voice_collections:
            raise HTTPException(status_code=404, detail=f"Voice collection '{voice_id}' not found")
        
        # Validaciones específicas para prosody
        if max_chunk_size < 50:
            raise HTTPException(status_code=400, detail="max_chunk_size must be at least 50 characters")
        if max_chunk_size > 300:
            raise HTTPException(status_code=400, detail="max_chunk_size cannot exceed 300 characters")
        
        if crossfade_duration < 0.01:
            raise HTTPException(status_code=400, detail="crossfade_duration must be at least 0.01 seconds")
        if crossfade_duration > 0.2:
            raise HTTPException(status_code=400, detail="crossfade_duration cannot exceed 0.2 seconds")
        
        if len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Generar audio con configuraciones de prosodia (con control de concurrencia)
        logger.info(f"🎭 Prosody-enhanced generation: {len(text)} chars, max_chunk: {max_chunk_size}")
        
        # Usar siempre división en oraciones para este endpoint
        audio = await safe_inference_call(
            manager.clone_voice_with_sentences,
            text=text,
            voice_id=voice_id,
            sample_name=sample_name,
            enable_sentence_splitting=True,
            max_chars=max_chunk_size,
            temperature=temperature,
            max_tokens=max_tokens,
            turbo=turbo,
            client_info=client_info
        )
        
        # Crear nombre de archivo único
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        voice_suffix = f"_{voice_id}" if voice_id else "_default"
        sample_suffix = f"_{sample_name}" if sample_name else ""
        filename = f"prosody{voice_suffix}{sample_suffix}_{max_chunk_size}ch_{text_hash}.{output_format}"
        
        output_path = Path("outputs") / filename
        
        # Guardar audio
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
            audio_tensor = torch.from_numpy(audio)
        else:
            audio_tensor = audio.float()
        
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        audio_numpy = audio_tensor.squeeze().numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor
        import soundfile as sf
        sf.write(output_path, audio_numpy, 24000)
        
        # Calcular estadísticas
        duration = len(audio_numpy) / 24000
        estimated_chunks = len(text) // max_chunk_size + (1 if len(text) % max_chunk_size > 0 else 0)
        
        logger.info(f"✅ Prosody-enhanced audio generated: {output_path}")
        logger.info(f"📊 Duration: {duration:.2f}s, Estimated chunks: {estimated_chunks}")
        
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=filename,
            headers={
                "X-Audio-Duration": str(duration),
                "X-Chunk-Size": str(max_chunk_size),
                "X-Estimated-Chunks": str(estimated_chunks),
                "X-Prosody-Enhanced": "true"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prosody-enhanced voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prosody-enhanced voice cloning failed: {str(e)}")

if __name__ == "__main__":
    logger.info("🎤 Voice Cloning API Complete - Starting...")
    logger.info("🔍 Checking system requirements...")
    
    # Verificar GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("⚠️ No GPU available, using CPU")
    
    # Verificar SOLO modelo turbo
    turbo_model_path = Path("./models/csm-1b-turbo")
    if turbo_model_path.exists():
        logger.info("✅ Turbo model directory found")
    else:
        logger.error("❌ Turbo model directory not found")
        sys.exit(1)
    
    try:
        logger.info("🚀 Starting server on http://0.0.0.0:7860")
        logger.info("📖 API Documentation: http://0.0.0.0:7860/docs")
        
        # Iniciar servidor
        uvicorn.run(
            "voice_api_complete:app",
            host="0.0.0.0",
            port=7860,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        traceback.print_exc()
        sys.exit(1) 
