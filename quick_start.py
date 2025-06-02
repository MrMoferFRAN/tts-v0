#!/usr/bin/env python3
"""
Voice Cloning API usando CSM-1B nativo de Transformers
Configurado para máximo rendimiento en NVIDIA A100
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import tempfile
import shutil

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
from transformers import CsmForConditionalGeneration, AutoProcessor
import numpy as np

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/voice_api.log', mode='a') if Path('logs').exists() else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuración del entorno
os.environ.setdefault('NO_TORCH_COMPILE', '1')
os.environ.setdefault('HF_TOKEN', '|==>REMOVED')

class CSMVoiceCloner:
    """Clonador de voz usando CSM-1B nativo"""
    
    def __init__(self, model_path: str = "./models/sesame-csm-1b"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.voice_profiles = {}
        
        logger.info(f"🎤 Initializing CSM Voice Cloner")
        logger.info(f"📁 Model path: {model_path}")
        logger.info(f"🖥️ Device: {self.device}")
        
        # Verificar archivos del modelo
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        model_file = Path(model_path) / "model.safetensors"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Cargar modelo y processor
        self._load_model()
        self._load_voice_profiles()
    
    def _load_model(self):
        """Carga el modelo y processor CSM-1B"""
        try:
            logger.info("📥 Loading CSM processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            logger.info("📥 Loading CSM model...")
            self.model = CsmForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map=self.device,
                torch_dtype=torch.float32  # Usar float32 para evitar problemas de tipos mixtos
            )
            
            logger.info("✅ CSM model loaded successfully")
            
            if torch.cuda.is_available():
                gpu_info = torch.cuda.get_device_properties(0)
                memory_gb = gpu_info.total_memory / 1024**3
                logger.info(f"🖥️ GPU: {gpu_info.name} ({memory_gb:.1f} GB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load CSM model: {e}")
            raise
    
    def _load_voice_profiles(self):
        """Carga perfiles de voz desde el directorio voices/"""
        voices_dir = Path("voices")
        if not voices_dir.exists():
            logger.warning("⚠️ Voices directory not found")
            return
        
        audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}
        
        for audio_file in voices_dir.iterdir():
            if audio_file.suffix.lower() in audio_extensions:
                voice_name = audio_file.stem
                
                try:
                    # Cargar audio
                    waveform, sample_rate = torchaudio.load(audio_file)
                    
                    # Resample a 24kHz si es necesario
                    if sample_rate != 24000:
                        resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                        waveform = resampler(waveform)
                    
                    # Convertir a mono si es estéreo
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    
                    # Buscar transcripción
                    transcript_file = audio_file.with_suffix('.txt')
                    if transcript_file.exists():
                        transcript = transcript_file.read_text().strip()
                    else:
                        # Usar el nombre del archivo como transcript básico
                        transcript = voice_name.replace('_', ' ').replace('-', ' ')
                    
                    self.voice_profiles[voice_name] = {
                        'audio_path': str(audio_file),
                        'waveform': waveform.squeeze().numpy(),
                        'transcript': transcript,
                        'sample_rate': 24000
                    }
                    
                    logger.info(f"✅ Loaded voice profile: {voice_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load voice {audio_file}: {e}")
        
        logger.info(f"📢 Loaded {len(self.voice_profiles)} voice profiles")
    
    def clone_voice(
        self, 
        text: str, 
        voice_name: str = None,
        context_audio: np.ndarray = None,
        context_text: str = None,
        temperature: float = 0.8,
        max_tokens: int = 512
    ) -> np.ndarray:
        """
        Clona una voz usando CSM-1B
        
        Args:
            text: Texto a sintetizar
            voice_name: Nombre del perfil de voz a usar
            context_audio: Audio de contexto directo
            context_text: Texto de contexto directo
            temperature: Temperatura de muestreo
            max_tokens: Máximo de tokens a generar
            
        Returns:
            Audio sintetizado como array numpy
        """
        try:
            # Preparar contexto
            conversation = []
            
            if voice_name and voice_name in self.voice_profiles:
                profile = self.voice_profiles[voice_name]
                conversation.append({
                    "role": "0",
                    "content": [
                        {"type": "text", "text": profile['transcript']},
                        {"type": "audio", "path": profile['waveform']}
                    ]
                })
            elif context_audio is not None and context_text:
                conversation.append({
                    "role": "0", 
                    "content": [
                        {"type": "text", "text": context_text},
                        {"type": "audio", "path": context_audio}
                    ]
                })
            
            # Agregar el texto a sintetizar
            conversation.append({
                "role": "0",
                "content": [{"type": "text", "text": text}]
            })
            
            # Procesar entrada
            if conversation:
                inputs = self.processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_dict=True,
                ).to(self.device)
            else:
                # Sin contexto, usar formato simple
                formatted_text = f"[0]{text}"
                inputs = self.processor(formatted_text, add_special_tokens=True).to(self.device)
            
            # Generar audio
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    output_audio=True,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True
                )
            
            # Extraer audio de los outputs del modelo CSM
            if hasattr(outputs, 'audio_values'):
                # Output con formato estructurado
                audio = outputs.audio_values
            elif isinstance(outputs, dict) and 'audio_values' in outputs:
                # Output como diccionario
                audio = outputs['audio_values']
            elif isinstance(outputs, (list, tuple)) and len(outputs) > 1:
                # Output como tupla/lista, el audio suele estar en el segundo elemento
                audio = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                # Usar directamente el output
                audio = outputs
            
            # Convertir a numpy y asegurar formato correcto
            if isinstance(audio, torch.Tensor):
                # Convertir a float32 antes de numpy para evitar problemas de dtype
                audio = audio.float().cpu().numpy()
            elif isinstance(audio, list):
                # El modelo puede devolver una lista de tensors
                if len(audio) > 0:
                    audio = audio[0]
                    if isinstance(audio, torch.Tensor):
                        audio = audio.float().cpu().numpy()
                    else:
                        audio = np.array(audio, dtype=np.float32)
                else:
                    # Lista vacía, crear audio de silencio
                    logger.warning("⚠️ Model returned empty audio, generating silence")
                    audio = np.zeros(24000, dtype=np.float32)  # 1 segundo de silencio
            else:
                # Convertir a numpy con dtype explícito
                audio = np.array(audio, dtype=np.float32)
            
            # Asegurar que es un array 1D y en el rango correcto
            if len(audio.shape) > 1:
                audio = audio.flatten()
            
            # Normalizar audio si es necesario
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            logger.info(f"✅ Generated audio shape: {audio.shape}, dtype: {audio.dtype}")
            return audio
            
        except Exception as e:
            logger.error(f"❌ Voice cloning failed: {e}")
            raise

# Inicializar clonador global
cloner = None

def get_cloner():
    """Obtiene la instancia global del clonador"""
    global cloner
    if cloner is None:
        cloner = CSMVoiceCloner()
    return cloner

# Configurar FastAPI
app = FastAPI(
    title="🎤 Voice Cloning API - CSM-1B",
    description="API de clonación de voz usando CSM-1B nativo de Transformers",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Inicialización del servidor"""
    logger.info("🚀 Starting Voice Cloning API server...")
    
    # Crear directorios necesarios
    for directory in ['outputs', 'temp', 'logs']:
        Path(directory).mkdir(exist_ok=True)
    
    # Inicializar clonador
    try:
        get_cloner()
        logger.info("✅ Voice Cloning API ready")
    except Exception as e:
        logger.error(f"❌ Failed to initialize cloner: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def home():
    """Página principal"""
    return """
    <html>
        <head>
            <title>🎤 Voice Cloning API - CSM-1B</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; text-align: center; }
                .api-section { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
                .endpoint { background: #e9ecef; padding: 10px; margin: 10px 0; border-radius: 5px; font-family: monospace; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .status { padding: 10px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎤 Voice Cloning API - CSM-1B</h1>
                
                <div class="status">
                    ✅ API is running and ready to use!
                </div>
                
                <div class="api-section">
                    <h2>📋 Available Endpoints</h2>
                    <div class="endpoint">GET /health - Health check</div>
                    <div class="endpoint">GET /voices - List available voice profiles</div>
                    <div class="endpoint">POST /clone-voice - Clone voice with text</div>
                    <div class="endpoint">POST /upload-voice - Upload new voice profile</div>
                    <div class="endpoint">GET /docs - Interactive API documentation</div>
                </div>
                
                <div class="api-section">
                    <h2>🔗 Quick Links</h2>
                    <p><a href="/docs">📖 Interactive API Documentation (Swagger UI)</a></p>
                    <p><a href="/health">🔍 Health Check</a></p>
                    <p><a href="/voices">📢 Available Voices</a></p>
                </div>
                
                <div class="api-section">
                    <h2>🎯 Example Usage</h2>
                    <p><strong>List voices:</strong></p>
                    <div class="endpoint">curl http://localhost:7860/voices</div>
                    
                    <p><strong>Clone voice:</strong></p>
                    <div class="endpoint">curl -X POST 'http://localhost:7860/clone-voice' \\<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;-F 'text=Hello world' \\<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;-F 'voice_name=voices' \\<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;-F 'temperature=0.7'</div>
                </div>
            </div>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        cloner = get_cloner()
        gpu_available = torch.cuda.is_available()
        
        return {
            "status": "healthy",
            "model_loaded": cloner.model is not None,
            "processor_loaded": cloner.processor is not None,
            "gpu_available": gpu_available,
            "voice_profiles": len(cloner.voice_profiles),
            "device": cloner.device
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/voices")
async def list_voices():
    """Lista perfiles de voz disponibles"""
    try:
        cloner = get_cloner()
        
        voices_info = {}
        for name, profile in cloner.voice_profiles.items():
            voices_info[name] = {
                "transcript": profile['transcript'],
                "duration_seconds": len(profile['waveform']) / profile['sample_rate']
            }
        
        return {
            "voices": voices_info,
            "total": len(voices_info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list voices: {str(e)}")

@app.post("/clone-voice")
async def clone_voice_endpoint(
    text: str = Form(..., description="Text to synthesize"),
    voice_name: Optional[str] = Form(None, description="Voice profile name"),
    temperature: float = Form(0.8, description="Sampling temperature"),
    max_tokens: int = Form(512, description="Maximum tokens to generate"),
    context_audio: Optional[UploadFile] = File(None, description="Context audio file"),
    context_text: Optional[str] = Form(None, description="Context text transcript")
):
    """Clona una voz con el texto especificado"""
    try:
        cloner = get_cloner()
        
        # Procesar audio de contexto si se proporciona
        context_audio_array = None
        if context_audio:
            # Guardar archivo temporal
            temp_path = Path("temp") / f"context_{context_audio.filename}"
            
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await context_audio.read()
                await f.write(content)
            
            try:
                # Cargar y procesar audio
                waveform, sample_rate = torchaudio.load(temp_path)
                
                if sample_rate != 24000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 24000)
                    waveform = resampler(waveform)
                
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                context_audio_array = waveform.squeeze().numpy()
                
            finally:
                # Limpiar archivo temporal
                if temp_path.exists():
                    temp_path.unlink()
        
        # Generar audio
        audio = cloner.clone_voice(
            text=text,
            voice_name=voice_name,
            context_audio=context_audio_array,
            context_text=context_text,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Guardar audio
        output_path = Path("outputs") / f"cloned_{hash(text + str(voice_name))}_{np.random.randint(1000, 9999)}.wav"
        
        # Convertir audio a tensor y guardar
        if isinstance(audio, np.ndarray):
            # Asegurar que el audio está en float32
            audio = audio.astype(np.float32)
            audio_tensor = torch.from_numpy(audio)
        else:
            audio_tensor = audio.float()  # Convertir a float32
        
        # Asegurar que tiene la forma correcta (1, N) para torchaudio
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Guardar con dtype específico para evitar errores
        torchaudio.save(output_path, audio_tensor, 24000)
        
        logger.info(f"✅ Generated audio: {output_path}")
        
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=f"cloned_voice_{voice_name or 'custom'}.wav"
        )
        
    except Exception as e:
        logger.error(f"❌ Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")

@app.post("/upload-voice")
async def upload_voice_profile(
    name: str = Form(..., description="Voice profile name"),
    audio_file: UploadFile = File(..., description="Audio file"),
    transcript: str = Form(..., description="Audio transcript")
):
    """Sube un nuevo perfil de voz"""
    try:
        # Validar archivo de audio
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Guardar archivo
        voices_dir = Path("voices")
        voices_dir.mkdir(exist_ok=True)
        
        audio_path = voices_dir / f"{name}.wav"
        transcript_path = voices_dir / f"{name}.txt"
        
        # Guardar audio
        async with aiofiles.open(audio_path, 'wb') as f:
            content = await audio_file.read()
            await f.write(content)
        
        # Guardar transcripción
        async with aiofiles.open(transcript_path, 'w') as f:
            await f.write(transcript)
        
        # Recargar perfiles
        cloner = get_cloner()
        cloner._load_voice_profiles()
        
        logger.info(f"✅ Uploaded voice profile: {name}")
        
        return {
            "message": f"Voice profile '{name}' uploaded successfully",
            "audio_path": str(audio_path),
            "transcript_path": str(transcript_path)
        }
        
    except Exception as e:
        logger.error(f"❌ Voice upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice upload failed: {str(e)}")

if __name__ == "__main__":
    logger.info("🎤 Voice Cloning API - Quick Start")
    logger.info("🔍 Checking system requirements...")
    
    # Verificar GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("⚠️ No GPU available, using CPU")
    
    # Verificar modelo
    model_path = Path("./models/sesame-csm-1b")
    if model_path.exists():
        logger.info("✅ Model directory found")
    else:
        logger.error("❌ Model directory not found")
        sys.exit(1)
    
    try:
        # Inicializar sistema
        logger.info("🎤 Setting up voice profiles...")
        cloner = get_cloner()
        
        logger.info(f"Loaded {len(cloner.voice_profiles)} voice profiles")
        
        # Verificar archivo de referencia
        reference_voice = "voices/Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3"
        if Path(reference_voice).exists():
            logger.info(f"✅ Found reference audio: {reference_voice}")
        else:
            logger.warning("⚠️ Reference audio not found")
        
        logger.info("🚀 Starting server on http://0.0.0.0:7860")
        logger.info("📖 API Documentation: http://0.0.0.0:7860/docs")
        
        # Iniciar servidor
        uvicorn.run(
            "quick_start:app",
            host="0.0.0.0",
            port=7860,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        traceback.print_exc()
        sys.exit(1)
