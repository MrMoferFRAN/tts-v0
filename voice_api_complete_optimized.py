#!/usr/bin/env python3
"""
Voice Cloning API Completa - CSM-1B (OPTIMIZED)
Versión con control de pronunciación, prosodia y velocidad mejorada
Compatible con RTX 4090 / 6000 Ada
"""

import os
import sys
import logging
import traceback
import json
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import tempfile
import shutil

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
from transformers import CsmForConditionalGeneration, AutoProcessor
import numpy as np
from pydantic import BaseModel

# =========================================================
# 1. CUDA COMPATIBILITY (igual que tu versión original)
# =========================================================

def setup_cuda_compatibility():
    """Configura las variables de entorno para máxima compatibilidad CUDA"""
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("TORCH_USE_CUDA_DSA", "0")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:True")
    os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
    os.environ.setdefault("NO_TORCH_COMPILE", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        print(f"🖥️ GPU: {device_props.name}")
        print(f"🔧 Compute Capability: {device_props.major}.{device_props.minor}")
        print(f"💾 Memory: {device_props.total_memory / 1024**3:.1f} GB")
    else:
        print("💻 No CUDA available, using CPU mode")

    return torch.cuda.is_available()

cuda_available = setup_cuda_compatibility()

# =========================================================
# 2. LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/voice_api.log", mode="a") if Path("logs").exists() else logging.NullHandler(),
    ],
)
logger = logging.getLogger(__name__)

# =========================================================
# 3. MODEL & VOICE DATA CLASSES
# =========================================================

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

# =========================================================
# 4. CSM VOICE MANAGER
# =========================================================

class CSMVoiceManager:
    """Gestor de voces CSM-1B con controles avanzados de generación"""

    def __init__(
        self,
        model_path: str = "./models/sesame-csm-1b",
        turbo_model_path: str = "./models/csm-1b-turbo",
        voices_dir: str = "./voices",
        dtype_preference: str = os.getenv("CSM_PRECISION", "int8"),  # "int8", "fp16" o "fp32"
    ):
        self.model_path = model_path
        self.turbo_model_path = turbo_model_path
        self.voices_dir = Path(voices_dir)
        self.dtype_preference = dtype_preference.lower()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️ Device: {self.device}")

        # Config por defecto para generación (puede sobreescribirse en cada llamada)
        self.default_gen_cfg = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,  # 0 = desactivar top‑k y solo usar nucleus
            "repetition_penalty": 1.2,
            "depth_decoder_do_sample": False,
            "depth_decoder_temperature": 0.4,
        }

        self.model = None
        self.processor = None
        self._load_models()
        self._ensure_dirs()
        self._load_voice_collections()

    # -----------------------------------------------------
    # 4.1 Utils
    # -----------------------------------------------------
    def _ensure_dirs(self):
        Path("outputs").mkdir(exist_ok=True)
        Path("temp").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        self.voices_dir.mkdir(exist_ok=True)

    # -----------------------------------------------------
    # 4.2 Model Loading
    # -----------------------------------------------------
    def _load_models(self):
        logger.info("🚀 Loading CSM‑1B turbo model…")
        if not Path(self.turbo_model_path).exists():
            raise FileNotFoundError(f"Turbo model not found at {self.turbo_model_path}")

        torch_dtype = {
            "int8": torch.int8,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(self.dtype_preference, torch.int8)

        # Si preferencia int8 => use_safetensors + load_in_8bit
        extra_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.dtype_preference == "int8":
            extra_kwargs.update({"load_in_8bit": True})
        else:
            extra_kwargs.update({"torch_dtype": torch_dtype})

        self.processor = AutoProcessor.from_pretrained(self.turbo_model_path)
        self.model = CsmForConditionalGeneration.from_pretrained(
            self.turbo_model_path,
            device_map="auto" if self.device == "cuda" else "cpu",
            **extra_kwargs,
        )
        logger.info("✅ CSM model loaded")

    # -----------------------------------------------------
    # 4.3 Voice collections (idéntico a tu código salvo minimos cambios)
    # -----------------------------------------------------
    def _load_voice_collections(self):
        self.voice_collections: Dict[str, VoiceCollection] = {}
        for voice_dir in self.voices_dir.iterdir():
            if not voice_dir.is_dir():
                continue
            profiles_file = voice_dir / "profiles.json"
            if not profiles_file.exists():
                continue
            with open(profiles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            profiles = [VoiceProfile(**p) for p in data.get("profiles", [])]
            collection = VoiceCollection(
                voice_id=data.get("voice_id", voice_dir.name),
                profiles=profiles,
                total_samples=len(profiles),
                average_duration=sum(p.duration for p in profiles) / len(profiles) if profiles else 0.0,
                created_at=data.get("created_at", datetime.now().isoformat()),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
            )
            self.voice_collections[collection.voice_id] = collection
        logger.info(f"📢 Loaded {len(self.voice_collections)} voice collections")

    # -----------------------------------------------------
    # 4.4  Core generation
    # -----------------------------------------------------
    def clone_voice(
        self,
        text: str,
        voice_id: Optional[str] = None,
        sample_name: Optional[str] = None,
        **gen_overrides,
    ) -> np.ndarray:
        """Genera audio con controles avanzados de sampling"""
        # 1) Merge config (overrides > defaults)
        gen_cfg = {**self.default_gen_cfg, **gen_overrides}

        # 2) Build conversation list
        conversation = []
        if voice_id and voice_id in self.voice_collections and self.voice_collections[voice_id].profiles:
            profile = self.voice_collections[voice_id].profiles[0]
            # (Opcional: elegir por sample_name)
            if sample_name:
                profile = next((p for p in self.voice_collections[voice_id].profiles if p.name == sample_name), profile)

            waveform, sr = torchaudio.load(profile.audio_path)
            if sr != 24000:
                waveform = torchaudio.transforms.Resample(sr, 24000)(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(0, keepdim=True)
            conversation.append(
                {
                    "role": "0",
                    "content": [
                        {"type": "text", "text": profile.transcription},
                        {"type": "audio", "path": waveform.squeeze().numpy()},
                    ],
                }
            )

        conversation.append({"role": "0", "content": [{"type": "text", "text": text}]})
        inputs = self.processor.apply_chat_template(conversation, tokenize=True, return_dict=True).to(self.device)

        # 3) Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                output_audio=True,
                do_sample=True,
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                top_k=gen_cfg["top_k"],
                repetition_penalty=gen_cfg["repetition_penalty"],
                depth_decoder_do_sample=gen_cfg["depth_decoder_do_sample"],
                depth_decoder_temperature=gen_cfg["depth_decoder_temperature"],
            )

        audio = outputs.audio_values if hasattr(outputs, "audio_values") else outputs[1]
        audio = torch.as_tensor(audio, dtype=torch.float32).cpu().numpy()
        if audio.ndim > 1:
            audio = audio.flatten()
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        return audio

    # -----------------------------------------------------
    # 4.5 Extended generation (segmentación)
    # -----------------------------------------------------
    def clone_voice_extended(
        self,
        text: str,
        target_duration: int = 60,
        **kwargs,
    ) -> np.ndarray:
        """Divide el texto en frases y las sintetiza por separado para mayor estabilidad."""
        # División básica por puntuación
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        audios: List[np.ndarray] = []
        fade = np.linspace(0, 1, int(0.01 * 24000))  # 10 ms fade

        for sent in sentences:
            if not sent:
                continue
            audio = self.clone_voice(sent, **kwargs)
            # Fade in/out para cada segmento
            if audio.shape[0] > fade.shape[0] * 2:
                audio[: fade.shape[0]] *= fade
                audio[-fade.shape[0] :] *= fade[::-1]
            audios.append(audio)

        if not audios:
            raise ValueError("Texto vacío tras segmentación")

        final_audio = np.concatenate(audios)
        # Ajustar duración si target_duration explícito
        if target_duration:
            desired_len = int(target_duration * 24000)
            if final_audio.shape[0] > desired_len:
                final_audio = final_audio[:desired_len]
            else:
                # Rellenar con silencio si sobra
                pad = np.zeros(desired_len - final_audio.shape[0], dtype=final_audio.dtype)
                final_audio = np.concatenate([final_audio, pad])
        return final_audio

# =========================================================
# 5. GLOBAL MANAGER INSTANCE
# =========================================================
voice_manager: Optional[CSMVoiceManager] = None

def get_voice_manager():
    global voice_manager
    if voice_manager is None:
        voice_manager = CSMVoiceManager()
    return voice_manager

# =========================================================
# 6. FASTAPI APP
# =========================================================
app = FastAPI(
    title="🎤 Voice Cloning API Complete - Optimized",
    description="API de clonación de voz con controles avanzados de pronunciación y prosodia",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 6.1 Health
# ---------------------------------------------------------
@app.get("/health")
async def health_check():
    mgr = get_voice_manager()
    return {"status": "ok", "voices": len(mgr.voice_collections)}

# ---------------------------------------------------------
# 6.2 Clone endpoints (simplificados)
# ---------------------------------------------------------
@app.post("/clone")
async def clone_endpoint(
    text: str = Form(...),
    voice_id: Optional[str] = Form(None),
    sample_name: Optional[str] = Form(None),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    repetition_penalty: float = Form(1.2),
    depth_decoder_do_sample: bool = Form(False),
    depth_decoder_temperature: float = Form(0.4),
    output_format: str = Form("wav"),
):
    mgr = get_voice_manager()
    audio = mgr.clone_voice(
        text,
        voice_id=voice_id,
        sample_name=sample_name,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        depth_decoder_do_sample=depth_decoder_do_sample,
        depth_decoder_temperature=depth_decoder_temperature,
    )

    filename = f"cloned_{hashlib.md5(text.encode()).hexdigest()[:8]}.{output_format}"
    out_path = Path("outputs") / filename
    import soundfile as sf

    sf.write(out_path, audio.astype(np.float32), 24000)
    return FileResponse(out_path, media_type="audio/wav", filename=filename)

@app.post("/clone_extended")
async def clone_extended_endpoint(
    text: str = Form(...),
    voice_id: Optional[str] = Form(None),
    sample_name: Optional[str] = Form(None),
    target_duration: int = Form(60),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    repetition_penalty: float = Form(1.2),
    depth_decoder_do_sample: bool = Form(False),
    depth_decoder_temperature: float = Form(0.4),
    output_format: str = Form("wav"),
):
    mgr = get_voice_manager()
    audio = mgr.clone_voice_extended(
        text,
        voice_id=voice_id,
        sample_name=sample_name,
        target_duration=target_duration,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        depth_decoder_do_sample=depth_decoder_do_sample,
        depth_decoder_temperature=depth_decoder_temperature,
    )
    filename = f"extended_{hashlib.md5(text.encode()).hexdigest()[:8]}.{output_format}"
    out_path = Path("outputs") / filename
    import soundfile as sf

    sf.write(out_path, audio.astype(np.float32), 24000)
    return FileResponse(out_path, media_type="audio/wav", filename=filename)

# =========================================================
# 7. MAIN
# =========================================================
if __name__ == "__main__":
    try:
        get_voice_manager()
        uvicorn.run("voice_api_complete_optimized:app", host="0.0.0.0", port=7860, log_level="info")
    except Exception as e:
        logger.error(f"❌ Failed to launch API: {e}")
        traceback.print_exc()
