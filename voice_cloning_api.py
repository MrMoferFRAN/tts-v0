#!/usr/bin/env python3
"""
Robust Voice Cloning API with streaming, performance monitoring, and advanced features
"""

import asyncio
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from typing import List, Optional, AsyncGenerator, Dict, Any
import logging

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
from threading import Lock
import psutil
import GPUtil

# Import voice cloning components
from voice_cloning.voice_clone import VoiceCloner
from voice_cloning_optimizer import get_optimizer, optimize_model_loading, OptimizationConfig
from voice_manager import get_voice_manager, initialize_voices, VoiceProfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    start_time: float = 0.0
    end_time: float = 0.0
    processing_time: float = 0.0
    audio_duration: float = 0.0
    realtime_factor: float = 0.0  # processing_time / audio_duration
    gpu_memory_used: float = 0.0
    cpu_usage: float = 0.0
    ram_usage: float = 0.0
    text_length: int = 0
    chunk_count: int = 0
    tokens_per_second: float = 0.0
    optimization_stats: Dict[str, Any] = None

class VoiceCloneRequest(BaseModel):
    """Voice cloning request model"""
    text: str = Field(..., description="Text to synthesize")
    voice_name: Optional[str] = Field(None, description="Name of voice profile to use")
    reference_text: Optional[str] = Field(None, description="Reference audio transcript")
    speaker_id: str = Field("0", description="Speaker ID")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    chunk_size: Optional[int] = Field(None, ge=50, le=500, description="Text chunk size for processing (auto-optimized if None)")
    remove_silence: bool = Field(True, description="Remove excessive silence")
    streaming: bool = Field(False, description="Enable streaming response")
    max_silence_duration: float = Field(0.5, description="Max silence duration in seconds")
    use_optimization: bool = Field(True, description="Enable automatic optimization")

class BatchVoiceCloneRequest(BaseModel):
    """Batch voice cloning request"""
    texts: List[str] = Field(..., description="List of texts to synthesize")
    voice_name: Optional[str] = Field(None, description="Name of voice profile to use")
    reference_text: Optional[str] = Field(None, description="Reference audio transcript")
    speaker_id: str = Field("0", description="Speaker ID")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Generation temperature")
    chunk_size: Optional[int] = Field(None, ge=50, le=500, description="Text chunk size for processing (auto-optimized if None)")
    remove_silence: bool = Field(True, description="Remove excessive silence")
    max_silence_duration: float = Field(0.5, description="Max silence duration in seconds")
    use_optimization: bool = Field(True, description="Enable automatic optimization")

class VoiceCloneResponse(BaseModel):
    """Voice cloning response model"""
    success: bool
    audio_url: Optional[str] = None
    performance_metrics: Dict[str, Any]
    processing_info: Dict[str, Any]
    optimization_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class TextChunker:
    """Advanced text chunking for optimal processing"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 100, overlap: int = 10) -> List[str]:
        """
        Chunk text intelligently at sentence boundaries
        """
        import re
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap from previous
                if overlap > 0 and chunks:
                    words = current_chunk.split()
                    overlap_text = " ".join(words[-overlap:]) if len(words) >= overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add remaining text
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks

class AudioProcessor:
    """Advanced audio processing utilities"""
    
    @staticmethod
    def remove_silence(audio: np.ndarray, sample_rate: int = 24000, 
                      max_silence_duration: float = 0.5) -> np.ndarray:
        """
        Remove excessive silence from audio with optimization
        """
        optimizer = get_optimizer()
        optimizer.profiler.start_profile("silence_removal")
        
        try:
            # Detect silence using librosa
            silence_threshold = 0.01  # Adjust based on your needs
            frame_length = int(sample_rate * 0.025)  # 25ms frames
            hop_length = frame_length // 4
            
            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Find non-silent frames
            non_silent_frames = rms > silence_threshold
            
            # Convert frame indices to sample indices
            non_silent_samples = []
            for i, is_sound in enumerate(non_silent_frames):
                start_sample = i * hop_length
                end_sample = min(start_sample + hop_length, len(audio))
                if is_sound:
                    non_silent_samples.extend(range(start_sample, end_sample))
            
            if not non_silent_samples:
                return audio  # Return original if all silent
            
            # Extract non-silent audio
            cleaned_audio = audio[non_silent_samples]
            
            return cleaned_audio
        finally:
            optimizer.profiler.end_profile()
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
        """
        Normalize audio to target LUFS
        """
        # Simple normalization - can be enhanced with pyloudnorm for proper LUFS
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio * 0.8  # Prevent clipping

class PerformanceMonitor:
    """System performance monitoring"""
    
    @staticmethod
    def get_gpu_memory() -> float:
        """Get GPU memory usage"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024**3  # GB
        except:
            pass
        return 0.0
    
    @staticmethod
    def get_system_metrics() -> Dict[str, float]:
        """Get system performance metrics"""
        return {
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().percent,
            "gpu_memory": PerformanceMonitor.get_gpu_memory()
        }

class VoiceCloneService:
    """Main voice cloning service with caching and optimization"""
    
    def __init__(self):
        self.cloner: Optional[VoiceCloner] = None
        self.reference_audio_cache: Dict[str, np.ndarray] = {}
        self.lock = Lock()
        self.chunker = TextChunker()
        self.audio_processor = AudioProcessor()
        self.monitor = PerformanceMonitor()
        self.optimizer = get_optimizer()
        self.voice_manager = get_voice_manager()
        
    async def initialize(self):
        """Initialize the voice cloner with optimization"""
        logger.info("Initializing Voice Cloning Service with optimization...")
        
        with self.lock:
            if self.cloner is None:
                # Get optimization settings for model loading
                optimization_settings = optimize_model_loading("./models/sesame-csm-1b")
                
                # Initialize cloner with optimized settings
                self.cloner = VoiceCloner(
                    model_path="./models/sesame-csm-1b",
                    device=optimization_settings["device"]
                )
                
                # Initialize voice profiles
                initialize_voices()
                
        logger.info("Voice Cloning Service initialized successfully with optimization")
    
    def _resolve_voice_reference(self, request: VoiceCloneRequest, reference_audio: Optional[UploadFile] = None) -> tuple:
        """Resolve voice reference from voice name or uploaded file"""
        voice_profile = None
        reference_audio_path = None
        reference_text = request.reference_text
        
        # Check if voice_name is provided
        if request.voice_name:
            voice_profile = self.voice_manager.get_voice(request.voice_name)
            if voice_profile:
                reference_audio_path = voice_profile.audio_path
                if not reference_text:  # Use profile transcription if not provided
                    reference_text = voice_profile.transcription
                logger.info(f"Using voice profile '{request.voice_name}': {voice_profile.audio_path}")
            else:
                logger.warning(f"Voice profile '{request.voice_name}' not found, falling back to uploaded audio")
        
        return voice_profile, reference_audio_path, reference_text

    def _get_performance_metrics(self, start_time: float, text: str, 
                                audio_duration: float, chunk_count: int, 
                                optimization_stats: Dict = None) -> PerformanceMetrics:
        """Calculate performance metrics with optimization data"""
        end_time = time.time()
        processing_time = end_time - start_time
        system_metrics = self.monitor.get_system_metrics()
        
        return PerformanceMetrics(
            start_time=start_time,
            end_time=end_time,
            processing_time=processing_time,
            audio_duration=audio_duration,
            realtime_factor=processing_time / audio_duration if audio_duration > 0 else 0,
            gpu_memory_used=system_metrics["gpu_memory"],
            cpu_usage=system_metrics["cpu_usage"],
            ram_usage=system_metrics["ram_usage"],
            text_length=len(text),
            chunk_count=chunk_count,
            tokens_per_second=len(text.split()) / processing_time if processing_time > 0 else 0,
            optimization_stats=optimization_stats
        )
    
    async def clone_voice(self, request: VoiceCloneRequest, 
                         reference_audio: Optional[UploadFile] = None) -> VoiceCloneResponse:
        """
        Clone voice with advanced features and optimization
        """
        start_time = time.time()
        optimization_info = {}
        
        try:
            # Get optimization settings
            if request.use_optimization:
                optimization_settings = self.optimizer.optimize_for_request(
                    request.text, request.streaming
                )
                optimization_info = optimization_settings
                
                # Use optimized chunk size if not specified
                if request.chunk_size is None:
                    request.chunk_size = optimization_settings["chunk_size"]
                    
                # Force garbage collection if recommended
                if optimization_settings.get("force_gc", False):
                    self.optimizer.memory_manager.force_garbage_collection()
            else:
                request.chunk_size = request.chunk_size or 100
            
            # Resolve voice reference (from profile or uploaded file)
            voice_profile, reference_audio_path, reference_text = self._resolve_voice_reference(request, reference_audio)
            reference_audio_key = None
            
            # If no voice profile, handle uploaded audio
            if not voice_profile and reference_audio:
                # Create cache key for reference audio
                reference_audio_key = f"ref_{hash(await reference_audio.read())}"
                await reference_audio.seek(0)  # Reset file pointer
                
                # Check cache first
                cached_audio = self.optimizer.memory_manager.get_cached_audio(reference_audio_key)
                
                if cached_audio is None:
                    # Save uploaded file temporarily
                    reference_audio_path = f"temp_reference_{uuid.uuid4().hex}.wav"
                    with open(reference_audio_path, "wb") as f:
                        content = await reference_audio.read()
                        f.write(content)
                    
                    # Cache the audio for future use
                    audio_data, _ = librosa.load(reference_audio_path, sr=24000)
                    self.optimizer.memory_manager.cache_audio_data(reference_audio_key, audio_data)
                else:
                    # Use cached audio
                    reference_audio_path = f"temp_cached_{uuid.uuid4().hex}.wav"
                    sf.write(reference_audio_path, cached_audio, 24000)
            
            # Chunk the text for processing
            chunks = self.chunker.chunk_text(request.text, request.chunk_size)
            logger.info(f"Text chunked into {len(chunks)} pieces (chunk_size: {request.chunk_size})")
            
            # Generate audio for each chunk
            audio_segments = []
            total_duration = 0.0
            
            for i, chunk in enumerate(chunks):
                chunk_start_time = time.time()
                logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
                
                # Generate audio for this chunk
                chunk_output = f"temp_chunk_{uuid.uuid4().hex}.wav"
                
                if reference_audio_path and reference_text:
                    # Use voice cloning
                    self.cloner.clone_voice_from_file(
                        reference_audio=reference_audio_path,
                        reference_transcript=reference_text,
                        target_text=chunk,
                        output_path=chunk_output,
                        speaker_id=request.speaker_id
                    )
                else:
                    # Use simple TTS
                    self.cloner.simple_generate(
                        text=chunk,
                        output_path=chunk_output,
                        speaker_id=request.speaker_id
                    )
                
                # Load and process the generated audio
                audio, sr = librosa.load(chunk_output, sr=24000)
                
                # Remove silence if requested
                if request.remove_silence:
                    audio = self.audio_processor.remove_silence(
                        audio, sr, request.max_silence_duration
                    )
                
                # Normalize audio
                audio = self.audio_processor.normalize_audio(audio)
                
                audio_segments.append(audio)
                chunk_duration = len(audio) / sr
                total_duration += chunk_duration
                
                # Record performance for optimization
                chunk_processing_time = time.time() - chunk_start_time
                if request.use_optimization:
                    self.optimizer.record_request_performance(
                        request.chunk_size, chunk_processing_time, chunk_duration
                    )
                
                # Clean up temporary file
                if os.path.exists(chunk_output):
                    os.remove(chunk_output)
            
            # Concatenate all audio segments
            if audio_segments:
                final_audio = np.concatenate(audio_segments)
            else:
                raise HTTPException(status_code=500, detail="No audio generated")
            
            # Save final audio
            output_path = f"outputs/cloned_voice_{uuid.uuid4().hex}.wav"
            os.makedirs("outputs", exist_ok=True)
            sf.write(output_path, final_audio, 24000)
            
            # Calculate performance metrics
            optimization_stats = self.optimizer.get_optimization_stats() if request.use_optimization else None
            metrics = self._get_performance_metrics(
                start_time, request.text, total_duration, len(chunks), optimization_stats
            )
            
            # Clean up temporary reference audio (only if not from voice profile)
            if not voice_profile and reference_audio_path and os.path.exists(reference_audio_path) and "temp_" in reference_audio_path:
                os.remove(reference_audio_path)
            
            return VoiceCloneResponse(
                success=True,
                audio_url=output_path,
                performance_metrics=asdict(metrics),
                processing_info={
                    "chunks_processed": len(chunks),
                    "total_audio_duration": total_duration,
                    "average_chunk_size": len(request.text) // len(chunks) if chunks else 0,
                    "silence_removed": request.remove_silence,
                    "optimization_enabled": request.use_optimization,
                    "voice_profile_used": voice_profile.name if voice_profile else None,
                    "cache_hit": reference_audio_key and self.optimizer.memory_manager.get_cached_audio(reference_audio_key) is not None
                },
                optimization_info=optimization_info
            )
            
        except Exception as e:
            logger.error(f"Error in voice cloning: {str(e)}")
            return VoiceCloneResponse(
                success=False,
                error=str(e),
                performance_metrics={},
                processing_info={},
                optimization_info=optimization_info
            )
    
    async def stream_voice_clone(self, request: VoiceCloneRequest, 
                                reference_audio: Optional[UploadFile] = None) -> AsyncGenerator[bytes, None]:
        """
        Stream voice cloning with real-time chunks and optimization
        """
        try:
            # Get optimization settings for streaming
            if request.use_optimization:
                optimization_settings = self.optimizer.optimize_for_request(
                    request.text, streaming=True
                )
                if request.chunk_size is None:
                    request.chunk_size = optimization_settings["chunk_size"]
            else:
                request.chunk_size = request.chunk_size or 75  # Default for streaming
            
            # Process reference audio if provided
            reference_audio_path = None
            if reference_audio:
                reference_audio_path = f"temp_reference_{uuid.uuid4().hex}.wav"
                with open(reference_audio_path, "wb") as f:
                    content = await reference_audio.read()
                    f.write(content)
            
            # Chunk the text
            chunks = self.chunker.chunk_text(request.text, request.chunk_size)
            
            for i, chunk in enumerate(chunks):
                chunk_start_time = time.time()
                logger.info(f"Streaming chunk {i+1}/{len(chunks)}")
                
                # Generate audio for this chunk
                chunk_output = f"temp_stream_chunk_{uuid.uuid4().hex}.wav"
                
                if reference_audio_path and request.reference_text:
                    self.cloner.clone_voice_from_file(
                        reference_audio=reference_audio_path,
                        reference_transcript=request.reference_text,
                        target_text=chunk,
                        output_path=chunk_output,
                        speaker_id=request.speaker_id
                    )
                else:
                    self.cloner.simple_generate(
                        text=chunk,
                        output_path=chunk_output,
                        speaker_id=request.speaker_id
                    )
                
                # Process and stream the audio
                audio, sr = librosa.load(chunk_output, sr=24000)
                
                if request.remove_silence:
                    audio = self.audio_processor.remove_silence(audio, sr)
                
                audio = self.audio_processor.normalize_audio(audio)
                
                # Record performance for streaming optimization
                chunk_processing_time = time.time() - chunk_start_time
                chunk_duration = len(audio) / sr
                if request.use_optimization:
                    self.optimizer.record_request_performance(
                        request.chunk_size, chunk_processing_time, chunk_duration
                    )
                
                # Convert to bytes for streaming
                buffer = io.BytesIO()
                sf.write(buffer, audio, sr, format='WAV')
                buffer.seek(0)
                
                yield buffer.read()
                
                # Clean up
                if os.path.exists(chunk_output):
                    os.remove(chunk_output)
                
                # Adaptive delay based on performance
                delay = 0.05 if chunk_processing_time < chunk_duration else 0.1
                await asyncio.sleep(delay)
            
            # Clean up reference audio
            if reference_audio_path and os.path.exists(reference_audio_path):
                os.remove(reference_audio_path)
                
        except Exception as e:
            logger.error(f"Error in streaming: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Global service instance
voice_service = VoiceCloneService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await voice_service.initialize()
    yield
    # Shutdown
    pass

# FastAPI app
app = FastAPI(
    title="Advanced Voice Cloning API",
    description="Robust voice cloning with streaming, performance monitoring, and advanced features",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    metrics = voice_service.monitor.get_system_metrics()
    return {
        "status": "healthy",
        "system_metrics": metrics,
        "model_loaded": voice_service.cloner is not None
    }

@app.post("/clone-voice", response_model=VoiceCloneResponse)
async def clone_voice_endpoint(
    text: str = Form(...),
    voice_name: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    speaker_id: str = Form("0"),
    temperature: float = Form(0.7),
    chunk_size: Optional[int] = Form(None),
    remove_silence: bool = Form(True),
    streaming: bool = Form(False),
    max_silence_duration: float = Form(0.5),
    use_optimization: bool = Form(True),
    reference_audio: Optional[UploadFile] = File(None)
):
    """
    Clone voice with advanced features
    """
    request = VoiceCloneRequest(
        text=text,
        voice_name=voice_name,
        reference_text=reference_text,
        speaker_id=speaker_id,
        temperature=temperature,
        chunk_size=chunk_size,
        remove_silence=remove_silence,
        streaming=streaming,
        max_silence_duration=max_silence_duration,
        use_optimization=use_optimization
    )
    return await voice_service.clone_voice(request, reference_audio)

@app.post("/clone-voice-stream")
async def stream_voice_clone_endpoint(
    text: str = Form(...),
    voice_name: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    speaker_id: str = Form("0"),
    temperature: float = Form(0.7),
    chunk_size: Optional[int] = Form(None),
    remove_silence: bool = Form(True),
    streaming: bool = Form(True),
    max_silence_duration: float = Form(0.5),
    use_optimization: bool = Form(True),
    reference_audio: Optional[UploadFile] = File(None)
):
    """
    Stream voice cloning in real-time chunks
    """
    request = VoiceCloneRequest(
        text=text,
        voice_name=voice_name,
        reference_text=reference_text,
        speaker_id=speaker_id,
        temperature=temperature,
        chunk_size=chunk_size,
        remove_silence=remove_silence,
        streaming=streaming,
        max_silence_duration=max_silence_duration,
        use_optimization=use_optimization
    )
    
    if not request.streaming:
        raise HTTPException(status_code=400, detail="Streaming must be enabled")
    
    return StreamingResponse(
        voice_service.stream_voice_clone(request, reference_audio),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=streamed_voice.wav"}
    )

@app.post("/batch-clone-voice")
async def batch_clone_voice_endpoint(
    request: BatchVoiceCloneRequest = Depends(),
    reference_audio: Optional[UploadFile] = File(None),
    background_tasks: BackgroundTasks = None
):
    """
    Batch voice cloning for multiple texts
    """
    results = []
    
    for i, text in enumerate(request.texts):
        voice_request = VoiceCloneRequest(
            text=text,
            voice_name=request.voice_name,
            reference_text=request.reference_text,
            speaker_id=request.speaker_id,
            temperature=request.temperature,
            chunk_size=request.chunk_size,
            remove_silence=request.remove_silence,
            max_silence_duration=request.max_silence_duration
        )
        
        result = await voice_service.clone_voice(voice_request, reference_audio)
        results.append({
            "index": i,
            "text": text,
            "result": result
        })
    
    return {
        "success": True,
        "total_processed": len(results),
        "results": results
    }

@app.get("/performance-stats")
async def get_performance_stats():
    """
    Get current system performance statistics with optimization data
    """
    base_stats = {
        "system_metrics": voice_service.monitor.get_system_metrics(),
        "model_info": {
            "device": str(voice_service.cloner.device) if voice_service.cloner else "unknown",
            "model_loaded": voice_service.cloner is not None
        }
    }
    
    # Add optimization statistics
    if hasattr(voice_service, 'optimizer'):
        optimization_stats = voice_service.optimizer.get_optimization_stats()
        base_stats.update({
            "optimization_stats": optimization_stats,
            "cache_efficiency": {
                "cache_hit_ratio": "calculated_on_demand",  # Would need request tracking
                "memory_savings": optimization_stats.get("cache_stats", {}).get("cache_size_mb", 0)
            }
        })
    
    return base_stats

@app.get("/optimization-config")
async def get_optimization_config():
    """
    Get current optimization configuration
    """
    if hasattr(voice_service, 'optimizer'):
        return {
            "config": {
                "gpu_optimization_enabled": voice_service.optimizer.config.enable_gpu_optimization,
                "mixed_precision": voice_service.optimizer.config.enable_mixed_precision,
                "adaptive_chunking": voice_service.optimizer.config.adaptive_chunk_sizing,
                "max_cache_size_mb": voice_service.optimizer.config.max_cache_size_mb,
                "optimal_chunk_sizes": voice_service.optimizer.config.optimal_chunk_sizes,
                "max_concurrent_requests": voice_service.optimizer.config.max_concurrent_requests
            },
            "current_stats": voice_service.optimizer.get_optimization_stats()
        }
    else:
        return {"error": "Optimizer not available"}

@app.post("/optimize-settings")
async def update_optimization_settings(
    enable_gpu_optimization: Optional[bool] = None,
    max_cache_size_mb: Optional[int] = None,
    adaptive_chunking: Optional[bool] = None,
    max_concurrent_requests: Optional[int] = None
):
    """
    Update optimization settings dynamically
    """
    if not hasattr(voice_service, 'optimizer'):
        raise HTTPException(status_code=500, detail="Optimizer not available")
    
    updated_settings = {}
    config = voice_service.optimizer.config
    
    if enable_gpu_optimization is not None:
        config.enable_gpu_optimization = enable_gpu_optimization
        updated_settings["enable_gpu_optimization"] = enable_gpu_optimization
        
        # Re-apply GPU optimization
        if enable_gpu_optimization:
            voice_service.optimizer.gpu_optimizer.optimize_gpu_settings(config)
    
    if max_cache_size_mb is not None:
        config.max_cache_size_mb = max_cache_size_mb
        voice_service.optimizer.memory_manager.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        updated_settings["max_cache_size_mb"] = max_cache_size_mb
    
    if adaptive_chunking is not None:
        config.adaptive_chunk_sizing = adaptive_chunking
        updated_settings["adaptive_chunking"] = adaptive_chunking
    
    if max_concurrent_requests is not None:
        config.max_concurrent_requests = max_concurrent_requests
        updated_settings["max_concurrent_requests"] = max_concurrent_requests
    
    return {
        "success": True,
        "updated_settings": updated_settings,
        "current_config": {
            "gpu_optimization_enabled": config.enable_gpu_optimization,
            "max_cache_size_mb": config.max_cache_size_mb,
            "adaptive_chunking": config.adaptive_chunk_sizing,
            "max_concurrent_requests": config.max_concurrent_requests
        }
    }

@app.post("/clear-cache")
async def clear_audio_cache():
    """
    Clear the audio cache to free up memory
    """
    if hasattr(voice_service, 'optimizer'):
        cache_size_before = voice_service.optimizer.memory_manager.cache_size_bytes / 1024**2
        cache_items_before = len(voice_service.optimizer.memory_manager.cache)
        
        # Clear cache
        voice_service.optimizer.memory_manager.cache.clear()
        voice_service.optimizer.memory_manager.access_count.clear()
        voice_service.optimizer.memory_manager.cache_size_bytes = 0
        
        # Force garbage collection
        voice_service.optimizer.memory_manager.force_garbage_collection()
        
        return {
            "success": True,
            "cache_cleared": {
                "size_mb_freed": cache_size_before,
                "items_removed": cache_items_before
            },
            "current_memory": voice_service.optimizer.memory_manager.get_memory_stats()
        }
    else:
        raise HTTPException(status_code=500, detail="Optimizer not available")

@app.get("/chunk-size-recommendation")
async def get_chunk_size_recommendation(
    text: str,
    streaming: bool = False,
    target_realtime_factor: float = 1.0
):
    """
    Get recommended chunk size for a specific text and configuration
    """
    if not hasattr(voice_service, 'optimizer'):
        return {"recommended_chunk_size": 100}  # Default fallback
    
    # Get current system load
    system_stats = voice_service.optimizer.memory_manager.get_memory_stats()
    system_load = system_stats["ram_usage_percent"]
    
    # Get recommended chunk size
    recommended_size = voice_service.optimizer.adaptive_chunker.get_optimal_chunk_size(
        len(text), system_load
    )
    
    # Adjust for streaming
    if streaming:
        recommended_size = min(recommended_size, 
                             voice_service.optimizer.config.optimal_chunk_sizes["streaming"])
    
    return {
        "recommended_chunk_size": recommended_size,
        "text_length": len(text),
        "system_load": system_load,
        "streaming_mode": streaming,
        "reasoning": {
            "base_recommendation": recommended_size,
            "system_load_factor": "high" if system_load > 80 else "medium" if system_load > 30 else "low",
            "text_length_factor": "short" if len(text) < 100 else "long" if len(text) > 1000 else "medium"
        }
    }

@app.get("/voices")
async def list_voices():
    """List all available voice profiles"""
    vm = get_voice_manager()
    voices = vm.list_voices()
    stats = vm.get_voice_stats()
    
    return {
        "voices": voices,
        "total_voices": len(voices),
        "voice_stats": stats,
        "profiles": {name: vm.get_voice(name).to_dict() for name in voices}
    }

@app.post("/voices/{voice_name}")
async def add_voice_profile(
    voice_name: str,
    transcription: str,
    language: str = "es",
    audio_file: UploadFile = File(...)
):
    """Add a new voice profile"""
    vm = get_voice_manager()
    
    try:
        # Save uploaded audio temporarily
        temp_audio = f"temp_voice_{uuid.uuid4().hex}{Path(audio_file.filename).suffix}"
        with open(temp_audio, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Add voice profile
        success = vm.add_voice(
            name=voice_name,
            audio_path=temp_audio,
            transcription=transcription,
            language=language,
            copy_file=True
        )
        
        # Clean up temp file
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        if success:
            profile = vm.get_voice(voice_name)
            return {
                "success": True,
                "message": f"Voice profile '{voice_name}' added successfully",
                "profile": profile.to_dict()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add voice profile")
    
    except Exception as e:
        # Clean up temp file in case of error
        if 'temp_audio' in locals() and os.path.exists(temp_audio):
            os.remove(temp_audio)
        raise HTTPException(status_code=500, detail=f"Error adding voice profile: {str(e)}")

@app.delete("/voices/{voice_name}")
async def remove_voice_profile(voice_name: str):
    """Remove a voice profile"""
    vm = get_voice_manager()
    
    success = vm.remove_voice(voice_name)
    if success:
        return {
            "success": True,
            "message": f"Voice profile '{voice_name}' removed successfully"
        }
    else:
        raise HTTPException(status_code=404, detail=f"Voice profile '{voice_name}' not found")

@app.get("/voices/{voice_name}")
async def get_voice_profile(voice_name: str):
    """Get details of a specific voice profile"""
    vm = get_voice_manager()
    profile = vm.get_voice(voice_name)
    
    if profile:
        return {
            "success": True,
            "profile": profile.to_dict()
        }
    else:
        raise HTTPException(status_code=404, detail=f"Voice profile '{voice_name}' not found")

if __name__ == "__main__":
    uvicorn.run(
        "voice_cloning_api:app",
        host="0.0.0.0",
        port=7860,  # Changed to 7860
        reload=False,  # Set to True for development
        workers=1  # Single worker for GPU models
    ) 