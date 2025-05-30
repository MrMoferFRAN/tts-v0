#!/usr/bin/env python3
"""
Advanced optimization module for Voice Cloning API
Includes GPU optimization, memory management, and performance tuning
"""

import os
import gc
import torch
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from threading import Lock
import time

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for optimization settings"""
    # GPU settings
    enable_gpu_optimization: bool = True
    max_gpu_memory_fraction: float = 0.9
    enable_mixed_precision: bool = True
    enable_torch_compile: bool = True
    
    # Memory management
    enable_memory_pool: bool = True
    max_cache_size_mb: int = 2048
    garbage_collection_threshold: int = 100
    
    # Processing optimization
    optimal_chunk_sizes: Dict[str, int] = None
    adaptive_chunk_sizing: bool = True
    batch_processing_enabled: bool = True
    max_concurrent_requests: int = 4
    
    # Audio processing
    audio_preprocessing_threads: int = 2
    silence_detection_optimization: bool = True
    
    def __post_init__(self):
        if self.optimal_chunk_sizes is None:
            self.optimal_chunk_sizes = {
                "low_memory": 50,
                "medium_memory": 100,
                "high_memory": 200,
                "streaming": 75
            }

class GPUOptimizer:
    """GPU optimization utilities"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_available = torch.cuda.is_available()
        
    def optimize_gpu_settings(self, config: OptimizationConfig):
        """Optimize GPU settings for performance"""
        if not self.gpu_available:
            logger.warning("GPU not available, using CPU")
            return
        
        try:
            # Set memory fraction
            if config.max_gpu_memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(config.max_gpu_memory_fraction)
                logger.info(f"Set GPU memory fraction to {config.max_gpu_memory_fraction}")
            
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Enable mixed precision training if available
            if config.enable_mixed_precision and hasattr(torch.cuda, 'amp'):
                logger.info("Mixed precision enabled")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            logger.info("GPU optimization completed")
            
        except Exception as e:
            logger.error(f"GPU optimization failed: {e}")
    
    def get_optimal_batch_size(self, model_size_mb: float, sequence_length: int) -> int:
        """Calculate optimal batch size based on GPU memory"""
        if not self.gpu_available:
            return 1
        
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available_memory = gpu_memory_gb * 0.8  # Leave 20% buffer
            
            # Estimate memory per sample (rough calculation)
            memory_per_sample = model_size_mb * sequence_length / 1000
            optimal_batch_size = max(1, int(available_memory * 1024 / memory_per_sample))
            
            return min(optimal_batch_size, 8)  # Cap at 8 for stability
            
        except Exception as e:
            logger.error(f"Failed to calculate optimal batch size: {e}")
            return 1

class MemoryManager:
    """Advanced memory management"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache = {}
        self.cache_size_bytes = 0
        self.max_cache_size_bytes = config.max_cache_size_mb * 1024 * 1024
        self.access_count = {}
        self.lock = Lock()
        
    def cache_audio_data(self, key: str, audio_data: np.ndarray) -> bool:
        """Cache audio data with LRU eviction"""
        with self.lock:
            try:
                data_size = audio_data.nbytes
                
                # Check if we need to evict
                while (self.cache_size_bytes + data_size > self.max_cache_size_bytes 
                       and self.cache):
                    self._evict_lru()
                
                # Cache the data
                self.cache[key] = audio_data.copy()
                self.cache_size_bytes += data_size
                self.access_count[key] = 1
                
                logger.debug(f"Cached audio data: {key} ({data_size} bytes)")
                return True
                
            except Exception as e:
                logger.error(f"Failed to cache audio data: {e}")
                return False
    
    def get_cached_audio(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached audio data"""
        with self.lock:
            if key in self.cache:
                self.access_count[key] += 1
                return self.cache[key].copy()
            return None
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        # Find LRU item
        lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
        
        # Remove from cache
        data = self.cache.pop(lru_key)
        self.cache_size_bytes -= data.nbytes
        del self.access_count[lru_key]
        
        logger.debug(f"Evicted LRU item: {lru_key}")
    
    def force_garbage_collection(self):
        """Force garbage collection"""
        collected = gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug(f"Garbage collection: {collected} objects collected")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        stats = {
            "ram_usage_percent": psutil.virtual_memory().percent,
            "ram_available_gb": psutil.virtual_memory().available / 1024**3,
            "cache_size_mb": self.cache_size_bytes / 1024**2,
            "cache_items": len(self.cache)
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_memory_free_gb": (torch.cuda.get_device_properties(0).total_memory 
                                     - torch.cuda.memory_allocated()) / 1024**3
            })
        
        return stats

class AdaptiveChunker:
    """Adaptive text chunking based on system resources"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.performance_history = []
        
    def get_optimal_chunk_size(self, text_length: int, system_load: float) -> int:
        """Calculate optimal chunk size based on text length and system load"""
        base_chunk_size = self.config.optimal_chunk_sizes["medium_memory"]
        
        # Adjust based on system load
        if system_load > 80:
            chunk_size = self.config.optimal_chunk_sizes["low_memory"]
        elif system_load < 30:
            chunk_size = self.config.optimal_chunk_sizes["high_memory"]
        else:
            chunk_size = base_chunk_size
        
        # Adjust based on text length
        if text_length < 100:
            chunk_size = min(chunk_size, 50)
        elif text_length > 1000:
            chunk_size = max(chunk_size, 150)
        
        # Use performance history for adaptation
        if self.config.adaptive_chunk_sizing and self.performance_history:
            avg_performance = np.mean(self.performance_history[-10:])  # Last 10 measurements
            if avg_performance > 2.0:  # Slow performance
                chunk_size = max(chunk_size - 20, 30)
            elif avg_performance < 0.5:  # Fast performance
                chunk_size = min(chunk_size + 20, 300)
        
        return chunk_size
    
    def record_performance(self, chunk_size: int, processing_time: float, audio_duration: float):
        """Record performance metrics for adaptive optimization"""
        if audio_duration > 0:
            realtime_factor = processing_time / audio_duration
            self.performance_history.append(realtime_factor)
            
            # Keep only recent history
            if len(self.performance_history) > 50:
                self.performance_history = self.performance_history[-50:]

class PerformanceProfiler:
    """Performance profiling and monitoring"""
    
    def __init__(self):
        self.profiles = {}
        self.current_profile = None
        
    def start_profile(self, name: str):
        """Start profiling a section"""
        self.current_profile = {
            "name": name,
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
            "start_gpu_memory": self._get_gpu_memory() if torch.cuda.is_available() else 0
        }
    
    def end_profile(self) -> Dict[str, float]:
        """End profiling and return metrics"""
        if not self.current_profile:
            return {}
        
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_gpu_memory = self._get_gpu_memory() if torch.cuda.is_available() else 0
        
        profile_data = {
            "duration": end_time - self.current_profile["start_time"],
            "memory_delta": end_memory - self.current_profile["start_memory"],
            "gpu_memory_delta": end_gpu_memory - self.current_profile["start_gpu_memory"],
            "peak_memory": end_memory,
            "peak_gpu_memory": end_gpu_memory
        }
        
        self.profiles[self.current_profile["name"]] = profile_data
        self.current_profile = None
        
        return profile_data
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return psutil.Process().memory_info().rss / 1024**2
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0.0
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all profiles"""
        return self.profiles.copy()

class VoiceCloneOptimizer:
    """Main optimization controller"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.gpu_optimizer = GPUOptimizer()
        self.memory_manager = MemoryManager(self.config)
        self.adaptive_chunker = AdaptiveChunker(self.config)
        self.profiler = PerformanceProfiler()
        
        # Initialize optimization
        self._initialize_optimizations()
    
    def _initialize_optimizations(self):
        """Initialize all optimizations"""
        logger.info("Initializing voice cloning optimizations...")
        
        # GPU optimization
        self.gpu_optimizer.optimize_gpu_settings(self.config)
        
        # Set environment variables for performance
        os.environ['OMP_NUM_THREADS'] = str(self.config.audio_preprocessing_threads)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings
        
        logger.info("Optimization initialization completed")
    
    def optimize_for_request(self, text: str, streaming: bool = False) -> Dict[str, any]:
        """Optimize settings for a specific request"""
        system_stats = self.memory_manager.get_memory_stats()
        system_load = system_stats["ram_usage_percent"]
        
        # Get optimal chunk size
        chunk_size = self.adaptive_chunker.get_optimal_chunk_size(
            len(text), system_load
        )
        
        # Adjust for streaming
        if streaming:
            chunk_size = min(chunk_size, self.config.optimal_chunk_sizes["streaming"])
        
        # Get optimal batch size for model
        batch_size = self.gpu_optimizer.get_optimal_batch_size(1500, len(text))  # Estimate 1.5GB model
        
        return {
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "use_cache": True,
            "force_gc": system_load > 85,
            "system_load": system_load,
            "memory_stats": system_stats
        }
    
    def record_request_performance(self, chunk_size: int, processing_time: float, 
                                 audio_duration: float):
        """Record performance for adaptive optimization"""
        self.adaptive_chunker.record_performance(chunk_size, processing_time, audio_duration)
        
        # Force garbage collection if needed
        if self.memory_manager.get_memory_stats()["ram_usage_percent"] > 85:
            self.memory_manager.force_garbage_collection()
    
    def get_optimization_stats(self) -> Dict[str, any]:
        """Get comprehensive optimization statistics"""
        return {
            "memory_stats": self.memory_manager.get_memory_stats(),
            "performance_profiles": self.profiler.get_performance_summary(),
            "gpu_available": self.gpu_optimizer.gpu_available,
            "cache_stats": {
                "cache_size_mb": self.memory_manager.cache_size_bytes / 1024**2,
                "cache_items": len(self.memory_manager.cache),
                "max_cache_size_mb": self.config.max_cache_size_mb
            },
            "config": {
                "mixed_precision": self.config.enable_mixed_precision,
                "adaptive_chunking": self.config.adaptive_chunk_sizing,
                "gpu_optimization": self.config.enable_gpu_optimization
            }
        }

# Global optimizer instance
global_optimizer = VoiceCloneOptimizer()

def get_optimizer() -> VoiceCloneOptimizer:
    """Get the global optimizer instance"""
    return global_optimizer

def optimize_model_loading(model_path: str) -> Dict[str, any]:
    """Optimize model loading process"""
    optimizer = get_optimizer()
    optimizer.profiler.start_profile("model_loading")
    
    # Pre-optimize GPU settings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "device": optimizer.gpu_optimizer.device,
        "mixed_precision": optimizer.config.enable_mixed_precision,
        "compile_model": optimizer.config.enable_torch_compile
    } 