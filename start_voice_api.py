#!/usr/bin/env python3
"""
Startup script for Voice Cloning API with automatic configuration and monitoring
"""

import os
import sys
import argparse
import logging
import asyncio
import signal
import uvicorn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from voice_cloning_optimizer import OptimizationConfig, get_optimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('voice_api.log')
    ]
)
logger = logging.getLogger(__name__)

class VoiceAPIServer:
    """Voice API server with configuration and monitoring"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.server = None
        self.optimization_config = None
        
    def load_configuration(self):
        """Load and validate configuration"""
        logger.info("Loading API configuration...")
        
        # Check for model directory
        model_path = Path("./models/sesame-csm-1b")
        if not model_path.exists():
            logger.error(f"Model directory not found: {model_path}")
            logger.info("Please download the CSM-1B model to ./models/sesame-csm-1b")
            sys.exit(1)
        
        # Check required files
        required_files = ["config.json", "tokenizer.json"]  # Adjust based on actual model files
        missing_files = []
        for file_name in required_files:
            if not (model_path / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.warning(f"Some model files may be missing: {missing_files}")
        
        # Create output directories
        output_dirs = ["outputs", "temp", "logs"]
        for dir_name in output_dirs:
            Path(dir_name).mkdir(exist_ok=True)
        
        logger.info("Configuration loaded successfully")
    
    def setup_optimization(self, gpu_optimization: bool = True, 
                         max_cache_mb: int = 4096,
                         adaptive_chunking: bool = True,
                         max_concurrent: int = 2,
                         production: bool = False,
                         gpu_memory_fraction: float = 0.85):
        """Setup optimization configuration"""
        logger.info("Setting up optimization configuration...")
        
        # Production optimizations
        if production:
            logger.info("🚀 Enabling production optimizations...")
            max_cache_mb = max(max_cache_mb, 4096)  # Minimum 4GB cache for production
            gpu_memory_fraction = min(gpu_memory_fraction, 0.9)  # Conservative GPU usage
        
        self.optimization_config = OptimizationConfig(
            enable_gpu_optimization=gpu_optimization,
            max_cache_size_mb=max_cache_mb,
            adaptive_chunk_sizing=adaptive_chunking,
            enable_mixed_precision=True,
            enable_torch_compile=False if production else True,  # Disable torch compile in production for stability
            audio_preprocessing_threads=4 if production else 2,
            max_concurrent_requests=max_concurrent,
            max_gpu_memory_fraction=gpu_memory_fraction
        )
        
        # Log optimization settings
        logger.info(f"GPU Optimization: {gpu_optimization}")
        logger.info(f"Max Cache Size: {max_cache_mb} MB")
        logger.info(f"Adaptive Chunking: {adaptive_chunking}")
        logger.info(f"Max Concurrent Requests: {max_concurrent}")
        logger.info(f"GPU Memory Fraction: {gpu_memory_fraction}")
        logger.info(f"Production Mode: {production}")
        
        # Set environment variables for optimal performance
        if production:
            os.environ['OMP_NUM_THREADS'] = '4'
            os.environ['MKL_NUM_THREADS'] = '4'
            os.environ['NUMEXPR_NUM_THREADS'] = '4'
            logger.info("🔧 Set environment variables for production")
    
    def check_system_requirements(self):
        """Check system requirements and recommendations"""
        logger.info("Checking system requirements...")
        
        import torch
        import psutil
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Available: {gpu_count} device(s), {gpu_memory:.1f} GB memory")
            
            if gpu_memory < 8:
                logger.warning("GPU memory < 8GB. Performance may be limited.")
        else:
            logger.warning("No GPU available. Using CPU (will be slower)")
        
        # Check RAM
        ram_gb = psutil.virtual_memory().total / 1024**3
        logger.info(f"System RAM: {ram_gb:.1f} GB")
        
        if ram_gb < 16:
            logger.warning("RAM < 16GB. Consider reducing cache size.")
        
        # Check disk space
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / 1024**3
        logger.info(f"Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 5:
            logger.warning("Low disk space. May affect temporary file storage.")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}. Shutting down gracefully...")
            if self.server:
                self.server.should_exit = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8000, 
                   workers: int = 1, reload: bool = False):
        """Run the API server"""
        logger.info(f"Starting Voice Cloning API server on {host}:{port}")
        
        # Configure uvicorn
        config = uvicorn.Config(
            "voice_cloning_api:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level="info",
            access_log=True,
            server_header=False,
            date_header=False
        )
        
        self.server = uvicorn.Server(config)
        
        try:
            self.server.run()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            logger.info("Server shutdown complete")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Voice Cloning API Server")
    
    # Server configuration
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Optimization configuration - Tuned for production
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU optimization")
    parser.add_argument("--cache-size", type=int, default=4096, help="Cache size in MB (default: 4096)")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive chunking")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent requests")
    
    # Performance tuning
    parser.add_argument("--production", action="store_true", help="Enable production optimizations")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.85, help="GPU memory fraction to use")
    
    # Other options
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--check-only", action="store_true", help="Only check requirements")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create server instance
    server = VoiceAPIServer(config_file=args.config)
    
    try:
        # Load configuration
        server.load_configuration()
        
        # Check system requirements
        server.check_system_requirements()
        
        if args.check_only:
            logger.info("System check completed. Exiting.")
            return
        
        # Setup optimization
        server.setup_optimization(
            gpu_optimization=not args.no_gpu,
            max_cache_mb=args.cache_size,
            adaptive_chunking=not args.no_adaptive,
            max_concurrent=args.max_concurrent,
            production=args.production,
            gpu_memory_fraction=args.gpu_memory_fraction
        )
        
        # Setup signal handlers
        server.setup_signal_handlers()
        
        # Display startup information
        logger.info("=" * 60)
        logger.info("🎤 Advanced Voice Cloning API Server")
        logger.info("=" * 60)
        logger.info(f"Server URL: http://{args.host}:{args.port}")
        logger.info(f"Health Check: http://{args.host}:{args.port}/health")
        logger.info(f"API Docs: http://{args.host}:{args.port}/docs")
        logger.info(f"Workers: {args.workers}")
        logger.info(f"Reload: {args.reload}")
        logger.info("=" * 60)
        
        # Run server
        server.run_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload
        )
        
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 