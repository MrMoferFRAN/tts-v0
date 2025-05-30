#!/usr/bin/env python3
"""
Quick Start Script for Voice Cloning API - Optimized Configuration
Specially configured for port 7860 with 'voices' profile
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_voices():
    """Setup default voice profiles"""
    logger.info("🎤 Setting up voice profiles...")
    
    from voice_manager import get_voice_manager
    
    vm = get_voice_manager()
    
    # Check for the existing audio file
    reference_audio = "voices/Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3"
    
    if Path(reference_audio).exists():
        logger.info(f"✅ Found reference audio: {reference_audio}")
        
        # Add 'voices' profile
        success = vm.add_voice(
            name="voices",
            audio_path=reference_audio,
            transcription="Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo.",
            language="es",
            copy_file=False  # Keep original file location
        )
        
        if success:
            logger.info("✅ Voice profile 'voices' configured successfully")
            
            # Show voice stats
            stats = vm.get_voice_stats()
            profile = vm.get_voice("voices")
            logger.info(f"📊 Voice Quality Score: {profile.quality_score:.2f}")
            logger.info(f"📊 Audio Duration: {profile.duration:.2f}s")
            
            return True
        else:
            logger.error("❌ Failed to setup voice profile")
            return False
    else:
        logger.error(f"❌ Reference audio not found: {reference_audio}")
        logger.info("💡 Please ensure the audio file is in the current directory")
        return False

def check_system():
    """Quick system check"""
    logger.info("🔍 Checking system requirements...")
    
    try:
        import torch
        import librosa
        import fastapi
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("⚠️  No GPU detected - will use CPU (slower)")
        
        # Check model directory
        model_dir = Path("./models/sesame-csm-1b")
        if model_dir.exists():
            logger.info("✅ Model directory found")
        else:
            logger.error("❌ Model directory not found: ./models/sesame-csm-1b")
            return False
        
        return True
    
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def start_optimized_server():
    """Start server with optimized configuration"""
    logger.info("🚀 Starting Voice Cloning API with optimized configuration...")
    
    # Import after checking dependencies
    from start_voice_api import VoiceAPIServer
    
    # Create server with optimal settings
    server = VoiceAPIServer()
    
    try:
        # Load configuration
        server.load_configuration()
        
        # Setup optimization for maximum performance
        server.setup_optimization(
            gpu_optimization=True,
            max_cache_mb=6144,  # 6GB cache for optimal performance
            adaptive_chunking=True,
            max_concurrent=3,   # Allow 3 concurrent requests
            production=True,    # Enable production optimizations
            gpu_memory_fraction=0.85
        )
        
        # Check system requirements
        server.check_system_requirements()
        
        # Setup signal handlers
        server.setup_signal_handlers()
        
        # Display configuration
        logger.info("=" * 60)
        logger.info("🎤 VOICE CLONING API - OPTIMIZED CONFIGURATION")
        logger.info("=" * 60)
        logger.info("🌐 Server URL: http://0.0.0.0:7860")
        logger.info("🔍 Health Check: http://0.0.0.0:7860/health")
        logger.info("📚 API Docs: http://0.0.0.0:7860/docs")
        logger.info("🎭 Voice Profiles: http://0.0.0.0:7860/voices")
        logger.info("=" * 60)
        logger.info("🔧 OPTIMIZATION SETTINGS:")
        logger.info("   • GPU Optimization: ENABLED")
        logger.info("   • Cache Size: 6GB")
        logger.info("   • Adaptive Chunking: ENABLED")
        logger.info("   • Max Concurrent: 3 requests")
        logger.info("   • Production Mode: ENABLED")
        logger.info("   • Voice Profile: 'voices' ready")
        logger.info("=" * 60)
        logger.info("")
        logger.info("💡 QUICK TEST COMMANDS:")
        logger.info("   # Test with voice profile:")
        logger.info("   curl -X POST 'http://localhost:7860/clone-voice' \\")
        logger.info("        -F 'text=Hola, esto es una prueba de clonación de voz' \\")
        logger.info("        -F 'voice_name=voices' \\")
        logger.info("        -F 'temperature=0.7'")
        logger.info("")
        logger.info("   # List available voices:")
        logger.info("   curl http://localhost:7860/voices")
        logger.info("")
        logger.info("🎯 Ready for high-performance voice cloning!")
        logger.info("=" * 60)
        
        # Run server
        server.run_server(
            host="0.0.0.0",
            port=7860,
            workers=1,
            reload=False
        )
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)

def main():
    """Main function"""
    logger.info("🎤 Voice Cloning API - Quick Start")
    
    # Check system
    if not check_system():
        logger.error("❌ System check failed")
        sys.exit(1)
    
    # Setup voices
    if not setup_voices():
        logger.error("❌ Voice setup failed")
        sys.exit(1)
    
    # Start server
    start_optimized_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1) 