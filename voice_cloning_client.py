#!/usr/bin/env python3
"""
Client example for Voice Cloning API
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path
from typing import Optional

class VoiceCloneClient:
    """
    Client for Voice Cloning API
    """
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self):
        """Check API health"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def clone_voice(self, text: str, reference_audio_path: Optional[str] = None,
                         reference_text: Optional[str] = None, **kwargs):
        """
        Clone voice using the API
        """
        data = {
            "text": text,
            "reference_text": reference_text,
            "speaker_id": kwargs.get("speaker_id", "0"),
            "temperature": kwargs.get("temperature", 0.7),
            "chunk_size": kwargs.get("chunk_size", 100),
            "remove_silence": kwargs.get("remove_silence", True),
            "streaming": kwargs.get("streaming", False),
            "max_silence_duration": kwargs.get("max_silence_duration", 0.5)
        }
        
        # Prepare form data
        form_data = aiohttp.FormData()
        
        # Add JSON data
        for key, value in data.items():
            if value is not None:
                form_data.add_field(key, str(value))
        
        # Add audio file if provided
        if reference_audio_path and Path(reference_audio_path).exists():
            form_data.add_field('reference_audio', 
                              open(reference_audio_path, 'rb'),
                              filename=Path(reference_audio_path).name,
                              content_type='audio/wav')
        
        async with self.session.post(f"{self.base_url}/clone-voice", 
                                   data=form_data) as response:
            return await response.json()
    
    async def stream_voice_clone(self, text: str, reference_audio_path: Optional[str] = None,
                                reference_text: Optional[str] = None, output_path: str = "streamed_output.wav",
                                **kwargs):
        """
        Stream voice cloning
        """
        data = {
            "text": text,
            "reference_text": reference_text,
            "speaker_id": kwargs.get("speaker_id", "0"),
            "temperature": kwargs.get("temperature", 0.7),
            "chunk_size": kwargs.get("chunk_size", 100),
            "remove_silence": kwargs.get("remove_silence", True),
            "streaming": "true",  # Enable streaming
            "max_silence_duration": kwargs.get("max_silence_duration", 0.5)
        }
        
        # Prepare form data
        form_data = aiohttp.FormData()
        
        for key, value in data.items():
            if value is not None:
                form_data.add_field(key, str(value))
        
        if reference_audio_path and Path(reference_audio_path).exists():
            form_data.add_field('reference_audio', 
                              open(reference_audio_path, 'rb'),
                              filename=Path(reference_audio_path).name,
                              content_type='audio/wav')
        
        # Stream the response
        async with self.session.post(f"{self.base_url}/clone-voice-stream", 
                                   data=form_data) as response:
            with open(output_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)
        
        return output_path
    
    async def batch_clone_voice(self, texts: list, reference_audio_path: Optional[str] = None,
                               reference_text: Optional[str] = None, **kwargs):
        """
        Batch voice cloning
        """
        data = {
            "texts": texts,
            "reference_text": reference_text,
            "speaker_id": kwargs.get("speaker_id", "0"),
            "temperature": kwargs.get("temperature", 0.7),
            "chunk_size": kwargs.get("chunk_size", 100),
            "remove_silence": kwargs.get("remove_silence", True),
            "max_silence_duration": kwargs.get("max_silence_duration", 0.5)
        }
        
        # Prepare form data
        form_data = aiohttp.FormData()
        
        # Add JSON data
        form_data.add_field('texts', json.dumps(texts))
        for key, value in data.items():
            if key != 'texts' and value is not None:
                form_data.add_field(key, str(value))
        
        if reference_audio_path and Path(reference_audio_path).exists():
            form_data.add_field('reference_audio', 
                              open(reference_audio_path, 'rb'),
                              filename=Path(reference_audio_path).name,
                              content_type='audio/wav')
        
        async with self.session.post(f"{self.base_url}/batch-clone-voice", 
                                   data=form_data) as response:
            return await response.json()
    
    async def get_performance_stats(self):
        """Get performance statistics"""
        async with self.session.get(f"{self.base_url}/performance-stats") as response:
            return await response.json()

async def demo_client():
    """
    Demo function showing how to use the client
    """
    async with VoiceCloneClient() as client:
        print("🚀 Voice Cloning API Client Demo")
        print("=" * 50)
        
        # 1. Health check
        print("\n1. Checking API health...")
        health = await client.health_check()
        print(f"   Status: {health.get('status')}")
        print(f"   Model loaded: {health.get('model_loaded')}")
        
        if not health.get('model_loaded'):
            print("❌ Model not loaded. Please start the API server first.")
            return
        
        # 2. Performance stats
        print("\n2. Getting performance stats...")
        stats = await client.get_performance_stats()
        metrics = stats.get('system_metrics', {})
        print(f"   CPU Usage: {metrics.get('cpu_usage', 0):.1f}%")
        print(f"   RAM Usage: {metrics.get('ram_usage', 0):.1f}%")
        print(f"   GPU Memory: {metrics.get('gpu_memory', 0):.1f} GB")
        
        # 3. Simple voice cloning
        print("\n3. Simple voice cloning...")
        simple_text = "Hola, esto es una prueba de clonación de voz simple."
        
        start_time = time.time()
        result = await client.clone_voice(
            text=simple_text,
            chunk_size=50,
            temperature=0.8
        )
        end_time = time.time()
        
        if result.get('success'):
            print(f"   ✅ Generated: {result.get('audio_url')}")
            metrics = result.get('performance_metrics', {})
            print(f"   📊 Processing time: {end_time - start_time:.2f}s")
            print(f"   📊 Realtime factor: {metrics.get('realtime_factor', 0):.2f}")
            print(f"   📊 Tokens/second: {metrics.get('tokens_per_second', 0):.1f}")
        else:
            print(f"   ❌ Error: {result.get('error')}")
        
        # 4. Voice cloning with reference
        reference_audio = "Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3"
        if Path(reference_audio).exists():
            print("\n4. Voice cloning with reference audio...")
            reference_text = "Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo."
            clone_text = "Esta es una demostración de clonación de voz con audio de referencia."
            
            start_time = time.time()
            result = await client.clone_voice(
                text=clone_text,
                reference_audio_path=reference_audio,
                reference_text=reference_text,
                chunk_size=80,
                temperature=0.7,
                remove_silence=True
            )
            end_time = time.time()
            
            if result.get('success'):
                print(f"   ✅ Generated: {result.get('audio_url')}")
                metrics = result.get('performance_metrics', {})
                processing_info = result.get('processing_info', {})
                print(f"   📊 Processing time: {end_time - start_time:.2f}s")
                print(f"   📊 Chunks processed: {processing_info.get('chunks_processed', 0)}")
                print(f"   📊 Audio duration: {processing_info.get('total_audio_duration', 0):.2f}s")
                print(f"   📊 Silence removed: {processing_info.get('silence_removed', False)}")
            else:
                print(f"   ❌ Error: {result.get('error')}")
        
        # 5. Streaming demo
        print("\n5. Streaming voice cloning...")
        stream_text = "Esta es una demostración de generación de voz en streaming. El audio se genera en chunks en tiempo real."
        
        try:
            start_time = time.time()
            output_path = await client.stream_voice_clone(
                text=stream_text,
                reference_audio_path=reference_audio if Path(reference_audio).exists() else None,
                reference_text=reference_text if Path(reference_audio).exists() else None,
                output_path="demo_streamed.wav",
                chunk_size=60
            )
            end_time = time.time()
            
            print(f"   ✅ Streamed to: {output_path}")
            print(f"   📊 Streaming time: {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"   ❌ Streaming error: {str(e)}")
        
        # 6. Batch processing
        print("\n6. Batch voice cloning...")
        batch_texts = [
            "Primera frase del lote de procesamiento.",
            "Segunda frase con diferentes características.",
            "Tercera y última frase del demo de lote."
        ]
        
        start_time = time.time()
        batch_result = await client.batch_clone_voice(
            texts=batch_texts,
            reference_audio_path=reference_audio if Path(reference_audio).exists() else None,
            reference_text=reference_text if Path(reference_audio).exists() else None,
            chunk_size=70
        )
        end_time = time.time()
        
        if batch_result.get('success'):
            print(f"   ✅ Processed {batch_result.get('total_processed')} texts")
            print(f"   📊 Batch processing time: {end_time - start_time:.2f}s")
            
            for result in batch_result.get('results', []):
                if result.get('result', {}).get('success'):
                    print(f"      {result.get('index')+1}. {result.get('result', {}).get('audio_url')}")
        else:
            print(f"   ❌ Batch error: {batch_result.get('error', 'Unknown error')}")
        
        # 7. Final performance stats
        print("\n7. Final performance stats...")
        final_stats = await client.get_performance_stats()
        final_metrics = final_stats.get('system_metrics', {})
        print(f"   CPU Usage: {final_metrics.get('cpu_usage', 0):.1f}%")
        print(f"   RAM Usage: {final_metrics.get('ram_usage', 0):.1f}%")
        print(f"   GPU Memory: {final_metrics.get('gpu_memory', 0):.1f} GB")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Usage Tips:")
        print("   - Adjust chunk_size based on memory and speed requirements")
        print("   - Use streaming for real-time applications")
        print("   - Monitor performance metrics for optimization")
        print("   - Enable silence removal for cleaner audio")

if __name__ == "__main__":
    asyncio.run(demo_client()) 