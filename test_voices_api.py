#!/usr/bin/env python3
"""
Test Script for Voice Cloning API - Optimized for 'voices' profile
Tests the API with the configured voice profile and shows performance metrics
"""

import asyncio
import aiohttp
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any

class VoiceAPITester:
    """Tester for Voice Cloning API with performance monitoring"""
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def check_health(self) -> Dict[str, Any]:
        """Check API health and system status"""
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "error", "code": response.status}
        except Exception as e:
            return {"status": "connection_error", "error": str(e)}
    
    async def list_voices(self) -> Dict[str, Any]:
        """List available voice profiles"""
        try:
            async with self.session.get(f"{self.base_url}/voices") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def test_voice_cloning(self, text: str, voice_name: str = "voices", 
                               temperature: float = 0.7, chunk_size: int = None) -> Dict[str, Any]:
        """Test voice cloning with specified parameters"""
        start_time = time.time()
        
        try:
            # Prepare request data
            data = {
                "text": text,
                "voice_name": voice_name,
                "temperature": temperature,
                "remove_silence": True,
                "use_optimization": True
            }
            
            if chunk_size:
                data["chunk_size"] = chunk_size
            
            # Prepare form data
            form_data = aiohttp.FormData()
            for key, value in data.items():
                form_data.add_field(key, str(value))
            
            # Make request
            async with self.session.post(f"{self.base_url}/clone-voice", data=form_data) as response:
                result = await response.json()
                
                # Add timing information
                total_time = time.time() - start_time
                result["client_timing"] = {
                    "total_request_time": total_time,
                    "status_code": response.status
                }
                
                return result
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "client_timing": {"total_request_time": time.time() - start_time}
            }
    
    async def test_streaming(self, text: str, voice_name: str = "voices", 
                           output_path: str = "test_stream.wav") -> Dict[str, Any]:
        """Test streaming voice cloning"""
        start_time = time.time()
        
        try:
            data = {
                "text": text,
                "voice_name": voice_name,
                "streaming": "true",
                "temperature": 0.7,
                "chunk_size": 60,
                "remove_silence": True
            }
            
            form_data = aiohttp.FormData()
            for key, value in data.items():
                form_data.add_field(key, str(value))
            
            chunks_received = 0
            total_bytes = 0
            
            async with self.session.post(f"{self.base_url}/clone-voice-stream", data=form_data) as response:
                if response.status == 200:
                    with open(output_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            chunks_received += 1
                            total_bytes += len(chunk)
                    
                    total_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "output_path": output_path,
                        "streaming_stats": {
                            "chunks_received": chunks_received,
                            "total_bytes": total_bytes,
                            "total_time": total_time,
                            "avg_chunk_size": total_bytes / chunks_received if chunks_received > 0 else 0
                        }
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "streaming_stats": {"chunks_received": 0, "total_bytes": 0}
                    }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "streaming_stats": {"chunks_received": 0, "total_bytes": 0}
            }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        try:
            async with self.session.get(f"{self.base_url}/performance-stats") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def get_chunk_recommendation(self, text: str, streaming: bool = False) -> Dict[str, Any]:
        """Get chunk size recommendation for text"""
        try:
            params = {"text": text, "streaming": streaming}
            async with self.session.get(f"{self.base_url}/chunk-size-recommendation", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}

async def run_comprehensive_test():
    """Run comprehensive test suite"""
    
    print("🎤 Voice Cloning API - Comprehensive Test Suite")
    print("=" * 60)
    
    async with VoiceAPITester() as tester:
        
        # 1. Health Check
        print("\n1️⃣ Health Check...")
        health = await tester.check_health()
        
        if health.get("status") == "healthy":
            print("   ✅ API is healthy")
            metrics = health.get("system_metrics", {})
            print(f"   📊 CPU: {metrics.get('cpu_usage', 0):.1f}%")
            print(f"   📊 RAM: {metrics.get('ram_usage', 0):.1f}%")
            print(f"   📊 GPU Memory: {metrics.get('gpu_memory', 0):.2f} GB")
        else:
            print(f"   ❌ API Health Check Failed: {health}")
            return
        
        # 2. Voice Profiles
        print("\n2️⃣ Voice Profiles...")
        voices = await tester.list_voices()
        
        if "voices" in voices.get("voices", []):
            print("   ✅ 'voices' profile found")
            voice_stats = voices.get("voice_stats", {})
            print(f"   📊 Total Voices: {voice_stats.get('total_voices', 0)}")
            print(f"   📊 Avg Quality: {voice_stats.get('avg_quality', 0):.2f}")
            print(f"   📊 Total Audio Time: {voice_stats.get('total_audio_time', 0):.1f}s")
        else:
            print(f"   ❌ 'voices' profile not found. Available: {voices.get('voices', [])}")
            return
        
        # 3. Chunk Size Recommendation
        test_text = "Esta es una prueba de clonación de voz con el perfil optimizado. Queremos evaluar la calidad y el rendimiento del sistema."
        
        print("\n3️⃣ Getting Chunk Size Recommendation...")
        chunk_rec = await tester.get_chunk_recommendation(test_text, streaming=False)
        if "recommended_chunk_size" in chunk_rec:
            recommended_size = chunk_rec["recommended_chunk_size"]
            print(f"   ✅ Recommended chunk size: {recommended_size}")
            print(f"   📊 Text length: {chunk_rec.get('text_length', 0)}")
            print(f"   📊 System load: {chunk_rec.get('system_load', 0):.1f}%")
        else:
            print(f"   ❌ Could not get recommendation: {chunk_rec}")
            recommended_size = 80  # fallback
        
        # 4. Basic Voice Cloning Test
        print("\n4️⃣ Basic Voice Cloning Test...")
        basic_result = await tester.test_voice_cloning(
            text=test_text,
            voice_name="voices",
            temperature=0.7,
            chunk_size=recommended_size
        )
        
        if basic_result.get("success"):
            print("   ✅ Voice cloning successful")
            metrics = basic_result.get("performance_metrics", {})
            processing_info = basic_result.get("processing_info", {})
            
            print(f"   📊 Processing time: {metrics.get('processing_time', 0):.2f}s")
            print(f"   📊 Audio duration: {processing_info.get('total_audio_duration', 0):.2f}s")
            print(f"   📊 Realtime factor: {metrics.get('realtime_factor', 0):.2f}")
            print(f"   📊 Tokens/second: {metrics.get('tokens_per_second', 0):.1f}")
            print(f"   📊 Chunks processed: {processing_info.get('chunks_processed', 0)}")
            print(f"   🎭 Voice profile used: {processing_info.get('voice_profile_used', 'None')}")
            print(f"   📁 Output: {basic_result.get('audio_url', 'Not specified')}")
        else:
            print(f"   ❌ Voice cloning failed: {basic_result.get('error', 'Unknown error')}")
        
        # 5. Streaming Test
        print("\n5️⃣ Streaming Test...")
        stream_text = "Prueba de streaming con chunks en tiempo real. Este audio se genera de forma continua."
        
        stream_result = await tester.test_streaming(
            text=stream_text,
            voice_name="voices",
            output_path="test_stream_output.wav"
        )
        
        if stream_result.get("success"):
            print("   ✅ Streaming successful")
            stats = stream_result.get("streaming_stats", {})
            print(f"   📊 Chunks received: {stats.get('chunks_received', 0)}")
            print(f"   📊 Total bytes: {stats.get('total_bytes', 0):,}")
            print(f"   📊 Streaming time: {stats.get('total_time', 0):.2f}s")
            print(f"   📊 Avg chunk size: {stats.get('avg_chunk_size', 0):.0f} bytes")
            print(f"   📁 Stream output: {stream_result.get('output_path', 'Not specified')}")
        else:
            print(f"   ❌ Streaming failed: {stream_result.get('error', 'Unknown error')}")
        
        # 6. Performance Comparison
        print("\n6️⃣ Performance Comparison (Different Temperatures)...")
        
        temps = [0.5, 0.7, 0.9]
        comparison_text = "Texto corto para comparar temperaturas."
        
        for temp in temps:
            result = await tester.test_voice_cloning(
                text=comparison_text,
                voice_name="voices",
                temperature=temp,
                chunk_size=50  # Small chunk for quick test
            )
            
            if result.get("success"):
                metrics = result.get("performance_metrics", {})
                print(f"   🌡️  Temp {temp}: {metrics.get('processing_time', 0):.2f}s "
                      f"(RT: {metrics.get('realtime_factor', 0):.2f})")
            else:
                print(f"   ❌ Temp {temp}: Failed")
        
        # 7. Final Performance Stats
        print("\n7️⃣ Final Performance Statistics...")
        final_stats = await tester.get_performance_stats()
        
        if "system_metrics" in final_stats:
            system = final_stats["system_metrics"]
            print(f"   📊 Final CPU Usage: {system.get('cpu_usage', 0):.1f}%")
            print(f"   📊 Final RAM Usage: {system.get('ram_usage', 0):.1f}%")
            print(f"   📊 Final GPU Memory: {system.get('gpu_memory', 0):.2f} GB")
            
            if "optimization_stats" in final_stats:
                opt_stats = final_stats["optimization_stats"]
                cache_stats = opt_stats.get("cache_stats", {})
                print(f"   📊 Cache Size: {cache_stats.get('cache_size_mb', 0):.1f} MB")
                print(f"   📊 Cache Items: {cache_stats.get('cache_items', 0)}")
        
        print("\n🎉 Test Suite Completed!")
        print("=" * 60)

async def quick_test():
    """Quick single test"""
    print("🚀 Quick Voice Cloning Test")
    
    async with VoiceAPITester() as tester:
        # Quick health check
        health = await tester.check_health()
        if health.get("status") != "healthy":
            print(f"❌ API not healthy: {health}")
            return
        
        # Quick voice clone test
        result = await tester.test_voice_cloning(
            text="Hola, esta es una prueba rápida de clonación de voz.",
            voice_name="voices",
            temperature=0.7
        )
        
        if result.get("success"):
            metrics = result.get("performance_metrics", {})
            print(f"✅ Success! Processing time: {metrics.get('processing_time', 0):.2f}s")
            print(f"📁 Output: {result.get('audio_url', 'Not specified')}")
        else:
            print(f"❌ Failed: {result.get('error', 'Unknown error')}")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        asyncio.run(quick_test())
    else:
        asyncio.run(run_comprehensive_test())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1) 