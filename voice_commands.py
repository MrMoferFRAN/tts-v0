#!/usr/bin/env python3
"""
Simple command-line interface for Voice Cloning API
Quick commands for common operations
"""

import argparse
import asyncio
import aiohttp
import json
import sys
from pathlib import Path

async def clone_voice_simple(text: str, voice: str = "voices", temperature: float = 0.7, 
                           output: str = None, streaming: bool = False):
    """Simple voice cloning command"""
    
    base_url = "http://localhost:7860"
    
    async with aiohttp.ClientSession() as session:
        try:
            # Prepare data for the 'request' field in the JSON payload
            request_data = {
                "text": text,
                "voice_name": voice,
                "temperature": temperature,
                "remove_silence": True,
                "use_optimization": True,
                "streaming": streaming # Send boolean directly
            }
            
            # Choose endpoint
            endpoint = "/clone-voice-stream" if streaming else "/clone-voice"
            
            print(f"🎤 Cloning voice with profile '{voice}'...")
            print(f"📝 Text: {text[:50]}{'...' if len(text) > 50 else ''}")
            print(f"🌡️  Temperature: {temperature}")
            print(f"🌊 Streaming: {'Yes' if streaming else 'No'}")
            
            # Send JSON payload
            async with session.post(f"{base_url}{endpoint}", json={"request": request_data}) as response:
                if response.status == 200:
                    if streaming:
                        # Handle streaming response
                        output_file = output or f"voice_output_stream.wav"
                        with open(output_file, 'wb') as f:
                            async for chunk in response.content.iter_chunked(8192):
                                f.write(chunk)
                        print(f"✅ Streaming completed: {output_file}")
                    else:
                        # Handle regular response
                        result = await response.json()
                        if result.get("success"):
                            audio_url = result.get("audio_url")
                            metrics = result.get("performance_metrics", {})
                            
                            print(f"✅ Voice cloning successful!")
                            print(f"📁 Output: {audio_url}")
                            print(f"⏱️  Processing time: {metrics.get('processing_time', 0):.2f}s")
                            print(f"📊 Realtime factor: {metrics.get('realtime_factor', 0):.2f}")
                        else:
                            print(f"❌ Error: {result.get('error', 'Unknown error')}")
                else:
                    error_text = await response.text()
                    print(f"❌ HTTP Error {response.status}: {error_text}")
        
        except Exception as e:
            print(f"❌ Connection error: {e}")

async def list_voices_simple():
    """List available voices"""
    
    base_url = "http://localhost:7860"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/voices") as response:
                if response.status == 200:
                    result = await response.json()
                    voices = result.get("voices", [])
                    stats = result.get("voice_stats", {})
                    
                    print("🎭 Available Voice Profiles:")
                    print("=" * 40)
                    
                    if not voices:
                        print("No voice profiles found.")
                        return
                    
                    for voice_name in voices:
                        profile = result.get("profiles", {}).get(voice_name, {})
                        duration = profile.get("duration", 0)
                        quality = profile.get("quality_score", 0)
                        language = profile.get("language", "unknown")
                        
                        print(f"• {voice_name}")
                        print(f"  Duration: {duration:.1f}s")
                        print(f"  Quality: {quality:.2f}")
                        print(f"  Language: {language}")
                        print()
                    
                    print(f"📊 Total voices: {stats.get('total_voices', 0)}")
                    print(f"📊 Average quality: {stats.get('avg_quality', 0):.2f}")
                    print(f"📊 Total audio time: {stats.get('total_audio_time', 0):.1f}s")
                
                else:
                    print(f"❌ HTTP Error {response.status}")
        
        except Exception as e:
            print(f"❌ Connection error: {e}")

async def check_status_simple():
    """Check API status"""
    
    base_url = "http://localhost:7860"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    result = await response.json()
                    status = result.get("status")
                    metrics = result.get("system_metrics", {})
                    
                    if status == "healthy":
                        print("✅ API is healthy and ready")
                        print(f"📊 CPU Usage: {metrics.get('cpu_usage', 0):.1f}%")
                        print(f"📊 RAM Usage: {metrics.get('ram_usage', 0):.1f}%")
                        print(f"📊 GPU Memory: {metrics.get('gpu_memory', 0):.2f} GB")
                        print(f"🤖 Model loaded: {result.get('model_loaded', False)}")
                    else:
                        print(f"⚠️  API status: {status}")
                
                else:
                    print(f"❌ HTTP Error {response.status}")
        
        except Exception as e:
            print(f"❌ API not accessible: {e}")
            print("💡 Make sure the server is running on port 7860")

async def add_voice_simple(name: str, audio_file: str, transcription: str, language: str = "es"):
    """Add a new voice profile"""
    
    base_url = "http://localhost:7860"
    
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    async with aiohttp.ClientSession() as session:
        try:
            # Prepare form data
            form_data = aiohttp.FormData()
            form_data.add_field('transcription', transcription)
            form_data.add_field('language', language)
            
            with open(audio_file, 'rb') as f:
                form_data.add_field('audio_file', f, 
                                  filename=Path(audio_file).name,
                                  content_type='audio/mpeg')
                
                print(f"🎤 Adding voice profile '{name}'...")
                print(f"📁 Audio file: {audio_file}")
                print(f"📝 Transcription: {transcription[:50]}{'...' if len(transcription) > 50 else ''}")
                
                async with session.post(f"{base_url}/voices/{name}", data=form_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("success"):
                            profile = result.get("profile", {})
                            print(f"✅ Voice profile '{name}' added successfully!")
                            print(f"📊 Quality score: {profile.get('quality_score', 0):.2f}")
                            print(f"📊 Duration: {profile.get('duration', 0):.1f}s")
                        else:
                            print(f"❌ Failed to add voice profile")
                    else:
                        error_text = await response.text()
                        print(f"❌ HTTP Error {response.status}: {error_text}")
        
        except Exception as e:
            print(f"❌ Error adding voice: {e}")

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(description="Voice Cloning API Commands")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Clone command
    clone_parser = subparsers.add_parser('clone', help='Clone a voice')
    clone_parser.add_argument('text', help='Text to synthesize')
    clone_parser.add_argument('--voice', default='voices', help='Voice profile name (default: voices)')
    clone_parser.add_argument('--temperature', type=float, default=0.7, help='Temperature (0.0-2.0)')
    clone_parser.add_argument('--output', help='Output file path')
    clone_parser.add_argument('--stream', action='store_true', help='Use streaming')
    
    # List voices command
    list_parser = subparsers.add_parser('voices', help='List available voice profiles')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check API status')
    
    # Add voice command
    add_parser = subparsers.add_parser('add', help='Add new voice profile')
    add_parser.add_argument('name', help='Voice profile name')
    add_parser.add_argument('audio', help='Audio file path')
    add_parser.add_argument('transcription', help='Audio transcription')
    add_parser.add_argument('--language', default='es', help='Language code (default: es)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'clone':
            asyncio.run(clone_voice_simple(
                text=args.text,
                voice=args.voice,
                temperature=args.temperature,
                output=args.output,
                streaming=args.stream
            ))
        
        elif args.command == 'voices':
            asyncio.run(list_voices_simple())
        
        elif args.command == 'status':
            asyncio.run(check_status_simple())
        
        elif args.command == 'add':
            asyncio.run(add_voice_simple(
                name=args.name,
                audio_file=args.audio,
                transcription=args.transcription,
                language=args.language
            ))
    
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 