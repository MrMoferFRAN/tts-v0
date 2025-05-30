#!/usr/bin/env python3
"""
Voice Manager for organizing multiple voice references
Manages voice profiles with audio files and transcriptions
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple
import librosa
import numpy as np
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class VoiceProfile:
    """Voice profile with audio reference and metadata"""
    name: str
    audio_path: str
    transcription: str
    language: str = "es"
    quality_score: Optional[float] = None
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VoiceProfile':
        return cls(**data)

class VoiceManager:
    """Manager for voice profiles and references"""
    
    def __init__(self, voices_dir: str = "voices"):
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(exist_ok=True)
        self.profiles_file = self.voices_dir / "profiles.json"
        self.profiles: Dict[str, VoiceProfile] = {}
        self.load_profiles()
    
    def load_profiles(self):
        """Load voice profiles from JSON file"""
        try:
            if self.profiles_file.exists():
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.profiles = {
                        name: VoiceProfile.from_dict(profile_data) 
                        for name, profile_data in data.items()
                    }
                logger.info(f"Loaded {len(self.profiles)} voice profiles")
            else:
                logger.info("No existing profiles found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            self.profiles = {}
    
    def save_profiles(self):
        """Save voice profiles to JSON file"""
        try:
            data = {name: profile.to_dict() for name, profile in self.profiles.items()}
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.profiles)} voice profiles")
        except Exception as e:
            logger.error(f"Error saving profiles: {e}")
    
    def add_voice(self, name: str, audio_path: str, transcription: str, 
                  language: str = "es", copy_file: bool = True) -> bool:
        """Add a new voice profile"""
        try:
            audio_path = Path(audio_path)
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_path}")
                return False
            
            # Create voice directory
            voice_dir = self.voices_dir / name
            voice_dir.mkdir(exist_ok=True)
            
            # Copy or link audio file to voices directory
            if copy_file:
                import shutil
                target_audio = voice_dir / f"reference{audio_path.suffix}"
                shutil.copy2(audio_path, target_audio)
                final_audio_path = str(target_audio)
            else:
                final_audio_path = str(audio_path.absolute())
            
            # Analyze audio
            audio_data, sr = librosa.load(final_audio_path, sr=None)
            duration = len(audio_data) / sr
            
            # Calculate quality score (simple metric based on RMS and frequency content)
            rms = np.sqrt(np.mean(audio_data**2))
            quality_score = min(1.0, rms * 10)  # Simple quality estimation
            
            # Create profile
            from datetime import datetime
            profile = VoiceProfile(
                name=name,
                audio_path=final_audio_path,
                transcription=transcription,
                language=language,
                quality_score=quality_score,
                duration=duration,
                sample_rate=sr,
                created_at=datetime.now().isoformat()
            )
            
            self.profiles[name] = profile
            self.save_profiles()
            
            logger.info(f"Added voice profile '{name}' - Duration: {duration:.2f}s, Quality: {quality_score:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding voice '{name}': {e}")
            return False
    
    def get_voice(self, name: str) -> Optional[VoiceProfile]:
        """Get voice profile by name"""
        return self.profiles.get(name)
    
    def list_voices(self) -> List[str]:
        """List all available voice names"""
        return list(self.profiles.keys())
    
    def remove_voice(self, name: str) -> bool:
        """Remove a voice profile"""
        try:
            if name in self.profiles:
                profile = self.profiles[name]
                
                # Remove audio file if it's in voices directory
                audio_path = Path(profile.audio_path)
                if audio_path.parent == self.voices_dir / name:
                    if audio_path.exists():
                        audio_path.unlink()
                    # Remove voice directory if empty
                    try:
                        (self.voices_dir / name).rmdir()
                    except OSError:
                        pass
                
                del self.profiles[name]
                self.save_profiles()
                logger.info(f"Removed voice profile '{name}'")
                return True
            else:
                logger.warning(f"Voice '{name}' not found")
                return False
        except Exception as e:
            logger.error(f"Error removing voice '{name}': {e}")
            return False
    
    def get_voice_stats(self) -> Dict:
        """Get statistics about voice profiles"""
        if not self.profiles:
            return {"total_voices": 0}
        
        durations = [p.duration for p in self.profiles.values() if p.duration]
        quality_scores = [p.quality_score for p in self.profiles.values() if p.quality_score]
        
        return {
            "total_voices": len(self.profiles),
            "avg_duration": np.mean(durations) if durations else 0,
            "avg_quality": np.mean(quality_scores) if quality_scores else 0,
            "languages": list(set(p.language for p in self.profiles.values())),
            "total_audio_time": sum(durations) if durations else 0
        }
    
    def setup_default_voices(self):
        """Setup default voice profiles from existing files"""
        # Check for the existing audio file in multiple locations
        possible_paths = [
            "Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3",
            "voices/Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3",
            "./voices/Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3"
        ]
        
        existing_audio = None
        for path in possible_paths:
            if Path(path).exists():
                existing_audio = path
                break
        
        if existing_audio:
            logger.info(f"Found reference audio at: {existing_audio}")
            success = self.add_voice(
                name="voices", 
                audio_path=existing_audio,
                transcription="Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo.",
                language="es",
                copy_file=False  # Keep original file location
            )
            if success:
                logger.info("Added default 'voices' profile")
                return True
            else:
                logger.error("Failed to add default 'voices' profile")
                return False
        else:
            logger.warning("Default audio file not found in any expected location")
            logger.info("Expected locations:")
            for path in possible_paths:
                logger.info(f"  - {path}")
            return False

# Global voice manager instance
voice_manager = VoiceManager()

def get_voice_manager() -> VoiceManager:
    """Get the global voice manager instance"""
    return voice_manager

def initialize_voices():
    """Initialize voice manager with default voices"""
    vm = get_voice_manager()
    vm.setup_default_voices()
    return vm 