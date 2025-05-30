"""
Voice Cloning Module using Sesame CSM-1B
Based on: https://github.com/isaiahbjork/csm-voice-cloning
"""

from .voice_clone import VoiceCloner
from .models import load_csm_model
from .watermarking import apply_watermark

__all__ = ['VoiceCloner', 'load_csm_model', 'apply_watermark'] 