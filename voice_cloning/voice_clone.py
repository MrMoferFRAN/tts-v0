"""
Main Voice Cloning implementation using Sesame CSM-1B
Based on: https://github.com/isaiahbjork/csm-voice-cloning
"""

import torch
import torchaudio
import os
import numpy as np
from typing import Optional, Tuple
from .models import load_csm_model, CSMModelConfig
from .watermarking import apply_watermark
import librosa
import soundfile as sf

class VoiceCloner:
    """
    Voice Cloning class using Sesame CSM-1B model
    """
    
    def __init__(self, model_path: str = "./models/sesame-csm-1b", 
                 max_length: int = 2048, 
                 device: Optional[str] = None):
        """
        Initialize the VoiceCloner
        
        Args:
            model_path: Path to the CSM-1B model
            max_length: Maximum sequence length for the model
            device: Device to run on (auto-detected if None)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = CSMModelConfig(max_length=max_length)
        
        # Initialize model and processor
        self.model = None
        self.processor = None
        self.load_model()
        
    def load_model(self):
        """Load the CSM model and processor"""
        print(f"Loading model on device: {self.device}")
        self.model, self.processor = load_csm_model(self.model_path, self.config)
        
    def preprocess_audio(self, audio_path: str, target_sample_rate: int = 24000) -> np.ndarray:
        """
        Preprocess audio file for voice cloning (CSM expects 24kHz)
        
        Args:
            audio_path: Path to the audio file
            target_sample_rate: Target sample rate for processing (CSM uses 24kHz)
            
        Returns:
            Preprocessed audio array as float32
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, dtype=np.float32)
        
        # Resample if necessary - CSM expects 24kHz
        if sr != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Ensure float32 type
        audio = audio.astype(np.float32)
        
        return audio
        
    def create_conversation(self, context_text: str, target_text: str, 
                           context_audio: Optional[np.ndarray] = None, 
                           speaker_id: str = "0") -> list:
        """
        Create conversation format for CSM model
        
        Args:
            context_text: Transcription of the reference audio
            target_text: Text to synthesize
            context_audio: Reference audio array (optional)
            speaker_id: Speaker ID for the conversation
            
        Returns:
            Conversation list in CSM format
        """
        conversation = []
        
        # Add context if available
        if context_audio is not None:
            conversation.append({
                "role": speaker_id,
                "content": [
                    {"type": "text", "text": context_text},
                    {"type": "audio", "path": context_audio}
                ]
            })
        
        # Add target text to synthesize
        conversation.append({
            "role": speaker_id,
            "content": [{"type": "text", "text": target_text}]
        })
        
        return conversation
        
    def generate_speech(self, context_text: str, target_text: str, 
                       context_audio_path: Optional[str] = None,
                       output_path: str = "output.wav",
                       temperature: float = 0.7,
                       speaker_id: str = "0") -> str:
        """
        Generate speech with voice cloning using CSM
        
        Args:
            context_text: Transcription of the reference voice
            target_text: Text to synthesize
            context_audio_path: Path to reference audio file
            output_path: Path to save the generated audio
            temperature: Generation temperature
            speaker_id: Speaker ID for the conversation
            
        Returns:
            Path to the generated audio file
        """
        print(f"Generating speech for: '{target_text}'")
        print(f"Using reference text: '{context_text}'")
        
        # Load and preprocess context audio if provided
        context_audio = None
        if context_audio_path and os.path.exists(context_audio_path):
            print(f"Loading reference audio: {context_audio_path}")
            context_audio = self.preprocess_audio(context_audio_path)
        
        # Create conversation in CSM format
        conversation = self.create_conversation(
            context_text, target_text, context_audio, speaker_id
        )
        
        # Process inputs using the CSM processor
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            "output_audio": True,
            "temperature": temperature,
            "do_sample": True if temperature > 0 else False,
        }
        
        # Generate with the model
        print("Generating audio...")
        with torch.no_grad():
            audio = self.model.generate(**inputs, **gen_kwargs)
        
        # Save the generated audio
        self.processor.save_audio(audio, output_path)
        
        print(f"Audio generated and saved to: {output_path}")
        return output_path
        
    def clone_voice_from_file(self, reference_audio: str, reference_transcript: str,
                             target_text: str, output_path: str = "cloned_voice.wav",
                             speaker_id: str = "0") -> str:
        """
        Convenience method to clone voice from a reference file
        
        Args:
            reference_audio: Path to reference audio file
            reference_transcript: Transcript of the reference audio
            target_text: Text to synthesize with the cloned voice
            output_path: Path to save the output
            speaker_id: Speaker ID for the conversation
            
        Returns:
            Path to the generated audio
        """
        return self.generate_speech(
            context_text=reference_transcript,
            target_text=target_text,
            context_audio_path=reference_audio,
            output_path=output_path,
            speaker_id=speaker_id
        )
        
    def batch_generate(self, text_list: list, context_text: str,
                      context_audio_path: Optional[str] = None,
                      output_dir: str = "outputs",
                      speaker_id: str = "0") -> list:
        """
        Generate multiple audio files in batch
        
        Args:
            text_list: List of texts to synthesize
            context_text: Reference transcript
            context_audio_path: Path to reference audio
            output_dir: Directory to save outputs
            speaker_id: Speaker ID for the conversation
            
        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for i, text in enumerate(text_list):
            output_path = os.path.join(output_dir, f"generated_{i:03d}.wav")
            result_path = self.generate_speech(
                context_text=context_text,
                target_text=text,
                context_audio_path=context_audio_path,
                output_path=output_path,
                speaker_id=speaker_id
            )
            output_paths.append(result_path)
            
        return output_paths
        
    def simple_generate(self, text: str, output_path: str = "simple_output.wav",
                       speaker_id: str = "0") -> str:
        """
        Simple text-to-speech without context audio
        
        Args:
            text: Text to synthesize
            output_path: Path to save the output
            speaker_id: Speaker ID
            
        Returns:
            Path to the generated audio
        """
        # Create simple conversation with just text
        conversation = [{
            "role": speaker_id,
            "content": [{"type": "text", "text": text}]
        }]
        
        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(self.device)
        
        # Generate
        print(f"Generating simple TTS for: '{text}'")
        with torch.no_grad():
            audio = self.model.generate(**inputs, output_audio=True)
        
        # Save
        self.processor.save_audio(audio, output_path)
        print(f"Simple TTS saved to: {output_path}")
        
        return output_path 