"""
Generator module for CSM-1B Voice Cloning
Based on: https://github.com/isaiahbjork/csm-voice-cloning
"""

import torch
import torchaudio
import os
from typing import Optional, Dict, Any
from .models import load_csm_model
import numpy as np
import librosa

class VoiceGenerator:
    """
    Voice generation class compatible with the original repository structure
    """
    
    def __init__(self, model_path: str = "./models/sesame-csm-1b"):
        """
        Initialize the voice generator
        
        Args:
            model_path: Path to the CSM model
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the model and processor"""
        if self.model is None:
            self.model, self.processor = load_csm_model(self.model_path)
            
    def generate(self, 
                context_audio_path: str,
                context_text: str, 
                text: str,
                output_filename: str = "output.wav",
                temperature: float = 0.7,
                speaker_id: str = "0") -> str:
        """
        Generate voice cloned audio using CSM
        
        Args:
            context_audio_path: Path to reference audio
            context_text: Transcript of reference audio
            text: Text to synthesize
            output_filename: Output file name
            temperature: Generation temperature
            speaker_id: Speaker ID for the conversation
            
        Returns:
            Path to generated audio file
        """
        self.load_model()
        
        print(f"Generating audio for: '{text}'")
        print(f"Using context: '{context_text}'")
        print(f"Reference audio: {context_audio_path}")
        
        # Load and preprocess reference audio
        context_audio = None
        if context_audio_path and os.path.exists(context_audio_path):
            context_audio = self._preprocess_audio(context_audio_path)
        
        # Create conversation format
        conversation = self._create_conversation(
            context_text, text, context_audio, speaker_id
        )
        
        # Process inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        ).to(self.device)
        
        # Generate
        gen_kwargs = {
            "output_audio": True,
            "temperature": temperature,
            "do_sample": True if temperature > 0 else False,
        }
        
        with torch.no_grad():
            audio = self.model.generate(**inputs, **gen_kwargs)
        
        # Save the generated audio
        self.processor.save_audio(audio, output_filename)
        
        print(f"Audio generated and saved to: {output_filename}")
        return output_filename
        
    def _preprocess_audio(self, audio_path: str, target_sample_rate: int = 24000) -> np.ndarray:
        """
        Preprocess audio for CSM model
        
        Args:
            audio_path: Path to audio file
            target_sample_rate: Target sample rate (CSM uses 24kHz)
            
        Returns:
            Preprocessed audio array
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Resample if necessary
        if sr != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio
        
    def _create_conversation(self, context_text: str, target_text: str, 
                           context_audio: Optional[np.ndarray] = None, 
                           speaker_id: str = "0") -> list:
        """
        Create conversation format for CSM
        
        Args:
            context_text: Reference transcript
            target_text: Target text to synthesize
            context_audio: Reference audio array
            speaker_id: Speaker ID
            
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
        
        # Add target text
        conversation.append({
            "role": speaker_id,
            "content": [{"type": "text", "text": target_text}]
        })
        
        return conversation

def main():
    """
    Main function for direct script execution (compatibility with original repo)
    """
    # Configuration (matching original repository style)
    context_audio_path = "Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo..mp3"
    context_text = "Ah, ¿en serio? Vaya, eso debe ser un poco incómodo para tu equipo."
    text = "Hola, esta es una prueba de clonación de voz usando el modelo CSM-1B."
    output_filename = "generated_voice.wav"
    
    # Create generator
    generator = VoiceGenerator()
    
    # Generate audio
    result = generator.generate(
        context_audio_path=context_audio_path,
        context_text=context_text,
        text=text,
        output_filename=output_filename
    )
    
    print(f"Voice cloning completed! Output saved to: {result}")

if __name__ == "__main__":
    main() 