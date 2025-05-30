"""
Model loading utilities for Sesame CSM-1B Voice Cloning
"""

import torch
import os
from transformers import CsmForConditionalGeneration, AutoProcessor
from typing import Tuple, Optional

class CSMModelConfig:
    """Configuration class for CSM model parameters"""
    def __init__(self, max_length: int = 2048, temperature: float = 0.7):
        self.max_length = max_length
        self.temperature = temperature

def load_csm_model(model_path: str = "./models/sesame-csm-1b", 
                   config: Optional[CSMModelConfig] = None) -> Tuple[CsmForConditionalGeneration, AutoProcessor]:
    """
    Load the Sesame CSM-1B model from local path
    
    Args:
        model_path: Path to the locally downloaded model
        config: Model configuration parameters
        
    Returns:
        Tuple of (model, processor)
    """
    if config is None:
        config = CSMModelConfig()
    
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"Loading CSM-1B model from: {model_path}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Load model with consistent float32 to avoid type mismatches
    model = CsmForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use float32 instead of float16 to avoid type mismatches
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Configure generation settings
    model.generation_config.max_length = config.max_length
    model.generation_config.temperature = config.temperature
    
    # Set model to evaluation mode
    model.eval()
    
    print("CSM-1B model loaded successfully!")
    return model, processor

def get_model_info(model_path: str = "./models/sesame-csm-1b") -> dict:
    """
    Get information about the model
    
    Args:
        model_path: Path to the model
        
    Returns:
        Dictionary with model information
    """
    config_path = os.path.join(model_path, "config.json")
    
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    else:
        return {"error": "Config file not found"}

# Compatibility functions for the original repository structure
def llama3_2_1B(max_seq_len: int = 2048):
    """
    Compatibility function for the original repository structure
    """
    config = CSMModelConfig(max_length=max_seq_len)
    return load_csm_model(config=config)

def llama3_2_100M(max_seq_len: int = 2048):
    """
    Compatibility function for smaller model (if available)
    """
    # For now, use the same 1B model
    config = CSMModelConfig(max_length=max_seq_len)
    return load_csm_model(config=config) 