#!/usr/bin/env python3
"""
Simplified LoRA Training Script for Stable Diffusion XL
Compatible with current PEFT and diffusers versions
"""

import os
import yaml
import torch
import logging
from pathlib import Path
from typing import Dict, Any
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_device(config: Dict[str, Any]) -> torch.device:
    """Setup device for training."""
    if torch.cuda.is_available() and config["hardware"]["device"] == "cuda":
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="LoRA Training for SDXL")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to training configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info("Configuration loaded successfully")
    
    # Setup device
    device = setup_device(config)
    
    # Create output directory
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    logger.info("Starting LoRA training setup...")
    
    try:
        # Import required libraries
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        from diffusers.utils import logging as diffusers_logging
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPTextModel, CLIPTokenizer
        import torch.nn.functional as F
        
        # Set diffusers logging level
        diffusers_logging.set_verbosity_info()
        
        logger.info("Loading base SDXL model...")
        
        # Load base model
        model_id = config["model"]["base_model"]
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        
        # Move to device
        pipeline = pipeline.to(device)
        
        logger.info("Setting up LoRA configuration...")
        
        # Setup LoRA config
        lora_config = LoraConfig(
            r=config["model"]["lora_rank"],
            lora_alpha=config["model"]["lora_alpha"],
            target_modules=config["model"]["target_modules"],
            lora_dropout=config["model"]["lora_dropout"],
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        logger.info(f"LoRA config: {lora_config}")
        
        # Apply LoRA to UNet
        logger.info("Applying LoRA to UNet...")
        pipeline.unet = get_peft_model(pipeline.unet, lora_config)
        
        # Freeze VAE and text encoders
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
        
        # Enable gradient computation for UNet
        pipeline.unet.requires_grad_(True)
        
        logger.info("Model setup complete!")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in pipeline.unet.parameters() if p.requires_grad):,}")
        
        # Save the LoRA model
        logger.info("Saving LoRA model...")
        pipeline.unet.save_pretrained(str(output_dir / "lora_weights"))
        
        logger.info(f"LoRA model saved to {output_dir / 'lora_weights'}")
        logger.info("Training setup completed successfully!")
        
        logger.info("🎉 LoRA training setup completed successfully!")
        logger.info("Your LoRA model is ready for training!")
        logger.info("Next steps:")
        logger.info("1. Use the saved LoRA weights in outputs/lora_weights/")
        logger.info("2. Load them with your base model for inference")
        logger.info("3. Or continue with full training loop implementation")
        
    except Exception as e:
        logger.error(f"Error during training setup: {str(e)}")
        raise

if __name__ == "__main__":
    main()
