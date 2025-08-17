#!/usr/bin/env python3
"""
Inference Script for LoRA Fine-tuned SDXL Model
Loads trained LoRA weights and generates images
"""

import os
import yaml
import torch
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_device(config: Dict[str, Any]) -> torch.device:
    """Setup device for inference."""
    if torch.cuda.is_available() and config["hardware"]["device"] == "cuda":
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device

def load_lora_model(base_model_id: str, lora_path: str, device: torch.device):
    """Load base model with LoRA weights."""
    from diffusers import StableDiffusionXLPipeline
    from peft import PeftModel
    
    logger.info(f"Loading base model: {base_model_id}")
    
    # Load base pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device.type == "cuda" else None
    )
    
    # Move to device
    pipeline = pipeline.to(device)
    
    # Load LoRA weights
    logger.info(f"Loading LoRA weights from: {lora_path}")
    pipeline.unet = PeftModel.from_pretrained(pipeline.unet, lora_path)
    
    # Fix PEFT wrapper issue by setting the forward method
    def fixed_forward(*args, **kwargs):
        # Remove input_ids if present (PEFT wrapper issue)
        if 'input_ids' in kwargs:
            del kwargs['input_ids']
        return pipeline.unet.base_model.forward(*args, **kwargs)
    
    pipeline.unet.forward = fixed_forward
    
    # Enable memory efficient attention if available
    if hasattr(torch.backends, "cuda") and torch.backends.cuda.is_built():
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
    
    logger.info("LoRA model loaded successfully!")
    return pipeline

def generate_image(pipeline, prompt: str, negative_prompt: str = "", 
                  height: int = 1024, width: int = 1024, 
                  num_inference_steps: int = 50, guidance_scale: float = 7.5,
                  seed: int = None):
    """Generate image using the LoRA model."""
    
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    
    logger.info(f"Generating image with prompt: {prompt}")
    logger.info(f"Parameters: {height}x{width}, steps={num_inference_steps}, guidance={guidance_scale}")
    
    # Generate image
    with torch.no_grad():
        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=pipeline.device).manual_seed(seed) if seed else None
        ).images[0]
    
    logger.info("Image generation completed!")
    return image

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="LoRA Model Inference")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--lora_path", type=str, required=True,
                       help="Path to LoRA weights directory")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="",
                       help="Negative prompt")
    parser.add_argument("--output_path", type=str, default="generated_image.png",
                       help="Output image path")
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                       help="Image width")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info("Configuration loaded successfully")
    
    # Setup device
    device = setup_device(config)
    
    try:
        # Load LoRA model
        pipeline = load_lora_model(
            config["model"]["base_model"],
            args.lora_path,
            device
        )
        
        # Generate image
        image = generate_image(
            pipeline,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )
        
        # Save image
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        
        logger.info(f"🎉 Image saved to: {output_path}")
        
        # Display image info
        logger.info(f"Image size: {image.size}")
        logger.info(f"Image mode: {image.mode}")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()
