"""
Model utilities for text-to-image generation training.

This module provides functions for loading models, setting up LoRA configurations,
and managing model weights for Stable Diffusion fine-tuning.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL
)
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import CLIPTextModel, CLIPTokenizer

import logging

logger = logging.getLogger(__name__)


def load_base_model(
    model_path: str,
    model_type: str = "sdxl"
) -> Union[StableDiffusionPipeline, StableDiffusionXLPipeline]:
    """
    Load base Stable Diffusion model.
    
    Args:
        model_path: Path to the base model or HuggingFace model ID
        model_type: Type of model ('sdxl', 'sd2.1', 'sd1.5')
    
    Returns:
        Loaded pipeline
    """
    logger.info(f"Loading base model: {model_path}")
    
    try:
        if model_type.lower() == "sdxl":
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
        elif model_type.lower() in ["sd2.1", "sd1.5"]:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Successfully loaded {model_type} model")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def setup_lora_config(
    rank: int = 16,
    alpha: int = 32,
    target_modules: Optional[list] = None,
    dropout: float = 0.1
) -> LoraConfig:
    """
    Setup LoRA configuration.
    
    Args:
        rank: LoRA rank
        alpha: LoRA alpha parameter
        target_modules: List of target modules for LoRA
        dropout: Dropout rate
    
    Returns:
        LoRA configuration
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
    
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    logger.info(f"LoRA config: rank={rank}, alpha={alpha}, targets={target_modules}")
    return config


def apply_lora_to_model(
    model: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
    lora_config: LoraConfig
) -> Union[StableDiffusionPipeline, StableDiffusionXLPipeline]:
    """
    Apply LoRA to the UNet of a Stable Diffusion model.
    
    Args:
        model: Base model pipeline
        lora_config: LoRA configuration
    
    Returns:
        Model with LoRA applied
    """
    logger.info("Applying LoRA to UNet...")
    
    # Apply LoRA to UNet
    model.unet = get_peft_model(model.unet, lora_config)
    
    # Enable gradient computation for LoRA parameters
    model.unet.enable_input_require_grads()
    
    logger.info("LoRA applied successfully")
    return model


def save_lora_weights(
    model: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
    output_dir: Union[str, Path],
    save_metadata: bool = True
) -> None:
    """
    Save LoRA weights from a fine-tuned model.
    
    Args:
        model: Fine-tuned model
        output_dir: Directory to save weights
        save_metadata: Whether to save metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Saving LoRA weights to {output_path}")
    
    # Save LoRA weights
    if hasattr(model.unet, 'peft_config'):
        model.unet.save_pretrained(output_path / "unet_lora")
        logger.info("UNet LoRA weights saved")
    
    # Save VAE if it was modified
    if hasattr(model.vae, 'peft_config'):
        model.vae.save_pretrained(output_path / "vae_lora")
        logger.info("VAE LoRA weights saved")
    
    # Save text encoder if it was modified
    if hasattr(model.text_encoder, 'peft_config'):
        model.text_encoder.save_pretrained(output_path / "text_encoder_lora")
        logger.info("Text encoder LoRA weights saved")
    
    # Save metadata
    if save_metadata:
        metadata = {
            "model_type": "lora_fine_tuned",
            "base_model": getattr(model, 'base_model', "unknown"),
            "lora_config": {
                "rank": getattr(model.unet.peft_config, 'r', None),
                "alpha": getattr(model.unet.peft_config, 'lora_alpha', None),
                "target_modules": getattr(model.unet.peft_config, 'target_modules', None)
            },
            "training_info": {
                "timestamp": torch.datetime.now().isoformat(),
                "total_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Metadata saved")


def load_lora_weights(
    base_model: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
    lora_path: Union[str, Path]
) -> Union[StableDiffusionPipeline, StableDiffusionXLPipeline]:
    """
    Load LoRA weights into a base model.
    
    Args:
        base_model: Base model pipeline
        lora_path: Path to LoRA weights
    
    Returns:
        Model with LoRA weights loaded
    """
    lora_path = Path(lora_path)
    
    if not lora_path.exists():
        raise ValueError(f"LoRA path does not exist: {lora_path}")
    
    logger.info(f"Loading LoRA weights from {lora_path}")
    
    # Load UNet LoRA weights
    unet_lora_path = lora_path / "unet_lora"
    if unet_lora_path.exists():
        base_model.unet = PeftModel.from_pretrained(base_model.unet, unet_lora_path)
        logger.info("UNet LoRA weights loaded")
    
    # Load VAE LoRA weights if they exist
    vae_lora_path = lora_path / "vae_lora"
    if vae_lora_path.exists():
        base_model.vae = PeftModel.from_pretrained(base_model.vae, vae_lora_path)
        logger.info("VAE LoRA weights loaded")
    
    # Load text encoder LoRA weights if they exist
    text_encoder_lora_path = lora_path / "text_encoder_lora"
    if text_encoder_lora_path.exists():
        base_model.text_encoder = PeftModel.from_pretrained(
            base_model.text_encoder, text_encoder_lora_path
        )
        logger.info("Text encoder LoRA weights loaded")
    
    return base_model


def merge_lora_weights(
    base_model: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
    lora_path: Union[str, Path],
    output_path: Union[str, Path]
) -> None:
    """
    Merge LoRA weights with base model and save as a standalone model.
    
    Args:
        base_model: Base model pipeline
        lora_path: Path to LoRA weights
        output_path: Path to save merged model
    """
    logger.info("Merging LoRA weights with base model...")
    
    # Load LoRA weights
    model_with_lora = load_lora_weights(base_model, lora_path)
    
    # Merge LoRA weights
    model_with_lora.unet = model_with_lora.unet.merge_and_unload()
    
    if hasattr(model_with_lora.vae, 'peft_config'):
        model_with_lora.vae = model_with_lora.vae.merge_and_unload()
    
    if hasattr(model_with_lora.text_encoder, 'peft_config'):
        model_with_lora.text_encoder = model_with_lora.text_encoder.merge_and_unload()
    
    # Save merged model
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)
    
    model_with_lora.save_pretrained(output_path)
    logger.info(f"Merged model saved to {output_path}")


def get_model_info(model: Union[StableDiffusionPipeline, StableDiffusionXLPipeline]) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: Model pipeline
    
    Returns:
        Dictionary with model information
    """
    info = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    }
    
    # Add LoRA info if available
    if hasattr(model.unet, 'peft_config'):
        info["lora_config"] = {
            "rank": model.unet.peft_config.r,
            "alpha": model.unet.peft_config.lora_alpha,
            "target_modules": model.unet.peft_config.target_modules
        }
    
    return info


def optimize_model_for_inference(
    model: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
    device: str = "cuda"
) -> Union[StableDiffusionPipeline, StableDiffusionXLPipeline]:
    """
    Optimize model for inference.
    
    Args:
        model: Model to optimize
        device: Target device
    
    Returns:
        Optimized model
    """
    logger.info("Optimizing model for inference...")
    
    # Move to device
    model.to(device)
    
    # Enable memory efficient attention if available
    if hasattr(torch.backends, "cuda") and torch.backends.cuda.is_built():
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
    
    # Set to evaluation mode
    model.eval()
    
    # Enable model compilation if available (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        try:
            model.unet = torch.compile(model.unet)
            logger.info("Model compiled successfully")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
    
    logger.info("Model optimization complete")
    return model


def validate_model_weights(
    model_path: Union[str, Path]
) -> Dict[str, Any]:
    """
    Validate model weights and return information.
    
    Args:
        model_path: Path to model
    
    Returns:
        Validation results
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    validation_results = {
        "path": str(model_path),
        "exists": True,
        "is_directory": model_path.is_dir(),
        "files": [],
        "has_lora": False,
        "has_base_model": False
    }
    
    if model_path.is_dir():
        # Check for model files
        model_files = list(model_path.glob("*"))
        validation_results["files"] = [f.name for f in model_files]
        
        # Check for LoRA weights
        lora_dirs = ["unet_lora", "vae_lora", "text_encoder_lora"]
        validation_results["has_lora"] = any(
            (model_path / d).exists() for d in lora_dirs
        )
        
        # Check for base model files
        base_model_files = ["config.json", "model_index.json", "scheduler", "tokenizer"]
        validation_results["has_base_model"] = any(
            (model_path / f).exists() for f in base_model_files
        )
    
    return validation_results


if __name__ == "__main__":
    # Test functions
    print("Model utilities module loaded successfully")

