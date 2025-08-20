#!/usr/bin/env python3
"""
Complete LoRA Training Script for Stable Diffusion XL
Implements full training loop with dataset
"""

import os
import yaml
import torch
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextImageDataset(torch.utils.data.Dataset):
    """Dataset for text-image pairs."""
    
    def __init__(self, data_dir: str, tokenizer, resolution: int = 1024, max_length: int = 77):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.max_length = max_length
        
        # Load metadata
        metadata_file = self.data_dir / "metadata.jsonl"
        self.samples = []
        with open(metadata_file, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load and preprocess image
        image_path = self.data_dir / sample["file_name"]
        image = Image.open(image_path).convert("RGB")
        
        # Resize and center crop
        image = self.preprocess_image(image)
        
        # Tokenize text
        text = sample["text"]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze()
        }
    
    def preprocess_image(self, image):
        """Preprocess image for training."""
        # Resize to resolution
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC to CHW
        
        # Normalize to [-1, 1]
        image = image * 2.0 - 1.0
        
        return image

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

def create_dataloader(dataset, batch_size: int, num_workers: int = 0):
    """Create data loader."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

def train_epoch(pipeline, dataloader, optimizer, scheduler, device, epoch: int, config: Dict[str, Any]):
    """Train for one epoch."""
    pipeline.unet.train()  # Only train the UNet
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = pipeline.text_encoder(input_ids, attention_mask=attention_mask)[0]
            text_embeddings_2 = pipeline.text_encoder_2(input_ids, attention_mask=attention_mask)[0]
        
        # Sample noise
        batch_size = pixel_values.shape[0]
        latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * pipeline.vae.config.scaling_factor
        
        # Sample random timesteps
        timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (batch_size,), device=device)
        timesteps = timesteps.long()
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        # Ensure both text embeddings have the same dimensions
        if text_embeddings_2.dim() == 2:
            # Expand to match sequence length: [batch, 1280] -> [batch, seq_len, 1280]
            text_embeddings_2 = text_embeddings_2.unsqueeze(1).expand(-1, text_embeddings.shape[1], -1)
        
        combined_embeddings = torch.cat([text_embeddings, text_embeddings_2], dim=-1)
        
        # Call the base model directly to avoid PEFT wrapper issues
        # For SDXL, we need to provide added_cond_kwargs
        noise_pred = pipeline.unet.base_model(
            noisy_latents,
            timesteps,
            encoder_hidden_states=combined_embeddings,
            added_cond_kwargs={
                "text_embeds": text_embeddings_2.mean(dim=1),  # Average across sequence
                "time_ids": torch.zeros(noisy_latents.shape[0], 6, device=device)  # SDXL time embeddings
            }
        ).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config["training"]["gradient_clip_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(pipeline.unet.parameters(), config["training"]["gradient_clip_norm"])
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
        
        # Log every log_steps
        if batch_idx % config["training"]["log_steps"] == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)

def save_checkpoint(pipeline, optimizer, scheduler, epoch: int, loss: float, output_dir: Path):
    """Save training checkpoint."""
    checkpoint_dir = output_dir / f"checkpoint-{epoch}"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save LoRA weights
    pipeline.unet.save_pretrained(str(checkpoint_dir / "lora_weights"))
    
    # Save optimizer and scheduler
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, checkpoint_dir / "training_state.pt")
    
    logger.info(f"Checkpoint saved to {checkpoint_dir}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Full LoRA Training for SDXL")
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
    
    logger.info("Starting full LoRA training...")
    
    try:
        # Import required libraries
        from diffusers import StableDiffusionXLPipeline
        from diffusers.utils import logging as diffusers_logging
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPTokenizer
        
        # Set diffusers logging level
        diffusers_logging.set_verbosity_info()
        
        logger.info("Loading base SDXL model...")
        
        # Load base model
        model_id = config["model"]["base_model"]
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if device.type == "cuda" else None
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
        
        # Setup datasets
        logger.info("Setting up datasets...")
        train_dataset = TextImageDataset(
            config["data"]["train_data_dir"],
            pipeline.tokenizer,
            config["data"]["resolution"],
            config["data"]["max_length"]
        )
        
        val_dataset = TextImageDataset(
            config["data"]["validation_data_dir"],
            pipeline.tokenizer,
            config["data"]["resolution"],
            config["data"]["max_length"]
        )
        
        # Create dataloaders
        train_dataloader = create_dataloader(
            train_dataset,
            config["training"]["batch_size"],
            config["data"]["num_workers"]
        )
        
        val_dataloader = create_dataloader(
            val_dataset,
            config["training"]["batch_size"],
            config["data"]["num_workers"]
        )
        
        # Setup optimizer
        logger.info("Setting up optimizer...")
        
        # Ensure numeric values are properly typed
        lr = float(config["training"]["learning_rate"])
        weight_decay = float(config["training"]["weight_decay"])
        eps = float(config["training"]["eps"])
        
        logger.info(f"Learning rate: {lr}, Weight decay: {weight_decay}, Eps: {eps}")
        
        optimizer = torch.optim.AdamW(
            pipeline.unet.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=eps
        )
        
        # Setup scheduler
        logger.info("Setting up scheduler...")
        total_steps = len(train_dataloader) * config["training"]["num_epochs"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=lr * 0.1
        )
        
        # Training loop
        logger.info("Starting training loop...")
        best_loss = float('inf')
        
        for epoch in range(config["training"]["num_epochs"]):
            logger.info(f"Starting epoch {epoch + 1}/{config['training']['num_epochs']}")
            
            # Train
            train_loss = train_epoch(
                pipeline, train_dataloader, optimizer, scheduler, device, epoch + 1, config
            )
            
            # Save checkpoint
            if (epoch + 1) % config["training"]["save_steps"] == 0:
                save_checkpoint(pipeline, optimizer, scheduler, epoch + 1, train_loss, output_dir)
            
            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                save_checkpoint(pipeline, optimizer, scheduler, epoch + 1, train_loss, output_dir / "best")
                logger.info(f"New best model saved with loss: {best_loss:.4f}")
            
            logger.info(f"Epoch {epoch + 1} completed. Train Loss: {train_loss:.4f}")
        
        # Save final model
        logger.info("Training completed! Saving final model...")
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        pipeline.unet.save_pretrained(str(final_dir / "lora_weights"))
        
        logger.info("🎉 Full LoRA training completed successfully!")
        logger.info(f"Final model saved to {final_dir}")
        logger.info(f"Best loss achieved: {best_loss:.4f}")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
