#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Stable Diffusion Models

This script implements parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
for Stable Diffusion models. It supports both SD 2.1 and SDXL models.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline
)
from diffusers.optimization import get_scheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from datasets import load_dataset
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import (
    create_dataloader,
    collate_fn,
    preprocess_image,
    tokenize_prompt
)
from utils.model_utils import (
    load_base_model,
    setup_lora_config,
    save_lora_weights
)
from utils.training_utils import (
    setup_training_args,
    create_optimizer,
    create_scheduler,
    log_training_info
)

logger = get_logger(__name__)


class LoRATrainer:
    """Main trainer class for LoRA fine-tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            mixed_precision=config["training"]["mixed_precision"],
            log_with=config["training"]["log_with"],
            project_dir=config["training"]["output_dir"]
        )
        
        # Set seed for reproducibility
        set_seed(config["training"]["seed"])
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.setup_models()
        self.setup_data()
        self.setup_optimizer()
        self.setup_scheduler()
        
        # Prepare for training
        self.prepare_training()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        
        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(log_dir=self.config["training"]["output_dir"])
        else:
            self.writer = None
            
    def setup_models(self):
        """Initialize and setup all models."""
        logger.info("Setting up models...")
        
        # Load base model
        self.base_model = load_base_model(
            self.config["model"]["base_model"],
            self.config["model"]["model_type"]
        )
        
        # Setup LoRA configuration
        lora_config = setup_lora_config(
            self.config["model"]["lora_rank"],
            self.config["model"]["lora_alpha"],
            self.config["model"]["target_modules"]
        )
        
        # Apply LoRA to UNet
        self.unet = get_peft_model(self.base_model.unet, lora_config)
        
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config["model"]["base_model"],
            subfolder="tokenizer"
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config["model"]["base_model"],
            subfolder="text_encoder"
        )
        
        # Freeze text encoder and vae
        self.text_encoder.requires_grad_(False)
        self.base_model.vae.requires_grad_(False)
        
        # Setup noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config["model"]["base_model"],
            subfolder="scheduler"
        )
        
        logger.info("Models setup complete!")
        
    def setup_data(self):
        """Setup training and validation datasets."""
        logger.info("Setting up datasets...")
        
        # Load dataset
        dataset = load_dataset(
            "imagefolder",
            data_dir=self.config["data"]["train_data_dir"],
            split="train"
        )
        
        # Create dataloader
        self.train_dataloader = create_dataloader(
            dataset,
            self.tokenizer,
            self.config["data"]["resolution"],
            self.config["data"]["max_length"],
            self.config["training"]["batch_size"],
            self.config["training"]["num_workers"]
        )
        
        logger.info(f"Dataset loaded with {len(dataset)} samples")
        
    def setup_optimizer(self):
        """Setup optimizer for training."""
        logger.info("Setting up optimizer...")
        
        # Get trainable parameters
        trainable_params = list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config["training"]["learning_rate"],
            betas=(0.9, 0.999),
            weight_decay=self.config["training"]["weight_decay"],
            eps=self.config["training"]["eps"]
        )
        
    def setup_scheduler(self):
        """Setup learning rate scheduler."""
        logger.info("Setting up scheduler...")
        
        self.lr_scheduler = create_scheduler(
            self.optimizer,
            self.config["training"]["num_epochs"],
            len(self.train_dataloader),
            self.config["training"]["warmup_steps"]
        )
        
    def prepare_training(self):
        """Prepare models and optimizers for training."""
        logger.info("Preparing for training...")
        
        # Prepare everything with accelerator
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        )
        
        # Move text encoder and vae to device
        self.text_encoder.to(self.accelerator.device)
        self.base_model.vae.to(self.accelerator.device)
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends, "cuda") and torch.backends.cuda.is_built():
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            
    def training_step(self, batch):
        """Single training step."""
        # Get batch data
        pixel_values = batch["pixel_values"].to(self.accelerator.device)
        input_ids = batch["input_ids"].to(self.accelerator.device)
        
        # Convert images to latent space
        latents = self.base_model.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.base_model.vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(input_ids)[0]
            
        # Predict noise
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
        
        return loss
        
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        progress_bar = self.accelerator.init_progress_bar()
        global_step = 0
        
        for epoch in range(self.config["training"]["num_epochs"]):
            self.unet.train()
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    loss = self.training_step(batch)
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                        
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                # Log progress
                if global_step % self.config["training"]["log_steps"] == 0:
                    self.log_training_step(global_step, loss, epoch)
                    
                # Save checkpoint
                if global_step % self.config["training"]["save_steps"] == 0:
                    self.save_checkpoint(global_step)
                    
                # Evaluation
                if global_step % self.config["training"]["eval_steps"] == 0:
                    self.evaluate(global_step)
                    
                global_step += 1
                progress_bar.update(1)
                
        # Save final model
        self.save_checkpoint(global_step, is_final=True)
        logger.info("Training completed!")
        
    def log_training_step(self, global_step: int, loss: torch.Tensor, epoch: int):
        """Log training information."""
        if self.accelerator.is_main_process and self.writer:
            self.writer.add_scalar("Loss/train", loss.item(), global_step)
            self.writer.add_scalar("LR", self.lr_scheduler.get_last_lr()[0], global_step)
            self.writer.add_scalar("Epoch", epoch, global_step)
            
        logger.info(f"Step {global_step}: loss = {loss.item():.4f}")
        
    def save_checkpoint(self, global_step: int, is_final: bool = False):
        """Save model checkpoint."""
        if self.accelerator.is_main_process:
            checkpoint_dir = Path(self.config["training"]["output_dir"]) / f"checkpoint-{global_step}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save LoRA weights
            save_lora_weights(self.unet, checkpoint_dir)
            
            # Save training state
            self.accelerator.save_state(checkpoint_dir)
            
            logger.info(f"Checkpoint saved to {checkpoint_dir}")
            
    def evaluate(self, global_step: int):
        """Run evaluation on validation set."""
        logger.info(f"Running evaluation at step {global_step}")
        
        # Generate sample images
        self.generate_sample_images(global_step)
        
    def generate_sample_images(self, global_step: int):
        """Generate sample images for evaluation."""
        if not self.accelerator.is_main_process:
            return
            
        # Set to eval mode
        self.unet.eval()
        
        # Sample prompts for evaluation
        eval_prompts = [
            "a beautiful landscape with mountains and lake",
            "a cute cat sitting on a windowsill",
            "a futuristic city skyline at night"
        ]
        
        with torch.no_grad():
            for i, prompt in enumerate(eval_prompts):
                # Generate image
                image = self.generate_image(prompt)
                
                # Save image
                if self.writer:
                    self.writer.add_image(f"eval_{i}", image, global_step, dataformats="HWC")
                    
        # Set back to train mode
        self.unet.train()
        
    def generate_image(self, prompt: str) -> np.ndarray:
        """Generate a single image from prompt."""
        # This is a simplified generation - in practice, you'd use the full pipeline
        # For now, we'll return a placeholder
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Stable Diffusion")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override output directory if specified
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
        
    # Create trainer and start training
    trainer = LoRATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
