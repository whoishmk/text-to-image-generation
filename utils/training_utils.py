"""
Training utilities for text-to-image generation.

This module provides functions for setting up training configurations,
optimizers, schedulers, and monitoring tools.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LinearLR
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logger = logging.getLogger(__name__)


def setup_training_args(
    config_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Setup training arguments from config file and command line.
    
    Args:
        config_path: Path to config file
        **kwargs: Override arguments
    
    Returns:
        Training configuration dictionary
    """
    config = {}
    
    # Load from config file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Override with command line arguments
    config.update(kwargs)
    
    # Set defaults
    config.setdefault("training", {})
    config.setdefault("model", {})
    config.setdefault("data", {})
    
    # Training defaults
    training_defaults = {
        "learning_rate": 1e-4,
        "batch_size": 4,
        "num_epochs": 100,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "eps": 1e-8,
        "mixed_precision": "fp16",
        "log_steps": 10,
        "save_steps": 500,
        "eval_steps": 100,
        "seed": 42,
        "output_dir": "./outputs",
        "log_with": "tensorboard"
    }
    
    for key, default_value in training_defaults.items():
        config["training"].setdefault(key, default_value)
    
    # Model defaults
    model_defaults = {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_type": "sdxl",
        "lora_rank": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"]
    }
    
    for key, default_value in model_defaults.items():
        config["model"].setdefault(key, default_value)
    
    # Data defaults
    data_defaults = {
        "train_data_dir": "./data/train",
        "validation_data_dir": "./data/val",
        "resolution": 1024,
        "max_length": 77,
        "num_workers": 4
    }
    
    for key, default_value in data_defaults.items():
        config["data"].setdefault(key, default_value)
    
    return config


def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    optimizer_type: str = "adamw"
) -> torch.optim.Optimizer:
    """
    Create optimizer for training.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        eps: Epsilon value
        optimizer_type: Type of optimizer
    
    Returns:
        Optimizer instance
    """
    # Get trainable parameters
    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=eps
        )
    elif optimizer_type.lower() == "adam":
        optimizer = Adam(
            trainable_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=eps
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    logger.info(f"Created {optimizer_type} optimizer with {len(trainable_params)} trainable parameters")
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    steps_per_epoch: int,
    warmup_steps: int = 0,
    scheduler_type: str = "cosine"
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        num_epochs: Number of training epochs
        steps_per_epoch: Steps per epoch
        warmup_steps: Number of warmup steps
        scheduler_type: Type of scheduler
    
    Returns:
        Learning rate scheduler
    """
    total_steps = num_epochs * steps_per_epoch
    
    if scheduler_type.lower() == "cosine":
        if warmup_steps > 0:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    elif scheduler_type.lower() == "linear":
        if warmup_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    
    elif scheduler_type.lower() == "constant":
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    logger.info(f"Created {scheduler_type} scheduler with {warmup_steps} warmup steps")
    return scheduler


def log_training_info(config: Dict[str, Any]) -> None:
    """
    Log training configuration information.
    
    Args:
        config: Training configuration
    """
    console.print("\n[bold blue]Training Configuration[/bold blue]")
    
    # Create table for training config
    training_table = Table(title="Training Settings")
    training_table.add_column("Parameter", style="cyan")
    training_table.add_column("Value", style="green")
    
    for key, value in config["training"].items():
        training_table.add_row(key, str(value))
    
    console.print(training_table)
    
    # Create table for model config
    model_table = Table(title="Model Settings")
    model_table.add_column("Parameter", style="cyan")
    model_table.add_column("Value", style="green")
    
    for key, value in config["model"].items():
        model_table.add_row(key, str(value))
    
    console.print(model_table)
    
    # Create table for data config
    data_table = Table(title="Data Settings")
    data_table.add_column("Parameter", style="cyan")
    data_table.add_column("Value", style="green")
    
    for key, value in config["data"].items():
        data_table.add_row(key, str(value))
    
    console.print(data_table)


def setup_logging(
    output_dir: str,
    log_level: str = "INFO"
) -> None:
    """
    Setup logging configuration.
    
    Args:
        output_dir: Output directory for logs
        log_level: Logging level
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Setup file handler
    log_file = output_path / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Setup formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging setup complete. Log file: {log_file}")


def save_training_config(
    config: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Save training configuration to file.
    
    Args:
        config: Training configuration
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    config_file = output_path / "training_config.yaml"
    
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Training configuration saved to {config_file}")


def create_progress_bar(description: str = "Training"):
    """
    Create a rich progress bar.
    
    Args:
        description: Description for the progress bar
    
    Returns:
        Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )


def log_training_step(
    step: int,
    loss: float,
    learning_rate: float,
    epoch: int,
    writer=None
) -> None:
    """
    Log training step information.
    
    Args:
        step: Current training step
        loss: Loss value
        learning_rate: Current learning rate
        epoch: Current epoch
        writer: TensorBoard writer
    """
    # Console logging
    console.print(f"Step {step}: loss={loss:.4f}, lr={learning_rate:.2e}, epoch={epoch}")
    
    # TensorBoard logging
    if writer:
        writer.add_scalar("Loss/train", loss, step)
        writer.add_scalar("LR", learning_rate, step)
        writer.add_scalar("Epoch", epoch, step)


def log_validation_results(
    step: int,
    metrics: Dict[str, float],
    writer=None
) -> None:
    """
    Log validation results.
    
    Args:
        step: Current training step
        metrics: Validation metrics
        writer: TensorBoard writer
    """
    # Console logging
    console.print(f"\n[bold green]Validation Results at Step {step}[/bold green]")
    
    for metric_name, metric_value in metrics.items():
        console.print(f"  {metric_name}: {metric_value:.4f}")
        
        # TensorBoard logging
        if writer:
            writer.add_scalar(f"Validation/{metric_name}", metric_value, step)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    loss: float,
    output_dir: str,
    filename: str = None
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        output_dir: Output directory
        filename: Checkpoint filename
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
    
    checkpoint_path = output_path / filename
    
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": {
            "epoch": epoch,
            "step": step,
            "loss": loss
        }
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
    
    Returns:
        Checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(f"Resuming from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    
    return checkpoint


def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Calculate model size information.
    
    Args:
        model: Model to analyze
    
    Returns:
        Dictionary with size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Calculate size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    total_size_mb = total_size / (1024 * 1024)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "total_size_mb": total_size_mb,
        "param_size_mb": param_size / (1024 * 1024),
        "buffer_size_mb": buffer_size / (1024 * 1024)
    }


if __name__ == "__main__":
    # Test functions
    config = setup_training_args()
    print("Training utilities module loaded successfully")

