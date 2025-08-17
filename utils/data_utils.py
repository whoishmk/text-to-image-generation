"""
Data utilities for text-to-image generation training.

This module provides functions for loading, preprocessing, and creating datasets
for training Stable Diffusion models with LoRA fine-tuning.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

from transformers import CLIPTokenizer
from datasets import Dataset


class TextImageDataset(Dataset):
    """Dataset class for text-image pairs."""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer: CLIPTokenizer,
        resolution: int = 512,
        max_length: int = 77,
        center_crop: bool = True,
        random_flip: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.max_length = max_length
        self.center_crop = center_crop
        self.random_flip = random_flip
        
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> List[Dict[str, str]]:
        """Load dataset metadata."""
        metadata_file = self.data_dir / "metadata.jsonl"
        
        if metadata_file.exists():
            # Load from JSONL file
            metadata = []
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        metadata.append(json.loads(line))
            return metadata
        else:
            # Auto-generate metadata from directory structure
            return self._generate_metadata()
    
    def _generate_metadata(self) -> List[Dict[str, str]]:
        """Generate metadata from directory structure."""
        metadata = []
        
        # Look for common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for image_file in self.data_dir.rglob('*'):
            if image_file.suffix.lower() in image_extensions:
                # Try to find corresponding caption file
                caption_file = image_file.with_suffix('.txt')
                caption = ""
                
                if caption_file.exists():
                    with open(caption_file, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                else:
                    # Use filename as caption
                    caption = image_file.stem.replace('_', ' ').replace('-', ' ')
                
                metadata.append({
                    "file_name": str(image_file.relative_to(self.data_dir)),
                    "text": caption
                })
        
        return metadata
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.metadata[idx]
        
        # Load and preprocess image
        image_path = self.data_dir / item["file_name"]
        image = Image.open(image_path).convert("RGB")
        image = preprocess_image(
            image,
            self.resolution,
            self.center_crop,
            self.random_flip
        )
        
        # Tokenize text
        text = item["text"]
        tokens = tokenize_prompt(self.tokenizer, text, self.max_length)
        
        return {
            "pixel_values": image,
            "input_ids": tokens,
            "text": text
        }


def preprocess_image(
    image: Image.Image,
    resolution: int,
    center_crop: bool = True,
    random_flip: bool = True
) -> torch.Tensor:
    """Preprocess image for training."""
    # Resize
    if center_crop:
        # Center crop to square
        width, height = image.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        image = image.crop((left, top, right, bottom))
    
    # Resize to target resolution
    image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    
    # Random horizontal flip
    if random_flip and np.random.random() > 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    
    # Convert to tensor and normalize
    image = np.array(image).astype(np.float32) / 127.5 - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    
    return image


def tokenize_prompt(
    tokenizer: CLIPTokenizer,
    text: str,
    max_length: int
) -> torch.Tensor:
    """Tokenize text prompt."""
    tokens = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    
    return tokens.input_ids.squeeze()


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "texts": [item["text"] for item in batch]
    }


def create_dataloader(
    dataset: Dataset,
    tokenizer: CLIPTokenizer,
    resolution: int,
    max_length: int,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create DataLoader for training."""
    # Convert HuggingFace dataset to PyTorch dataset
    torch_dataset = TextImageDataset(
        data_dir=dataset.data_dir,
        tokenizer=tokenizer,
        resolution=resolution,
        max_length=max_length
    )
    
    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )


def validate_dataset(data_dir: str) -> Dict[str, any]:
    """Validate dataset structure and return statistics."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Dataset directory {data_dir} does not exist")
    
    # Count files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    caption_files = []
    
    for file_path in data_path.rglob('*'):
        if file_path.is_file():
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
            elif file_path.suffix == '.txt':
                caption_files.append(file_path)
    
    # Check for metadata file
    metadata_file = data_path / "metadata.jsonl"
    has_metadata = metadata_file.exists()
    
    # Validate image-caption pairs
    valid_pairs = 0
    missing_captions = 0
    
    for image_file in image_files:
        caption_file = image_file.with_suffix('.txt')
        if caption_file.exists():
            valid_pairs += 1
        else:
            missing_captions += 1
    
    return {
        "total_images": len(image_files),
        "total_captions": len(caption_files),
        "valid_pairs": valid_pairs,
        "missing_captions": missing_captions,
        "has_metadata": has_metadata,
        "image_extensions": list(image_extensions),
        "dataset_size_mb": sum(f.stat().st_size for f in image_files) / (1024 * 1024)
    }


def create_sample_dataset(
    output_dir: str,
    num_samples: int = 10,
    resolution: int = 512
) -> None:
    """Create a sample dataset for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create sample images and captions
    sample_data = [
        ("a beautiful sunset over mountains", "sunset_mountains"),
        ("a cute cat playing with yarn", "cat_yarn"),
        ("a futuristic city skyline", "futuristic_city"),
        ("a peaceful forest scene", "forest_scene"),
        ("a vintage car on a country road", "vintage_car"),
        ("a colorful butterfly on a flower", "butterfly_flower"),
        ("a cozy coffee shop interior", "coffee_shop"),
        ("a stormy ocean waves", "ocean_waves"),
        ("a magical forest with glowing mushrooms", "magical_forest"),
        ("a modern minimalist bedroom", "minimalist_bedroom")
    ]
    
    metadata = []
    
    for i, (caption, filename) in enumerate(sample_data[:num_samples]):
        # Create a simple colored image
        image = Image.new('RGB', (resolution, resolution), 
                         color=(np.random.randint(0, 255), 
                                np.random.randint(0, 255), 
                                np.random.randint(0, 255)))
        
        # Save image
        image_path = output_path / f"{filename}.png"
        image.save(image_path)
        
        # Save caption
        caption_path = output_path / f"{filename}.txt"
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(caption)
        
        # Add to metadata
        metadata.append({
            "file_name": f"{filename}.png",
            "text": caption
        })
    
    # Save metadata
    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    
    print(f"Sample dataset created in {output_dir} with {len(metadata)} samples")


if __name__ == "__main__":
    # Test dataset creation
    create_sample_dataset("sample_data", num_samples=5)
    
    # Test validation
    stats = validate_dataset("sample_data")
    print("Dataset statistics:", stats)
