import os
import shutil
import json
from pathlib import Path
import random

def organize_dataset(metadata_file, images_dir, train_dir, val_dir, train_split=0.9):
    """Organize dataset into train/validation splits"""
    
    # Read metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = [json.loads(line) for line in f]
    
    print(f"Total samples: {len(metadata)}")
    
    # Shuffle and split
    random.shuffle(metadata)
    split_idx = int(len(metadata) * train_split)
    train_data = metadata[:split_idx]
    val_data = metadata[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Copy images and create metadata files
    def copy_data(data, target_dir, metadata_file):
        os.makedirs(target_dir, exist_ok=True)
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for item in data:
                # Copy image
                src_path = os.path.join(images_dir, item['file_name'])
                dst_path = os.path.join(target_dir, item['file_name'])
                
                if os.path.exists(src_path):
                    shutil.copy2(src_path, dst_path)
                    f.write(json.dumps(item) + '\n')
                else:
                    print(f"Warning: Image not found: {src_path}")
    
    # Organize train data
    print("Organizing training data...")
    copy_data(train_data, train_dir, os.path.join(train_dir, 'metadata.jsonl'))
    
    # Organize validation data
    print("Organizing validation data...")
    copy_data(val_data, val_dir, os.path.join(val_dir, 'metadata.jsonl'))
    
    print("Dataset organization complete!")
    print(f"Training data: {train_dir}")
    print(f"Validation data: {val_dir}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Organize dataset
    organize_dataset(
        metadata_file="data/metadata.jsonl",
        images_dir="dataset/Images",
        train_dir="data/train",
        val_dir="data/val",
        train_split=0.9  # 90% train, 10% validation
    )
