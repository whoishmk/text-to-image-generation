import os
import csv
import json
from pathlib import Path

def convert_csv_to_jsonl(csv_file, output_file, images_dir):
    """Convert CSV captions file to JSONL format expected by training script"""
    
    # Check if images directory exists
    if not os.path.exists(images_dir):
        print(f"Images directory {images_dir} not found!")
        return
    
    # Get list of available images
    available_images = set()
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        available_images.update([f.name for f in Path(images_dir).glob(ext)])
    
    print(f"Found {len(available_images)} images in {images_dir}")
    
    # Convert CSV to JSONL
    converted_count = 0
    skipped_count = 0
    
    with open(csv_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        
        for row in reader:
            image_name = row['image'].strip()
            caption = row['caption'].strip()
            
            # Check if image exists
            if image_name in available_images:
                json_line = {
                    "file_name": image_name,
                    "text": caption
                }
                f_out.write(json.dumps(json_line) + '\n')
                converted_count += 1
            else:
                skipped_count += 1
                print(f"Skipping {image_name} - file not found")
    
    print(f"Conversion complete!")
    print(f"Converted: {converted_count} entries")
    print(f"Skipped: {skipped_count} entries")
    print(f"Output saved to: {output_file}")

def create_training_structure():
    """Create the training directory structure"""
    
    # Create data directories
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/val", exist_ok=True)
    
    print("Created training directory structure:")
    print("  data/train/")
    print("  data/val/")

if __name__ == "__main__":
    # Convert dataset
    csv_file = "dataset/captions.txt"
    output_file = "data/metadata.jsonl"
    images_dir = "dataset/Images"
    
    print("Converting dataset format...")
    convert_csv_to_jsonl(csv_file, output_file, images_dir)
    
    # Create training structure
    print("\nCreating training directory structure...")
    create_training_structure()
    
    print("\nDataset preparation complete!")
    print("You can now run the training script:")
    print("python training/train_lora.py --config configs/training_config.yaml")

