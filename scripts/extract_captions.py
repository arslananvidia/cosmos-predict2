#!/usr/bin/env python3

import json
import os
from pathlib import Path

# Define directories
source_dir = "/home/arslana/codes/github/cosmos-predict2/assets/watermarked/metas/v0"
output_dir = "/home/arslana/codes/github/cosmos-predict2/datasets/metropolis/train/metas"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Process each JSON file
for json_file in Path(source_dir).glob("*.json"):
    try:
        # Read and parse JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract qwen_caption from first window
        if 'windows' in data and len(data['windows']) > 0:
            caption = data['windows'][0].get('qwen_caption', '')
            
            # Create output text file
            output_file = Path(output_dir) / f"{json_file.stem}.txt"
            
            # Write caption to text file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            print(f"Processed: {json_file.stem}")
        else:
            print(f"Warning: No windows found in {json_file.name}")
            
    except Exception as e:
        print(f"Error processing {json_file.name}: {e}")

print("All captions extracted successfully!")