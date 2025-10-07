#!/bin/bash

# Create output directory
mkdir -p /home/arslana/codes/github/cosmos-predict2/datasets/metropolis/train/metas

# Source directory containing JSON files
SOURCE_DIR="/home/arslana/codes/github/cosmos-predict2/assets/watermarked/metas/v0"

# Output directory for text files
OUTPUT_DIR="/home/arslana/codes/github/cosmos-predict2/datasets/metropolis/train/metas"

# Process each JSON file
for json_file in "$SOURCE_DIR"/*.json; do
    if [ -f "$json_file" ]; then
        # Get base filename without extension
        basename=$(basename "$json_file" .json)
        
        # Extract qwen_caption using jq and save to text file
        jq -r '.windows[0].qwen_caption' "$json_file" > "$OUTPUT_DIR/${basename}.txt"
        
        echo "Processed: $basename"
    fi
done

echo "All captions extracted successfully!" 