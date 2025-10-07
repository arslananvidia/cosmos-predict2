#!/usr/bin/env python3
"""
Script to extract 'qwen_caption' from all JSON files in the source directory
and save them as text files in the destination directory.
"""

import json
import os
import glob
from pathlib import Path

def extract_qwen_captions():
    # Define source and destination directories
    source_dir = "/home/arslana/codes/cosmos-predict2/amol_its_it2/metas/v0/"
    dest_dir = "/home/arslana/codes/cosmos-predict2/datasets/v3/train/metas/"
    
    # Get all JSON files in the source directory
    json_files = glob.glob(os.path.join(source_dir, "*.json"))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    processed_count = 0
    error_count = 0
    
    for json_file in json_files:
        try:
            # Get the base filename without extension
            filename = os.path.basename(json_file)
            base_name = os.path.splitext(filename)[0]
            
            # Read the JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract qwen_caption
            qwen_caption = None
            
            # Look for qwen_caption in the JSON structure
            if 'windows' in data and isinstance(data['windows'], list):
                for window in data['windows']:
                    if 'qwen_caption' in window:
                        qwen_caption = window['qwen_caption']
                        break
            
            if qwen_caption is None:
                # Try to find qwen_caption at the root level
                if 'qwen_caption' in data:
                    qwen_caption = data['qwen_caption']
            
            if qwen_caption is not None:
                # Create the output text file path
                output_file = os.path.join(dest_dir, f"{base_name}.txt")
                
                # Write the caption to the text file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(qwen_caption)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files...")
            else:
                print(f"Warning: No qwen_caption found in {filename}")
                error_count += 1
                
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            error_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors encountered: {error_count} files")

if __name__ == "__main__":
    extract_qwen_captions() 