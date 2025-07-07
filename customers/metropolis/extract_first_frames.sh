#!/bin/bash

# Script to extract first frame from all videos in high quality
# Source: datasets/metropolis/benchmark/v2/videos_1280x704/
# Output: /home/arslana/codes/cosmos-predict2/datasets/metropolis/benchmark/v2/frames_1280x704/

SOURCE_DIR="datasets/metropolis/benchmark/v2/videos_1280x704"
OUTPUT_DIR="/home/arslana/codes/cosmos-predict2/datasets/metropolis/benchmark/v2/frames_1280x704"

echo "=================================="
echo "Extracting First Frames (High Quality)"
echo "=================================="
echo "Source: $SOURCE_DIR"
echo "Output: $OUTPUT_DIR"
echo "=================================="

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Counter for progress
count=0
total=$(find "$SOURCE_DIR" -name "*.mp4" | wc -l)

# Process each MP4 file
for video_file in "$SOURCE_DIR"/*.mp4; do
    if [ -f "$video_file" ]; then
        # Get base filename without extension
        base_name=$(basename "$video_file" .mp4)
        
        # Define output path
        output_file="$OUTPUT_DIR/${base_name}.jpg"
        
        # Increment counter
        ((count++))
        
        echo "[$count/$total] Processing: $base_name"
        
        # Extract first frame with high quality
        ffmpeg -i "$video_file" \
               -vframes 1 \
               -q:v 2 \
               -y \
               "$output_file" \
               -loglevel quiet
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully extracted: $base_name.jpg"
        else
            echo "✗ Failed to extract: $base_name"
        fi
    fi
done

echo "=================================="
echo "Extraction Complete!"
echo "Processed: $count videos"
echo "Output directory: $OUTPUT_DIR"
echo "==================================" 