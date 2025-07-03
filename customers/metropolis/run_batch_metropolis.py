#!/usr/bin/env python3
# run_batch_metropolis.py
"""
Batch-launch video2world jobs for the Metropolis benchmark.

Automatically processes all frames from datasets/metropolis/benchmark/v2/frames_480_640
and matches them with corresponding prompts from datasets/metropolis/benchmark/v2/prompts.

Assumes:
  • torchrun is in PATH
  • NUM_GPUS is set below (or override via env)
  • Frame files are .jpg and prompt files are .txt with matching base names
"""

import os
import shlex
import subprocess
from pathlib import Path
import glob

# ------------------------------------------------------------------
# CONFIG – adjust only if your paths / flags change
# ------------------------------------------------------------------
FRAMES_DIR  = Path("datasets/metropolis/benchmark/v2/frames_480_640")
PROMPTS_DIR = Path("datasets/metropolis/benchmark/v2/prompts")
OUTPUT_DIR  = Path("output")
NUM_GPUS    = int(os.getenv("NUM_GPUS", 8))
MODEL_SIZE  = "14B"  # Changed from 14B to 2B based on the checkpoint path in original
DIT_PATH    = (
    "checkpoints/posttraining/video2world/14b_metropolis/"
    "checkpoints/model/iter_000002000.pt"
)
TORCHRUN    = "torchrun"                             # or full path if needed
SCRIPT      = "examples/video2world.py"              # entry script

# ------------------------------------------------------------------
def load_prompt(prompt_file: Path) -> str:
    """Load prompt text from file, returning first non-empty line."""
    try:
        with prompt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # Return first non-empty line
                    return line
        return ""  # Return empty string if no non-empty lines found
    except Exception as e:
        print(f"Warning: Could not read prompt file {prompt_file}: {e}")
        return ""

def main() -> None:
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames directory not found: {FRAMES_DIR}")
    
    if not PROMPTS_DIR.exists():
        raise FileNotFoundError(f"Prompts directory not found: {PROMPTS_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all JPG files in the frames directory
    frame_files = list(FRAMES_DIR.glob("*.jpg"))
    
    if not frame_files:
        raise FileNotFoundError(f"No .jpg files found in {FRAMES_DIR}")

    print(f"Found {len(frame_files)} frame files to process")

    # Process each frame file
    successful_jobs = 0
    failed_jobs = []

    for idx, frame_file in enumerate(sorted(frame_files), 1):
        # Get base name without extension
        base_name = frame_file.stem
        
        # Find corresponding prompt file
        prompt_file = PROMPTS_DIR / f"{base_name}.txt"
        
        if not prompt_file.exists():
            print(f"Warning: No prompt file found for {base_name}, skipping...")
            failed_jobs.append(f"{base_name} - missing prompt file")
            continue
        
        # Load prompt
        prompt = load_prompt(prompt_file)
        if not prompt:
            print(f"Warning: Empty or unreadable prompt for {base_name}, skipping...")
            failed_jobs.append(f"{base_name} - empty/unreadable prompt")
            continue
        
        # Generate output path
        output_file = OUTPUT_DIR / f"{base_name}.mp4"
        
        # Compose the torchrun command
        cmd = [
            TORCHRUN,
            f"--nproc_per_node={NUM_GPUS}",
            SCRIPT,
            "--model_size", MODEL_SIZE,
            "--dit_path", DIT_PATH,
            "--input_path", str(frame_file),
            "--prompt", prompt,
            "--save_path", str(output_file),
            "--num_gpus", str(NUM_GPUS),
            "--offload_guardrail",
            "--offload_prompt_refiner",
        ]

        # Print nicely for the log
        pretty = " \\\n    ".join(map(shlex.quote, cmd))
        border = f"[{idx:02}/{len(frame_files)}]"
        print(f"\n{border} Processing: {base_name}")
        print(f"Input: {frame_file}")
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print(f"Output: {output_file}")
        print(f"Command:\n    {pretty}\n")

        # Launch
        try:
            subprocess.run(cmd, check=True)
            successful_jobs += 1
            print(f"✓ Successfully completed {base_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to process {base_name}: {e}")
            failed_jobs.append(f"{base_name} - subprocess error: {e}")
            continue

    # Print summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files found: {len(frame_files)}")
    print(f"Successfully processed: {successful_jobs}")
    print(f"Failed: {len(failed_jobs)}")
    
    if failed_jobs:
        print(f"\nFailed jobs:")
        for failure in failed_jobs:
            print(f"  - {failure}")

if __name__ == "__main__":
    main()
