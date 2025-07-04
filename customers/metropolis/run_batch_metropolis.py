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
  
Usage:
  python run_batch_metropolis.py                    # Standard inference
  python run_batch_metropolis.py --lora             # LoRA inference
  python run_batch_metropolis.py --lora --model_size 2B  # LoRA with 2B model
"""

import os
import shlex
import subprocess
import argparse
from pathlib import Path
import glob

# ------------------------------------------------------------------
# CONFIG – adjust only if your paths / flags change
# ------------------------------------------------------------------
FRAMES_DIR  = Path("datasets/metropolis/benchmark/v2/frames_1280x704")
PROMPTS_DIR = Path("datasets/metropolis/benchmark/v2/prompts")
OUTPUT_DIR  = Path("output")
NUM_GPUS    = int(os.getenv("NUM_GPUS", 8))
TORCHRUN    = "torchrun"                             # or full path if needed

# Standard inference configuration
STANDARD_CONFIG = {
    "script": "examples/video2world.py",
    "model_size": "2B",
    "dit_path": "checkpoints/posttraining/video2world/2b_metropolis/checkpoints/model/iter_000002000.pt"
}

# LoRA inference configuration
LORA_CONFIG = {
    "script": "examples/video2world_lora.py",
    "model_size": "2B",
    "dit_path": "checkpoints/posttraining/video2world/2b_metropolis/checkpoints/model/iter_000003000.pt",
    "lora_rank": 16,
    "lora_alpha": 16,
    "lora_target_modules": "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
}

# ------------------------------------------------------------------
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch inference for Metropolis dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_batch_metropolis.py                    # Standard inference with 14B model
  python run_batch_metropolis.py --lora             # LoRA inference with 2B model
  python run_batch_metropolis.py --lora --model_size 2B  # LoRA with specific model size
        """
    )
    
    parser.add_argument(
        "--lora", 
        action="store_true", 
        help="Enable LoRA inference mode"
    )
    
    parser.add_argument(
        "--model_size", 
        choices=["2B", "14B"], 
        help="Model size (overrides default based on inference type)"
    )
    
    parser.add_argument(
        "--dit_path", 
        type=str, 
        help="Custom path to DiT checkpoint (overrides default)"
    )
    
    parser.add_argument(
        "--lora_rank", 
        type=int, 
        default=16, 
        help="LoRA rank (only used with --lora flag)"
    )
    
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=16, 
        help="LoRA alpha (only used with --lora flag)"
    )
    
    parser.add_argument(
        "--lora_target_modules", 
        type=str, 
        default="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        help="LoRA target modules (only used with --lora flag)"
    )
    
    return parser.parse_args()

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

def get_inference_config(args):
    """Get inference configuration based on arguments."""
    if args.lora:
        config = LORA_CONFIG.copy()
        # Override with command line arguments if provided
        if args.model_size:
            config["model_size"] = args.model_size
        if args.dit_path:
            config["dit_path"] = args.dit_path
        config["lora_rank"] = args.lora_rank
        config["lora_alpha"] = args.lora_alpha
        config["lora_target_modules"] = args.lora_target_modules
    else:
        config = STANDARD_CONFIG.copy()
        # Override with command line arguments if provided
        if args.model_size:
            config["model_size"] = args.model_size
        if args.dit_path:
            config["dit_path"] = args.dit_path
    
    return config

def build_command(config, frame_file, prompt, output_file, args):
    """Build the torchrun command based on configuration."""
    cmd = [
        TORCHRUN,
        f"--nproc_per_node={NUM_GPUS}",
        config["script"],
        "--model_size", config["model_size"],
        "--dit_path", config["dit_path"],
        "--input_path", str(frame_file),
        "--prompt", prompt,
        "--save_path", str(output_file),
        "--num_gpus", str(NUM_GPUS),
        "--offload_guardrail",
        "--offload_prompt_refiner",
    ]
    
    # Add LoRA-specific parameters if using LoRA inference
    if args.lora:
        cmd.extend([
            "--use_lora",
            "--lora_rank", str(config["lora_rank"]),
            "--lora_alpha", str(config["lora_alpha"]),
            "--lora_target_modules", config["lora_target_modules"]
        ])
    
    return cmd

def main() -> None:
    # Parse command line arguments
    args = parse_args()
    
    # Get inference configuration
    config = get_inference_config(args)
    
    # Print configuration
    inference_type = "LoRA" if args.lora else "Standard"
    print(f"{'='*60}")
    print(f"BATCH METROPOLIS INFERENCE - {inference_type} Mode")
    print(f"{'='*60}")
    print(f"Script: {config['script']}")
    print(f"Model Size: {config['model_size']}")
    print(f"Checkpoint: {config['dit_path']}")
    if args.lora:
        print(f"LoRA Rank: {config['lora_rank']}")
        print(f"LoRA Alpha: {config['lora_alpha']}")
        print(f"LoRA Target Modules: {config['lora_target_modules']}")
    print(f"{'='*60}\n")
    
    if not FRAMES_DIR.exists():
        raise FileNotFoundError(f"Frames directory not found: {FRAMES_DIR}")
    
    if not PROMPTS_DIR.exists():
        raise FileNotFoundError(f"Prompts directory not found: {PROMPTS_DIR}")

    # Create output directory with inference type suffix
    output_suffix = "_lora" if args.lora else "_standard"
    output_dir = OUTPUT_DIR / f"metropolis_batch{output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JPG files in the frames directory
    frame_files = list(FRAMES_DIR.glob("*.jpg"))
    
    if not frame_files:
        raise FileNotFoundError(f"No .jpg files found in {FRAMES_DIR}")

    print(f"Found {len(frame_files)} frame files to process")
    print(f"Output directory: {output_dir}\n")

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
        output_file = output_dir / f"{base_name}.mp4"
        
        # Build command
        cmd = build_command(config, frame_file, prompt, output_file, args)

        # Print nicely for the log
        pretty = " \\\n    ".join(map(shlex.quote, cmd))
        border = f"[{idx:02}/{len(frame_files)}]"
        print(f"\n{border} Processing: {base_name} ({inference_type})")
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
    print(f"BATCH PROCESSING COMPLETE - {inference_type} Mode")
    print(f"{'='*60}")
    print(f"Total files found: {len(frame_files)}")
    print(f"Successfully processed: {successful_jobs}")
    print(f"Failed: {len(failed_jobs)}")
    print(f"Output directory: {output_dir}")
    
    if failed_jobs:
        print(f"\nFailed jobs:")
        for failure in failed_jobs:
            print(f"  - {failure}")

if __name__ == "__main__":
    main()
