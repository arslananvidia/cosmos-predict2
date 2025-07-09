# Batch Inference with LoRA Support for Metropolis Dataset

This directory contains scripts for running batch inference on the Metropolis dataset with both standard and LoRA inference modes.

## Files

- `run_batch_metropolis.py` - Main batch inference script with LoRA support
- `run_batch_examples.sh` - Interactive script with usage examples
- `README_BATCH_LORA.md` - This documentation file

## Features

### Standard Inference
- Uses `examples/video2world.py` script
- Default: 14B model with standard checkpoint
- Full model inference

### LoRA Inference
- Uses `examples/video2world_lora.py` script  
- Default: 2B model with LoRA checkpoint
- Parameter-efficient inference with LoRA adapters

## Usage

### Basic Commands

```bash
# Standard inference (14B model)
python customers/metropolis/run_batch_metropolis.py

# LoRA inference (2B model)
python customers/metropolis/run_batch_metropolis.py --lora
```

### Advanced Options

```bash
# LoRA inference with custom parameters
python customers/metropolis/run_batch_metropolis.py \
    --lora \
    --model_size 2B \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"

# Custom checkpoint path
python customers/metropolis/run_batch_metropolis.py \
    --lora \
    --dit_path "path/to/your/lora_checkpoint.pt"
```

### Interactive Examples

```bash
# Run interactive examples script
./customers/metropolis/run_batch_examples.sh
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--lora` | Enable LoRA inference mode | False |
| `--model_size` | Model size (2B or 14B) | 14B (standard), 2B (LoRA) |
| `--dit_path` | Custom checkpoint path | Auto-selected based on mode |
| `--lora_rank` | LoRA rank parameter | 16 |
| `--lora_alpha` | LoRA alpha parameter | 16 |
| `--lora_target_modules` | LoRA target modules | Default attention and MLP modules |

## Input/Output Structure

### Input Requirements
- **Frames**: `datasets/metropolis/benchmark/v2/frames_480_640/*.jpg`
- **Prompts**: `datasets/metropolis/benchmark/v2/prompts/*.txt`
- Frame and prompt files must have matching base names

### Output Structure
```
output/
├── metropolis_batch_standard/    # Standard inference outputs
│   ├── frame1.mp4
│   ├── frame2.mp4
│   └── ...
└── metropolis_batch_lora/        # LoRA inference outputs
    ├── frame1.mp4
    ├── frame2.mp4
    └── ...
```

## Default Configurations

### Standard Inference
- **Script**: `examples/video2world.py`
- **Model**: 14B
- **Checkpoint**: `checkpoints/posttraining/video2world/14b_metropolis/checkpoints/model/iter_000002000.pt`

### LoRA Inference  
- **Script**: `examples/video2world_lora.py`
- **Model**: 2B
- **Checkpoint**: `checkpoints/posttraining/video2world/2b_metropolis/checkpoints/model/iter_000002000.pt`
- **LoRA Parameters**:
  - Rank: 16
  - Alpha: 16
  - Target Modules: `q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2`

## Environment Setup

```bash
export NUM_GPUS=8
export PYTHONPATH=$(pwd)
```

## Performance Comparison

| Mode | Model Size | Memory Usage | Speed | Quality |
|------|------------|--------------|-------|---------|
| Standard | 14B | High | Slower | Full model quality |
| LoRA | 2B + adapters | Lower | Faster | Domain-adapted quality |

## Troubleshooting

### Common Issues

1. **Missing checkpoint files**: Ensure checkpoint paths exist
2. **Missing input files**: Verify frames and prompts directories exist
3. **Memory issues**: Reduce `NUM_GPUS` or use LoRA mode for lower memory usage
4. **LoRA parameter mismatch**: Ensure LoRA parameters match training configuration

### Getting Help

```bash
python customers/metropolis/run_batch_metropolis.py --help
```

## Examples

### Quick Start - LoRA Inference
```bash
# Set environment
export NUM_GPUS=8
export PYTHONPATH=$(pwd)

# Run LoRA batch inference
python customers/metropolis/run_batch_metropolis.py --lora
```

### Custom LoRA Configuration
```bash
python customers/metropolis/run_batch_metropolis.py \
    --lora \
    --model_size 2B \
    --dit_path "checkpoints/my_custom_lora.pt" \
    --lora_rank 32 \
    --lora_alpha 32
```

This setup provides flexible batch inference with both standard and LoRA modes while maintaining all the original functionality of the batch processing system. 