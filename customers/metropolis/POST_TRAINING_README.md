# Cosmos-Predict2 Post-Training Configuration Guide

This guide provides a comprehensive overview of the post-training methodology, configuration system, and current settings for Cosmos-Predict2 models.

## Overview

Cosmos-Predict2 supports post-training of Video2World models using a flexible configuration system built on Hydra/OmegaConf. The training command used is:

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

## Configuration System Architecture

The configuration system is hierarchically organized to support flexible experimentation:

```
cosmos_predict2/configs/base/
├── config.py                   # Main configuration class and registration
├── defaults/                   # Default configuration groups
│   ├── callbacks.py            # Training callbacks (logging, monitoring)
│   ├── checkpoint.py           # Checkpoint saving/loading settings
│   ├── data.py                 # Dataset and dataloader configurations
│   ├── ema.py                  # Exponential Moving Average settings
│   ├── model.py                # Model architecture configurations
│   ├── optimizer.py            # Optimizer configurations
│   └── scheduler.py            # Learning rate scheduler configurations
└── experiment/                 # Experiment-specific configurations
    ├── groot.py                # GR00T robot experiments
    ├── cosmos_nemo_assets.py   # Cosmos Nemo Asset experiments
    └── agibot_head_center_fisheye_color.py  # AgiBOT experiments
```

## Current Default Settings

### Base Configuration (`config.py`)

- **Trainer Type**: `ImaginaireTrainer`
- **Max Iterations**: 400,000
- **Logging Frequency**: Every 10 iterations
- **Validation Frequency**: Every 100 iterations (disabled by default)
- **Project Name**: `cosmos_predict2`
- **Job Group**: `debug`

### Model Configurations (`defaults/model.py`)

#### Cosmos-Predict2-2B-Video2World (`predict2_video2world_fsdp_2b`)
- **Model Path**: `checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt`
- **Parallelism**: FSDP (Fully Sharded Data Parallel)
- **FSDP Shard Size**: 8
- **High Sigma Ratio**: 0.05

#### Cosmos-Predict2-14B-Video2World (`predict2_video2world_fsdp_14b`)
- **Model Path**: `checkpoints/nvidia/Cosmos-Predict2-14B-Video2World/model-720p-16fps.pt`
- **Parallelism**: FSDP (Fully Sharded Data Parallel)
- **FSDP Shard Size**: 32
- **High Sigma Ratio**: 0.05

### Optimizer Configuration (`defaults/optimizer.py`)

#### FusedAdamW (`fusedadamw`)
- **Learning Rate**: 1e-4 (default, often overridden)
- **Weight Decay**: 0.1
- **Betas**: [0.9, 0.99]
- **Epsilon**: 1e-8
- **Master Weights**: Enabled
- **Capturable**: Enabled

### Scheduler Configuration (`defaults/scheduler.py`)

#### Lambda Linear Scheduler (`lambdalinear`)
- **Warm-up Steps**: [1000]
- **Cycle Lengths**: [10000000000000] (effectively infinite)
- **F Start**: [1.0e-6]
- **F Max**: [1.0]
- **F Min**: [1.0]

#### Constant Scheduler (`constant`)
- **Rate**: 1.0 (constant throughout training)

## Available Experiments

### GR00T Experiments (`experiment/groot.py`)

#### 2B Model GR00T Training
```bash
EXP=predict2_video2world_training_2b_groot_gr1_480
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

**Configuration:**
- **Learning Rate**: 2^(-14.5) ≈ 3.05e-5
- **Scheduler**: Lambda Linear (f_max=0.2, f_min=0.1, warm_up=1000, cycle=100000)
- **Context Parallel Size**: 1
- **Save Frequency**: Every 200 iterations
- **Video Size**: 432x768, 93 frames
- **Batch Size**: 1 per GPU

#### 14B Model GR00T Training
```bash
EXP=predict2_video2world_training_14b_groot_gr1_480
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

**Configuration:**
- **Learning Rate**: 2^(-14.5) ≈ 3.05e-5
- **Context Parallel Size**: 4
- **FSDP Shard Size**: 32
- **Other settings**: Same as 2B variant

### Other Available Experiments
- **Cosmos Nemo Assets**: Various experiments with asset-based training
- **AgiBOT**: Fisheye camera training configurations

## Key Training Parameters

### Common Overrides in Experiments

1. **Learning Rate**: Typically set to 2^(-14.5) ≈ 3.05e-5
2. **Context Parallelism**: 
   - 2B models: 1-2 GPUs
   - 14B models: 4 GPUs
3. **Guardrail**: Disabled during training (`guardrail_config.enabled=False`)
4. **EMA**: Often enabled for better model stability
5. **Validation**: Usually set to mock/disabled

### Parallelism Strategy

- **Data Parallelism**: Distributed across 8 GPUs (`--nproc_per_node=8`)
- **Model Parallelism**: FSDP with different shard sizes
  - 2B: shard_size=8
  - 14B: shard_size=32
- **Context Parallelism**: Used for longer sequences
  - 2B: context_parallel_size=1-2
  - 14B: context_parallel_size=4

## Usage Examples

### Basic Post-Training Command
```bash
# Set experiment name
export EXP=your_experiment_name

# Run training
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

### With LoRA Training
```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP} model.config.train_architecture=lora
```

### Environment Variables
```bash
# Disable NVTE fused attention if needed
export NVTE_FUSED_ATTN=0

# Set CUDA home
export CUDA_HOME=$CONDA_PREFIX
```

## Checkpoint Management

### Save Locations
Checkpoints are saved to: `checkpoints/{PROJECT}/{GROUP}/{NAME}/checkpoints/`

Example structure:
```
checkpoints/posttraining/video2world/2b_groot_gr1_480/checkpoints/
├── model/
│   ├── iter_000000200.pt
│   ├── iter_000000400.pt
├── optim/
├── scheduler/
├── trainer/
├── latest_checkpoint.txt
```

### Loading Checkpoints for Inference
```bash
python examples/video2world.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/video2world/your_experiment/checkpoints/model/iter_001000.pt" \
  --prompt "Your descriptive prompt" \
  --input_path "path/to/input.jpg" \
  --save_path "path/to/output.mp4"
```

## Performance Considerations

- **Memory Requirements**: 
  - 2B model: ~8 A100/H100 GPUs
  - 14B model: ~8 A100/H100 GPUs with higher memory usage
- **Context Parallelism**: Used to handle longer video sequences
- **FSDP Sharding**: Enables training of large models across multiple GPUs

## Custom Experiment Creation

To create a new experiment:

1. Define your dataset configuration
2. Create experiment config dictionary
3. Register with ConfigStore
4. Run with your experiment name

Example template:
```python
your_custom_experiment = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="your_project",
        group="your_group", 
        name="your_experiment_name",
    ),
    # ... other configurations
)
```

## Troubleshooting

- **OOM Errors**: Reduce batch size or increase context_parallel_size
- **Slow Training**: Check iter_speed callback hit_thres settings
- **Checkpoint Issues**: Verify checkpoint save paths and permissions
- **NVTE Issues**: Set `NVTE_FUSED_ATTN=0` environment variable 