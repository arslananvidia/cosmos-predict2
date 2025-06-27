# Cosmos-Predict2 Post-Training: Metropolis Dataset

This README provides complete instructions for post-training Cosmos-Predict2 Video2World models on the Metropolis traffic dataset.

## Dataset Overview

- **Dataset Name**: Metropolis Traffic Dataset
- **Location**: `/home/arslana/codes/github/cosmos-predict2/datasets/metropolis/train`
- **Size**: 1,191 video-text pairs
- **Video Format**: 1280×740 resolution, ~8.2 seconds duration, ~246 frames
- **Text Format**: Traffic scene descriptions extracted from qwen_caption

## Dataset Structure

```
datasets/metropolis/train/
├── metas/          # Text descriptions (.txt files)
├── videos/         # Video files (.mp4 files)
└── t5_xxl/         # Pre-computed T5-XXL embeddings (.pickle files)
```

## Configuration

### Model Configuration
- **Model**: Cosmos-Predict2-2B-Video2World (FSDP)
- **Input Frames**: 93 (cropped/padded from original ~246 frames)
- **Video Size**: 740×1280 (Height×Width format)
- **Batch Size**: 1 per GPU
- **Context Parallelism**: 2 (2B model) / 4 (14B model)

### Training Configuration
- **Learning Rate**: 3.05e-5 (2^-14.5)
- **Max Iterations**: 2,000
- **Save Frequency**: Every 500 iterations
- **Optimizer**: FusedAdamW
- **Scheduler**: Lambda Linear
- **EMA**: Enabled for stability
- **Guardrails**: Disabled during training

## Quick Start

### Step 1: Set Up Environment
```bash
# Make environment setup script executable
chmod +x setup_training_env.sh

# Source environment variables
source setup_training_env.sh
```

### Step 2: Run Training (2B Model)
```bash
# Set experiment name
export EXP=predict2_video2world_training_2b_metropolis

# Run training
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

### Step 3: Monitor Training Progress
Watch for:
- Iteration speed (~20-30 seconds per iteration)
- Loss decreasing over time
- No NCCL timeout errors
- GPU memory usage

## Training Options

### Option 1: Standard Training (2B Model)
```bash
source setup_training_env.sh
export EXP=predict2_video2world_training_2b_metropolis
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

### Option 2: LoRA Training (Parameter-Efficient)
```bash
source setup_training_env.sh
export EXP=predict2_video2world_training_2b_metropolis
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP} model.config.train_architecture=lora
```

### Option 3: 14B Model (Requires More GPU Memory)
```bash
source setup_training_env.sh
export EXP=predict2_video2world_training_14b_metropolis
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}
```

## File Structure After Training

### Configuration Files
- `cosmos_predict2/configs/base/experiment/metropolis.py` - Training configuration
- `setup_training_env.sh` - Environment setup script

### Checkpoint Structure
```
checkpoints/posttraining/video2world/2b_metropolis/checkpoints/
├── model/
│   ├── iter_000000500.pt    # Checkpoint at 500 iterations
│   ├── iter_000001000.pt    # Checkpoint at 1000 iterations  
│   ├── iter_000001500.pt    # Checkpoint at 1500 iterations
│   ├── iter_000002000.pt    # Final checkpoint
├── optim/                   # Optimizer states
├── scheduler/               # Scheduler states
├── trainer/                 # Trainer states
└── latest_checkpoint.txt    # Points to latest checkpoint
```

## Training Parameters Details

### Dataset Configuration
```python
example_video_dataset_metropolis = L(Dataset)(
    dataset_dir="datasets/metropolis/train",
    num_frames=93,  # Standard frame count
    video_size=(740, 1280),  # Height, Width format
)
```

### DataLoader Configuration
```python
dataloader_train_metropolis = L(DataLoader)(
    dataset=example_video_dataset_metropolis,
    sampler=L(get_sampler)(dataset=example_video_dataset_metropolis),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)
```

### Training Configuration
```python
trainer=dict(
    distributed_parallelism="fsdp",
    max_iter=2000,  # Total training iterations
    callbacks=dict(
        iter_speed=dict(hit_thres=10),  # Report every 10 iterations
    ),
)
```

### Optimizer Configuration
```python
optimizer=dict(
    lr=2 ** (-14.5),  # Learning rate: ~3.05e-5
)
```

### Scheduler Configuration
```python
scheduler=dict(
    warm_up_steps=[2_000],      # Warm-up steps
    cycle_lengths=[400_000],    # Cycle length
    f_max=[0.6],               # Maximum factor
    f_min=[0.3],               # Minimum factor
)
```

## Environment Variables

The `setup_training_env.sh` script sets these important variables:

```bash
export CUDA_HOME=$CONDA_PREFIX              # CUDA path
export NCCL_TIMEOUT=3600                    # 1 hour timeout
export NCCL_ASYNC_ERROR_HANDLING=1          # Better error handling
export NCCL_DEBUG=WARN                      # NCCL debug level
export PYTHONPATH=$(pwd):$PYTHONPATH       # Python path
```

## Inference After Training

### Test Your Trained Model
```bash
# Test with a sample from your dataset
python examples/video2world.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/video2world/2b_metropolis/checkpoints/model/iter_000002000.pt" \
  --prompt "A traffic scene with vehicles navigating through an urban intersection" \
  --input_path "datasets/metropolis/train/videos/000f9c15-06af-58de-833b-606042201f53.mp4" \
  --save_path "results/metropolis_test_output.mp4"
```

### Custom Inference
```bash
# Use your own input image/video and prompt
python examples/video2world.py \
  --model_size 2B \
  --dit_path "checkpoints/posttraining/video2world/2b_metropolis/checkpoints/model/iter_000002000.pt" \
  --prompt "Your custom traffic scene description" \
  --input_path "path/to/your/input.jpg" \
  --save_path "path/to/your/output.mp4"
```

## Performance Expectations

### Hardware Requirements
- **2B Model**: 8× A100/H100 GPUs (recommended)
- **14B Model**: 8× A100/H100 GPUs with higher memory
- **Memory**: ~50-70GB GPU memory per GPU during training
- **Storage**: Ensure sufficient space for checkpoints (~2-5GB per checkpoint)

### Training Time Estimates
- **2B Model**: ~11-17 hours for 2,000 iterations (based on 20-30 sec/iter)
- **14B Model**: ~17-28 hours for 2,000 iterations
- **Checkpointing**: Additional time for saving every 500 iterations

### Expected Metrics
- **Iteration Speed**: 20-30 seconds per iteration
- **Loss**: Should decrease from ~0.5-1.0 to ~0.1-0.3
- **GPU Utilization**: ~90-100%
- **Memory Usage**: ~50-70GB per GPU

## Troubleshooting

### Common Issues and Solutions

#### 1. NCCL Timeout Errors
```bash
# Already handled in setup_training_env.sh, but if issues persist:
export NCCL_TIMEOUT=7200  # Increase to 2 hours
export NCCL_DEBUG=INFO    # Get more detailed logs
```

#### 2. Out of Memory (OOM) Errors
```bash
# Use LoRA training instead
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP} model.config.train_architecture=lora

# Or reduce context parallelism in config:
# context_parallel_size=1  # Instead of 2
```

#### 3. Configuration Not Found
```bash
# Verify the metropolis.py file exists:
ls -la cosmos_predict2/configs/base/experiment/metropolis.py

# Check for syntax errors:
python -m py_compile cosmos_predict2/configs/base/experiment/metropolis.py
```

#### 4. Dataset Loading Issues
```bash
# Verify dataset structure:
ls -la datasets/metropolis/train/
ls datasets/metropolis/train/videos/ | wc -l    # Should show 1191
ls datasets/metropolis/train/metas/ | wc -l     # Should show 1191
ls datasets/metropolis/train/t5_xxl/ | wc -l    # Should show 1191
```

#### 5. Slow Training
```bash
# Check GPU utilization:
nvidia-smi -l 1

# Check I/O performance:
iotop -ao

# Increase number of workers if CPU/I/O bound:
# num_workers=16  # Instead of 8 in config
```

## Monitoring Training

### Key Logs to Watch
```bash
# Monitor training progress
tail -f <log_file>

# Key indicators:
# - "iter_speed X.XX seconds per iteration"
# - "Loss: X.XXXX" (should decrease)
# - "DeviceMonitor Stats" (GPU usage)
# - No NCCL timeout errors
```

### Training Success Indicators
- ✅ Consistent iteration speed (20-30 sec/iter)
- ✅ Decreasing loss values
- ✅ Regular checkpoint saves
- ✅ No NCCL communication errors
- ✅ High GPU utilization (>90%)

## Customization Options

### Modify Training Parameters
Edit `cosmos_predict2/configs/base/experiment/metropolis.py`:

```python
# Change training iterations
max_iter=5000,  # Instead of 2000

# Change learning rate
lr=2 ** (-15),  # Lower learning rate

# Change save frequency
save_iter=1000,  # Save every 1000 iterations

# Change batch size (if memory allows)
batch_size=2,  # Instead of 1
```

### Add Custom Callbacks
```python
# In the training config
callbacks=dict(
    iter_speed=dict(hit_thres=10),
    # Add custom callbacks here
),
```

## Next Steps

1. **Monitor Training**: Watch logs for progress and issues
2. **Evaluate Results**: Test inference after training completes
3. **Fine-tune Parameters**: Adjust learning rate, iterations based on results
4. **Scale Up**: Try 14B model if 2B results are promising
5. **Production**: Deploy trained model for traffic scene generation

## Support

For issues specific to this Metropolis dataset setup:
1. Check this README troubleshooting section
2. Verify dataset integrity and structure
3. Check environment variables and dependencies
4. Monitor GPU resources and NCCL communications

For general Cosmos-Predict2 issues, refer to the main documentation in `/documentations/`. 