# LoRA Inference for Video2World Generation

This document describes how to perform inference with LoRA (Low-Rank Adaptation) post-trained models in Cosmos Predict2.

## Overview

LoRA is a parameter-efficient fine-tuning technique that allows you to adapt large pre-trained models to specific domains or tasks by training only a small number of additional parameters. The `video2world_lora.py` script enables inference with LoRA-trained checkpoints.

## Key Differences from Standard Inference

### Standard Inference (`video2world.py`)
- Loads the full model checkpoint
- Uses all model parameters as they were trained
- Suitable for base models or fully fine-tuned models

### LoRA Inference (`video2world_lora.py`)
- Loads base model + LoRA adapters
- Only LoRA parameters were trained during post-training
- Memory efficient and faster training
- Maintains base model capabilities while adding domain-specific adaptations

## Prerequisites

1. **Post-trained LoRA checkpoint**: You need a checkpoint created with LoRA training (using `model.config.train_architecture=lora`)
2. **PEFT library**: The script uses the PEFT (Parameter-Efficient Fine-Tuning) library
3. **Base model**: The underlying base model should be available (automatically handled by the script)

## Usage

### Basic Command Structure

```bash
export NUM_GPUS=8
export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=${NUM_GPUS} examples/video2world_lora.py \
    --model_size 2B \
    --dit_path <path_to_lora_checkpoint> \
    --input_path <input_image_or_video> \
    --prompt "<your_prompt>" \
    --save_path <output_path> \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
```

### LoRA-Specific Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--use_lora` | Enable LoRA inference mode | False | **Required** for LoRA inference |
| `--lora_rank` | Rank of LoRA adaptation | 16 | Should match training config |
| `--lora_alpha` | LoRA alpha parameter | 16 | Should match training config |
| `--lora_target_modules` | Target modules for LoRA | "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2" | Should match training config |
| `--init_lora_weights` | Initialize LoRA weights | True | Usually keep as True |

### Example: Metropolis Post-Training

```bash
#!/bin/bash

export NUM_GPUS=8
export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=${NUM_GPUS} examples/video2world_lora.py \
    --model_size 2B \
    --dit_path checkpoints/posttraining/video2world/2b_metropolis/checkpoints/model/iter_000002000.pt \
    --input_path "/path/to/your/input.jpg" \
    --prompt "A static traffic cctv camera captures an urban intersection where multiple vehicles navigate through the busy streets." \
    --save_path results/2b_metropolis_lora/generated_video.mp4 \
    --num_gpus ${NUM_GPUS} \
    --offload_guardrail \
    --offload_prompt_refiner \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
```

## How It Works

### 1. Model Initialization
The script initializes the base model architecture without loading any weights:

```python
pipe.dit = instantiate(dit_config).eval()
```

### 2. LoRA Addition
LoRA adapters are added to the specified target modules:

```python
pipe.dit = add_lora_to_model(
    pipe.dit,
    lora_rank=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_target_modules=args.lora_target_modules,
    init_lora_weights=args.init_lora_weights,
)
```

### 3. Checkpoint Loading
The LoRA-trained checkpoint is loaded with `strict=False` to accommodate the additional LoRA parameters:

```python
missing_keys = pipe.dit.load_state_dict(state_dict_dit_regular, strict=False, assign=True)
```

### 4. Parameter Statistics
The script reports parameter counts to show the efficiency of LoRA:

```
Total parameters: 2,345,678,901
Trainable LoRA parameters: 1,234,567
LoRA parameter ratio: 0.05%
```

## LoRA Configuration Matching

⚠️ **Important**: The LoRA parameters used during inference **must match** those used during training:

### From Training Command
```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py \
    -- experiment=predict2_video2world_training_2b_metropolis \
    model.config.train_architecture=lora
```

### Default LoRA Parameters (in `video2world_model.py`)
```python
lora_rank: int = 16
lora_alpha: int = 16
lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
init_lora_weights: bool = True
```

## Troubleshooting

### Common Issues

1. **Missing LoRA Parameters**
   ```
   Error: Missing keys in regular model: ['blocks.0.self_attn.q_proj.lora_A.default.weight', ...]
   ```
   **Solution**: Ensure `--use_lora` flag is set and LoRA parameters match training configuration.

2. **Unexpected Keys**
   ```
   Warning: Unexpected keys in regular model: ['blocks.0.self_attn.q_proj.lora_B.default.weight', ...]
   ```
   **Solution**: This usually means the checkpoint contains LoRA weights but `--use_lora` is not enabled.

3. **Parameter Mismatch**
   ```
   Error: size mismatch for blocks.0.self_attn.q_proj.lora_A.default.weight
   ```
   **Solution**: Check that `--lora_rank` matches the value used during training.

4. **Module Not Found**
   ```
   Warning: No module named 'blocks.0.self_attn.k_proj' found for LoRA
   ```
   **Solution**: Verify `--lora_target_modules` matches the training configuration.

### Debug Mode

To debug LoRA loading, you can check the model structure:

```python
# Print all parameter names
for name, param in pipe.dit.named_parameters():
    if 'lora' in name.lower():
        print(f"LoRA parameter: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}")
```

## Performance Considerations

### Memory Usage
- LoRA inference uses slightly more memory than base model inference due to additional LoRA parameters
- Memory overhead is typically < 1% of base model size

### Speed
- LoRA inference speed is comparable to base model inference
- Additional computation from LoRA layers is minimal

### Quality
- LoRA models retain base model capabilities while adding domain-specific improvements
- Quality depends on the quality of post-training data and LoRA configuration

## Comparison with Standard Inference

| Aspect | Standard Inference | LoRA Inference |
|--------|-------------------|----------------|
| **Checkpoint Size** | Full model (~5-50GB) | Base + LoRA (~5-50GB + 10-100MB) |
| **Training Time** | Full model retraining | Only LoRA parameters |
| **Memory Usage** | Base memory | Base + small LoRA overhead |
| **Flexibility** | Fixed model | Can switch LoRA adapters |
| **Domain Adaptation** | Requires full fine-tuning | Efficient adaptation |

## Best Practices

1. **Always match LoRA parameters** between training and inference
2. **Use same target modules** as specified in training configuration
3. **Monitor parameter counts** to ensure LoRA is loaded correctly
4. **Test with known inputs** to validate model behavior
5. **Keep LoRA parameters small** (rank 8-64) for efficiency

## Integration with Existing Workflows

The LoRA inference script is fully compatible with existing workflows:

- **Batch processing**: Use `--batch_input_json` for multiple inputs
- **Guardrails**: All guardrail functionality works as normal
- **Prompt refinement**: Compatible with prompt refiner
- **Multi-GPU**: Supports context parallelism for large videos
- **Different resolutions**: Support for 480p and 720p outputs

## Future Enhancements

Potential improvements for LoRA inference:

1. **Multiple LoRA adapters**: Load multiple LoRA adapters for different domains
2. **Dynamic LoRA switching**: Switch between different LoRA adapters during inference
3. **LoRA composition**: Combine multiple LoRA adapters for multi-domain capabilities
4. **Quantized LoRA**: Support for quantized LoRA parameters for further efficiency 