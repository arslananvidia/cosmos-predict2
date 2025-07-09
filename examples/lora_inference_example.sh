#!/bin/bash

# Example script for running LoRA inference with Cosmos Predict2
# This script demonstrates how to use the post-trained LoRA checkpoint

# Set environment variables
    export NUM_GPUS=8
    export PYTHONPATH=$(pwd)

# Run LoRA inference with your post-trained checkpoint
torchrun --nproc_per_node=${NUM_GPUS} examples/video2world_lora.py \
    --model_size 2B \
    --dit_path checkpoints/posttraining/video2world/2b_metropolis/checkpoints/model/iter_000002000.pt \
    --input_path "/home/arslana/codes/cosmos-predict2/datasets/metropolis/benchmark/v2/frames/0a2cac1d-96ef-5132-9b34-67921af97ffb.jpg" \
    --prompt "A static traffic cctv camera captures an urban intersection where multiple vehicles navigate through the busy streets. A motorcycle enters the scene, and it crashes with a pedestrian which is crossing the street. Other cars in the scene drives normally without interruption." \
    --save_path results/2b_metropolis_lora/generated_test_lora.mp4 \
    --num_gpus ${NUM_GPUS} \
    --offload_guardrail \
    --offload_prompt_refiner \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"

echo "LoRA inference completed. Check results/2b_metropolis_lora/generated_test_lora.mp4" 