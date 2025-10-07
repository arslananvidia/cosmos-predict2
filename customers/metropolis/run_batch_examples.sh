#!/bin/bash

# Examples for running batch inference with Metropolis dataset

echo "==================================================================="
echo "Batch Inference Examples for Metropolis Dataset"
echo "==================================================================="

# Set environment variables
export NUM_GPUS=8
export PYTHONPATH=$(pwd)

echo "Choose an option:"
echo "1. Standard inference (14B model)"
echo "2. LoRA inference (2B model)"
echo "3. LoRA inference with custom parameters"
echo "4. Show help"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Running standard inference with 14B model..."
        python customers/metropolis/run_batch_metropolis.py
        ;;
    2)
        echo "Running LoRA inference with 2B model..."
        python customers/metropolis/run_batch_metropolis.py --lora
        ;;
    3)
        echo "Running LoRA inference with custom parameters..."
        python customers/metropolis/run_batch_metropolis.py \
            --lora \
            --model_size 2B \
            --lora_rank 16 \
            --lora_alpha 16 \
            --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
        ;;
    4)
        echo "Showing help..."
        python customers/metropolis/run_batch_metropolis.py --help
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo "Batch inference completed!" 