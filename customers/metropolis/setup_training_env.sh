#!/bin/bash

# Set up environment variables for Cosmos-Predict2 training
# Recommended settings to avoid common issues

# CUDA settings
export CUDA_HOME=$CONDA_PREFIX

# NCCL settings (to avoid timeout issues)
export NCCL_TIMEOUT=3600          # 1 hour timeout
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN             # Set to INFO for more detailed logs

# Optional: Disable NVTE fused attention if needed
# export NVTE_FUSED_ATTN=0

# PyTorch settings
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Environment variables set for Cosmos-Predict2 training" 