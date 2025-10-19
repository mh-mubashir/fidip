#!/bin/bash

# FiDIP Environment Recreation Script
# This script recreates the fidip_cuda12.2 environment

echo "🚀 Recreating FiDIP CUDA 12.2 Environment..."

# Load required modules
echo "📦 Loading required modules..."
module load cuda/12.3.0
module load miniconda3/24.11.1

# Create conda environment from exported file
echo "🔧 Creating conda environment from exported file..."
conda env create -f fidip_cuda12.2_exported.yml

# Activate environment
echo "✅ Activating environment..."
conda activate fidip_cuda12.2

# Verify installation
echo "🔍 Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo "✅ Environment recreation complete!"
echo "📝 To activate in future sessions: conda activate fidip_cuda12.2"

