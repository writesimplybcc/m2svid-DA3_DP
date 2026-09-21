#!/bin/bash

# =============================================================================
# 1. Launch FileBrowser immediately (top priority for instant access)
# =============================================================================
echo "Starting FileBrowser in background..."
filebrowser -r /workspace/m2svid-DA3_DP -p 7879 -a 0.0.0.0 --noauth &
sleep 2
echo "✅ FileBrowser is running on port 7879"

# =============================================================================
# 2. GPU Pre-flight & Autodetection
# =============================================================================
echo "==============================================================="
echo "🔍 StereoFaster Pre-flight & GPU Detection"
echo "==============================================================="

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)

    echo "GPU Model:          $GPU_NAME"
    echo "Total VRAM:         $GPU_VRAM"
    echo "Compute Capability: sm_$COMPUTE_CAP"
    echo "Driver Version:     $DRIVER_VER"

    # Check for RTX 5090 / Blackwell architecture (Compute Capability 12.0+)
    COMPUTE_MAJOR=$(echo "$COMPUTE_CAP" | cut -d'.' -f1)
    if [ "$COMPUTE_MAJOR" -ge 12 ] 2>/dev/null || [[ "$GPU_NAME" =~ "5090" ]]; then
        echo "⚡ Blackwell architecture detected ($GPU_NAME)."
        
        # Check installed PyTorch CUDA build
        CURRENT_CU=$(python3 -c "import torch; print(torch.version.cuda or 'cpu')" 2>/dev/null)
        echo "Current PyTorch CUDA build: $CURRENT_CU"

        if [[ "$CURRENT_CU" != 12.8* && "$CURRENT_CU" != 12.9* ]]; then
            echo "⚠️ PyTorch cu128 is required for Blackwell sm_120. Updating PyTorch..."
            pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
            echo "✅ PyTorch updated to cu128 successfully."
        else
            echo "✅ Compatible PyTorch cu128 detected."
        fi
    else
        echo "✅ GPU architecture sm_$COMPUTE_CAP is fully compatible with existing environment."
    fi
else
    echo "⚠️ nvidia-smi not found. Running in CPU mode or container lacks NVIDIA runtime."
fi

# =============================================================================
# 3. Memory, Caching & Performance Optimizations
# =============================================================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_MODULE_LOADING="LAZY"

# Override temporary directory to prevent Vast.ai's tiny 64MB /tmp partition from crashing video uploads
export GRADIO_TEMP_DIR="/workspace/m2svid-DA3_DP/tmp"
export TMPDIR="/workspace/m2svid-DA3_DP/tmp"
mkdir -p /workspace/m2svid-DA3_DP/tmp

# =============================================================================
# 4. Launch StereoFaster WebUI
# =============================================================================
echo "==============================================================="
echo "🌐 Starting StereoFaster WebUI on port 7878..."
echo "==============================================================="
python webui.py --server-name 0.0.0.0 --server-port 7878

