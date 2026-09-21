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
        
        # Test if PyTorch CUDA is already working properly on sm_120
        CUDA_OK=$(python3 -c "import torch; print(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 12)" 2>/dev/null)

        if [ "$CUDA_OK" != "True" ]; then
            echo "⚠️ PyTorch cu128 (sm_120) is required for Blackwell. Installing torch 2.11.0+cu128..."
            pip uninstall -y torchaudio 2>/dev/null || true
            pip install --no-cache-dir torch==2.11.0+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
            echo "✅ PyTorch 2.11.0+cu128 installed successfully."
        else
            echo "✅ Compatible PyTorch with sm_120 support detected."
            # Ensure leftover broken torchaudio doesn't crash diffusers
            if python3 -c "import torchaudio" 2>&1 | grep -q "libcudart"; then
                echo "🧹 Removing broken torchaudio build..."
                pip uninstall -y torchaudio 2>/dev/null || true
            fi
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

