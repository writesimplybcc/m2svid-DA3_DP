#!/bin/bash
# download_weights.sh
# This script downloads all massive weight files that are excluded from GitHub via .gitignore.
# It is designed to be run during the Docker build process to bake the weights into the image.

echo "Downloading and placing weights..."

# Create necessary directories
mkdir -p checkpoints
mkdir -p ckpts
mkdir -p m2svid/ckpts/metric_models

# ==============================================================================
# IMPORTANT: Replace the placeholder URLs below with the actual direct download 
# links to your hosted weights (e.g., from HuggingFace, Google Drive, or AWS S3).
# ==============================================================================

# 1. Depth Pro Weights
wget -q -O checkpoints/depth_pro.pt "https://huggingface.co/apple/DepthPro/resolve/main/depth_pro.pt"

# 1.5 Depth Anything 3 Weights (Automatically caches via huggingface-cli)
echo "Caching Depth Anything 3 weights..."
huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1
huggingface-cli download depth-anything/DA3MONO-LARGE

# 2. LPIPS VGG weights
# wget -q -O ckpts/vgg.pth "YOUR_VGG_URL"

# 3. M2SVid Main Weights
wget -q -O m2svid/ckpts/m2svid_weights.pt "https://storage.googleapis.com/gresearch/m2svid/m2svid_weights.pt"
wget -q -O m2svid/ckpts/m2svid_no_full_atten_weights.pt "https://storage.googleapis.com/gresearch/m2svid/m2svid_no_full_atten_weights.pt"

# 4. Other dependencies
# wget -q -O m2svid/ckpts/ViT-L-14.pt "YOUR_VIT_URL"
# wget -q -O m2svid/ckpts/dpt_hybrid_384.pt "YOUR_DPT_URL"
# wget -q -O m2svid/ckpts/metric_models/sac+logos+ava1-l14-linearMSE.pth "YOUR_METRIC_URL"

echo "Weights downloaded and placed successfully!"
