#!/bin/bash
# download_weights.sh
# This script downloads all massive weight files that are excluded from GitHub via .gitignore.
# It is designed to be run during the Docker build process to bake the weights into the image.

echo "Downloading and placing weights for StereoFaster..."

# Create necessary directories
mkdir -p ckpts
mkdir -p m2svid/ckpts/metric_models
mkdir -p m2svid/third_party/Hi3D-Official/ckpts

# ==============================================================================
# 1. DepthCrafter Weights (Primary Depth Estimator)
# ==============================================================================
echo "Caching DepthCrafter & SVD weights..."
huggingface-cli download tencent/DepthCrafter
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt

# ==============================================================================
# 1.5 ICDepth (Wan2.1 In-Context DiT + DINOv2) Weights (Primary Depth Estimator)
# ==============================================================================
echo "Caching ICDepth (Wan2.1-depth + Wan2.1 VAE + DINOv2-Large) weights..."
huggingface-cli download iris0329/Wan2.1-depth diffusion_pytorch_model.safetensors config.json prompt_embeds.pt
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B Wan2.1_VAE.pth google/umt5-xxl/spiece.model google/umt5-xxl/tokenizer.json google/umt5-xxl/tokenizer_config.json google/umt5-xxl/special_tokens_map.json
huggingface-cli download facebook/dinov2-large

# ==============================================================================
# 2. Video Depth Anything (VDA) Weights (Fast Alternative Estimator)
# ==============================================================================
echo "Caching Video Depth Anything weights..."
huggingface-cli download depth-anything/Video-Depth-Anything-Large
huggingface-cli download depth-anything/Video-Depth-Anything-Base

# ==============================================================================
# 2.5 FlashDepth Weights (Real-Time 2K Streaming Estimator)
# ==============================================================================
echo "Caching FlashDepth weights..."
mkdir -p m2svid/third_party/FlashDepth/configs/flashdepth
mkdir -p m2svid/third_party/FlashDepth/configs/flashdepth-l
huggingface-cli download Eyeline-Labs/FlashDepth flashdepth/iter_43002.pth --local-dir m2svid/third_party/FlashDepth/configs/flashdepth
huggingface-cli download Eyeline-Labs/FlashDepth flashdepth-l/iter_10001.pth --local-dir m2svid/third_party/FlashDepth/configs/flashdepth-l

# ==============================================================================
# 3. M2SVid Main Weights
# ==============================================================================
echo "Downloading M2SVid weights..."
wget -q -O m2svid/ckpts/m2svid_weights.pt "https://storage.googleapis.com/gresearch/m2svid/m2svid_weights.pt"
wget -q -O m2svid/ckpts/m2svid_no_full_atten_weights.pt "https://storage.googleapis.com/gresearch/m2svid/m2svid_no_full_atten_weights.pt"

# OpenCLIP model used by Hi3D / SGM conditioning
wget -q -O m2svid/third_party/Hi3D-Official/ckpts/open_clip_pytorch_model.bin "https://huggingface.co/hbyang/Hi3D/resolve/main/open_clip_pytorch_model.bin"

# Pre-cache OpenCLIP ViT-H-14 weights for SGM
echo "Caching OpenCLIP ViT-H-14 weights..."
huggingface-cli download laion/CLIP-ViT-H-14-laion2B-s32B-b79K open_clip_model.safetensors

echo "All weights downloaded and placed successfully!"
