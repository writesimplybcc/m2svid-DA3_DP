#!/bin/bash
# Modern DA3 + M2SVid pipeline (recommended)
#
# Two primary heavy processes:
#   1. DA3 depth estimation (monocular depth, optionally metric or streaming for long videos)
#   2. M2SVid stage — lightweight geometric warping (depth → right view + disocclusion mask)
#      + single 1-step conditioned SVD generation (left + warped + mask → refined right view)
#
# This replaces the legacy DepthCrafter-based pipeline with a faster, higher-quality
# modern monocular depth model (Depth Anything 3).
#
# Prerequisites:
#   - DA3 installed / importable (PYTHONPATH pointing to Depth-Anything-3/src)
#   - M2SVid sgm environment active for the warping + generation stages
#
# Example:
#   bash m2svid/inference_da3.sh

set -euo pipefail
set -x

VIDEO_PATH="${1:-demo/input.mp4}"
OUT_ROOT="${2:-outputs/da3_m2svid}"

# Recommended:
#   DA3NESTED-GIANT-LARGE-1.1   → best overall (preferred)
#   DA3MONO-LARGE               → best for pure relative depth / warping quality
DA3_MODEL="${DA3_MODEL:-depth-anything/DA3NESTED-GIANT-LARGE-1.1}"
PROCESS_RES="${PROCESS_RES:-720}"   # higher than legacy 504 for better quality
BATCH_SIZE="${BATCH_SIZE:-4}"

# 1) Heavy stage 1: DA3 monocular depth (feed-forward, very fast compared to diffusion depth)
echo "=== [1/3] DA3 Depth Estimation (heavy) ==="
mkdir -p "${OUT_ROOT}/da3"
PYTHONPATH="Depth-Anything-3/src:${PYTHONPATH:-}" python -m m2svid.m2svid.prepare_da3_depth \
    --video "${VIDEO_PATH}" \
    --output "${OUT_ROOT}/da3/depth.npz" \
    --model "${DA3_MODEL}" \
    --process-res "${PROCESS_RES}" \
    --batch-size "${BATCH_SIZE}" \
    --fps 0 \
    # --invert   # uncomment if the synthesized right view has inverted parallax

# 2) Lightweight geometric warping (depth → right view + mask)
echo "=== [2/3] Geometric Warping (light) ==="
mkdir -p "${OUT_ROOT}/reprojected"
# Activate the M2SVid/SGM environment if needed (adjust as per your setup)
# source /opt/conda/bin/activate "" && conda activate sgm
PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH:-}" \
python warping.py \
    --video_path "${VIDEO_PATH}" \
    --depth_path "${OUT_ROOT}/da3/depth.npz" \
    --output_path_reprojected "${OUT_ROOT}/reprojected/input_reprojected.mp4" \
    --output_path_mask "${OUT_ROOT}/reprojected/input_reprojected_mask.mp4" \
    --disparity_perc 0.05

# 3) Heavy stage 2: M2SVid 1-step conditioned generation
echo "=== [3/3] M2SVid Inpainting & Refinement (heavy 1-step SVD) ==="
mkdir -p "${OUT_ROOT}/m2svid"
# source /opt/conda/bin/activate "" && conda activate sgm
PYTHONPATH="./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/:${PYTHONPATH:-}" \
python inpaint_and_refine.py \
    --mask_antialias 0 \
    --model_config configs/m2svid.yaml \
    --ckpt ckpts/m2svid_weights.pt \
    --video_path "${VIDEO_PATH}" \
    --reprojected_path "${OUT_ROOT}/reprojected/input_reprojected.mp4" \
    --reprojected_mask_path "${OUT_ROOT}/reprojected/input_reprojected_mask.mp4" \
    --output_folder "${OUT_ROOT}/m2svid"

echo "✅ Modern DA3 + M2SVid pipeline complete."
echo "   Outputs in: ${OUT_ROOT}/m2svid/"
echo ""
echo "Primary heavy stages (as designed):"
echo "  1. DA3 depth estimation"
echo "  2. M2SVid single-step conditioned generation"
