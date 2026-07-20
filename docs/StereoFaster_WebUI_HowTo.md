# StereoFaster WebUI – Complete How-To Guide

**Modern two-step monocular-to-stereo video conversion using Depth Anything 3 + M2SVid**

This Gradio interface gives you a clean, two-click workflow:

1. **Step 1** – High-quality depth estimation with **Depth Anything 3** (DA3)
2. **Step 2** – Geometric warping + 1-step conditioned generation with **M2SVid**

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites & Launching](#prerequisites--launching)
- [Step 1: DA3 Depth Estimation](#step-1-da3-depth-estimation)
- [Step 2: M2SVid Stereography](#step-2-m2svid-stereography)
- [Outputs Explained](#outputs-explained)
- [Recommended Settings & Best Practices](#recommended-settings--best-practices)
- [Advanced Usage & Tips](#advanced-usage--tips)
- [Troubleshooting](#troubleshooting)

---

## Overview

StereoFaster replaces older multi-stage pipelines (DepthCrafter + complex inpainting) with a modern, faster, higher-quality two-stage approach:

- **DA3** produces excellent, temporally consistent monocular depth (including a strong "Nested Giant" model that combines relative + metric cues).
- **M2SVid** performs precise geometric warping using that depth, then uses a single conditioned 1-step video diffusion pass to hallucinate the missing right-eye content.

The UI is deliberately minimal: two heavy steps, clear progress, and direct access to the most important controls.

---

## Prerequisites & Launching

### Requirements
- NVIDIA GPU with at least 10–12 GB VRAM recommended (RTX 3060 12 GB works well)
- Proper **CUDA-enabled PyTorch** (CPU-only builds will be extremely slow)
- `xformers` installed (highly recommended for speed)
- M2SVid weights placed in `m2svid/ckpts/`
- Depth-Anything-3 source in the Python path

### Launch Command

From the project root:

```bash
PYTHONPATH="Depth-Anything-3/src:.:${PYTHONPATH}" python webui.py --server-port 7878
```

For public sharing (temporary tunnel):

```bash
python webui.py --share
```

The UI will show your current GPU and available VRAM at the top.

---

## Step 1: DA3 Depth Estimation (Heavy Process #1)

This is the first (and usually faster) heavy step. It runs Depth Anything 3 on every frame of your video.

### Input Controls

| Control                    | Type          | Default     | Description |
|---------------------------|---------------|-------------|-----------|
| **Input Monocular Video** | Video upload  | —           | Your source left-eye video (mp4 recommended). The entire video is loaded into RAM. |
| **DA3 Model**             | Dropdown      | NESTED-GIANT-LARGE-1.1 | Choice of Depth Anything 3 model. See model explanations below. |
| **Processing Resolution** | Slider (384–1024, step 32) | 720 | Internal resolution for DA3 inference. Controls the longest side of frames fed to the model. |
    | **Batch Size**            | Slider (1–16) | 4           | Number of frames processed together by DA3. Higher = faster (if you have VRAM), lower = safer on limited VRAM. |

### DA3 Model Choices

- **DA3NESTED-GIANT-LARGE-1.1** (default, recommended)  
  Best overall quality. Combines a giant any-view model with a metric model. Excellent for both geometric accuracy and scale.

- **DA3MONO-LARGE**  
  Pure relative monocular depth specialist. Often gives the most geometrically accurate depth for warping purposes.

- **DA3-GIANT-1.1** / **DA3-LARGE-1.1**  
  Strong general-purpose models (newer 1.1 retrained versions).

- **DA3METRIC-LARGE**  
  Specialized for metric (real-world scale) depth.

### Processing Resolution Explained

This is the most important quality/speed knob.

- Uses `upper_bound_resize`: the longest side of each frame is resized to ≤ this value (aspect ratio preserved).
- The model then runs at this resolution.
- Depth is automatically upsampled back to your original video resolution afterward.

**Guidelines for 854×480 video (longest side = 854):**
- 720 (default) → excellent balance
- 854 or nearest multiple of 14 → maximum quality (if you have VRAM)
- 576–640 → faster iteration, lower VRAM
- **Never set higher than your video's longest side** unless you specifically want upscaling (usually not helpful)

Higher values = quadratic increase in memory and time.

Depth is automatically inverted after inference so that the output `.npz` follows the convention expected by M2SVid (high value = nearer objects), matching DepthCrafter and most other stereo tools.

### What Step 1 Produces

- A `depth.npz` file (compatible with M2SVid warping)
- A grayscale preview video showing the depth
- Automatic hand-off to Step 2 via internal state

---

## Step 2: M2SVid Stereography (Heavy Process #2)

This is the more VRAM-intensive step. It performs geometric warping using the depth from Step 1, then runs the M2SVid 1-step diffusion model to generate the missing right-eye content.

### Input Controls

| Control                          | Type          | Default                  | Description |
|----------------------------------|---------------|--------------------------|-----------|
| **Depth .npz**                   | File upload   | Auto-filled from Step 1  | Depth file produced by Step 1 (or any compatible DA3/M2SVid depth npz). |
| **Disparity Scale**              | Slider 0.01–0.2 | 0.05                   | Strength of the stereo effect. Percentage of image width used as maximum disparity. Higher = stronger 3D pop-out / more eye strain. |
| **Mask Closing Kernel**          | Slider 3–21   | 11                       | Size of the morphological closing operation applied to the disocclusion mask. Larger values fill bigger holes in the warped view before generation. |
| **Mask Antialias (downsample)**  | Checkbox      | Off                      | Whether to use antialiasing when downsampling the mask 8× for the diffusion model. Usually left off. |
| **M2SVid Config**                | Textbox       | `m2svid/configs/m2svid.yaml` | Path to the M2SVid model configuration. Advanced users only. |
| **M2SVid Checkpoint**            | Textbox       | `m2svid/ckpts/m2svid_weights.pt` | Path to the trained M2SVid weights. |

### How Step 2 Works Internally

1. **Geometric Warping** (`warping.py`): Uses the depth map to reproject the left view into an approximate right view. Produces a reprojected video + occlusion mask.
2. **Mask Cleanup**: Applies morphological closing + dilation to the mask.
3. **1-Step Conditioned Generation**: The M2SVid model (a fine-tuned Stable Video Diffusion style model) receives the original left view + the warped right view + the cleaned mask and hallucinates the final right-eye frames in a single denoising step.
4. **Outputs**: Clean right view, side-by-side video, and anaglyph video.

---

## Outputs Explained

After Step 2 completes you will see:

- **Generated Right View** – The synthesized right-eye video.
- **Side-by-Side (SBS)** – Left + Right concatenated horizontally (ready for VR headsets or 3D TVs).
- **Anaglyph** – Red-cyan anaglyph version for cheap 3D glasses.
- **Output folder** – Contains all generated files plus intermediate reprojected video/mask.

All videos are saved with CRF 17 for good quality.

---

## Recommended Settings & Best Practices

### For RTX 3060 12 GB
- **DA3 Model**: `DA3NESTED-GIANT-LARGE-1.1`
- **Processing Resolution**: 640–720
- **Batch Size**: 4
- **Disparity Scale**: 0.04–0.06 (start here)

### General Tips
- Start with the defaults. They are chosen for good quality/speed balance.
- Use **DA3MONO-LARGE** when you care most about pure geometric accuracy for warping.
- Use the **Nested Giant** model when you want the best combination of relative geometry + reasonable metric scale.
- For long videos (> ~1000 frames), consider running DA3 streaming separately and feeding the resulting depth npz here.
- Always keep an eye on the GPU memory line at the top of the UI.

---

## Advanced Usage & Tips

- You can upload your own `depth.npz` in Step 2 (skipping Step 1) if you have depth from another source that follows the same format (key = `"depth"`, shape `(T, H, W)`).
- The two config/ckpt textboxes allow you to point at custom fine-tuned M2SVid weights without editing code.
- The UI uses lazy loading: heavy M2SVid/sgm imports only happen when you actually press "Run Step 2".
- Depth is always upsampled to the original video resolution before warping for best quality.

---

## Troubleshooting

**"xFormers can't load C++/CUDA extensions"**  
→ You are still on a CPU-only PyTorch build. Install a proper CUDA version (see earlier conversation for exact commands).

**Out of Memory during Step 1**  
→ Lower Processing Resolution (try 576 or 512) and/or reduce Batch Size.

**Out of Memory during Step 2**  
→ This is the heavier step. Close other applications, lower resolution in Step 1, or use a smaller M2SVid model if available.

**Stereo effect looks too strong / too weak**  
→ Adjust the **Disparity Scale** slider. 0.05 is a good starting point for most content.

**Depth looks inverted (objects pop backward)**  
→ This should no longer occur. Depth inversion is now applied automatically for correct M2SVid stereoscopy. If you still see it, the issue is likely with your input video or M2SVid settings.

**Preview video is blank or wrong**  
→ This is just a quick visualization. The actual `depth.npz` is what matters for Step 2.

---

**Enjoy creating high-quality stereo video with modern tools!**

If you improve the pipeline or add new features, please update this guide accordingly.
