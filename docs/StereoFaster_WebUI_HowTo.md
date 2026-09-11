# StereoFaster WebUI – Complete User & Optimization Guide

**High-Performance Monocular-to-Stereoscopic 3D Video Conversion**  
*Powered by DepthCrafter / Depth Anything 3 + M2SVid Diffusion Inpainting*

---

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [Quickstart & Launching](#quickstart--launching)
3. [The 4-Tab WebUI Interface](#the-4-tab-webui-interface)
   - [Tab 1: 📁 File & Preview Hub](#tab-1--file--preview-hub-input-center)
   - [Tab 2: 🚀 Step 1 — DepthCrafter Estimation](#tab-2--step-1--depthcrafter-estimation-recommended)
   - [Tab 3: 🎬 Step 2 — M2SVid Stereography](#tab-3--step-2--m2svid-stereography)
   - [Tab 4: 🛠️ Tab 3 — Other Depth Models (DA3 & DepthPro)](#tab-4-️-tab-3--other-depth-models-da3--depthpro)
4. [Auto-Crop & Letterbox Intelligence](#auto-crop--letterbox-intelligence)
5. [Stereoscopy & Convergence Math Explained](#stereoscopy--convergence-math-explained)
6. [GPU Optimization & Recommended Settings Matrix](#gpu-optimization--recommended-settings-matrix)
   - [RTX 3060 (12 GB)](#1-rtx-3060-12-gb-local-workhorse)
   - [RTX 3090 / 4090 (24 GB)](#2-rtx-3090--rtx-4090-24-gb-prosumer-standard)
   - [RTX 5090 (32 GB)](#3-rtx-5090-32-gb-next-gen-powerhouse)
   - [RTX 6000 Ada (48 GB)](#4-rtx-6000-ada-48-gb-enterprise-workstation)
   - [RTX Pro 6000 / Blackwell Workstation (96 GB)](#5-rtx-pro-6000--blackwell-workstation-96-gb-extreme)
7. [Outputs & Formats Explained](#outputs--formats-explained)
8. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview & Architecture

StereoFaster transforms 2D monocular videos into true stereoscopic 3D (Half-SBS / Full-SBS / Red-Cyan Anaglyph) through a decoupled, two-stage AI pipeline:

```
[2D Video] ──► [Step 1: Video Depth Diffusion] ──► [Step 2: Geometric Warping & M2SVid Inpainting] ──► [Stereo 3D Video]
                   (DepthCrafter / DA3)                           (1-Step SVD Latent Diffusion)
```

1. **Step 1 (Temporal Depth Estimation):**
   - **DepthCrafter** uses a video-conditioned diffusion model to estimate continuous, flicker-free depth. Unlike per-frame image depth estimators, it enforces strong temporal consistency across consecutive frames without geometric breathing or jitter.
   - **Depth Anything 3 (DA3)** & **Apple DepthPro** remain available in Tab 3 for metric depth and single-frame comparisons.

2. **Step 2 (Stereo Synthesis & Inpainting):**
   - Performs geometric forward projection (warping) to create a candidate right-eye view from the left eye and depth map.
   - Identifies disoccluded regions (blind spots revealed when shifting perspective).
   - Utilizes **M2SVid** (a specialized 1-step video diffusion model) to inpaint missing pixels and hallucinate natural scene occlusions.
   - Blends hallucinated pixels seamlessly into the sharp original reprojected plate and applies zero-parallax framing.

---

## Quickstart & Launching

### Local Environment (Windows / Linux)

Ensure your Python environment has CUDA PyTorch 2.3+ and requirements installed:

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run WebUI
python webui.py --server-port 7878
```

To create a public Gradio tunnel:
```bash
python webui.py --share
```

### Docker / Vast.ai

For cloud GPU instances (e.g. Vast.ai, RunPod):
```bash
# Using the pre-built Docker image
docker run -d --gpus all \
  -p 7878:7878 -p 8080:8080 \
  -v /local/path/source_videos:/workspace/m2svid-DA3_DP/source_videos \
  -v /local/path/final_videos:/workspace/m2svid-DA3_DP/final_videos \
  writesimplybcc/stereofaster:latest
```

---

## The 4-Tab WebUI Interface

### Tab 1: 📁 File & Preview Hub (Input Center)

The hub is your central control point for video assets.

- **Select Prefix for Match & Preview (Master Dropdown):**
  - **CRITICAL:** What is selected here dictates which video is loaded into system memory for **Step 2**.
  - Selecting a prefix loads the source clip in the **Source Video Preview** player and links any existing `_depth.npz` files in `depthmaps_videos/`.
- **Upload Source Video:** Drag and drop videos (`.mp4`, `.mov`, `.mkv`). Files are saved into `source_videos/`.
- **Auto-Crop Aspect Ratio:** Detects and removes black pillarboxes and letterboxes.

---

### Tab 2: 🚀 Step 1 — DepthCrafter Estimation (Recommended)

Computes temporally consistent depth maps using the DepthCrafter video diffusion model.

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Max Resolution (Longest Edge)** | `1024` | 256–3840 | Maximum dimension (width or height) fed to DepthCrafter. Aspect ratio is preserved. |
| **Inference Steps** | `5` | 1–50 | Denoising steps. `5` for rapid draft, `20–25` for high-fidelity production depth. |
| **Window Size** | `110` | 10–200 | Temporal context frame count processed per sliding window. |
| **Overlap** | `25` | 0–100 | Frame overlap between consecutive windows to prevent seam artifacts. |
| **Guidance Scale** | `1.0` | 0.1–10.0 | Classifier-free guidance strength (keep at 1.0 for video depth). |

- **Estimated Depth Outputs:**
  - `depthmaps_videos/<stem>_DC_depth.npz`: Compressed array containing normalized relative disparity.
  - `depthmaps_videos/<stem>_DC_depth.mp4`: Grayscale video preview of the depth field.

---

### Tab 3: 🎬 Step 2 — M2SVid Stereography

Synthesizes the missing eye view through geometric warping and video diffusion inpainting.

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Disparity Scale** | `0.05` | 0.01–0.20 | Interocular separation (3D depth intensity). `0.04`–`0.06` is natural; higher values cause eye fatigue. |
| **Convergence Point (Zero Parallax)** | `0.5` | 0.0–1.0 | Screen-plane placement. Sets which depth layer sits flush with the monitor/glasses. |
| **M2SVid Processing Resolution** | `Native` or `1024x576` | Dropdown | Internal resolution for AI inpainting. The hallucinated pixels are blended back into native resolution plates. |
| **Generation Chunk Size** | `10` | 2–35 | Number of frames processed simultaneously by the UNet. Lower this if encountering CUDA OOM. |
| **Warping Batch Size** | `8` | 1–16 | Frame batching during geometric reprojection. |
| **Mask Closing Kernel** | `11` | 3–21 | Morphological dilation kernel size for cleaning disocclusion boundaries. |

---

### Tab 4: 🛠️ Tab 3 — Other Depth Models (DA3 & DepthPro)

Legacy and research tab for frame-by-frame and metric depth estimators:
- **DA3NESTED-GIANT-LARGE-1.1:** Combines relative geometric understanding with metric scale cues.
- **DA3MONO-LARGE:** High-precision relative depth for still scenes.
- **Apple DepthPro:** Zero-shot metric depth with sharp focal estimation.

---

## Auto-Crop & Letterbox Intelligence

**Black bars are fatal to stereoscopic conversion.** If letterbox or pillarbox bars remain in the video, depth estimators treat them as real flat geometry, causing the warping algorithm to tear borders and project jagged AI artifacts.

### How Auto-Crop Works:
1. **Intelligent Mid-Point Sampling:** The analyzer samples a frame from the middle of the video to avoid false positives on opening fade-ins from black.
2. **Edge Threshold Detection:** It computes the active non-black bounding box. If the content fills within 10 pixels of the frame borders, it automatically flags the video as **`16:9 (None)`** and **leaves it untouched**.
3. **Preset Matching:** If black borders exist, it matches the active video to standard presets (`2.39:1 CinemaScope`, `4:3 Pillarbox`, `1:1 Square`, `9:16 Vertical`, etc.).
4. **Duplicate Prevention:**
   - Files created by the cropper receive the `_cropped` suffix.
   - Batch auto-crop automatically filters out existing `_cropped` files so clips are never re-cropped.

---

## Stereoscopy & Convergence Math Explained

### Why Convergence (Zero Parallax) Matters
- **Convergence Point = 0.0:** The entire scene floats in front of the screen (pure negative parallax pop-out). Causes extreme border violations.
- **Convergence Point = 1.0:** The entire scene recedes behind the screen like looking through a window (pure positive parallax).
- **Convergence Point = 0.5 (Default):** The main subject sits comfortably on the display plane, foreground elements pop slightly outward, and background elements sink naturally into the display.

### The Border Tear Prevention Architecture
M2SVid's diffusion model is conditioned on rightward camera translation. Shifting the background to the right during geometric warping tears open the left border of the frame. 

StereoFaster uses a **two-phase convergence pipeline**:
1. **Geometric Warping:** Warps strictly with `convergence = 0.0` to keep the left boundary locked and sealed, preventing AI generation tearing.
2. **Post-Synthesis Convergence Shift:** Shifts the synthesized right-eye plate relative to the left-eye plate by `shift = int(width * disparity_perc * convergence_point)` at the output stage, ensuring mathematically pristine zero parallax without edge artifacts.

---

## GPU Optimization & Recommended Settings Matrix

Diffusion models scale attention memory quadratically with resolution ($O(N^2)$). Use this reference table for optimal speed and stability:

| GPU Tier | VRAM | Step 1 DepthCrafter Res | DepthCrafter Steps | M2SVid Process Res | Gen Chunk Size | Warping Batch |
|---|---|---|---|---|---|---|
| **RTX 3060** | 12 GB | `640` – `768` (or Native 480p) | 20–25 | `Native` (≤480p) / `1024x576` | 8–10 | 8 |
| **RTX 3090 / 4090** | 24 GB | `1024` – `1280` | 25 | `1280x720 (Faster)` / `Native` | 14–16 | 16 |
| **RTX 5090** | 32 GB | `1280` – `1536` | 25–30 | `Native` (up to 1080p) | 18–22 | 16 |
| **RTX 6000 Ada** | 48 GB | `1536` – `1920` | 25–30 | `Native` (up to 1440p) | 24–28 | 16 |
| **RTX Pro 6000 / Blackwell** | 96 GB | `1920` – `2048` | 30 | `Native` (Full 1080p/2K) | 30–35 | 16 |

---

### Detailed GPU Profiles

#### 1. RTX 3060 (12 GB Local Workhorse)
- **Target Profile:** 480p / 640×480 / 720p footage.
- **DepthCrafter Max Res:** Set to `640` or `768`. For standard definition clips (e.g. 632×480), match the native longest edge (`640`).
- **M2SVid Settings:**
  - For clips ≤ 720p: Select **`Native`**.
  - For 1080p source clips: Select **`1024x576 (Optimal 12GB)`** to prevent out-of-memory errors during the diffusion pass.
  - **Gen Chunk Size:** `8` or `10`.

#### 2. RTX 3090 / RTX 4090 (24 GB Prosumer Standard)
- **Target Profile:** 720p / 1080p footage.
- **DepthCrafter Max Res:** `1024` (optimal speed-to-quality balance) or `1280`.
- **M2SVid Settings:**
  - **M2SVid Processing Resolution:** `1280x720` or `Native` (for 1080p).
  - **Gen Chunk Size:** `14` to `16`.
  - **Warping Batch:** `16`.

#### 3. RTX 5090 (32 GB Next-Gen Powerhouse)
- **Target Profile:** Native 1080p high frame-rate footage.
- **DepthCrafter Max Res:** `1280` to `1536`.
- **M2SVid Settings:**
  - **M2SVid Processing Resolution:** `Native` (runs full 1080p AI generation smoothly).
  - **Gen Chunk Size:** `18` to `22` frames per batch.
  - Generates full 10-second scenes in minimal chunks with zero boundary seams.

#### 4. RTX 6000 Ada (48 GB Enterprise Workstation)
- **Target Profile:** Native 1080p / 2K Master stereography.
- **DepthCrafter Max Res:** `1536` to `1920`.
- **DepthCrafter Window Size:** Can safely increase from `110` to `150` for deeper temporal context.
- **M2SVid Settings:**
  - **M2SVid Processing Resolution:** `Native`.
  - **Gen Chunk Size:** `24` to `28`.

#### 5. RTX Pro 6000 / Blackwell Workstation (96 GB Extreme)
- **Target Profile:** Maximum quality archival and cinematic conversion.
- **DepthCrafter Max Res:** `1920` or `2048` longest edge.
- **M2SVid Settings:**
  - **M2SVid Processing Resolution:** `Native` across all source resolutions.
  - **Gen Chunk Size:** `32` to `35` (maximum supported window).
  - *Note on 4K:* Even with 96 GB VRAM, diffusion self-attention at native 3840×2160 requires terabytes of unrolled memory. Always let M2SVid generate the disocclusions at 1080p/2K and let the blended compositor output the final 4K plate.

---

## Outputs & Formats Explained

All processed results are organized into `final_videos/<stem>_stereo/`:

1. **`stereo_sbs.mp4` (Side-by-Side 3D):**
   - Left-eye video on the left, right-eye on the right.
   - Padded to a 16:9 canvas with black pillarboxes if the source is vertical or 4:3, ensuring 3D projectors and VR video players (Meta Quest, Apple Vision Pro, 3D TVs) do not distort the stereo geometry.
2. **`anaglyph.mp4` (Red/Cyan):**
   - Full-color dubois anaglyph rendering for immediate previewing with standard red/cyan glasses.
3. **`generated_right.mp4`:**
   - Standalone synthesized right-eye stream.
4. **`reprojected/`:**
   - Intermediate warped frame plate and binary inpainting occlusion mask used during AI conditioning.

---

## Troubleshooting & FAQ

### Q: Why do the edges of the video look jagged or torn?
- **Cause:** Your source video likely has black pillarbox or letterbox bars encoded into the image.
- **Solution:** Go to **📁 File & Preview Hub**, expand **✂️ Auto-Crop Aspect Ratio**, click **Apply Crop**, and run Step 1 & 2 on the resulting `_cropped.mp4` file.

### Q: Why did Step 2 process the wrong video?
- **Cause:** Step 2 uses whatever video is currently loaded into the **File & Preview Hub** video player.
- **Solution:** In the **File & Preview Hub**, ensure the **`Select Prefix for Match & Preview`** dropdown matches your desired clip (e.g. `myvideo_cropped`) before switching to Step 2.

### Q: The console seems frozen during DepthCrafter or M2SVid!
- **Cause:** Diffusion models run intensive matrix math in CUDA memory without emitting output between steps.
- **Status Check:** StereoFaster logs step markers in the console:
  - DepthCrafter: `[DepthCrafter] Denoising step X/25...` every 5 steps.
  - M2SVid: `[M2SVid] ⏳ Executing heavy AI generation...` followed by completion elapsed time.

### Q: CUDA Out of Memory (OOM) during generation?
1. In Step 2, lower **Generation Chunk Size** (e.g., reduce from 14 to 8 or 6).
2. Set **M2SVid Processing Resolution** to `1024x576 (Optimal 12GB)`.
3. In Step 1, reduce **Max Resolution (Longest Edge)** to `640` or `768`.
