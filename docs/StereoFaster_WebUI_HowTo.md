# StereoFaster WebUI – Complete User & High-Fidelity Stereoscopy Guide

**High-Performance Monocular-to-Stereoscopic 3D Video Conversion**  
*Optimized for 1:1 Native Resolution Depth, Hair & Micro-Detail Preservation, and Multi-Tier GPU Acceleration*

---

## Table of Contents

1. [Core Philosophy: 1:1 Native Resolution Depth](#core-philosophy-11-native-resolution-depth)
2. [Quickstart & Launching](#quickstart--launching)
3. [The 4-Tab WebUI Interface](#the-4-tab-webui-interface)
   - [Tab 1: 📁 File & Preview Hub](#tab-1--file--preview-hub-input-center)
   - [Tab 2: 🚀 Step 1 — DepthCrafter Estimation (Temporal 3D Depth)](#tab-2--step-1--depthcrafter-estimation-temporal-3d-depth)
   - [Tab 3: 🎬 Step 2 — M2SVid Stereography (Inpainting & Synthesis)](#tab-3--step-2--m2svid-stereography-inpainting--synthesis)
   - [Tab 4: 🛠️ Tab 3 — Other Depth Models (DA3 & Apple DepthPro)](#tab-4-️-tab-3--other-depth-models-da3--apple-depthpro)
4. [The Fine-Detail Preservation Law: Hair, Faces & Thin Edges](#the-fine-detail-preservation-law-hair-faces--thin-edges)
5. [GPU Optimization & 1:1 Native Resolution Settings Matrix](#gpu-optimization--11-native-resolution-settings-matrix)
   - [RTX 3060 (12 GB)](#1-rtx-3060-12-gb-local-workhorse)
   - [RTX 3090 / 4090 (24 GB)](#2-rtx-3090--rtx-4090-24-gb-native-1080p-standard)
   - [RTX 5090 (32 GB)](#3-rtx-5090-32-gb-high-fidelity-powerhouse)
   - [RTX 6000 Ada (48 GB)](#4-rtx-6000-ada-48-gb-enterprise-mastering)
   - [RTX Pro 6000 / Blackwell Workstation (96 GB)](#5-rtx-pro-6000--blackwell-workstation-96-gb-extreme-master)
6. [Auto-Crop & Letterbox Intelligence](#auto-crop--letterbox-intelligence)
7. [Zero-Parallax Convergence Math & Border Protection](#zero-parallax-convergence-math--border-protection)
8. [Outputs & Delivery Formats Explained](#outputs--delivery-formats-explained)
9. [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Core Philosophy: 1:1 Native Resolution Depth

Stereoscopic 3D is not merely about broad shapes or cardboard silhouettes. The visceral human perception of 3D depth relies heavily on **micro-geometry**:
- Individual strands of flyaway hair
- Eyelashes, facial contours, and skin folds
- Fine architectural lines, telephone wires, window frames, and fences

When a 1080p video ($1920 \times 1080$ or $1440 \times 1080$) is downscaled to 1024 or 768 during depth estimation, a 1-pixel strand of hair is mathematically averaged into the background wall pixels. The depth estimator consequently assigns the hair the same depth layer as the wall behind it. When shifted to synthesize the right eye, the hair is glued to the background instead of naturally separating in 3D space.

**Rule of Thumb:**
> **For professional and prosumer GPUs (24 GB VRAM and above), always match your DepthCrafter Max Resolution 1:1 to your source video's longest edge.**  
> If VRAM limits are approached, **reduce Window Size** instead of sacrificing spatial resolution.

---

## Quickstart & Launching

### Local Environment (Windows / Linux)

```bash
# Activate Python virtual environment
.\venv\Scripts\activate

# Run WebUI on local port 7878
python webui.py --server-port 7878
```

To create an encrypted public Gradio sharing link:
```bash
python webui.py --share
```

### Docker / Vast.ai Cloud Deployment

```bash
docker run -d --gpus all \
  -p 7878:7878 -p 8080:8080 \
  -v /local/path/source_videos:/workspace/m2svid-DA3_DP/source_videos \
  -v /local/path/final_videos:/workspace/m2svid-DA3_DP/final_videos \
  writesimplybcc/stereofaster:latest
```

---

## The 4-Tab WebUI Interface

### Tab 1: 📁 File & Preview Hub (Input Center)

- **Select Prefix for Match & Preview (Master Dropdown):**
  - **CRITICAL:** What is selected here dictates which video is loaded into system memory for **Step 2**.
  - Selecting a clip loads it into the **Source Video Preview** player and links its corresponding depth map in `depthmaps_videos/`.
- **Upload Source Video:** Drag and drop `.mp4`, `.mov`, `.mkv` files directly into `source_videos/`.
- **Auto-Crop Aspect Ratio:** Detects and removes black pillarboxes and letterboxes.

---

### Tab 2: 🚀 Step 1 — DepthCrafter Estimation (Temporal 3D Depth)

Estimates continuous, flicker-free video depth using the DepthCrafter video diffusion model.

| Parameter | Recommended (24GB+ GPUs) | Range | Purpose |
|---|---|---|---|
| **Max Resolution (Longest Edge)** | **Native** (e.g. `1920`) | 256–3840 | Set equal to your video's longest edge for 1:1 hair & edge preservation. |
| **Window Size** | `65` – `110` | 10–200 | Temporal sliding window. Balance against resolution to fit within GPU VRAM. |
| **Inference Steps** | `7` (Speed) / `20–25` (Master) | 1–50 | `7` steps for fast turnaround; `20–25` for pristine edge definition. |
| **Overlap** | `25` | 0–100 | Frame overlap between consecutive windows to prevent seam artifacts. |
| **Guidance Scale** | `1.0` | 0.1–10.0 | Classifier-free guidance strength (keep at 1.0 for video depth). |

- **Outputs:**
  - `depthmaps_videos/<stem>_DC_depth.npz`: Compressed array containing normalized relative disparity.
  - `depthmaps_videos/<stem>_DC_depth.mp4`: Grayscale preview video.

---

### Tab 3: 🎬 Step 2 — M2SVid Stereography (Inpainting & Synthesis)

Synthesizes the right-eye view through geometric warping and 1-step video diffusion inpainting.

| Parameter | Value | Description |
|---|---|---|
| **M2SVid Processing Resolution** | **`Native`** | Keeps the AI inpainting pass aligned with your full-resolution source plate. |
| **Disparity Scale** | `0.045` – `0.055` | 3D depth separation. `0.05` is natural and comfortable; higher values increase pop-out. |
| **Convergence Point (Zero Parallax)** | `0.5` | Sets the screen-plane depth. Main subjects sit flush with the display; foreground pops out, background recedes. |
| **Generation Chunk Size** | `10` – `35` | Frames evaluated per inpainting pass. Dynamically scales based on your GPU VRAM tier. |
| **Warping Batch Size** | `8` – `16` | Geometric reprojection frame batching. |
| **Mask Closing Kernel** | `11` | Morphological dilation kernel for disocclusion cleanup. |

---

### Tab 4: 🛠️ Tab 3 — Other Depth Models (DA3 & Apple DepthPro)

- **Apple DepthPro:** Zero-shot metric depth with specialized multi-scale patch transformers. If you have scenes with extreme flyaway hair or foliage where temporal consistency is secondary, DepthPro generates sharp edge boundaries.
- **DA3NESTED-GIANT-LARGE-1.1:** Combines relative geometric understanding with metric scale cues.

---

## The Fine-Detail Preservation Law: Hair, Faces & Thin Edges

DepthCrafter uses an $8\times$ spatial VAE downsampler:
* At **$1024 \times 576$**, the internal latent grid is only **$128 \times 72$** cells. An entire clump of hair is represented by just 2 or 3 latent pixels.
* At **$1920 \times 1080$**, the latent grid expands to **$240 \times 135$** cells—giving the model **nearly 4× more spatial resolution** to isolate individual hair boundaries from the background.

```
Low-Res Depth (1024):   [Hair Strand] + [Background Wall] ──► Averaged into 1 Latent Pixel ──► Glued to Wall in 3D
Native Depth (1920):    [Hair Strand] ──► Distinct Latent Tokens ──► Separate 3D Layer ──► Floats in Front in 3D
```

To prevent out-of-memory errors when running native 1080p/1440p depth on cards under 48GB, **trade temporal window size for spatial resolution**:
- **Do not** drop resolution to 1024 if you care about hair strands.
- **Do** keep resolution at 1920, and set `Window Size` to `65`–`75` on 24GB/32GB cards.

---

## GPU Optimization & 1:1 Native Resolution Settings Matrix

Diffusion self-attention memory scales quadratically ($O(N^2)$) with spatial tokens, while temporal attention scales with window size. Use this table to achieve **1:1 Native Resolution Depth**:

| GPU Hardware | VRAM | Target Depth Resolution | Window Size | Inference Steps | M2SVid Resolution | Gen Chunk Size |
|---|---|---|---|---|---|---|
| **RTX 3060** | 12 GB | `Native` (≤720p) / `1024` (for 1080p)* | 110 (or 50 for native 1080p) | 7 – 20 | `1024x576` / `Native` | 8 – 10 |
| **RTX 3090 / 4090** | 24 GB | **`1920` (Native 1080p)** | **`65` – `75`** | 20 – 25 | **`Native`** | 14 – 16 |
| **RTX 5090** | 32 GB | **`1920` – `2048` (Native 1080p/1440p)** | **`75` – `90`** | 20 – 25 | **`Native`** | 18 – 22 |
| **RTX 6000 Ada** | 48 GB | **`1920` – `2560` (Native 1080p/2K)** | **`110` – `120`** | 25 – 30 | **`Native`** | 24 – 28 |
| **RTX Pro 6000 / Blackwell** | 96 GB | **`1920` – `3840` (Native 1080p/2K/4K)** | **`110` – `150`** | 25 – 30 | **`Native`** | 32 – 35 |

*\*On RTX 3060 12GB: To achieve native 1920 depth for fine hair on 1080p footage, lower Window Size to `45`–`50`.*

---

### Detailed GPU Profiles

#### 1. RTX 3060 (12 GB Local Workhorse)
- **High-Fidelity SD / 720p:** Set `Max Res = Native` (e.g. 632, 720, 768), `Window Size = 110`. Captures all hair and fine geometry natively.
- **1080p Hair Strategy:** Set `Max Res = 1920`, `Window Size = 45–50`, `Inference Steps = 7`.
- **M2SVid:** `1024x576 (Optimal 12GB)`, `Gen Chunk Size = 8–10`.

#### 2. RTX 3090 / RTX 4090 (24 GB Native 1080p Standard)
- **DepthCrafter:** Set `Max Res = 1920` (1:1 with source), `Window Size = 65–75`, `Inference Steps = 20–25`.
- **VRAM Footprint:** ~20–22 GB. Fits comfortably in 24 GB GDDR6X without spilling into system memory.
- **M2SVid:** Set to `Native`, `Gen Chunk Size = 14–16`. Full 1080p AI inpainting.

#### 3. RTX 5090 (32 GB High-Fidelity Powerhouse)
- **DepthCrafter:** Set `Max Res = 1920` to `2048` (handles 1440×1080 and 1920×1080 natively), `Window Size = 75–90`, `Inference Steps = 25`.
- **VRAM Footprint:** ~26–28 GB GDDR7.
- **M2SVid:** Set to `Native`, `Gen Chunk Size = 18–22`. Fast parallel inpainting with full temporal cross-attention (`m2svid.yaml`).

#### 4. RTX 6000 Ada (48 GB Enterprise Mastering)
- **DepthCrafter:** Set `Max Res = 1920` to `2560`, `Window Size = 110–120`, `Inference Steps = 25–30`.
- **VRAM Footprint:** ~36–42 GB.
- **M2SVid:** Set to `Native`, `Gen Chunk Size = 24–28`. Massive temporal consistency across long takes.

#### 5. RTX Pro 6000 / Blackwell Workstation (96 GB Extreme Master)
- **DepthCrafter:** Full unconstrained native processing. Set `Max Res = 1920` to `3840`, `Window Size = 110–150`.
- **M2SVid:** Set to `Native`, `Gen Chunk Size = 35` (maximum supported window). Renders complete 10-second cinematic scenes in single seamless passes.

---

## Auto-Crop & Letterbox Intelligence

**Black bars destroy 3D stereoscopy.** If black letterbox or pillarbox bars remain in your clip, depth estimators treat them as physical objects, and the warping algorithm violently tears borders apart.

1. **Mid-Frame Sampling:** The auto-crop analyzer samples from the middle of the video, ignoring opening fade-ins from black.
2. **Threshold Detection:** If the active picture fills within 10 pixels of the frame edges, it automatically marks the clip as **`16:9 (None)`** and **leaves it completely untouched**.
3. **Preset Cropping:** If black borders exist, matching presets (`4:3 Pillarbox`, `2.39:1 CinemaScope`, `1:1 Square`, etc.) crop the file cleanly.
4. **Duplicate Safeguard:** Files created by the cropper receive the `_cropped` suffix and are automatically excluded from future batch runs.

---

## Zero-Parallax Convergence Math & Border Protection

StereoFaster uses a **two-phase decoupled convergence architecture**:

1. **Phase 1 — Inpainting Warping:**  
   Warping executes strictly with `convergence = 0.0` (zero background shift). This ensures the left frame boundary remains sealed, preventing the SVD inpainting UNet from hallucinating jagged edge tears.
2. **Phase 2 — Post-Synthesis Convergence Shift:**  
   The right-eye plate is shifted relative to the left-eye plate by `shift = int(width * disparity_perc * convergence_point)` at the compositing stage.
   - `Convergence = 0.5` (Default): The main character sits flush on the screen glass; foreground objects pop outward, backgrounds recede into the display.
   - `Convergence > 0.5`: The entire scene recedes into a deep theater window.

---

## Outputs & Delivery Formats Explained

All processed files are output to `final_videos/<stem>_stereo/`:

1. **`stereo_sbs.mp4` (Side-by-Side 3D):**
   - Left-eye on the left, right-eye on the right.
   - Automatically padded with black pillarboxes to a 16:9 aspect ratio if the source is 4:3 or vertical, ensuring 3D projectors, Meta Quest, and Apple Vision Pro players do not distort the stereo geometry.
2. **`anaglyph.mp4` (Red/Cyan):**
   - Full-color dubois anaglyph rendering for immediate previewing with standard red/cyan 3D glasses.
3. **`generated_right.mp4`:**
   - Standalone synthesized right-eye stream.
4. **Unified Stream Writing:**
   - All three video streams are rendered in a **single streaming pass**, using less than 50 MB of system RAM without memory bloat.

---

## Troubleshooting & FAQ

### Q: Why did DepthCrafter give `RuntimeError: input tensor must fit into 32-bit index math` on long clips?
- **Fixed:** StereoFaster now automatically chunks antialiasing and noise generation into 14-frame batches. You can process clips of any length without hitting PyTorch's 2.14-billion-element index limit.

### Q: Why did Step 2 give `DefaultCPUAllocator: not enough memory`?
- **Fixed:** StereoFaster blends and upscales in 16-frame temporal chunks directly into compact `uint8` plates, reducing system RAM usage from 90 GB down to ~5 GB.

### Q: How do I make sure a strand of hair pops out in 3D?
- Set **Max Resolution** in Step 1 equal to your video's native resolution (e.g. `1920` for 1080p).
- On 24GB / 32GB GPUs, set **Window Size** to `65`–`75`.
- In Step 2, keep **M2SVid Processing Resolution** set to **`Native`**.
