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

| Parameter | Recommended (RTX 5090 32GB) | Recommended (24GB GPUs) | Range | Purpose |
|---|---|---|---|---|
| **Max Resolution (Longest Edge)** | **Native** (e.g. `1920`) | **Native** (e.g. `1920`) | 256–3840 | Set equal to your video's longest edge for 1:1 hair & edge preservation. |
| **Window Size** | **`80`** (Fastest) / **`110`** (w/ Slicing) | **`65` – `75`** | 10–200 | Temporal sliding window. On 5090, `80` runs at 100% speed without slicing; `110` requires attention slicing. |
| **Inference Steps** | `20–25` (Master) / `5–7` (Fast) | `20–25` (Master) / `5–7` (Fast) | 1–50 | `5–7` steps for rapid turnaround; `20–25` for pristine edge and hair definition. |
| **Overlap** | `16` – `20` (for Window 80) / `24` (for 110) | `16` – `20` (for Window 75) | 0–100 | Frame overlap between consecutive windows to prevent seam artifacts. |
| **Guidance Scale** | `1.0` | `1.0` | 0.1–10.0 | Classifier-free guidance strength (keep at 1.0 for video depth). |
| **Attention Slicing** | **`Disabled`** (for Window 80) / **`Enabled`** (for 110) | `Auto` | Auto / Disabled / Enabled | Chunks attention calculation. Cuts peak allocation by ~50% at a ~5% speed cost. |
| **CPU Offload** | **`none`** (or `Auto`) | `Auto` / `none` | Auto / none / model / sequential | Offloads VAE/text encoder to system RAM. Frees ~4 GB VRAM on tighter setups. |

- **Outputs:**
  - `depthmaps_videos/<stem>_DC_depth.npz`: Compressed array containing normalized relative disparity.
  - `depthmaps_videos/<stem>_DC_depth.mp4`: Grayscale preview video.

---

### Tab 3: 🎬 Step 2 — M2SVid Stereography (Inpainting & Synthesis)

Synthesizes the right-eye view through geometric warping and 1-step video diffusion inpainting.

| Parameter | Recommended (RTX 5090 32GB) | Value (General 24GB+) | Description |
|---|---|---|---|
| **M2SVid Processing Resolution** | **`1024x576 (Optimal 12GB)`** *(Sweet Spot)*<br>• `768x432 (Fastest)`<br>• `Native` *(Max Inpainting)* | `1024x576` / `Native` | **Sweet Spot:** `1024x576` only inpaints narrow disocclusion holes and alpha-blends the original 1080p plate everywhere else—giving **100% native sharpness** at 2× speed. Use `768x432` for pure speed (~12–15 fps). |
| **Generation Chunk Size** | **`14`** *(Fastest & Recommended)* | `14` – `16` | Frames evaluated per inpainting pass. **Keep at `14` for maximum speed:** avoids $O(T^2)$ quadratic temporal attention compute and decodes VAE in 1 fast shot without triggering chunked OOM fallbacks. Max `18`–`20` for Native 1080p. |
| **Parallel Chunk Batch Size** | **`2`** *(Non-Native only)* | `1` | **High-VRAM Saturation:** Batches multiple 14-frame chunks into a single GPU forward pass to utilize the RTX 5090's 32GB VRAM (~19–21 GB peak). **Available at non-native resolutions only** (`1024x576`, `768x432`, `1280x720`). Automatically clamped to `1` on Native 1080p to prevent OOM. |
| **Warping Batch Size** | **`16`** *(up to `64`)* | `8` – `12` | Geometric reprojection frame batching. `16` maximizes GPU parallelism, finishing in ~1 second. |
| **Disparity Scale** | `0.045` – `0.055` (Default `0.05`) | `0.045` – `0.055` | 3D depth separation. `0.05` is natural and comfortable; higher values increase pop-out. |
| **Convergence Point (Zero Parallax)** | `0.5` | `0.5` | Sets the screen-plane depth. Main subjects sit flush with the display; foreground pops out, background recedes. |
| **Mask Closing Kernel** | `11` | `11` | Morphological dilation kernel for disocclusion cleanup. |
| **Mask Antialias** | `False` | `False` | Prevents edge softening on disocclusion holes. |

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
- **Do** keep resolution at 1920:
  - On **24GB cards (RTX 3090/4090)**: set `Window Size` to `65`–`75`.
  - On **32GB cards (RTX 5090)**: set `Window Size` to `80` (full speed, zero slicing) or `110` (with `Attention Slicing: Auto`).

---

## GPU Optimization & 1:1 Native Resolution Settings Matrix

Diffusion self-attention memory scales quadratically ($O(N^2)$) with spatial tokens, while temporal attention scales with window size. Use this table to achieve **1:1 Native Resolution Depth**:

| GPU Hardware | VRAM | Target Depth Resolution | Window Size | Inference Steps | M2SVid Resolution | Gen Chunk Size | Warping Batch Size |
|---|---|---|---|---|---|---|---|
| **RTX 3060** | 12 GB | `Native` (≤720p) / `1024` (for 1080p)* | 110 (or 50 for native 1080p) | 7 – 20 | `1024x576` / `Native` | 8 – 10 | 2 |
| **RTX 3090 / 4090** | 24 GB | **`1920` (Native 1080p)** | **`65` – `75`** | 20 – 25 | **`Native`** | 14 – 16 | 8 – 12 |
| **RTX 5090** | 32 GB | **`1920` – `2048` (Native 1080p/1440p)** | **`80`** (Fastest) / **`110`** (w/ Slicing) | 20 – 25 | **`1024x576`** (Sweet Spot) / **`768x432`** / **`Native`** | **`14`** (Fastest) / **`18–20`** (Native) | **`16`** |
| **RTX 6000 Ada** | 48 GB | **`1920` – `2560` (Native 1080p/2K)** | **`110` – `120`** | 25 – 30 | **`Native`** | 24 – 28 | 16 |
| **RTX Pro 6000 / Blackwell** | 96 GB | **`1920` – `3840` (Native 1080p/2K/4K)** | **`110` – `150`** | 25 – 30 | **`Native`** | 32 – 35 | 16 |

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
- **Architecture & VRAM:** Blackwell `sm_120`, 32 GB GDDR7 @ 1,792 GB/s. Target 80–90% VRAM saturation (~25–28 GB) without hitting PyTorch CUDA OOM.
- **DepthCrafter (Step 1):**
  - **Max Resolution:** **`Native`** (e.g. `1920` for 1080p, `2048` for 2K). Preserves 1:1 micro-geometry, hair strands, and razor-sharp depth boundaries.
  - **Option A: Pure Speed / Zero Slicing:**
    - `Window Size = 60`, `Overlap = 12–15`, `Inference Steps = 20–25`.
    - `Attention Slicing: Disabled`, `CPU Offload: none`.
    - **Memory Profile:** Baseline model + VAE holds ~27.6 GB, leaving 3.76 GB free. A 60-frame attention spike is ~3.57 GB, fitting safely inside the 31.36 GB headroom with zero paging at **100% unconstrained hardware speed**.
  - **Option B: Maximum Context & Stability (Window 80–110 — Recommended):**
    - `Window Size = 80` (or `110`), `Overlap = 16–24`, `Inference Steps = 20–25`.
    - **`Attention Slicing: Auto`** (or `Enabled`).
    - **Why:** An 80-frame window at 1920x832 demands a **4.76 GiB spike**, which exceeds the 3.76 GB headroom without slicing. Slicing drops the spike from 4.76 GiB down to **~2.2 GiB**, running 80 or 110 frames effortlessly with only a ~5% speed difference.
    - *Note:* In StereoFaster, `Attention Slicing: Auto` now automatically detects `process_res >= 1536` and `window_size > 60` on 32GB GPUs to prevent this OOM automatically.
- **M2SVid (Step 2):**
  - **Profile A: Sweet Spot (Max Quality & Speed — Recommended):**
    - **Processing Resolution:** **`1024x576 (Optimal 12GB)`** (Inpaints disocclusion holes at 576p and alpha-blends the native 1080p source plate everywhere else for 100% sharp visuals at 2× speed).
    - **Generation Chunk Size:** **`14`** (~1.2s/chunk, avoids $O(T^2)$ quadratic temporal penalty, 1-shot VAE decode).
    - **Parallel Chunk Batch Size:** **`3`** (Saturates ~24–26 GB VRAM on the RTX 5090, processing 3 chunks simultaneously).
    - **Warping Batch Size:** **`32`** (max GPU parallelism).
    - **Video Export:** **GPU NVENC Hardware Encoder (`h264_nvenc`)** with bulk DMA transfers, reducing multi-stream video export from ~20s to ~2s per clip.
    - **Throughput:** **~12–15 fps** (~22–25 GB peak VRAM).
  - **Profile B: Absolute Fastest Throughput:**
    - **Processing Resolution:** **`768x432 (Fastest)`**, **Generation Chunk Size:** **`14`**, **Warping Batch:** **`16`**.
    - **Throughput:** **~12–15 fps** (~3.5s for 57 frames; ~4.5 mins for 3-minute video; ~8 GB peak VRAM).
  - **Profile C: Full Native 1080p Inpainting:**
    - **Processing Resolution:** **`Native`**, **Generation Chunk Size:** **`14` – `18`** (keep $\le 20$ to prevent VAE decode spikes), **Warping Batch:** **`16`**.
    - **Throughput:** **~4 fps** (~17 mins for 3-minute video; ~26 GB peak VRAM).
  - **Disparity Scale:** `0.045` – `0.055` (default `0.05`).
  - **Convergence Point:** `0.5` (zero parallax subject lock).
  - **Config:** Automatically loads `m2svid.yaml` (full temporal cross-attention).
- **Environment & Drivers:**
  - Requires `torch==2.11.0+cu128` (CUDA 12.8/12.9) for native `sm_120` support.
  - Automatically runs with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to eliminate memory fragmentation.

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
- Set **Window Size**:
  - On **24GB GPUs (RTX 3090/4090)**: `65`–`75`.
  - On **32GB GPUs (RTX 5090)**: `80` (or `110` with `Attention Slicing: Auto`).
- In Step 2, keep **M2SVid Processing Resolution** set to **`Native`**.

### Q: Why did DepthCrafter give `CUDA out of memory` (OOM) on an RTX 5090?
- **Root Cause:** At native 1080p ($1920 \times 1080$ / $1920 \times 832$), DepthCrafter's baseline model, VAE, and initial cache occupy ~27.59 GB of VRAM, leaving **3.76 GB of free headroom** on a 32 GB card.
- When running an 80-frame window, the temporal attention and feed-forward layer (`self.ff` / GeLU activation) demands a **4.76 GiB spike** ($4.76 > 3.76$ GiB free by ~1.0 GiB), crashing with CUDA OOM.
- **The Solution:**
  1. **Option A (Recommended — Window 80 to 110 with Attention Slicing):**
     - Set **`Attention Slicing = Auto`** (or `Enabled`).
     - StereoFaster's updated `Auto` logic now detects $1920\text{p}$ + window $> 60$ on 24GB/32GB cards and automatically slices attention heads, reducing the 4.76 GiB spike down to **~2.2 GiB** (fits easily in 3.76 GB free with zero OOM and only a ~5% speed difference).
  2. **Option B (CPU Offload):**
     - Set **`CPU Offload = model`** in the WebUI. This offloads the VAE and text encoder to system RAM during UNet execution, freeing **~4.0 GB of VRAM** and expanding free headroom to **~7.7 GB**.
  3. **Option C (Pure Speed / Zero Slicing / Zero Offload):**
     - Set **`Window Size = 60`** (Overlap: `12–15`). At 60 frames, the spike drops to **~3.57 GiB**, fitting inside the 3.76 GiB headroom without slicing.

### Q: What PyTorch version is required for RTX 5090 (Blackwell `sm_120`)?
- Standard PyTorch builds (`cu121`, `cu124`, `cu126`) only support architectures up to `sm_90` (Hopper) and will throw `UserWarning: sm_120 is not compatible`.
- Unpinned `pip install torch ... cu128` can pull CUDA 13 wheels (`2.14.0+cu130`), which fail with `The NVIDIA driver on your system is too old (found version 12090)`.
- **The Working Build:** Pin `torch==2.11.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128`.
- If `torchaudio` causes a `libcudart.so.13` error on launch, run `pip uninstall -y torchaudio` (StereoFaster does not require audio libraries).

### Q: Why does Generation Chunk Size 14 run faster than Chunk Size 35 in M2SVid?
- **Root Cause:** Temporal self-attention in diffusion models scales **quadratically ($O(T^2)$)** with frame count:
  - For **14 frames**: $14^2 = 196$ attention operations per spatial token.
  - For **35 frames**: $35^2 = 1,225$ attention operations per spatial token (**$6.25\times$ more compute!**).
- Furthermore, decoding 35 frames in one shot in the temporal VAE can exceed memory limits and trigger the chunked fallback.
- **Result:** Chunk Size `14` processes frames at **~0.08s/frame** (~1.2s per 14-frame chunk), whereas Chunk Size `35` runs at **~0.38s/frame** (~13s per 35-frame chunk). Always use **`14`** for maximum speed.

### Q: Why does `1024x576 (Optimal 12GB)` look identical to `Native 1080p`?
- **Disocclusion Inpainting vs. Full Generation:** M2SVid does not regenerate the entire video from scratch; it **only inpaints the narrow disocclusion holes** (the blind spots behind moving foreground objects revealed by warping).
- The pipeline alpha-blends the original, uncompressed native 1080p source plate everywhere else (>95% of the frame).
- Therefore, running M2SVid at `1024x576` retains **100% native 1080p clarity across all visible details**, runs nearly **2× faster**, and uses under 12 GB VRAM.

### Q: Why is `1280x720` labeled `(Faster)` in the resolution dropdown?
- The label `(Faster)` on `1280x720` is relative to **`Native (1920x1080)`** (i.e., 720p is faster than 1080p).
- However, **`1024x576` is faster than `1280x720`** (~35% fewer pixels), and **`768x432` is the fastest**.

### Q: What does `[M2SVid] ⚠️ VAE Decode OOM detected... Running chunked VAE decode fallback` mean?
- **Explanation:** When processing high-resolution frames or large chunks (e.g., 35 frames), SVD's 3D temporal VAE decoder generates large 5D intermediate tensor buffers that exceed the single-operation VRAM buffer limit.
- **Safety Net:** This is **not a crash**. StereoFaster includes an automatic chunked VAE fallback that intercepts the OOM, splits the latents temporally into 8-frame sub-chunks, decodes them cleanly, and reconstructs the full video. To avoid triggering the fallback entirely, keep **Generation Chunk Size at `14`**.
