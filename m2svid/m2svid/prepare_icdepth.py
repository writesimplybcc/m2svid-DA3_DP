"""
ICDepth (In-Context Depth / Wan2.1 DiT) Module for StereoFaster
Temporal video depth estimation using Alibaba's Wan2.1 Diffusion Transformer.
Bypasses the 11.36 GB UMT5 text encoder via pre-computed static prompt context (8 MB).
Produces normalized disparity arrays compatible with Step 2 (M2SVid Warping & Inpainting).
"""

import os
import sys
import gc
import cv2
import time
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Suppress harmless cuDNN benchmark/plan fallback warnings on Ampere (RTX 3060)
warnings.filterwarnings("ignore", message=".*cudnnException.*")
warnings.filterwarnings("ignore", message=".*CUDNN_STATUS_NOT_SUPPORTED.*")

SF_LOG = logging.getLogger("stereofaster")

# Ensure ICDepth root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ICDEPTH_ROOT = PROJECT_ROOT / "m2svid" / "third_party" / "ICDepth"

for candidate in [
    ICDEPTH_ROOT,
    Path("/app/m2svid/third_party/ICDepth"),
    PROJECT_ROOT / "third_party" / "ICDepth",
]:
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

_cached_icd_pipe = None


def find_checkpoint_path(filename: str, fallback_patterns: List[str] = None) -> Optional[Path]:
    """Search for checkpoint files in standard StereoFaster locations and HF cache."""
    search_dirs = [
        ICDEPTH_ROOT / "checkpoints",
        Path("/app/m2svid/third_party/ICDepth/checkpoints"),
        PROJECT_ROOT / "checkpoints",
        Path("/app/checkpoints"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        direct = s_dir / filename
        if direct.exists() and direct.is_file():
            return direct
        # Recursive glob if needed
        try:
            matches = list(s_dir.glob(f"**/{filename}"))
            if matches:
                return matches[0]
        except Exception:
            pass

    if fallback_patterns:
        for s_dir in search_dirs:
            if not s_dir.exists():
                continue
            for pattern in fallback_patterns:
                try:
                    matches = list(s_dir.glob(f"**/{pattern}"))
                    if matches:
                        return matches[0]
                except Exception:
                    pass
    return None


class StaticPromptEmbedder:
    """Replaces PipelineUnit PromptEmbedder to avoid loading the 11.36 GB UMT5-XXL text encoder.
    Returns pre-baked embeddings for 'Estimate the depth' (positive) and '' (negative)."""
    def __init__(self, static_context_path: str):
        self.seperate_cfg = True
        self.input_params_posi = {"prompt": "prompt", "positive": "positive"}
        self.input_params_nega = {"prompt": "negative_prompt", "positive": "positive"}
        self.onload_model_names = ()
        self.input_params = None
        self.data = torch.load(static_context_path, map_location="cpu")
        SF_LOG.info(f"[ICDepth] Initialized StaticPromptEmbedder from {static_context_path} (8 MB)")

    def process(self, pipe, prompt, positive):
        key = "positive" if positive else "negative"
        emb = self.data[key].to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"context": emb}


def get_icdepth_pipeline(device: str = "cuda:0", torch_dtype=torch.bfloat16):
    """Load and cache the ICDepth DiT + VAE + DINOv2 pipeline with VRAM management."""
    global _cached_icd_pipe
    if _cached_icd_pipe is not None:
        return _cached_icd_pipe

    dit_path = find_checkpoint_path("Wan2.1-depth-1212_with_quant_epoch-5.safetensors")
    vae_path = find_checkpoint_path("Wan2.1_VAE.pth")
    dino_path = find_checkpoint_path("video_depth_anything_vitl.pth")
    static_context_path = find_checkpoint_path("icdepth_static_context.pt")

    if not dit_path:
        raise FileNotFoundError(
            "ICDepth DiT checkpoint 'Wan2.1-depth-1212_with_quant_epoch-5.safetensors' not found.\n"
            "Please ensure it is placed in m2svid/third_party/ICDepth/checkpoints/."
        )
    if not vae_path:
        raise FileNotFoundError(
            "Wan2.1 VAE checkpoint 'Wan2.1_VAE.pth' not found.\n"
            "Please ensure it is placed in m2svid/third_party/ICDepth/checkpoints/."
        )
    if not dino_path:
        raise FileNotFoundError(
            "Video Depth Anything prior 'video_depth_anything_vitl.pth' not found."
        )
    if not static_context_path:
        raise FileNotFoundError(
            "Pre-computed static prompt context 'icdepth_static_context.pt' not found."
        )

    SF_LOG.info(f"[ICDepth] Loading DiT from {dit_path}")
    SF_LOG.info(f"[ICDepth] Loading VAE from {vae_path}")
    SF_LOG.info(f"[ICDepth] Loading DINOv2 prior from {dino_path}")

    from icdepth import ICDepthPipeline, ModelManager

    target_dev = "cuda:0" if "cuda" in str(device) and (str(device) == "cuda" or (isinstance(device, torch.device) and device.index is None)) else device
    model_manager = ModelManager(torch_dtype=torch_dtype, device=target_dev)
    model_manager.load_models([str(dit_path), str(vae_path)])

    pipe = ICDepthPipeline.from_model_manager(
        model_manager,
        torch_dtype=torch_dtype,
        device=target_dev,
        dino_checkpoint=str(dino_path),
    )

    # Patch PromptEmbedder with lightweight StaticPromptEmbedder
    new_units = []
    for unit in pipe.units:
        if unit.__class__.__name__ == "PromptEmbedder":
            new_units.append(StaticPromptEmbedder(str(static_context_path)))
        else:
            new_units.append(unit)
    pipe.units = new_units

    # Enable intelligent VRAM offload/onload management
    pipe.enable_vram_management()
    _cached_icd_pipe = pipe
    SF_LOG.info("[ICDepth] Pipeline successfully cached and ready for inference.")
    return _cached_icd_pipe


def unload_icdepth_model():
    """Unload ICDepth model from memory and free GPU VRAM."""
    global _cached_icd_pipe
    if _cached_icd_pipe is not None:
        try:
            if hasattr(_cached_icd_pipe, "load_models_to_device"):
                _cached_icd_pipe.load_models_to_device([])
            if hasattr(_cached_icd_pipe, "dino_extractor") and hasattr(_cached_icd_pipe.dino_extractor, "model"):
                _cached_icd_pipe.dino_extractor.model.to("cpu")
            if hasattr(_cached_icd_pipe, "vae") and _cached_icd_pipe.vae is not None:
                _cached_icd_pipe.vae.to("cpu")
        except Exception:
            pass
        _cached_icd_pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    SF_LOG.info("[ICDepth] Model successfully unloaded and VRAM cleared.")


def _directional_median_2d(d: np.ndarray, step: int = 1) -> np.ndarray:
    """4-direction 1D median filter (0°, 45°, 90°, 135°) with given pixel stride.

    Acts as an exact mathematical identity on all monotonic surfaces and step edges
    (preserving 1px solid silhouettes and thin hair strands), while eliminating 2D
    period-2 and period-3 VAE halftone/checkerboard stippling.
    """
    p = np.pad(d, step, mode="edge")
    c = p[step:-step, step:-step]

    def _med3(a, b, c_val):
        return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c_val))

    s2 = 2 * step
    h_med = _med3(p[step:-step, :-s2], c, p[step:-step, s2:])
    v_med = _med3(p[:-s2, step:-step], c, p[s2:, step:-step])
    d1_med = _med3(p[:-s2, :-s2], c, p[s2:, s2:])
    d2_med = _med3(p[:-s2, s2:], c, p[s2:, :-s2])

    lo1, hi1 = np.minimum(h_med, v_med), np.maximum(h_med, v_med)
    lo2, hi2 = np.minimum(d1_med, d2_med), np.maximum(d1_med, d2_med)
    return (0.5 * (np.maximum(lo1, lo2) + np.minimum(hi1, hi2))).astype(np.float32)


def descreen_icdepth_disparity(disparity_seq: np.ndarray) -> np.ndarray:
    """Inverse-halftoning pass for ICDepth Wan2.1 VAE output.

    Removes the 2-3px diagonal checkerboard/halftone stipple synthesized by the
    frozen RGB Video VAE on fine hair boundaries while keeping solid object edges
    and thin curly hair strands 100% crisp.
    """
    out_frames = np.empty_like(disparity_seq, dtype=np.float32)
    kernel_3x3 = np.ones((3, 3), np.float32)
    for i in range(len(disparity_seq)):
        d = disparity_seq[i].astype(np.float32)
        dm1 = _directional_median_2d(d, step=1)
        dm2 = _directional_median_2d(dm1, step=2)
        dm3 = _directional_median_2d(dm2, step=1)

        ht_residual = np.abs(d - dm3)
        ht_density = cv2.GaussianBlur(cv2.dilate(ht_residual, kernel_3x3), (5, 5), 1.2)
        mask = np.clip((ht_density - 0.0015) / 0.008, 0.0, 1.0)

        demod = cv2.bilateralFilter(dm3, d=5, sigmaColor=0.12, sigmaSpace=1.3)
        out_frames[i] = dm3 * (1.0 - mask) + demod * mask
    return out_frames


def recover_black_void_and_titles(depth_seq: np.ndarray, rgb_frames) -> np.ndarray:
    """Automatic Black-Void & 3D Title-Card Recovery for ICDepth.

    Detects frames where the active image region (excluding top/bottom letterbox bars)
    is >= 60% pure digital black void (e.g., studio logos, flying 3D title cards like
    INCEPTION). Suppresses the Wan2.1/DINOv2 hallucinated room/corridor background on
    the black void while preserving genuine 3D perspective depth on moving objects and
    synthesizing clean 3D bevel/luminance relief when blocks lock into flat 2D typography.
    """
    T, H, W = depth_seq.shape
    if T == 0 or rgb_frames is None or len(rgb_frames) != T:
        return depth_seq

    y0, y1 = int(H * 0.14), int(H * 0.86)

    def _get_rgb(idx: int) -> np.ndarray:
        f = rgb_frames[idx]
        arr = np.asarray(f, dtype=np.uint8)
        if arr.shape[:2] != (H, W):
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)
        return arr

    # 1. Measure pure-black void fraction in the active image region per frame
    active_void_ratio = np.zeros(T, dtype=np.float32)
    for t in range(T):
        rgb_t = _get_rgb(t)
        cmax_act = rgb_t[y0:y1].max(axis=-1)
        active_void_ratio[t] = float((cmax_act <= 8).mean())

    # Smooth void ratio temporally so transitions into/out of title cards are seamless
    if T >= 5:
        active_void_ratio = np.convolve(active_void_ratio, np.ones(5) / 5.0, mode="same")

    if active_void_ratio.max() < 0.60:
        return depth_seq

    SF_LOG.info(
        f"[ICDepth] Black-Void / Title-Card shot detected (max void ratio={active_void_ratio.max():.1%}); "
        "applying 3D foreground & title relief recovery."
    )
    out = depth_seq.copy().astype(np.float32)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for t in range(T):
        v_ratio = float(active_void_ratio[t])
        if v_ratio < 0.50:
            continue

        # Blend weight for frames transitioning into/out of a black-void shot
        shot_alpha = float(np.clip((v_ratio - 0.50) / 0.15, 0.0, 1.0))

        rgb = _get_rgb(t).astype(np.float32)
        luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        cmax = rgb.max(axis=-1)
        d_orig = depth_seq[t].astype(np.float32)

        # Soft foreground matte from non-void pixels
        fg_hard = (cmax > 10.0).astype(np.float32)
        fg_dilated = cv2.dilate(fg_hard, k_dilate, iterations=1)
        fg_alpha = np.clip((cmax - 5.0) / 14.0, 0.0, 1.0)
        fg_alpha = np.maximum(fg_alpha, cv2.GaussianBlur(fg_dilated, (5, 5), 1.0) * 0.35)
        fg_alpha = np.where(cmax <= 4.0, 0.0, fg_alpha)

        fg_mask = cmax > 12.0
        void_mask = cmax <= 5.0

        if fg_mask.sum() < 50:
            d_fixed = np.zeros_like(d_orig)
        else:
            # Estimate local hallucinated background floor from surrounding void
            void_vals = d_orig[void_mask]
            bg_floor = float(np.percentile(void_vals, 20)) if void_vals.size > 100 else 0.0

            fg_d = d_orig[fg_mask]
            fg_spread = float(np.percentile(fg_d, 95) - np.percentile(fg_d, 10))
            fg_hi = float(np.percentile(fg_d, 90))

            # Genuine 3D depth component (preserves perspective gradients on flying 3D blocks)
            d_3d = np.clip(d_orig - bg_floor, 0.0, 1.0)
            if fg_hi - bg_floor > 0.05:
                d_3d = d_3d / max(0.25, float(np.percentile(d_3d[fg_mask], 98)))
                d_3d = np.clip(d_3d, 0.0, 1.0)

            # 3D bevel + luminance relief component (activates when blocks lock into flat 2D letters)
            fg_luma_p95 = max(25.0, float(np.percentile(luma[fg_mask], 95)))
            luma_norm = np.clip(luma / fg_luma_p95, 0.0, 1.0)

            dist = cv2.distanceTransform((cmax > 10.0).astype(np.uint8), cv2.DIST_L2, 5)
            dist_max = max(3.0, float(np.percentile(dist[fg_mask], 90)) if fg_mask.any() else 3.0)
            bevel = np.clip(dist / dist_max, 0.0, 1.0)

            # Target stereo pop depth (~0.42 disparity) with metallic face detail & 3D bevel
            d_relief = 0.22 + 0.14 * luma_norm + 0.08 * bevel

            # Smoothly blend: when ICDepth has strong 3D spread (flying blocks), trust d_3d;
            # when ICDepth collapses to flat background (locked 2D title), transition to d_relief
            w_3d = float(np.clip((fg_spread - 0.04) / 0.14, 0.0, 1.0))
            d_fg = w_3d * np.maximum(d_3d, d_relief * 0.65) + (1.0 - w_3d) * d_relief
            d_fixed = d_fg * fg_alpha

        out[t] = (1.0 - shot_alpha) * d_orig + shot_alpha * d_fixed

    # Light 3-frame temporal smoothing on pure-flat title frames to prevent micro-flicker
    out_smooth = out.copy()
    for t in range(1, T - 1):
        if active_void_ratio[t] >= 0.70:
            out_smooth[t] = 0.2 * out[t - 1] + 0.6 * out[t] + 0.2 * out[t + 1]
    return np.clip(out_smooth, 0.0, 1.0)


def recover_volumetric_splash_aeration(
    depth_seq: np.ndarray,
    rgb_frames,
    repair_strength: float = 1.0,
) -> np.ndarray:
    """Volumetric Splash & Water Aeration Recovery for ICDepth.

    Automatically detects underwater plunge aeration, cavitation, and turbulent bubble plumes
    where monocular depth models hallucinate an unnatural dark sunken hole behind the subject.
    Reconstructs continuous foreground/midground depth from the water surface plunge entry
    down around the occluded torso and connecting to the limbs.
    """
    strength = float(np.clip(repair_strength, 0.0, 1.0))
    if strength <= 1e-4 or depth_seq is None or rgb_frames is None:
        return depth_seq

    T, H, W = depth_seq.shape
    if T == 0 or len(rgb_frames) != T:
        return depth_seq

    def _get_bgr(idx: int) -> np.ndarray:
        f = rgb_frames[idx]
        if isinstance(f, Image.Image):
            arr = np.array(f, dtype=np.uint8)
            if arr.shape[:2] != (H, W):
                arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        arr = np.asarray(f, dtype=np.uint8)
        if arr.shape[:2] != (H, W):
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)
        return arr

    underwater_frames = []
    X_2d, Y_2d = np.meshgrid(np.arange(W), np.arange(H))

    for t in range(T):
        bgr = _get_bgr(t)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H_ch, S_ch, V_ch = cv2.split(hsv)
        b, g, r = cv2.split(bgr)

        film_content = (V_ch > 15.0)
        if np.mean(film_content) < 0.2:
            continue

        lower_region = Y_2d > int(H * 0.28)
        water_pixels = (H_ch > 60) & (H_ch < 135) & (V_ch > 25) & lower_region & film_content
        water_ratio = float(np.mean(water_pixels))

        if water_ratio > 0.20:
            froth = (b > 85) & (g > 95) & lower_region & film_content
            froth_ratio = float(np.mean(froth))

            if froth_ratio > 0.04:
                froth_depth = float(np.median(depth_seq[t][froth]))
                upper_water = (Y_2d < int(H * 0.25)) & film_content
                surface_depth = float(np.median(depth_seq[t][upper_water])) if np.any(upper_water) else 0.30

                # Deep sunken cavity check
                if froth_depth < (surface_depth - 0.06) or froth_depth < 0.18:
                    underwater_frames.append(t)

    if not underwater_frames:
        return depth_seq

    SF_LOG.info(f"[ICDepth] Volumetric splash recovery: processing {len(underwater_frames)} aerated plunge frames (strength={strength:.2f}).")
    fixed_depth = depth_seq.copy()
    start_t = min(underwater_frames)

    for t in range(max(0, start_t - 2), T):
        bgr = _get_bgr(t)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        V_ch = hsv[:, :, 2].astype(np.float32)
        d = fixed_depth[t]

        film_content = (V_ch > 15.0).astype(np.float32)
        film_content = cv2.erode(film_content, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        t_weight = min(1.0, max(0.0, (t - max(0, start_t - 2)) / 5.0))

        valid_side = ((X_2d < int(W * 0.22)) | (X_2d > int(W * 0.78))) & (film_content > 0.5)
        bg_water_V = float(np.percentile(V_ch[valid_side], 25)) if np.any(valid_side) else 45.0
        excess_V = np.maximum(0.0, V_ch - bg_water_V)

        bubble_mask = np.clip(excess_V / 45.0, 0.0, 1.0) * film_content
        bubble_mask = cv2.GaussianBlur(bubble_mask, (15, 15), 0)

        top_mask = (Y_2d < int(H * 0.25)) & (film_content > 0.5)
        surface_ref = float(np.median(d[top_mask])) if np.any(top_mask) else 0.33
        surface_ref = max(0.28, min(0.40, surface_ref))

        bot_mask = (Y_2d > int(H * 0.75)) & (film_content > 0.5) & (d > 0.18)
        bot_ref = float(np.median(d[bot_mask])) if np.any(bot_mask) else 0.25
        bot_ref = max(0.22, min(0.32, bot_ref))

        y_norm = np.clip((Y_2d - int(H * 0.16)) / float(max(1, int(H * 0.68))), 0.0, 1.0)
        torso_plunge_guide = surface_ref * (1.0 - y_norm) + bot_ref * y_norm
        bubble_target_depth = torso_plunge_guide + 0.05 * bubble_mask

        elevate = np.maximum(0.0, bubble_target_depth - d)
        fixed_depth[t] = d + elevate * bubble_mask * t_weight * strength

    for t in range(max(1, start_t), T - 1):
        fixed_depth[t] = 0.20 * fixed_depth[t - 1] + 0.60 * fixed_depth[t] + 0.20 * fixed_depth[t + 1]

    return np.clip(fixed_depth, 0.0, 1.0)


def load_clip_credit_sidecar(video_path: str) -> str:
    """Load per-clip credit text hint from a sidecar file next to the source video if present.

    Looks for `{stem}.credits.txt` or `{stem}.txt` (and base stem without `_cropped`).
    """
    if not video_path:
        return ""
    vp = Path(video_path)
    parent = vp.parent if vp.parent.exists() else (PROJECT_ROOT / "source_videos")
    stem = vp.stem
    stems_to_check = [stem]
    if stem.endswith("_cropped"):
        stems_to_check.append(stem[:-8])
    for s in stems_to_check:
        for ext in (".credits.txt", ".txt"):
            cand = parent / f"{s}{ext}"
            if cand.is_file():
                try:
                    return cand.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
    return ""


def save_clip_credit_sidecar(video_path: str, credit_text: str) -> Optional[Path]:
    """Save or remove the per-clip `{stem}.credits.txt` sidecar file next to the source video."""
    if not video_path:
        return None
    vp = Path(video_path)
    parent = vp.parent if vp.parent.exists() else (PROJECT_ROOT / "source_videos")
    sidecar_path = parent / f"{vp.stem}.credits.txt"
    clean = (credit_text or "").strip()
    try:
        if clean:
            sidecar_path.write_text(clean + "\n", encoding="utf-8")
            SF_LOG.info(f"[ICDepth] Saved credit sidecar: {sidecar_path}")
            return sidecar_path
    except Exception as e:
        SF_LOG.warning(f"[ICDepth] Could not write credit sidecar {sidecar_path}: {e}")
    return None


def detect_and_elevate_credits(
    depth_seq: np.ndarray,
    rgb_frames,
    credit_text_hint: str = "",
    scene_max_disp: float = 0.85,
    repair_strength: float = 1.0,
) -> np.ndarray:
    """Automatic & Text-Guided On-Screen Credit / Title Foreground Pop for ICDepth.

    1. Accurately detects static 2D on-screen credits/titles (e.g. 'Directed by Vania Heymann / Gal Muggia',
       rainbow 'COLDPLAY', or mid-clip 3-line 'UP & UP') over moving live-action scenes.
    2. When `credit_text_hint` is provided by the user (or via `{stem}.credits.txt`), uses half/full-clip
       temporal-median RGB + Gaussian chroma upsampling (undoing 4:2:0 blockiness) + continuous soft stroke
       anti-aliasing for smooth, grit-free display letters.
    3. Preserves dark and mid-ground scene depth 100% untouched (soft knee only above 0.72 -> scene_max_disp)
       so background/mid-ground subjects (e.g. the kid swinging from the gantry) never lose brightness or contrast.
    4. Blends by `repair_strength` in [0.0, 1.0] so the user can interactively pull back or dial in the repair.
    """
    strength = float(np.clip(repair_strength, 0.0, 1.0))
    if strength <= 1e-4:
        return depth_seq

    T, H, W = depth_seq.shape
    if T < 6 or rgb_frames is None or len(rgb_frames) != T:
        return depth_seq

    hint_clean = (credit_text_hint or "").strip()
    hint_lines = [ln.strip() for ln in hint_clean.replace("|", "\n").splitlines() if ln.strip()]
    expected_chars = sum(len(ln.replace(" ", "")) for ln in hint_lines)
    has_hint = expected_chars >= 2

    # Skip pure black-void shots (already handled by recover_black_void_and_titles) unless a hint is explicitly given
    y0, y1 = int(H * 0.14), int(H * 0.86)
    mid_rgb = np.asarray(rgb_frames[T // 2], dtype=np.uint8)
    if mid_rgb.shape[:2] != (H, W):
        mid_rgb = cv2.resize(mid_rgb, (W, H), interpolation=cv2.INTER_AREA)
    if not has_hint and float((mid_rgb[y0:y1].max(axis=-1) <= 8).mean()) >= 0.60:
        return depth_seq

    k_sm_sz = max(9, int(round(min(H, W) * 0.03)) | 1)
    k_lg_sz = max(21, int(round(min(H, W) * 0.07)) | 1)
    k_sm = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_sm_sz, k_sm_sz))
    k_lg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_lg_sz, k_lg_sz))

    n_samp = min(T, 36)
    sample_idx = np.linspace(0, T - 1, n_samp, dtype=int)
    rgbs = []
    for idx in sample_idx:
        rgb = np.asarray(rgb_frames[idx], dtype=np.float32)
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        rgbs.append(rgb)
    rgb_stack = np.stack(rgbs, axis=0)
    med_rgb = np.median(rgb_stack, axis=0)

    if has_hint:
        half = max(4, n_samp // 2)
        windows = [rgb_stack, rgb_stack[: min(n_samp, half + 2)], rgb_stack[max(0, half - 2) :]]
        best_seed = np.zeros((H, W), dtype=bool)
        best_soft = np.zeros((H, W), dtype=np.float32)
        for w_arr in windows:
            med_rgb = np.median(w_arr, axis=0)
            lu = 0.299 * med_rgb[:, :, 0] + 0.587 * med_rgb[:, :, 1] + 0.114 * med_rgb[:, :, 2]
            # Smooth 4:2:0 chroma subsampling blocks before computing saturation Top-Hat
            rgb_blur = cv2.GaussianBlur(med_rgb, (3, 3), 0.75)
            sat = rgb_blur.max(axis=-1) - rgb_blur.min(axis=-1)

            th_lu = np.maximum(
                cv2.morphologyEx(lu, cv2.MORPH_TOPHAT, k_sm),
                cv2.morphologyEx(lu, cv2.MORPH_TOPHAT, k_lg),
            )
            th_ch = cv2.morphologyEx(sat, cv2.MORPH_TOPHAT, k_lg)

            temp_std = np.std(w_arr, axis=0).max(axis=-1)

            w_seed = (th_lu > 32.0) & (lu > 130.0) & (sat < 65.0) & (temp_std < 40.0)
            c_seed = (np.maximum(th_ch, th_lu) > 28.0) & (sat > 40.0) & (temp_std < 35.0)
            best_seed |= (w_seed | c_seed)

            stability = np.clip(1.0 - (temp_std - 10.0) / 25.0, 0.0, 1.0)
            w_soft = np.clip((th_lu - 15.0) / 40.0, 0.0, 1.0) * np.clip((lu - 100.0) / 50.0, 0.0, 1.0) * stability
            c_soft = np.clip((np.maximum(th_ch, th_lu) - 15.0) / 38.0, 0.0, 1.0) * np.clip((sat - 25.0) / 35.0, 0.0, 1.0) * stability
            best_soft = np.maximum(best_soft, np.maximum(w_soft, c_soft))

        cand = best_seed
        min_chars_req = max(2, min(4, expected_chars // 2))
    else:
        lu_stack = 0.299 * rgb_stack[:, :, :, 0] + 0.587 * rgb_stack[:, :, :, 1] + 0.114 * rgb_stack[:, :, :, 2]
        sat_stack = rgb_stack.max(axis=-1) - rgb_stack.min(axis=-1)
        th_lu_stack = np.stack([cv2.morphologyEx(lu, cv2.MORPH_TOPHAT, k_sm) for lu in lu_stack], axis=0)
        p60_th = np.percentile(th_lu_stack, 60, axis=0)
        p30_th = np.percentile(th_lu_stack, 30, axis=0)
        med_sat = np.percentile(sat_stack, 50, axis=0)
        med_lu = np.percentile(lu_stack, 50, axis=0)
        cand = (p30_th > 28.0) & (p60_th > 42.0) & (med_lu > 125.0) & (med_sat < 45.0)
        best_soft = np.clip((p60_th - 16.0) / max(10.0, float(np.percentile(p60_th, 99)) - 16.0), 0.0, 1.0)
        min_chars_req = 5

    margin_y, margin_x = int(H * 0.06), int(W * 0.06)
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[margin_y:H - margin_y, margin_x:W - margin_x] = True
    cand &= border_mask

    if cand.sum() < (30 if has_hint else 80):
        return depth_seq

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    max_glyph_h = int(H * (0.25 if has_hint else 0.14))
    max_glyph_w = int(W * (0.25 if has_hint else 0.20))
    max_glyph_area = int(H * W * (0.025 if has_hint else 0.01))

    raw_glyphs = []
    for lbl in range(1, num_labels):
        x, y, bw, bh, area = stats[lbl]
        if 3 <= area <= max_glyph_area and 3 <= bh <= max_glyph_h and 2 <= bw <= max_glyph_w:
            raw_glyphs.append((lbl, x, y, bw, bh, area))

    if len(raw_glyphs) < min_chars_req:
        return depth_seq

    # When large display letters are present, drop tiny isolated noise specks (<10% of top-5 glyph area)
    sorted_areas = sorted([g[5] for g in raw_glyphs], reverse=True)
    ref_area = float(np.median(sorted_areas[: min(5, len(sorted_areas))]))
    min_glyph_area = max(3, int(ref_area * 0.10)) if ref_area > 250 else 3

    valid_glyph_mask = np.zeros((H, W), dtype=np.uint8)
    glyph_boxes = []
    for lbl, x, y, bw, bh, area in raw_glyphs:
        if area >= min_glyph_area:
            valid_glyph_mask[labels == lbl] = 1
            glyph_boxes.append((x, y, bw, bh))

    if len(glyph_boxes) < min_chars_req:
        return depth_seq

    k_group = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, int(W * 0.065)), max(14, int(H * 0.05))))
    grouped = cv2.morphologyEx(valid_glyph_mask, cv2.MORPH_CLOSE, k_group)
    n_blk, blk_labels, blk_stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)

    min_aspect = 0.60 if has_hint else 1.10
    candidate_blocks = []
    for b_id in range(1, n_blk):
        bx, by, bw, bh, barea = blk_stats[b_id]
        n_chars = sum(1 for (gx, gy, gw, gh) in glyph_boxes if bx <= gx <= bx + bw and by <= gy <= by + bh)
        if n_chars >= min_chars_req and (bw / max(1, bh)) >= min_aspect and bw <= int(W * 0.92) and bh <= int(H * 0.65):
            candidate_blocks.append((b_id, n_chars, bx, by, bw, bh))

    if not candidate_blocks:
        return depth_seq

    block_mask = np.zeros((H, W), dtype=bool)
    if has_hint and len(candidate_blocks) > 1:
        def _score_block(item):
            b_id, n_chars, bx, by, bw, bh = item
            char_diff = abs(n_chars - expected_chars) / max(1.0, float(expected_chars))
            cx, cy = (bx + 0.5 * bw) / W, (by + 0.5 * bh) / H
            center_dist = np.hypot(cx - 0.5, cy - 0.5)
            return -(char_diff + 0.5 * center_dist)

        candidate_blocks.sort(key=_score_block, reverse=True)
        best_id, _, best_x, best_y, best_w, best_h = candidate_blocks[0]
        best_cy = best_y + 0.5 * best_h
        block_mask[blk_labels == best_id] = True
        for b_id, n_c, bx, by, bw, bh in candidate_blocks[1:]:
            cy = by + 0.5 * bh
            same_line = abs(cy - best_cy) <= int(H * 0.08)
            vert_stack = abs(by - best_y) <= int(H * 0.18) and not (bx + bw < best_x - int(W * 0.1) or bx > best_x + best_w + int(W * 0.1))
            if same_line or vert_stack:
                block_mask[blk_labels == b_id] = True
        # Also include any valid glyphs on the same horizontal title line separated by wide kerning (e.g. L -> D, L -> A)
        line_y0 = max(0, best_y - int(H * 0.03))
        line_y1 = min(H, best_y + best_h + int(H * 0.03))
        block_mask[line_y0:line_y1, :] |= (valid_glyph_mask[line_y0:line_y1, :] > 0)
    else:
        for b_id, _, _, by, _, bh in candidate_blocks:
            block_mask[blk_labels == b_id] = True
            line_y0 = max(0, by - int(H * 0.03))
            line_y1 = min(H, by + bh + int(H * 0.03))
            block_mask[line_y0:line_y1, :] |= (valid_glyph_mask[line_y0:line_y1, :] > 0)

    credit_region = (valid_glyph_mask > 0) & block_mask
    # Define tight text lines / bounds zone
    k_zone = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(W * 0.015)), max(5, int(H * 0.015))))
    text_zone = cv2.dilate(block_mask.astype(np.uint8), k_zone, iterations=1).astype(bool)

    # Continuous sub-pixel typographic luminance signal extracted from noise-free temporal median
    med_lu_ref = 0.299 * med_rgb[:, :, 0] + 0.587 * med_rgb[:, :, 1] + 0.114 * med_rgb[:, :, 2]
    med_sat_ref = med_rgb.max(axis=-1) - med_rgb.min(axis=-1)
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sig_lu = np.clip(med_lu_ref - cv2.morphologyEx(med_lu_ref, cv2.MORPH_OPEN, k_bg), 0.0, 255.0)
    sig_sat = np.clip(med_sat_ref - cv2.morphologyEx(med_sat_ref, cv2.MORPH_OPEN, k_bg), 0.0, 255.0)
    sig_text = np.maximum(sig_lu, sig_sat * 1.2)

    tz_sig = sig_text[text_zone]
    if len(tz_sig) > 0 and float(tz_sig.max()) > 25.0:
        lo = max(14.0, float(np.percentile(tz_sig, 55)))
        hi = max(lo + 30.0, float(np.percentile(tz_sig, 96)))
        raw_alpha = np.zeros((H, W), dtype=np.float32)
        raw_alpha[text_zone] = np.clip((sig_text[text_zone] - lo) / (hi - lo), 0.0, 1.0)
        # Continuous sub-pixel Gaussian feathering: preserves exact typographic serifs & optical anti-aliasing
        static_alpha = cv2.GaussianBlur(raw_alpha, (3, 3), 0.60)
        static_alpha = np.where(static_alpha > 0.02, static_alpha, 0.0)
        static_alpha = np.clip(static_alpha, 0.0, 1.0)
    else:
        # Fallback to stroke dilation if temporal signal is flat
        k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        stroke_edge = cv2.dilate(credit_region.astype(np.uint8), k_edge, iterations=1).astype(bool)
        base_matte = np.where(stroke_edge, np.maximum(best_soft * (credit_region > 0), (credit_region > 0).astype(np.float32) * 0.90), 0.0)
        static_alpha = cv2.GaussianBlur(base_matte, (3, 3), 0.60)
        static_alpha = np.where(static_alpha > 0.05, static_alpha, 0.0)
        static_alpha = np.clip(static_alpha, 0.0, 1.0)

    base_credit_alpha = static_alpha

    SF_LOG.info(
        f"[ICDepth] On-screen credit detected ({int((static_alpha > 0.2).sum())} stroke pixels"
        f"{f', hint={hint_clean!r}' if has_hint else ''}, strength={strength:.2f}); elevating credit to foreground."
    )

    glyph_ref_mask = credit_region if credit_region.any() else stroke_edge
    # Compute reference Top-Hat across active frames for per-frame visibility gating
    ref_ths = []
    for idx in sample_idx:
        rgb_s = np.asarray(rgb_frames[idx], dtype=np.float32)
        if rgb_s.shape[:2] != (H, W):
            rgb_s = cv2.resize(rgb_s, (W, H), interpolation=cv2.INTER_AREA)
        lu_s = 0.299 * rgb_s[:, :, 0] + 0.587 * rgb_s[:, :, 1] + 0.114 * rgb_s[:, :, 2]
        th_s = cv2.morphologyEx(lu_s, cv2.MORPH_TOPHAT, k_sm)
        if has_hint:
            sat_s = rgb_s.max(axis=-1) - rgb_s.min(axis=-1)
            th_s = np.maximum(th_s, np.maximum(
                cv2.morphologyEx(lu_s, cv2.MORPH_TOPHAT, k_lg),
                cv2.morphologyEx(sat_s, cv2.MORPH_TOPHAT, k_lg),
            ))
        ref_ths.append(float(np.percentile(th_s[glyph_ref_mask], 75)))
    th_hi = max(45.0, float(np.percentile(ref_ths, 90)))

    knee = 0.72
    max_s = float(np.clip(scene_max_disp, knee + 0.05, 0.95))

    out = np.empty_like(depth_seq, dtype=np.float32)
    for t in range(T):
        rgb_t = np.asarray(rgb_frames[t], dtype=np.float32)
        if rgb_t.shape[:2] != (H, W):
            rgb_t = cv2.resize(rgb_t, (W, H), interpolation=cv2.INTER_AREA)
        lu_t = 0.299 * rgb_t[:, :, 0] + 0.587 * rgb_t[:, :, 1] + 0.114 * rgb_t[:, :, 2]
        th_t = cv2.morphologyEx(lu_t, cv2.MORPH_TOPHAT, k_sm)
        if has_hint:
            sat_t = rgb_t.max(axis=-1) - rgb_t.min(axis=-1)
            th_t = np.maximum(th_t, np.maximum(
                cv2.morphologyEx(lu_t, cv2.MORPH_TOPHAT, k_lg),
                cv2.morphologyEx(sat_t, cv2.MORPH_TOPHAT, k_lg),
            ))

        frame_vis = float(np.percentile(th_t[glyph_ref_mask], 75)) / th_hi
        frame_vis = float(np.clip((frame_vis - 0.28) / 0.42, 0.0, 1.0))

        d_raw = depth_seq[t].astype(np.float32)
        eff_strength = float(np.clip(strength, 0.0, 1.0))

        if frame_vis > 0.02:
            alpha_t = base_credit_alpha * frame_vis
            # Absolute foreground disparity: on-screen titles sit strictly in front of the live-action scene (0.94..1.00)
            d_credit = 0.94 + 0.06 * static_alpha
            # Cap the scene behind credits so nothing clips or occludes the foreground text plane
            d_scene = np.minimum(d_raw, 0.85)
            d_repaired = d_scene * (1.0 - alpha_t) + d_credit * alpha_t
        else:
            if has_hint:
                # Suppress Wan2.1 pre-appearance temporal attention ghosting before mid-clip titles appear
                d_clean_bg = cv2.morphologyEx(d_raw, cv2.MORPH_OPEN, k_lg)
                ghost_mask = np.clip(base_credit_alpha * 1.15, 0.0, 1.0)
                d_repaired = d_raw * (1.0 - ghost_mask) + np.minimum(d_raw, d_clean_bg) * ghost_mask
            else:
                d_repaired = d_raw

        out[t] = (1.0 - eff_strength) * d_raw + eff_strength * d_repaired

    return np.clip(out, 0.0, 1.0)


def compute_icdepth_size(height: int, width: int, max_side: int) -> Tuple[int, int]:
    """Calculate inference dimensions: maintain aspect ratio, cap longest side, align to multiples of 16."""
    scale = min(1.0, float(max_side) / max(height, width))
    new_h = max(16, int(round(height * scale / 16.0) * 16))
    new_w = max(16, int(round(width * scale / 16.0) * 16))
    return new_h, new_w


def run_icdepth_depth(
    video_path: str,
    max_res: int = 720,
    window_size: int = 81,
    overlap: int = 32,
    num_inference_steps: int = 5,
    cfg_scale: float = 5.0,
    seed: int = 42,
    tiled: bool = False,
    void_title_recovery: bool = True,
    volumetric_splash_recovery: bool = True,
    credit_pop: bool = True,
    credit_text_hint: str = "",
    progress=None,
) -> Tuple[np.ndarray, float]:
    """
    Run ICDepth inference on an input video.

    Args:
        video_path: Path to the input video.
        max_res: Longest side for inference (e.g. 720, 1024).
        window_size: Number of frames per temporal DiT window (default 81, clamped to 4k+1).
        overlap: Overlap frames between windows (default 32, clamped to 4k).
        num_inference_steps: FlowMatch diffusion steps (default 5).
        cfg_scale: Classifier-free guidance scale (default 5.0; 1.0 runs fast single-pass).
        seed: Random seed for latent noise.
        tiled: Use tiled VAE encode/decode to protect VRAM.
        void_title_recovery: Automatically detect pure-black void / 3D title shots and recover foreground & title relief.
        progress: Optional Gradio progress callback.

    Returns:
        depth_array: float32 ndarray of shape (T, orig_h, orig_w) with disparity in [0, 1].
        fps: Source video frame rate.
    """
    vp_check = Path(video_path)
    if not vp_check.is_file() or vp_check.suffix.lower() not in ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv'):
        stem = vp_check.stem
        if stem.endswith('.credits'):
            stem = stem[:-8]
        parent = vp_check.parent if vp_check.parent.exists() else Path("/app/source_videos")
        for ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv'):
            cand = parent / f"{stem}{ext}"
            if cand.is_file():
                video_path = str(cand)
                break

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read frames into PIL Images
    pil_frames: List[Image.Image] = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frames.append(Image.fromarray(frame_rgb))
    cap.release()

    n_frames = len(pil_frames)
    if n_frames == 0:
        raise ValueError(f"No valid frames read from {video_path}")

    # Compute inference resolution matching aspect ratio and aligned to 16
    proc_h, proc_w = compute_icdepth_size(orig_h, orig_w, max_res)
    SF_LOG.info(
        f"[ICDepth] Input: {n_frames} frames ({orig_w}x{orig_h}) -> Inference ({proc_w}x{proc_h}) | "
        f"Window: {window_size}, Overlap: {overlap}, Steps: {num_inference_steps}, CFG: {cfg_scale}"
    )

    if progress:
        progress(0.05, desc="ICDepth: Initializing model...")

    pipe = get_icdepth_pipeline()

    # Align window_size to 4k+1 and overlap to 4k (temporal compression factor = 4)
    ws_aligned = max(5, ((window_size - 1) // 4) * 4 + 1)
    ov_aligned = max(4, (overlap // 4) * 4)
    if ov_aligned >= ws_aligned:
        ov_aligned = max(4, ws_aligned - 4)

    # Resize input frames for inference
    resized_frames = [
        frame.resize((proc_w, proc_h), Image.Resampling.BILINEAR)
        if frame.size != (proc_w, proc_h)
        else frame
        for frame in pil_frames
    ]

    # Calculate exact sliding windows for progress tracking
    window_latent = (ws_aligned - 1) // 4
    overlap_latent = ov_aligned // 4
    stride_latent = max(1, window_latent - overlap_latent)
    num_latent_frames = (n_frames - 1) // 4 + 1
    if n_frames > ws_aligned:
        windows_list = []
        for start in range(0, num_latent_frames, stride_latent):
            end = min(start + window_latent, num_latent_frames)
            if start > 0 and end - start <= overlap_latent:
                break
            if end == num_latent_frames and (end - start) < window_latent and num_latent_frames >= window_latent:
                start = num_latent_frames - window_latent
            windows_list.append((start, end))
        total_windows = max(1, len(windows_list))
    else:
        total_windows = 1

    current_window_idx = [0]
    t_denoise_start = [time.time()]
    last_step_time = [time.time()]
    total_steps_all = total_windows * num_inference_steps

    def custom_progress_cmd(iterable):
        w_idx = min(current_window_idx[0], total_windows - 1)
        w_start_time = time.time()
        print(f"\n[ICDepth] ━━━ Window {w_idx + 1}/{total_windows} Starting ━━━", flush=True)

        for step, item in enumerate(iterable):
            current_step_global = w_idx * num_inference_steps + step + 1
            now = time.time()
            if w_idx == 0 and step == 0:
                t_denoise_start[0] = now
                last_step_time[0] = now

            elapsed = max(0.1, now - t_denoise_start[0])
            step_dur = max(0.1, now - last_step_time[0])
            last_step_time[0] = now

            steps_done = current_step_global
            steps_left = max(0, total_steps_all - steps_done)
            # Use average step duration for ETA calculation
            avg_step = elapsed / max(1, steps_done)
            eta_sec = int(steps_left * avg_step)
            if eta_sec >= 3600:
                eta_str = f"{eta_sec // 3600}h {(eta_sec % 3600) // 60}m"
            elif eta_sec >= 60:
                eta_str = f"{eta_sec // 60}m {eta_sec % 60}s"
            else:
                eta_str = f"{eta_sec}s"

            msg = (
                f"[ICDepth] Win {w_idx + 1}/{total_windows} | "
                f"Step {step + 1}/{num_inference_steps} "
                f"({current_step_global}/{total_steps_all}) | "
                f"Step: {step_dur:.1f}s | "
                f"Elapsed: {int(elapsed)}s | "
                f"ETA: {eta_str}"
            )
            SF_LOG.info(msg)
            print(msg, flush=True)

            if progress:
                try:
                    frac = min(
                        0.95,
                        0.1 + 0.85 * (current_step_global / max(1, total_steps_all))
                    )
                    progress(
                        frac,
                        desc=f"ICDepth: Win {w_idx + 1}/{total_windows} (Step {step + 1}/{num_inference_steps}) | ETA: {eta_str}"
                    )
                except Exception:
                    pass
            yield item
        current_window_idx[0] += 1

    t0 = time.time()
    with torch.no_grad():
        _, raw_disparity = pipe(
            video=resized_frames,
            height=proc_h,
            width=proc_w,
            num_frames=n_frames,
            seed=seed,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            tiled=tiled,
            window_size=ws_aligned if n_frames > ws_aligned else None,
            overlap=ov_aligned,
            progress_bar_cmd=custom_progress_cmd,
        )

    elapsed = max(1e-5, time.time() - t0)
    SF_LOG.info(f"[ICDepth] Denoising completed in {elapsed:.2f}s ({n_frames / elapsed:.2f} FPS)")

    # raw_disparity is (T, proc_h, proc_w) in [0, 1] float32
    # Apply directional-median inverse halftoning on the native VAE grid before resizing
    if progress:
        progress(0.95, desc="ICDepth: Descreening fine hair edges...")
    raw_disparity = descreen_icdepth_disparity(raw_disparity)

    # Resize back to native original video resolution (orig_h, orig_w) if needed
    # Use INTER_AREA when downscaling (e.g. 640->632) or INTER_LINEAR when upscaling to avoid bicubic Moiré/overshoot
    if raw_disparity.shape[1:] != (orig_h, orig_w):
        if progress:
            progress(0.97, desc=f"ICDepth: Resizing disparity to native {orig_w}x{orig_h}...")
        interp = cv2.INTER_AREA if (orig_w <= proc_w and orig_h <= proc_h) else cv2.INTER_LINEAR
        depth_array = np.stack([
            cv2.resize(raw_disparity[i], (orig_w, orig_h), interpolation=interp)
            for i in range(len(raw_disparity))
        ])
    else:
        depth_array = raw_disparity

    depth_array = np.clip(depth_array.astype(np.float32), 0.0, 1.0)

    # Save an untouched raw backup before any post-processing repair so the user can always pull back or revert
    try:
        vp = Path(video_path)
        depth_dir = vp.resolve().parent.parent / "depthmaps_videos"
        if depth_dir.exists():
            raw_backup_path = depth_dir / f"{vp.stem}_ICD_depth.raw.npz"
            np.savez_compressed(str(raw_backup_path), depth=depth_array.astype(np.float16))
            SF_LOG.info(f"[ICDepth] Saved untouched raw backup: {raw_backup_path}")
    except Exception as e:
        SF_LOG.warning(f"[ICDepth] Could not save raw backup: {e}")

    if void_title_recovery:
        if progress:
            progress(0.98, desc="ICDepth: Checking Black-Void / Title-Card 3D recovery...")
        depth_array = recover_black_void_and_titles(depth_array, pil_frames)

    if volumetric_splash_recovery:
        if progress:
            progress(0.985, desc="ICDepth: Checking Volumetric Splash & Water Aeration...")
        depth_array = recover_volumetric_splash_aeration(depth_array, pil_frames)

    effective_hint = str(credit_text_hint or "").strip()
    if effective_hint:
        save_clip_credit_sidecar(video_path, effective_hint)
    else:
        effective_hint = load_clip_credit_sidecar(video_path)

    if credit_pop:
        if progress:
            progress(0.99, desc="ICDepth: Checking On-Screen Credit Foreground Pop...")
        depth_array = detect_and_elevate_credits(
            depth_array,
            pil_frames,
            credit_text_hint=effective_hint,
        )

    # Check for {stem}.layers.json sidecar and apply object depth layers
    try:
        from m2svid.object_layer_depth import (
            load_layers_sidecar,
            segment_object_keyframe,
            track_object_across_video,
            apply_object_layer_to_depth_seq,
        )
        layers_data = load_layers_sidecar(video_path)
        if layers_data and "objects" in layers_data and len(layers_data["objects"]) > 0:
            if progress:
                progress(0.995, desc=f"ICDepth: Applying {len(layers_data['objects'])} Object Depth Layers...")
            bgr_frames = [
                cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR) if np.array(f).shape[-1] == 3 else np.array(f)
                for f in pil_frames
            ]
            apply_skyline = bool(layers_data.get("skyline_split", False))
            for obj in layers_data["objects"]:
                t_d = float(obj.get("target_depth", 0.20))
                pts = [tuple(p) for p in obj.get("points", [])]
                txt = str(obj.get("text_prompt", "") or "")
                ref_idx = min(len(bgr_frames) - 1, max(0, int(obj.get("ref_frame", len(bgr_frames) // 2))))
                key_mask = segment_object_keyframe(bgr_frames[ref_idx], point_clicks=pts, text_prompt=txt)
                if np.any(key_mask):
                    if obj.get("tophat_snap", True):
                        from m2svid.object_layer_depth import snap_mask_with_tophat_and_volume
                        key_mask, _ = snap_mask_with_tophat_and_volume(bgr_frames[ref_idx], key_mask)
                    obj_masks = track_object_across_video(bgr_frames, key_mask, ref_idx)
                    depth_array = apply_object_layer_to_depth_seq(
                        depth_array, obj_masks, target_depth=t_d,
                        apply_skyline_split=apply_skyline, rgb_frames=bgr_frames
                    )
                    SF_LOG.info(f"[ICDepth] Applied object layer '{obj.get('name', 'object')}' at target depth {t_d:.2f}")
    except Exception as e:
        SF_LOG.warning(f"[ICDepth] Could not apply object layer sidecar: {e}")

    if progress:
        progress(1.0, desc="ICDepth: Complete!")

    return depth_array, fps
