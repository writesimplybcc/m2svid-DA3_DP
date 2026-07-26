"""
StereoFaster WebUI - Gradio Interface
Modern 2-step DA3 + M2SVid pipeline for monocular-to-stereo video conversion.

Based on the conceptual structure of the StereoCrafter combined webui,
but simplified to exactly two primary heavy processes as designed for StereoFaster:

1. DA3 Depth Estimation (monocular / optionally metric or streaming)
2. M2SVid stage (lightweight geometric warping + single 1-step conditioned SVD)

Launch:
    PYTHONPATH="Depth-Anything-3/src:.:${PYTHONPATH}" python webui.py --server-port 7878

Requirements:
- DA3 installed or src in path
- M2SVid dependencies + third_party/Hi3D-Official etc. in path (as for normal inference)
- M2SVid weights in ckpts/
"""

import os
import sys

# Provide immediate feedback to the console before heavy imports lock up the thread
print("===============================================================")
print("🚀 Starting StereoFaster WebUI Initialization...")
print("⏳ Please wait while heavy ML libraries (PyTorch, Gradio) are loaded into memory.")
print("   This may take up to a minute depending on your disk speed.")
print("===============================================================")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging
import os
import sys
import tempfile
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import torch
import numpy as np
import cv2
import ffmpeg

# Ensure project paths (match what inference scripts use)
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "m2svid"))          # for m2svid.m2svid.*
sys.path.insert(0, str(PROJECT_ROOT / "m2svid" / "m2svid"))  # extra safety
sys.path.insert(0, str(PROJECT_ROOT / "Depth-Anything-3" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "m2svid" / "third_party" / "Hi3D-Official"))
sys.path.insert(0, str(PROJECT_ROOT / "m2svid" / "third_party" / "pytorch-msssim"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- Logging setup ----
log_level = logging.DEBUG if os.getenv("STEREOFASTER_DEBUG") == "1" else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
SF_LOG = logging.getLogger("stereofaster")

# Batch folders for automatic processing
SOURCE_DIR = PROJECT_ROOT / "source_videos"
DEPTH_DIR = PROJECT_ROOT / "depthmaps_videos"
FINAL_DIR = PROJECT_ROOT / "final_videos"
for _d in (SOURCE_DIR, DEPTH_DIR, FINAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


ASPECT_RATIO_PRESETS = {
    "16:9 (None)": 1.7777,
    "1.85:1 (Letterbox)": 1.85,
    "2.00:1 (Letterbox)": 2.00,
    "2.20:1 (Letterbox)": 2.20,
    "2.35:1 (Letterbox)": 2.35,
    "2.39:1 (Letterbox)": 2.39,
    "2.76:1 (Letterbox)": 2.76,
    "4:3 (Pillarbox)": 1.3333,
    "1.37:1 (Pillarbox)": 1.37,
    "1.43:1 (IMAX Pillarbox)": 1.43,
    "1.66:1 (Pillarbox)": 1.66,
    "1:1 (Square Pillarbox)": 1.00,
    "9:16 (Vertical Pillarbox)": 0.5625
}

def analyze_letterbox(video_path: str) -> tuple[str, str, str]:
    """Analyzes the first frame of a video to detect black bars, matching it to the closest preset."""
    if not video_path or not os.path.exists(video_path):
        return "16:9 (None)", "", ""
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return "16:9 (None)", "", ""
        
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "16:9 (None)", f"Full: {w}x{h}", video_path
        
    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    
    if bw == 0 or bh == 0 or (bw >= w - 4 and bh >= h - 4):
        return "16:9 (None)", f"Full: {w}x{h}", video_path
        
    detected_ratio = bw / bh
    best_preset = "16:9 (None)"
    min_diff = 999.0
    for preset_name, ratio in ASPECT_RATIO_PRESETS.items():
        diff = abs(detected_ratio - ratio)
        if diff < min_diff:
            min_diff = diff
            best_preset = preset_name
            
    if best_preset == "16:9 (None)":
        return best_preset, f"Full: {w}x{h}", video_path
        
    # Calculate crop
    target_ratio = ASPECT_RATIO_PRESETS[best_preset]
    if target_ratio > (w/h): # Letterbox
        new_h = int(w / target_ratio)
        new_h = new_h - (new_h % 8)
        new_w = w
    else: # Pillarbox
        new_w = int(h * target_ratio)
        new_w = new_w - (new_w % 8)
        new_h = h
        
    res_str = f"{new_w}x{new_h}"
    
    # Generate 1-second preview
    preview_path = str(PROJECT_ROOT / "scratch" / f"crop_preview_{Path(video_path).stem}.mp4")
    crop_filter = f"crop={new_w}:{new_h}:(in_w-{new_w})/2:(in_h-{new_h})/2"
    import subprocess
    cmd = ["ffmpeg", "-y", "-i", video_path, "-t", "1", "-vf", crop_filter, "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast", preview_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(preview_path):
        return best_preset, res_str, preview_path
    return best_preset, res_str, video_path

def execute_crop(video_path: str, preset: str, progress=gr.Progress(track_tqdm=True)) -> str:
    if not video_path or not os.path.exists(video_path) or preset == "16:9 (None)":
        return video_path
        
    import subprocess
    target_ratio = ASPECT_RATIO_PRESETS.get(preset, 1.7777)
    if target_ratio == 1.7777: return video_path
    
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if target_ratio > (w/h):
        new_w = w
        new_h = int(w / target_ratio)
        new_h = new_h - (new_h % 8)
    else:
        new_h = h
        new_w = int(h * target_ratio)
        new_w = new_w - (new_w % 8)
        
    out_path = str(SOURCE_DIR / f"{Path(video_path).stem}_cropped.mp4")
    crop_filter = f"crop={new_w}:{new_h}:(in_w-{new_w})/2:(in_h-{new_h})/2"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", crop_filter, "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-c:a", "copy", out_path]
    
    if progress:
        progress(0, desc=f"Cropping video to {new_w}x{new_h}...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def batch_auto_crop_all(progress=gr.Progress(track_tqdm=True)):
    vids = [p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')]
    vids = [v for v in vids if not v.stem.endswith("_cropped")]
    if not vids:
        return gr.update(), gr.update()
        
    for i, vp in enumerate(vids):
        progress(float(i)/len(vids), desc=f"Auto-Cropping {vp.name}")
        preset, _, _ = analyze_letterbox(str(vp))
        if preset != "16:9 (None)":
            execute_crop(str(vp), preset, progress=None)
            
    progress(1.0, desc="Batch Crop Complete!")
    src = get_source_video_list()
    return gr.update(choices=[""] + src), gr.update(choices=[""] + src)




def convert_depth_video_to_npz(video_path: str, out_npz_path: str):
    """Extract frames from a grayscale depth video and save as compatible .npz file."""
    SF_LOG.info(f"Converting depth video {video_path} to M2SVid compatible .npz")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open depth video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    depth = np.stack(frames, axis=0).astype(np.float32)
    save_m2svid_compatible_npz(depth, out_npz_path)


def run_depth_pro_depth(
    frames: list[np.ndarray],
    device: str = "cuda",
    progress=None,
) -> np.ndarray:
    """Run Depth Pro inference on a list of RGB frames."""
    import depth_pro
    from PIL import Image
    
    total_frames = len(frames)
    SF_LOG.info(f"Loading Apple Depth Pro model... processing {total_frames} frames")
    device_obj = torch.device(device)
    precision = torch.half if device == "cuda" else torch.float32
    model, transform = depth_pro.create_model_and_transforms(
        device=device_obj,
        precision=precision,
    )
    model.eval()
    
    depths = []
    for i in range(total_frames):
        frame = frames[i]
        pil_img = Image.fromarray(frame)
        x = transform(pil_img)
        with torch.inference_mode():
            pred = model.infer(x, f_px=None)
        depth_val = pred["depth"].detach().cpu().numpy().squeeze()
        depths.append(depth_val)
        
        if progress is not None:
            try:
                progress((i + 1) / total_frames, desc=f"DepthPro processing frame {i+1}/{total_frames}")
            except Exception:
                pass
                
        # Console logging every 10 frames or the last frame
        if (i + 1) % 10 == 0 or i == total_frames - 1:
            SF_LOG.info(f"DepthPro processing frame {i+1}/{total_frames}...")
        
    depth = np.stack(depths, axis=0)
    return depth

def run_depth_on_source_videos(progress=None, model_name=None, process_res=720, batch_size=4):
    """Process all videos in SOURCE_DIR and save depth .npz and depth_depth.mp4 in DEPTH_DIR if missing."""
    SF_LOG.info(f"Starting batch depth processing with model {model_name} on source_videos directory")
    global DEFAULT_DA3_MODEL
    model_name = model_name or DEFAULT_DA3_MODEL
    suffix = get_model_suffix(model_name)
    
    vids = [p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')]
    total = len(vids)
    SF_LOG.info(f"Found {total} video(s) in source_videos")
    
    for i, vp in enumerate(vids):
        stem = vp.stem
        out_npz = DEPTH_DIR / f"{stem}{suffix}_depth.npz"
        out_mp4 = DEPTH_DIR / f"{stem}{suffix}_depth.mp4"
        
        if out_npz.exists() and out_mp4.exists():
            SF_LOG.info(f"Skipping {stem}: depth already exists at {out_npz} and {out_mp4}")
            if progress:
                try: progress(float(i)/max(1,total), desc=f"Skipping {stem}, depth exists")
                except Exception: pass
            continue
            
        SF_LOG.info(f"Depth processing [{i+1}/{total}]: {stem} using model {model_name}")
        try:
            frames, fps, (h, w) = load_video_frames(str(vp), target_fps=0.0)
            SF_LOG.debug(f"Loaded {len(frames)} frames @ {fps}fps, size {w}x{h}")
            
            is_depth_pro = "DepthPro" in model_name or "depth-pro" in model_name
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if is_depth_pro:
                depth = run_depth_pro_depth(frames, device=device, progress=progress)
                inv_depth = 1.0 / np.clip(depth, 1e-4, 1e5)
                inv_min, inv_max = inv_depth.min(), inv_depth.max()
                if inv_max - inv_min > 1e-6:
                    depth = (inv_depth - inv_min) / (inv_max - inv_min)
                else:
                    depth = np.zeros_like(inv_depth)
            else:
                depth = run_da3_depth(frames, model_name=model_name, process_res=process_res, device=device, batch_size=batch_size)
                depth = -depth
                
            if depth.shape[1:] != (h, w):
                depth = np.stack([cv2.resize(d, (w, h), cv2.INTER_CUBIC) for d in depth])
                
            save_m2svid_compatible_npz(depth, str(out_npz))
            _create_depth_preview_video(depth, str(out_mp4), fps)
            SF_LOG.info(f"Saved depth files: {out_npz} and {out_mp4}")
            
            if progress:
                try: progress(float(i+1)/max(1,total), desc=f"Processed {stem}")
                except Exception: pass
        except Exception as e:
            SF_LOG.error(f"Depth error on {stem}: {e}")
            if progress:
                try: progress(float(i+1)/max(1,total), desc=f"Error {stem}")
                except Exception: pass
            import traceback
            traceback.print_exc()
    SF_LOG.info("Batch depth processing complete")


def run_m2svid_on_pairs(m2svid_config, m2svid_ckpt, disparity_perc, closing_kernel, mask_antialias, warping_batch_size, gen_chunk_size, m2svid_process_res, progress=gr.Progress(track_tqdm=True)):
    """Process pairs in SOURCE_DIR and DEPTH_DIR and save outputs into FINAL_DIR."""
    SF_LOG.info("Starting batch M2SVid processing on source_videos + depthmaps_videos")
    cfg = m2svid_config or DEFAULT_M2SVID_CONFIG
    ckpt = m2svid_ckpt or DEFAULT_M2SVID_CKPT
    vids = [p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')]
    total = len(vids)
    SF_LOG.info(f"Found {total} video(s) to process")
    for i, vp in enumerate(vids):
        stem = vp.stem
        depth_npz = None

        legacy_npz = DEPTH_DIR / f"{stem}_depth.npz"
        if legacy_npz.exists():
            depth_npz = legacy_npz
        else:
            candidates = sorted([p for p in DEPTH_DIR.iterdir() if p.is_file() and p.suffix == ".npz" and p.stem.startswith(stem + "_") and "_depth" in p.stem])
            if candidates:
                depth_npz = candidates[0]

        if depth_npz is None:
            SF_LOG.warning(f"M2SVid skipping {stem}: missing depth npz")
            if progress:
                try: progress(float(i)/max(1,total), desc=f"M2SVid: missing depth for {stem}")
                except Exception: pass
            continue

        final_out = FINAL_DIR / f"{stem}_generated_right.mp4"
        if final_out.exists():
            SF_LOG.info(f"Skipping {stem}: already processed at {final_out}")
            if progress:
                try: progress(float(i)/max(1,total), desc=f"M2SVid: skipping {stem}, already done")
                except Exception: pass
            continue
        SF_LOG.info(f"M2SVid processing [{i+1}/{total}]: {stem}")
        try:
            prev_input = STATE.get("input_video")
            STATE["input_video"] = str(vp)
            status, gen_right, sbs, anaglyph, out_dir = step2_run_m2svid(str(depth_npz), disparity_perc, closing_kernel, mask_antialias, cfg, ckpt, input_video_path=str(vp), warping_batch_size=warping_batch_size, gen_chunk_size=gen_chunk_size, m2svid_process_res=m2svid_process_res, progress=progress)
            if out_dir and os.path.exists(out_dir):
                try:
                    src = Path(out_dir) / "generated_right.mp4"
                    if src.exists():
                        dst = FINAL_DIR / f"{stem}_generated_right.mp4"
                        shutil.move(str(src), str(dst))
                        SF_LOG.info(f"Moved generated_right to {dst}")
                    src = Path(out_dir) / "stereo_sbs.mp4"
                    if src.exists():
                        shutil.move(str(src), str(FINAL_DIR / f"{stem}_stereo_sbs.mp4"))
                    src = Path(out_dir) / "anaglyph.mp4"
                    if src.exists():
                        shutil.move(str(src), str(FINAL_DIR / f"{stem}_anaglyph.mp4"))
                except Exception as e:
                    SF_LOG.error(f"Error moving outputs for {stem}: {e}")
                    import traceback
                    traceback.print_exc()
            STATE["input_video"] = prev_input
            if progress:
                try: progress(float(i+1)/max(1,total), desc=f"M2SVid: processed {stem}")
                except Exception: pass
        except Exception as e:
            SF_LOG.error(f"M2SVid error on {stem}: {e}")
            import traceback
            traceback.print_exc()
            STATE["input_video"] = prev_input
    SF_LOG.info("Batch M2SVid processing complete")


def _background_batch_worker():
    """Run batch Depth+M2SVid processing on source_videos folder. Intended for background/threaded use."""
    SF_LOG.info("Starting background batch worker")
    try:
        run_depth_on_source_videos(progress=None)
        run_m2svid_on_pairs()
        SF_LOG.info("Background batch worker completed")
    except Exception as e:
        SF_LOG.error(f"Background batch worker failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        SF_LOG.error(f"Background batch worker failed: {e}")
        import traceback
        traceback.print_exc()



# Load .env file if present (supports HF_TOKEN, HF_HOME, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
except ImportError:
    # Fallback: very lightweight manual loader for HF_TOKEN only
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import random

# Note: We deliberately avoid importing from depth_anything_3.api at module level
# to prevent pulling in 3DGS / gsplat code paths (which are not used in this StereoFaster webui).

# M2SVid internals (reuse existing logic)
# Robust import: try normal package import first, then direct file load as fallback
def _load_prepare_da3_module():
    try:
        import m2svid.m2svid.prepare_da3_depth as mod
        return mod
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "prepare_da3_depth",
            PROJECT_ROOT / "m2svid" / "m2svid" / "prepare_da3_depth.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

_prepare_mod = _load_prepare_da3_module()
load_video_frames = _prepare_mod.load_video_frames
run_da3_depth = _prepare_mod.run_da3_depth
save_m2svid_compatible_npz = _prepare_mod.save_m2svid_compatible_npz

# Note: M2SVid / sgm heavy imports are intentionally lazy-loaded inside step2_run_m2svid
# to avoid pulling in kornia, sgm, and related code until the user actually runs Step 2.
# This saves startup time and VRAM (consistent with the DA3 lazy loading).

def _load_warping_module():
    """Robustly load the standalone m2svid/warping.py (not the nested package version)."""
    SF_LOG.info("Loading warping module...")
    import importlib.util
    warping_path = PROJECT_ROOT / "m2svid" / "warping.py"
    spec = importlib.util.spec_from_file_location("m2svid_warping", warping_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    SF_LOG.info("Warping module loaded successfully")
    return mod

# --- Configuration ---
# Recommended defaults for StereoFaster (DA3 + M2SVid) use case:
# - DA3NESTED-GIANT-LARGE-1.1 : Best overall quality (official preferred -1.1 retrained version)
# - DA3MONO-LARGE             : Excellent pure relative monocular depth (often best for warping accuracy)
DEFAULT_DA3_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
def _is_fast_gpu(gpu_name: str) -> bool:
    n = gpu_name.lower()
    markers = (
        "rtx 4080", "rtx 4090", "rtx 5090",
        "rtx 6000 ada", "rtx 5000 ada",
        "a100", "a40", "h100", "l40", "l4", "a10", "a6000",
    )
    return any(m in n for m in markers)


def get_vram_defaults():
    """Returns dynamic batch sizes based on VRAM capacity (targeting 12GB, 24GB, 32GB, 48GB, 96GB)."""
    if not torch.cuda.is_available():
        return {"da3": 2, "warp": 1, "vae": 2, "gen_chunk": 2}
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if vram_gb >= 90: # 96GB class (e.g., A100 96GB/Mac 128GB)
        return {"da3": 32, "warp": 16, "vae": 35, "gen_chunk": 35}
    if vram_gb >= 45: # 48GB class (e.g., RTX 6000 Ada / A6000)
        return {"da3": 16, "warp": 8, "vae": 28, "gen_chunk": 28}
    if vram_gb >= 30: # 32GB class (e.g., V100 32GB)
        return {"da3": 12, "warp": 6, "vae": 20, "gen_chunk": 21}
    if vram_gb >= 22: # 24GB class (e.g., RTX 3090 / 4090)
        return {"da3": 8, "warp": 4, "vae": 14, "gen_chunk": 14}
    if vram_gb >= 11: # 12GB class (e.g., RTX 3060 / 4070)
        return {"da3": 4, "warp": 2, "vae": 4, "gen_chunk": 5}
        
    # Fallback for <12GB (e.g., 8GB cards)
    return {"da3": 2, "warp": 1, "vae": 2, "gen_chunk": 3}

_VRAM_DEFAULTS = get_vram_defaults()
_FAST_GPU = False
if torch.cuda.is_available():
    _gpu_name = torch.cuda.get_device_name(0)
    _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if _vram_gb >= 18 or (_vram_gb >= 14 and _is_fast_gpu(_gpu_name)):
        _FAST_GPU = True
    SF_LOG.info(
        f"GPU detected: {_gpu_name} ({_vram_gb:.1f} GB) -> "
        f"{'high-quality m2svid.yaml' if _FAST_GPU else 'fast m2svid_no_fullatten.yaml'} "
        f"| VRAM Profile: DA3={_VRAM_DEFAULTS['da3']}, Warp={_VRAM_DEFAULTS['warp']}, VAE={_VRAM_DEFAULTS['vae']}"
    )

DEFAULT_M2SVID_CONFIG = str(
    PROJECT_ROOT / "m2svid" / "configs" / ("m2svid.yaml" if _FAST_GPU else "m2svid_no_fullatten.yaml")
)
DEFAULT_M2SVID_CKPT = str(PROJECT_ROOT / "m2svid" / "ckpts" / "m2svid_weights.pt")

# In-memory state between steps (simple for single-user local UI)
STATE = {
    "input_video": None,
    "depth_npz": None,
    "da3_depth": None,          # numpy array (T,H,W)
    "original_fps": 24.0,
    "original_hw": (720, 1280),
}


def get_gpu_info() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return f"GPU: {name} | VRAM: {reserved:.1f} / {total:.1f} GB"
    return "GPU: CPU only"


def get_source_video_list():
    """Return list of video stems available in source_videos folder."""
    if not SOURCE_DIR.exists():
        return []
    vids = sorted([p.stem for p in SOURCE_DIR.iterdir() 
                   if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')])
    return vids


MODEL_SUFFIX_MAP = {
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1": "_NGL",
    "depth-anything/DA3MONO-LARGE": "_ML",
    "depth-anything/DA3-GIANT-1.1": "_G",
    "depth-anything/DA3-LARGE-1.1": "_L",
    "depth-anything/DA3METRIC-LARGE": "_ML",
    "apple/DepthPro": "_DP",
}


def get_model_suffix(model_name: str) -> str:
    """Return the suffix for a given depth model identifier."""
    return MODEL_SUFFIX_MAP.get(model_name, "_depth")


def get_model_depth_paths(stem: str, include_all: bool = False):
    """Return depth paths for a source stem.
    
    If include_all is False, returns the legacy _depth.npz path (backward compat).
    If include_all is True, returns a list of (suffix, npz_path, mp4_path) tuples
    for all model-tagged depth files matching the stem.
    """
    if include_all:
        results = []
        if DEPTH_DIR.exists():
            for p in DEPTH_DIR.iterdir():
                if not p.is_file():
                    continue
                name = p.stem
                if name.startswith(stem + "_") and "_depth" in name:
                    suffix = name[len(stem):]
                    npz = DEPTH_DIR / f"{stem}{suffix}.npz"
                    mp4 = DEPTH_DIR / f"{stem}{suffix}.mp4"
                    if npz.exists() or mp4.exists():
                        results.append((suffix, str(npz) if npz.exists() else None, str(mp4) if mp4.exists() else None))
        return results
    else:
        npz_path = DEPTH_DIR / f"{stem}_depth.npz"
        mp4_path = DEPTH_DIR / f"{stem}_depth.mp4"
        return [("", str(npz_path) if npz_path.exists() else None, str(mp4_path) if mp4_path.exists() else None)]


def select_source_video(stem):
    """Load video and depth paths for a stem selected from dropdown."""
    if stem is None or (isinstance(stem, dict) and not stem.get("path")):
        return None, None
    if isinstance(stem, dict):
        stem = stem.get("orig_name", "") or ""
    if not stem:
        return None, None
    video_path = SOURCE_DIR / f"{stem}.mp4"
    if not video_path.exists():
        for ext in ('.mov', '.avi', '.mkv'):
            alt = SOURCE_DIR / f"{stem}{ext}"
            if alt.exists():
                video_path = alt
                break
    
    depth_npz = None
    legacy_npz = DEPTH_DIR / f"{stem}_depth.npz"
    if legacy_npz.exists():
        depth_npz = legacy_npz
    else:
        candidates = sorted([p for p in DEPTH_DIR.iterdir() if p.is_file() and p.suffix == ".npz" and p.stem.startswith(stem + "_") and "_depth" in p.stem])
        if candidates:
            depth_npz = candidates[0]
    
    return str(video_path) if video_path.exists() else None, str(depth_npz) if depth_npz else None


def clear_cuda():
    if torch.cuda.is_available():
        SF_LOG.debug("Clearing CUDA cache")
        torch.cuda.empty_cache()
    import gc
    gc.collect()


# =============================================================================
# STEP 1: DA3 Depth Estimation
# =============================================================================

def step1_run_da3_depth(
    input_video: str,
    model_name: str,
    process_res: int,
    batch_size: int,
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[str, str, str, Optional[str]]:
    """
    Run depth estimation (DA3 or Depth Pro) on the uploaded video.
    Produces both a depth.npz (for M2SVid warping) and a visual depth.mp4 video.
    Returns: (status, depth_preview_video, depth_npz_path, depth_state_for_step2)
    """
    if input_video is None or input_video == "":
        return "Please upload or select a video first.", None, None, None
        
    # If the input is just a stem (from dropdown), convert it to an absolute path
    if isinstance(input_video, str) and not os.path.isabs(input_video) and not input_video.startswith("/workspace") and not ":" in input_video:
        possible = list(SOURCE_DIR.glob(f"{input_video}.*"))
        if possible:
            input_video = str(possible[0])
        else:
            return f"Video {input_video} not found in source_videos.", None, None, None
    
    is_depth_pro = "DepthPro" in model_name or "depth-pro" in model_name
    model_type_str = "Depth Pro" if is_depth_pro else "DA3"
    model_suffix = get_model_suffix(model_name)

    SF_LOG.info(f"Starting {model_type_str} depth estimation")
    progress(0, desc=f"Preparing {model_type_str} depth estimation...")

    video_path = input_video
    if isinstance(input_video, dict) and "name" in input_video:
        src_path = input_video["name"]
        dst_path = SOURCE_DIR / Path(src_path).name
        shutil.copy(src_path, str(dst_path))
        video_path = str(dst_path)
        SF_LOG.info(f"Uploaded video copied to {dst_path}")

    STATE["input_video"] = video_path
    SF_LOG.debug(f"Video path set to: {video_path}")

    try:
        frames, fps, (h, w) = load_video_frames(video_path, target_fps=0.0)
        STATE["original_fps"] = fps
        STATE["original_hw"] = (h, w)
        SF_LOG.info(f"Loaded {len(frames)} frames @ {fps:.1f}fps, resolution {w}x{h}")

        progress(0.1, desc=f"Loaded {len(frames)} frames @ {fps:.1f} fps")

        SF_LOG.info(f"Running {model_type_str} inference...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if is_depth_pro:
            depth = run_depth_pro_depth(frames, device=device)
            # Depth Pro metric depth -> inverse depth (disparity) -> normalized to [0, 1]
            inv_depth = 1.0 / np.clip(depth, 1e-4, 1e5)
            inv_min, inv_max = inv_depth.min(), inv_depth.max()
            if inv_max - inv_min > 1e-6:
                depth = (inv_depth - inv_min) / (inv_max - inv_min)
            else:
                depth = np.zeros_like(inv_depth)
        else:
            depth = run_da3_depth(
                frames,
                model_name=model_name,
                process_res=process_res,
                device=device,
                batch_size=batch_size,
            )
            depth = -depth

        if depth.shape[1:] != (h, w):
            depth = np.stack([cv2.resize(d, (w, h), cv2.INTER_CUBIC) for d in depth])

        STATE["da3_depth"] = depth

        # Save compatible npz and visual mp4 to depthmaps_videos with model-tagged suffix
        stem = Path(video_path).stem
        depth_npz = DEPTH_DIR / f"{stem}{model_suffix}_depth.npz"
        depth_mp4 = DEPTH_DIR / f"{stem}{model_suffix}_depth.mp4"
        
        save_m2svid_compatible_npz(depth, str(depth_npz))
        _create_depth_preview_video(depth, str(depth_mp4), fps)
        
        STATE["depth_npz"] = str(depth_npz)
        SF_LOG.info(f"Saved depth files: {depth_npz} and {depth_mp4}")

        # Also save to temp dir for immediate Step 2 use
        out_dir = Path(tempfile.mkdtemp(prefix="stereofaster_depth_"))
        temp_depth_npz = out_dir / "depth.npz"
        save_m2svid_compatible_npz(depth, str(temp_depth_npz))

        depth_vis_path = out_dir / "depth_preview.mp4"
        _create_depth_preview_video(depth, str(depth_vis_path), fps)

        progress(1.0, desc="Depth estimation complete")
        clear_cuda()

        status = f"✅ {model_type_str} depth computed ({depth.shape[0]} frames). Ready for Step 2."
        return status, str(depth_vis_path), str(depth_npz), str(depth_npz)

    except Exception as e:
        clear_cuda()
        import traceback
        traceback.print_exc()
        return f"❌ Error in depth step: {str(e)}", None, None, None


def _create_depth_preview_video(depth: np.ndarray, out_path: str, fps: float):
    """Create a contrast-enhanced grayscale video showing depth for preview.
    
    Uses per-frame 1st/99th percentile stretching so that both near and far
    objects are clearly visible, even when the depth range is highly skewed
    (e.g. Depth Pro inverse-depth where most values cluster near zero).
    """
    if depth.size == 0:
        raise RuntimeError("Cannot create preview video: depth array is empty")

    n, h, w = depth.shape
    if n == 0:
        raise RuntimeError("Cannot create preview video: no frames in depth array")

    if w <= 0 or h <= 0:
        raise RuntimeError(f"Cannot create preview video: invalid frame size {w}x{h}")

    SF_LOG.info(f"Creating depth preview video: {n} frames @ {w}x{h}, {fps} fps -> {out_path}")

    fourcc_candidates = [
        cv2.VideoWriter_fourcc(*"avc1"), # H.264 (Most browser compatible)
        cv2.VideoWriter_fourcc(*"X264"),
        cv2.VideoWriter_fourcc(*"H264"),
        cv2.VideoWriter_fourcc(*"mp4v"), # Fallback (Will trigger Gradio warning)
    ]

    writer = None
    used_fourcc = None
    for fourcc in fourcc_candidates:
        trial_path = out_path if writer is None else out_path + ".tmp"
        writer = cv2.VideoWriter(trial_path, fourcc, fps, (w, h), isColor=False)
        if writer.isOpened():
            used_fourcc = fourcc
            if trial_path != out_path:
                try:
                    os.replace(trial_path, out_path)
                except OSError:
                    pass
            break
        writer.release()
        writer = None

    if writer is None:
        raise RuntimeError(
            f"Failed to open VideoWriter for {out_path} with any codec {fourcc_candidates}. "
            "Check that ffmpeg is installed and a H.264/MP4 codec is available."
        )

    written = 0
    last_error = None
    try:
        for idx, d in enumerate(depth):
            p_lo = np.percentile(d, 1)
            p_hi = np.percentile(d, 99)
            if p_hi - p_lo < 1e-6:
                p_lo = d.min()
                p_hi = d.max()
            if p_hi - p_lo < 1e-6:
                p_hi = p_lo + 1
            norm = np.clip((d - p_lo) / (p_hi - p_lo), 0, 1)
            frame = (norm * 255).astype(np.uint8)

            if frame.shape != (h, w):
                writer.release()
                raise RuntimeError(
                    f"Frame shape mismatch at frame {idx}: expected ({h},{w}), got {frame.shape}"
                )

            writer.write(frame)
            written += 1
    except Exception as e:
        writer.release()
        raise RuntimeError(f"Error writing depth video at frame {written}: {e}") from e

    writer.release()

    written_bytes = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    SF_LOG.info(f"Depth preview video written: {written}/{n} frames, {written_bytes} bytes")

    if written < n:
        # Partial write; the video may not play correctly
        os.remove(out_path)
        raise RuntimeError(
            f"Depth preview video write aborted: only {written}/{n} frames written. "
            f"Removed incomplete file: {out_path}"
        )

    if written_bytes < 1024:
        os.remove(out_path)
        raise RuntimeError(
            f"Depth preview video is suspiciously small ({written_bytes} bytes). "
            f"Removed file: {out_path}"
        )


# =============================================================================
# STEP 2: M2SVid Warping + 1-step Generation (the second heavy process)
# =============================================================================

_m2svid_model = None  # singleton


def _load_m2svid_model(config_path: str, ckpt_path: str):
    global _m2svid_model
    if _m2svid_model is not None:
        SF_LOG.debug("M2SVid model already loaded, reusing cached model")
        return _m2svid_model

    # Lazy import for sgm (only when actually loading the model in Step 2)
    from sgm.util import instantiate_from_config

    SF_LOG.info("[StereoFaster] Loading M2SVid model (this is the second heavy process)...")
    SF_LOG.debug(f"Config path: {config_path}")
    SF_LOG.debug(f"Checkpoint path: {ckpt_path}")
    config = OmegaConf.load(config_path)
    
    # Dynamically inject VRAM-aware parameters
    if hasattr(config.model, 'params'):
        config.model.params.en_and_decode_n_samples_a_time = _VRAM_DEFAULTS['vae']
        SF_LOG.info(f"Dynamically set VAE decode chunk size to {_VRAM_DEFAULTS['vae']}")
        
    model = instantiate_from_config(config.model).cpu()
    model.init_from_ckpt(ckpt_path)
    model = model.cuda().half().eval()
    _m2svid_model = model
    SF_LOG.info("M2SVid model loaded successfully")
    return model


def step2_run_m2svid(
    depth_npz_path: str,
    disparity_perc: float,
    reprojected_closing_kernel: int,
    mask_antialias: bool,
    m2svid_config: str,
    m2svid_ckpt: str,
    input_video_path: Optional[str] = None,
    warping_batch_size: int = 2,
    gen_chunk_size: int = 14,
    m2svid_process_res: str = "Native",
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[str, str, str, str, str]:
    """
    Performs the M2SVid stage:
    - Geometric warping using the DA3 depth
    - 1-step conditioned generation
    Returns paths to: generated_right, sbs, anaglyph, and status
    """
    SF_LOG.info("Starting M2SVid stage (warping + generation)")
    SF_LOG.debug(f"Depth npz: {depth_npz_path}")
    if not depth_npz_path or not os.path.exists(depth_npz_path):
        SF_LOG.error("No depth file provided")
        return "No depth file from Step 1. Please run Step 1 first or upload a compatible depth.npz.", "", "", "", ""

    if input_video_path is None:
        SF_LOG.error("No input video provided")
        return "Original video not provided. Please select or upload a source video in Step 1 or File Hub.", "", "", "", ""

    video_path = input_video_path
    stem = Path(video_path).stem
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = FINAL_DIR / f"{stem}_stereo_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    reprojected_dir = out_dir / "reprojected"
    reprojected_dir.mkdir(parents=True, exist_ok=True)

    try:
        from m2svid.utils.video_utils import get_video_fps, get_total_frames
        from m2svid.data.utils import apply_closing, apply_dilation, get_video_frames
        from m2svid.utils.anaglyph import make_anaglyph_video
        SF_LOG.debug("Loaded M2SVid/sgm dependencies")

        # Use direct file load for the standalone warping module (avoids nested package conflict)
        warping_mod = _load_warping_module()
        m2s_process_video_with_depth = warping_mod.process_video_with_depth

        # --- Lightweight warping (first part of step 2) ---
        SF_LOG.info("Running geometric warping")
        reprojected_video = reprojected_dir / "input_reprojected.mp4"
        reprojected_mask = reprojected_dir / "input_reprojected_mask.mp4"

        # We call the existing warping logic directly
        m2s_process_video_with_depth(
            video_path=str(video_path),
            depth_path=depth_npz_path,
            output_path_reprojected=str(reprojected_video),
            output_path_mask=str(reprojected_mask),
            disparity_perc=disparity_perc,
            batch_size=warping_batch_size,
        )
        SF_LOG.info("Geometric warping complete")

        try:
            if progress:
                progress(0.35, desc="Geometric warping complete. Starting 1-step M2SVid generation...")
        except TypeError:
            pass

        # --- Heavy part: load M2SVid model and run 1-step generation ---
        model = _load_m2svid_model(m2svid_config, m2svid_ckpt)

        # Replicate the exact preprocessing from inpaint_and_refine.py
        input_video = get_video_frames(video_path)
        reprojected = get_video_frames(str(reprojected_video))
        reprojected_mask_arr = get_video_frames(str(reprojected_mask), video_is_grayscale=True)

        probe = ffmpeg.probe(video_path)
        fps = get_video_fps(video_path, probe)

        reprojected_mask_arr = apply_closing(reprojected_mask_arr, reprojected_closing_kernel)
        reprojected[reprojected_mask_arr.repeat(1, 3, 1, 1) > 0.5] = 0
        reprojected_mask_arr = apply_dilation(reprojected_mask_arr, 3)
        reprojected_mask_arr = reprojected_mask_arr.repeat(1, 3, 1, 1)

        input_video = input_video.permute(1, 0, 2, 3).float() * 2 - 1
        reprojected = reprojected.permute(1, 0, 2, 3).float() * 2 - 1
        reprojected_mask_t = reprojected_mask_arr.permute(1, 0, 2, 3).float() * 2 - 1

        import torchvision.transforms.functional as TF
        orig_shape = input_video.shape[-2:]
        original_input_video = input_video.clone()
        original_reprojected = reprojected.clone()
        # Create a feathered mask [0, 1] for high-quality alpha blending later
        original_mask = reprojected_mask_arr.permute(1, 0, 2, 3).float()
        original_mask_soft = TF.gaussian_blur(original_mask.transpose(0, 1), kernel_size=[11, 11], sigma=[3.0, 3.0]).transpose(0, 1)

        if m2svid_process_res != "Native":
            res_str = m2svid_process_res.split(" ")[0]
            tw = int(res_str.split("x")[0])
            # Calculate height proportionally to preserve aspect ratio, preventing geometric distortion
            th = int((orig_shape[0] / orig_shape[1]) * tw)
            
            # SVD requires dimensions to be exactly divisible by 8 for its VAE latent space
            th = th - (th % 8)
            tw = tw - (tw % 8)
            if orig_shape[0] != th or orig_shape[1] != tw:
                SF_LOG.info(f"Downscaling inputs for M2SVid from {orig_shape} to {(th, tw)}")
                input_video = torch.nn.functional.interpolate(input_video, size=(th, tw), mode="bilinear", align_corners=False)
                reprojected = torch.nn.functional.interpolate(reprojected, size=(th, tw), mode="bilinear", align_corners=False)
                reprojected_mask_t = torch.nn.functional.interpolate(reprojected_mask_t, size=(th, tw), mode="bilinear", align_corners=False)

        c, t, h, w = reprojected_mask_t.shape
        downsampled_resolution = [int(h / 8), int(w / 8)]
        reprojected_mask_t = reprojected_mask_t.permute(1, 0, 2, 3)
        reprojected_mask_t = torch.nn.functional.interpolate(
            reprojected_mask_t.float(), size=downsampled_resolution, mode="bilinear", antialias=bool(mask_antialias)
        )[:, [0]]
        reprojected_mask_t = reprojected_mask_t.permute(1, 0, 2, 3)

        # prepare for generation
        num_samples = gen_chunk_size
        T = input_video.shape[1]
        SF_LOG.info(f"Video length {T} frames; model.max_frames={num_samples}")

        def _save_video(video_tensor, fps_val, path):
            frames = video_tensor.cpu().numpy().transpose(0, 2, 3, 4, 1)
            frames = np.concatenate(frames)
            frames = (((frames + 1) / 2).clip(0, 1) * 255).astype(np.uint8)
            import torchvision.io
            torchvision.io.write_video(path, frames, fps=int(fps_val), options={"crf": "17"})
            written = os.path.getsize(path) if os.path.exists(path) else 0
            SF_LOG.info(f"Wrote {written} bytes to {path}")
            if written < 1024:
                os.remove(path)
                raise RuntimeError(f"Written file suspiciously small ({written} bytes), removed: {path}")

        final_generated = torch.empty(0)
        if T <= num_samples:
            input_batch = {
                "video": input_video[None].cuda(),
                "video_2nd_view": input_video[None].cuda(),
                "reprojected_video": reprojected[None].cuda(),
                "reprojected_mask": reprojected_mask_t[None].cuda(),
                "fps_id": torch.tensor([fps]).cuda(),
                "caption": [""],
                "motion_bucket_id": torch.tensor([127]).cuda(),
            }
            SF_LOG.info("Starting single-chunk generation")
            t0 = time.time()
            clear_cuda()
            try:
                if progress:
                    progress(0.55 if not T else 0.55, desc="Running 1-step conditioned generation (heavy)...")
            except TypeError:
                pass
            with torch.inference_mode():
                final_generated = model.generate(input_batch)["generated-video"][0].cpu()
            t1 = time.time()
            SF_LOG.info(f"Single-chunk generation complete in {t1 - t0:.1f}s")
        else:
            total_chunks = (T + num_samples - 1) // num_samples
            chunk_outputs = []
            SF_LOG.info(f"Video has {T} frames, using chunking: {total_chunks} chunks (padded to {num_samples} frames each)")
            for idx in range(total_chunks):
                s = idx * num_samples
                e = min(s + num_samples, T)
                chunk_len = e - s
                pad_len = num_samples - chunk_len
                SF_LOG.info(f"Chunk {idx+1}/{total_chunks}: frames {s}-{e}{f' (padding {pad_len})' if pad_len else ''}")
                sys.stdout.flush()
                if pad_len > 0:
                    input_chunk = torch.nn.functional.pad(input_video[:, s:e, :, :], (0, 0, 0, 0, 0, pad_len))
                    reproj_chunk = torch.nn.functional.pad(reprojected[:, s:e, :, :], (0, 0, 0, 0, 0, pad_len))
                    mask_chunk = torch.nn.functional.pad(reprojected_mask_t[:, s:e, :, :], (0, 0, 0, 0, 0, pad_len))
                else:
                    input_chunk = input_video[:, s:e, :, :]
                    reproj_chunk = reprojected[:, s:e, :, :]
                    mask_chunk = reprojected_mask_t[:, s:e, :, :]
                input_batch = {
                    "video": input_chunk[None].cuda(),
                    "video_2nd_view": input_chunk[None].cuda(),
                    "reprojected_video": reproj_chunk[None].cuda(),
                    "reprojected_mask": mask_chunk[None].cuda(),
                    "fps_id": torch.tensor([fps]).cuda(),
                    "caption": [""],
                    "motion_bucket_id": torch.tensor([127]).cuda(),
                }
                t0 = time.time()
                try:
                    if progress:
                        progress(0.55 + 0.30 * (idx / total_chunks), desc=f"Generating chunk {idx+1}/{total_chunks}...")
                except TypeError:
                    pass
                with torch.inference_mode():
                    gen_chunk = model.generate(input_batch)["generated-video"][0].cpu()
                t1 = time.time()
                sys.stdout.flush()
                SF_LOG.info(f"Chunk {idx+1}/{total_chunks} done in {t1 - t0:.1f}s")
                if pad_len > 0:
                    gen_chunk = gen_chunk[:, :chunk_len, :, :]
                chunk_outputs.append(gen_chunk)
                clear_cuda()
            final_generated = torch.cat(chunk_outputs, dim=1)

        final_generated = torch.nn.functional.interpolate(
            final_generated, size=orig_shape, mode="bilinear", align_corners=False
        )
        
        # Blend the AI hallucinated pixels ONLY into the masked holes of the razor-sharp original reprojection
        final_generated = original_reprojected * (1.0 - original_mask_soft) + final_generated * original_mask_soft
        
        # Restore the perfectly sharp original left eye
        input_video = original_input_video
        
        # Ensure outputs are padded back to 16:9 standard resolution for hardware compatibility
        c, t, h, w = final_generated.shape
        target_h, target_w = h, w
        if w < int(h * 16 / 9):  # Pillarbox, pad width
            target_w = int(h * 16 / 9)
            target_w = target_w - (target_w % 8)
        elif h < int(w * 9 / 16):  # Letterbox, pad height
            target_h = int(w * 9 / 16)
            target_h = target_h - (target_h % 8)
            
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        
        padded_input = torch.nn.functional.pad(input_video, (pad_left, pad_right, pad_top, pad_bottom), value=-1.0)
        padded_final = torch.nn.functional.pad(final_generated, (pad_left, pad_right, pad_top, pad_bottom), value=-1.0)
        
        SF_LOG.info(f"Padded output from {w}x{h} to 16:9 standard ({target_w}x{target_h})")

        generated_right = out_dir / "generated_right.mp4"
        sbs = out_dir / "stereo_sbs.mp4"
        anaglyph = out_dir / "anaglyph.mp4"

        _save_video(padded_final[None], fps, str(generated_right))
        sbs_tensor = torch.cat([padded_input, padded_final], dim=-1)
        _save_video(sbs_tensor[None], fps, str(sbs))

        try:
            anaglyph_tensor = make_anaglyph_video(padded_input, padded_final, unnormalized_videos=True)
        except Exception as e:
            SF_LOG.error(f"Anaglyph generation failed (non-fatal): {e}")
            anaglyph_tensor = None

        if anaglyph_tensor is not None:
            try:
                _save_video(anaglyph_tensor[None], fps, str(anaglyph))
            except Exception as e:
                SF_LOG.error(f"Anaglyph save failed: {e}")
                anaglyph = None
        else:
            anaglyph = None

        progress(1.0, desc="M2SVid stage complete!")
        clear_cuda()

        status = "✅ M2SVid conversion finished. Download the results below."
        return status, str(generated_right) if os.path.exists(generated_right) else None, str(sbs) if os.path.exists(sbs) else None, str(anaglyph) if anaglyph and os.path.exists(anaglyph) else None, f"Saved to {out_dir.name}"


    except Exception as e:
        clear_cuda()
        return f"❌ Error in M2SVid step: {str(e)}", None, None, None, None


# =============================================================================
# Gradio Interface
# =============================================================================


# =============================================================================
# Gradio Interface
# =============================================================================

def handle_source_upload(file):
    if file is None:
        return None, gr.update()
    src_path = file.name if hasattr(file, "name") else str(file)
    dst_path = SOURCE_DIR / Path(src_path).name
    shutil.copy(src_path, str(dst_path))
    SF_LOG.info(f"Uploaded source video saved to {dst_path}")
    stems = get_source_video_list()
    return str(dst_path), gr.update(choices=[""] + stems, value=Path(src_path).stem)


def handle_depth_upload(file):
    if file is None:
        return None, gr.update()
    src_path = file.name if hasattr(file, "name") else str(file)
    dst_path = DEPTH_DIR / Path(src_path).name
    shutil.copy(src_path, str(dst_path))
    SF_LOG.info(f"Uploaded depth map video saved to {dst_path}")
    
    stem = Path(src_path).stem
    if stem.endswith("_depth"):
        base_stem = stem[:-6]
    else:
        base_stem = stem
        
    out_npz = DEPTH_DIR / f"{base_stem}_depth.npz"
    try:
        convert_depth_video_to_npz(str(dst_path), str(out_npz))
    except Exception as e:
        SF_LOG.error(f"Failed to automatically convert depth video to npz: {e}")
        
    stems = get_source_video_list()
    return str(dst_path), gr.update(choices=[""] + stems, value=base_stem)


def get_depth_video_list():
    """Return list of depth video stems with model suffix info available in depthmaps_videos folder."""
    if not DEPTH_DIR.exists():
        return []
    vids = sorted([p.stem for p in DEPTH_DIR.iterdir() 
                   if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')])
    return vids


SUFFIX_TO_LABEL = {
    "_NGL": "DA3: Nested Giant Large",
    "_ML": "DA3: Mono Large / Metric Large",
    "_G": "DA3: Giant",
    "_L": "DA3: Large",
    "_DP": "Depth Pro",
}


def select_depth_video(stem):
    """Load depth preview for a stem selected from depth dropdown."""
    if not stem:
        return None
    depth_path = DEPTH_DIR / f"{stem}.mp4"
    if not depth_path.exists():
        for ext in ('.mov', '.avi', '.mkv'):
            alt = DEPTH_DIR / f"{stem}{ext}"
            if alt.exists():
                depth_path = alt
                break
    return str(depth_path) if depth_path.exists() else None


def create_stereofaster_ui():
    # Detect default starting file for playback
    stems = get_source_video_list()
    default_video = None
    default_stem = ""
    default_depth_mp4 = None
    default_depth_npz = None
    
    if stems:
        default_stem = stems[0]
        v_p, d_npz = select_source_video(default_stem)
        default_video = v_p
        default_depth_npz = d_npz
        d_mp4 = DEPTH_DIR / f"{default_stem}_depth.mp4"
        if d_mp4.exists():
            default_depth_mp4 = str(d_mp4)

    with gr.Blocks(title="StereoFaster Hub", fill_width=True) as demo:
        gr.HTML(
            """
            <style>
                .gradio-container { max-width: 100% !important; padding: 0 !important; }
                .contain { max-width: 100% !important; padding: 0 !important; }
                #component-0 { max-width: 100% !important; padding: 0 !important; }
            </style>
            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #101827, #0B2545); border-radius: 12px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.05);'>
                <h1 style='color: #00F5FF; font-family: "Outfit", sans-serif; font-size: 2.8em; margin: 0; text-shadow: 0 0 20px rgba(0,245,255,0.3); font-weight: 800;'>StereoFaster</h1>
                <p style='color: #8D99AE; font-size: 1.1em; margin-top: 5px; font-weight: 300;'>Premium Multi-Model Depth & Stereoscopy Pipeline</p>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(f"### ⚙️ Engine Status: `{get_gpu_info()}`")
            with gr.Column(scale=1):
                refresh_btn = gr.Button("🔄 Refresh File Hub", variant="secondary", size="sm")

        with gr.Tabs():
            # ===================== FILE HUB & PREVIEWS =====================
            with gr.Tab("📁 File & Preview Hub"):
                gr.Markdown("#### Match uploaded videos or estimated depth files automatically using the Hub.")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("##### 📤 Source Video Management")
                        source_uploader = gr.File(
                            label="Upload Source Video (saved to source_videos/)",
                            file_types=["video"],
                            type="filepath"
                        )
                        source_dropdown = gr.Dropdown(
                            choices=[""] + stems,
                            value=default_stem,
                            label="Select Prefix for Match & Preview",
                            interactive=True,
                        )
                        preview_video = gr.Video(
                            value=default_video,
                            label="Source Video Preview",
                            interactive=False,
                            format="mp4"
                        )
                        
                        with gr.Accordion("✂️ Auto-Crop Aspect Ratio", open=False):
                            crop_preset = gr.Dropdown(choices=list(ASPECT_RATIO_PRESETS.keys()), value="16:9 (None)", label="Detected Preset", interactive=True)
                            crop_res = gr.Textbox(value="", label="Remaining Resolution", interactive=False)
                            toggle_preview_btn = gr.Button("🔄 Toggle View: Full vs Cropped Preview")
                            apply_crop_btn = gr.Button("✂️ Apply Crop and Save as New Video", variant="primary")
                            crop_preview_state = gr.State(None)
                        
                    with gr.Column():
                        gr.Markdown("##### 🗂️ Depth Map Management")
                        depth_uploader = gr.File(
                            label="Upload External Depth Video (saved to depthmaps_videos/)",
                            file_types=["video"],
                            type="filepath"
                        )
                        depth_dropdown = gr.Dropdown(
                            choices=get_depth_video_list(),
                            value="",
                            label="Select Depth Video to Review",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        preview_depth = gr.Video(
                            value=default_depth_mp4,
                            label="Depth Map Preview",
                            interactive=False,
                            format="mp4"
                        )

            # ===================== STEP 1 =====================
            with gr.Tab("🚀 Step 1 — Depth Estimation"):
                gr.Markdown("#### Compute depth using Depth Anything 3 or Apple Depth Pro.")
                with gr.Row():
                    with gr.Column(scale=2):
                        da3_model = gr.Dropdown(
                            choices=[
                                "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                                "depth-anything/DA3MONO-LARGE",
                                "depth-anything/DA3-GIANT-1.1",
                                "depth-anything/DA3-LARGE-1.1",
                                "depth-anything/DA3METRIC-LARGE",
                                "apple/DepthPro",
                            ],
                            value=DEFAULT_DA3_MODEL,
                            label="Depth Estimation Model",
                        )
                        process_res = gr.Slider(384, 1024, value=720, step=32, label="DA3 Resolution")
                        batch_size = gr.Slider(1, 32, value=_VRAM_DEFAULTS["da3"], step=1, label="DA3 Batch Size")
                        
                        batch_depth_btn = gr.Button("📦 Run Batch Depth Processing on All Source Videos", variant="secondary")
                        batch_crop_btn = gr.Button("✂️ Batch Auto-Crop All Widescreen/IMAX Videos", variant="secondary")
                    with gr.Column(scale=1):
                        step1_dropdown = gr.Dropdown(
                            choices=get_source_video_list(),
                            value="",
                            label="Select Source Video",
                            interactive=True,
                        )
                        step1_btn = gr.Button("⚡ Estimate Depth for Selected Video", variant="primary", size="lg")
                        step1_status = gr.Textbox(label="Estimation Progress", interactive=False)
                        depth_file = gr.File(label="Download Depth .npz", type="filepath")

            # ===================== STEP 2 =====================
            with gr.Tab("🎬 Step 2 — M2SVid Stereography"):
                gr.Markdown("#### Perform geometric warping and single-stepconditioned video generation.")
                with gr.Row():
                    with gr.Column(scale=2):
                        disparity_perc = gr.Slider(0.01, 0.2, value=0.05, step=0.005, label="Disparity Scale")
                        closing_kernel = gr.Slider(3, 21, value=11, step=2, label="Mask Closing Kernel")
                        mask_antialias = gr.Checkbox(label="Mask Antialias", value=False)
                        
                        m2svid_config = gr.Textbox(value=DEFAULT_M2SVID_CONFIG, label="Config Path", visible=False)
                        m2svid_ckpt = gr.Textbox(value=DEFAULT_M2SVID_CKPT, label="Checkpoint Path", visible=False)
                        
                        batch_m2svid_btn = gr.Button("📦 Run Batch Stereography on All Matched Pairs", variant="secondary")
                        warping_batch_size = gr.Slider(1, 16, value=_VRAM_DEFAULTS["warp"], step=1, label="Warping Batch Size (lower = less VRAM)")
                        gen_chunk_size = gr.Slider(2, 35, value=_VRAM_DEFAULTS["gen_chunk"], step=1, label="Generation Chunk Size (lower = less VRAM)")
                        m2svid_process_res = gr.Dropdown(
                            choices=["Native", "1280x720 (Faster)", "1024x576 (Optimal 12GB)", "768x432 (Fastest)"],
                            value="1024x576 (Optimal 12GB)" if _VRAM_DEFAULTS["gen_chunk"] <= 5 else "Native",
                            label="M2SVid Processing Resolution"
                        )

                    with gr.Column(scale=1):
                        step2_btn = gr.Button("🎬 Convert Selected to Stereo", variant="primary", size="lg")
                        step2_status = gr.Textbox(label="Stereography Progress", interactive=False)

                with gr.Row():
                    out_right = gr.Video(label="Generated Right View")
                    out_sbs = gr.Video(label="Side-by-Side (SBS)")
                    out_anaglyph = gr.Video(label="Anaglyph 3D")
                    
                out_dir_box = gr.Textbox(label="Output Directory (all files)", interactive=False)

        # In-memory depth tracking
        depth_input = gr.File(value=default_depth_npz, visible=False)
        depth_state = gr.State(default_depth_npz)

        # Wire dropdown updates
        def _on_hub_select(stem):
            if not stem:
                return None, None, None, None, "16:9 (None)", "", None

            v_p, d_npz = select_source_video(stem)
            d_mp4 = DEPTH_DIR / f"{stem}_depth.mp4"
            d_mp4_p = str(d_mp4) if d_mp4.exists() else None
            
            SF_LOG.info(f"Loaded preview for '{stem}'")
            if d_mp4_p:
                SF_LOG.info(f"  -> Found Depth Video: {d_mp4_p}")
            else:
                SF_LOG.info(f"  -> No Depth Video found yet for '{stem}'")
                
            preset_val, res_val, preview_crop = "16:9 (None)", "", None
            if v_p:
                preset_val, res_val, preview_crop = analyze_letterbox(v_p)
                
            return v_p, d_mp4_p, d_npz, d_npz, preset_val, res_val, preview_crop
            
        source_dropdown.change(
            fn=_on_hub_select,
            inputs=[source_dropdown],
            outputs=[preview_video, preview_depth, depth_input, depth_state, crop_preset, crop_res, crop_preview_state],
        )
        
        def _toggle_preview(current_video, crop_preview):
            # If current video is the full one, show crop preview
            if current_video and crop_preview and "crop_preview" not in current_video:
                return crop_preview
            else:
                # If currently showing cropped, or no crop exists, switch back to full
                stem = source_dropdown.value
                if stem:
                    v_p, _ = select_source_video(stem)
                    return str(v_p) if v_p and v_p.exists() else None
            return None
            
        toggle_preview_btn.click(
            fn=_toggle_preview,
            inputs=[preview_video, crop_preview_state],
            outputs=[preview_video],
        )
        
        def _apply_crop_and_refresh(stem, preset):
            if not stem: return gr.update(), gr.update()
            v_p, _ = select_source_video(stem)
            if v_p and v_p.exists():
                execute_crop(str(v_p), preset)
            
            src = get_source_video_list()
            new_stem = f"{stem}_cropped"
            if new_stem in src:
                return gr.update(choices=[""] + src, value=new_stem), gr.update(choices=[""] + src, value=new_stem)
            return gr.update(choices=[""] + src), gr.update(choices=[""] + src)
            
        apply_crop_btn.click(
            fn=_apply_crop_and_refresh,
            inputs=[source_dropdown, crop_preset],
            outputs=[source_dropdown, step1_dropdown],
        )

        # Wire uploaders
        source_uploader.upload(
            fn=handle_source_upload,
            inputs=[source_uploader],
            outputs=[preview_video, source_dropdown],
        )
        depth_uploader.upload(
            fn=handle_depth_upload,
            inputs=[depth_uploader],
            outputs=[preview_depth, source_dropdown],
        )

        # Wire step 1
        step1_btn.click(
            fn=step1_run_da3_depth,
            inputs=[step1_dropdown, da3_model, process_res, batch_size],
            outputs=[step1_status, preview_depth, depth_file, depth_state],
        )

        # Sync state
        depth_state.change(
            fn=lambda x: x,
            inputs=[depth_state],
            outputs=[depth_input],
        )

        batch_crop_btn.click(
            fn=batch_auto_crop_all,
            inputs=[],
            outputs=[source_dropdown, step1_dropdown],
        )

        # Wire step 2
        step2_btn.click(
            fn=step2_run_m2svid,
            inputs=[
                depth_input,
                disparity_perc,
                closing_kernel,
                mask_antialias,
                m2svid_config,
                m2svid_ckpt,
                preview_video,
                warping_batch_size,
                gen_chunk_size,
                m2svid_process_res,
            ],
            outputs=[step2_status, out_right, out_sbs, out_anaglyph, out_dir_box],
        )

        # Batch buttons
        batch_depth_btn.click(
            fn=run_depth_on_source_videos,
            inputs=[gr.State(), da3_model, process_res, batch_size],
            outputs=None,
        )
        batch_m2svid_btn.click(
            fn=run_m2svid_on_pairs,
            inputs=[m2svid_config, m2svid_ckpt, disparity_perc, closing_kernel, mask_antialias, warping_batch_size, gen_chunk_size, m2svid_process_res],
            outputs=None,
        )

        # Refresh button
        def _refresh_hub():
            src = get_source_video_list()
            dep = get_depth_video_list()
            return (
                gr.update(choices=[""] + src),
                gr.update(choices=[""] + dep),
                gr.update(choices=[""] + src)
            )
            
        refresh_btn.click(
            fn=_refresh_hub,
            inputs=[],
            outputs=[source_dropdown, depth_dropdown, step1_dropdown],
        )
        
        # Wire depth dropdown to update preview
        depth_dropdown.change(
            fn=select_depth_video,
            inputs=[depth_dropdown],
            outputs=[preview_depth],
        )

    return demo


def launch(server_name="127.0.0.1", server_port=7878, share=False):
    demo = create_stereofaster_ui()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        prevent_thread_lock=False,
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 1100px !important; }
            .step-header { font-size: 1.3em; font-weight: 600; margin-bottom: 8px; }
        """
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7878)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("Launching StereoFaster WebUI (DA3 + M2SVid 2-step pipeline)...")
    launch(server_name=args.server_name, server_port=args.server_port, share=args.share)
