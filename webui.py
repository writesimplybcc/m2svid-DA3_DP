"""
StereoFaster WebUI - Gradio Interface
Modern 2-step Video Depth Anything (VDA) / DepthCrafter + M2SVid pipeline for monocular-to-stereo video conversion.

Based on the conceptual structure of the StereoCrafter combined webui,
optimized for StereoFaster:

1. Video Depth Anything (VDA Base & Large) / DepthCrafter Estimation
2. M2SVid stage (lightweight geometric warping + single 1-step conditioned SVD)

Launch:
    python webui.py --server-port 7878

Requirements:
- Video Depth Anything in third_party
- M2SVid dependencies + third_party in path
- M2SVid weights in ckpts/
"""

import os
import sys

# Configure PyTorch CUDA memory allocator to prevent memory fragmentation OOM
# Note: Avoid expandable_segments:True on PyTorch 2.3.0+cu121 due to CUDACachingAllocator.cpp:418 (!handles_.at(i)) assert bug
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.8"

# Ensure UTF-8 stdout/stderr on Windows consoles to prevent charmap UnicodeEncodeErrors with emojis
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Provide immediate feedback to the console before heavy imports lock up the thread
print("===============================================================")
print("🚀 Starting StereoFaster WebUI Initialization...")
print("⏳ Please wait while heavy ML libraries (PyTorch, Gradio) are loaded into memory.")
print("   This may take up to a minute depending on your disk speed.")
print("===============================================================")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")

import logging
import os
import sys
import tempfile
import time
import shutil
import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

def _fix_wsl2_libcuda_for_triton():
    """Triton JIT on WSL2/Docker Desktop requires libcuda.so, but WSL2 only provides libcuda.so.1."""
    if os.name != "posix":
        return
    import glob
    candidates = glob.glob("/usr/lib/wsl/**/libcuda.so.1", recursive=True) + glob.glob("/usr/lib/**/libcuda.so.1", recursive=True)
    if not candidates:
        return
    src = candidates[0]
    targets = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        os.path.join(os.path.dirname(src), "libcuda.so"),
    ]
    for target in targets:
        try:
            if not os.path.exists(target):
                os.symlink(src, target)
        except Exception:
            pass

_fix_wsl2_libcuda_for_triton()

import gradio as gr
import torch

# Polyfill torch.accelerator for PyTorch < 2.6
if not hasattr(torch, "accelerator"):
    class _TorchAccelerator:
        @staticmethod
        def is_available():
            return torch.cuda.is_available()
        @staticmethod
        def device_count():
            return torch.cuda.device_count()
        @staticmethod
        def current_accelerator():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.accelerator = _TorchAccelerator()

import numpy as np
import cv2
import ffmpeg
import threading
import json
from pipeline_sidecar import (
    HARDWARE_PROFILES,
    detect_gpu_hardware,
    get_default_sidecar_config,
    load_sidecar_file,
    save_sidecar_file,
    scan_source_library,
    validate_preflight,
    find_matching_depth_npz,
    run_pipeline_benchmark,
    run_pipeline_batch,
    remux_audio_if_present,
)

# --- PyTorch VRAM & Speed Optimizations for RTX 3000/4000 series ---
# Disable brutal 3-minute benchmarking on Chunk 1 (forces fast heuristics instead)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
# Enable TensorCores for insane speedups on RTX cards
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
# -------------------------------------------------------------------

# Ensure project paths (match what inference scripts use)
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "m2svid"))          # for m2svid.m2svid.*
sys.path.insert(0, str(PROJECT_ROOT / "m2svid" / "m2svid"))  # extra safety
sys.path.insert(0, str(PROJECT_ROOT / "m2svid" / "third_party" / "Video-Depth-Anything"))
sys.path.insert(0, str(PROJECT_ROOT / "m2svid" / "third_party" / "FlashDepth"))
sys.path.insert(0, str(PROJECT_ROOT / "m2svid" / "third_party" / "Hi3D-Official"))
sys.path.insert(0, str(PROJECT_ROOT / "m2svid" / "third_party" / "pytorch-msssim"))
sys.path.insert(0, str(PROJECT_ROOT / "m2svid" / "third_party" / "ICDepth"))

try:
    from m2svid.prepare_flashdepth import run_flashdepth, unload_flashdepth_model
    FLASHDEPTH_AVAILABLE = True
except Exception as e:
    run_flashdepth, unload_flashdepth_model = None, None
    FLASHDEPTH_AVAILABLE = False

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")

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
    """Analyzes a video to detect black bars, matching it to the closest preset."""
    if not video_path or not os.path.exists(video_path):
        return "16:9 (None)", "", ""
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    best_frame = None
    max_variance = -1.0

    if total_frames > 0:
        # Sample candidates across the video to avoid black fade-ins, black transitions, or end credits
        if total_frames >= 10:
            fractions = [0.15, 0.30, 0.50, 0.70, 0.85]
            sample_indices = [int(total_frames * f) for f in fractions]
        elif total_frames > 1:
            sample_indices = [total_frames // 2, 0, total_frames - 1]
        else:
            sample_indices = [0]

        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                var = float(np.var(gray))
                if var > max_variance:
                    max_variance = var
                    best_frame = frame
    cap.release()
    
    # Fallback to FFmpeg if OpenCV couldn't read valid frames or all sampled frames were pure black
    if best_frame is None or max_variance < 1.0:
        import subprocess
        tmp_img = str(PROJECT_ROOT / "scratch" / "tmp_detect_frame.jpg")
        (PROJECT_ROOT / "scratch").mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-ss", "00:00:01", "-i", video_path, "-vframes", "1", "-q:v", "2", tmp_img]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(tmp_img):
            frame = cv2.imread(tmp_img)
            if frame is not None:
                best_frame = frame
            try:
                os.remove(tmp_img)
            except: pass

    if best_frame is None:
        return "16:9 (None)", f"Full: {w}x{h}" if w and h else "", ""
        
    fh, fw = best_frame.shape[:2]
    gray = cv2.cvtColor(best_frame, cv2.COLOR_BGR2GRAY)
    # Threshold at 18 to cleanly exclude limited-range broadcast/codec black bar noise (typically <= 16)
    _, thresh = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
    
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return "16:9 (None)", f"Full: {fw}x{fh}", video_path
        
    x, y, bw, bh = cv2.boundingRect(coords)
    
    if bw == 0 or bh == 0 or (bw >= fw - 8 and bh >= fh - 8):
        return "16:9 (None)", f"Full: {fw}x{fh}", video_path
        
    detected_ratio = bw / bh
    best_preset = "16:9 (None)"
    min_diff = 999.0
    for preset_name, ratio in ASPECT_RATIO_PRESETS.items():
        diff = abs(detected_ratio - ratio)
        if diff < min_diff:
            min_diff = diff
            best_preset = preset_name
            
    if best_preset == "16:9 (None)":
        return best_preset, f"Full: {fw}x{fh}", video_path
        
    # Calculate crop
    target_ratio = ASPECT_RATIO_PRESETS[best_preset]
    if target_ratio > (fw / fh): # Letterbox
        new_h = int(fw / target_ratio)
        new_h = new_h - (new_h % 8)
        new_w = fw
    else: # Pillarbox
        new_w = int(fh * target_ratio)
        new_w = new_w - (new_w % 8)
        new_h = fh
        
    res_str = f"{new_w}x{new_h}"
    
    # Generate 1-second preview
    preview_dir = PROJECT_ROOT / "scratch"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = str(preview_dir / f"crop_preview_{Path(video_path).stem}.mp4")
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

def batch_auto_crop_all(force_crop_enabled=False, manual_preset="16:9 (None)", progress=gr.Progress(track_tqdm=True)):
    vids = [p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')]
    vids = [v for v in vids if not v.stem.endswith("_cropped")]
    if not vids:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        
    for i, vp in enumerate(vids):
        progress(float(i)/len(vids), desc=f"Auto-Cropping {vp.name}")
        
        if force_crop_enabled:
            preset = manual_preset
        else:
            preset, _, _ = analyze_letterbox(str(vp))
            
        if preset != "16:9 (None)":
            execute_crop(str(vp), preset, progress=None)
            
    progress(1.0, desc="Batch Crop Complete!")
    src = get_source_video_list()
    return (
        gr.update(),
        gr.update(choices=[""] + src),
        gr.update(choices=[""] + src),
        gr.update(choices=[""] + src),
        gr.update(choices=[""] + src),
        gr.update(choices=[""] + src),
    )




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
    if depth.max() > 1.0:
        d_min = float(depth.min())
        d_max = float(depth.max())
        denom = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1.0
        depth = (depth - d_min) / denom
    save_m2svid_compatible_npz(depth, out_npz_path)


def save_m2svid_compatible_npz(depth: np.ndarray, out_path: str):
    """Save depth array with the single key 'depth' that m2svid/warping.py expects."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(p), depth=depth.astype(np.float32))


def load_video_frames(video_path: str, target_fps: float = 0.0) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    """Extract frames from video. target_fps=0 means every frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_interval = 1
    if target_fps > 0:
        frame_interval = max(1, int(round(video_fps / target_fps)))
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()
    if not frames:
        raise RuntimeError("No frames extracted")
    h, w = frames[0].shape[:2]
    actual_fps = video_fps / frame_interval if frame_interval > 0 else video_fps
    return frames, actual_fps, (h, w)


def run_depth_on_source_videos(
    model_name="tencent/DepthCrafter",
    process_res=720,
    batch_size=4,
    guidance_scale=1.0,
    inference_steps=5,
    window_size=30,
    overlap=10,
    attn_slicing="Auto (Adapts to GPU VRAM)",
    cpu_offload="Auto (Adapts to GPU VRAM)",
    progress=gr.Progress(track_tqdm=True)
):
    """Process all videos in SOURCE_DIR and save depth .npz and depth_depth.mp4 in DEPTH_DIR if missing."""
    model_name = model_name or "tencent/DepthCrafter"
    is_depthcrafter = True
    suffix = get_model_suffix(model_name)
    SF_LOG.info(f"Starting batch DepthCrafter processing on source_videos directory")
    unload_m2svid_model()
    
    vids = sorted([p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')], key=lambda p: p.name)
    
    final_vids = []
    stems_in_dir = {p.stem for p in vids}
    for vp in vids:
        if not vp.stem.endswith("_cropped"):
            if f"{vp.stem}_cropped" in stems_in_dir:
                SF_LOG.info(f"Skipping {vp.name} because a _cropped version exists")
                continue
        final_vids.append(vp)
        
    vids = final_vids
    total = len(vids)
    SF_LOG.info(f"Found {total} video(s) in source_videos to process after filtering")
    
    # Pre-scan videos to detect which files already have depth maps (e.g. processed with lower settings)
    skipped_already_done = []
    to_process = []
    for vp in vids:
        stem = vp.stem
        out_npz = DEPTH_DIR / f"{stem}{suffix}_depth.npz"
        out_mp4 = DEPTH_DIR / f"{stem}{suffix}_depth.mp4"
        if out_npz.exists() and out_mp4.exists():
            skipped_already_done.append(vp.name)
        else:
            to_process.append(vp)

    SF_LOG.info("===============================================================")
    SF_LOG.info(f"[Batch Depth] 📁 Found {total} total source video(s).")
    SF_LOG.info(f"[Batch Depth] ⏭️ {len(skipped_already_done)}/{total} video(s) already have existing depth maps (skipped).")
    SF_LOG.info(f"[Batch Depth] ⏳ {len(to_process)}/{total} video(s) remaining to process with current settings.")
    SF_LOG.info("===============================================================")

    src = get_source_video_list()
    dep = get_depth_video_list()

    if total == 0:
        summary_msg = "ℹ️ No source videos found in source_videos/ directory."
        SF_LOG.info(summary_msg)
        return gr.update(), gr.update(choices=[""] + src), gr.update(choices=[""] + dep), gr.update(choices=[""] + src), summary_msg

    if len(to_process) == 0:
        summary_lines = [
            "===============================================================",
            "📊 Batch Depth Processing Summary:",
            f"• Total source videos: {total}",
            f"• Skipped (already completed / existing depth): {len(skipped_already_done)}/{total}",
            f"• Successfully processed in this run: 0/{total}",
            f"• Skipped due to CUDA OOM: 0/{total}",
            "ℹ️ All videos already have existing depth maps (skipped). No new videos to process.",
            "===============================================================",
        ]
        summary_msg = "\n".join(summary_lines)
        for line in summary_lines:
            SF_LOG.info(line)
        return gr.update(), gr.update(choices=[""] + src), gr.update(choices=[""] + dep), gr.update(choices=[""] + src), summary_msg

    successful_files = []
    oom_files = []
    error_files = []

    for i, vp in enumerate(to_process):
        global GLOBAL_CANCEL
        if GLOBAL_CANCEL:
            GLOBAL_CANCEL = False
            SF_LOG.info("Batch depth processing cancelled by user.")
            break
        stem = vp.stem
        out_npz = DEPTH_DIR / f"{stem}{suffix}_depth.npz"
        out_mp4 = DEPTH_DIR / f"{stem}{suffix}_depth.mp4"
        
        # Double check in case created during run
        if out_npz.exists() and out_mp4.exists():
            if vp.name not in skipped_already_done:
                skipped_already_done.append(vp.name)
            SF_LOG.info(f"Skipping {stem}: depth already exists at {out_npz} and {out_mp4}")
            continue
            
        SF_LOG.info(f"Depth processing [{i+1}/{len(to_process)} of remaining, total {total}]: {stem} using model {model_name}")
        if progress:
            try: progress(float(i)/max(1, len(to_process)), desc=f"Processing [{i+1}/{len(to_process)}]: {stem}")
            except Exception: pass
            
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cap = cv2.VideoCapture(str(vp))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0: fps = 30.0
            cap.release()
            
            from m2svid.prepare_depthcrafter import run_depthcrafter_depth
            depth = run_depthcrafter_depth(
                str(vp),
                process_res=process_res,
                guidance_scale=guidance_scale,
                num_inference_steps=inference_steps,
                window_size=window_size,
                overlap=overlap,
                attn_slicing=attn_slicing,
                cpu_offload=cpu_offload,
                progress=progress
            )
                
            if depth.shape[1:] != (h, w):
                depth = np.stack([cv2.resize(d, (w, h), cv2.INTER_CUBIC) for d in depth])
                
            save_m2svid_compatible_npz(depth, str(out_npz))
            _create_depth_preview_video(depth, str(out_mp4), fps)
            SF_LOG.info(f"Saved depth files: {out_npz} and {out_mp4}")
            
            successful_files.append(vp.name)
            # Immediately free depth numpy array and flush GPU cache before next video
            del depth
            clear_cuda()
            
            if progress:
                try: progress(float(i+1)/max(1, len(to_process)), desc=f"Processed {stem}")
                except Exception: pass
        except Exception as e:
            clear_cuda()
            if is_depthcrafter:
                try:
                    from m2svid.prepare_depthcrafter import unload_depthcrafter_model
                    unload_depthcrafter_model()
                except Exception:
                    pass
                clear_cuda()

            is_oom = is_cuda_oom(e)
            if is_oom:
                oom_files.append(vp.name)
                SF_LOG.error(f"❌ [CUDA OOM] Skipped {vp.name} due to Out of Memory ({len(oom_files)}/{total} OOM so far).")
                SF_LOG.error(f"   💡 Recommendation for {vp.name}: Reduce Window Size (e.g. 60 or 40), enable CPU Offload: Model, or lower Resolution.")
            elif "Stopped by user" in str(e):
                SF_LOG.info("Batch depth processing cancelled by user.")
                break
            else:
                error_files.append((vp.name, str(e)))
                SF_LOG.error(f"❌ Error on {vp.name}: {e}")
                import traceback
                traceback.print_exc()

            if progress:
                try:
                    status_desc = f"OOM: {stem}" if is_oom else f"Error: {stem}"
                    progress(float(i+1)/max(1, len(to_process)), desc=status_desc)
                except Exception: pass

    SF_LOG.info("Batch depth processing complete")
    
    summary_lines = [
        "===============================================================",
        "📊 Batch Depth Processing Summary:",
        f"• Total source videos: {total}",
        f"• Skipped (already completed / existing depth): {len(skipped_already_done)}/{total}",
        f"• Successfully processed in this run: {len(successful_files)}/{total}",
    ]
    if oom_files:
        summary_lines.append(f"• ❌ Skipped due to CUDA OOM: {len(oom_files)}/{total}")
        summary_lines.append("  ⚠️ OOM Video(s):")
        for f in oom_files:
            summary_lines.append(f"    - {f}")
        summary_lines.append("  💡 Recommendations for OOM files:")
        summary_lines.append("     1. Lower Window Size (e.g. 60 or 40 frames)")
        summary_lines.append("     2. Set CPU Offload to 'Model'")
        summary_lines.append("     3. Lower Resolution (e.g. 1080 -> 720)")
        summary_lines.append("     4. Set Attention Slicing to 'Auto' or 'Enabled'")
    else:
        summary_lines.append(f"• Skipped due to CUDA OOM: 0/{total}")

    if error_files:
        summary_lines.append(f"• ⚠️ Other errors: {len(error_files)}/{total}")
        for f, err in error_files:
            summary_lines.append(f"    - {f}: {err[:80]}")

    summary_lines.append("===============================================================")
    summary_msg = "\n".join(summary_lines)
    for line in summary_lines:
        SF_LOG.info(line)

    try:
        from m2svid.prepare_depthcrafter import unload_depthcrafter_model
        unload_depthcrafter_model()
    except Exception:
        pass
    clear_cuda()

    src = get_source_video_list()
    dep = get_depth_video_list()

    return (
        gr.update(),
        gr.update(choices=[""] + src),
        gr.update(choices=[""] + dep),
        gr.update(choices=[""] + src),
        summary_msg
    )


def run_vda_on_source_videos(
    encoder="Video-Depth-Anything-Large (vitl - 381.8M, Highest Quality)",
    mode="Sliding Window (Optimal Quality & Alignment)",
    input_size=518,
    max_res="1280 (Recommended)",
    target_fps="-1 (Native Source FPS)",
    max_len=-1,
    fp32=False,
    global_norm=True,
    invert_depth=False,
    progress=gr.Progress(track_tqdm=True)
):
    """Process all videos in SOURCE_DIR with Video Depth Anything and save into DEPTH_DIR."""
    SF_LOG.info("Starting batch VDA depth processing on source_videos")
    unload_m2svid_model()
    enc_code = "vitl" if "large" in str(encoder).lower() or "vitl" in str(encoder).lower() else "vitb"
    suffix = "_VDAL" if enc_code == "vitl" else "_VDAB"
    
    clean_res = 1280
    if isinstance(max_res, str):
        for part in max_res.split():
            if part.isdigit():
                clean_res = int(part)
                break
    elif isinstance(max_res, (int, float)):
        clean_res = int(max_res)
        
    clean_fps = -1.0
    if isinstance(target_fps, str):
        for part in target_fps.split():
            try:
                clean_fps = float(part)
                break
            except ValueError:
                pass
    elif isinstance(target_fps, (int, float)):
        clean_fps = float(target_fps)
        
    clean_mode = "streaming" if "stream" in str(mode).lower() else "sliding_window"

    vids = sorted([p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')], key=lambda p: p.name)
    vids = [vp for vp in vids if not (not vp.stem.endswith("_cropped") and f"{vp.stem}_cropped" in {p.stem for p in vids})]
    total = len(vids)
    SF_LOG.info(f"Found {total} video(s) to process in batch VDA")

    successful_files = []
    error_files = []
    for i, vp in enumerate(vids):
        global GLOBAL_CANCEL
        if GLOBAL_CANCEL:
            GLOBAL_CANCEL = False
            SF_LOG.info("Batch VDA processing cancelled by user.")
            break
            
        stem = vp.stem
        out_npz = DEPTH_DIR / f"{stem}{suffix}_depth.npz"
        out_mp4 = DEPTH_DIR / f"{stem}{suffix}_depth.mp4"
        if out_npz.exists() and out_mp4.exists():
            SF_LOG.info(f"Skipping {stem}: already has VDA depth at {out_npz.name}")
            continue

        if progress:
            try: progress(float(i) / max(1, total), desc=f"VDA [{i+1}/{total}]: {stem}")
            except Exception: pass

        try:
            depth, fps = run_vda_depth(
                video_path=str(vp),
                encoder=enc_code,
                input_size=int(input_size),
                max_res=clean_res,
                mode=clean_mode,
                target_fps=clean_fps,
                max_len=int(max_len),
                fp32=fp32,
                global_norm=global_norm,
                invert_depth=invert_depth,
                progress=progress
            )
            save_m2svid_compatible_npz(depth, str(out_npz))
            _create_depth_preview_video(depth, str(out_mp4), fps)
            successful_files.append(vp.name)
            del depth
            clear_cuda()
        except Exception as e:
            clear_cuda()
            SF_LOG.error(f"Error processing {stem} with VDA: {e}")
            error_files.append((vp.name, str(e)))
            if "stopped by user" in str(e).lower():
                break

    unload_vda_model()
    clear_cuda()
    src = get_source_video_list()
    dep = get_depth_video_list()
    msg = f"✅ Batch VDA Complete! Successfully processed {len(successful_files)} video(s)."
    if error_files:
        msg += f" (⚠️ {len(error_files)} errors)"
    return gr.update(), gr.update(choices=[""] + src), gr.update(choices=[""] + dep), gr.update(choices=[""] + src), msg
 
 
def run_flashdepth_on_source_videos(
    model_name="FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
    max_res="1280 (Recommended)",
    target_fps="-1 (Native Source FPS)",
    max_len=-1,
    global_norm=True,
    invert_depth=False,
    progress=gr.Progress(track_tqdm=True)
):
    """Process all videos in SOURCE_DIR with FlashDepth and save into DEPTH_DIR."""
    SF_LOG.info("Starting batch FlashDepth processing on source_videos")
    unload_m2svid_model()
    
    clean_mod = str(model_name).lower()
    suffix = "_FDL" if "large" in clean_mod or "vitl" in clean_mod else ("_FDS" if "small" in clean_mod or "vits" in clean_mod else "_FDF")
    
    clean_res = 1280
    if isinstance(max_res, str):
        for part in max_res.split():
            if part.isdigit():
                clean_res = int(part)
                break
    elif isinstance(max_res, (int, float)):
        clean_res = int(max_res)
        
    clean_fps = -1.0
    if isinstance(target_fps, str):
        for part in target_fps.split():
            try:
                clean_fps = float(part)
                break
            except ValueError:
                pass
    elif isinstance(target_fps, (int, float)):
        clean_fps = float(target_fps)

    vids = sorted([p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')], key=lambda p: p.name)
    vids = [vp for vp in vids if not (not vp.stem.endswith("_cropped") and f"{vp.stem}_cropped" in {p.stem for p in vids})]
    total = len(vids)
    SF_LOG.info(f"Found {total} video(s) to process in batch FlashDepth")

    successful_files = []
    error_files = []
    for i, vp in enumerate(vids):
        global GLOBAL_CANCEL
        if GLOBAL_CANCEL:
            GLOBAL_CANCEL = False
            SF_LOG.info("Batch FlashDepth processing cancelled by user.")
            break
            
        stem = vp.stem
        out_npz = DEPTH_DIR / f"{stem}{suffix}_depth.npz"
        out_mp4 = DEPTH_DIR / f"{stem}{suffix}_depth.mp4"
        if out_npz.exists() and out_mp4.exists():
            SF_LOG.info(f"Skipping {stem}: already has FlashDepth output at {out_npz.name}")
            continue

        if progress:
            try: progress(float(i) / max(1, total), desc=f"FlashDepth [{i+1}/{total}]: {stem}")
            except Exception: pass

        try:
            depth, fps = run_flashdepth(
                video_path=str(vp),
                model_name=model_name,
                max_res=clean_res,
                target_fps=clean_fps,
                max_len=int(max_len),
                global_norm=global_norm,
                invert_depth=invert_depth,
                progress=progress
            )
            save_m2svid_compatible_npz(depth, str(out_npz))
            _create_depth_preview_video(depth, str(out_mp4), fps)
            successful_files.append(vp.name)
            del depth
            clear_cuda()
        except Exception as e:
            clear_cuda()
            SF_LOG.error(f"Error processing {stem} with FlashDepth: {e}")
            error_files.append((vp.name, str(e)))
            if "stopped by user" in str(e).lower():
                break

    try:
        unload_flashdepth_model()
    except Exception:
        pass
    clear_cuda()
    src = get_source_video_list()
    dep = get_depth_video_list()
    msg = f"✅ Batch FlashDepth Complete! Successfully processed {len(successful_files)} video(s)."
    if error_files:
        msg += f" (⚠️ {len(error_files)} errors)"
    return gr.update(), gr.update(choices=[""] + src), gr.update(choices=[""] + dep), gr.update(choices=[""] + src), msg


def run_icdepth_on_source_videos(
    max_res=720,
    window_size=81,
    overlap=32,
    steps=5,
    cfg_scale=5.0,
    tiled=False,
    void_title_recovery=True,
    volumetric_splash_recovery=False,
    credit_pop=True,
    credit_text_hint="",
    progress=gr.Progress(track_tqdm=True),
):
    """Process all videos in SOURCE_DIR with ICDepth (Wan2.1 In-Context DiT) and save into DEPTH_DIR."""
    suffix = "_ICD"
    SF_LOG.info("Starting batch ICDepth processing on source_videos directory")
    unload_m2svid_model()

    vids = sorted(
        [p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')],
        key=lambda p: p.name,
    )
    stems_in_dir = {p.stem for p in vids}
    final_vids = []
    for vp in vids:
        if not vp.stem.endswith("_cropped") and f"{vp.stem}_cropped" in stems_in_dir:
            SF_LOG.info(f"Skipping {vp.name} because a _cropped version exists")
            continue
        final_vids.append(vp)

    vids = final_vids
    total = len(vids)
    SF_LOG.info(f"Found {total} video(s) in source_videos to process after filtering")

    skipped_already_done = []
    to_process = []
    for vp in vids:
        stem = vp.stem
        out_npz = DEPTH_DIR / f"{stem}{suffix}_depth.npz"
        out_mp4 = DEPTH_DIR / f"{stem}{suffix}_depth.mp4"
        if out_npz.exists() and out_mp4.exists():
            skipped_already_done.append(vp.name)
        else:
            to_process.append(vp)

    SF_LOG.info("===============================================================")
    SF_LOG.info(f"[Batch ICDepth] 📁 Found {total} total source video(s).")
    SF_LOG.info(f"[Batch ICDepth] ⏭️ {len(skipped_already_done)}/{total} video(s) already have existing ICDepth maps (skipped).")
    SF_LOG.info(f"[Batch ICDepth] ⏳ {len(to_process)}/{total} video(s) remaining to process with current settings.")
    SF_LOG.info("===============================================================")

    src = get_source_video_list()
    dep = get_depth_video_list()

    if total == 0:
        summary_msg = "ℹ️ No source videos found in source_videos/ directory."
        SF_LOG.info(summary_msg)
        return gr.update(), gr.update(choices=[""] + src), gr.update(choices=[""] + dep), gr.update(choices=[""] + src), summary_msg

    if len(to_process) == 0:
        summary_lines = [
            "===============================================================",
            "📊 Batch ICDepth Processing Summary:",
            f"• Total source videos: {total}",
            f"• Skipped (already completed / existing _ICD_depth): {len(skipped_already_done)}/{total}",
            f"• Successfully processed in this run: 0/{total}",
            f"• Skipped due to CUDA OOM: 0/{total}",
            "ℹ️ All videos already have existing ICDepth maps (skipped). No new videos to process.",
            "===============================================================",
        ]
        summary_msg = "\n".join(summary_lines)
        for line in summary_lines:
            SF_LOG.info(line)
        return gr.update(), gr.update(choices=[""] + src), gr.update(choices=[""] + dep), gr.update(choices=[""] + src), summary_msg

    from m2svid.prepare_icdepth import run_icdepth_depth, unload_icdepth_model

    successful_files = []
    oom_files = []
    error_files = []

    for i, vp in enumerate(to_process):
        global GLOBAL_CANCEL
        if GLOBAL_CANCEL:
            GLOBAL_CANCEL = False
            SF_LOG.info("Batch ICDepth processing cancelled by user.")
            break

        stem = vp.stem
        out_npz = DEPTH_DIR / f"{stem}{suffix}_depth.npz"
        out_mp4 = DEPTH_DIR / f"{stem}{suffix}_depth.mp4"
        if out_npz.exists() and out_mp4.exists():
            if vp.name not in skipped_already_done:
                skipped_already_done.append(vp.name)
            SF_LOG.info(f"Skipping {stem}: ICDepth output already exists at {out_npz} and {out_mp4}")
            continue

        SF_LOG.info(f"ICDepth processing [{i+1}/{len(to_process)} of remaining, total {total}]: {stem}")
        if progress:
            try:
                progress(float(i) / max(1, len(to_process)), desc=f"ICDepth [{i+1}/{len(to_process)}]: {stem}")
            except Exception:
                pass

        try:
            cap = cv2.VideoCapture(str(vp))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # In batch mode, each clip automatically loads its own {stem}.credits.txt sidecar if present
            depth, out_fps = run_icdepth_depth(
                str(vp),
                max_res=int(max_res),
                window_size=int(window_size),
                overlap=int(overlap),
                num_inference_steps=int(steps),
                cfg_scale=float(cfg_scale),
                tiled=bool(tiled),
                void_title_recovery=bool(void_title_recovery),
                volumetric_splash_recovery=bool(volumetric_splash_recovery),
                credit_pop=bool(credit_pop),
                credit_text_hint="",
                progress=progress,
            )

            if depth.shape[1:] != (h, w):
                interp = cv2.INTER_AREA if (w <= depth.shape[2] and h <= depth.shape[1]) else cv2.INTER_LINEAR
                depth = np.stack([cv2.resize(d, (w, h), interpolation=interp) for d in depth])

            save_m2svid_compatible_npz(depth, str(out_npz))
            _create_depth_preview_video(depth, str(out_mp4), out_fps)
            SF_LOG.info(f"Saved ICDepth files: {out_npz} and {out_mp4}")

            successful_files.append(vp.name)
            del depth
            clear_cuda()

            if progress:
                try:
                    progress(float(i + 1) / max(1, len(to_process)), desc=f"Processed {stem}")
                except Exception:
                    pass
        except Exception as e:
            clear_cuda()
            is_oom = is_cuda_oom(e)
            if is_oom:
                try:
                    unload_icdepth_model()
                except Exception:
                    pass
                clear_cuda()
                oom_files.append(vp.name)
                SF_LOG.error(f"❌ [CUDA OOM] Skipped {vp.name} due to Out of Memory ({len(oom_files)}/{total} OOM so far).")
            elif "stopped by user" in str(e).lower():
                SF_LOG.info("Batch ICDepth processing cancelled by user.")
                break
            else:
                error_files.append((vp.name, str(e)))
                SF_LOG.error(f"❌ Error on {vp.name} in ICDepth: {e}")
                import traceback
                traceback.print_exc()

    try:
        unload_icdepth_model()
    except Exception:
        pass
    clear_cuda()

    summary_lines = [
        "===============================================================",
        "📊 Batch ICDepth Processing Summary:",
        f"• Total source videos: {total}",
        f"• Skipped (already completed / existing _ICD_depth): {len(skipped_already_done)}/{total}",
        f"• Successfully processed in this run: {len(successful_files)}/{total}",
    ]
    if oom_files:
        summary_lines.append(f"• ❌ Skipped due to CUDA OOM: {len(oom_files)}/{total}")
        for f in oom_files:
            summary_lines.append(f"    - {f}")
    else:
        summary_lines.append(f"• Skipped due to CUDA OOM: 0/{total}")

    if error_files:
        summary_lines.append(f"• ⚠️ Other errors: {len(error_files)}/{total}")
        for f, err in error_files:
            summary_lines.append(f"    - {f}: {err}")
    summary_lines.append("===============================================================")
    summary_msg = "\n".join(summary_lines)
    for line in summary_lines:
        SF_LOG.info(line)

    src = get_source_video_list()
    dep = get_depth_video_list()
    return (
        gr.update(),
        gr.update(choices=[""] + src),
        gr.update(choices=[""] + dep),
        gr.update(choices=[""] + src),
        summary_msg,
    )



def run_m2svid_on_pairs(
    m2svid_config=None,
    m2svid_ckpt=None,
    disparity_perc=20.0,
    convergence_point=0.5,
    closing_kernel=11,
    mask_antialias=False,
    warping_batch_size=None,
    gen_chunk_size=None,
    m2svid_process_res="1024x576 (Optimal 12GB)",
    gen_batch_size=None,
    progress=gr.Progress(track_tqdm=True)
):
    """Process pairs in SOURCE_DIR and DEPTH_DIR and save outputs into FINAL_DIR."""
    SF_LOG.info("Starting batch M2SVid processing on source_videos + depthmaps_videos")
    try:
        from m2svid.prepare_depthcrafter import unload_depthcrafter_model
        unload_depthcrafter_model()
    except Exception:
        pass
    cfg = m2svid_config or DEFAULT_M2SVID_CONFIG
    ckpt = m2svid_ckpt or DEFAULT_M2SVID_CKPT
    warping_batch_size = warping_batch_size or _VRAM_DEFAULTS["warp"]
    gen_chunk_size = gen_chunk_size or _VRAM_DEFAULTS["gen_chunk"]
    gen_batch_size = gen_batch_size or _VRAM_DEFAULTS.get("gen_batch", 1)
    m2svid_process_res = m2svid_process_res or "1024x576 (Optimal 12GB)"
    vids = sorted([p for p in SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')], key=lambda p: p.name)
    total = len(vids)
    SF_LOG.info(f"Found {total} video(s) to process")
    for i, vp in enumerate(vids):
        global GLOBAL_CANCEL
        if GLOBAL_CANCEL:
            GLOBAL_CANCEL = False
            SF_LOG.info("Batch M2SVid processing cancelled by user.")
            break
        stem = vp.stem
        hybrid_npz = DEPTH_DIR / f"{stem}_hybrid_depth.npz"
        dc_npz = DEPTH_DIR / f"{stem}_DC_depth.npz"
        icd_npz = DEPTH_DIR / f"{stem}_ICD_depth.npz"
        fdf_npz = DEPTH_DIR / f"{stem}_FDF_depth.npz"
        fdl_npz = DEPTH_DIR / f"{stem}_FDL_depth.npz"
        fd_npz = DEPTH_DIR / f"{stem}_FD_depth.npz"
        vdal_npz = DEPTH_DIR / f"{stem}_VDAL_depth.npz"
        vda_npz = DEPTH_DIR / f"{stem}_VDA_depth.npz"
        vdab_npz = DEPTH_DIR / f"{stem}_VDAB_depth.npz"
        legacy_npz = DEPTH_DIR / f"{stem}_depth.npz"
        if hybrid_npz.exists():
            depth_npz = hybrid_npz
        elif dc_npz.exists():
            depth_npz = dc_npz
        elif icd_npz.exists():
            depth_npz = icd_npz
        elif fdf_npz.exists():
            depth_npz = fdf_npz
        elif fdl_npz.exists():
            depth_npz = fdl_npz
        elif fd_npz.exists():
            depth_npz = fd_npz
        elif vdal_npz.exists():
            depth_npz = vdal_npz
        elif vda_npz.exists():
            depth_npz = vda_npz
        elif vdab_npz.exists():
            depth_npz = vdab_npz
        elif legacy_npz.exists():
            depth_npz = legacy_npz
        else:
            candidates = sorted([p for p in DEPTH_DIR.iterdir() if p.is_file() and p.suffix == ".npz" and p.stem.startswith(stem + "_") and "_depth" in p.stem])
            if candidates:
                depth_npz = candidates[0]
            else:
                depth_npz = None

        if depth_npz is None:
            # Check if an orphan depth MP4 exists (e.g. uploaded via FileBrowser or kept for space saving)
            mp4_candidates = [
                DEPTH_DIR / f"{stem}_ICD_depth.mp4",
                DEPTH_DIR / f"{stem}_VDAL_depth.mp4",
                DEPTH_DIR / f"{stem}_VDA_depth.mp4",
                DEPTH_DIR / f"{stem}_VDAB_depth.mp4",
                DEPTH_DIR / f"{stem}_DC_depth.mp4",
                DEPTH_DIR / f"{stem}_depth.mp4",
            ]
            if DEPTH_DIR.exists():
                for p in sorted(DEPTH_DIR.glob(f"{stem}_*_depth.mp4")):
                    if p not in mp4_candidates:
                        mp4_candidates.append(p)
            found_mp4 = None
            for mc in mp4_candidates:
                if mc.exists():
                    found_mp4 = mc
                    break
            if found_mp4:
                target_npz = found_mp4.parent / f"{found_mp4.name.rsplit('.', 1)[0]}.npz"
                try:
                    SF_LOG.info(f"[Batch M2SVid] Auto-converting {found_mp4.name} -> {target_npz.name} for {stem}")
                    convert_depth_video_to_npz(str(found_mp4), str(target_npz))
                    if target_npz.exists():
                        depth_npz = target_npz
                except Exception as e:
                    SF_LOG.error(f"Failed to auto-convert {found_mp4.name} to npz: {e}")

        if depth_npz is None:
            SF_LOG.warning(f"M2SVid skipping {stem}: missing depth npz")
            if progress:
                try: progress(float(i)/max(1,total), desc=f"M2SVid: missing depth for {stem}")
                except Exception: pass
            continue

        out_clip_dir = FINAL_DIR / f"{stem}_stereo"
        final_out = out_clip_dir / f"{stem}_generated_right.mp4"
        alt_out = out_clip_dir / "generated_right.mp4"
        legacy_out = FINAL_DIR / f"{stem}_generated_right.mp4"

        # Organize any files currently outside in final_videos into their clip folder
        if legacy_out.exists():
            out_clip_dir.mkdir(parents=True, exist_ok=True)
            for suffix in ["generated_right.mp4", "stereo_sbs.mp4", "anaglyph.mp4"]:
                leg_f = FINAL_DIR / f"{stem}_{suffix}"
                if leg_f.exists():
                    target_f = out_clip_dir / f"{stem}_{suffix}"
                    if not target_f.exists():
                        try:
                            shutil.move(str(leg_f), str(target_f))
                            SF_LOG.info(f"Moved {leg_f.name} into {out_clip_dir.name}/")
                        except Exception as e:
                            SF_LOG.warning(f"Could not move {leg_f.name} to {target_f}: {e}")

        if final_out.exists() or alt_out.exists():
            SF_LOG.info(f"Skipping {stem}: already processed at {out_clip_dir}")
            if progress:
                try: progress(float(i)/max(1,total), desc=f"M2SVid: skipping {stem}, already done")
                except Exception: pass
            continue
        SF_LOG.info(f"M2SVid processing [{i+1}/{total}]: {stem}")
        try:
            prev_input = STATE.get("input_video")
            STATE["input_video"] = str(vp)
            prefix = f"[{i+1}/{total}: {stem}] "
            status, gen_right, sbs, anaglyph, out_dir = step2_run_m2svid(str(depth_npz), disparity_perc, convergence_point, closing_kernel, mask_antialias, cfg, ckpt, input_video_path=str(vp), warping_batch_size=warping_batch_size, gen_chunk_size=gen_chunk_size, m2svid_process_res=m2svid_process_res, gen_batch_size=gen_batch_size, progress=progress, progress_prefix=prefix)
            if not gen_right or "Error" in status:
                SF_LOG.error(f"M2SVid failed on {stem}: {status}")
                STATE["input_video"] = prev_input
                if "Stopped by user" in status:
                    SF_LOG.info("Batch M2SVid processing cancelled by user.")
                    break
                continue

            if out_dir and os.path.exists(out_dir):
                SF_LOG.info(f"All outputs for {stem} successfully saved in folder: {out_dir}")
            STATE["input_video"] = prev_input
            clear_cuda()
            if progress:
                try: progress(float(i+1)/max(1,total), desc=f"M2SVid: processed {stem}")
                except Exception: pass
        except Exception as e:
            SF_LOG.error(f"M2SVid error on {stem}: {e}")
            STATE["input_video"] = prev_input
            clear_cuda()
            if "Stopped by user" in str(e):
                SF_LOG.info("Batch M2SVid processing cancelled by user.")
                break
            import traceback
            traceback.print_exc()
    unload_m2svid_model()
    SF_LOG.info("Batch M2SVid processing complete")
    return "✅ Batch M2SVid processing complete! All outputs saved to final_videos/."


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

# Video Depth Anything (VDA) module loader
def _load_prepare_vda_module():
    try:
        import m2svid.m2svid.prepare_vda_depth as mod
        return mod
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "prepare_vda_depth",
            PROJECT_ROOT / "m2svid" / "m2svid" / "prepare_vda_depth.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

_vda_mod = _load_prepare_vda_module()
run_vda_depth = _vda_mod.run_vda_depth
unload_vda_model = _vda_mod.unload_vda_model

# Note: M2SVid / sgm heavy imports are intentionally lazy-loaded inside step2_run_m2svid
# to avoid pulling in kornia, sgm, and related code until the user actually runs Step 2.
# This saves startup time and VRAM.

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
# Recommended defaults for StereoFaster (VDA / DepthCrafter + M2SVid) use case:
# - Video-Depth-Anything-Large (vitl - 381.8M) : Highest detail & accuracy
# - Video-Depth-Anything-Base (vitb - 113.1M)  : ~2x faster, lightweight
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
        return {"vda": 2, "warp": 1, "vae": 2, "gen_chunk": 2}
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if vram_gb >= 90: # 96GB class (e.g., A100 96GB/Mac 128GB)
        return {"vda": 32, "warp": 32, "vae": 14, "gen_chunk": 14, "gen_batch": 4}
    if vram_gb >= 45: # 48GB class (e.g., RTX 6000 Ada / A6000)
        return {"vda": 16, "warp": 32, "vae": 14, "gen_chunk": 14, "gen_batch": 3}
    if vram_gb >= 30: # 32GB class (e.g., RTX 5090 / V100 32GB)
        return {"vda": 12, "warp": 32, "vae": 14, "gen_chunk": 14, "gen_batch": 3}
    if vram_gb >= 22: # 24GB class (e.g., RTX 3090 / 4090)
        return {"vda": 8, "warp": 32, "vae": 14, "gen_chunk": 14, "gen_batch": 2}
    if vram_gb >= 11: # 12GB class (e.g., RTX 3060 / 4070)
        return {"vda": 4, "warp": 8, "vae": 4, "gen_chunk": 5, "gen_batch": 1}
        
    # Fallback for <12GB (e.g., 8GB cards)
    return {"vda": 2, "warp": 2, "vae": 2, "gen_chunk": 3, "gen_batch": 1}

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
        f"| VRAM Profile: VDA={_VRAM_DEFAULTS['vda']}, Warp={_VRAM_DEFAULTS['warp']}, VAE={_VRAM_DEFAULTS['vae']}"
    )

DEFAULT_M2SVID_CONFIG = str(
    PROJECT_ROOT / "m2svid" / "configs" / ("m2svid.yaml" if _FAST_GPU else "m2svid_no_fullatten.yaml")
)
DEFAULT_M2SVID_CKPT = str(PROJECT_ROOT / "m2svid" / "ckpts" / "m2svid_weights.pt")

# In-memory state between steps (simple for single-user local UI)
STATE = {
    "input_video": None,
    "depth_npz": None,
    "vda_depth": None,          # numpy array (T,H,W)
    "original_fps": 24.0,
    "original_hw": (720, 1280),
}
GLOBAL_CANCEL = False
PIPELINE_STOP_EVENT = threading.Event()
SIDECAR_CONFIG_PATH = PROJECT_ROOT / "pipeline_sidecar.json"

def trigger_cancel():
    global GLOBAL_CANCEL
    GLOBAL_CANCEL = True
    PIPELINE_STOP_EVENT.set()
    return "Stopping gracefully... (waiting for active task to finish and release CUDA)"

def get_gpu_info() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return f"GPU: {name} | VRAM: {reserved:.1f} / {total:.1f} GB"
    return "GPU: CPU only"


def get_icdepth_res_advisory(res: int) -> str:
    """Generates dynamic advisory/warning message for ICDepth resolution based on detected GPU VRAM."""
    try:
        res = int(res)
    except Exception:
        res = 720

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_label = f"<b>{gpu_name}</b> ({vram_gb:.1f} GB VRAM)"
    else:
        gpu_name = "CPU"
        vram_gb = 0.0
        gpu_label = "<b>CPU Only (No CUDA)</b>"

    def _box(bg, border, text_color, icon, title, desc):
        return (
            f"<div style='margin-top: 6px; padding: 10px 14px; background: {bg}; "
            f"border-left: 4px solid {border}; border-radius: 6px; font-size: 0.9em; "
            f"color: {text_color}; line-height: 1.45;'>"
            f"<div style='font-weight: 700; margin-bottom: 2px;'>{icon} {title}</div>"
            f"<div>{desc}</div>"
            f"</div>"
        )

    if vram_gb >= 28.0:  # 32GB - 96GB (RTX 5090, RTX 6000 Ada, A100)
        if res <= 1024:
            return _box(
                "rgba(16, 185, 129, 0.12)", "#10B981", "#D1FAE5",
                "🟢", f"Safe & Ultra-Fast on {gpu_label}",
                f"{res}p runs smoothly within high-capacity memory. Excellent speed and headroom."
            )
        elif res <= 1536:
            return _box(
                "rgba(16, 185, 129, 0.12)", "#10B981", "#D1FAE5",
                "🟢", f"Native High-Res Supported on {gpu_label}",
                f"{res}p fits well in your 32GB+ VRAM. Tiled VAE is optional."
            )
        else:  # 1920 (1080p)
            return _box(
                "rgba(59, 130, 246, 0.15)", "#3B82F6", "#DBEAFE",
                "🚀", f"Native 1080p ({res}p) Unlocked on {gpu_label}",
                f"Full 1:1 1080p DiT spatio-temporal inference active. For long video clips, enable <b>Tiled VAE Decoding</b> to maintain comfortable headroom."
            )

    elif vram_gb >= 20.0:  # 24GB (RTX 3090, RTX 4090)
        if res <= 1024:
            return _box(
                "rgba(16, 185, 129, 0.12)", "#10B981", "#D1FAE5",
                "🟢", f"Optimal for {gpu_label}",
                f"{res}p is the recommended high-quality setting for 24GB VRAM (~12–16 GB peak)."
            )
        elif res <= 1280:
            return _box(
                "rgba(245, 158, 11, 0.15)", "#F59E0B", "#FEF3C7",
                "🟡", f"High VRAM Load on {gpu_label}",
                f"{res}p consumes ~18–21 GB VRAM. Ensure <b>Tiled VAE Decoding</b> is enabled."
            )
        else:  # > 1280 (up to 1920 / 1080p)
            return _box(
                "rgba(239, 68, 68, 0.15)", "#EF4444", "#FEE2E2",
                "⚠️", f"Tight VRAM on 24GB ({res}p)",
                f"Native 1080p DiT spatio-temporal attention can peak near the 24GB limit (~22–24 GB). Set Window Size ≤ 61 and check <b>Tiled VAE Decoding</b> to avoid CUDA OOM."
            )

    elif vram_gb >= 11.0:  # 12GB - 16GB (RTX 3060, RTX 4060, RTX 4070)
        if res <= 720:
            return _box(
                "rgba(16, 185, 129, 0.12)", "#10B981", "#D1FAE5",
                "🟢", f"Recommended Sweet Spot for {gpu_label}",
                f"<b>{res}p</b> uses ~7–9 GB peak VRAM. StereoFaster will <b>automatically upscale the output depth map to full 1080p source dimensions</b> with clean sub-pixel interpolation."
            )
        elif res <= 1024:
            return _box(
                "rgba(245, 158, 11, 0.15)", "#F59E0B", "#FEF3C7",
                "🟡", f"Warning: Near 12GB Limit for {gpu_label}",
                f"<b>{res}p</b> consumes ~10.5–12 GB VRAM. You <b>MUST enable Tiled VAE Decoding</b> and reduce Window Size (≤ 61) to prevent CUDA Out of Memory (OOM)."
            )
        else:  # > 1024 (e.g. 1080p / 1280–1920)
            return _box(
                "rgba(239, 68, 68, 0.18)", "#EF4444", "#FEE2E2",
                "🔴", f"Critical OOM / Crash Hazard for {gpu_label}",
                f"Native DiT processing at <b>{res}p</b> requires <b>>20–28 GB VRAM</b>. Running this on a 12GB GPU will trigger <b>CUDA Out of Memory</b> or a <b>Windows GPU Timeout crash (VIDEO_TDR_FAILURE 0x116 BSOD)</b>.<br/>"
                f"👉 <b>Recommended:</b> Leave at <b>720p</b>. Your final depth map and 3D video are still rendered at full 1080p quality!"
            )
    else:  # <11GB or CPU
        if res <= 576:
            return _box(
                "rgba(245, 158, 11, 0.15)", "#F59E0B", "#FEF3C7",
                "🟡", f"Low VRAM Mode ({gpu_label})",
                f"{res}p requires strict VAE tiling and short window sizes."
            )
        else:
            return _box(
                "rgba(239, 68, 68, 0.18)", "#EF4444", "#FEE2E2",
                "🔴", f"OOM Inevitable on {gpu_label}",
                f"{res}p requires far more memory than available. Please reduce resolution to 480p or 576p."
            )


VALID_SOURCE_VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv', '.wmv')


def resolve_source_video_path(input_video: str) -> Optional[Path]:
    """Resolve user input string/stem/path strictly to an existing video file in SOURCE_DIR or on disk.

    Guarantees that non-video sidecar files (e.g. {stem}.credits.txt or .json) are NEVER selected as video files.
    """
    if not input_video:
        return None
    raw = str(input_video).strip()
    p_direct = Path(raw)

    # 1. Direct path exists and has video extension
    if p_direct.is_file() and p_direct.suffix.lower() in VALID_SOURCE_VIDEO_EXTS:
        return p_direct

    # 2. Extract base stem, stripping .credits or other auxiliary extensions if accidentally included
    stem = p_direct.stem
    if stem.endswith('.credits'):
        stem = stem[:-8]

    # 3. Check known video extensions in SOURCE_DIR
    for ext in VALID_SOURCE_VIDEO_EXTS:
        cand = SOURCE_DIR / f"{stem}{ext}"
        if cand.is_file():
            return cand

    # 4. Fallback search among matching stems in SOURCE_DIR with valid video extensions
    for cand in SOURCE_DIR.glob(f"{stem}.*"):
        if cand.is_file() and cand.suffix.lower() in VALID_SOURCE_VIDEO_EXTS:
            return cand

    return None


def get_source_video_list():
    """Return list of video stems available in source_videos folder."""
    if not SOURCE_DIR.exists():
        return []
    vids = sorted([p.stem for p in SOURCE_DIR.iterdir() 
                   if p.is_file() and p.suffix.lower() in VALID_SOURCE_VIDEO_EXTS])
    return vids


DEFAULT_DEPTH_MODEL = "depth-anything/Video-Depth-Anything-Large"

MODEL_SUFFIX_MAP = {
    "depth-anything/Video-Depth-Anything-Large": "_VDAL",
    "depth-anything/Video-Depth-Anything-Base": "_VDAB",
    "Video-Depth-Anything-Large": "_VDAL",
    "Video-Depth-Anything-Base": "_VDAB",
    "Video-Depth-Anything": "_VDA",
    "tencent/DepthCrafter": "_DC",
    "ICDepth": "_ICD",
    "Wan2.1-depth": "_ICD",
}


def get_model_suffix(model_name: str) -> str:
    """Return the suffix for a given depth model identifier."""
    return MODEL_SUFFIX_MAP.get(model_name, "")


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
    """Load video, depth npz, and depth mp4 paths for a stem selected from dropdown."""
    if stem is None or (isinstance(stem, dict) and not stem.get("path")):
        return None, None, None
    if isinstance(stem, dict):
        stem = stem.get("orig_name", "") or ""
    if not stem:
        return None, None, None
    video_path = SOURCE_DIR / f"{stem}.mp4"
    if not video_path.exists():
        for ext in ('.mov', '.avi', '.mkv'):
            alt = SOURCE_DIR / f"{stem}{ext}"
            if alt.exists():
                video_path = alt
                break
    
    depth_npz = None
    depth_mp4 = None
    
    # Priority order: Hybrid Depth (Fused 1080p) -> DepthCrafter (Primary) -> FlashDepth -> standard/legacy -> VDA Large -> VDA -> VDA Base
    base_stems = [
        f"{stem}_hybrid_depth",
        f"{stem}_DC_depth",
        f"{stem}_ICD_depth",
        f"{stem}_FDF_depth",
        f"{stem}_FDL_depth",
        f"{stem}_FD_depth",
        f"{stem}_depth",
        f"{stem}_VDAL_depth",
        f"{stem}_VDA_depth",
        f"{stem}_VDAB_depth",
    ]
    if DEPTH_DIR.exists():
        for p in sorted(DEPTH_DIR.glob(f"{stem}_*_depth.*")):
            b_name = p.name.rsplit(".", 1)[0]
            if b_name not in base_stems:
                base_stems.append(b_name)
                
    for b_stem in base_stems:
        npz = DEPTH_DIR / f"{b_stem}.npz"
        mp4 = DEPTH_DIR / f"{b_stem}.mp4"
        if depth_npz is None and npz.exists():
            depth_npz = npz
        if depth_mp4 is None and mp4.exists():
            depth_mp4 = mp4
        if depth_npz and depth_mp4:
            break

    # Auto-convert if depth MP4 exists but NPZ is missing (e.g. uploaded via FileBrowser or kept for space saving)
    if depth_mp4 and not depth_npz:
        target_npz = depth_mp4.parent / f"{depth_mp4.name.rsplit('.', 1)[0]}.npz"
        try:
            SF_LOG.info(f"Auto-converting orphan depth video {depth_mp4.name} -> {target_npz.name}")
            convert_depth_video_to_npz(str(depth_mp4), str(target_npz))
            if target_npz.exists():
                depth_npz = target_npz
        except Exception as e:
            SF_LOG.error(f"Failed to auto-convert {depth_mp4.name} to npz: {e}")
    
    return (
        str(video_path) if video_path.exists() else None,
        str(depth_npz) if depth_npz else None,
        str(depth_mp4) if depth_mp4 else None
    )


def clear_cuda():
    if torch.cuda.is_available():
        SF_LOG.debug("Clearing CUDA cache")
        torch.cuda.empty_cache()
    import gc
    gc.collect()


def is_cuda_oom(e: Exception) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls and isinstance(e, oom_cls):
        return True
    oom_cls_root = getattr(torch, "OutOfMemoryError", None)
    if oom_cls_root and isinstance(e, oom_cls_root):
        return True
    err_str = str(e).lower()
    return (
        "out of memory" in err_str
        or "cuda oom" in err_str
        or "device not ready" in err_str
        or "driver error" in err_str
    )


# =============================================================================
# STEP 1: Depth Estimation (DepthCrafter & Video Depth Anything)
# =============================================================================

def step1_run_depthcrafter(video_path: str, process_res: int, guidance_scale: float, inference_steps: int, window_size: int, overlap: int, attn_slicing: str = "Auto (Adapts to GPU VRAM)", cpu_offload: str = "Auto (Adapts to GPU VRAM)", progress=gr.Progress(track_tqdm=True)) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    if not video_path:
        return "No video selected.", None, None, None
    
    vp = resolve_source_video_path(video_path)
    if vp is None or not vp.exists():
        return f"File not found: {video_path}", None, None, None
        
    stem = vp.stem
    out_npz = DEPTH_DIR / f"{stem}_DC_depth.npz"
    out_mp4 = DEPTH_DIR / f"{stem}_DC_depth.mp4"
    
    try:
        unload_m2svid_model()
        from m2svid.prepare_depthcrafter import run_depthcrafter_depth
        import cv2
        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0: fps = 30.0
        cap.release()
        depth = run_depthcrafter_depth(
            str(vp),
            process_res=process_res,
            guidance_scale=guidance_scale,
            num_inference_steps=inference_steps,
            window_size=window_size,
            overlap=overlap,
            attn_slicing=attn_slicing,
            cpu_offload=cpu_offload,
            progress=progress
        )
        
        if depth.shape[1:] != (h, w):
            depth = np.stack([cv2.resize(d, (w, h), cv2.INTER_CUBIC) for d in depth])
            
        save_m2svid_compatible_npz(depth, str(out_npz))
        _create_depth_preview_video(depth, str(out_mp4), fps)
        STATE["depth_npz"] = str(out_npz)
        return "DepthCrafter estimation complete!", str(out_mp4), str(out_npz), str(out_npz)
    except Exception as e:
        clear_cuda()
        try:
            from m2svid.prepare_depthcrafter import unload_depthcrafter_model
            unload_depthcrafter_model()
        except Exception:
            pass
        clear_cuda()
        is_oom = is_cuda_oom(e)
        if is_oom:
            msg = (
                f"❌ CUDA Out of Memory (OOM) on '{vp.name}'.\n"
                f"💡 Recommendations for 24GB/32GB GPUs (RTX 5090/4090/3090):\n"
                f"  • Reduce Window Size (try 60 or 40 instead of {window_size})\n"
                f"  • Set CPU Offload to 'Model'\n"
                f"  • Lower Resolution (e.g. 1080 -> 720)\n"
                f"  • Keep Attention Slicing on 'Auto' or 'Enabled'"
            )
            SF_LOG.error(f"[DepthCrafter] {msg}\nDetail: {e}")
            return msg, None, None, None
        SF_LOG.error(f"DepthCrafter error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {e}", None, None, None
    finally:
        try:
            from m2svid.prepare_depthcrafter import unload_depthcrafter_model
            unload_depthcrafter_model()
        except Exception:
            pass
        clear_cuda()


def get_depth_npz_list():
    """Return all available depth npz files in DEPTH_DIR."""
    if not DEPTH_DIR.exists():
        return []
    return sorted([p.name for p in DEPTH_DIR.glob("*_depth.npz")])


def step1_run_hybrid_fusion(
    video_stem: str,
    dc_npz_choice: str,
    fd_npz_choice: str,
    fusion_mode: str = "RGB-Guided Super-Resolution (DC -> 1080p) [Recommended]",
    particle_boost: float = 0.45,
    min_circ: float = 0.35,
    min_solidity: float = 0.60,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Any]:
    """Run Guided 1080p Hybrid Depth Fusion:
    Upscales 480p/720p DepthCrafter particles using 1080p RGB edge guidance and fuses them
    with FlashDepth 1080p subject/background depth map using perspective diameter scaling.
    """
    if not video_stem:
        return "❌ Error: No source video selected.", None, None, None, gr.update()

    vp = SOURCE_DIR / (video_stem if "." in video_stem else f"{video_stem}.mp4")
    if not vp.exists():
        for ext in (".mp4", ".mkv", ".mov", ".avi", ".webm"):
            candidate = SOURCE_DIR / f"{video_stem}{ext}"
            if candidate.exists():
                vp = candidate
                break
    if not vp.exists():
        return f"❌ Source video not found: {vp}", None, None, None, gr.update()

    clean_stem = vp.stem

    # 1. Resolve DepthCrafter NPZ
    dc_path = None
    if dc_npz_choice and dc_npz_choice != "[Auto-Detect from Video]":
        candidate = DEPTH_DIR / dc_npz_choice
        if candidate.exists():
            dc_path = candidate
    if dc_path is None:
        dc_candidates = [
            DEPTH_DIR / f"{clean_stem}_DC_depth.npz",
            DEPTH_DIR / f"{clean_stem}_ICD_depth.npz",
            DEPTH_DIR / f"{clean_stem}_depth.npz",
        ]
        for c in dc_candidates:
            if c.exists():
                dc_path = c
                break
        if dc_path is None:
            found = list(DEPTH_DIR.glob(f"{clean_stem}*DC*.npz"))
            if found:
                dc_path = found[0]

    if dc_path is None:
        return (
            f"❌ DepthCrafter depth array (.npz) not found for '{clean_stem}'.\n"
            f"Please run Step 1 (DepthCrafter) at 480p/720p first to extract particle & motion layers.",
            None, None, None, gr.update()
        )

    # 2. Resolve FlashDepth NPZ
    fd_path = None
    if fd_npz_choice and fd_npz_choice != "[Auto-Detect from Video]":
        candidate = DEPTH_DIR / fd_npz_choice
        if candidate.exists():
            fd_path = candidate
    if fd_path is None:
        fd_candidates = [
            DEPTH_DIR / f"{clean_stem}_FDF_depth.npz",
            DEPTH_DIR / f"{clean_stem}_FDL_depth.npz",
            DEPTH_DIR / f"{clean_stem}_FD_depth.npz",
            DEPTH_DIR / f"{clean_stem}_FDS_depth.npz",
        ]
        for c in fd_candidates:
            if c.exists():
                fd_path = c
                break
        if fd_path is None:
            found = list(DEPTH_DIR.glob(f"{clean_stem}*FD*.npz"))
            if found:
                fd_path = found[0]

    if fd_path is None and "RGB-Guided" not in fusion_mode:
        return (
            f"❌ FlashDepth depth array (.npz) not found for '{clean_stem}'.\n"
            f"Please run Step 1 (Alt) (FlashDepth) at 1080p first, or switch Mode to 'RGB-Guided Super-Resolution'.",
            None, None, None, gr.update()
        )

    out_npz = DEPTH_DIR / f"{clean_stem}_hybrid_depth.npz"
    out_mp4 = DEPTH_DIR / f"{clean_stem}_hybrid_depth.mp4"

    try:
        try:
            from m2svid.hybrid_depth import run_hybrid_depth_fusion
        except ImportError:
            from m2svid.m2svid.hybrid_depth import run_hybrid_depth_fusion

        fused, fps = run_hybrid_depth_fusion(
            video_path=str(vp),
            dc_depth_path=str(dc_path),
            fd_depth_path=str(fd_path) if fd_path else None,
            output_npz_path=str(out_npz),
            output_mp4_path=str(out_mp4),
            fusion_mode=fusion_mode,
            particle_boost=float(particle_boost),
            min_circularity=float(min_circ),
            min_solidity=float(min_solidity),
            progress=progress,
        )

        STATE["depth_npz"] = str(out_npz)
        msg = (
            f"✅ 1080p Guided Hybrid Depth Fusion complete!\n"
            f"📁 Output: {out_npz.name} ({fused.shape[0]} frames @ {fused.shape[2]}x{fused.shape[1]})\n"
            f"🎬 DC Particle Source: {dc_path.name}\n"
            f"⚡ FD Base Source: {fd_path.name}\n"
            f"Ready for Step 2 (M2SVid Stereography)!"
        )
        new_depth_dropdown = gr.update(
            choices=[""] + get_depth_video_list(),
            value=f"{clean_stem}_hybrid_depth"
        )
        return msg, str(out_mp4), str(out_npz), str(out_npz), new_depth_dropdown
    except Exception as e:
        SF_LOG.exception(f"Hybrid Depth Fusion failed: {e}")
        return f"❌ Hybrid Fusion failed: {e}", None, None, None, gr.update()


def step1_run_vda_depth(
    input_video: str,
    encoder: str = "Video-Depth-Anything-Large (vitl - 381.8M, Highest Quality)",
    mode: str = "Sliding Window (Optimal Quality & Alignment)",
    input_size: int = 518,
    max_res: str = "1280 (Recommended)",
    target_fps: str = "-1 (Native Source FPS)",
    max_len: int = -1,
    fp32: bool = False,
    global_norm: bool = True,
    invert_depth: bool = False,
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[str, str, str, Optional[str]]:
    """
    Run Video Depth Anything (VDA Base or Large) on the uploaded/selected video.
    Produces both a depth.npz (for M2SVid warping) and a visual depth.mp4 video.
    Returns: (status, depth_preview_video, depth_npz_path, depth_state_for_step2)
    """
    if input_video is None or input_video == "":
        return "Please upload or select a video first.", None, None, None
        
    vp = resolve_source_video_path(input_video)
    if vp is None or not vp.exists():
        return f"Video {input_video} not found in source_videos.", None, None, None
    input_video = str(vp)

    enc_code = "vitl" if "large" in str(encoder).lower() or "vitl" in str(encoder).lower() else "vitb"
    model_suffix = "_VDAL" if enc_code == "vitl" else "_VDAB"
    model_type_str = f"Video Depth Anything ({'Large' if enc_code == 'vitl' else 'Base'})"

    clean_res = 1280
    if isinstance(max_res, str):
        for part in max_res.split():
            if part.isdigit():
                clean_res = int(part)
                break
    elif isinstance(max_res, (int, float)):
        clean_res = int(max_res)
        
    clean_fps = -1.0
    if isinstance(target_fps, str):
        for part in target_fps.split():
            try:
                clean_fps = float(part)
                break
            except ValueError:
                pass
    elif isinstance(target_fps, (int, float)):
        clean_fps = float(target_fps)
        
    clean_mode = "streaming" if "stream" in str(mode).lower() else "sliding_window"

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
    stem = Path(video_path).stem

    try:
        unload_m2svid_model()
        depth, fps = run_vda_depth(
            video_path=video_path,
            encoder=enc_code,
            input_size=int(input_size),
            max_res=clean_res,
            mode=clean_mode,
            target_fps=clean_fps,
            max_len=int(max_len),
            fp32=fp32,
            global_norm=global_norm,
            invert_depth=invert_depth,
            progress=progress
        )

        STATE["vda_depth"] = depth
        STATE["original_fps"] = fps

        # Save compatible npz and visual mp4 to depthmaps_videos
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

        progress(1.0, desc="VDA Depth estimation complete!")
        unload_vda_model()
        clear_cuda()

        status = f"✅ {model_type_str} depth computed ({depth.shape[0]} frames). Ready for Step 2."
        return status, str(depth_vis_path), str(depth_npz), str(depth_npz)

    except Exception as e:
        clear_cuda()
        try:
            unload_vda_model()
        except Exception:
            pass
        clear_cuda()
        if "stopped by user" in str(e).lower():
            return "🛑 Estimation stopped by user.", None, None, None
        is_oom = is_cuda_oom(e)
        if is_oom:
            msg = (
                f"❌ CUDA Out of Memory (OOM) on '{Path(video_path).name}'.\n"
                f"💡 Recommendations:\n"
                f"  • Reduce Input Size (e.g. 518 -> 392)\n"
                f"  • Lower Max Resolution (e.g. 1280 -> 720)\n"
                f"  • Use Streaming mode or VDA-Base"
            )
            SF_LOG.error(f"[{model_type_str}] {msg}\nDetail: {e}")
            return msg, None, None, None
        import traceback
        traceback.print_exc()
        return f"❌ Error in VDA depth step: {str(e)}", None, None, None


def step1_run_flashdepth(
    input_video: str,
    model_name: str = "FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
    max_res: str = "1280 (Recommended)",
    target_fps: str = "-1 (Native Source FPS)",
    max_len: int = -1,
    global_norm: bool = True,
    invert_depth: bool = False,
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Run FlashDepth (Real-Time 2K Streaming) on the uploaded/selected video.
    Produces both a depth.npz (for M2SVid warping) and a visual depth.mp4 video.
    Returns: (status, depth_preview_video, depth_npz_path, depth_state_for_step2)
    """
    if input_video is None or input_video == "":
        return "Please upload or select a video first.", None, None, None
        
    vp = resolve_source_video_path(input_video)
    if vp is None or not vp.exists():
        return f"Video {input_video} not found in source_videos.", None, None, None
    input_video = str(vp)

    clean_mod = str(model_name).lower()
    suffix = "_FDL" if "large" in clean_mod or "vitl" in clean_mod else ("_FDS" if "small" in clean_mod or "vits" in clean_mod else "_FDF")
    model_type_str = f"FlashDepth ({'Large' if suffix == '_FDL' else ('Small' if suffix == '_FDS' else 'Full 2K')})"

    clean_res = 1280
    if isinstance(max_res, str):
        for part in max_res.split():
            if part.isdigit():
                clean_res = int(part)
                break
    elif isinstance(max_res, (int, float)):
        clean_res = int(max_res)
        
    clean_fps = -1.0
    if isinstance(target_fps, str):
        for part in target_fps.split():
            try:
                clean_fps = float(part)
                break
            except ValueError:
                pass
    elif isinstance(target_fps, (int, float)):
        clean_fps = float(target_fps)

    SF_LOG.info(f"Starting {model_type_str} depth estimation")
    progress(0, desc=f"Preparing {model_type_str} streaming depth estimation...")

    video_path = input_video
    if isinstance(input_video, dict) and "name" in input_video:
        src_path = input_video["name"]
        dst_path = SOURCE_DIR / Path(src_path).name
        shutil.copy(src_path, str(dst_path))
        video_path = str(dst_path)
        SF_LOG.info(f"Uploaded video copied to {dst_path}")

    STATE["input_video"] = video_path
    stem = Path(video_path).stem

    try:
        unload_m2svid_model()
        depth, fps = run_flashdepth(
            video_path=video_path,
            model_name=model_name,
            max_res=clean_res,
            target_fps=clean_fps,
            max_len=int(max_len),
            global_norm=global_norm,
            invert_depth=invert_depth,
            progress=progress
        )

        STATE["flashdepth_depth"] = depth
        STATE["original_fps"] = fps

        # Save compatible npz and visual mp4 to depthmaps_videos
        depth_npz = DEPTH_DIR / f"{stem}{suffix}_depth.npz"
        depth_mp4 = DEPTH_DIR / f"{stem}{suffix}_depth.mp4"

        save_m2svid_compatible_npz(depth, str(depth_npz))
        _create_depth_preview_video(depth, str(depth_mp4), fps)

        STATE["depth_npz"] = str(depth_npz)
        SF_LOG.info(f"Saved FlashDepth files: {depth_npz} and {depth_mp4}")

        # Also save to temp dir for immediate Step 2 use
        out_dir = Path(tempfile.mkdtemp(prefix="stereofaster_depth_"))
        temp_depth_npz = out_dir / "depth.npz"
        save_m2svid_compatible_npz(depth, str(temp_depth_npz))

        depth_vis_path = out_dir / "depth_preview.mp4"
        _create_depth_preview_video(depth, str(depth_vis_path), fps)

        progress(1.0, desc="FlashDepth estimation complete!")
        try:
            unload_flashdepth_model()
        except Exception:
            pass
        clear_cuda()

        status = f"✅ {model_type_str} depth computed ({depth.shape[0]} frames). Ready for Step 2."
        return status, str(depth_vis_path), str(depth_npz), str(depth_npz)

    except Exception as e:
        clear_cuda()
        try:
            unload_flashdepth_model()
        except Exception:
            pass
        clear_cuda()
        if "stopped by user" in str(e).lower():
            return "🛑 Estimation stopped by user.", None, None, None
        is_oom = is_cuda_oom(e)
        if is_oom:
            msg = (
                f"❌ CUDA Out of Memory (OOM) on '{Path(video_path).name}'.\n"
                f"💡 Recommendations:\n"
                f"  • Lower Max Resolution (e.g. 1280 -> 720)\n"
                f"  • Use FlashDepth-S instead of Full/Large"
            )
            SF_LOG.error(f"[{model_type_str}] {msg}\nDetail: {e}")
            return msg, None, None, None
        import traceback
        traceback.print_exc()
        return f"❌ Error in FlashDepth step: {str(e)}", None, None, None


def step1_run_icdepth(
    input_video: str,
    max_res: int = 720,
    window_size: int = 81,
    overlap: int = 32,
    steps: int = 5,
    cfg_scale: float = 5.0,
    tiled: bool = False,
    void_title_recovery: bool = True,
    volumetric_splash_recovery: bool = False,
    credit_pop: bool = True,
    credit_text_hint: str = "",
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Run ICDepth (Wan2.1 In-Context DiT) depth estimation on the selected video.
    Produces both a depth.npz (for M2SVid warping) and a visual depth.mp4 video.
    Returns: (status, depth_preview_video, depth_npz_path, depth_state_for_step2)
    """
    if input_video is None or input_video == "":
        return "Please upload or select a video first.", None, None, None

    vp = resolve_source_video_path(input_video)
    if vp is None or not vp.exists():
        return f"File not found: {input_video}", None, None, None

    stem = vp.stem
    out_npz = DEPTH_DIR / f"{stem}_ICD_depth.npz"
    out_mp4 = DEPTH_DIR / f"{stem}_ICD_depth.mp4"

    try:
        unload_m2svid_model()
        from m2svid.prepare_icdepth import run_icdepth_depth, unload_icdepth_model
        import cv2

        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        depth, out_fps = run_icdepth_depth(
            str(vp),
            max_res=int(max_res),
            window_size=int(window_size),
            overlap=int(overlap),
            num_inference_steps=int(steps),
            cfg_scale=float(cfg_scale),
            tiled=tiled,
            void_title_recovery=bool(void_title_recovery),
            volumetric_splash_recovery=bool(volumetric_splash_recovery),
            credit_pop=bool(credit_pop),
            credit_text_hint=str(credit_text_hint or ""),
            progress=progress,
        )

        if depth.shape[1:] != (h, w):
            interp = cv2.INTER_AREA if (w <= depth.shape[2] and h <= depth.shape[1]) else cv2.INTER_LINEAR
            depth = np.stack([cv2.resize(d, (w, h), interpolation=interp) for d in depth])

        save_m2svid_compatible_npz(depth, str(out_npz))
        _create_depth_preview_video(depth, str(out_mp4), out_fps)
        STATE["depth_npz"] = str(out_npz)

        # Also save to temp dir for immediate Step 2 use
        out_dir = Path(tempfile.mkdtemp(prefix="stereofaster_depth_"))
        temp_depth_npz = out_dir / "depth.npz"
        save_m2svid_compatible_npz(depth, str(temp_depth_npz))
        depth_vis_path = out_dir / "depth_preview.mp4"
        _create_depth_preview_video(depth, str(depth_vis_path), out_fps)

        try:
            unload_icdepth_model()
        except Exception:
            pass
        clear_cuda()

        status = f"✅ ICDepth (Wan2.1 In-Context DiT) depth computed ({depth.shape[0]} frames). Ready for Step 2."
        return status, str(depth_vis_path), str(out_npz), str(out_npz)

    except Exception as e:
        clear_cuda()
        try:
            from m2svid.prepare_icdepth import unload_icdepth_model
            unload_icdepth_model()
        except Exception:
            pass
        clear_cuda()
        if "stopped by user" in str(e).lower():
            return "🛑 Estimation stopped by user.", None, None, None
        is_oom = is_cuda_oom(e)
        if is_oom:
            msg = (
                f"⚠️ CUDA Out of Memory (OOM) on '{vp.name}'.\n"
                f"💡 Recommendations for 12GB GPUs (RTX 3060):\n"
                f"  • Reduce Max Resolution (try 720 or 576 instead of {max_res})\n"
                f"  • Reduce Window Size (try 61 or 41 instead of {window_size})\n"
                f"  • Ensure Tiled VAE Decoding is checked\n"
                f"  • Set CFG Scale to 1.0 for single-pass fast inference"
            )
            SF_LOG.error(f"[ICDepth] {msg}\nDetail: {e}")
            return msg, None, None, None
        SF_LOG.error(f"ICDepth error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {e}", None, None, None


def repair_existing_icdepth_void_title(
    input_video: str,
    void_title_recovery: bool = True,
    credit_pop: bool = True,
    credit_text_hint: str = "",
    volumetric_splash_recovery: bool = False,
    repair_strength: float = 1.0,
    credit_strength: float = 1.0,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, Optional[str], Optional[str], Optional[str], dict]:
    """Apply or pull back Black-Void & Title-Card 3D Recovery and On-Screen Credit Foreground Pop using untouched .raw.npz backup."""
    if input_video is None or input_video == "":
        return "Please select a source video first.", None, None, None, gr.update()

    vp = resolve_source_video_path(input_video)
    if vp is None or not vp.exists():
        return f"❌ Source video not found: {input_video}", None, None, None, gr.update()

    stem = vp.stem
    out_npz = DEPTH_DIR / f"{stem}_ICD_depth.npz"
    raw_npz = DEPTH_DIR / f"{stem}_ICD_depth.raw.npz"
    out_mp4 = DEPTH_DIR / f"{stem}_ICD_depth.mp4"

    if not out_npz.exists() and out_mp4.exists():
        convert_depth_video_to_npz(str(out_mp4), str(out_npz))

    if not out_npz.exists() and not raw_npz.exists():
        return (
            f"❌ Existing ICDepth map '{out_npz.name}' not found.\n"
            f"Please click '⚡ Estimate Depth with ICDepth' first.",
            None, None, None, gr.update()
        )

    try:
        import importlib
        import m2svid.prepare_icdepth
        importlib.reload(m2svid.prepare_icdepth)
        from m2svid.prepare_icdepth import (
            recover_black_void_and_titles,
            recover_volumetric_splash_aeration,
            detect_and_elevate_credits,
            load_clip_credit_sidecar,
            save_clip_credit_sidecar,
        )
        # Preserve an untouched raw backup on first repair so repairs can always be pulled back or reverted
        if not raw_npz.exists() and out_npz.exists():
            shutil.copy2(str(out_npz), str(raw_npz))
            SF_LOG.info(f"[ICDepth] Created untouched raw backup: {raw_npz}")

        raw_credits_npz = DEPTH_DIR / f"{stem}_raw_credits_depth.npz"
        src_npz = raw_credits_npz if raw_credits_npz.exists() else (raw_npz if raw_npz.exists() else out_npz)
        progress(0.1, desc=f"Loading untouched raw depth from {src_npz.name}...")

        cap = cv2.VideoCapture(str(vp))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        rgb_frames = []
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            rgb_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()

        with np.load(str(src_npz)) as data:
            key = "depth" if "depth" in data else list(data.keys())[0]
            depth_raw = data[key].astype(np.float32)

        strength_scene = float(np.clip(repair_strength, 0.0, 1.0))
        strength_credit = float(np.clip(credit_strength, 0.0, 1.0))
        repaired = depth_raw.copy()

        if strength_scene > 1e-4:
            # Scaled engine response: 80% maximum aggressiveness with gentle ease-in curve
            # Ensures a 0.05 repair step is subtle (~0.016) rather than a steep cliff jump
            eff_strength = 0.80 * (strength_scene ** 1.30)

            if void_title_recovery:
                progress(0.4, desc="Checking Black-Void & Title-Card 3D Recovery...")
                void_fixed = recover_black_void_and_titles(repaired, rgb_frames)
                repaired = (1.0 - eff_strength) * repaired + eff_strength * void_fixed

            if volumetric_splash_recovery:
                progress(0.5, desc=f"Applying Volumetric Splash & Water Aeration Recovery (strength={strength_scene:.2f})...")
                repaired = recover_volumetric_splash_aeration(repaired, rgb_frames, repair_strength=strength_scene)

        effective_hint = str(credit_text_hint or "").strip()
        if effective_hint:
            save_clip_credit_sidecar(str(vp), effective_hint)
        else:
            effective_hint = load_clip_credit_sidecar(str(vp))

        if credit_pop and strength_credit > 1e-4:
            progress(0.6, desc=f"Applying Typography / Credit Foreground Pop (strength={strength_credit:.2f})...")
            repaired = detect_and_elevate_credits(
                repaired,
                rgb_frames,
                credit_text_hint=effective_hint,
                repair_strength=strength_credit,
            )

        progress(0.8, desc="Saving .npz and .mp4 preview...")
        save_m2svid_compatible_npz(repaired, str(out_npz))
        _create_depth_preview_video(repaired, str(out_mp4), fps)
        STATE["depth_npz"] = str(out_npz)

        out_dir = Path(tempfile.mkdtemp(prefix="stereofaster_depth_"))
        depth_vis_path = out_dir / "depth_preview.mp4"
        shutil.copy(str(out_mp4), str(depth_vis_path))

        _prog(1.0, desc="Complete!")
        new_dep_dd = gr.update(choices=[""] + get_depth_video_list(), value=f"{stem}_ICD_depth")
        if strength_scene <= 1e-4 and strength_credit <= 1e-4:
            status_msg = f"⏪ Reverted {out_npz.name} to 100% untouched raw ICDepth ({repaired.shape[0]} frames)."
        else:
            status_msg = (
                f"✅ Applied ICDepth repair to {out_npz.name} [Scene/Void: {int(round(strength_scene * 100))}%, "
                f"Typography: {int(round(strength_credit * 100))}%] "
                f"({repaired.shape[0]} frames; raw backup preserved at {raw_npz.name})."
            )
        return (
            status_msg,
            str(depth_vis_path),
            str(out_npz),
            str(out_npz),
            new_dep_dd,
        )
    except Exception as e:
        SF_LOG.exception(f"Failed to apply/adjust ICDepth recovery: {e}")
        return f"❌ Error: {e}", None, None, None, gr.update()


def revert_icdepth_to_raw(
    input_video: str,
    progress=gr.Progress(track_tqdm=True),
):
    """One-click restore of {stem}_ICD_depth.npz and .mp4 from {stem}_ICD_depth.raw.npz."""
    res = repair_existing_icdepth_void_title(
        input_video=input_video,
        void_title_recovery=False,
        credit_pop=False,
        credit_text_hint="",
        volumetric_splash_recovery=False,
        repair_strength=0.0,
        credit_strength=0.0,
        progress=progress,
    )
    return (*res, gr.update(value=0.0), gr.update(value=0.0))


def check_icd_status_and_update_toggle(input_video: str):
    """Check if selected video has already been processed by ICDepth to enable toggle switch and set slider bounds."""
    if not input_video:
        return (
            gr.update(interactive=False, value=False, info="Select a video to check ICDepth status."),
            gr.update(maximum=100, value=0),
        )
    vp = resolve_source_video_path(input_video)
    if not vp or not vp.exists():
        return (
            gr.update(interactive=False, value=False, info="Video not found."),
            gr.update(maximum=100, value=0),
        )
    stem = vp.stem
    npz_path = DEPTH_DIR / f"{stem}_ICD_depth.npz"
    raw_path = DEPTH_DIR / f"{stem}_ICD_depth.raw.npz"
    has_icd = npz_path.exists() or raw_path.exists()

    frames = get_cached_video_frames(str(vp))
    total_frames = len(frames) if frames else 1
    max_f = max(1, total_frames - 1)
    init_val = min(36, max_f)

    if has_icd:
        return (
            gr.update(
                interactive=True,
                value=False,
                info="✅ Video processed by ICDepth. Toggle ON to activate Depth Editor.",
            ),
            gr.update(maximum=max_f, value=init_val),
        )
    else:
        return (
            gr.update(
                interactive=False,
                value=False,
                info="⚠️ Please run 'Estimate Depth with ICDepth' first to enable Depth Editor.",
            ),
            gr.update(maximum=max_f, value=init_val),
        )


def extract_mask_from_image_editor(editor_data: Any, target_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Extract a binary float mask (H, W) in [0.0, 1.0] from gr.ImageEditor dict."""
    if not editor_data or not isinstance(editor_data, dict):
        return None
    layers = editor_data.get("layers", [])
    if not layers:
        return None

    combined_mask = None
    for layer in layers:
        if layer is None:
            continue
        arr = np.array(layer)
        if arr.size == 0:
            continue
        if arr.ndim == 3:
            if arr.shape[-1] == 4:
                alpha = arr[..., 3]
                mask = (alpha > 5).astype(np.float32)
            else:
                mask = (arr.max(axis=-1) > 10).astype(np.float32)
        elif arr.ndim == 2:
            mask = (arr > 5).astype(np.float32)
        else:
            continue

        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = np.maximum(combined_mask, mask)

    if combined_mask is not None and target_shape is not None:
        if combined_mask.shape[:2] != target_shape[:2]:
            combined_mask = cv2.resize(combined_mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    return combined_mask


_VIDEO_FRAME_CACHE: Dict[str, List[np.ndarray]] = {}

def get_cached_video_frames(video_path: str) -> List[np.ndarray]:
    """Load and cache all video frames sequentially in memory for instantaneous, 100% reliable frame-accurate seeking."""
    global _VIDEO_FRAME_CACHE
    if not video_path:
        return []
    vp = Path(video_path).resolve()
    key = str(vp)
    if key in _VIDEO_FRAME_CACHE and len(_VIDEO_FRAME_CACHE[key]) > 0:
        return _VIDEO_FRAME_CACHE[key]

    cap = cv2.VideoCapture(str(vp))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()

    if len(_VIDEO_FRAME_CACHE) > 2:
        _VIDEO_FRAME_CACHE.clear()

    _VIDEO_FRAME_CACHE[key] = frames
    return frames


def build_editor_value_with_keyframe_mask(
    rgb_canvas: np.ndarray,
    frame_idx: int,
    keyframe_dict: Optional[Dict[int, Any]] = None,
) -> Dict[str, Any]:
    """Construct ImageEditor dict loading existing painted strokes for frame_idx if recorded."""
    layers_list = []
    kf_dict = dict(keyframe_dict or {})
    idx_int = int(frame_idx)
    mask = kf_dict.get(idx_int)
    if mask is not None and np.any(mask > 0.01):
        m_float = np.clip(mask.astype(np.float32), 0.0, 1.0)
        H_c, W_c = rgb_canvas.shape[:2]
        if m_float.shape[:2] != (H_c, W_c):
            m_float = cv2.resize(m_float, (W_c, H_c), interpolation=cv2.INTER_NEAREST)
        rgba_layer = np.zeros((H_c, W_c, 4), dtype=np.uint8)
        rgba_layer[..., 0] = 255  # Red
        rgba_layer[..., 1] = 34   # Green
        rgba_layer[..., 2] = 68   # Blue
        rgba_layer[..., 3] = (m_float * 128).astype(np.uint8)  # 50% translucent
        layers_list.append(rgba_layer)

        alpha_norm = (rgba_layer[..., 3:4] / 255.0).astype(np.float32)
        comp_rgb = (rgb_canvas.astype(np.float32) * (1.0 - alpha_norm) + rgba_layer[..., :3].astype(np.float32) * alpha_norm)
        composite_canvas = np.clip(comp_rgb, 0, 255).astype(np.uint8)
    else:
        composite_canvas = rgb_canvas

    return {
        "background": rgb_canvas,
        "layers": layers_list,
        "composite": composite_canvas,
    }


def load_reference_frame_for_editor(
    input_video: str,
    frame_idx: int = 36,
    show_markers: bool = False,
    keyframe_dict: Optional[Dict[int, Any]] = None,
    unlocked_choice: str = "🔒 All Layers Locked",
):
    """Load reference frame, dynamically analyze NPZ into 3 layers (all locked by default), and render canvas."""
    default_rows = [
        ["Layer 1", "Foreground (Characters / Foremost Objects)", "0.54 – 1.00", "0.0%", "🔒 Locked"],
        ["Layer 2", "Midground (Structures / Environment)", "0.21 – 0.54", "0.0%", "🔒 Locked"],
        ["Layer 3", "Background (Sky / Distant Scenery)", "0.00 – 0.21", "0.0%", "🔒 Locked"],
    ]
    default_choices = [
        "🔒 All Layers Locked",
        "Layer 1: Foreground (Characters / Foremost Objects)",
        "Layer 2: Midground (Structures / Environment)",
        "Layer 3: Background (Sky / Distant Scenery)",
    ]
    default_badge = "📌 **Keyframes recorded:** None *(Brush on canvas, then click 'Record Keyframe')*"
    default_banner = "🔒 **All layers are currently locked.** In the table above, all layers are protected. Select a layer from **'Unlock Layer for Editing'** to enable editing controls."

    if not input_video:
        return None, default_rows, gr.update(choices=default_choices, value="🔒 All Layers Locked"), gr.update(visible=False), default_banner, 0.18, 100, default_badge, []
    vp = resolve_source_video_path(input_video)
    if not vp or not vp.exists():
        return None, default_rows, gr.update(choices=default_choices, value="🔒 All Layers Locked"), gr.update(visible=False), default_banner, 0.18, 100, default_badge, []

    stem = vp.stem
    out_npz = DEPTH_DIR / f"{stem}_ICD_depth.npz"
    raw_npz = DEPTH_DIR / f"{stem}_ICD_depth.raw.npz"
    raw_credits_npz = DEPTH_DIR / f"{stem}_raw_credits_depth.npz"
    src_npz = raw_credits_npz if raw_credits_npz.exists() else (raw_npz if raw_npz.exists() else out_npz)

    if not src_npz.exists():
        return None, default_rows, gr.update(choices=default_choices, value="🔒 All Layers Locked"), gr.update(visible=False), f"⚠️ ICDepth map for '{stem}' not found. Please run ICDepth first.", 0.18, 100, default_badge, []

    from m2svid.object_layer_depth import (
        detect_scene_layer_hierarchy,
        layers_to_dataframe_data,
        load_layers_sidecar,
        analyze_scene_depth_anchors,
        render_annotated_reference_frame,
    )
    from m2svid.prepare_icdepth import load_clip_credit_sidecar

    frames = get_cached_video_frames(str(vp))
    if not frames:
        return None, default_rows, gr.update(choices=default_choices, value="🔒 All Layers Locked"), gr.update(visible=False), default_banner, 0.18, 100, default_badge, []

    total_frames = len(frames)
    idx = max(0, min(total_frames - 1, int(frame_idx)))
    bgr = frames[idx]

    if bgr is None:
        return None, default_rows, gr.update(choices=default_choices, value="🔒 All Layers Locked"), gr.update(visible=False), default_banner, 0.18, max(1, total_frames - 1), default_badge, []

    with np.load(str(src_npz)) as data:
        key = "depth" if "depth" in data else list(data.keys())[0]
        depth_all = data[key]
        idx_clamped = min(depth_all.shape[0] - 1, idx)
        d_frame = depth_all[idx_clamped].astype(np.float32)

    # Check sidecar for user-saved descriptions
    sidecar_data = load_layers_sidecar(str(vp))
    sidecar_layers = sidecar_data.get("layers") if sidecar_data else None
    credit_hint = load_clip_credit_sidecar(str(vp))

    # Dynamically analyze NPZ into semantic layers (including On-Screen Credits if present, all locked by default)
    layers = detect_scene_layer_hierarchy(d_frame, sidecar_layers=sidecar_layers, credit_text_hint=credit_hint)

    # Determine unlocked ID if any
    unlocked_id = None
    if unlocked_choice and unlocked_choice != "🔒 All Layers Locked":
        for l in layers:
            lid = l.get("id")
            if f"Layer {lid}" in unlocked_choice:
                unlocked_id = lid
                break
    elif sidecar_data and sidecar_data.get("unlocked_layer"):
        saved_lid = sidecar_data.get("unlocked_layer")
        for l in layers:
            if l.get("id") == saved_lid:
                unlocked_id = saved_lid
                unlocked_choice = f"Layer {saved_lid}: {l.get('desc', '')}"
                break

    table_data = layers_to_dataframe_data(layers, unlocked_id=unlocked_id)

    unlock_choices = ["🔒 All Layers Locked"]
    for l in layers:
        unlock_choices.append(f"Layer {l['id']}: {l.get('desc', '')}")

    controls_visible = (unlocked_id is not None)
    if controls_visible:
        banner_text = f"✏️ **Layer {unlocked_id}** is unlocked and active for editing! All other layers are protected as locked holdout mattes."
    else:
        banner_text = default_banner

    if show_markers:
        anchors_info = analyze_scene_depth_anchors(bgr, d_frame)
        annotated_bgr = render_annotated_reference_frame(
            bgr, d_frame, anchors_info, target_depth=0.18
        )
        rgb_canvas = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb_canvas = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    kf_dict = dict(keyframe_dict or {})
    if vp:
        masks_npz_path = vp.parent / f"{stem}.masks.npz"
        if masks_npz_path.exists():
            try:
                with np.load(str(masks_npz_path)) as m_data:
                    for k in m_data.files:
                        if k.isdigit():
                            k_int = int(k)
                            if k_int not in kf_dict:
                                kf_dict[k_int] = m_data[k].astype(np.float32)
            except Exception as e:
                SF_LOG.warning(f"Could not load keyframe masks from {masks_npz_path}: {e}")

    editor_value = build_editor_value_with_keyframe_mask(rgb_canvas, idx, kf_dict)

    kf_keys = sorted(list(kf_dict.keys()))
    if len(kf_keys) > 10:
        badge_text = f"📍 **Frame {idx}:** Painted strokes loaded for all {len(kf_keys)} frames on canvas"
    elif kf_keys:
        badge_text = f"📌 **Keyframes recorded:** `{kf_keys}` (Total: {len(kf_keys)})"
    else:
        badge_text = default_badge

    suggested_depth = 0.18
    if unlocked_id is not None:
        for l in layers:
            if l.get("id") == unlocked_id:
                suggested_depth = float(l.get("depth_val", 0.18))
                break

    return (
        editor_value,
        table_data,
        gr.update(choices=unlock_choices, value=(unlocked_choice if unlocked_choice in unlock_choices else "🔒 All Layers Locked")),
        gr.update(visible=controls_visible),
        banner_text,
        suggested_depth,
        max(1, total_frames - 1),
        badge_text,
        layers,
    )


def on_toggle_interactive_layering(
    is_enabled: bool,
    input_video: str,
    frame_idx: int,
    show_markers: bool = False,
    keyframe_dict: Optional[Dict[int, Any]] = None,
):
    """Show/hide Depth Editor section and load reference frame when enabled."""
    if not is_enabled:
        return (
            gr.update(visible=False),
            None,
            [],
            gr.update(),
            gr.update(visible=False),
            "",
            gr.update(),
            gr.update(),
            "",
            [],
        )

    editor_val, table_data, unlock_update, controls_update, banner_text, t_depth, max_f, badge_text, layers = load_reference_frame_for_editor(
        input_video, frame_idx, show_markers=show_markers, keyframe_dict=keyframe_dict, unlocked_choice="🔒 All Layers Locked"
    )
    return (
        gr.update(visible=True),
        editor_val,
        table_data,
        unlock_update,
        controls_update,
        banner_text,
        gr.update(maximum=max_f, value=min(frame_idx, max_f)),
        gr.update(value=t_depth),
        badge_text,
        layers,
    )


def on_select_unlocked_layer(
    selected_choice: str,
    current_df_data: Any,
    layers_state: Any,
    input_video: str = "",
):
    """Update table lock status and toggle editing controls based on unlock selection, and persist to sidecar."""
    updated_layers = list(layers_state or [])
    if not selected_choice or selected_choice == "🔒 All Layers Locked":
        updated_rows = []
        rows = current_df_data if isinstance(current_df_data, list) else (current_df_data.values if hasattr(current_df_data, "values") else [])
        for row in rows:
            r = list(row)
            if len(r) >= 5:
                r[4] = "🔒 Locked"
            updated_rows.append(r)

        for l in updated_layers:
            l["locked"] = True

        if input_video:
            vp = resolve_source_video_path(input_video)
            if vp:
                from m2svid.object_layer_depth import load_layers_sidecar, save_layers_sidecar
                sidecar_data = load_layers_sidecar(str(vp)) or {}
                sidecar_data["video"] = vp.name
                sidecar_data["masks_file"] = f"{vp.stem}.masks.npz"
                sidecar_data["layers"] = updated_layers
                sidecar_data["unlocked_layer"] = None
                save_layers_sidecar(str(vp), sidecar_data)

        banner_text = "🔒 **All layers are currently locked.** In the table above, all layers are protected. Select a layer from **'Unlock Layer for Editing'** to enable editing controls."
        return updated_rows, gr.update(visible=False), banner_text, 0.18, updated_layers

    # Identify unlocked layer ID
    unlocked_id = None
    for l in updated_layers:
        lid = l.get("id")
        if f"Layer {lid}" in selected_choice:
            unlocked_id = lid
            break
    if unlocked_id is None:
        for lid in range(1, 25):
            if f"Layer {lid}" in selected_choice:
                unlocked_id = lid
                break

    updated_rows = []
    rows = current_df_data if isinstance(current_df_data, list) else (current_df_data.values if hasattr(current_df_data, "values") else [])
    for row in rows:
        r = list(row)
        if len(r) >= 5:
            if f"Layer {unlocked_id}" in str(r[0]):
                r[4] = "✏️ Unlocked (Active)"
            else:
                r[4] = "🔒 Locked"
        updated_rows.append(r)

    suggested_depth = 0.18
    for l in updated_layers:
        if l.get("id") == unlocked_id:
            l["locked"] = False
            suggested_depth = float(l.get("depth_val", 0.18))
        else:
            l["locked"] = True

    if input_video:
        vp = resolve_source_video_path(input_video)
        if vp:
            from m2svid.object_layer_depth import load_layers_sidecar, save_layers_sidecar
            sidecar_data = load_layers_sidecar(str(vp)) or {}
            sidecar_data["video"] = vp.name
            sidecar_data["masks_file"] = f"{vp.stem}.masks.npz"
            sidecar_data["layers"] = updated_layers
            sidecar_data["unlocked_layer"] = unlocked_id
            save_layers_sidecar(str(vp), sidecar_data)

    banner_text = (
        f"✏️ **{selected_choice}** is unlocked and active for editing! "
        f"All other layers are protected as locked holdout mattes. "
        f"Brush over target objects on the canvas below to sculpt this layer."
    )
    return updated_rows, gr.update(visible=True), banner_text, suggested_depth, updated_layers


def on_target_depth_change(
    target_depth: float,
    unlocked_choice: str,
    layers_state: Any,
    input_video: str = "",
):
    """Save user-assigned target depth disparity to the active layer and persist to sidecar."""
    updated_layers = list(layers_state or [])
    unlocked_id = None
    if unlocked_choice and unlocked_choice != "🔒 All Layers Locked":
        for l in updated_layers:
            lid = l.get("id")
            if f"Layer {lid}" in unlocked_choice:
                unlocked_id = lid
                l["depth_val"] = float(round(target_depth, 3))
                break

    if input_video and unlocked_id is not None:
        vp = resolve_source_video_path(input_video)
        if vp:
            from m2svid.object_layer_depth import load_layers_sidecar, save_layers_sidecar
            sidecar_data = load_layers_sidecar(str(vp)) or {}
            sidecar_data["video"] = vp.name
            sidecar_data["masks_file"] = f"{vp.stem}.masks.npz"
            sidecar_data["layers"] = updated_layers
            sidecar_data["unlocked_layer"] = unlocked_id
            save_layers_sidecar(str(vp), sidecar_data)

    return updated_layers


def on_layer_table_change(table_data: Any, layers_state: Any, input_video: str = ""):
    """Keep layer state synchronized when the user edits description cells in the table and persist to sidecar."""
    updated_layers = list(layers_state or [])
    rows = table_data if isinstance(table_data, list) else (table_data.values if hasattr(table_data, "values") else [])

    for row in rows:
        if len(row) >= 2:
            lid_str = str(row[0])
            desc_str = str(row[1])
            for l in updated_layers:
                if f"Layer {l.get('id')}" in lid_str:
                    l["desc"] = desc_str
                    break

    unlock_choices = ["🔒 All Layers Locked"]
    for l in updated_layers:
        unlock_choices.append(f"Layer {l.get('id')}: {l.get('desc', '')}")

    if input_video:
        vp = resolve_source_video_path(input_video)
        if vp:
            from m2svid.object_layer_depth import load_layers_sidecar, save_layers_sidecar
            sidecar_data = load_layers_sidecar(str(vp)) or {}
            sidecar_data["video"] = vp.name
            sidecar_data["masks_file"] = f"{vp.stem}.masks.npz"
            sidecar_data["layers"] = updated_layers
            save_layers_sidecar(str(vp), sidecar_data)

    return updated_layers, gr.update(choices=unlock_choices)


def on_add_layer_action(current_df_data: Any, layers_state: Any, input_video: str = ""):
    """Dynamically append a new custom depth layer row to the table and persist immediately to sidecar."""
    layers = list(layers_state or [])
    new_id = len(layers) + 1
    new_layer = {
        "id": new_id,
        "name": f"Layer {new_id}: Custom Layer",
        "desc": f"Custom Depth Layer {new_id}",
        "range": "0.10 – 0.25",
        "depth_range": [0.10, 0.25],
        "depth_min": 0.10,
        "depth_max": 0.25,
        "depth_val": 0.18,
        "locked": True,
        "pixel_pct": 0.0,
    }
    layers.append(new_layer)

    rows = []
    existing = current_df_data if isinstance(current_df_data, list) else (current_df_data.values if hasattr(current_df_data, "values") else [])
    for r in existing:
        rows.append(list(r))

    rows.append([f"Layer {new_id}", f"Custom Depth Layer {new_id}", "0.10 – 0.25", "0.0%", "🔒 Locked"])

    unlock_choices = ["🔒 All Layers Locked"]
    for l in layers:
        unlock_choices.append(f"Layer {l.get('id')}: {l.get('desc', '')}")

    if input_video:
        vp = resolve_source_video_path(input_video)
        if vp:
            from m2svid.object_layer_depth import load_layers_sidecar, save_layers_sidecar
            sidecar_data = load_layers_sidecar(str(vp)) or {}
            sidecar_data["video"] = vp.name
            sidecar_data["masks_file"] = f"{vp.stem}.masks.npz"
            sidecar_data["layers"] = layers
            save_layers_sidecar(str(vp), sidecar_data)

    return rows, layers, gr.update(choices=unlock_choices)


def on_revert_to_raw_baseline(input_video: str, frame_idx: int = 36):
    """Revert current depth npz and mp4 back to the untouched raw baseline."""
    if not input_video:
        return "Please select a video first.", None, gr.update(), gr.update(), gr.update()
    vp = resolve_source_video_path(input_video)
    if not vp or not vp.exists():
        return f"Video not found: {input_video}", None, gr.update(), gr.update(), gr.update()

    stem = vp.stem
    out_npz = DEPTH_DIR / f"{stem}_ICD_depth.npz"
    raw_npz = DEPTH_DIR / f"{stem}_ICD_depth.raw.npz"
    out_mp4 = DEPTH_DIR / f"{stem}_ICD_depth.mp4"

    if not raw_npz.exists():
        return f"Raw baseline '{raw_npz.name}' not found.", None, gr.update(), gr.update(), gr.update()

    shutil.copy2(str(raw_npz), str(out_npz))
    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    with np.load(str(out_npz)) as data:
        key = "depth" if "depth" in data else list(data.keys())[0]
        depth = data[key]
    _create_depth_preview_video(depth, str(out_mp4), fps)
    STATE["depth_npz"] = str(out_npz)

    # Re-detect layers on clean depth
    d_clean = depth[min(len(depth) - 1, int(frame_idx))].astype(np.float32)
    from m2svid.object_layer_depth import detect_scene_layer_hierarchy, layers_to_dataframe_data, load_layers_sidecar
    from m2svid.prepare_icdepth import load_clip_credit_sidecar
    credit_hint = load_clip_credit_sidecar(str(vp))
    sidecar_data = load_layers_sidecar(str(vp))
    sidecar_layers = sidecar_data.get("layers") if sidecar_data else None
    layers = detect_scene_layer_hierarchy(d_clean, sidecar_layers=sidecar_layers, credit_text_hint=credit_hint)
    table_data = layers_to_dataframe_data(layers, unlocked_id=None)
    unlock_choices = ["🔒 All Layers Locked"]
    for l in layers:
        unlock_choices.append(f"Layer {l['id']}: {l.get('desc', '')}")

    msg = f"✅ Reverted {stem} to untouched raw baseline! All layers reset to locked."
    return msg, str(out_mp4), table_data, layers, gr.update(choices=unlock_choices, value="🔒 All Layers Locked")


def on_scrub_reference_frame(
    input_video: str,
    frame_idx: int,
    show_markers: bool = False,
    keyframe_dict: Optional[Dict[int, Any]] = None,
):
    """Update canvas background when user scrubs the frame slider, keeping the exact slider position."""
    default_badge = "📌 **Keyframes recorded:** None *(Brush on canvas, then click 'Record Keyframe')*"
    if not input_video:
        return None, default_badge
    vp = resolve_source_video_path(input_video)
    if not vp or not vp.exists():
        return None, default_badge

    frames = get_cached_video_frames(str(vp))
    if not frames:
        return None, default_badge

    total_frames = len(frames)
    idx = max(0, min(total_frames - 1, int(frame_idx)))
    bgr = frames[idx]

    if bgr is None:
        return None, f"⚠️ Could not read frame {idx}."

    if show_markers:
        stem = vp.stem
        out_npz = DEPTH_DIR / f"{stem}_ICD_depth.npz"
        raw_npz = DEPTH_DIR / f"{stem}_ICD_depth.raw.npz"
        raw_credits_npz = DEPTH_DIR / f"{stem}_raw_credits_depth.npz"
        src_npz = raw_credits_npz if raw_credits_npz.exists() else (raw_npz if raw_npz.exists() else out_npz)
        if src_npz.exists():
            with np.load(str(src_npz)) as data:
                key = "depth" if "depth" in data else list(data.keys())[0]
                depth_all = data[key]
                idx_clamped = min(depth_all.shape[0] - 1, idx)
                d_frame = depth_all[idx_clamped].astype(np.float32)

            from m2svid.object_layer_depth import (
                analyze_scene_depth_anchors,
                render_annotated_reference_frame,
            )
            anchors_info = analyze_scene_depth_anchors(bgr, d_frame)
            annotated_bgr = render_annotated_reference_frame(bgr, d_frame, anchors_info, target_depth=0.18)
            rgb_canvas = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb_canvas = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb_canvas = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    kf_dict = dict(keyframe_dict or {})
    if vp:
        masks_npz_path = vp.parent / f"{vp.stem}.masks.npz"
        if masks_npz_path.exists():
            try:
                with np.load(str(masks_npz_path)) as m_data:
                    for k in m_data.files:
                        if k.isdigit():
                            k_int = int(k)
                            if k_int not in kf_dict:
                                kf_dict[k_int] = m_data[k].astype(np.float32)
            except Exception as e:
                SF_LOG.warning(f"Could not load keyframe masks from {masks_npz_path}: {e}")

    editor_value = build_editor_value_with_keyframe_mask(rgb_canvas, idx, kf_dict)

    kf_keys = sorted(list(kf_dict.keys()))
    has_mask = (kf_dict.get(idx) is not None and np.any(kf_dict[idx] > 0.05))
    mask_info = f"🎨 Mask active ({int(np.sum(kf_dict[idx] > 0.1))} px)" if has_mask else "⚪ No mask (exited/empty)"
    if len(kf_keys) > 10:
        badge = f"📍 **Frame {idx}:** {mask_info} | All {len(kf_keys)} frames tracked & painted on canvas"
    elif idx in kf_keys:
        badge = f"📌 **Keyframe recorded for Frame {idx}!** Recorded keyframes: `{kf_keys}`"
    elif kf_keys:
        badge = f"📍 Frame {idx} ({mask_info}). Recorded keyframes: `{kf_keys}`"
    else:
        badge = f"📍 Frame {idx} (Unrecorded). Brush and click 'Record Keyframe'."

    return editor_value, badge


def record_keyframe_action(
    input_video: str,
    frame_idx: int,
    editor_data: Any,
    keyframe_dict: Optional[Dict[int, Any]] = None,
    unlocked_choice: str = "🔒 All Layers Locked",
    layers_state: Optional[List[Dict[str, Any]]] = None,
    target_depth: float = 0.18,
):
    """Record painted strokes for current frame into keyframe state dictionary and persist masks & sidecar to disk."""
    kf_dict = dict(keyframe_dict or {})
    idx = int(frame_idx)

    # Determine unlocked ID
    unlocked_id = None
    if unlocked_choice and unlocked_choice != "🔒 All Layers Locked":
        for l in (layers_state or []):
            lid = l.get("id")
            if f"Layer {lid}" in unlocked_choice:
                unlocked_id = lid
                break

    # Always merge previously recorded keyframes from disk so no prior work is ever lost
    vp = None
    if input_video:
        vp = resolve_source_video_path(input_video)
        if vp:
            masks_npz_path = vp.parent / f"{vp.stem}.masks.npz"
            if masks_npz_path.exists():
                try:
                    with np.load(str(masks_npz_path)) as m_data:
                        for k in m_data.files:
                            if k.isdigit():
                                k_int = int(k)
                                if k_int not in kf_dict:
                                    kf_dict[k_int] = m_data[k].astype(np.float32)
                except Exception as e:
                    SF_LOG.warning(f"Could not load existing keyframes from disk: {e}")

    target_shape = None
    if vp and vp.exists():
        frames = get_cached_video_frames(str(vp))
        if frames:
            target_shape = frames[0].shape[:2]
        else:
            cap = cv2.VideoCapture(str(vp))
            ret, f = cap.read()
            cap.release()
            if ret and f is not None:
                target_shape = f.shape[:2]

    mask = extract_mask_from_image_editor(editor_data, target_shape=target_shape)
    if mask is None or np.max(mask) < 0.01:
        keys = sorted(list(kf_dict.keys()))
        badge = f"📌 **Keyframes recorded:** `{keys}`" if keys else "📌 **Keyframes recorded:** None"
        return (
            kf_dict,
            f"⚠️ No painted brush strokes detected on canvas for frame {idx}! Brush over the target object first.",
            badge,
            layers_state,
        )

    # Force exact native dimensions to eliminate any browser CSS/canvas scaling artifacts
    if target_shape is not None and mask.shape[:2] != target_shape[:2]:
        mask = cv2.resize(mask.astype(np.float32), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    kf_dict[idx] = mask

    # Update layer metadata in layers_state
    updated_layers = list(layers_state or [])
    if unlocked_id is not None:
        for l in updated_layers:
            if l.get("id") == unlocked_id:
                l["depth_val"] = float(round(target_depth, 3))
                kf_list = set(l.get("keyframes", []))
                kf_list.add(idx)
                l["keyframes"] = sorted(list(kf_list))
                break

    if vp:
        masks_npz_path = vp.parent / f"{vp.stem}.masks.npz"
        try:
            save_payload = {}
            for k, v in kf_dict.items():
                save_payload[str(k)] = v.astype(np.float32)
                if unlocked_id is not None:
                    save_payload[f"L{unlocked_id}_{k}"] = v.astype(np.float32)
            np.savez_compressed(str(masks_npz_path), **save_payload)
        except Exception as e:
            SF_LOG.warning(f"Could not autosave keyframe masks: {e}")

        try:
            from m2svid.object_layer_depth import load_layers_sidecar, save_layers_sidecar
            sidecar_data = load_layers_sidecar(str(vp)) or {}
            sidecar_data["video"] = vp.name
            sidecar_data["masks_file"] = f"{vp.stem}.masks.npz"
            if updated_layers:
                sidecar_data["layers"] = updated_layers
            if unlocked_id is not None:
                sidecar_data["unlocked_layer"] = unlocked_id
            sidecar_data["keyframes"] = sorted(list(kf_dict.keys()))
            save_layers_sidecar(str(vp), sidecar_data)
        except Exception as e:
            SF_LOG.warning(f"Could not autosave layer sidecar: {e}")

    keys = sorted(list(kf_dict.keys()))
    painted_pixels = int(np.sum(mask > 0.5))
    layer_tag = f" on Layer {unlocked_id}" if unlocked_id is not None else ""
    status_msg = f"✅ Recorded keyframe for frame {idx}{layer_tag} ({painted_pixels} pixels painted at {mask.shape[1]}x{mask.shape[0]}, target depth={target_depth:.2f}). Total keyframes: {keys}"
    badge_text = f"📌 **Keyframes recorded:** `{keys}` (Total: {len(keys)})"
    return kf_dict, status_msg, badge_text, updated_layers


def clear_keyframe_action(input_video: str = ""):
    """Clear all recorded keyframes and remove from disk."""
    if input_video:
        vp = resolve_source_video_path(input_video)
        if vp:
            masks_npz_path = vp.parent / f"{vp.stem}.masks.npz"
            if masks_npz_path.exists():
                try:
                    masks_npz_path.unlink()
                except Exception:
                    pass
            try:
                from m2svid.object_layer_depth import load_layers_sidecar, save_layers_sidecar
                sidecar_data = load_layers_sidecar(str(vp))
                if sidecar_data:
                    sidecar_data["keyframes"] = []
                    for l in sidecar_data.get("layers", []):
                        if "keyframes" in l:
                            l["keyframes"] = []
                    save_layers_sidecar(str(vp), sidecar_data)
            except Exception:
                pass
    return (
        {},
        "🗑️ All recorded keyframes cleared.",
        "📌 **Keyframes recorded:** None *(Brush on canvas, then click 'Record Keyframe')*",
    )


def track_and_paint_all_frames_action(
    input_video: str,
    frame_idx: int,
    editor_data: Any,
    keyframe_dict: Optional[Dict[int, Any]] = None,
    progress=gr.Progress(track_tqdm=True),
):
    """Run optical flow tracking on painted keyframes and paint masks on Depth Paint Canvas for all frames."""
    _prog = progress if callable(progress) else (lambda *args, **kwargs: None)
    if not input_video:
        return None, keyframe_dict, "Please select a source video first.", "📌 **Keyframes recorded:** None"

    vp = resolve_source_video_path(input_video)
    if not vp or not vp.exists():
        return None, keyframe_dict, f"Source video not found: {input_video}", "📌 **Keyframes recorded:** None"

    frames = get_cached_video_frames(str(vp))
    if not frames:
        cap = cv2.VideoCapture(str(vp))
        frames = []
        while True:
            ret, f = cap.read()
            if not ret: break
            frames.append(f)
        cap.release()

    if not frames:
        return None, keyframe_dict, "Could not load video frames.", "📌 **Keyframes recorded:** None"

    H, W = frames[0].shape[:2]
    kf_dict = dict(keyframe_dict or {})

    masks_npz_path = vp.parent / f"{vp.stem}.masks.npz"
    if masks_npz_path.exists():
        try:
            with np.load(str(masks_npz_path)) as m_data:
                for k in m_data.files:
                    if k.isdigit():
                        k_int = int(k)
                        if k_int not in kf_dict:
                            kf_dict[k_int] = m_data[k].astype(np.float32)
        except Exception:
            pass

    curr_mask = extract_mask_from_image_editor(editor_data, target_shape=(H, W))
    if curr_mask is not None and np.max(curr_mask) > 0.01:
        kf_dict[int(frame_idx)] = curr_mask

    if not kf_dict:
        return None, kf_dict, "⚠️ No painted keyframes found! Please brush on canvas and record at least one keyframe first.", "📌 **Keyframes recorded:** None"

    _prog(0.2, desc="Running optical flow tracking across all frames...")
    import importlib
    import m2svid.object_layer_depth
    importlib.reload(m2svid.object_layer_depth)
    from m2svid.object_layer_depth import track_object_across_video
    tracked = track_object_across_video(frames, multi_keyframes=kf_dict)

    save_dict = {}
    updated_dict = {}
    for t in range(len(frames)):
        m_t = tracked[t].astype(np.float32)
        save_dict[str(t)] = m_t
        updated_dict[t] = m_t

    try:
        np.savez_compressed(str(masks_npz_path), **save_dict)
    except Exception as e:
        SF_LOG.warning(f"Could not save tracked masks to {masks_npz_path}: {e}")

    idx = max(0, min(len(frames) - 1, int(frame_idx)))
    rgb_canvas = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
    editor_val = build_editor_value_with_keyframe_mask(rgb_canvas, idx, updated_dict)

    badge = f"🎯 **Tracking complete!** All {len(frames)} frames tracked and painted on canvas. Scrub the slider to inspect any frame!"
    status = f"✅ Tracked painted object across all {len(frames)} frames. Canvas updated with tracked strokes!"
    _prog(1.0, desc="Complete!")
    return editor_val, updated_dict, status, badge


def run_depth_editor_action(
    input_video: str,
    frame_idx: int,
    editor_data: Any,
    keyframe_dict: Optional[Dict[int, Any]],
    target_creatures_depth: float,
    target_city_depth: float,
    target_sky_depth: float,
    skyline_split: bool = True,
    tophat_snap: bool = True,
    save_sidecar: bool = True,
    enable_tracking: bool = True,
    unlocked_choice: str = "🔒 All Layers Locked",
    layers_state: Optional[List[Dict[str, Any]]] = None,
    progress=gr.Progress(track_tqdm=True),
):
    """Execute full Depth Editor holdout-matte layering, optical flow tracking, and NPZ generation."""
    _prog = progress if callable(progress) else (lambda *args, **kwargs: None)
    if not unlocked_choice or unlocked_choice == "🔒 All Layers Locked":
        return (
            "⚠️ All layers are currently locked! Please unlock a layer from 'Unlock Layer for Editing' before applying.",
            None, None, None, gr.update()
        )

    if not input_video:
        return "Please select a source video first.", None, None, None, gr.update()
    vp = resolve_source_video_path(input_video)
    if not vp or not vp.exists():
        return f"Source video not found: {input_video}", None, None, None, gr.update()

    stem = vp.stem
    out_npz = DEPTH_DIR / f"{stem}_ICD_depth.npz"
    raw_npz = DEPTH_DIR / f"{stem}_ICD_depth.raw.npz"
    out_mp4 = DEPTH_DIR / f"{stem}_ICD_depth.mp4"

    if not raw_npz.exists() and out_npz.exists():
        shutil.copy2(str(out_npz), str(raw_npz))
        SF_LOG.info(f"[DepthEditor] Created untouched raw backup: {raw_npz}")

    raw_credits_npz = DEPTH_DIR / f"{stem}_raw_credits_depth.npz"
    src_npz = raw_credits_npz if raw_credits_npz.exists() else (raw_npz if raw_npz.exists() else out_npz)
    if not src_npz.exists():
        return f"ICDepth map '{out_npz.name}' not found. Run ICDepth first.", None, None, None, gr.update()

    import importlib
    import m2svid.object_layer_depth
    importlib.reload(m2svid.object_layer_depth)
    from m2svid.object_layer_depth import (
        apply_depth_editor_tracking,
        save_layers_sidecar,
        get_layer_depth_bounds,
    )
    from m2svid.prepare_icdepth import load_clip_credit_sidecar, detect_and_elevate_credits

    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    if not frames:
        return f"Could not read video frames from {vp.name}", None, None, None, gr.update()

    H, W = frames[0].shape[:2]

    # Resolve keyframe masks
    k_dict = {}
    for k, v in dict(keyframe_dict or {}).items():
        if v is not None and np.max(v) > 0.01:
            if v.shape[:2] != (H, W):
                v = cv2.resize(v.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            k_dict[int(k)] = v

    if not k_dict:
        curr_mask = extract_mask_from_image_editor(editor_data, target_shape=(H, W))
        if curr_mask is not None and np.max(curr_mask) > 0.01:
            if curr_mask.shape[:2] != (H, W):
                curr_mask = cv2.resize(curr_mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            k_dict[int(frame_idx)] = curr_mask
        else:
            return (
                "⚠️ No painted strokes detected on canvas and no keyframes recorded! "
                "Please brush over the target object(s) or record at least one keyframe before applying.",
                None, None, None, gr.update()
            )

    # Determine unlocked ID
    unlocked_id = 3
    for lid in [1, 2, 3, 4, 5, 6, 7]:
        if f"Layer {lid}" in unlocked_choice:
            unlocked_id = lid
            break

    # Determine lock threshold and layer bounds based on layer
    lock_threshold = 0.41
    if layers_state:
        for l in layers_state:
            if l.get("id") == unlocked_id:
                if "depth_range" in l:
                    lock_threshold = float(l["depth_range"][1])
                else:
                    lock_threshold = float(l.get("depth_max", 0.41))
                break

    _prog = progress if callable(progress) else (lambda *args, **kwargs: None)
    _prog(0.1, desc="Loading base ICDepth sequence...")
    with np.load(str(src_npz)) as data:
        key = "depth" if "depth" in data else list(data.keys())[0]
        depth_seq = data[key].astype(np.float32)

    _prog(0.2, desc=f"Executing Depth Editor tracking for Layer {unlocked_id}...")
    updated_depth = apply_depth_editor_tracking(
        rgb_frames=frames,
        base_depth_seq=depth_seq,
        keyframe_masks=k_dict,
        target_depth=float(target_creatures_depth),
        city_depth=float(target_city_depth),
        sky_depth=float(target_sky_depth),
        separate_sky_city=bool(skyline_split),
        tophat_snap=bool(tophat_snap),
        lock_threshold=lock_threshold,
        unlocked_id=unlocked_id,
        layers_state=layers_state,
        enable_tracking=bool(enable_tracking),
        progress=_prog,
    )

    # Absolute bitwise holdout for all locked layers
    if layers_state:
        for l in layers_state:
            lid = l.get("id")
            if lid != unlocked_id and l.get("locked", True):
                l_min, l_max = get_layer_depth_bounds(l)
                l_mask = (depth_seq >= l_min) & (depth_seq <= l_max)
                if l_min >= (float(target_creatures_depth) - 0.02):
                    # Locked foreground structure (bridge truss, foreground): 100% IMMUTABLE HOLDOUT
                    # Even if the user painted over it on canvas, the locked bridge remains in front
                    updated_depth[l_mask] = depth_seq[l_mask]
                else:
                    # Locked background layer (deep floor): restore pixels outside the sculpted object
                    sculpted_mask = (np.abs(updated_depth - depth_seq) > 0.005)
                    restore_mask = l_mask & (~sculpted_mask)
                    updated_depth[restore_mask] = depth_seq[restore_mask]

    # Final compositing step: Permanently elevate credits to the foremost plane (>= 0.95)
    credit_hint = load_clip_credit_sidecar(str(vp))
    try:
        updated_depth = detect_and_elevate_credits(
            updated_depth,
            frames,
            credit_text_hint=credit_hint,
            scene_max_disp=0.85,
            repair_strength=1.0,
        )
    except Exception as e:
        SF_LOG.warning(f"[DepthEditor] Credit elevation error: {e}")

    _prog(0.85, desc="Saving updated depth .npz and .mp4 preview...")
    save_m2svid_compatible_npz(updated_depth, str(out_npz))
    _create_depth_preview_video(updated_depth, str(out_mp4), fps)
    STATE["depth_npz"] = str(out_npz)

    if save_sidecar:
        sidecar_layers_list = []
        if layers_state:
            for l in layers_state:
                l_min, l_max = get_layer_depth_bounds(l)
                d_val = float(target_creatures_depth) if l.get("id") == unlocked_id else float(l.get("depth_val", (l_min + l_max) / 2.0))
                sidecar_layers_list.append({
                    "id": l.get("id"),
                    "name": l.get("name", f"Layer {l.get('id')}"),
                    "desc": l.get("desc", ""),
                    "range": l.get("range", ""),
                    "depth_range": [l_min, l_max],
                    "depth_min": l_min,
                    "depth_max": l_max,
                    "depth_val": d_val,
                    "locked": (l.get("id") != unlocked_id),
                    "keyframes": l.get("keyframes", []),
                })
        else:
            sidecar_layers_list = [
                {"id": 1, "name": "Layer 1: Foreground", "desc": "Foreground", "locked": (unlocked_id != 1)},
                {"id": 2, "name": "Layer 2: Midground", "desc": "Midground", "locked": (unlocked_id != 2)},
                {"id": 3, "name": "Layer 3: Background", "desc": "Background", "locked": (unlocked_id != 3)},
            ]

        sidecar_data = {
            "video": vp.name,
            "masks_file": f"{vp.stem}.masks.npz",
            "base_depth_file": src_npz.name,
            "layers": sidecar_layers_list,
            "unlocked_layer": unlocked_id,
            "keyframes": sorted(list(k_dict.keys())),
        }
        save_layers_sidecar(str(vp), sidecar_data)

    out_dir = Path(tempfile.mkdtemp(prefix="stereofaster_depth_editor_"))
    depth_vis_path = out_dir / "depth_preview.mp4"
    shutil.copy(str(out_mp4), str(depth_vis_path))

    _prog(1.0, desc="Complete!")
    new_dep_dd = gr.update(choices=[""] + get_depth_video_list(), value=f"{stem}_ICD_depth")
    msg = (
        f"✅ Depth Editor successfully processed {len(frames)} frames for Layer {unlocked_id} across keyframes {sorted(list(k_dict.keys()))}! "
        f"Protected Holdout Layers are 100% bitwise preserved. "
        f"Updated {out_npz.name} and preview {out_mp4.name}."
    )
    return msg, str(depth_vis_path), str(out_npz), str(out_npz), new_dep_dd


def run_depth_editor_option_a(
    input_video: str,
    frame_idx: int,
    editor_data: Any,
    keyframe_dict: Optional[Dict[int, Any]],
    target_creatures_depth: float,
    target_city_depth: float,
    target_sky_depth: float,
    skyline_split: bool = True,
    tophat_snap: bool = True,
    save_sidecar: bool = True,
    enable_tracking: bool = True,
    unlocked_choice: str = "🔒 All Layers Locked",
    layers_state: Optional[List[Dict[str, Any]]] = None,
    icd_steps: int = 5,
    icd_cfg: float = 1.0,
    icd_res: int = 512,
    progress=gr.Progress(track_tqdm=True),
):
    """Option A: In-Context Keyframe Propagation (ICDepth DiT)."""
    if not unlocked_choice or unlocked_choice == "🔒 All Layers Locked":
        return (
            "⚠️ All layers are currently locked! Please unlock a layer from 'Unlock Layer for Editing' before applying.",
            None, None, None, gr.update()
        )

    if not input_video:
        return "Please select a source video first.", None, None, None, gr.update()
    vp = resolve_source_video_path(input_video)
    if not vp or not vp.exists():
        return f"Source video not found: {input_video}", None, None, None, gr.update()

    stem = vp.stem
    out_npz = DEPTH_DIR / f"{stem}_ICD_depth.npz"
    raw_npz = DEPTH_DIR / f"{stem}_ICD_depth.raw.npz"
    out_mp4 = DEPTH_DIR / f"{stem}_ICD_depth.mp4"

    if not raw_npz.exists() and out_npz.exists():
        shutil.copy2(str(out_npz), str(raw_npz))
        SF_LOG.info(f"[DepthEditor] Created untouched raw backup: {raw_npz}")

    raw_credits_npz = DEPTH_DIR / f"{stem}_raw_credits_depth.npz"
    src_npz = raw_credits_npz if raw_credits_npz.exists() else (raw_npz if raw_npz.exists() else out_npz)
    if not src_npz.exists():
        return f"ICDepth map '{out_npz.name}' not found. Run ICDepth first.", None, None, None, gr.update()

    from m2svid.object_layer_depth import (
        apply_option_a_keyframe_propagation,
        save_layers_sidecar,
    )
    from m2svid.prepare_icdepth import load_clip_credit_sidecar, detect_and_elevate_credits

    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    if not frames:
        return f"Could not read video frames from {vp.name}", None, None, None, gr.update()

    H, W = frames[0].shape[:2]

    # Resolve keyframe masks
    k_dict = {}
    for k, v in dict(keyframe_dict or {}).items():
        if v is not None and np.max(v) > 0.01:
            if v.shape[:2] != (H, W):
                v = cv2.resize(v.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            k_dict[int(k)] = v

    if not k_dict:
        curr_mask = extract_mask_from_image_editor(editor_data, target_shape=(H, W))
        if curr_mask is not None and np.max(curr_mask) > 0.01:
            if curr_mask.shape[:2] != (H, W):
                curr_mask = cv2.resize(curr_mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            k_dict[int(frame_idx)] = curr_mask

    # Determine unlocked ID
    unlocked_id = 3
    for lid in [1, 2, 3, 4, 5, 6, 7]:
        if f"Layer {lid}" in unlocked_choice:
            unlocked_id = lid
            break

    progress(0.05, desc="Loading base ICDepth sequence...")
    with np.load(str(src_npz)) as data:
        key = "depth" if "depth" in data else list(data.keys())[0]
        depth_seq = data[key].astype(np.float32)

    updated_depth = apply_option_a_keyframe_propagation(
        rgb_frames=frames,
        base_depth_seq=depth_seq,
        keyframe_masks=k_dict,
        unlocked_id=unlocked_id,
        layers_state=layers_state,
        target_depth=float(target_creatures_depth),
        tophat_snap=bool(tophat_snap),
        icd_steps=int(icd_steps),
        icd_cfg=float(icd_cfg),
        icd_res=int(icd_res),
        fps=fps,
        progress=progress,
    )

    # Absolute bitwise holdout for all locked layers
    if layers_state:
        active_d_max = float(target_creatures_depth) + 0.04
        for l in layers_state:
            if l.get("id") == unlocked_id and "depth_range" in l:
                active_d_max = max(active_d_max, float(l["depth_range"][1]))
                break

        for l in layers_state:
            lid = l.get("id")
            if lid != unlocked_id:
                if "depth_range" in l:
                    l_min, l_max = float(l["depth_range"][0]), float(l["depth_range"][1])
                else:
                    l_min = float(l.get("depth_min", 0.0))
                    l_max = float(l.get("depth_max", 1.0))
                l_mask = (depth_seq >= l_min) & (depth_seq <= l_max)
                # CRUCIAL: Protect the user's sculpted object from being overwritten by old base depth.
                # All pixels outside the sculpted object are 100% bitwise restored to base depth.
                sculpted_mask = (np.abs(updated_depth - depth_seq) > 0.005)
                restore_mask = l_mask & (~sculpted_mask)
                updated_depth[restore_mask] = depth_seq[restore_mask]

    # Final compositing step: Permanently elevate credits to the foremost plane (>= 0.95)
    credit_hint = load_clip_credit_sidecar(str(vp))
    try:
        updated_depth = detect_and_elevate_credits(
            updated_depth,
            frames,
            credit_text_hint=credit_hint,
            scene_max_disp=0.85,
            repair_strength=1.0,
        )
    except Exception as e:
        SF_LOG.warning(f"[DepthEditor] Credit elevation error: {e}")

    progress(0.9, desc="Saving updated depth .npz and .mp4 preview...")
    save_m2svid_compatible_npz(updated_depth, str(out_npz))
    _create_depth_preview_video(updated_depth, str(out_mp4), fps)
    STATE["depth_npz"] = str(out_npz)

    if save_sidecar:
        sidecar_layers_list = []
        if layers_state:
            for l in layers_state:
                sidecar_layers_list.append({
                    "id": l.get("id"),
                    "name": l.get("name", f"Layer {l.get('id')}"),
                    "desc": l.get("desc", ""),
                    "range": l.get("range", ""),
                    "depth_range": [l.get("depth_min", 0.0), l.get("depth_max", 1.0)],
                    "locked": (l.get("id") != unlocked_id),
                })
        else:
            sidecar_layers_list = [
                {"id": 1, "name": "Layer 1: Foreground", "desc": "Foreground", "locked": (unlocked_id != 1)},
                {"id": 2, "name": "Layer 2: Midground", "desc": "Midground", "locked": (unlocked_id != 2)},
                {"id": 3, "name": "Layer 3: Background", "desc": "Background", "locked": (unlocked_id != 3)},
            ]

        sidecar_data = {
            "video": vp.name,
            "layers": sidecar_layers_list,
            "unlocked_layer": unlocked_id,
            "method": "Option A (In-Context Keyframe Propagation)",
            "keyframes": sorted(list(k_dict.keys())),
        }
        save_layers_sidecar(str(vp), sidecar_data)

    out_dir = Path(tempfile.mkdtemp(prefix="stereofaster_depth_editor_opt_a_"))
    depth_vis_path = out_dir / "depth_preview.mp4"
    shutil.copy(str(out_mp4), str(depth_vis_path))

    _prog(1.0, desc="Complete!")
    new_dep_dd = gr.update(choices=[""] + get_depth_video_list(), value=f"{stem}_ICD_depth")
    msg = (
        f"✅ Option A (In-Context Keyframe Propagation) successfully completed for Layer {unlocked_id}! "
        f"Keyframe relief propagated with DiT temporal coherence across {len(frames)} frames. "
        f"Protected Holdout Layers are 100% bitwise preserved. "
        f"Updated {out_npz.name} and preview {out_mp4.name}."
    )
    return msg, str(depth_vis_path), str(out_npz), str(out_npz), new_dep_dd


def run_depth_editor_option_b(
    input_video: str,
    frame_idx: int,
    editor_data: Any,
    keyframe_dict: Optional[Dict[int, Any]],
    target_creatures_depth: float,
    target_city_depth: float,
    target_sky_depth: float,
    skyline_split: bool = True,
    tophat_snap: bool = True,
    save_sidecar: bool = True,
    enable_tracking: bool = True,
    unlocked_choice: str = "🔒 All Layers Locked",
    layers_state: Optional[List[Dict[str, Any]]] = None,
    engine: str = "DepthCrafter (SVD U-Net)",
    cavity_mode: str = "Inpaint Occluders (Pyramid Infill)",
    dc_steps: int = 5,
    dc_res: int = 512,
    dc_guidance: float = 1.0,
    icd_steps: int = 5,
    icd_cfg: float = 1.0,
    icd_res: int = 512,
    progress=gr.Progress(track_tqdm=True),
):
    """Option B: Occluder-Removed Cavity Pass (DepthCrafter / ICDepth Cavity)."""
    if not unlocked_choice or unlocked_choice == "🔒 All Layers Locked":
        return (
            "⚠️ All layers are currently locked! Please unlock a layer from 'Unlock Layer for Editing' before applying.",
            None, None, None, gr.update()
        )

    if not input_video:
        return "Please select a source video first.", None, None, None, gr.update()
    vp = resolve_source_video_path(input_video)
    if not vp or not vp.exists():
        return f"Source video not found: {input_video}", None, None, None, gr.update()

    stem = vp.stem
    out_npz = DEPTH_DIR / f"{stem}_ICD_depth.npz"
    raw_npz = DEPTH_DIR / f"{stem}_ICD_depth.raw.npz"
    out_mp4 = DEPTH_DIR / f"{stem}_ICD_depth.mp4"

    if not raw_npz.exists() and out_npz.exists():
        shutil.copy2(str(out_npz), str(raw_npz))
        SF_LOG.info(f"[DepthEditor] Created untouched raw backup: {raw_npz}")

    raw_credits_npz = DEPTH_DIR / f"{stem}_raw_credits_depth.npz"
    src_npz = raw_credits_npz if raw_credits_npz.exists() else (raw_npz if raw_npz.exists() else out_npz)
    if not src_npz.exists():
        return f"ICDepth map '{out_npz.name}' not found. Run ICDepth first.", None, None, None, gr.update()

    from m2svid.object_layer_depth import (
        apply_option_b_cavity_pass,
        save_layers_sidecar,
    )
    from m2svid.prepare_icdepth import load_clip_credit_sidecar, detect_and_elevate_credits

    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    if not frames:
        return f"Could not read video frames from {vp.name}", None, None, None, gr.update()

    # Determine unlocked ID
    unlocked_id = 3
    for lid in [1, 2, 3, 4, 5, 6, 7]:
        if f"Layer {lid}" in unlocked_choice:
            unlocked_id = lid
            break

    # Resolve model-specific parameters
    if "depthcrafter" in str(engine).lower():
        effective_steps = int(dc_steps)
        effective_cfg = float(dc_guidance)
        effective_res = int(dc_res)
    else:
        effective_steps = int(icd_steps)
        effective_cfg = float(icd_cfg)
        effective_res = int(icd_res)

    progress(0.05, desc="Loading base ICDepth sequence...")
    with np.load(str(src_npz)) as data:
        key = "depth" if "depth" in data else list(data.keys())[0]
        depth_seq = data[key].astype(np.float32)

    updated_depth = apply_option_b_cavity_pass(
        rgb_frames=frames,
        base_depth_seq=depth_seq,
        unlocked_id=unlocked_id,
        layers_state=layers_state,
        cavity_mode="crop" if "crop" in cavity_mode.lower() else "inpaint",
        engine=engine,
        icd_steps=effective_steps,
        icd_cfg=effective_cfg,
        icd_res=effective_res,
        target_depth=float(target_creatures_depth) if target_creatures_depth else None,
        fps=fps,
        progress=progress,
    )

    # Absolute bitwise holdout for all locked layers
    if layers_state:
        active_d_max = float(target_creatures_depth) + 0.04
        for l in layers_state:
            if l.get("id") == unlocked_id and "depth_range" in l:
                active_d_max = max(active_d_max, float(l["depth_range"][1]))
                break

        for l in layers_state:
            lid = l.get("id")
            if lid != unlocked_id:
                if "depth_range" in l:
                    l_min, l_max = float(l["depth_range"][0]), float(l["depth_range"][1])
                else:
                    l_min = float(l.get("depth_min", 0.0))
                    l_max = float(l.get("depth_max", 1.0))
                l_mask = (depth_seq >= l_min) & (depth_seq <= l_max)
                # CRUCIAL: Protect the user's sculpted object from being overwritten by old base depth.
                # All pixels outside the sculpted object are 100% bitwise restored to base depth.
                sculpted_mask = (np.abs(updated_depth - depth_seq) > 0.005)
                restore_mask = l_mask & (~sculpted_mask)
                updated_depth[restore_mask] = depth_seq[restore_mask]

    # Final compositing step: Permanently elevate credits to the foremost plane (>= 0.95)
    credit_hint = load_clip_credit_sidecar(str(vp))
    try:
        updated_depth = detect_and_elevate_credits(
            updated_depth,
            frames,
            credit_text_hint=credit_hint,
            scene_max_disp=0.85,
            repair_strength=1.0,
        )
    except Exception as e:
        SF_LOG.warning(f"[DepthEditor] Credit elevation error: {e}")

    progress(0.9, desc="Saving updated depth .npz and .mp4 preview...")
    save_m2svid_compatible_npz(updated_depth, str(out_npz))
    _create_depth_preview_video(updated_depth, str(out_mp4), fps)
    STATE["depth_npz"] = str(out_npz)

    if save_sidecar:
        sidecar_layers_list = []
        if layers_state:
            for l in layers_state:
                sidecar_layers_list.append({
                    "id": l.get("id"),
                    "name": l.get("name", f"Layer {l.get('id')}"),
                    "desc": l.get("desc", ""),
                    "range": l.get("range", ""),
                    "depth_range": [l.get("depth_min", 0.0), l.get("depth_max", 1.0)],
                    "locked": (l.get("id") != unlocked_id),
                })
        else:
            sidecar_layers_list = [
                {"id": 1, "name": "Layer 1: Foreground", "desc": "Foreground", "locked": (unlocked_id != 1)},
                {"id": 2, "name": "Layer 2: Midground", "desc": "Midground", "locked": (unlocked_id != 2)},
                {"id": 3, "name": "Layer 3: Background", "desc": "Background", "locked": (unlocked_id != 3)},
            ]

        sidecar_data = {
            "video": vp.name,
            "layers": sidecar_layers_list,
            "unlocked_layer": unlocked_id,
            "method": f"Option B (Occluder-Removed Cavity Pass via {engine})",
            "cavity_mode": cavity_mode,
            "engine": engine,
        }
        save_layers_sidecar(str(vp), sidecar_data)

    out_dir = Path(tempfile.mkdtemp(prefix="stereofaster_depth_editor_opt_b_"))
    depth_vis_path = out_dir / "depth_preview.mp4"
    shutil.copy(str(out_mp4), str(depth_vis_path))

    _prog(1.0, desc="Complete!")
    new_dep_dd = gr.update(choices=[""] + get_depth_video_list(), value=f"{stem}_ICD_depth")
    msg = (
        f"✅ Option B (Occluder-Removed Cavity Pass via {engine}) successfully completed for Layer {unlocked_id}! "
        f"All locked foreground occluders removed via {cavity_mode}. "
        f"Aperture dynamic range restored and locked layers are 100% bitwise preserved. "
        f"Updated {out_npz.name} and preview {out_mp4.name}."
    )
    return msg, str(depth_vis_path), str(out_npz), str(out_npz), new_dep_dd


def apply_painted_object_layer(
    input_video: str,
    frame_idx: int,
    editor_data: Any,
    target_depth: float,
    object_name: str = "painted_object",
    skyline_split: bool = True,
    tophat_snap: bool = True,
    save_sidecar: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    """Backward compatibility wrapper redirecting to run_depth_editor_action."""
    return run_depth_editor_action(
        input_video=input_video,
        frame_idx=frame_idx,
        editor_data=editor_data,
        keyframe_dict=None,
        target_creatures_depth=target_depth,
        target_city_depth=0.06,
        target_sky_depth=0.00,
        skyline_split=skyline_split,
        tophat_snap=tophat_snap,
        save_sidecar=save_sidecar,
        progress=progress,
    )



_NVENC_AVAILABLE = None

def _check_nvenc_available() -> bool:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is None:
        if not torch.cuda.is_available() or not shutil.which("ffmpeg"):
            _NVENC_AVAILABLE = False
        else:
            try:
                # Test real NVENC encoding of 1 frame instead of assuming from ffmpeg -encoders
                res = subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64", "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                    capture_output=True, text=True, timeout=5
                )
                _NVENC_AVAILABLE = (res.returncode == 0)
            except Exception:
                _NVENC_AVAILABLE = False
        if _NVENC_AVAILABLE:
            SF_LOG.info("🚀 NVIDIA NVENC hardware encoder detected and enabled for video export.")
        else:
            SF_LOG.info("ℹ️ NVIDIA NVENC not available or supported in this environment; using CPU libx264 encoder.")
    return _NVENC_AVAILABLE


class FFmpegVideoWriter:
    """Streams raw frames directly to an FFmpeg subprocess encoding H.264 MP4.
    Uses GPU-accelerated h264_nvenc when available, with automatic fallback to libx264 and cv2.VideoWriter.
    """
    def __init__(self, path: str, width: int, height: int, fps: float, is_color: bool = True, crf: int = 17, preset: str = "fast"):
        self.path = str(path)
        self.width = width
        self.height = height
        self.fps = max(1.0, float(fps))
        self.is_color = is_color
        self.proc = None
        self.cv2_writer = None

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        if os.path.exists(self.path):
            try: os.remove(self.path)
            except Exception: pass

        if shutil.which("ffmpeg"):
            in_pix_fmt = "rgb24" if is_color else "gray"
            use_nvenc = _check_nvenc_available()

            def _build_cmd(encoder_type):
                if encoder_type == "nvenc":
                    codec_args = [
                        "-c:v", "h264_nvenc",
                        "-pix_fmt", "yuv420p",
                        "-cq", str(crf),
                        "-preset", "p4",
                    ]
                else:
                    codec_args = [
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-crf", str(crf),
                        "-preset", preset,
                    ]
                return [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-s", f"{width}x{height}",
                    "-pix_fmt", in_pix_fmt,
                    "-r", f"{self.fps:.4f}",
                    "-i", "-",
                    *codec_args,
                    self.path
                ]

            if use_nvenc:
                try:
                    self.proc = subprocess.Popen(
                        _build_cmd("nvenc"),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE
                    )
                    import time
                    time.sleep(0.05)
                    if self.proc.poll() is not None:
                        err = self.proc.stderr.read().decode('utf-8', errors='ignore') if self.proc.stderr else ""
                        SF_LOG.warning(f"NVENC exited immediately ({err[:200]}), falling back to libx264")
                        self.proc = None
                except Exception as e:
                    SF_LOG.warning(f"NVENC start failed ({e}), falling back to libx264")
                    self.proc = None

            if self.proc is None:
                try:
                    self.proc = subprocess.Popen(
                        _build_cmd("libx264"),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except Exception as e:
                    SF_LOG.warning(f"Failed to start ffmpeg subprocess for {path}: {e}")
                    self.proc = None

        if self.proc is None:
            import cv2
            fourcc_candidates = [
                cv2.VideoWriter_fourcc(*"mp4v"),
                cv2.VideoWriter_fourcc(*"avc1"),
                cv2.VideoWriter_fourcc(*"X264"),
            ]
            for fourcc in fourcc_candidates:
                w = cv2.VideoWriter(self.path, fourcc, self.fps, (width, height), isColor=is_color)
                if w.isOpened():
                    self.cv2_writer = w
                    break
                w.release()

    def write(self, frame: np.ndarray):
        """Writes a single frame: [H, W, 3] RGB if is_color else [H, W] uint8."""
        if self.proc and self.proc.stdin:
            self.proc.stdin.write(frame.tobytes())
        elif self.cv2_writer:
            import cv2
            if self.is_color:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.cv2_writer.write(frame_bgr)
            else:
                self.cv2_writer.write(frame)

    def close(self):
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                self.proc.wait()
            except Exception:
                pass
            self.proc = None
        if self.cv2_writer:
            self.cv2_writer.release()
            self.cv2_writer = None

    def is_valid(self) -> bool:
        return (self.proc is not None) or (self.cv2_writer is not None)


def _create_depth_preview_video_opencv(frames_u8: np.ndarray, out_path: str, fps: float):
    n, h, w = frames_u8.shape
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.exists(out_path):
        try: os.remove(out_path)
        except Exception: pass
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, max(1.0, float(fps)), (w, h), isColor=False)
    for frame in frames_u8:
        vw.write(frame)
    vw.release()
    SF_LOG.info(f"Depth video successfully written via OpenCV fallback: {n} frames -> {out_path}")


def _create_depth_preview_video(depth: np.ndarray, out_path: str, fps: float):
    """Create a contrast-enhanced grayscale video showing depth for preview and archiving.
    
    Uses global normalization across the video to preserve true temporal consistency across frames,
    vectorized NumPy scaling, and fast bulk GPU NVENC streaming.
    """
    if depth.size == 0:
        raise RuntimeError("Cannot create preview video: depth array is empty")

    n, h, w = depth.shape
    if n == 0:
        raise RuntimeError("Cannot create preview video: no frames in depth array")

    if w <= 0 or h <= 0:
        raise RuntimeError(f"Cannot create preview video: invalid frame size {w}x{h}")

    SF_LOG.info(f"Creating depth video: {n} frames @ {w}x{h}, {fps} fps -> {out_path}")

    # Fast global normalization (preserves 1:1 temporal consistency for future NPZ rebuilds)
    d_min = float(depth.min())
    d_max = float(depth.max())
    denom = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1.0
    frames_u8 = np.clip((depth - d_min) / denom, 0, 1)
    frames_u8 = (frames_u8 * 255.0).astype(np.uint8)

    writer = FFmpegVideoWriter(out_path, w, h, fps, is_color=False)
    if not writer.is_valid():
        SF_LOG.warning("FFmpeg video writer is not valid, writing preview video via OpenCV...")
        _create_depth_preview_video_opencv(frames_u8, out_path, fps)
        return

    written = 0
    try:
        # Fast bulk write to FFmpeg stdin (NVENC or libx264)
        if writer.proc and writer.proc.stdin:
            try:
                writer.proc.stdin.write(frames_u8.tobytes())
                written = n
            except Exception as pipe_err:
                SF_LOG.warning(f"FFmpeg write pipe failed ({pipe_err}), falling back to OpenCV VideoWriter...")
                writer.close()
                _create_depth_preview_video_opencv(frames_u8, out_path, fps)
                return
        else:
            for frame in frames_u8:
                writer.write(frame)
                written += 1
    except Exception as e:
        writer.close()
        SF_LOG.warning(f"FFmpeg writing error ({e}), falling back to OpenCV VideoWriter...")
        _create_depth_preview_video_opencv(frames_u8, out_path, fps)
        return

    writer.close()

    written_bytes = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    SF_LOG.info(f"Depth video written: {written}/{n} frames, {written_bytes} bytes")

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


def sync_orphan_depth_mp4s():
    """Scan depthmaps_videos/ and auto-convert any .mp4 depth videos that lack an .npz file."""
    if not DEPTH_DIR.exists():
        return
    for mp4 in sorted(DEPTH_DIR.glob("*_depth.mp4")):
        npz = mp4.parent / f"{mp4.name.rsplit('.', 1)[0]}.npz"
        if not npz.exists():
            try:
                SF_LOG.info(f"[Auto-Sync] Found orphan depth video {mp4.name}. Auto-converting to {npz.name}...")
                convert_depth_video_to_npz(str(mp4), str(npz))
            except Exception as e:
                SF_LOG.error(f"[Auto-Sync] Failed to convert {mp4.name} to npz: {e}")


def generate_title_desaturation_depth(
    source_stem: str,
    auto_bg: bool = True,
    invert_depth: bool = False,
    progress=gr.Progress(track_tqdm=True)
):
    """Generate clean, artifact-free depth for flat 2D title cards on any solid background (black, white, green screen, etc.) using desaturation & color distance keying."""
    if not source_stem:
        return "No source clip selected.", None, gr.update(), None
    
    video_path = SOURCE_DIR / f"{source_stem}.mp4"
    if not video_path.exists():
        for ext in ('.mov', '.avi', '.mkv'):
            alt = SOURCE_DIR / f"{source_stem}{ext}"
            if alt.exists():
                video_path = alt
                break
    if not video_path.exists():
        return f"Source video '{source_stem}' not found in source_videos/", None, gr.update(), None

    SF_LOG.info(f"[Title Desaturation] Processing '{source_stem}' (auto_bg={auto_bg}, invert={invert_depth})...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"Cannot open video: {video_path}", None, gr.update(), None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    depth_frames = []
    f_idx = 0
    corner_size = max(5, min(15, min(w, h) // 20))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if auto_bg:
            # Sample 4 corners to detect the solid background color
            corners = [
                frame[:corner_size, :corner_size],
                frame[:corner_size, -corner_size:],
                frame[-corner_size:, :corner_size],
                frame[-corner_size:, -corner_size:]
            ]
            bg_color = np.median(np.concatenate([c.reshape(-1, 3) for c in corners], axis=0), axis=0)
            # Chebyshev (max channel) color distance from solid background
            diff = np.max(np.abs(frame.astype(np.float32) - bg_color), axis=-1)
            # Noise threshold eliminates video compression macroblocking in solid background
            mask = diff > 12.0
            depth_f = np.zeros((h, w), dtype=np.float32)
            depth_f[mask] = np.clip((diff[mask] - 12.0) / (255.0 - 12.0), 0.0, 1.0)
        else:
            depth_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        if invert_depth:
            depth_f = np.clip(1.0 - depth_f, 0.0, 1.0)

        depth_frames.append(depth_f)
        f_idx += 1
        if progress and f_idx % 10 == 0:
            try:
                progress(f_idx / total_frames, desc=f"Desaturating: frame {f_idx}/{total_frames}")
            except Exception:
                pass
    cap.release()

    if not depth_frames:
        return "No frames decoded from video.", None, gr.update(), None

    depth = np.stack(depth_frames, axis=0).astype(np.float32)
    depth = np.clip(depth, 0.0, 1.0)

    out_npz = DEPTH_DIR / f"{source_stem}_depth.npz"
    out_mp4 = DEPTH_DIR / f"{source_stem}_depth.mp4"

    save_m2svid_compatible_npz(depth, str(out_npz))
    _create_depth_preview_video(depth, str(out_mp4), fps)

    msg = f"✅ Flat Title Depth created: {len(depth_frames)} frames ({w}x{h}) saved to {out_npz.name} and {out_mp4.name}."
    SF_LOG.info(msg)
    dep_list = get_depth_video_list()
    return msg, str(out_mp4), gr.update(choices=[""] + dep_list, value=out_mp4.stem), str(out_npz)


def generate_3d_title_isolation_depth(
    source_stem: str,
    color_target: str = "Universal (Auto Solid BG)",
    base_elevation: float = 0.60,
    relief_scale: float = 0.32,
    auto_bg: bool = True,
    invert_depth: bool = False,
    progress=gr.Progress(track_tqdm=True)
):
    """Generate 3D depth for sculpted or claymation title cards on any solid background using color isolation & surface relief."""
    if not source_stem:
        return "No source clip selected.", None, gr.update(), None

    video_path = SOURCE_DIR / f"{source_stem}.mp4"
    if not video_path.exists():
        for ext in ('.mov', '.avi', '.mkv'):
            alt = SOURCE_DIR / f"{source_stem}{ext}"
            if alt.exists():
                video_path = alt
                break
    if not video_path.exists():
        return f"Source video '{source_stem}' not found in source_videos/", None, gr.update(), None

    SF_LOG.info(f"[3D Title Isolation] Processing '{source_stem}' with target '{color_target}', base={base_elevation}, relief={relief_scale}, auto_bg={auto_bg}, invert={invert_depth}...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"Cannot open video: {video_path}", None, gr.update(), None

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    depth_frames = []
    f_idx = 0
    corner_size = max(5, min(15, min(w, h) // 20))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        b, g, r = cv2.split(frame)
        b_f, g_f, r_f = b.astype(np.float32), g.astype(np.float32), r.astype(np.float32)

        if "Red" in color_target:
            diff = np.clip(r_f - np.maximum(g_f, b_f), 0, 255)
        elif "Green" in color_target:
            diff = np.clip(g_f - np.maximum(r_f, b_f), 0, 255)
        elif "Blue" in color_target:
            diff = np.clip(b_f - np.maximum(r_f, g_f), 0, 255)
        elif "Universal" in color_target and auto_bg:
            corners = [
                frame[:corner_size, :corner_size],
                frame[:corner_size, -corner_size:],
                frame[-corner_size:, :corner_size],
                frame[-corner_size:, -corner_size:]
            ]
            bg_color = np.median(np.concatenate([c.reshape(-1, 3) for c in corners], axis=0), axis=0)
            diff = np.max(np.abs(frame.astype(np.float32) - bg_color), axis=-1)
        else: # Universal Luminance
            diff = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        mask = diff > 15.0
        text_vals = diff[mask]
        if len(text_vals) > 0:
            max_v = max(float(np.percentile(text_vals, 99)), 30.0)
            norm_relief = np.clip(diff / max_v, 0.0, 1.0)
            depth_f = np.where(mask, float(base_elevation) + float(relief_scale) * norm_relief, 0.0).astype(np.float32)
        else:
            depth_f = np.zeros((h, w), dtype=np.float32)

        if invert_depth:
            depth_f = np.clip(1.0 - depth_f, 0.0, 1.0)

        depth_frames.append(depth_f)
        f_idx += 1
        if progress and f_idx % 10 == 0:
            try:
                progress(f_idx / total_frames, desc=f"Isolating 3D Title: frame {f_idx}/{total_frames}")
            except Exception:
                pass
    cap.release()

    if not depth_frames:
        return "No frames decoded from video.", None, gr.update(), None

    depth = np.stack(depth_frames, axis=0).astype(np.float32)
    depth = np.clip(depth, 0.0, 1.0)

    out_npz = DEPTH_DIR / f"{source_stem}_depth.npz"
    out_mp4 = DEPTH_DIR / f"{source_stem}_depth.mp4"

    save_m2svid_compatible_npz(depth, str(out_npz))
    _create_depth_preview_video(depth, str(out_mp4), fps)

    msg = f"✅ 3D Title Depth created: {len(depth_frames)} frames ({w}x{h}) saved to {out_npz.name} and {out_mp4.name}."
    SF_LOG.info(msg)
    dep_list = get_depth_video_list()
    return msg, str(out_mp4), gr.update(choices=[""] + dep_list, value=out_mp4.stem), str(out_npz)


# =============================================================================
# STEP 2: M2SVid Warping + 1-step Generation (the second heavy process)
# =============================================================================

_m2svid_model = None  # singleton


def unload_m2svid_model():
    """Unload M2SVid model from GPU VRAM to allow DepthCrafter to have full access to GPU."""
    global _m2svid_model
    if _m2svid_model is not None:
        SF_LOG.info("Unloading M2SVid model from GPU and flushing VRAM...")
        try:
            del _m2svid_model
        except Exception:
            pass
        _m2svid_model = None
    clear_cuda()


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
    # Cap VAE decode chunk to 14 to prevent massive 13GB VAE activation spikes on native 1080p
    if hasattr(config.model, 'params'):
        vae_chunk = min(_VRAM_DEFAULTS.get('vae', 14), 14)
        config.model.params.en_and_decode_n_samples_a_time = vae_chunk
        SF_LOG.info(f"Dynamically set VAE decode chunk size to {vae_chunk}")
        
    model = instantiate_from_config(config.model).cpu()
    model.init_from_ckpt(ckpt_path)
    model = model.cuda().half().eval()
    _m2svid_model = model
    SF_LOG.info("M2SVid model loaded successfully")
    return model


def step2_run_m2svid(
    depth_npz_path: str,
    disparity_perc: float,
    convergence_point: float,
    reprojected_closing_kernel: int,
    mask_antialias: bool,
    m2svid_config: str,
    m2svid_ckpt: str,
    input_video_path: Optional[str] = None,
    warping_batch_size: int = 2,
    gen_chunk_size: int = 14,
    m2svid_process_res: str = "1024x576 (Optimal 12GB)",
    gen_batch_size: int = 1,
    progress=gr.Progress(track_tqdm=True),
    progress_prefix: str = ""
) -> Tuple[str, str, str, str, str]:
    """
    Performs the M2SVid stage:
    - Geometric warping using the estimated depth
    - 1-step conditioned generation
    Returns paths to: generated_right, sbs, anaglyph, and status
    """
    SF_LOG.info("Starting M2SVid stage (warping + generation)")
    SF_LOG.debug(f"Depth npz: {depth_npz_path}")
    if not depth_npz_path or not os.path.exists(depth_npz_path):
        SF_LOG.error("No depth file provided")
        return "No depth file from Step 1. Please run Step 1 first or upload a compatible depth.npz.", None, None, None, None

    if input_video_path is None:
        SF_LOG.error("No input video provided")
        return "No source video selected.", None, None, None, None

    video_path = input_video_path
    stem = Path(video_path).stem
    if not progress_prefix:
        progress_prefix = f"[{stem}] "
        
    out_dir = FINAL_DIR / f"{stem}_stereo"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    reprojected_dir = out_dir / "reprojected"
    reprojected_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            from m2svid.prepare_depthcrafter import unload_depthcrafter_model
            unload_depthcrafter_model()
        except Exception:
            pass
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

        # Support both StereoCrafter scale (> 1.0, e.g. 20.0 -> 0.020) and legacy fraction (<= 1.0, e.g. 0.02)
        eff_disp_perc = disparity_perc / 1000.0 if disparity_perc > 1.0 else disparity_perc

        # We call the existing warping logic with in-memory handoff
        warp_res = m2s_process_video_with_depth(
            video_path=str(video_path),
            depth_path=depth_npz_path,
            output_path_reprojected=str(reprojected_video),
            output_path_mask=str(reprojected_mask),
            disparity_perc=eff_disp_perc,
            batch_size=warping_batch_size,
            convergence_point=0.0, # MUST BE 0.0 TO PREVENT UNET GLITCHES!
            return_in_memory=True,
        )
        SF_LOG.info("Geometric warping complete")

        try:
            if progress:
                progress(0.35, desc=f"{progress_prefix}Geometric warping complete. Starting 1-step M2SVid generation...")
        except TypeError:
            pass

        # --- Heavy part: load M2SVid model and run 1-step generation ---
        model = _load_m2svid_model(m2svid_config, m2svid_ckpt)

        # Use fast in-memory tensors if available, with robust fallback to disk reading
        if warp_res is not None:
            in_mem_reproj, in_mem_mask, in_mem_left, fps = warp_res
            input_video = torch.from_numpy(in_mem_left).permute(0, 3, 1, 2).float() / 255.0
            reprojected = torch.from_numpy(in_mem_reproj).permute(0, 3, 1, 2).float() / 255.0
            reprojected_mask_arr = torch.from_numpy(in_mem_mask[:, None, :, :]).float() / 255.0
            del in_mem_reproj, in_mem_mask, in_mem_left, warp_res
            SF_LOG.info("Used direct in-memory warping handoff (bypassed disk read)")
        else:
            # Fallback to reading from disk
            input_video = get_video_frames(video_path)
            reprojected = get_video_frames(str(reprojected_video))
            reprojected_mask_arr = get_video_frames(str(reprojected_mask), video_is_grayscale=True)
            probe = ffmpeg.probe(video_path)
            fps = get_video_fps(video_path, probe)
            SF_LOG.info("Read warping frames from disk via FFmpeg fallback")

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
        # Create a hard mask [0, 1] for alpha blending later (no blur, avoids leaking black holes)
        original_mask = reprojected_mask_arr.permute(1, 0, 2, 3).float()

        m2svid_process_res = m2svid_process_res or "1024x576 (Optimal 12GB)"
        if m2svid_process_res != "Native":
            res_str = m2svid_process_res.split(" ")[0]
            tw = int(res_str.split("x")[0])
            # Calculate height proportionally to preserve aspect ratio, preventing geometric distortion
            th = int((orig_shape[0] / orig_shape[1]) * tw)
        else:
            th, tw = orig_shape[0], orig_shape[1]
            
        # SVD requires dimensions to be exactly divisible by 8 for its VAE latent space
        th = th - (th % 8)
        tw = tw - (tw % 8)

        num_samples = gen_chunk_size
        T = input_video.shape[1]

        # Resolution safety check: Parallel chunk batching is only allowed on non-native resolutions to prevent OOM
        is_native_res = (m2svid_process_res == "Native")
        if is_native_res and gen_batch_size > 1:
            SF_LOG.warning(f"[M2SVid] Parallel chunk batching (batch_size={gen_batch_size}) is only available at non-native resolutions. Automatically reducing to batch_size=1 to prevent Native 1080p OOM.")
            print(f"\n[M2SVid] ⚠️ Parallel chunk batching ({gen_batch_size}) is only available at non-native resolutions. Automatically using batch_size=1 for Native 1080p to prevent OOM.\n")
            gen_batch_size = 1
        elif gen_batch_size > 1:
            SF_LOG.info(f"[M2SVid] 🚀 Parallel Chunk Batching active: processing {gen_batch_size} chunks simultaneously at {tw}x{th} (~19-22GB VRAM)")
            print(f"\n[M2SVid] 🚀 Parallel Chunk Batching active: processing {gen_batch_size} chunks simultaneously at {tw}x{th} (~19-22GB VRAM)\n")

        SF_LOG.info(f"[M2SVid] Configuration: resolution={tw}x{th} (setting='{m2svid_process_res}'), chunk_size={num_samples}, chunk_batch_size={gen_batch_size}, total_frames={T}")
        print(f"\n[M2SVid] ⚙️ Processing resolution: {tw}x{th} (setting: '{m2svid_process_res}'), Chunk size: {num_samples}, Parallel batch: {gen_batch_size}, Total frames: {T}")

        if orig_shape[0] != th or orig_shape[1] != tw:
            SF_LOG.info(f"Adjusting inputs for M2SVid from {orig_shape} to {(th, tw)} (must be divisible by 8)")
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
        SF_LOG.info(f"Video length {T} frames; model.max_frames={num_samples}")

        def _save_video(video_tensor, fps_val, path):
            T_val = video_tensor.shape[2]
            H, W = video_tensor.shape[3], video_tensor.shape[4]
            writer = FFmpegVideoWriter(path, W, H, fps_val, is_color=True)
            if not writer.is_valid():
                raise RuntimeError(f"Failed to open video writer for {path}")

            for i in range(T_val):
                frame = video_tensor[0, :, i, :, :] # [C, H, W]
                frame = ((frame + 1.0) / 2.0).clamp(0, 1) * 255.0
                frame_np = frame.to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
                writer.write(frame_np)

            writer.close()
            written = os.path.getsize(path) if os.path.exists(path) else 0
            SF_LOG.info(f"Wrote {written} bytes to {path}")
            if written < 1024:
                try: os.remove(path)
                except: pass
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
            print(f"\n[M2SVid] ⏳ Executing heavy AI video generation for {T} frames...")
            print(f"[M2SVid] ⏳ Please wait. This can take anywhere from 15s to 90s depending on your GPU...")
            t0 = time.time()
            clear_cuda()
            try:
                if progress:
                    progress(0.55 if not T else 0.55, desc=f"{progress_prefix}Running 1-step conditioned generation (heavy)...")
            except TypeError:
                pass
            with torch.inference_mode():
                final_generated = model.generate(input_batch)["generated-video"][0].cpu()
            del input_batch
            clear_cuda()
            t1 = time.time()
            print(f"[M2SVid] ✅ Generation finished in {t1 - t0:.1f} seconds!")
            SF_LOG.info(f"Single-chunk generation complete in {t1 - t0:.1f}s")
        else:
            total_chunks = (T + num_samples - 1) // num_samples
            chunk_outputs = []
            SF_LOG.info(f"Video has {T} frames, using chunking: {total_chunks} chunks (padded to {num_samples} frames each, parallel_batch_size={gen_batch_size})")
            for b_idx in range(0, total_chunks, gen_batch_size):
                global GLOBAL_CANCEL
                if GLOBAL_CANCEL:
                    GLOBAL_CANCEL = False
                    raise RuntimeError("Stopped by user. VRAM flushed.")
                
                curr_batch_chunks = []
                curr_chunk_lens = []
                for sub_idx in range(b_idx, min(b_idx + gen_batch_size, total_chunks)):
                    s = sub_idx * num_samples
                    e = min(s + num_samples, T)
                    chunk_len = e - s
                    pad_len = num_samples - chunk_len
                    SF_LOG.info(f"Chunk {sub_idx+1}/{total_chunks}: frames {s}-{e}{f' (padding {pad_len})' if pad_len else ''}")
                    if pad_len > 0:
                        input_chunk = torch.nn.functional.pad(input_video[:, s:e, :, :], (0, 0, 0, 0, 0, pad_len))
                        reproj_chunk = torch.nn.functional.pad(reprojected[:, s:e, :, :], (0, 0, 0, 0, 0, pad_len))
                        mask_chunk = torch.nn.functional.pad(reprojected_mask_t[:, s:e, :, :], (0, 0, 0, 0, 0, pad_len))
                    else:
                        input_chunk = input_video[:, s:e, :, :]
                        reproj_chunk = reprojected[:, s:e, :, :]
                        mask_chunk = reprojected_mask_t[:, s:e, :, :]
                    
                    curr_batch_chunks.append({
                        "video": input_chunk,
                        "reprojected_video": reproj_chunk,
                        "reprojected_mask": mask_chunk,
                    })
                    curr_chunk_lens.append(chunk_len)

                B_curr = len(curr_batch_chunks)
                input_batch = {
                    "video": torch.stack([c["video"] for c in curr_batch_chunks], dim=0).cuda(),
                    "video_2nd_view": torch.stack([c["video"] for c in curr_batch_chunks], dim=0).cuda(),
                    "reprojected_video": torch.stack([c["reprojected_video"] for c in curr_batch_chunks], dim=0).cuda(),
                    "reprojected_mask": torch.stack([c["reprojected_mask"] for c in curr_batch_chunks], dim=0).cuda(),
                    "fps_id": torch.tensor([fps] * B_curr).cuda(),
                    "caption": [""] * B_curr,
                    "motion_bucket_id": torch.tensor([127] * B_curr).cuda(),
                }
                t0 = time.time()
                chunk_range_str = f"Chunk(s) {b_idx+1}-{b_idx+B_curr}/{total_chunks}" if B_curr > 1 else f"Chunk {b_idx+1}/{total_chunks}"
                print(f"\n[M2SVid] ⏳ Executing AI generation for {chunk_range_str} ({B_curr} parallel chunk{'s' if B_curr > 1 else ''})... Please wait.")
                try:
                    if progress:
                        progress(0.55 + 0.30 * (b_idx / total_chunks), desc=f"{progress_prefix}Generating {chunk_range_str}...")
                except TypeError:
                    pass
                with torch.inference_mode():
                    gen_batch = model.generate(input_batch)["generated-video"].cpu()
                del input_batch
                clear_cuda()
                t1 = time.time()
                sys.stdout.flush()
                print(f"[M2SVid] ✅ {chunk_range_str} finished in {t1 - t0:.1f} seconds!")
                SF_LOG.info(f"{chunk_range_str} done in {t1 - t0:.1f}s")
                
                for b_sub in range(B_curr):
                    c_len = curr_chunk_lens[b_sub]
                    gen_chunk = gen_batch[b_sub][:, :c_len, :, :]
                    chunk_outputs.append(gen_chunk)
                del gen_batch
                clear_cuda()
            final_generated = torch.cat(chunk_outputs, dim=1)
            del chunk_outputs

        # Chunked blending and upsampling in small batches (16 frames)
        # to completely eliminate the 21GB/42GB contiguous CPU RAM allocation crash!
        T_total = input_video.shape[1]
        orig_w = orig_shape[1]
        shift = int(orig_w * eff_disp_perc * convergence_point)
        
        final_uint8_frames = []
        orig_left_uint8 = []
        
        for i in range(0, T_total, 16):
            i_end = min(i + 16, T_total)
            fg_sub = final_generated[:, i:i_end]
            if fg_sub.shape[-2:] != orig_shape:
                fg_sub = torch.nn.functional.interpolate(fg_sub, size=orig_shape, mode="bilinear", align_corners=False)
            
            rep_sub = original_reprojected[:, i:i_end]
            mask_sub = original_mask[:, i:i_end]
            
            # Blend hallucinated pixels ONLY into disocclusion holes
            blended = rep_sub * (1.0 - mask_sub) + fg_sub * mask_sub
            
            # Convert directly to uint8 [0, 255]
            blended_u8 = (((blended + 1.0) / 2.0).clamp(0, 1) * 255.0).to(torch.uint8)
            left_u8 = (((original_input_video[:, i:i_end] + 1.0) / 2.0).clamp(0, 1) * 255.0).to(torch.uint8)
            
            final_uint8_frames.append(blended_u8)
            orig_left_uint8.append(left_u8)
            
        del final_generated, original_reprojected, original_mask, original_input_video
        right_video_u8 = torch.cat(final_uint8_frames, dim=1) # [C, T, H, W] in uint8
        left_video_u8 = torch.cat(orig_left_uint8, dim=1)
        del final_uint8_frames, orig_left_uint8
        
        # Mathematical Convergence Point (Zero Parallax) Shift
        if shift > 0:
            left_video_u8 = left_video_u8[:, :, :, shift:]
            right_video_u8 = right_video_u8[:, :, :, :-shift]
            
        # Ensure outputs are padded back to 16:9 standard resolution for hardware compatibility
        c, t, h, w = right_video_u8.shape
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
        
        padded_left = torch.nn.functional.pad(left_video_u8, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        padded_right = torch.nn.functional.pad(right_video_u8, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        del left_video_u8, right_video_u8
        
        SF_LOG.info(f"Padded output from {w}x{h} to 16:9 standard ({target_w}x{target_h})")

        generated_right = out_dir / f"{stem}_generated_right.mp4"
        sbs = out_dir / f"{stem}_stereo_sbs.mp4"
        anaglyph = out_dir / f"{stem}_anaglyph.mp4"

        # Stream write right, SBS, and red/cyan anaglyph videos in a single unified pass
        T_val = padded_left.shape[1]
        out_H = padded_left.shape[2]
        out_W = padded_left.shape[3]

        w_right = FFmpegVideoWriter(str(generated_right), out_W, out_H, fps, is_color=True)
        w_sbs = FFmpegVideoWriter(str(sbs), out_W * 2, out_H, fps, is_color=True)
        w_ana = FFmpegVideoWriter(str(anaglyph), out_W, out_H, fps, is_color=True)

        if not (w_right.is_valid() and w_sbs.is_valid() and w_ana.is_valid()):
            w_right.close()
            w_sbs.close()
            w_ana.close()
            raise RuntimeError("Failed to open VideoWriters for stereo output.")

        SF_LOG.info(f"Streaming {T_val} frames @ {out_W}x{out_H} into stereo video files (H.264 via FFmpeg)...")

        # Single bulk DMA transfer to host memory (avoids T_val separate GPU pipeline stalls)
        padded_left_np = padded_left.permute(1, 2, 3, 0).contiguous().cpu().numpy()
        padded_right_np = padded_right.permute(1, 2, 3, 0).contiguous().cpu().numpy()
        del padded_left, padded_right

        for i in range(T_val):
            left_f = padded_left_np[i]
            right_f = padded_right_np[i]

            # 1. Right eye view
            w_right.write(right_f)

            # 2. Side-by-Side (Left | Right)
            sbs_f = np.concatenate([left_f, right_f], axis=1)
            w_sbs.write(sbs_f)

            # 3. Red-Cyan Anaglyph: Left Red, Right Green/Blue
            ana_f = np.empty_like(left_f)
            ana_f[:, :, 0] = left_f[:, :, 0]   # Red from Left
            ana_f[:, :, 1] = right_f[:, :, 1]  # Green from Right
            ana_f[:, :, 2] = right_f[:, :, 2]  # Blue from Right
            w_ana.write(ana_f)

        w_right.close()
        w_sbs.close()
        w_ana.close()
        del padded_left_np, padded_right_np
        SF_LOG.info("All stereo video outputs completed successfully.")

        # Audio Passthrough: Copy source audio stream to stereo outputs if present
        try:
            from pipeline_sidecar import remux_audio_if_present
            remux_audio_if_present(video_path, str(generated_right))
            remux_audio_if_present(video_path, str(sbs))
            remux_audio_if_present(video_path, str(anaglyph))
        except Exception as _e_audio:
            SF_LOG.warning(f"Audio passthrough failed: {_e_audio}")

        if progress:
            progress(1.0, desc=f"{progress_prefix}M2SVid stage complete!")
        clear_cuda()

        status = "✅ M2SVid conversion finished. Download the results below."
        return status, str(generated_right) if os.path.exists(generated_right) else None, str(sbs) if os.path.exists(sbs) else None, str(anaglyph) if anaglyph and os.path.exists(anaglyph) else None, str(out_dir)


    except Exception as e:
        import traceback
        traceback.print_exc()
        clear_cuda()
        return f"❌ Error in M2SVid step: {str(e)}", None, None, None, None


# =============================================================================
# Gradio Interface
# =============================================================================


# =============================================================================
# Gradio Interface
# =============================================================================

def _extract_file_path_and_name(f):
    if isinstance(f, dict):
        src = f.get("name") or f.get("path") or ""
        name = f.get("orig_name") or (Path(src).name if src else "")
    elif hasattr(f, "name"):
        src = f.name
        name = getattr(f, "orig_name", Path(src).name if src else "")
    else:
        src = str(f)
        name = Path(src).name if src else ""
    return src, name


def handle_source_upload(files):
    if not files:
        return None, gr.update()
    if not isinstance(files, list):
        files = [files]
    last_dst = None
    last_stem = ""
    for f in files:
        src_path, orig_name = _extract_file_path_and_name(f)
        if not src_path or not os.path.exists(src_path):
            continue
        dst_path = SOURCE_DIR / orig_name
        shutil.copy(src_path, str(dst_path))
        SF_LOG.info(f"Uploaded source video saved to {dst_path}")
        last_dst = str(dst_path)
        last_stem = Path(orig_name).stem
    stems = get_source_video_list()
    return last_dst, gr.update(choices=[""] + stems, value=last_stem)


def handle_depth_upload(files):
    if not files:
        return None, gr.update()
    if not isinstance(files, list):
        files = [files]
    last_dst = None
    last_base_stem = ""
    for f in files:
        src_path, orig_name = _extract_file_path_and_name(f)
        if not src_path or not os.path.exists(src_path):
            continue
        dst_path = DEPTH_DIR / orig_name
        shutil.copy(src_path, str(dst_path))
        SF_LOG.info(f"Uploaded depth map video saved to {dst_path}")
        stem = Path(orig_name).stem
        base_stem = stem[:-6] if stem.endswith("_depth") else stem
        out_npz = DEPTH_DIR / f"{base_stem}_depth.npz"
        try:
            convert_depth_video_to_npz(str(dst_path), str(out_npz))
        except Exception as e:
            SF_LOG.error(f"Failed to automatically convert depth video to npz: {e}")
        last_dst = str(dst_path)
        last_base_stem = base_stem
    stems = get_source_video_list()
    return last_dst, gr.update(choices=[""] + stems, value=last_base_stem)


def get_depth_video_list():
    """Return list of depth video stems with model suffix info available in depthmaps_videos folder."""
    if not DEPTH_DIR.exists():
        return []
    vids = sorted([p.stem for p in DEPTH_DIR.iterdir() 
                   if p.is_file() and p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')])
    return vids


SUFFIX_TO_LABEL = {
    "_VDAL": "Video Depth Anything (Large)",
    "_VDAB": "Video Depth Anything (Base)",
    "_VDA": "Video Depth Anything",
    "_DC": "DepthCrafter",
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


STEREOFASTER_HEAD_HTML = """
<style>
    /* Enhanced ImageEditor Brush & Eraser Size Sliders: 3x wider track with high precision */
    .wrap.size_picker, .wrap.eraser_picker, div[class*="size_picker"], div[class*="eraser_picker"], .wrap.svelte-1y25b79 {
        min-width: 340px !important;
        width: 340px !important;
        max-width: 90vw !important;
        padding: 12px 18px !important;
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6) !important;
        border-radius: 8px !important;
        box-sizing: border-box !important;
        z-index: 9999 !important;
    }
    .wrap.size_picker .wrap, .wrap.eraser_picker .wrap, div[class*="size_picker"] .wrap, div[class*="eraser_picker"] .wrap {
        width: 100% !important;
    }
    .wrap.size_picker input[type="range"], .wrap.eraser_picker input[type="range"], div[class*="size_picker"] input[type="range"], div[class*="eraser_picker"] input[type="range"], .wrap.svelte-1y25b79 input[type="range"] {
        width: 100% !important;
        min-width: 280px !important;
        height: 10px !important;
        cursor: pointer !important;
        accent-color: #0284c7 !important;
    }
    .sf-popover-pills {
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
        margin-bottom: 10px !important;
        flex-wrap: wrap !important;
    }
    .sf-popover-pill {
        padding: 3px 8px !important;
        border-radius: 4px !important;
        border: 1px solid #475569 !important;
        background: #1e293b !important;
        color: #f1f5f9 !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
    }
    .sf-popover-pill:hover {
        background: #334155 !important;
        border-color: #38bdf8 !important;
    }
    .sf-popover-pill.active {
        background: #0284c7 !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        font-weight: bold !important;
    }
</style>
<script>
(function() {
    if (window.sfBrushControllerInitialized) return;
    window.sfBrushControllerInitialized = true;

    window.sfBrushSize = 4;
    window.sfEraserSize = 4;
    window.sfCurrentTool = 'draw';
    window.sfBrushOpacity = 0.50;

    function showToast(msg) {
        let toast = document.getElementById('sf-brush-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'sf-brush-toast';
            toast.style.cssText = 'position:fixed; bottom:24px; right:24px; background:rgba(15,23,42,0.92); border:1px solid #38bdf8; color:#f8fafc; font-weight:700; font-size:13px; padding:6px 14px; border-radius:6px; box-shadow:0 6px 20px rgba(0,0,0,0.5); z-index:999999; pointer-events:none; transition:opacity 0.2s ease, transform 0.2s ease; transform:translateY(0); opacity:0;';
            document.body.appendChild(toast);
        }
        toast.textContent = msg;
        toast.style.opacity = '1';
        toast.style.transform = 'translateY(0)';
        if (window._sfToastTimer) clearTimeout(window._sfToastTimer);
        window._sfToastTimer = setTimeout(function() {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(6px)';
        }, 850);
    }

    function getBrushTool() {
        if (window.editor && window.editor.tools) {
            return window.editor.tools.get('brush');
        }
        return null;
    }

    function getCurrentToolMode() {
        if (window.sfCurrentTool === 'erase') return 'erase';
        const brush = getBrushTool();
        if (brush && brush.state && brush.state.mode) {
            return brush.state.mode;
        }
        const eraserBtn = document.querySelector('#depth_paint_canvas button[aria-label="Eraser Size"]');
        if (eraserBtn) return 'erase';
        return 'draw';
    }

    function syncToolState() {
        const mode = getCurrentToolMode();
        window.sfCurrentTool = mode;
        const isEraser = (mode === 'erase');
        const size = isEraser ? (window.sfEraserSize || 4) : (window.sfBrushSize || 4);

        const modeLabel = document.getElementById('sf-mode-label');
        if (modeLabel) {
            modeLabel.innerText = isEraser ? '🧹 Eraser Size:' : '🖌️ Brush Size:';
            modeLabel.style.color = isEraser ? '#f59e0b' : '#38bdf8';
        }

        const valDisplay = document.getElementById('sf-brush-val');
        if (valDisplay) {
            valDisplay.innerText = size + ' px';
            valDisplay.style.color = isEraser ? '#f59e0b' : '#38bdf8';
        }

        const toolBrushBtn = document.getElementById('sf-tool-brush-btn');
        const toolEraserBtn = document.getElementById('sf-tool-eraser-btn');
        if (toolBrushBtn && toolEraserBtn) {
            if (isEraser) {
                toolEraserBtn.style.background = '#d97706';
                toolEraserBtn.style.color = '#ffffff';
                toolEraserBtn.style.fontWeight = 'bold';
                toolBrushBtn.style.background = 'transparent';
                toolBrushBtn.style.color = '#94a3b8';
                toolBrushBtn.style.fontWeight = 'normal';
            } else {
                toolBrushBtn.style.background = '#0284c7';
                toolBrushBtn.style.color = '#ffffff';
                toolBrushBtn.style.fontWeight = 'bold';
                toolEraserBtn.style.background = 'transparent';
                toolEraserBtn.style.color = '#94a3b8';
                toolEraserBtn.style.fontWeight = 'normal';
            }
        }

        document.querySelectorAll('.sf-brush-btn').forEach(function(btn) {
            const s = parseInt(btn.getAttribute('data-size'));
            if (s === size) {
                btn.style.border = isEraser ? '1px solid #f59e0b' : '1px solid #38bdf8';
                btn.style.background = isEraser ? '#d97706' : '#0284c7';
                btn.style.color = '#ffffff';
                btn.style.fontWeight = 'bold';
            } else {
                btn.style.border = '1px solid #475569';
                btn.style.background = '#0f172a';
                btn.style.color = '#f1f5f9';
                btn.style.fontWeight = 'normal';
            }
        });

        document.querySelectorAll('.sf-popover-pill').forEach(function(b) {
            const s = parseInt(b.getAttribute('data-size'));
            if (s === size) {
                b.classList.add('active');
                b.style.border = isEraser ? '1px solid #f59e0b' : '1px solid #38bdf8';
                b.style.background = isEraser ? '#d97706' : '#0284c7';
            } else {
                b.classList.remove('active');
                b.style.border = '1px solid #475569';
                b.style.background = '#1e293b';
            }
        });
    }

    window.sfSetTool = function(tool) {
        window.sfCurrentTool = tool;
        const editor = document.querySelector('#depth_paint_canvas');
        if (editor) {
            const ariaTarget = (tool === 'erase') ? 'Erase' : 'Brush';
            const btn = editor.querySelector('button[aria-label="' + ariaTarget + '"]');
            if (btn) btn.click();
        }
        if (window.editor && typeof window.editor.set_tool === 'function') {
            window.editor.set_tool(tool);
        }
        setTimeout(syncToolState, 40);
        showToast(tool === 'erase' ? '🧹 Switched to Eraser' : '🖌️ Switched to Brush');
    };

    function updateOpacityUI(opacity) {
        window.sfBrushOpacity = opacity;
        const pct = Math.round(opacity * 100);
        const valDisplay = document.getElementById('sf-brush-opacity-val');
        if (valDisplay) valDisplay.innerText = pct + '%';

        document.querySelectorAll('.sf-opacity-btn').forEach(function(btn) {
            const p = parseInt(btn.getAttribute('data-pct'));
            if (Math.abs(p - pct) < 5) {
                btn.style.border = '1px solid #a78bfa';
                btn.style.background = '#7c3aed';
                btn.style.color = '#ffffff';
                btn.style.fontWeight = 'bold';
            } else {
                btn.style.border = '1px solid #475569';
                btn.style.background = '#0f172a';
                btn.style.color = '#f1f5f9';
                btn.style.fontWeight = 'normal';
            }
        });
    }

    function sfApplyValToRange(slider, val) {
        val = Math.max(1, Math.min(100, Math.round(val)));
        try {
            const desc = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
            if (desc && desc.set) {
                desc.set.call(slider, val);
            } else {
                slider.value = val;
            }
        } catch(err) {
            slider.value = val;
        }
        slider.dispatchEvent(new Event('input', { bubbles: true }));
        slider.dispatchEvent(new Event('change', { bubbles: true }));
    }

    window.sfSetSize = function(size) {
        size = Math.max(1, Math.min(100, Math.round(size)));
        const isEraser = (getCurrentToolMode() === 'erase');

        if (isEraser) {
            window.sfEraserSize = size;
            showToast('🧹 Eraser: ' + size + ' px');
        } else {
            window.sfBrushSize = size;
            showToast('🖌️ Brush: ' + size + ' px');
        }

        syncToolState();

        const brush = getBrushTool();
        if (brush) {
            if (isEraser && typeof brush.set_eraser_size === 'function') {
                brush.set_eraser_size(size);
            } else if (!isEraser && typeof brush.set_brush_size === 'function') {
                brush.set_brush_size(size);
            }
        }

        const range = document.querySelector(
            isEraser
                ? '#depth_paint_canvas .eraser_picker input[type="range"], .eraser_picker input[type="range"]'
                : '#depth_paint_canvas .size_picker input[type="range"], .size_picker input[type="range"]'
        ) || document.querySelector('#depth_paint_canvas input[type="range"]');

        if (range) {
            sfApplyValToRange(range, size);
            return;
        }

        const editor = document.querySelector('#depth_paint_canvas');
        if (!editor) return;

        const targetLabel = isEraser ? 'Eraser Size' : 'Brush Size';
        let sizeBtn = editor.querySelector('button[aria-label="' + targetLabel + '"]');

        if (!sizeBtn) {
            const buttons = editor.querySelectorAll('button');
            for (let i = 0; i < buttons.length; i++) {
                const b = buttons[i];
                const aria = (b.getAttribute('aria-label') || '').toLowerCase();
                const title = (b.getAttribute('title') || '').toLowerCase();
                if (isEraser && (aria.includes('eraser') || title.includes('eraser')) && (aria.includes('size') || title.includes('size'))) {
                    sizeBtn = b;
                    break;
                } else if (!isEraser && (aria.includes('brush') || title.includes('brush')) && (aria.includes('size') || title.includes('size'))) {
                    sizeBtn = b;
                    break;
                }
            }
        }

        if (sizeBtn) {
            sizeBtn.click();
            setTimeout(function() {
                const newRange = document.querySelector(
                    isEraser
                        ? '#depth_paint_canvas .eraser_picker input[type="range"], .eraser_picker input[type="range"]'
                        : '#depth_paint_canvas .size_picker input[type="range"], .size_picker input[type="range"]'
                ) || document.querySelector('#depth_paint_canvas input[type="range"]');
                if (newRange) sfApplyValToRange(newRange, size);
                sizeBtn.click();
            }, 30);
        }
    };

    window.sfSetBrushSize = window.sfSetSize;
    window.sfSetEraserSize = function(size) {
        window.sfCurrentTool = 'erase';
        window.sfSetSize(size);
    };

    window.sfAdjustSize = function(delta) {
        const isEraser = (getCurrentToolMode() === 'erase');
        const cur = isEraser ? (window.sfEraserSize || 4) : (window.sfBrushSize || 4);
        window.sfSetSize(cur + delta);
    };

    window.sfAdjustBrush = window.sfAdjustSize;

    window.sfSetBrushOpacity = function(opacity) {
        opacity = Math.max(0.05, Math.min(1.0, Math.round(Number(opacity) * 100) / 100));
        updateOpacityUI(opacity);
        showToast('👁️ Opacity: ' + Math.round(opacity * 100) + '%');

        const brush = getBrushTool();
        if (brush && typeof brush.set_brush_opacity === 'function') {
            brush.set_brush_opacity(opacity);
        }
    };

    window.sfAdjustBrushOpacity = function(delta) {
        const cur = window.sfBrushOpacity !== undefined ? window.sfBrushOpacity : 0.50;
        window.sfSetBrushOpacity(cur + delta);
    };

    function ensureBrushSettings() {
        const brush = getBrushTool();
        if (brush) {
            if (window.sfBrushSize && brush.brush_size !== window.sfBrushSize) {
                brush.set_brush_size(window.sfBrushSize);
            }
            if (window.sfEraserSize && brush.eraser_size !== window.sfEraserSize) {
                brush.set_eraser_size(window.sfEraserSize);
            }
            if (window.sfBrushOpacity !== undefined && brush.state && Math.abs(brush.state.opacity - window.sfBrushOpacity) > 0.02) {
                brush.set_brush_opacity(window.sfBrushOpacity);
            }
        }
    }

    document.addEventListener('pointerdown', function(e) {
        if (e.target && e.target.closest && e.target.closest('#depth_paint_canvas')) {
            ensureBrushSettings();
        }
    }, true);

    document.addEventListener('click', function(e) {
        if (e.target && e.target.closest && e.target.closest('#depth_paint_canvas button')) {
            setTimeout(syncToolState, 50);
        }
    }, true);

    // Hotkeys: [ and ] for size, { and } (Shift+[ / Shift+]) for opacity
    document.addEventListener('keydown', function(e) {
        const active = document.activeElement;
        if (active && ['INPUT', 'TEXTAREA'].includes(active.tagName) && active.type !== 'range') return;
        if (active && active.isContentEditable) return;

        if (e.key === '{' || (e.shiftKey && e.key === '[')) {
            e.preventDefault();
            window.sfAdjustBrushOpacity(-0.10);
        } else if (e.key === '}' || (e.shiftKey && e.key === ']')) {
            e.preventDefault();
            window.sfAdjustBrushOpacity(0.10);
        } else if (e.key === '[') {
            e.preventDefault();
            window.sfAdjustSize(-1);
        } else if (e.key === ']') {
            e.preventDefault();
            window.sfAdjustSize(1);
        }
    });

    // Delegated input listener for manual slider changes
    document.addEventListener('input', function(e) {
        if (e.target && e.target.type === 'range') {
            const container = e.target.closest('#depth_paint_canvas, .size_picker, .eraser_picker, .wrap.svelte-1y25b79');
            if (container) {
                const val = parseInt(e.target.value);
                if (!isNaN(val)) {
                    const isEraser = container.classList.contains('eraser_picker') || (getCurrentToolMode() === 'erase');
                    if (isEraser) {
                        window.sfEraserSize = val;
                    } else {
                        window.sfBrushSize = val;
                    }
                    syncToolState();
                }
            }
        }
    }, true);

    function enhancePopover(popover) {
        const range = popover.querySelector('input[type="range"]');
        if (range && !popover.dataset.sfEnhanced) {
            popover.dataset.sfEnhanced = "true";
            range.style.width = "270px";
            range.style.cursor = "pointer";

            const isEraser = popover.classList.contains('eraser_picker') || (getCurrentToolMode() === 'erase');
            const currentSize = isEraser ? (window.sfEraserSize || 4) : (window.sfBrushSize || 4);

            sfApplyValToRange(range, currentSize);

            const internalBar = document.createElement('div');
            internalBar.className = 'sf-popover-pills';

            const pLabel = document.createElement('span');
            pLabel.innerText = isEraser ? '🧹 Eraser Presets:' : '🖌️ Brush Presets:';
            pLabel.style.cssText = (isEraser ? 'color:#f59e0b;' : 'color:#38bdf8;') + ' font-size:11px; font-weight:600; margin-right:2px;';
            internalBar.appendChild(pLabel);

            const sizes = [1, 2, 4, 6, 10, 16, 25];
            sizes.forEach(function(sz) {
                const pill = document.createElement('button');
                pill.type = 'button';
                pill.className = 'sf-popover-pill' + (sz === currentSize ? ' active' : '');
                pill.setAttribute('data-size', sz);
                pill.innerText = sz + 'px';
                if (isEraser && sz === currentSize) {
                    pill.style.background = '#d97706';
                    pill.style.borderColor = '#f59e0b';
                }
                pill.onclick = function(ev) {
                    ev.stopPropagation();
                    ev.preventDefault();
                    window.sfSetSize(sz);
                };
                internalBar.appendChild(pill);
            });

            popover.insertBefore(internalBar, popover.firstChild);
        }
    }

    const sfObserver = new MutationObserver(function() {
        const popovers = document.querySelectorAll('.size_picker, .eraser_picker, .wrap.svelte-1y25b79, div[class*="size_picker"], div[class*="eraser_picker"]');
        for (let i = 0; i < popovers.length; i++) {
            enhancePopover(popovers[i]);
        }
    });

    if (document.body) {
        sfObserver.observe(document.body, { childList: true, subtree: true });
    } else {
        document.addEventListener('DOMContentLoaded', function() {
            sfObserver.observe(document.body, { childList: true, subtree: true });
        });
    }
})();
</script>

    if (document.body) {
        sfObserver.observe(document.body, { childList: true, subtree: true });
    } else {
        document.addEventListener('DOMContentLoaded', function() {
            sfObserver.observe(document.body, { childList: true, subtree: true });
        });
    }
})();
</script>

    if (document.body) {
        sfObserver.observe(document.body, { childList: true, subtree: true });
    } else {
        document.addEventListener('DOMContentLoaded', function() {
            sfObserver.observe(document.body, { childList: true, subtree: true });
        });
    }
})();
</script>
"""


def create_stereofaster_ui():
    # Detect default starting file for playback
    stems = get_source_video_list()
    default_video = None
    default_stem = ""
    default_depth_mp4 = None
    default_depth_npz = None
    
    if stems:
        default_stem = stems[0]
        v_p, d_npz, d_mp4 = select_source_video(default_stem)
        default_video = v_p
        default_depth_npz = d_npz
        default_depth_mp4 = d_mp4

    default_preset = "16:9 (None)"
    default_res = ""
    default_crop_prev = None
    if default_video:
        default_preset, default_res, default_crop_prev = analyze_letterbox(default_video)

    init_sidecar = load_sidecar_file(SIDECAR_CONFIG_PATH)
    init_dc_cfg = init_sidecar.get("depthcrafter_settings", {})
    init_icd_cfg = init_sidecar.get("icdepth_settings", {})
    init_fd_cfg = init_sidecar.get("flashdepth_settings", {})
    init_vda_cfg = init_sidecar.get("vda_settings", {})
    init_m2s_cfg = init_sidecar.get("m2svid_settings", {})
    init_stages = init_sidecar.get("stages_enabled", {})
    init_prof = init_sidecar.get("hardware_profile", detect_gpu_hardware())
    init_rate = float(init_sidecar.get("cloud_vast_rate_per_hour", 0.50))
    init_depth_prio = init_sidecar.get("depth_priority", "DepthCrafter")
    init_audio_pass = init_sidecar.get("audio_passthrough", {}).get("enabled", True)
    init_json_str = json.dumps(init_sidecar, indent=2)
    init_table_rows, init_summary_md, _ = scan_source_library(SOURCE_DIR, DEPTH_DIR, FINAL_DIR)

    with gr.Blocks(title="StereoFaster Hub", fill_width=True) as demo:
        demo.head = STEREOFASTER_HEAD_HTML
        gr.HTML(
            """
            <style>
                .gradio-container { max-width: 100% !important; padding: 0 12px !important; }
                .contain { max-width: 100% !important; padding: 0 !important; }
                #component-0 { max-width: 100% !important; padding: 0 !important; }
                .tab-nav { flex-wrap: wrap !important; gap: 4px !important; }
                .tab-nav button { font-weight: 600 !important; padding: 10px 18px !important; }
            </style>
            <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #101827, #0B2545); border-radius: 12px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.05);'>
                <h1 style='color: #00F5FF; font-family: "Outfit", sans-serif; font-size: 2.8em; margin: 0; text-shadow: 0 0 20px rgba(0,245,255,0.3); font-weight: 800;'>StereoFaster</h1>
                <p style='color: #8D99AE; font-size: 1.1em; margin-top: 5px; font-weight: 300;'>DepthCrafter • ICDepth (Wan2.1 In-Context DiT) • M2SVid Stereography</p>
            </div>
            """
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(f"### ⚙️ Engine Status: `{get_gpu_info()}`")
            with gr.Column(scale=1):
                show_alt_tabs_chk = gr.Checkbox(
                    label="🔧 Show FlashDepth & VDA Tabs",
                    value=False,
                )
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
                            label="Upload Source Video(s) (saved to source_videos/)",
                            file_types=["video"],
                            file_count="multiple",
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
                        
                        with gr.Accordion("✂️ Auto-Crop Aspect Ratio", open=True):
                            crop_preset = gr.Dropdown(choices=list(ASPECT_RATIO_PRESETS.keys()), value=default_preset, label="Detected Preset", interactive=True)
                            crop_res = gr.Textbox(value=default_res, label="Remaining Resolution", interactive=False)
                            toggle_preview_btn = gr.Button("🔄 Toggle View: Full vs Cropped Preview")
                            apply_crop_btn = gr.Button("✂️ Apply Crop and Save as New Video", variant="primary")
                            gr.Markdown("---")
                            force_crop_preset = gr.Checkbox(label="Only use the aspect ratio from the 'Detected Preset' dropdown for batch cropping", value=False)
                            batch_crop_btn = gr.Button("✂️ Run Batch Auto-crop for All Source Videos", variant="secondary")
                            crop_preview_state = gr.State(default_crop_prev)
                        
                    with gr.Column():
                        gr.Markdown("##### 🗂️ Depth Map Management")
                        depth_uploader = gr.File(
                            label="Upload External Depth Video(s) (saved to depthmaps_videos/)",
                            file_types=["video"],
                            file_count="multiple",
                            type="filepath"
                        )
                        depth_dropdown = gr.Dropdown(
                            choices=[""] + get_depth_video_list(),
                            value="",
                            label="Select Depth Map from Folder",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        preview_depth = gr.Video(
                            value=default_depth_mp4,
                            label="Depth Map Preview",
                            interactive=False,
                            format="mp4"
                        )
                        
                        with gr.Accordion("🎬 Title Depth Tools (Solid Background Void)", open=False):
                            gr.Markdown("Generate instant, artifact-free depth maps for title cards and logos on any solid background (black, white, green screen, blue screen, etc.) without AI.")
                            with gr.Tab("🔲 Title Desaturation (Flat 2D Titles)"):
                                desat_clip_dropdown = gr.Dropdown(
                                    choices=[""] + stems,
                                    value=default_stem,
                                    label="Select Title Clip (from source_videos/)",
                                    interactive=True
                                )
                                with gr.Row():
                                    desat_auto_bg = gr.Checkbox(label="Auto-Detect Solid Background Color (Keys out Black, White, Green, etc.)", value=True)
                                    desat_invert = gr.Checkbox(label="Invert Depth (Reverses near/far)", value=False)
                                desat_gen_btn = gr.Button("🔲 Generate Flat Title Depth", variant="primary")
                                desat_status = gr.Textbox(label="Status", interactive=False, lines=2)
                            with gr.Tab("🏺 3D Title Isolation (Sculpted / Claymation)"):
                                iso_clip_dropdown = gr.Dropdown(
                                    choices=[""] + stems,
                                    value=default_stem,
                                    label="Select Title Clip (from source_videos/)",
                                    interactive=True
                                )
                                with gr.Row():
                                    iso_color_target = gr.Dropdown(
                                        choices=["Universal (Auto Solid BG)", "Red (Claymation)", "Green", "Blue", "Universal Luminance"],
                                        value="Universal (Auto Solid BG)",
                                        label="Target Color / Isolation Mode"
                                    )
                                with gr.Row():
                                    iso_auto_bg = gr.Checkbox(label="Auto-Detect Solid Background Color", value=True)
                                    iso_invert = gr.Checkbox(label="Invert Depth (Reverses near/far)", value=False)
                                with gr.Row():
                                    iso_base_elev = gr.Slider(0.0, 1.0, value=0.60, step=0.05, label="Base Elevation", info="Floating plane height")
                                    iso_relief = gr.Slider(0.0, 0.5, value=0.32, step=0.02, label="Surface Relief Scale", info="Sculpted contour depth")
                                iso_gen_btn = gr.Button("🏺 Generate 3D Sculpted Title Depth", variant="primary")
                                iso_status = gr.Textbox(label="Status", interactive=False, lines=2)

            # ===================== TAB 4: STEP 1 - ICDEPTH =====================
            with gr.Tab("💠 Step 1 - ICDepth"):
                gr.Markdown("### 💠 Step 1 — ICDepth (Wan2.1 In-Context DiT)\n#### Diffusion Transformer Video Depth Estimator: In-Context Video Depth using Alibaba Wan2.1 DiT + DINOv2 prior with Hann-window latent blending and directional halftone descreening.")
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            icd_max_res = gr.Slider(
                                384, 1920,
                                value=720,
                                step=16,
                                label="Max Inference Resolution (Longest Edge, Multiple of 16)",
                                info="Resolution passed to Wan2.1 DiT (unlocked up to 1920 for native 1080p). Check the GPU advisory below."
                            )
                            icd_steps = gr.Slider(
                                2, 20,
                                value=5,
                                step=1,
                                label="Inference Steps",
                                info="FlowMatch diffusion steps. 5 is recommended; 2-3 for faster previews."
                            )
                        icd_res_advisory = gr.HTML(
                            value=get_icdepth_res_advisory(720),
                            elem_id="icd_res_advisory"
                        )
                        with gr.Row():
                            icd_window_size = gr.Slider(
                                41, 121,
                                value=81,
                                step=4,
                                label="Temporal Window Size (Frames)",
                                info="Sliding window size in frames (aligned to 4k+1). Default 81."
                            )
                            icd_overlap = gr.Slider(
                                16, 64,
                                value=32,
                                step=4,
                                label="Window Overlap (Frames)",
                                info="Overlap between consecutive windows (aligned to 4k). Default 32."
                            )
                        with gr.Row():
                            icd_cfg_scale = gr.Slider(
                                1.0, 7.0,
                                value=5.0,
                                step=0.5,
                                label="CFG Scale",
                                info="Classifier-Free Guidance. 5.0 is standard; 1.0 is ~2x faster (skips negative pass)."
                            )
                            icd_tiled = gr.Checkbox(
                                label="Tiled VAE Decoding",
                                value=False,
                                info="Uncheck (default) for seamless untiled full-frame decoding (optimized with CPU streaming). Check only if RAM is constrained.",
                            )
                            icd_void_recovery = gr.Checkbox(
                                label="Black-Void & Title-Card 3D Recovery",
                                value=True,
                                info="Automatically detects pure-black void / 3D title shots (e.g. trailer logos), removes hallucinated room/horizon gradients on the black void, and preserves 3D block/letter relief.",
                            )
                            icd_credit_pop = gr.Checkbox(
                                label="On-Screen Credit / Subtitle Foreground Pop",
                                value=True,
                                info="Automatically detects static 2D credits/subtitles over moving scenes and places them at disparity=1.00 as the closest foreground layer.",
                            )
                            icd_splash_recovery = gr.Checkbox(
                                label="🌊 Volumetric Splash & Water Aeration Recovery",
                                value=False,
                                info="Detects underwater aerated bubble plumes and plunge cavitation, elevating the buoyant volume to prevent sunken dark voids and floating limbs.",
                            )
                        icd_credit_hint = gr.Textbox(
                            label="On-Screen Credit Text Hint (Optional — Improves Precision for Specific Credits)",
                            placeholder="e.g. Directed by Vania Heymann | Gal Muggia (leave blank for 100% automatic detection)",
                            value="",
                            info="Optional: Enter the on-screen credit text (use '|' or newline for multiple lines) to lock onto those exact lines and boost fine-serif stroke extraction.",
                        )

                        icd_batch_depth_btn = gr.Button("📦 Run Batch ICDepth on All Source Videos", variant="secondary")

                    with gr.Column(scale=1):
                        icd_step1_dropdown = gr.Dropdown(
                            choices=[""] + stems,
                            value=default_stem,
                            label="Select Source Video",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        icd_step1_btn = gr.Button("⚡ Estimate Depth with ICDepth", variant="primary", size="lg")
                        icd_scene_repair_strength = gr.Slider(
                            0.0, 1.0,
                            value=1.0,
                            step=0.05,
                            label="🎚️ Scene & Void 3D Repair Strength",
                            info="1.00 = Full repair. Controls black-void title cards, light-bleed suppression, and water splashes. Pull back toward 0.00 to revert to raw depth.",
                        )
                        icd_credit_repair_strength = gr.Slider(
                            0.0, 1.0,
                            value=1.0,
                            step=0.05,
                            label="🔤 Typography & Credit Pop Strength",
                            info="1.00 = Full stereoscopic foreground pop (cinema plane). Pull back toward 0.00 for subtle relief or untouched raw depth.",
                        )
                        with gr.Row():
                            icd_void_repair_btn = gr.Button("✨ Apply / Update ICDepth Repair", variant="secondary", size="sm")
                            icd_revert_raw_btn = gr.Button("⏪ Revert to Raw ICDepth", variant="secondary", size="sm")
                        icd_step1_stop_btn = gr.Button("🛑 STOP", variant="stop", size="sm")
                        icd_step1_status = gr.Textbox(label="Estimation Progress", interactive=False, lines=6)
                        icd_depth_file = gr.File(label="Download Depth .npz", type="filepath")

            # ===================== TAB 3: STEP 1 - DEPTHCRAFTER =====================
            with gr.Tab("🎬 Step 1 - Depthcrafter"):
                gr.Markdown("### 🎬 Step 1 — Depthcrafter\n#### Diffusion Video Depth Estimator: Robust temporal consistency and anatomical completeness across fast motion and particles.")
                with gr.Row():
                    with gr.Column(scale=2):
                        dc_guidance_scale = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Guidance Scale")
                        dc_inference_steps = gr.Slider(1, 50, value=5, step=1, label="Inference Steps (Default 5 for speed)")
                        dc_window_size = gr.Slider(10, 200, value=30, step=1, label="Window Size", info="GUIDELINE: 25-30 for 12GB VRAM GPUs (RTX 3060) to stay strictly in fast VRAM (~8-9GB) and prevent Windows PCIe memory paging. 80-110 for 24GB+ GPUs.")
                        dc_overlap = gr.Slider(0, 100, value=10, step=1, label="Overlap", info="Recommended: 10 for Window Size 30 (stride = 20 frames)")
                        dc_max_res = gr.Slider(256, 3840, value=1024, step=64, label="Max Resolution (Longest Edge)", info="GUIDELINE: 1024 = 12GB VRAM (RTX 3060/4070). 1536 = 24GB VRAM (RTX 4090). Setting this to 4K (3840) will crash almost any GPU due to the massive VRAM required for diffusion models.")
                        dc_attn_slicing = gr.Dropdown(
                            choices=["Auto (Adapts to GPU VRAM)", "Disabled (Fastest / 12GB+ GPUs)", "Enabled (Low VRAM / <=8GB)"],
                            value="Auto (Adapts to GPU VRAM)",
                            label="Attention Slicing",
                            info="Auto: Disables slicing for maximum speed on 12GB+ GPUs, automatically slices on <=8GB cards or huge workloads."
                        )
                        dc_cpu_offload = gr.Dropdown(
                            choices=[
                                "Auto (Adapts to GPU VRAM)",
                                "none (Fastest / Full VRAM)",
                                "model (Balanced / Saves ~4GB VRAM)",
                                "sequential (Low VRAM / <=8GB GPUs)",
                            ],
                            value="Auto (Adapts to GPU VRAM)",
                            label="CPU Offload",
                            info="Auto: Full VRAM on >=45GB GPUs (RTX 6000 Ada/96GB), 'model' on <45GB (RTX 3090/4090/5090) to save ~5GB VRAM. 'sequential' enables running on <=8GB GPUs."
                        )
                        
                        dc_batch_depth_btn = gr.Button("📦 Run Batch Depth Processing on All Source Videos", variant="secondary")
                    with gr.Column(scale=1):
                        dc_step1_dropdown = gr.Dropdown(
                            choices=[""] + stems,
                            value=default_stem,
                            label="Select Source Video",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        dc_step1_btn = gr.Button("⚡ Estimate Depth for Selected Video", variant="primary", size="lg")
                        dc_step1_stop_btn = gr.Button("🛑 STOP", variant="stop", size="sm")
                        dc_step1_status = gr.Textbox(label="Estimation Progress", interactive=False, lines=6)
                        dc_depth_file = gr.File(label="Download Depth .npz", type="filepath")

                gr.Markdown("---")
                with gr.Accordion("✨ Guided 1080p Hybrid Depth Fusion (480p/720p DC ➔ 1080p Super-Resolution)", open=False):
                    gr.Markdown(
                        "### ✨ Fast 1080p Hybrid Workflow: Best of Both Worlds\n"
                        "Running DepthCrafter at native 1080p is slow (~0.1–0.2 FPS) and VRAM heavy. Instead:\n"
                        "1. **Run DepthCrafter at 480p / 720p** above (takes ~1–2 min, captures flying micro-particles, erratic motion, and confetti).\n"
                        "2. **Run Hybrid Fusion** below: Uses 1080p RGB edge-guided super-resolution (or optional FlashDepth harmonization), fills particle centers 100% solid (no hollow rings), filters out facial contour artifacts, and scales depth by particle diameter ($d=\\sqrt{4A/\\pi}$).\n"
                        "**Result**: A pristine native 1080p depth map ready for M2SVid 3D stereo generation!"
                    )
                    with gr.Row():
                        with gr.Column(scale=2):
                            hybrid_mode = gr.Dropdown(
                                choices=[
                                    "RGB-Guided Super-Resolution (DC -> 1080p) [Recommended]",
                                    "DC-Anchored FlashDepth Harmonization (1080p)",
                                    "Floating Particle Enhancement (Confetti / Snow / Sparks)",
                                ],
                                value="RGB-Guided Super-Resolution (DC -> 1080p) [Recommended]",
                                label="⚙️ Hybrid Engine Mode",
                                info="Recommended: Multi-scale 1080p RGB edge guidance on DepthCrafter (zero flashing, zero periphery nodules)."
                            )
                            with gr.Row():
                                hybrid_dc_select = gr.Dropdown(
                                    choices=["[Auto-Detect from Video]"] + get_depth_npz_list(),
                                    value="[Auto-Detect from Video]",
                                    label="Select 480p/720p DepthCrafter NPZ",
                                    info="Leave on Auto-Detect to automatically use {stem}_DC_depth.npz"
                                )
                                hybrid_fd_select = gr.Dropdown(
                                    choices=["[Auto-Detect from Video]"] + get_depth_npz_list(),
                                    value="[Auto-Detect from Video]",
                                    label="Select 1080p FlashDepth NPZ",
                                    info="Leave on Auto-Detect to automatically use {stem}_FDL_depth.npz"
                                )
                            with gr.Accordion("🔧 Particle & Artifact Tuning Parameters", open=False):
                                with gr.Row():
                                    hybrid_boost = gr.Slider(0.1, 1.0, value=0.45, step=0.05, label="Max Foreground Particle Pop", info="Maximum 3D depth pop for large foreground confetti discs")
                                    hybrid_circ = gr.Slider(0.1, 0.8, value=0.35, step=0.05, label="Min Circularity (Artifact Filter)", info="Rejects elongated facial ridges & chin contours (default 0.35)")
                                    hybrid_solidity = gr.Slider(0.1, 0.9, value=0.60, step=0.05, label="Min Solidity", info="Rejects concave/hollow boundary slivers (default 0.60)")
                        with gr.Column(scale=1):
                            hybrid_run_btn = gr.Button("✨ Generate 1080p Hybrid Depth Map", variant="primary", size="lg")
                            hybrid_status = gr.Textbox(label="Hybrid Fusion Progress", interactive=False, lines=4)
                            hybrid_file = gr.File(label="Download Hybrid Depth .npz", type="filepath")

            # ===================== OPTIONAL ALT TAB: FLASHDEPTH (HIDDEN BY DEFAULT) =====================
            with gr.Tab("⚡ Step 1 (Alt) - FlashDepth", visible=False) as fd_tab:
                gr.Markdown("### ⚡ Step 1 (Alt) — FlashDepth (Real-Time 2K Streaming)\n#### Alternative Real-Time Streaming Depth Estimator: 2K-capable streaming depth engine from Netflix Eyeline Studios.")
                with gr.Row():
                    with gr.Column(scale=2):
                        fd_model = gr.Dropdown(
                            choices=[
                                "FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
                                "FlashDepth-L - ViT-Large Standalone (Best for 480p)",
                                "FlashDepth-S - ViT-Small (Fastest / Low VRAM)",
                            ],
                            value="FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
                            label="FlashDepth Model Variant",
                            info="Full: Dual-stream 2K hybrid for sharpest silhouette edges at 24+ FPS. Large: Highest depth accuracy, best at 480p/576p."
                        )
                        with gr.Row():
                            fd_max_res = gr.Slider(
                                256, 2048,
                                value=1280,
                                step=14,
                                label="Max Resolution (Longest Edge, Multiple of 14)",
                                info="Resolution to stream through FlashDepth. 1280 is optimal for 12GB GPUs. 2044 for 2K."
                            )
                            fd_target_fps = gr.Dropdown(
                                choices=["-1 (Native Source FPS)", "30 fps", "24 fps", "15 fps"],
                                value="-1 (Native Source FPS)",
                                label="Target FPS",
                                info="Native keeps original source timing. Setting to 24/30 FPS caps high-framerate clips."
                            )
                        with gr.Row():
                            fd_global_norm = gr.Checkbox(label="Global Normalization", value=True, info="Stabilizes near/far disparity bounds across entire video.")
                            fd_invert_depth = gr.Checkbox(label="Invert Depth Map", value=False, info="Invert black and white depth polarity if needed.")
                            fd_max_len = gr.Number(value=-1, label="Max Frames Limit (-1 for full video)", precision=0)

                        fd_batch_depth_btn = gr.Button("📦 Run Batch FlashDepth on All Source Videos", variant="secondary")

                    with gr.Column(scale=1):
                        fd_step1_dropdown = gr.Dropdown(
                            choices=[""] + stems,
                            value=default_stem,
                            label="Select Source Video",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        fd_step1_btn = gr.Button("⚡ Estimate Depth with FlashDepth", variant="primary", size="lg")
                        fd_step1_stop_btn = gr.Button("🛑 STOP", variant="stop", size="sm")
                        fd_step1_status = gr.Textbox(label="Estimation Progress", interactive=False, lines=6)
                        fd_depth_file = gr.File(label="Download Depth .npz", type="filepath")

            # ===================== OPTIONAL ALT TAB: VIDEO DEPTH ANYTHING (HIDDEN BY DEFAULT) =====================
            with gr.Tab("⚡ Step 1 (Alt) - VDA", visible=False) as vda_tab:
                gr.Markdown("### ⚡ Step 1 (Alt) — Video Depth Anything (VDA)\n#### Alternative Fast Depth Estimator: Ultra-fast ViT-based video depth (Base & Large).")
                with gr.Row():
                    with gr.Column(scale=2):
                        vda_model = gr.Dropdown(
                            choices=[
                                "Video-Depth-Anything-Large (vitl - 381.8M, Highest Quality)",
                                "Video-Depth-Anything-Base (vitb - 113.1M, 2x Faster)",
                            ],
                            value="Video-Depth-Anything-Large (vitl - 381.8M, Highest Quality)",
                            label="VDA Model / Encoder",
                            info="Large (vitl) provides highest precision details. Base (vitb) runs ~2x faster with minimal quality difference."
                        )
                        with gr.Row():
                            vda_mode = gr.Radio(
                                choices=["Sliding Window (Optimal Quality & Alignment)", "Streaming (Recurrent KV Cache)"],
                                value="Sliding Window (Optimal Quality & Alignment)",
                                label="Inference Processing Mode",
                                info="Sliding Window uses 32-frame overlapping chunks with keyframe alignment. Streaming uses recurrent temporal cache."
                            )
                        with gr.Row():
                            vda_input_size = gr.Slider(
                                256, 1024,
                                value=518,
                                step=14,
                                label="Model Input Size (Patch Multiple: 14)",
                                info="Resolution fed to ViT backbone. 518 is default/recommended. Lower (e.g. 392) saves VRAM; higher increases sharpness."
                            )
                            vda_max_res = gr.Slider(
                                384, 1920,
                                value=1280,
                                step=64,
                                label="Max Video Resolution (Longest Edge)",
                                info="Downscales large 4K/1080p video before processing if needed, then depth is matched to video. 1280 is optimal."
                            )
                        with gr.Row():
                            vda_target_fps = gr.Slider(
                                -1.0, 60.0,
                                value=-1.0,
                                step=1.0,
                                label="Target FPS (-1 = Native Source FPS)",
                                info="Subsamples video FPS to accelerate processing if desired. Set to -1 to process every frame at native FPS."
                            )
                            vda_max_len = gr.Slider(
                                -1, 3000,
                                value=-1,
                                step=10,
                                label="Max Frames Limit (-1 = Full Video)",
                                info="Limit max processed frames for quick testing. Set to -1 to process entire video."
                            )
                        with gr.Row():
                            vda_global_norm = gr.Checkbox(
                                label="Global Video Disparity Normalization",
                                value=True,
                                info="Normalizes disparity [0, 1] globally across the entire video. Prevents inter-frame breathing or depth pulsing."
                            )
                            vda_fp32 = gr.Checkbox(
                                label="FP32 Precision (Uncheck for FP16)",
                                value=False,
                                info="Leave unchecked for FP16 (faster, less VRAM). Enable if seeing gradient artifacts."
                            )
                            vda_invert_depth = gr.Checkbox(
                                label="Invert Depth Map",
                                value=False,
                                info="Default: Off (White = closest to camera, Black = furthest away)."
                            )

                        vda_batch_depth_btn = gr.Button("📦 Run Batch VDA Depth Processing on All Source Videos", variant="secondary")

                    with gr.Column(scale=1):
                        vda_step1_dropdown = gr.Dropdown(
                            choices=[""] + stems,
                            value=default_stem,
                            label="Select Source Video",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        vda_step1_btn = gr.Button("⚡ Estimate Depth with VDA", variant="primary", size="lg")
                        vda_step1_stop_btn = gr.Button("🛑 STOP", variant="stop", size="sm")
                        vda_step1_status = gr.Textbox(label="VDA Estimation Progress", interactive=False, lines=6)
                        vda_depth_file = gr.File(label="Download Depth .npz", type="filepath")

            # ===================== TAB: STEP 1.5 - DEPTH EDITOR =====================
            with gr.Tab("🎨 Step 1.5 - Depth Editor"):
                gr.Markdown(
                    r"""
                    ### 🎨 Step 1.5 — Depth Editor & Multi-Plane Semantic Layering
                    **Interactive 3D Depth Slicing & Holdout Matte Protection**:
                    * **🔒 Immutable Holdout Mattes:** Foremost credits/text and structural geometry (elevated road, trusses, passing cars) are automatically locked and 100% mathematically preserved.
                    * **✏️ Background Aperture Slicing:** The background cavity is partitioned into 3 distinct planes: Floating Creatures (Fish & Dugong), Cityscape Buildings, and Sky Floor.
                    * **🎯 Automatic Holdout Subtraction:** Painted strokes that cross trusses automatically subtract the trusses ($\text{Creatures} = \text{Brushed} \setminus \text{Trusses}$), tucking floating objects naturally behind the bridge beams.
                    * **📌 Multi-Keyframe Tracking:** Scrub to keyframes, brush objects, record keyframes, and track automatically across the shot with piecewise optical flow.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        obj_step1_dropdown = gr.Dropdown(
                            choices=[""] + stems,
                            value=default_stem,
                            label="Select Source Video (Must have an ICDepth map)",
                            interactive=True,
                            allow_custom_value=True,
                        )
                    with gr.Column(scale=2):
                        default_has_icd = bool(
                            default_stem and (
                                (DEPTH_DIR / f"{default_stem}_ICD_depth.npz").exists() or
                                (DEPTH_DIR / f"{default_stem}_ICD_depth.raw.npz").exists()
                            )
                        )
                        icd_obj_toggle = gr.Checkbox(
                            label="🎨 Enable Depth Editor",
                            value=False,
                            interactive=default_has_icd,
                            info="✅ Video processed by ICDepth. Toggle ON to activate Depth Editor." if default_has_icd else "Select a video processed by ICDepth to enable."
                        )

                with gr.Column(visible=False, variant="panel") as icd_obj_main_section:
                    gr.Markdown("### 📋 Scene Depth Hierarchy Table (Editable)")
                    gr.Markdown("💡 *Click directly into any Description cell in the table below to edit or customize names. By default, all layers are locked.*")

                    # Editable DataFrame table
                    icd_obj_table = gr.Dataframe(
                        headers=["Layer ID", "Description (Click to Edit)", "Depth Range", "% Pixels", "Status"],
                        datatype=["str", "str", "str", "str", "str"],
                        value=[
                            ["Layer 1", "Foreground (Characters / Foremost Objects)", "0.54 – 1.00", "0.0%", "🔒 Locked"],
                            ["Layer 2", "Midground (Structures / Environment)", "0.21 – 0.54", "0.0%", "🔒 Locked"],
                            ["Layer 3", "Background (Sky / Distant Scenery)", "0.00 – 0.21", "0.0%", "🔒 Locked"],
                        ],
                        interactive=True,
                        wrap=True,
                    )

                    with gr.Row():
                        with gr.Column(scale=3):
                            icd_obj_unlock_select = gr.Radio(
                                choices=[
                                    "🔒 All Layers Locked",
                                    "Layer 1: Foreground (Characters / Foremost Objects)",
                                    "Layer 2: Midground (Structures / Environment)",
                                    "Layer 3: Background (Sky / Distant Scenery)",
                                ],
                                value="🔒 All Layers Locked",
                                label="🔓 Unlock Layer for Editing",
                                info="By default all layers are locked. Select one layer to unlock and activate its editing controls.",
                                interactive=True,
                            )
                        with gr.Column(scale=1):
                            icd_obj_add_layer_btn = gr.Button("➕ Add Layer", variant="secondary", size="sm")
                            icd_obj_revert_raw_btn = gr.Button("🔄 Revert to Raw Baseline", variant="secondary", size="sm")

                    # Lock banner showing current state
                    icd_obj_active_banner = gr.Markdown("🔒 **All layers are currently locked.** In the table above, all layers are protected. Select a layer from **'Unlock Layer for Editing'** to enable editing controls.")

                    # Editing controls: hidden by default until a layer is unlocked!
                    with gr.Column(visible=False) as icd_obj_editing_controls:
                        # ==================== METHOD 1: PAINT & TRACK ====================
                        with gr.Group():
                            gr.Markdown(
                                "### 🎨 Method 1: Paint & Track (Interactive Object Sculpting)\n"
                                "*Brush over target objects (e.g. fish, dugong) on one or more reference keyframes. "
                                "StereoFaster snaps rough strokes to exact contours with Top-Hat filtering, calculates "
                                "organic 3D volumetric curved relief, separates the horizon/skyline, and tracks objects "
                                "bidirectionally across the clip using optical flow.*"
                            )
                            with gr.Row():
                                with gr.Column(scale=5):
                                    icd_obj_frame_slider = gr.Slider(
                                        0, 100, value=36, step=1,
                                        label="Reference Keyframe Index (Scrub to view / paint across keyframes)"
                                    )
                                with gr.Column(scale=3):
                                    icd_obj_target_depth = gr.Slider(
                                        0.00, 1.00, value=0.18, step=0.01,
                                        label="🎯 Assigned Depth Value (Disparity)",
                                        info="Depth of painted object (0.0 = deep background, 1.0 = closest foreground)."
                                    )
                                with gr.Column(scale=3):
                                    with gr.Row():
                                        icd_obj_record_kf_btn = gr.Button("📌 Record Keyframe", variant="primary", size="sm")
                                        icd_obj_track_preview_btn = gr.Button("🎯 Track & Paint All Frames on Canvas", variant="secondary", size="sm")
                                        icd_obj_clear_kf_btn = gr.Button("🗑️ Clear Keyframes", variant="secondary", size="sm")
                                    with gr.Row():
                                        icd_obj_show_markers = gr.Checkbox(label="Show Depth Anchor Markers", value=False)
                                        icd_obj_reload_btn = gr.Button("🔄 Reload Frame", variant="secondary", size="sm")

                            icd_obj_kf_badge = gr.Markdown("📌 **Keyframes recorded:** None *(Brush on canvas, then click 'Record Keyframe')*")

                            icd_obj_brush_bar = gr.HTML(
                                """
                                <div style="display: flex; flex-direction: column; gap: 8px; margin: 6px 0 10px 0; padding: 10px 14px; background: rgba(30, 41, 59, 0.75); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 8px;">
                                    <!-- Tool Switcher & Size Row -->
                                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                                        <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                                            <!-- Tool Switcher Buttons -->
                                            <div style="display: inline-flex; border-radius: 6px; background: #0f172a; padding: 2px; border: 1px solid #475569; margin-right: 6px;">
                                                <button type="button" id="sf-tool-brush-btn" onclick="window.sfSetTool && window.sfSetTool('draw')" style="padding: 3px 9px; border-radius: 4px; border: none; background: #0284c7; color: #ffffff; font-weight: bold; cursor: pointer; font-size: 0.85em;">🖌️ Brush</button>
                                                <button type="button" id="sf-tool-eraser-btn" onclick="window.sfSetTool && window.sfSetTool('erase')" style="padding: 3px 9px; border-radius: 4px; border: none; background: transparent; color: #94a3b8; cursor: pointer; font-size: 0.85em;">🧹 Eraser</button>
                                            </div>
                                            <span id="sf-mode-label" style="font-weight: 600; color: #38bdf8; font-size: 0.9em; min-width: 95px;">🖌️ Brush Size:</span>
                                            <button type="button" class="sf-brush-btn" data-size="1" onclick="window.sfSetSize && window.sfSetSize(1)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #f1f5f9; cursor: pointer; font-size: 0.85em;">1px</button>
                                            <button type="button" class="sf-brush-btn" data-size="2" onclick="window.sfSetSize && window.sfSetSize(2)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #f1f5f9; cursor: pointer; font-size: 0.85em;">2px</button>
                                            <button type="button" class="sf-brush-btn" data-size="4" onclick="window.sfSetSize && window.sfSetSize(4)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #38bdf8; background: #0284c7; color: #ffffff; cursor: pointer; font-size: 0.85em; font-weight: bold;">4px</button>
                                            <button type="button" class="sf-brush-btn" data-size="6" onclick="window.sfSetSize && window.sfSetSize(6)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #f1f5f9; cursor: pointer; font-size: 0.85em;">6px</button>
                                            <button type="button" class="sf-brush-btn" data-size="10" onclick="window.sfSetSize && window.sfSetSize(10)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #f1f5f9; cursor: pointer; font-size: 0.85em;">10px</button>
                                            <button type="button" class="sf-brush-btn" data-size="16" onclick="window.sfSetSize && window.sfSetSize(16)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #f1f5f9; cursor: pointer; font-size: 0.85em;">16px</button>
                                            <button type="button" class="sf-brush-btn" data-size="25" onclick="window.sfSetSize && window.sfSetSize(25)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #f1f5f9; cursor: pointer; font-size: 0.85em;">25px</button>
                                        </div>
                                        <div style="display: flex; align-items: center; gap: 8px;">
                                            <button type="button" onclick="window.sfAdjustSize && window.sfAdjustSize(-1)" style="padding: 3px 9px; border-radius: 4px; border: 1px solid #475569; background: #1e293b; color: #f1f5f9; cursor: pointer; font-size: 0.9em; font-weight: bold;" title="Decrease size by 1px (or press [)">- 1px</button>
                                            <span id="sf-brush-val" style="font-weight: 700; color: #38bdf8; min-width: 48px; text-align: center; font-size: 0.95em;">4 px</span>
                                            <button type="button" onclick="window.sfAdjustSize && window.sfAdjustSize(1)" style="padding: 3px 9px; border-radius: 4px; border: 1px solid #475569; background: #1e293b; color: #f1f5f9; cursor: pointer; font-size: 0.9em; font-weight: bold;" title="Increase size by 1px (or press ])">+ 1px</button>
                                            <span style="color: #94a3b8; font-size: 0.8em; margin-left: 6px;">(Hotkeys: <b>[</b> and <b>]</b>)</span>
                                        </div>
                                    </div>
                                    <!-- Brush Opacity (Translucency) Row -->
                                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 8px;">
                                        <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
                                            <span style="font-weight: 600; color: #a78bfa; font-size: 0.9em; min-width: 90px;">👁️ Brush Opacity:</span>
                                            <button type="button" class="sf-opacity-btn" data-pct="20" onclick="window.sfSetBrushOpacity && window.sfSetBrushOpacity(0.20)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #f1f5f9; cursor: pointer; font-size: 0.85em;">20%</button>
                                            <button type="button" class="sf-opacity-btn" data-pct="35" onclick="window.sfSetBrushOpacity && window.sfSetBrushOpacity(0.35)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #f1f5f9; cursor: pointer; font-size: 0.85em;">35%</button>
                                            <button type="button" class="sf-opacity-btn" data-pct="50" onclick="window.sfSetBrushOpacity && window.sfSetBrushOpacity(0.50)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #a78bfa; background: #7c3aed; color: #ffffff; cursor: pointer; font-size: 0.85em; font-weight: bold;">50%</button>
                                            <button type="button" class="sf-opacity-btn" data-pct="70" onclick="window.sfSetBrushOpacity && window.sfSetBrushOpacity(0.70)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #f1f5f9; cursor: pointer; font-size: 0.85em;">70%</button>
                                            <button type="button" class="sf-opacity-btn" data-pct="100" onclick="window.sfSetBrushOpacity && window.sfSetBrushOpacity(1.00)" style="padding: 3px 8px; border-radius: 4px; border: 1px solid #475569; background: #0f172a; color: #f1f5f9; cursor: pointer; font-size: 0.85em;">100%</button>
                                        </div>
                                        <div style="display: flex; align-items: center; gap: 8px;">
                                            <button type="button" onclick="window.sfAdjustBrushOpacity && window.sfAdjustBrushOpacity(-0.10)" style="padding: 3px 9px; border-radius: 4px; border: 1px solid #475569; background: #1e293b; color: #f1f5f9; cursor: pointer; font-size: 0.9em; font-weight: bold;" title="Decrease opacity by 10% (or press { / Shift+[)">- 10%</button>
                                            <span id="sf-brush-opacity-val" style="font-weight: 700; color: #a78bfa; min-width: 48px; text-align: center; font-size: 0.95em;">50%</span>
                                            <button type="button" onclick="window.sfAdjustBrushOpacity && window.sfAdjustBrushOpacity(0.10)" style="padding: 3px 9px; border-radius: 4px; border: 1px solid #475569; background: #1e293b; color: #f1f5f9; cursor: pointer; font-size: 0.9em; font-weight: bold;" title="Increase opacity by 10% (or press } / Shift+])">+ 10%</button>
                                            <span style="color: #94a3b8; font-size: 0.8em; margin-left: 6px;">(Hotkeys: <b>{</b> and <b>}</b> or <b>Shift+[</b> / <b>]</b>)</span>
                                        </div>
                                    </div>
                                </div>
                                """
                            )

                            icd_obj_editor = gr.ImageEditor(
                                label="🖌️ Depth Paint Canvas — Brush over target objects for the unlocked layer",
                                type="numpy",
                                image_mode="RGBA",
                                transforms=(),
                                sources=(),
                                height=540,
                                brush=gr.Brush(
                                    default_size=4,
                                    default_color=("rgb(255, 34, 68)", 0.50),
                                    color_mode="defaults"
                                ),
                                eraser=gr.Eraser(default_size=4),
                                elem_id="depth_paint_canvas",
                            )

                            with gr.Row():
                                icd_obj_city_depth = gr.Slider(
                                    0.00, 0.20, value=0.06, step=0.01,
                                    label="🏙️ Mid/Distant Silhouette Depth",
                                    info="Depth for intermediate aperture structures."
                                )
                                icd_obj_sky_depth = gr.Slider(
                                    0.00, 0.10, value=0.00, step=0.01,
                                    label="🌌 Deep Floor Depth",
                                    info="Depth of deepest floor (0.00 = infinite distance)."
                                )

                            with gr.Row():
                                icd_obj_tophat_snap = gr.Radio(
                                    choices=[
                                        "🖌️ Direct Brush Fidelity (100% User Silhouette & Organic 3D Volume)",
                                        "🎩 Top-Hat Edge Snapping (Auto-Snap Strokes to Source Image Contours)",
                                    ],
                                    value="🖌️ Direct Brush Fidelity (100% User Silhouette & Organic 3D Volume)",
                                    label="🎯 Contour Snapping Method",
                                    info="Direct Fidelity preserves your exact manual brushwork; Top-Hat automatically snaps strokes outward to high-contrast source edges (minus locked layers)."
                                )

                            with gr.Row():
                                icd_obj_enable_tracking = gr.Checkbox(
                                    label="🌊 Enable Optical Flow Tracking",
                                    value=True,
                                    info="When enabled, propagates painted strokes across unpainted frames using optical flow. When disabled, applies depth strictly to your explicitly painted frames."
                                )
                                icd_obj_skyline_split = gr.Checkbox(
                                    label="🌄 Separate Sky and Skyline in Aperture",
                                    value=False,
                                    info="Delineates horizon silhouette separating sky from backdrop (Coldplay Up&Up shot)."
                                )
                                icd_obj_save_sidecar = gr.Checkbox(
                                    label="Save to {stem}.layers.json Sidecar",
                                    value=True
                                )

                            icd_obj_apply_btn = gr.Button("⚡ Apply", variant="primary", size="lg")

                        # ==================== METHOD 2: OPTION A ====================
                        with gr.Accordion("✨ Method 2: Option A — In-Context Keyframe Propagation (ICDepth DiT)", open=False):
                            gr.Markdown(
                                "*Uses the keyframe(s) you painted above in **Paint & Track** as an in-context visual conditioning prompt, "
                                "running Wan2.1 Diffusion Transformer (DiT) inference to propagate fine structural 3D relief and cross-frame "
                                "temporal consistency across the unlocked layer.*"
                            )
                            with gr.Row():
                                icd_opt_a_steps = gr.Slider(
                                    1, 15, value=5, step=1,
                                    label="DiT Diffusion Steps",
                                    info="FlowMatch diffusion steps (default 5; 1-3 = rapid preview)."
                                )
                                icd_opt_a_cfg = gr.Slider(
                                    1.0, 7.5, value=1.0, step=0.5,
                                    label="CFG Scale",
                                    info="1.0 = Fast single-pass DiT; >1.0 = classifier-free guidance."
                                )
                                icd_opt_a_res = gr.Slider(
                                    384, 1920, value=512, step=64,
                                    label="Inference Resolution (Max Side)",
                                    info="Longest side for DiT inference (unlocked to 1080p)."
                                )

                            icd_opt_a_btn = gr.Button("✨ Run Option A: In-Context Keyframe Propagation", variant="primary", size="lg")

                        # ==================== METHOD 3: OPTION B ====================
                        with gr.Accordion("🕳️ Method 3: Option B — Occluder-Removed Cavity Pass (DepthCrafter / ICDepth)", open=False):
                            gr.Markdown(
                                "*Masks out all locked foreground occluders (road, bridge trusses, cars, and on-screen credits) "
                                "and runs generative depth estimation on the exposed background cavity with full dynamic range [0, 1]. "
                                "The newly generated depth is normalized and mapped strictly into the unlocked layer's depth bounds, "
                                "while all locked layers remain 100% bitwise preserved.*"
                            )
                            with gr.Row():
                                icd_opt_b_engine = gr.Radio(
                                    choices=["DepthCrafter (SVD U-Net)", "ICDepth (Wan2.1 DiT)"],
                                    value="DepthCrafter (SVD U-Net)",
                                    label="Cavity Depth Estimation Engine",
                                    info="Choose which generative depth model processes the occluder-removed cavity."
                                )
                                icd_opt_b_cavity_mode = gr.Radio(
                                    choices=["Inpaint Occluders (Pyramid Infill)", "Crop to Cavity Bounding Box"],
                                    value="Inpaint Occluders (Pyramid Infill)",
                                    label="Occluder Removal Mode",
                                    info="Inpaint fills locked foreground with smooth background; Crop focuses tightly on cavity bounding box."
                                )

                            # DepthCrafter controls (visible by default)
                            with gr.Column(visible=True) as icd_opt_b_dc_box:
                                with gr.Row():
                                    icd_opt_b_dc_steps = gr.Slider(1, 25, value=5, step=1, label="DepthCrafter Steps", info="SVD denoising steps (default 5)")
                                    icd_opt_b_dc_res = gr.Slider(384, 1024, value=512, step=64, label="DepthCrafter Resolution", info="Longest side for inference")
                                    icd_opt_b_dc_guidance = gr.Slider(1.0, 3.5, value=1.0, step=0.2, label="Guidance Scale", info="SVD classifier-free guidance")

                            # ICDepth controls (visible when ICDepth is selected)
                            with gr.Column(visible=False) as icd_opt_b_icd_box:
                                with gr.Row():
                                    icd_opt_b_icd_steps = gr.Slider(1, 15, value=5, step=1, label="ICDepth DiT Steps", info="FlowMatch diffusion steps (default 5)")
                                    icd_opt_b_icd_cfg = gr.Slider(1.0, 7.5, value=1.0, step=0.5, label="ICDepth CFG Scale", info="1.0 = Fast single-pass DiT")
                                    icd_opt_b_icd_res = gr.Slider(384, 1920, value=512, step=64, label="ICDepth Resolution", info="Longest side for DiT processing (unlocked to 1080p)")

                            icd_opt_b_btn = gr.Button("🕳️ Run Option B: Occluder-Removed Cavity Pass", variant="primary", size="lg")

                        with gr.Row():
                            with gr.Column(scale=1):
                                icd_obj_status = gr.Textbox(label="Depth Editor Status & Progress", interactive=False, lines=5)
                            with gr.Column(scale=1):
                                icd_obj_preview_video = gr.Video(label="Sculpted Depth Preview", interactive=False, format="mp4")
                                icd_obj_depth_file = gr.File(label="Download Sculpted Depth .npz", type="filepath")

                    # States for multi-keyframe masks and dynamic layer choices
                    keyframe_state = gr.State({})
                    layer_info_state = gr.State([])

            # ===================== TAB 5: M2SVID STEREOGRAPHY =====================
            with gr.Tab("🎬 Step 2 - M2SVid"):
                gr.Markdown("### 🎬 Step 2 — M2SVid Stereography\n#### Perform geometric warping and single-step conditioned 3D stereo video generation.")
                with gr.Row():
                    with gr.Column(scale=2):
                        disparity_perc = gr.Slider(
                            5.0, 60.0,
                            value=20.0,
                            step=1.0,
                            label="Disparity (StereoCrafter Scale)",
                            info="Parallax depth shift matching StereoCrafter scale. Default: 20.0 = 2.0% of width (38.4px on 1080p). 15-25: natural, comfortable depth; 30+: deeper 3D."
                        )
                        convergence_point = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Convergence Point (Zero Parallax)", info="Default is 0.5. Higher values = looking through window effect. Lower values = 3D pop-outs.")
                        closing_kernel = gr.Slider(3, 21, value=11, step=2, label="Mask Closing Kernel")
                        mask_antialias = gr.Checkbox(label="Mask Antialias", value=False)
                        
                        m2svid_config = gr.Textbox(value=DEFAULT_M2SVID_CONFIG, label="Config Path", visible=False)
                        m2svid_ckpt = gr.Textbox(value=DEFAULT_M2SVID_CKPT, label="Checkpoint Path", visible=False)
                        
                        warping_batch_size = gr.Slider(1, 64, value=_VRAM_DEFAULTS["warp"], step=1, label="Warping Batch Size (higher = faster parallel warping)")
                        gen_chunk_size = gr.Slider(2, 35, value=_VRAM_DEFAULTS["gen_chunk"], step=1, label="Generation Chunk Size (lower = less VRAM)")
                        gen_batch_size = gr.Slider(
                            1, 4,
                            value=_VRAM_DEFAULTS.get("gen_batch", 1),
                            step=1,
                            label="Parallel Chunk Batch Size (Non-Native Only)",
                            info="Batches multiple chunks into a single GPU forward pass to saturate 24GB/32GB GPUs (RTX 5090). Only available at non-native resolutions (1024x576, 768x432, 1280x720). Automatically clamped to 1 on Native 1080p to prevent OOM."
                        )
                        m2svid_process_res = gr.Dropdown(
                            choices=["1024x576 (Optimal 12GB)", "1280x720 (Faster)", "768x432 (Fastest)", "Native"],
                            value="1024x576 (Optimal 12GB)",
                            label="M2SVid Processing Resolution"
                        )
                        batch_m2svid_btn = gr.Button("📦 Run Batch Stereography on All Matched Pairs", variant="secondary")

                    with gr.Column(scale=1):
                        step2_btn = gr.Button("💫 Convert Selected to Stereo 3D", variant="primary", size="lg")
                        step2_stop_btn = gr.Button("🛑 STOP", variant="stop", size="sm")
                        step2_status = gr.Textbox(label="M2SVid Progress", interactive=False)
                        
                        out_right = gr.Video(label="Generated Right Eye", interactive=False, format="mp4")
                        out_sbs = gr.Video(label="Final Stereo SBS", interactive=False, format="mp4")
                        out_anaglyph = gr.Video(label="Final Anaglyph Red/Cyan", interactive=False, format="mp4")
                        out_dir_box = gr.Textbox(label="Output Directory (all files)", interactive=False)

            # ===================== TAB 2: AUTO PIPELINE AND BENCHMARKING =====================
            with gr.Tab("📊 Auto Pipeline and Benchmarking"):
                gr.Markdown(
                    """
                    ### 📊 Auto Pipeline and Benchmarking
                    **100% Explicit Configuration**: All parameters are saved to and read from `pipeline_sidecar.json`.
                    Run a 3-tier benchmark (Short, Medium, Long) to obtain accurate times, speed (FPS), and cost projections, plus an explicit recommended timer before starting your prioritized batch queue.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            sc_profile = gr.Dropdown(
                                choices=list(HARDWARE_PROFILES.keys()),
                                value=init_prof,
                                label="Hardware Profile (VRAM-Optimized)",
                                info="Auto-selects optimal window size, parallel batching, and chunk limits to prevent PCIe memory paging."
                            )
                            sc_vast_rate = gr.Number(
                                value=init_rate,
                                label="Cloud Rental Rate ($/hr)",
                                precision=2,
                                info="Used for benchmark cost extrapolation."
                            )
                            sc_depth_priority = gr.Dropdown(
                                choices=["DepthCrafter", "ICDepth", "FlashDepth", "VDA"],
                                value=init_depth_prio if init_depth_prio in ("DepthCrafter", "ICDepth", "FlashDepth", "VDA") else "DepthCrafter",
                                label="Step 2 Depth Map Priority (when multiple exist)",
                                info="Preferred depth map used by M2SVid Stereography when multiple depth maps exist for a clip."
                            )
                    with gr.Column(scale=1):
                        sc_save_btn = gr.Button("💾 Save Pipeline Sidecar", variant="primary", size="sm")
                        sc_reload_btn = gr.Button("🔄 Reload Sidecar from Disk", variant="secondary", size="sm")
                        sc_profile_defaults_btn = gr.Button("⚙️ Reset Profile Defaults", variant="secondary", size="sm")
                        sc_status_text = gr.Markdown("ℹ️ Sidecar loaded from `pipeline_sidecar.json`.")

                gr.Markdown("#### 📋 Active Pipeline Stages")
                with gr.Row():
                    with gr.Column(scale=2, variant="panel"):
                        gr.Markdown("##### 🔍 Step 1 — Depth Estimation Engines")
                        gr.Markdown("**⭐ Recommended Primary Depth Engines:**")
                        with gr.Row():
                            sc_stage_dc = gr.Checkbox(
                                label="🎬 DepthCrafter (Diffusion Video Depth)",
                                value=init_stages.get("depthcrafter", True),
                                info="Best for fast action, particles, and global anatomical structure."
                            )
                            sc_stage_icd = gr.Checkbox(
                                label="💠 ICDepth (Wan2.1 In-Context DiT)",
                                value=init_stages.get("icdepth", False),
                                info="Best for solid physical surfaces, indoor scenes, and DINOv2-guided detail."
                            )
                        gr.Markdown("**⚡ Alternative Streaming / ViT Depth Engines:**")
                        with gr.Row():
                            sc_stage_fd = gr.Checkbox(
                                label="⚡ FlashDepth (Real-Time 2K Streaming)",
                                value=init_stages.get("flashdepth", False),
                                info="20-30+ FPS Mamba recurrent streaming depth."
                            )
                            sc_stage_vda = gr.Checkbox(
                                label="⚡ Video Depth Anything (VDA)",
                                value=init_stages.get("vda", False),
                                info="Fast ViT-Large sliding-window depth."
                            )
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("##### 🎬 Step 2 — 3D Stereography Synthesis")
                        gr.Markdown("**👓 Stereo 3D View Synthesis (Requires Step 1 Depth Map):**")
                        sc_stage_m2svid = gr.Checkbox(
                            label="🎬 M2SVid Stereography (3D SBS & Anaglyph)",
                            value=init_stages.get("m2svid", True),
                            info="Warps source video using Step 1 depth map and synthesizes Right-Eye, SBS, and Anaglyph 3D."
                        )
                        gr.Markdown("**🔊 Output Multiplexing:**")
                        sc_audio_passthrough = gr.Checkbox(
                            label="🔊 Audio Passthrough (Copy Source Audio)",
                            value=init_audio_pass,
                            info="Automatically copies original audio track to generated 3D stereo videos."
                        )

                preflight_box = gr.HTML(
                    validate_preflight(
                        SOURCE_DIR,
                        DEPTH_DIR,
                        init_stages.get("depthcrafter", True),
                        init_stages.get("vda", False),
                        init_stages.get("m2svid", True),
                        do_icd=init_stages.get("icdepth", False),
                        do_fd=init_stages.get("flashdepth", False),
                    )[0]
                )

                with gr.Accordion("⚙️ Detailed Pipeline Sidecar Settings (Click to expand & customize)", open=False):
                    with gr.Tabs():
                        with gr.Tab("🎬 Step 1: DepthCrafter Settings"):
                            with gr.Row():
                                sc_dc_window = gr.Slider(10, 150, value=int(init_dc_cfg.get("window_size", 30)), step=1, label="Window Size", info="30 for 12GB GPUs, 75-110 for 24GB+ GPUs")
                                sc_dc_overlap = gr.Slider(0, 50, value=int(init_dc_cfg.get("overlap", 10)), step=1, label="Overlap", info="Frames overlapped between windows")
                                sc_dc_steps = gr.Slider(1, 25, value=int(init_dc_cfg.get("inference_steps", 5)), step=1, label="Inference Steps", info="Default 5 for fast inference")
                            with gr.Row():
                                sc_dc_max_res = gr.Slider(256, 1536, value=int(init_dc_cfg.get("max_res", 1024)), step=64, label="Max Resolution (Longest Edge)")
                                sc_dc_guidance = gr.Slider(0.1, 5.0, value=float(init_dc_cfg.get("guidance_scale", 1.0)), step=0.1, label="Guidance Scale")
                                sc_dc_offload = gr.Dropdown(choices=["none", "model", "sequential"], value=init_dc_cfg.get("cpu_offload", "model"), label="CPU Offload")
                                sc_dc_attn = gr.Dropdown(
                                    choices=["Auto (Adapts to GPU VRAM)", "Disabled (Fastest / 12GB+ GPUs)", "Enabled (Low VRAM / <=8GB)"],
                                    value=init_dc_cfg.get("attn_slicing", "Auto (Adapts to GPU VRAM)"),
                                    label="Attention Slicing"
                                )
                        with gr.Tab("💠 Step 1: ICDepth Settings"):
                            with gr.Row():
                                sc_icd_max_res = gr.Slider(384, 1920, value=int(init_icd_cfg.get("max_res", 720)), step=16, label="Max Inference Resolution (Multiple of 16)", info="720 for 12GB GPUs; 1024 for 24GB GPUs; 1920 for 32GB+ GPUs (unlocked to 1080p)")
                                sc_icd_steps = gr.Slider(2, 20, value=int(init_icd_cfg.get("inference_steps", 5)), step=1, label="Inference Steps", info="FlowMatch steps (default 5)")
                            sc_icd_res_advisory = gr.HTML(
                                value=get_icdepth_res_advisory(int(init_icd_cfg.get("max_res", 720))),
                                elem_id="sc_icd_res_advisory"
                            )
                            with gr.Row():
                                sc_icd_window = gr.Slider(41, 121, value=int(init_icd_cfg.get("window_size", 81)), step=4, label="Temporal Window Size (Frames, 4k+1)")
                                sc_icd_overlap = gr.Slider(16, 64, value=int(init_icd_cfg.get("overlap", 32)), step=4, label="Window Overlap (Frames, 4k)")
                            with gr.Row():
                                sc_icd_cfg = gr.Slider(1.0, 7.0, value=float(init_icd_cfg.get("cfg_scale", 5.0)), step=0.5, label="CFG Scale", info="5.0 standard; 1.0 is ~2x faster")
                                sc_icd_tiled = gr.Checkbox(label="Tiled VAE Decoding", value=bool(init_icd_cfg.get("tiled", False)), info="Leave unchecked (default) for seamless untiled decoding")
                        with gr.Tab("🎬 Step 2: M2SVid Stereography Settings"):
                            with gr.Row():
                                sc_m2s_chunk = gr.Slider(2, 14, value=int(init_m2s_cfg.get("gen_chunk_size", 5)), step=1, label="Generation Chunk Size", info="CRITICAL: 5 for 12GB GPUs to avoid PCIe paging; 14 for 24GB+ GPUs")
                                sc_m2s_batch = gr.Slider(1, 4, value=int(init_m2s_cfg.get("gen_batch_size", 1)), step=1, label="Parallel Chunk Batch Size", info="1 for 12GB/24GB GPUs; 2-3 for 32GB/48GB+ GPUs")
                                sc_m2s_res = gr.Dropdown(
                                    choices=["Native", "1024x576 (Optimal 12GB)", "1280x720 (Faster)", "768x432 (Fastest)"],
                                    value=init_m2s_cfg.get("processing_res", "Native"),
                                    label="Processing Resolution"
                                )
                            with gr.Row():
                                sc_m2s_disparity = gr.Slider(5.0, 60.0, value=float(init_m2s_cfg.get("disparity_perc", 20.0)), step=1.0, label="Disparity (StereoCrafter Scale)")
                                sc_m2s_convergence = gr.Slider(0.0, 1.0, value=float(init_m2s_cfg.get("convergence_point", 0.50)), step=0.05, label="Convergence Point (Zero Parallax)")
                                sc_m2s_warp_batch = gr.Slider(1, 64, value=int(init_m2s_cfg.get("warping_batch_size", 8)), step=1, label="Warping Batch Size")
                                sc_m2s_closing = gr.Slider(3, 21, value=int(init_m2s_cfg.get("closing_kernel", 11)), step=2, label="Mask Closing Kernel")
                                sc_m2s_antialias = gr.Checkbox(label="Mask Antialias", value=bool(init_m2s_cfg.get("mask_antialias", False)))
                        with gr.Tab("⚡ Step 1 (Alt): FlashDepth Settings"):
                            with gr.Row():
                                sc_fd_model = gr.Dropdown(
                                    choices=[
                                        "FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
                                        "FlashDepth-L - ViT-Large Standalone (Best for 480p)",
                                        "FlashDepth-S - ViT-Small (Fastest / Low VRAM)",
                                    ],
                                    value=init_fd_cfg.get("model", "FlashDepth (Full) - 2K VFX Hybrid (Recommended)"),
                                    label="FlashDepth Model Variant"
                                )
                                sc_fd_max_res = gr.Slider(256, 2044, value=int(init_fd_cfg.get("max_res", 1288)), step=14, label="Max Resolution (Multiple of 14)")
                            with gr.Row():
                                sc_fd_target_fps = gr.Slider(-1.0, 60.0, value=float(init_fd_cfg.get("target_fps", -1.0)), step=1.0, label="Target FPS (-1 = Native)")
                                sc_fd_max_len = gr.Slider(-1, 3000, value=int(init_fd_cfg.get("max_len", -1)), step=10, label="Max Frames (-1 = Full Video)")
                                sc_fd_global_norm = gr.Checkbox(label="Global Normalization", value=bool(init_fd_cfg.get("global_norm", True)))
                                sc_fd_invert = gr.Checkbox(label="Invert Depth", value=bool(init_fd_cfg.get("invert_depth", False)))
                        with gr.Tab("⚡ Step 1 (Alt): VDA Settings") as sc_vda_tab:
                            with gr.Row():
                                sc_vda_model = gr.Dropdown(
                                    choices=["Large (vitl - Recommended)", "Base (vitb - Fast)"],
                                    value=init_vda_cfg.get("model", "Large (vitl - Recommended)"),
                                    label="Model / Encoder"
                                )
                                sc_vda_mode = gr.Radio(choices=["sliding_window", "streaming"], value=init_vda_cfg.get("mode", "sliding_window"), label="Mode")
                                sc_vda_input_size = gr.Slider(256, 1024, value=int(init_vda_cfg.get("input_size", 644)), step=14, label="Model Input Size (Multiple of 14)")
                            with gr.Row():
                                sc_vda_max_res = gr.Slider(384, 1920, value=int(init_vda_cfg.get("max_res", 1024)), step=64, label="Max Video Resolution")
                                sc_vda_target_fps = gr.Slider(-1.0, 60.0, value=float(init_vda_cfg.get("target_fps", -1.0)), step=1.0, label="Target FPS (-1 = Native)")
                                sc_vda_max_len = gr.Slider(-1, 3000, value=int(init_vda_cfg.get("max_len", -1)), step=10, label="Max Frames (-1 = Full Video)")
                            with gr.Row():
                                sc_vda_global_norm = gr.Checkbox(label="Global Normalization", value=bool(init_vda_cfg.get("global_norm", True)))
                                sc_vda_fp32 = gr.Checkbox(label="FP32 Precision", value=bool(init_vda_cfg.get("fp32", False)))
                                sc_vda_invert = gr.Checkbox(label="Invert Depth", value=bool(init_vda_cfg.get("invert_depth", False)))
                        with gr.Tab("📄 Raw Sidecar JSON"):
                            sc_json_display = gr.Code(value=init_json_str, language="json", label="pipeline_sidecar.json Content", lines=16)

                with gr.Row():
                    with gr.Column(scale=3):
                        sc_scan_btn = gr.Button("🔍 Scan Library & Check Progress", variant="secondary")
                        sc_lib_summary = gr.Markdown(init_summary_md)
                        sc_lib_table = gr.Dataframe(
                            value=init_table_rows,
                            headers=["Filename", "Duration", "Format", "Category", "Pipeline Status"],
                            datatype=["str", "str", "str", "str", "str"],
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("#### 🚀 Pipeline Execution Controls")
                        sc_bench_btn = gr.Button("⏱️ Run Multi-Sample Benchmark & Predict Time/Cost", variant="secondary", size="lg")
                        sc_batch_btn = gr.Button("⚡ Start Prioritized Batch Queue (Short ➔ Long)", variant="primary", size="lg")
                        sc_stop_btn = gr.Button("🛑 STOP PIPELINE", variant="stop", size="sm")

                        sc_pipeline_status = gr.Textbox(label="Live Pipeline Activity", value="Idle", interactive=False, lines=2)
                        sc_progress_badge = gr.Textbox(label="Stage Progress", value="Idle", interactive=False)

                sc_bench_report = gr.Markdown("")

        # In-memory depth tracking
        depth_input = gr.File(value=default_depth_npz, visible=False)
        depth_state = gr.State(default_depth_npz)

        # Toggle visibility of optional FlashDepth & VDA tabs on user request
        def _toggle_alt_tabs(show: bool):
            vis = bool(show)
            return (
                gr.update(visible=vis),
                gr.update(visible=vis),
            )

        show_alt_tabs_chk.change(
            fn=_toggle_alt_tabs,
            inputs=[show_alt_tabs_chk],
            outputs=[fd_tab, vda_tab],
        )

        # Wire dropdown updates
        def _on_hub_select(stem):
            stems_curr = get_source_video_list()
            if not stem:
                return (
                    None, None, None, None,
                    gr.update(value="16:9 (None)"),
                    gr.update(value=""),
                    None,
                    gr.update(choices=[""] + stems_curr),
                    gr.update(choices=[""] + stems_curr),
                    gr.update(choices=[""] + stems_curr),
                    gr.update(choices=[""] + stems_curr),
                    gr.update(choices=[""] + stems_curr),
                    gr.update(choices=[""] + stems_curr),
                )

            v_p, d_npz, d_mp4_p = select_source_video(stem)
            
            SF_LOG.info(f"Loaded preview for '{stem}'")
            if d_mp4_p:
                SF_LOG.info(f"  -> Found Depth Video: {d_mp4_p}")
            else:
                SF_LOG.info(f"  -> No Depth Video found yet for '{stem}'")
                
            preset_val, res_val, preview_crop = "16:9 (None)", "", None
            if v_p:
                preset_val, res_val, preview_crop = analyze_letterbox(v_p)
                
            return (
                v_p,
                d_mp4_p,
                d_npz,
                d_npz,
                gr.update(value=preset_val),
                gr.update(value=res_val),
                preview_crop,
                gr.update(choices=[""] + stems_curr, value=stem),
                gr.update(choices=[""] + stems_curr, value=stem),
                gr.update(choices=[""] + stems_curr, value=stem),
                gr.update(choices=[""] + stems_curr, value=stem),
                gr.update(choices=[""] + stems_curr, value=stem),
                gr.update(choices=[""] + stems_curr, value=stem),
                gr.update(choices=[""] + stems_curr, value=stem),
            )
            
        hub_outputs = [
            preview_video,
            preview_depth,
            depth_input,
            depth_state,
            crop_preset,
            crop_res,
            crop_preview_state,
            desat_clip_dropdown,
            iso_clip_dropdown,
            vda_step1_dropdown,
            dc_step1_dropdown,
            fd_step1_dropdown,
            icd_step1_dropdown,
            obj_step1_dropdown,
        ]

        source_dropdown.change(
            fn=_on_hub_select,
            inputs=[source_dropdown],
            outputs=hub_outputs,
        )
        
        def _toggle_preview(current_video, crop_preview, stem):
            # If current video is the full one, show crop preview
            if current_video and crop_preview and "crop_preview" not in str(current_video):
                return crop_preview
            else:
                # If currently showing cropped, or no crop exists, switch back to full
                if stem:
                    v_p, _, _ = select_source_video(stem)
                    return str(v_p) if v_p and os.path.exists(str(v_p)) else None
            return None
            
        toggle_preview_btn.click(
            fn=_toggle_preview,
            inputs=[preview_video, crop_preview_state, source_dropdown],
            outputs=[preview_video],
        )
        
        def _apply_crop_and_refresh(stem, preset):
            if not stem:
                return (gr.update(),) * (1 + len(hub_outputs))
            v_p, _, _ = select_source_video(stem)
            if v_p:
                execute_crop(v_p, preset)
            
            src = get_source_video_list()
            new_stem = f"{stem}_cropped"
            target_stem = new_stem if new_stem in src else stem
            hub_vals = _on_hub_select(target_stem)
            return (gr.update(choices=[""] + src, value=target_stem),) + tuple(hub_vals)
            
        apply_crop_btn.click(
            fn=_apply_crop_and_refresh,
            inputs=[source_dropdown, crop_preset],
            outputs=[source_dropdown] + hub_outputs,
        )

        def _handle_source_upload(files):
            if not files:
                return (gr.update(),) * (1 + len(hub_outputs))
            if not isinstance(files, list):
                files = [files]
            last_stem = ""
            for f in files:
                src_path, orig_name = _extract_file_path_and_name(f)
                if not src_path or not os.path.exists(src_path):
                    continue
                dst_path = SOURCE_DIR / orig_name
                shutil.copy(src_path, str(dst_path))
                SF_LOG.info(f"Uploaded source video saved to {dst_path}")
                last_stem = Path(orig_name).stem

            stems_curr = get_source_video_list()
            if not last_stem:
                return (gr.update(choices=[""] + stems_curr),) + (gr.update(),) * len(hub_outputs)

            hub_vals = _on_hub_select(last_stem)
            return (gr.update(choices=[""] + stems_curr, value=last_stem),) + tuple(hub_vals)

        def _handle_depth_upload(files):
            if not files:
                return (gr.update(), gr.update()) + (gr.update(),) * len(hub_outputs)
            if not isinstance(files, list):
                files = [files]
            last_base_stem = ""
            last_depth_stem = ""
            for f in files:
                src_path, orig_name = _extract_file_path_and_name(f)
                if not src_path or not os.path.exists(src_path):
                    continue
                dst_path = DEPTH_DIR / orig_name
                shutil.copy(src_path, str(dst_path))
                SF_LOG.info(f"Uploaded depth map video saved to {dst_path}")
                stem = Path(orig_name).stem
                last_depth_stem = stem
                base_stem = stem[:-6] if stem.endswith("_depth") else stem
                for sfx in ("_DC", "_VDAL", "_VDAB", "_VDA"):
                    if base_stem.endswith(sfx):
                        base_stem = base_stem[:-len(sfx)]
                        break
                out_npz = DEPTH_DIR / f"{stem}.npz"
                try:
                    convert_depth_video_to_npz(str(dst_path), str(out_npz))
                except Exception as e:
                    SF_LOG.error(f"Failed to automatically convert depth video to npz: {e}")
                last_base_stem = base_stem

            dep_list = get_depth_video_list()
            stems_curr = get_source_video_list()
            if last_base_stem and last_base_stem in stems_curr:
                hub_vals = _on_hub_select(last_base_stem)
                return (
                    gr.update(choices=[""] + stems_curr, value=last_base_stem),
                    gr.update(choices=[""] + dep_list, value=last_depth_stem),
                ) + tuple(hub_vals)
            else:
                return (
                    gr.update(choices=[""] + stems_curr),
                    gr.update(choices=[""] + dep_list, value=last_depth_stem if last_depth_stem else None),
                ) + (gr.update(),) * len(hub_outputs)

        # Wire uploaders directly to avoid client-side race conditions
        source_uploader.upload(
            fn=_handle_source_upload,
            inputs=[source_uploader],
            outputs=[source_dropdown] + hub_outputs,
        )
        
        depth_uploader.upload(
            fn=_handle_depth_upload,
            inputs=[depth_uploader],
            outputs=[source_dropdown, depth_dropdown] + hub_outputs,
        )

        # Wire Step 1 (Video Depth Anything)
        vda_step1_btn.click(
            fn=step1_run_vda_depth,
            inputs=[
                vda_step1_dropdown,
                vda_model,
                vda_mode,
                vda_input_size,
                vda_max_res,
                vda_target_fps,
                vda_max_len,
                vda_fp32,
                vda_global_norm,
                vda_invert_depth,
            ],
            outputs=[vda_step1_status, preview_depth, vda_depth_file, depth_state],
        )
        vda_step1_stop_btn.click(fn=trigger_cancel, outputs=[vda_step1_status])

        vda_batch_depth_btn.click(
            fn=run_vda_on_source_videos,
            inputs=[
                vda_model,
                vda_mode,
                vda_input_size,
                vda_max_res,
                vda_target_fps,
                vda_max_len,
                vda_fp32,
                vda_global_norm,
                vda_invert_depth,
            ],
            outputs=[vda_batch_depth_btn, source_dropdown, depth_dropdown, vda_step1_dropdown, vda_step1_status],
        )

        # Wire Step 1 (DepthCrafter)
        dc_step1_btn.click(
            fn=step1_run_depthcrafter,
            inputs=[dc_step1_dropdown, dc_max_res, dc_guidance_scale, dc_inference_steps, dc_window_size, dc_overlap, dc_attn_slicing, dc_cpu_offload],
            outputs=[dc_step1_status, preview_depth, dc_depth_file, depth_state],
        )
        dc_step1_stop_btn.click(fn=trigger_cancel, outputs=[dc_step1_status])

        dc_batch_depth_btn.click(
            fn=run_depth_on_source_videos,
            inputs=[
                gr.Textbox(value="tencent/DepthCrafter", visible=False),
                dc_max_res,
                gr.Slider(value=1, visible=False),
                dc_guidance_scale,
                dc_inference_steps,
                dc_window_size,
                dc_overlap,
                dc_attn_slicing,
                dc_cpu_offload,
            ],
            outputs=[dc_batch_depth_btn, source_dropdown, depth_dropdown, dc_step1_dropdown, dc_step1_status],
        )

        # Wire Guided 1080p Hybrid Depth Fusion
        hybrid_run_btn.click(
            fn=step1_run_hybrid_fusion,
            inputs=[
                dc_step1_dropdown,
                hybrid_dc_select,
                hybrid_fd_select,
                hybrid_mode,
                hybrid_boost,
                hybrid_circ,
                hybrid_solidity,
            ],
            outputs=[hybrid_status, preview_depth, hybrid_file, depth_state, depth_dropdown],
        )

        def _refresh_hybrid_npz_choices(_):
            npz_list = ["[Auto-Detect from Video]"] + get_depth_npz_list()
            return gr.update(choices=npz_list), gr.update(choices=npz_list)

        dc_step1_dropdown.change(
            fn=_refresh_hybrid_npz_choices,
            inputs=[dc_step1_dropdown],
            outputs=[hybrid_dc_select, hybrid_fd_select],
        )

        # Wire Step 1 (ICDepth)
        def _load_icd_credit_sidecar_for_stem(stem_val):
            if not stem_val:
                return ""
            try:
                from m2svid.prepare_icdepth import load_clip_credit_sidecar
                return load_clip_credit_sidecar(str(SOURCE_DIR / f"{stem_val}.mp4"))
            except Exception:
                return ""

        icd_step1_dropdown.change(
            fn=_load_icd_credit_sidecar_for_stem,
            inputs=[icd_step1_dropdown],
            outputs=[icd_credit_hint],
        )

        icd_max_res.change(
            fn=get_icdepth_res_advisory,
            inputs=[icd_max_res],
            outputs=[icd_res_advisory],
        )
        icd_step1_btn.click(
            fn=step1_run_icdepth,
            inputs=[
                icd_step1_dropdown,
                icd_max_res,
                icd_window_size,
                icd_overlap,
                icd_steps,
                icd_cfg_scale,
                icd_tiled,
                icd_void_recovery,
                icd_splash_recovery,
                icd_credit_pop,
                icd_credit_hint,
            ],
            outputs=[icd_step1_status, preview_depth, icd_depth_file, depth_state],
        )
        icd_step1_stop_btn.click(fn=trigger_cancel, outputs=[icd_step1_status])
        icd_void_repair_btn.click(
            fn=repair_existing_icdepth_void_title,
            inputs=[
                icd_step1_dropdown,
                icd_void_recovery,
                icd_credit_pop,
                icd_credit_hint,
                icd_splash_recovery,
                icd_scene_repair_strength,
                icd_credit_repair_strength,
            ],
            outputs=[icd_step1_status, preview_depth, icd_depth_file, depth_state, depth_dropdown],
        )
        icd_scene_repair_strength.release(
            fn=repair_existing_icdepth_void_title,
            inputs=[
                icd_step1_dropdown,
                icd_void_recovery,
                icd_credit_pop,
                icd_credit_hint,
                icd_splash_recovery,
                icd_scene_repair_strength,
                icd_credit_repair_strength,
            ],
            outputs=[icd_step1_status, preview_depth, icd_depth_file, depth_state, depth_dropdown],
        )
        icd_credit_repair_strength.release(
            fn=repair_existing_icdepth_void_title,
            inputs=[
                icd_step1_dropdown,
                icd_void_recovery,
                icd_credit_pop,
                icd_credit_hint,
                icd_splash_recovery,
                icd_scene_repair_strength,
                icd_credit_repair_strength,
            ],
            outputs=[icd_step1_status, preview_depth, icd_depth_file, depth_state, depth_dropdown],
        )
        icd_revert_raw_btn.click(
            fn=revert_icdepth_to_raw,
            inputs=[icd_step1_dropdown],
            outputs=[
                icd_step1_status,
                preview_depth,
                icd_depth_file,
                depth_state,
                depth_dropdown,
                icd_scene_repair_strength,
                icd_credit_repair_strength,
            ],
        )

        # Depth Editor wiring
        obj_step1_dropdown.change(
            fn=check_icd_status_and_update_toggle,
            inputs=[obj_step1_dropdown],
            outputs=[icd_obj_toggle, icd_obj_frame_slider],
        )

        icd_obj_toggle.change(
            fn=on_toggle_interactive_layering,
            inputs=[
                icd_obj_toggle,
                obj_step1_dropdown,
                icd_obj_frame_slider,
                icd_obj_show_markers,
                keyframe_state,
            ],
            outputs=[
                icd_obj_main_section,
                icd_obj_editor,
                icd_obj_table,
                icd_obj_unlock_select,
                icd_obj_editing_controls,
                icd_obj_active_banner,
                icd_obj_frame_slider,
                icd_obj_target_depth,
                icd_obj_kf_badge,
                layer_info_state,
            ],
        )

        icd_obj_unlock_select.change(
            fn=on_select_unlocked_layer,
            inputs=[icd_obj_unlock_select, icd_obj_table, layer_info_state, obj_step1_dropdown],
            outputs=[
                icd_obj_table,
                icd_obj_editing_controls,
                icd_obj_active_banner,
                icd_obj_target_depth,
                layer_info_state,
            ],
        )

        icd_obj_table.change(
            fn=on_layer_table_change,
            inputs=[icd_obj_table, layer_info_state, obj_step1_dropdown],
            outputs=[layer_info_state, icd_obj_unlock_select],
        )

        icd_obj_target_depth.release(
            fn=on_target_depth_change,
            inputs=[icd_obj_target_depth, icd_obj_unlock_select, layer_info_state, obj_step1_dropdown],
            outputs=[layer_info_state],
        )

        icd_obj_frame_slider.release(
            fn=on_scrub_reference_frame,
            inputs=[obj_step1_dropdown, icd_obj_frame_slider, icd_obj_show_markers, keyframe_state],
            outputs=[icd_obj_editor, icd_obj_kf_badge],
        )

        icd_obj_reload_btn.click(
            fn=on_scrub_reference_frame,
            inputs=[obj_step1_dropdown, icd_obj_frame_slider, icd_obj_show_markers, keyframe_state],
            outputs=[icd_obj_editor, icd_obj_kf_badge],
        )

        icd_obj_show_markers.change(
            fn=on_scrub_reference_frame,
            inputs=[obj_step1_dropdown, icd_obj_frame_slider, icd_obj_show_markers, keyframe_state],
            outputs=[icd_obj_editor, icd_obj_kf_badge],
        )

        icd_obj_record_kf_btn.click(
            fn=record_keyframe_action,
            inputs=[
                obj_step1_dropdown,
                icd_obj_frame_slider,
                icd_obj_editor,
                keyframe_state,
                icd_obj_unlock_select,
                layer_info_state,
                icd_obj_target_depth,
            ],
            outputs=[keyframe_state, icd_obj_status, icd_obj_kf_badge, layer_info_state],
        )

        icd_obj_track_preview_btn.click(
            fn=track_and_paint_all_frames_action,
            inputs=[obj_step1_dropdown, icd_obj_frame_slider, icd_obj_editor, keyframe_state],
            outputs=[icd_obj_editor, keyframe_state, icd_obj_status, icd_obj_kf_badge],
        )

        icd_obj_clear_kf_btn.click(
            fn=clear_keyframe_action,
            inputs=[obj_step1_dropdown],
            outputs=[keyframe_state, icd_obj_status, icd_obj_kf_badge],
        )

        icd_obj_add_layer_btn.click(
            fn=on_add_layer_action,
            inputs=[icd_obj_table, layer_info_state, obj_step1_dropdown],
            outputs=[icd_obj_table, layer_info_state, icd_obj_unlock_select],
        )

        icd_obj_revert_raw_btn.click(
            fn=on_revert_to_raw_baseline,
            inputs=[obj_step1_dropdown, icd_obj_frame_slider],
            outputs=[icd_obj_status, icd_obj_preview_video, icd_obj_table, layer_info_state, icd_obj_unlock_select],
        )

        icd_obj_apply_btn.click(
            fn=run_depth_editor_action,
            inputs=[
                obj_step1_dropdown,
                icd_obj_frame_slider,
                icd_obj_editor,
                keyframe_state,
                icd_obj_target_depth,
                icd_obj_city_depth,
                icd_obj_sky_depth,
                icd_obj_skyline_split,
                icd_obj_tophat_snap,
                icd_obj_save_sidecar,
                icd_obj_enable_tracking,
                icd_obj_unlock_select,
                layer_info_state,
            ],
            outputs=[
                icd_obj_status,
                icd_obj_preview_video,
                icd_obj_depth_file,
                depth_state,
                depth_dropdown,
            ],
        )

        icd_opt_b_engine.change(
            fn=lambda eng: (
                gr.update(visible="depthcrafter" in str(eng).lower()),
                gr.update(visible="icdepth" in str(eng).lower())
            ),
            inputs=[icd_opt_b_engine],
            outputs=[icd_opt_b_dc_box, icd_opt_b_icd_box],
        )

        icd_opt_a_btn.click(
            fn=run_depth_editor_option_a,
            inputs=[
                obj_step1_dropdown,
                icd_obj_frame_slider,
                icd_obj_editor,
                keyframe_state,
                icd_obj_target_depth,
                icd_obj_city_depth,
                icd_obj_sky_depth,
                icd_obj_skyline_split,
                icd_obj_tophat_snap,
                icd_obj_save_sidecar,
                icd_obj_enable_tracking,
                icd_obj_unlock_select,
                layer_info_state,
                icd_opt_a_steps,
                icd_opt_a_cfg,
                icd_opt_a_res,
            ],
            outputs=[
                icd_obj_status,
                icd_obj_preview_video,
                icd_obj_depth_file,
                depth_state,
                depth_dropdown,
            ],
        )

        icd_opt_b_btn.click(
            fn=run_depth_editor_option_b,
            inputs=[
                obj_step1_dropdown,
                icd_obj_frame_slider,
                icd_obj_editor,
                keyframe_state,
                icd_obj_target_depth,
                icd_obj_city_depth,
                icd_obj_sky_depth,
                icd_obj_skyline_split,
                icd_obj_tophat_snap,
                icd_obj_save_sidecar,
                icd_obj_enable_tracking,
                icd_obj_unlock_select,
                layer_info_state,
                icd_opt_b_engine,
                icd_opt_b_cavity_mode,
                icd_opt_b_dc_steps,
                icd_opt_b_dc_res,
                icd_opt_b_dc_guidance,
                icd_opt_b_icd_steps,
                icd_opt_b_icd_cfg,
                icd_opt_b_icd_res,
            ],
            outputs=[
                icd_obj_status,
                icd_obj_preview_video,
                icd_obj_depth_file,
                depth_state,
                depth_dropdown,
            ],
        )

        icd_batch_depth_btn.click(
            fn=run_icdepth_on_source_videos,
            inputs=[
                icd_max_res,
                icd_window_size,
                icd_overlap,
                icd_steps,
                icd_cfg_scale,
                icd_tiled,
                icd_void_recovery,
                icd_splash_recovery,
                icd_credit_pop,
                icd_credit_hint,
            ],
            outputs=[icd_batch_depth_btn, source_dropdown, depth_dropdown, icd_step1_dropdown, icd_step1_status],
        )

        # Wire Step 1 (FlashDepth)
        fd_step1_btn.click(
            fn=step1_run_flashdepth,
            inputs=[
                fd_step1_dropdown,
                fd_model,
                fd_max_res,
                fd_target_fps,
                fd_max_len,
                fd_global_norm,
                fd_invert_depth,
            ],
            outputs=[fd_step1_status, preview_depth, fd_depth_file, depth_state],
        )
        fd_step1_stop_btn.click(fn=trigger_cancel, outputs=[fd_step1_status])

        fd_batch_depth_btn.click(
            fn=run_flashdepth_on_source_videos,
            inputs=[
                fd_model,
                fd_max_res,
                fd_target_fps,
                fd_max_len,
                fd_global_norm,
                fd_invert_depth,
            ],
            outputs=[fd_batch_depth_btn, source_dropdown, depth_dropdown, fd_step1_dropdown, fd_step1_status],
        )

        # Sync state
        depth_state.change(
            fn=lambda x: x,
            inputs=[depth_state],
            outputs=[depth_input],
        )

        batch_crop_btn.click(
            fn=batch_auto_crop_all,
            inputs=[force_crop_preset, crop_preset],
            outputs=[batch_crop_btn, source_dropdown, vda_step1_dropdown, dc_step1_dropdown, fd_step1_dropdown, icd_step1_dropdown],
        )

        # Wire step 2
        step2_btn.click(
            fn=step2_run_m2svid,
            inputs=[
                depth_input,
                disparity_perc,
                convergence_point,
                closing_kernel,
                mask_antialias,
                m2svid_config,
                m2svid_ckpt,
                preview_video,
                warping_batch_size,
                gen_chunk_size,
                m2svid_process_res,
                gen_batch_size,
            ],
            outputs=[step2_status, out_right, out_sbs, out_anaglyph, out_dir_box],
        )
        step2_stop_btn.click(fn=trigger_cancel, outputs=[step2_status])

        # Batch M2SVid stereography
        batch_m2svid_btn.click(
            fn=run_m2svid_on_pairs,
            inputs=[m2svid_config, m2svid_ckpt, disparity_perc, convergence_point, closing_kernel, mask_antialias, warping_batch_size, gen_chunk_size, m2svid_process_res, gen_batch_size],
            outputs=[step2_status],
        )

        # Wire Title Depth Tools
        desat_gen_btn.click(
            fn=generate_title_desaturation_depth,
            inputs=[desat_clip_dropdown, desat_auto_bg, desat_invert],
            outputs=[desat_status, preview_depth, depth_dropdown, depth_input],
        )

        iso_gen_btn.click(
            fn=generate_3d_title_isolation_depth,
            inputs=[iso_clip_dropdown, iso_color_target, iso_base_elev, iso_relief, iso_auto_bg, iso_invert],
            outputs=[iso_status, preview_depth, depth_dropdown, depth_input],
        )

        # Refresh button
        def _refresh_hub():
            sync_orphan_depth_mp4s()
            src = get_source_video_list()
            dep = get_depth_video_list()
            rows, sum_md, _ = scan_source_library(SOURCE_DIR, DEPTH_DIR, FINAL_DIR)
            return (
                gr.update(choices=[""] + src),
                gr.update(choices=[""] + dep),
                gr.update(choices=[""] + src),
                gr.update(choices=[""] + src),
                gr.update(choices=[""] + src),
                gr.update(choices=[""] + src),
                gr.update(choices=[""] + src),
                gr.update(choices=[""] + src),
                rows,
                sum_md,
            )
            
        refresh_btn.click(
            fn=_refresh_hub,
            inputs=[],
            outputs=[source_dropdown, depth_dropdown, vda_step1_dropdown, dc_step1_dropdown, fd_step1_dropdown, icd_step1_dropdown, desat_clip_dropdown, iso_clip_dropdown, sc_lib_table, sc_lib_summary],
        )
        
        # Wire depth dropdown to update preview
        depth_dropdown.change(
            fn=select_depth_video,
            inputs=[depth_dropdown],
            outputs=[preview_depth],
        )

        # =====================================================================
        # Auto-Pipeline & Sidecar Event Wiring
        # =====================================================================
        sidecar_ui_inputs = [
            sc_profile,
            sc_vast_rate,
            sc_depth_priority,
            sc_stage_dc,
            sc_stage_icd,
            sc_stage_fd,
            sc_stage_vda,
            sc_stage_m2svid,
            sc_dc_window,
            sc_dc_overlap,
            sc_dc_steps,
            sc_dc_max_res,
            sc_dc_guidance,
            sc_dc_attn,
            sc_dc_offload,
            sc_icd_max_res,
            sc_icd_window,
            sc_icd_overlap,
            sc_icd_steps,
            sc_icd_cfg,
            sc_icd_tiled,
            sc_fd_model,
            sc_fd_max_res,
            sc_fd_target_fps,
            sc_fd_max_len,
            sc_fd_global_norm,
            sc_fd_invert,
            sc_vda_model,
            sc_vda_mode,
            sc_vda_input_size,
            sc_vda_max_res,
            sc_vda_target_fps,
            sc_vda_max_len,
            sc_vda_fp32,
            sc_vda_global_norm,
            sc_vda_invert,
            sc_m2s_chunk,
            sc_m2s_batch,
            sc_m2s_res,
            sc_m2s_disparity,
            sc_m2s_convergence,
            sc_m2s_warp_batch,
            sc_m2s_closing,
            sc_m2s_antialias,
            sc_audio_passthrough,
        ]

        def _build_sidecar_dict_from_args(*args) -> dict:
            (
                prof_name, vast_rate, depth_prio,
                do_dc, do_icd, do_fd, do_vda, do_m2svid,
                dc_win, dc_over, dc_steps, dc_res, dc_guide, dc_attn, dc_offload,
                icd_res, icd_win, icd_over, icd_steps, icd_cfg_val, icd_tile,
                fd_mod, fd_res, fd_fps, fd_max_l, fd_norm, fd_inv,
                vda_mod, vda_mode, vda_in_sz, vda_res, vda_fps, vda_max_l, vda_fp32, vda_norm, vda_inv,
                m2s_chunk, m2s_batch, m2s_res, m2s_disp, m2s_conv, m2s_warp_b, m2s_close, m2s_antialias,
                audio_pass
            ) = args
            return {
                "version": "1.1",
                "hardware_profile": prof_name,
                "cloud_vast_rate_per_hour": float(vast_rate),
                "stages_enabled": {
                    "depthcrafter": bool(do_dc),
                    "icdepth": bool(do_icd),
                    "flashdepth": bool(do_fd),
                    "vda": bool(do_vda),
                    "m2svid": bool(do_m2svid),
                },
                "depth_priority": depth_prio,
                "depthcrafter_settings": {
                    "guidance_scale": float(dc_guide),
                    "inference_steps": int(dc_steps),
                    "window_size": int(dc_win),
                    "overlap": int(dc_over),
                    "max_res": int(dc_res),
                    "attn_slicing": dc_attn,
                    "cpu_offload": dc_offload,
                },
                "icdepth_settings": {
                    "max_res": int(icd_res),
                    "window_size": int(icd_win),
                    "overlap": int(icd_over),
                    "inference_steps": int(icd_steps),
                    "cfg_scale": float(icd_cfg_val),
                    "tiled": bool(icd_tile),
                },
                "flashdepth_settings": {
                    "model": fd_mod,
                    "max_res": int(fd_res) if str(fd_res).isdigit() else 1288,
                    "target_fps": float(fd_fps) if str(fd_fps).replace('-', '').replace('.', '').isdigit() else -1.0,
                    "max_len": int(fd_max_l) if str(fd_max_l).replace('-', '').isdigit() else -1,
                    "global_norm": bool(fd_norm),
                    "invert_depth": bool(fd_inv),
                },
                "vda_settings": {
                    "model": vda_mod,
                    "mode": vda_mode,
                    "input_size": int(vda_in_sz),
                    "max_res": int(vda_res) if str(vda_res).isdigit() else 1024,
                    "target_fps": float(vda_fps) if str(vda_fps).replace('-', '').replace('.', '').isdigit() else -1.0,
                    "max_len": int(vda_max_l) if str(vda_max_l).replace('-', '').isdigit() else -1,
                    "fp32": bool(vda_fp32),
                    "global_norm": bool(vda_norm),
                    "invert_depth": bool(vda_inv),
                },
                "m2svid_settings": {
                    "processing_res": m2s_res,
                    "disparity_perc": float(m2s_disp),
                    "convergence_point": float(m2s_conv),
                    "closing_kernel": int(m2s_close),
                    "mask_antialias": bool(m2s_antialias),
                    "warping_batch_size": int(m2s_warp_b),
                    "gen_chunk_size": int(m2s_chunk),
                    "gen_batch_size": int(m2s_batch),
                },
                "audio_passthrough": {
                    "enabled": bool(audio_pass),
                }
            }

        def _unpack_sidecar_dict_to_ui(cfg: dict, status_msg: str):
            dc_s = cfg.get("depthcrafter_settings", {})
            icd_s = cfg.get("icdepth_settings", {})
            fd_s = cfg.get("flashdepth_settings", {})
            vda_s = cfg.get("vda_settings", {})
            m2s_s = cfg.get("m2svid_settings", {})
            stages = cfg.get("stages_enabled", {})
            return (
                cfg.get("hardware_profile", detect_gpu_hardware()),
                cfg.get("cloud_vast_rate_per_hour", 0.50),
                cfg.get("depth_priority", "DepthCrafter"),
                stages.get("depthcrafter", True),
                stages.get("icdepth", False),
                stages.get("flashdepth", False),
                stages.get("vda", False),
                stages.get("m2svid", True),
                dc_s.get("window_size", 30),
                dc_s.get("overlap", 10),
                dc_s.get("inference_steps", 5),
                dc_s.get("max_res", 1024),
                dc_s.get("guidance_scale", 1.0),
                dc_s.get("attn_slicing", "Auto (Adapts to GPU VRAM)"),
                dc_s.get("cpu_offload", "model"),
                icd_s.get("max_res", 720),
                icd_s.get("window_size", 81),
                icd_s.get("overlap", 32),
                icd_s.get("inference_steps", 5),
                icd_s.get("cfg_scale", 5.0),
                icd_s.get("tiled", False),
                fd_s.get("model", "FlashDepth (Full) - 2K VFX Hybrid (Recommended)"),
                fd_s.get("max_res", 1288),
                fd_s.get("target_fps", -1.0),
                fd_s.get("max_len", -1),
                fd_s.get("global_norm", True),
                fd_s.get("invert_depth", False),
                vda_s.get("model", "Large (vitl - Recommended)"),
                vda_s.get("mode", "sliding_window"),
                vda_s.get("input_size", 644),
                vda_s.get("max_res", 1024),
                vda_s.get("target_fps", -1.0),
                vda_s.get("max_len", -1),
                vda_s.get("fp32", False),
                vda_s.get("global_norm", True),
                vda_s.get("invert_depth", False),
                m2s_s.get("gen_chunk_size", 5),
                m2s_s.get("gen_batch_size", 1),
                m2s_s.get("processing_res", "Native"),
                m2s_s.get("disparity_perc", 20.0),
                m2s_s.get("convergence_point", 0.50),
                m2s_s.get("warping_batch_size", 8),
                m2s_s.get("closing_kernel", 11),
                m2s_s.get("mask_antialias", False),
                cfg.get("audio_passthrough", {}).get("enabled", True),
                json.dumps(cfg, indent=2),
                status_msg
            )

        def _handle_save_sidecar(*args):
            cfg = _build_sidecar_dict_from_args(*args)
            msg = save_sidecar_file(cfg, SIDECAR_CONFIG_PATH)
            return json.dumps(cfg, indent=2), msg

        sc_save_btn.click(
            fn=_handle_save_sidecar,
            inputs=sidecar_ui_inputs,
            outputs=[sc_json_display, sc_status_text]
        )

        def _handle_reload_sidecar():
            cfg = load_sidecar_file(SIDECAR_CONFIG_PATH)
            return _unpack_sidecar_dict_to_ui(cfg, "✅ Reloaded sidecar from `pipeline_sidecar.json`.")

        sc_reload_btn.click(
            fn=_handle_reload_sidecar,
            inputs=[],
            outputs=sidecar_ui_inputs + [sc_json_display, sc_status_text]
        )

        def _handle_profile_change(prof_name):
            cfg = get_default_sidecar_config(prof_name)
            return _unpack_sidecar_dict_to_ui(cfg, f"✅ Applied `{prof_name}` profile defaults.")

        sc_profile_defaults_btn.click(
            fn=_handle_profile_change,
            inputs=[sc_profile],
            outputs=sidecar_ui_inputs + [sc_json_display, sc_status_text]
        )
        sc_profile.change(
            fn=_handle_profile_change,
            inputs=[sc_profile],
            outputs=sidecar_ui_inputs + [sc_json_display, sc_status_text]
        )
        sc_icd_max_res.change(
            fn=get_icdepth_res_advisory,
            inputs=[sc_icd_max_res],
            outputs=[sc_icd_res_advisory],
        )

        def _handle_preflight(dc_on, icd_on, fd_on, vda_on, m2s_on):
            html_msg, _ = validate_preflight(
                SOURCE_DIR, DEPTH_DIR, dc_on, vda_on, m2s_on, do_icd=icd_on, do_fd=fd_on
            )
            return html_msg

        for chk in (sc_stage_dc, sc_stage_icd, sc_stage_fd, sc_stage_vda, sc_stage_m2svid):
            chk.change(
                fn=_handle_preflight,
                inputs=[sc_stage_dc, sc_stage_icd, sc_stage_fd, sc_stage_vda, sc_stage_m2svid],
                outputs=[preflight_box]
            )

        def _handle_scan_library():
            rows, sum_md, _ = scan_source_library(SOURCE_DIR, DEPTH_DIR, FINAL_DIR)
            return rows, sum_md

        sc_scan_btn.click(
            fn=_handle_scan_library,
            inputs=[],
            outputs=[sc_lib_table, sc_lib_summary]
        )

        def _handle_run_benchmark(*args):
            cfg = _build_sidecar_dict_from_args(*args)
            save_sidecar_file(cfg, SIDECAR_CONFIG_PATH)
            do_dc = cfg["stages_enabled"].get("depthcrafter", False)
            do_icd = cfg["stages_enabled"].get("icdepth", False)
            do_fd = cfg["stages_enabled"].get("flashdepth", False)
            do_vda = cfg["stages_enabled"].get("vda", False)
            do_m2s = cfg["stages_enabled"].get("m2svid", False)

            html_msg, ok = validate_preflight(
                SOURCE_DIR, DEPTH_DIR, do_dc, do_vda, do_m2s, do_icd=do_icd, do_fd=do_fd
            )
            if not ok:
                yield "❌ Pre-flight check failed. Please review stages above.", "Pre-flight Error", "", gr.update(), gr.update()
                return

            global GLOBAL_CANCEL
            GLOBAL_CANCEL = False
            PIPELINE_STOP_EVENT.clear()

            rows, sum_md, _ = scan_source_library(SOURCE_DIR, DEPTH_DIR, FINAL_DIR)

            last_report = ""
            for status_msg, progress_badge, report_md in run_pipeline_benchmark(
                sidecar_config=cfg,
                do_dc=do_dc,
                do_vda=do_vda,
                do_m2svid=do_m2s,
                source_dir=SOURCE_DIR,
                depth_dir=DEPTH_DIR,
                final_dir=FINAL_DIR,
                dc_runner=step1_run_depthcrafter,
                vda_runner=step1_run_vda_depth,
                m2svid_runner=step2_run_m2svid,
                stop_event=PIPELINE_STOP_EVENT,
                do_icd=do_icd,
                do_fd=do_fd,
                icd_runner=step1_run_icdepth,
                fd_runner=step1_run_flashdepth,
            ):
                if report_md:
                    last_report = report_md
                yield status_msg, progress_badge, report_md or last_report, rows, sum_md

            new_rows, new_sum, _ = scan_source_library(SOURCE_DIR, DEPTH_DIR, FINAL_DIR)
            yield "✅ Benchmark Complete!", "Done", last_report, new_rows, new_sum

        sc_bench_btn.click(
            fn=_handle_run_benchmark,
            inputs=sidecar_ui_inputs,
            outputs=[sc_pipeline_status, sc_progress_badge, sc_bench_report, sc_lib_table, sc_lib_summary]
        )

        def _handle_run_batch(*args):
            cfg = _build_sidecar_dict_from_args(*args)
            save_sidecar_file(cfg, SIDECAR_CONFIG_PATH)
            do_dc = cfg["stages_enabled"].get("depthcrafter", False)
            do_icd = cfg["stages_enabled"].get("icdepth", False)
            do_fd = cfg["stages_enabled"].get("flashdepth", False)
            do_vda = cfg["stages_enabled"].get("vda", False)
            do_m2s = cfg["stages_enabled"].get("m2svid", False)

            html_msg, ok = validate_preflight(
                SOURCE_DIR, DEPTH_DIR, do_dc, do_vda, do_m2s, do_icd=do_icd, do_fd=do_fd
            )
            if not ok:
                yield "❌ Pre-flight check failed. Please review stages above.", "Pre-flight Error", gr.update(), gr.update()
                return

            global GLOBAL_CANCEL
            GLOBAL_CANCEL = False
            PIPELINE_STOP_EVENT.clear()

            rows, sum_md, _ = scan_source_library(SOURCE_DIR, DEPTH_DIR, FINAL_DIR)

            for status_msg, progress_badge in run_pipeline_batch(
                sidecar_config=cfg,
                do_dc=do_dc,
                do_vda=do_vda,
                do_m2svid=do_m2s,
                source_dir=SOURCE_DIR,
                depth_dir=DEPTH_DIR,
                final_dir=FINAL_DIR,
                dc_runner=step1_run_depthcrafter,
                vda_runner=step1_run_vda_depth,
                m2svid_runner=step2_run_m2svid,
                stop_event=PIPELINE_STOP_EVENT,
                do_icd=do_icd,
                do_fd=do_fd,
                icd_runner=step1_run_icdepth,
                fd_runner=step1_run_flashdepth,
            ):
                yield status_msg, progress_badge, rows, sum_md

            new_rows, new_sum, _ = scan_source_library(SOURCE_DIR, DEPTH_DIR, FINAL_DIR)
            yield "🎉 Full Batch Complete!", "Done", new_rows, new_sum

        sc_batch_btn.click(
            fn=_handle_run_batch,
            inputs=sidecar_ui_inputs,
            outputs=[sc_pipeline_status, sc_progress_badge, sc_lib_table, sc_lib_summary]
        )

        sc_stop_btn.click(
            fn=trigger_cancel,
            inputs=[],
            outputs=[sc_pipeline_status]
        )

    return demo


def launch(server_name="127.0.0.1", server_port=7878, share=False):
    demo = create_stereofaster_ui()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        prevent_thread_lock=False,
        head=STEREOFASTER_HEAD_HTML,
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 100% !important; padding: 0 12px !important; }
            .step-header { font-size: 1.3em; font-weight: 600; margin-bottom: 8px; }
            .tab-nav { flex-wrap: wrap !important; gap: 4px !important; }
            .tab-nav button { font-weight: 600 !important; padding: 10px 18px !important; }
            .wrap.size_picker, .wrap.eraser_picker, div[class*="size_picker"], div[class*="eraser_picker"], .wrap.svelte-1y25b79 { min-width: 340px !important; }
            .wrap.size_picker input[type="range"], .wrap.eraser_picker input[type="range"], div[class*="size_picker"] input[type="range"], div[class*="eraser_picker"] input[type="range"] { width: 270px !important; }
        """
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7878)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("Launching StereoFaster WebUI (VDA + DepthCrafter + M2SVid 2-step pipeline)...")
    launch(server_name=args.server_name, server_port=args.server_port, share=args.share)



