import os
import sys
import time
import gc
import cv2
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, "/app")
sys.path.insert(0, "/app/m2svid")

from webui import save_m2svid_compatible_npz, _create_depth_preview_video

VIDEO_PATH = "/app/source_videos/UNP854_V1-0003.mp4"
DEPTH_DIR = Path("/app/depthmaps_videos")
DEPTH_DIR.mkdir(parents=True, exist_ok=True)
STEM = Path(VIDEO_PATH).stem

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print("=" * 70)
print(f"GENERATING ALL DEPTHMAPS & PREVIEWS FOR: {STEM}.mp4")
print(f"Specs: {total_frames} frames | {w}x{h} | {fps:.2f} fps")
print(f"Output Directory: {DEPTH_DIR}")
print("=" * 70, flush=True)

def reset_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# -------------------------------------------------------------
# 1. Video Depth Anything (VDA) - ViT-Base
# -------------------------------------------------------------
out_npz_vda_b = DEPTH_DIR / f"{STEM}_VDA_vitb_depth.npz"
out_mp4_vda_b = DEPTH_DIR / f"{STEM}_VDA_vitb_depth.mp4"
if not out_npz_vda_b.exists():
    print(f"\n[1/6] Computing VDA ViT-Base -> {out_npz_vda_b.name}...", flush=True)
    reset_cuda()
    from m2svid.prepare_vda_depth import run_vda_depth, unload_vda_model
    t0 = time.time()
    depth, _ = run_vda_depth(
        video_path=VIDEO_PATH,
        encoder="vitb",
        input_size=518,
        max_res=854,
        mode="sliding",
        target_fps=-1,
        max_len=-1,
        fp32=False,
        global_norm=True,
        invert_depth=False,
    )
    save_m2svid_compatible_npz(depth, str(out_npz_vda_b))
    _create_depth_preview_video(depth, str(out_mp4_vda_b), fps)
    print(f"✅ Saved {out_npz_vda_b.name} and {out_mp4_vda_b.name} ({time.time() - t0:.1f}s)", flush=True)
    unload_vda_model()
else:
    print(f"⏭️ {out_npz_vda_b.name} already exists. Skipping.")

# -------------------------------------------------------------
# 2. FlashDepth - ViT-Large Standalone
# -------------------------------------------------------------
out_npz_fd_l = DEPTH_DIR / f"{STEM}_FD_vitl_depth.npz"
out_mp4_fd_l = DEPTH_DIR / f"{STEM}_FD_vitl_depth.mp4"
if not out_npz_fd_l.exists():
    print(f"\n[2/6] Computing FlashDepth ViT-L -> {out_npz_fd_l.name}...", flush=True)
    reset_cuda()
    from m2svid.prepare_flashdepth import run_flashdepth, unload_flashdepth_model
    t0 = time.time()
    depth, _ = run_flashdepth(
        video_path=VIDEO_PATH,
        model_name="FlashDepth-L - ViT-Large Standalone (Best for 480p)",
        max_res=854,
        target_fps=-1,
        max_len=-1,
        global_norm=True,
        invert_depth=False,
    )
    save_m2svid_compatible_npz(depth, str(out_npz_fd_l))
    _create_depth_preview_video(depth, str(out_mp4_fd_l), fps)
    print(f"✅ Saved {out_npz_fd_l.name} and {out_mp4_fd_l.name} ({time.time() - t0:.1f}s)", flush=True)
    unload_flashdepth_model()
else:
    print(f"⏭️ {out_npz_fd_l.name} already exists. Skipping.")

# -------------------------------------------------------------
# 3. Video Depth Anything (VDA) - ViT-Large
# -------------------------------------------------------------
out_npz_vda_l = DEPTH_DIR / f"{STEM}_VDA_vitl_depth.npz"
out_mp4_vda_l = DEPTH_DIR / f"{STEM}_VDA_vitl_depth.mp4"
if not out_npz_vda_l.exists():
    print(f"\n[3/6] Computing VDA ViT-Large -> {out_npz_vda_l.name}...", flush=True)
    reset_cuda()
    from m2svid.prepare_vda_depth import run_vda_depth, unload_vda_model
    t0 = time.time()
    depth, _ = run_vda_depth(
        video_path=VIDEO_PATH,
        encoder="vitl",
        input_size=518,
        max_res=854,
        mode="sliding",
        target_fps=-1,
        max_len=-1,
        fp32=False,
        global_norm=True,
        invert_depth=False,
    )
    save_m2svid_compatible_npz(depth, str(out_npz_vda_l))
    _create_depth_preview_video(depth, str(out_mp4_vda_l), fps)
    print(f"✅ Saved {out_npz_vda_l.name} and {out_mp4_vda_l.name} ({time.time() - t0:.1f}s)", flush=True)
    unload_vda_model()
else:
    print(f"⏭️ {out_npz_vda_l.name} already exists. Skipping.")

# -------------------------------------------------------------
# 4. FlashDepth - 2K VFX Hybrid
# -------------------------------------------------------------
out_npz_fd = DEPTH_DIR / f"{STEM}_FD_depth.npz"
out_mp4_fd = DEPTH_DIR / f"{STEM}_FD_depth.mp4"
if not out_npz_fd.exists():
    print(f"\n[4/6] Computing FlashDepth 2K Hybrid -> {out_npz_fd.name}...", flush=True)
    reset_cuda()
    from m2svid.prepare_flashdepth import run_flashdepth, unload_flashdepth_model
    t0 = time.time()
    depth, _ = run_flashdepth(
        video_path=VIDEO_PATH,
        model_name="FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
        max_res=854,
        target_fps=-1,
        max_len=-1,
        global_norm=True,
        invert_depth=False,
    )
    save_m2svid_compatible_npz(depth, str(out_npz_fd))
    _create_depth_preview_video(depth, str(out_mp4_fd), fps)
    print(f"✅ Saved {out_npz_fd.name} and {out_mp4_fd.name} ({time.time() - t0:.1f}s)", flush=True)
    unload_flashdepth_model()
else:
    print(f"⏭️ {out_npz_fd.name} already exists. Skipping.")

# -------------------------------------------------------------
# 5. DepthCrafter
# -------------------------------------------------------------
out_npz_dc = DEPTH_DIR / f"{STEM}_DC_depth.npz"
out_mp4_dc = DEPTH_DIR / f"{STEM}_DC_depth.mp4"
if not out_npz_dc.exists():
    print(f"\n[5/6] Computing DepthCrafter -> {out_npz_dc.name}...", flush=True)
    reset_cuda()
    from m2svid.prepare_depthcrafter import run_depthcrafter_depth
    t0 = time.time()
    depth = run_depthcrafter_depth(
        video_path=VIDEO_PATH,
        process_res=854,
        guidance_scale=1.2,
        num_inference_steps=5,
        window_size=30,
        overlap=10,
        attn_slicing="Auto (Adapts to GPU VRAM)",
        cpu_offload="Auto (Adapts to GPU VRAM)",
    )
    if depth.shape[1:] != (h, w):
        depth = np.stack([cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC) for d in depth])
    save_m2svid_compatible_npz(depth, str(out_npz_dc))
    _create_depth_preview_video(depth, str(out_mp4_dc), fps)
    print(f"✅ Saved {out_npz_dc.name} and {out_mp4_dc.name} ({time.time() - t0:.1f}s)", flush=True)
else:
    print(f"⏭️ {out_npz_dc.name} already exists. Skipping.")

# -------------------------------------------------------------
# 6. ICDepth (Fast Mode: 3 Steps, CFG 1.0, Credits ON)
# -------------------------------------------------------------
out_npz_icd = DEPTH_DIR / f"{STEM}_ICD_depth.npz"
out_mp4_icd = DEPTH_DIR / f"{STEM}_ICD_depth.mp4"
if not out_npz_icd.exists():
    print(f"\n[6/6] Computing ICDepth (Fast 3-step, CFG 1.0) -> {out_npz_icd.name}...", flush=True)
    reset_cuda()
    from m2svid.prepare_icdepth import run_icdepth_depth, unload_icdepth_model, load_clip_credit_sidecar
    t0 = time.time()
    hint = load_clip_credit_sidecar(VIDEO_PATH)
    depth, out_fps = run_icdepth_depth(
        video_path=VIDEO_PATH,
        max_res=720,
        window_size=81,
        overlap=32,
        num_inference_steps=3,
        cfg_scale=1.0,
        tiled=False,
        void_title_recovery=True,
        volumetric_splash_recovery=False,
        credit_pop=True,
        credit_text_hint=hint,
    )
    if depth.shape[1:] != (h, w):
        interp = cv2.INTER_AREA if (w <= depth.shape[2] and h <= depth.shape[1]) else cv2.INTER_LINEAR
        depth = np.stack([cv2.resize(d, (w, h), interpolation=interp) for d in depth])
    save_m2svid_compatible_npz(depth, str(out_npz_icd))
    _create_depth_preview_video(depth, str(out_mp4_icd), out_fps)
    print(f"✅ Saved {out_npz_icd.name} and {out_mp4_icd.name} ({time.time() - t0:.1f}s)", flush=True)
    unload_icdepth_model()
else:
    print(f"⏭️ {out_npz_icd.name} already exists. Skipping.")

reset_cuda()
print("\n" + "=" * 70)
print(f"ALL DEPTH FILES GENERATED & READY FOR STEP 2 (M2SVID) IN: {DEPTH_DIR}")
print("=" * 70, flush=True)
