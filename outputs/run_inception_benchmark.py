import os
import sys
import time
import shutil
import json
import torch
import cv2
import numpy as np
from pathlib import Path

sys.path.append('/app')
from webui import (
    step1_run_vda_depth,
    step1_run_depthcrafter,
    step1_run_flashdepth,
    step1_run_icdepth,
    _create_depth_preview_video,
    save_m2svid_compatible_npz,
    DEPTH_DIR,
    SOURCE_DIR,
)

VIDEO_NAME = "Inception - Final Trailer_V2-0037_cropped.mp4"
VIDEO_PATH = str(SOURCE_DIR / VIDEO_NAME)

cap = cv2.VideoCapture(VIDEO_PATH)
TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 23.976
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

STEM = Path(VIDEO_NAME).stem

print("=" * 80)
print(f"RESUMING BENCHMARK: INCEPTION 0037")
print(f"Specs: {TOTAL_FRAMES} frames | {W}x{H} | {FPS:.3f} fps | {TOTAL_FRAMES/FPS:.2f}s duration")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 80)

# Pre-populate completed runs
results = [
    {
        "engine": "Video Depth Anything",
        "config": "ViT-Base (vitb)",
        "time_sec": 27.13,
        "fps": 2.25,
        "peak_vram_gb": 4.00,
        "mp4": f"/app/depthmaps_videos/{STEM}_VDAB_depth.mp4",
        "npz": f"/app/depthmaps_videos/{STEM}_VDAB_depth.npz",
    },
    {
        "engine": "FlashDepth",
        "config": "2K VFX Hybrid",
        "time_sec": 53.88,
        "fps": 1.13,
        "peak_vram_gb": 2.19,
        "mp4": f"/app/depthmaps_videos/{STEM}_FDF_depth.mp4",
        "npz": f"/app/depthmaps_videos/{STEM}_FDF_depth.npz",
    },
    {
        "engine": "Video Depth Anything",
        "config": "ViT-Large (vitl)",
        "time_sec": 105.65,
        "fps": 0.58,
        "peak_vram_gb": 9.38,
        "mp4": f"/app/depthmaps_videos/{STEM}_VDAL_depth.mp4",
        "npz": f"/app/depthmaps_videos/{STEM}_VDAL_depth.npz",
    },
    {
        "engine": "FlashDepth",
        "config": "ViT-Large Standalone",
        "time_sec": 170.32,
        "fps": 0.36,
        "peak_vram_gb": 2.30,
        "mp4": f"/app/depthmaps_videos/{STEM}_FDL_depth.mp4",
        "npz": f"/app/depthmaps_videos/{STEM}_FDL_depth.npz",
    },
]

def run_test(engine_name, config_desc, fn, *args, **kwargs):
    print(f"\n>>> Running: {engine_name} ({config_desc})...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    t0 = time.perf_counter()
    status, mp4_path, npz_path, _ = fn(*args, **kwargs)
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    fps_val = TOTAL_FRAMES / elapsed if elapsed > 0 else 0.0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    print(f"Completed in {elapsed:.2f}s ({fps_val:.2f} FPS) | Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Status: {status[:100]}...")
    print(f"Outputs: MP4={mp4_path}, NPZ={npz_path}")
    
    res = {
        "engine": engine_name,
        "config": config_desc,
        "time_sec": round(elapsed, 2),
        "fps": round(fps_val, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "mp4": str(mp4_path),
        "npz": str(npz_path),
    }
    results.append(res)
    torch.cuda.empty_cache()
    return res

# 5. DepthCrafter (Standard process_res=720, 5-Step, Window 30)
run_test(
    "DepthCrafter",
    "Diffusion 5-Step, Win 30",
    step1_run_depthcrafter,
    video_path=VIDEO_NAME,
    process_res=720,
    guidance_scale=1.0,
    inference_steps=5,
    window_size=30,
    overlap=10,
)

# 6. ICDepth (Fast Mode: 3 Steps, CFG 1.0, Credits ON)
icd_fast_res = run_test(
    "ICDepth (Fast Mode)",
    "3 Steps, CFG 1.0, Credits ON",
    step1_run_icdepth,
    input_video=VIDEO_NAME,
    max_res=720,
    window_size=81,
    overlap=20,
    steps=3,
    cfg_scale=1.0,
    credit_pop=True,
    void_title_recovery=True,
    volumetric_splash_recovery=False,
)

# Copy fast mode to explicit backup
fast_npz = DEPTH_DIR / f"{STEM}_ICD_fast_depth.npz"
fast_mp4 = DEPTH_DIR / f"{STEM}_ICD_fast_depth.mp4"
if os.path.exists(icd_fast_res["npz"]):
    shutil.copy2(icd_fast_res["npz"], str(fast_npz))
if os.path.exists(icd_fast_res["mp4"]):
    shutil.copy2(icd_fast_res["mp4"], str(fast_mp4))

# 7. ICDepth (Baseline Mode: 5 Steps, CFG 5.0, Credits ON)
icd_base_res = run_test(
    "ICDepth (Baseline Mode)",
    "5 Steps, CFG 5.0, Credits ON",
    step1_run_icdepth,
    input_video=VIDEO_NAME,
    max_res=720,
    window_size=81,
    overlap=20,
    steps=5,
    cfg_scale=5.0,
    credit_pop=True,
    void_title_recovery=True,
    volumetric_splash_recovery=False,
)

# Copy baseline mode to explicit backup
base_npz = DEPTH_DIR / f"{STEM}_ICD_base_depth.npz"
base_mp4 = DEPTH_DIR / f"{STEM}_ICD_base_depth.mp4"
if os.path.exists(icd_base_res["npz"]):
    shutil.copy2(icd_base_res["npz"], str(base_npz))
if os.path.exists(icd_base_res["mp4"]):
    shutil.copy2(icd_base_res["mp4"], str(base_mp4))

# Restore Fast mode as the active {STEM}_ICD_depth
shutil.copy2(str(fast_npz), str(DEPTH_DIR / f"{STEM}_ICD_depth.npz"))
shutil.copy2(str(fast_mp4), str(DEPTH_DIR / f"{STEM}_ICD_depth.mp4"))

# Save results JSON
with open("/app/outputs/inception0037_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("ALL BENCHMARKS COMPLETE! SUMMARY TABLE:")
print(f"{'Engine':<26} | {'Config':<28} | {'Time (s)':<10} | {'FPS':<8} | {'Peak VRAM':<10}")
print("-" * 88)
for r in results:
    print(f"{r['engine']:<26} | {r['config']:<28} | {r['time_sec']:<10.2f} | {r['fps']:<8.2f} | {r['peak_vram_gb']:<8.2f} GB")
print("=" * 80)
