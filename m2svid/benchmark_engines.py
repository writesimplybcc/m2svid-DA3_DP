import os
import sys
import time
import gc
import json
import torch
import cv2
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, "/app")
sys.path.insert(0, "/app/m2svid")

VIDEO_PATH = "/app/source_videos/UNP854_V1-0003.mp4"
OUTPUT_DIR = Path("/app/depthmaps_videos/benchmark_unp0003")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def reset_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

def get_peak_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0

def inspect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return cnt, w, h, fps

results = {}

total_frames, width, height, fps = inspect_video(VIDEO_PATH)
print("=" * 70)
print(f"BENCHMARK: Video Depth Engines on UNP854_V1-0003.mp4")
print(f"Video Spec: {total_frames} frames | {width}x{height} | {fps:.2f} fps | {total_frames/fps:.2f}s duration")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 70, flush=True)

# -------------------------------------------------------------
# 1. Video Depth Anything (VDA) - ViT-Large (vitl)
# -------------------------------------------------------------
print("\n>>> [1/6] Running Video Depth Anything (VDA - ViT-Large)...", flush=True)
try:
    reset_cuda()
    from m2svid.prepare_vda_depth import run_vda_depth, unload_vda_model
    t0 = time.perf_counter()
    depth_vda_l, _ = run_vda_depth(
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
    t1 = time.perf_counter()
    dur = t1 - t0
    peak_vram = get_peak_vram_gb()
    results["VDA (ViT-Large)"] = {
        "time_sec": round(dur, 2),
        "fps": round(total_frames / max(1e-5, dur), 2),
        "peak_vram_gb": round(peak_vram, 2),
        "status": "SUCCESS",
        "shape": list(depth_vda_l.shape) if depth_vda_l is not None else None,
    }
    np.save(OUTPUT_DIR / "vda_large_depth.npy", depth_vda_l)
    print(f"-> VDA (ViT-Large): {dur:.2f}s ({total_frames/dur:.2f} FPS, Peak VRAM: {peak_vram:.2f} GB)", flush=True)
    unload_vda_model()
except Exception as e:
    print(f"-> VDA (ViT-Large) FAILED: {e}", flush=True)
    results["VDA (ViT-Large)"] = {"status": "FAILED", "error": str(e)}

# -------------------------------------------------------------
# 2. Video Depth Anything (VDA) - ViT-Base (vitb)
# -------------------------------------------------------------
print("\n>>> [2/6] Running Video Depth Anything (VDA - ViT-Base)...", flush=True)
try:
    reset_cuda()
    from m2svid.prepare_vda_depth import run_vda_depth, unload_vda_model
    t0 = time.perf_counter()
    depth_vda_b, _ = run_vda_depth(
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
    t1 = time.perf_counter()
    dur = t1 - t0
    peak_vram = get_peak_vram_gb()
    results["VDA (ViT-Base)"] = {
        "time_sec": round(dur, 2),
        "fps": round(total_frames / max(1e-5, dur), 2),
        "peak_vram_gb": round(peak_vram, 2),
        "status": "SUCCESS",
        "shape": list(depth_vda_b.shape) if depth_vda_b is not None else None,
    }
    np.save(OUTPUT_DIR / "vda_base_depth.npy", depth_vda_b)
    print(f"-> VDA (ViT-Base): {dur:.2f}s ({total_frames/dur:.2f} FPS, Peak VRAM: {peak_vram:.2f} GB)", flush=True)
    unload_vda_model()
except Exception as e:
    print(f"-> VDA (ViT-Base) FAILED: {e}", flush=True)
    results["VDA (ViT-Base)"] = {"status": "FAILED", "error": str(e)}

# -------------------------------------------------------------
# 3. FlashDepth - 2K VFX Hybrid (Recommended)
# -------------------------------------------------------------
print("\n>>> [3/6] Running FlashDepth (2K VFX Hybrid)...", flush=True)
try:
    reset_cuda()
    from m2svid.prepare_flashdepth import run_flashdepth, unload_flashdepth_model
    t0 = time.perf_counter()
    depth_fd_full, _ = run_flashdepth(
        video_path=VIDEO_PATH,
        model_name="FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
        max_res=854,
        target_fps=-1,
        max_len=-1,
        global_norm=True,
        invert_depth=False,
    )
    t1 = time.perf_counter()
    dur = t1 - t0
    peak_vram = get_peak_vram_gb()
    results["FlashDepth (2K VFX Hybrid)"] = {
        "time_sec": round(dur, 2),
        "fps": round(total_frames / max(1e-5, dur), 2),
        "peak_vram_gb": round(peak_vram, 2),
        "status": "SUCCESS",
        "shape": list(depth_fd_full.shape) if depth_fd_full is not None else None,
    }
    np.save(OUTPUT_DIR / "flashdepth_full_depth.npy", depth_fd_full)
    print(f"-> FlashDepth (Full): {dur:.2f}s ({total_frames/dur:.2f} FPS, Peak VRAM: {peak_vram:.2f} GB)", flush=True)
    unload_flashdepth_model()
except Exception as e:
    print(f"-> FlashDepth (Full) FAILED: {e}", flush=True)
    results["FlashDepth (2K VFX Hybrid)"] = {"status": "FAILED", "error": str(e)}

# -------------------------------------------------------------
# 4. FlashDepth - ViT-Large Standalone
# -------------------------------------------------------------
print("\n>>> [4/6] Running FlashDepth (ViT-Large Standalone)...", flush=True)
try:
    reset_cuda()
    from m2svid.prepare_flashdepth import run_flashdepth, unload_flashdepth_model
    t0 = time.perf_counter()
    depth_fd_l, _ = run_flashdepth(
        video_path=VIDEO_PATH,
        model_name="FlashDepth-L - ViT-Large Standalone (Best for 480p)",
        max_res=854,
        target_fps=-1,
        max_len=-1,
        global_norm=True,
        invert_depth=False,
    )
    t1 = time.perf_counter()
    dur = t1 - t0
    peak_vram = get_peak_vram_gb()
    results["FlashDepth (ViT-Large)"] = {
        "time_sec": round(dur, 2),
        "fps": round(total_frames / max(1e-5, dur), 2),
        "peak_vram_gb": round(peak_vram, 2),
        "status": "SUCCESS",
        "shape": list(depth_fd_l.shape) if depth_fd_l is not None else None,
    }
    np.save(OUTPUT_DIR / "flashdepth_vitl_depth.npy", depth_fd_l)
    print(f"-> FlashDepth (ViT-Large): {dur:.2f}s ({total_frames/dur:.2f} FPS, Peak VRAM: {peak_vram:.2f} GB)", flush=True)
    unload_flashdepth_model()
except Exception as e:
    print(f"-> FlashDepth (ViT-Large) FAILED: {e}", flush=True)
    results["FlashDepth (ViT-Large)"] = {"status": "FAILED", "error": str(e)}

# -------------------------------------------------------------
# 5. DepthCrafter (Diffusion Video Depth)
# -------------------------------------------------------------
print("\n>>> [5/6] Running DepthCrafter (Diffusion Model - 5 steps)...", flush=True)
try:
    reset_cuda()
    from m2svid.prepare_depthcrafter import run_depthcrafter_depth
    t0 = time.perf_counter()
    depth_dc = run_depthcrafter_depth(
        video_path=VIDEO_PATH,
        process_res=854,
        guidance_scale=1.2,
        num_inference_steps=5,
        window_size=30,
        overlap=10,
        attn_slicing="Auto (Adapts to GPU VRAM)",
        cpu_offload="Auto (Adapts to GPU VRAM)",
    )
    t1 = time.perf_counter()
    dur = t1 - t0
    peak_vram = get_peak_vram_gb()
    results["DepthCrafter (5 steps)"] = {
        "time_sec": round(dur, 2),
        "fps": round(total_frames / max(1e-5, dur), 2),
        "peak_vram_gb": round(peak_vram, 2),
        "status": "SUCCESS",
        "shape": list(depth_dc.shape) if depth_dc is not None else None,
    }
    np.save(OUTPUT_DIR / "depthcrafter_depth.npy", depth_dc)
    print(f"-> DepthCrafter: {dur:.2f}s ({total_frames/dur:.2f} FPS, Peak VRAM: {peak_vram:.2f} GB)", flush=True)
except Exception as e:
    print(f"-> DepthCrafter FAILED: {e}", flush=True)
    results["DepthCrafter (5 steps)"] = {"status": "FAILED", "error": str(e)}

# -------------------------------------------------------------
# 6. ICDepth (Wan2.1 In-Context DiT - 5 steps, CFG 5.0)
# -------------------------------------------------------------
print("\n>>> [6/6] Running ICDepth (Wan2.1 DiT - 5 steps, CFG 5.0)...", flush=True)
try:
    reset_cuda()
    from m2svid.prepare_icdepth import run_icdepth_depth, unload_icdepth_model
    t0 = time.perf_counter()
    depth_icd, _ = run_icdepth_depth(
        video_path=VIDEO_PATH,
        max_res=720,
        window_size=81,
        overlap=32,
        num_inference_steps=5,
        cfg_scale=5.0,
        tiled=False,
        void_title_recovery=True,
        volumetric_splash_recovery=False,
        credit_pop=True,
    )
    t1 = time.perf_counter()
    dur = t1 - t0
    peak_vram = get_peak_vram_gb()
    results["ICDepth (5 steps, CFG 5.0)"] = {
        "time_sec": round(dur, 2),
        "fps": round(total_frames / max(1e-5, dur), 2),
        "peak_vram_gb": round(peak_vram, 2),
        "status": "SUCCESS",
        "shape": list(depth_icd.shape) if depth_icd is not None else None,
    }
    np.save(OUTPUT_DIR / "icdepth_depth.npy", depth_icd)
    print(f"-> ICDepth: {dur:.2f}s ({total_frames/dur:.2f} FPS, Peak VRAM: {peak_vram:.2f} GB)", flush=True)
    unload_icdepth_model()
except Exception as e:
    print(f"-> ICDepth FAILED: {e}", flush=True)
    results["ICDepth (5 steps, CFG 5.0)"] = {"status": "FAILED", "error": str(e)}

reset_cuda()

# Save JSON results
with open(OUTPUT_DIR / "benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("FINAL BENCHMARK SUMMARY (Sorted from Fastest to Slowest):")
print("=" * 70)
successful = [(k, v) for k, v in results.items() if v.get("status") == "SUCCESS"]
successful.sort(key=lambda x: x[1]["time_sec"])

for rank, (name, data) in enumerate(successful, 1):
    print(f"#{rank} | {name:<32} | Time: {data['time_sec']:>6.2f}s | Speed: {data['fps']:>6.2f} FPS | Peak VRAM: {data['peak_vram_gb']:>5.2f} GB")

print("=" * 70, flush=True)
