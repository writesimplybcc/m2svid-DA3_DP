import os
import sys
import time
import gc
import json
import torch
import cv2
import numpy as np
from pathlib import Path

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

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print("=" * 70)
print(f"ICDEPTH BENCHMARK (FAST MODE): 3 Steps, CFG 1.0, Credit Detection ON")
print(f"Video Spec: {total_frames} frames | {w}x{h} | {fps:.2f} fps | {total_frames/fps:.2f}s duration")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 70, flush=True)

reset_cuda()

from m2svid.prepare_icdepth import run_icdepth_depth, unload_icdepth_model, load_clip_credit_sidecar

hint = load_clip_credit_sidecar(VIDEO_PATH)
print(f"Loaded credit sidecar hint: {hint!r}", flush=True)

t0 = time.perf_counter()
depth_icd_fast, out_fps = run_icdepth_depth(
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
t1 = time.perf_counter()
dur = t1 - t0
peak_vram = get_peak_vram_gb()
fps_speed = total_frames / max(1e-5, dur)

print("\n" + "=" * 70)
print(f"ICDEPTH (3 Steps, CFG 1.0, Credit ON) FINISHED:")
print(f"Total Time:    {dur:.2f} seconds ({dur/60:.2f} minutes)")
print(f"Speed:         {fps_speed:.2f} FPS")
print(f"Peak VRAM:     {peak_vram:.2f} GB")
print(f"Output Shape:  {depth_icd_fast.shape}")
print(f"Depth Range:   [{depth_icd_fast.min():.4f}, {depth_icd_fast.max():.4f}]")
print("=" * 70, flush=True)

# Save result
np.save(OUTPUT_DIR / "icdepth_fast_3steps_cfg1_depth.npy", depth_icd_fast)

# Unload
unload_icdepth_model()
reset_cuda()

# Save JSON result
res = {
    "ICDepth (3 steps, CFG 1.0, Credit ON)": {
        "time_sec": round(dur, 2),
        "fps": round(fps_speed, 2),
        "peak_vram_gb": round(peak_vram, 2),
        "shape": list(depth_icd_fast.shape),
    }
}
with open(OUTPUT_DIR / "benchmark_icdepth_fast_results.json", "w") as f:
    json.dump(res, f, indent=2)

# Generate a visual comparison of Frame 36: Baseline (5-step, CFG 5.0) vs Fast (3-step, CFG 1.0)
f_idx = 36
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frames.append(frame)
cap.release()
rgb_bgr = frames[f_idx]
H_orig, W_orig = rgb_bgr.shape[:2]

d_baseline = np.load(OUTPUT_DIR / "icdepth_depth.npy")[f_idx] if (OUTPUT_DIR / "icdepth_depth.npy").exists() else None
d_fast = depth_icd_fast[f_idx]

def _make_pane(title, subtitle, d_arr):
    if d_arr.shape != (H_orig, W_orig):
        d_arr = cv2.resize(d_arr, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    vis = cv2.applyColorMap((np.clip(d_arr, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    banner = np.zeros((55, W_orig, 3), dtype=np.uint8)
    cv2.putText(banner, title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(banner, subtitle, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, vis])

banner_rgb = np.zeros((55, W_orig, 3), dtype=np.uint8)
cv2.putText(banner_rgb, "Source RGB Frame 36", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(banner_rgb, "UNP854_V1-0003 (73 frames)", (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
pane_rgb = np.vstack([banner_rgb, rgb_bgr])

pane_fast = _make_pane("ICDepth: Fast Mode (3 Steps, CFG 1.0)", f"{dur:.1f}s | {fps_speed:.2f} FPS | Credit Elevation ON", d_fast)

if d_baseline is not None:
    pane_baseline = _make_pane("ICDepth: Standard (5 Steps, CFG 5.0)", "397.4s | 0.18 FPS | Baseline", d_baseline)
    compare_img = np.hstack([pane_rgb, pane_baseline, pane_fast])
else:
    compare_img = np.hstack([pane_rgb, pane_fast])

cv2.imwrite(str(OUTPUT_DIR / "icd_baseline_vs_fast.png"), compare_img)
print("Saved comparison image: icd_baseline_vs_fast.png", flush=True)
