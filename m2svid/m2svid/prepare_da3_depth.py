"""
DA3 (Depth Anything 3) to M2SVid depth preparer.

Produces a depth .npz in the exact format expected by m2svid/warping.py
(i.e. file with top-level key 'depth' containing (T, H, W) array).

This enables the modern DA3 + M2SVid pipeline, replacing the legacy
DepthCrafter stage with a faster, higher-quality monocular depth estimator.

Usage (after installing DA3):
    PYTHONPATH="Depth-Anything-3/src:${PYTHONPATH}" python -m m2svid.m2svid.prepare_da3_depth \
        --video demo/input.mp4 \
        --output outputs/da3/input.npz \
        --model depth-anything/DA3NESTED-GIANT-LARGE-1.1 \
        --process-res 504 \
        --fps 0   # 0 = all frames (native video rate)

The resulting .npz can be fed directly to:
    python warping.py --video_path ... --depth_path outputs/da3/input.npz ...

Recommended models for M2SVid stereo conversion:
  - depth-anything/DA3NESTED-GIANT-LARGE-1.1   (best overall quality, official -1.1 retrained version)
  - depth-anything/DA3MONO-LARGE               (best pure relative monocular depth for geometric warping)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def load_video_frames(video_path: str, target_fps: float = 0.0) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    """Extract frames from video. target_fps=0 means every frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
            # BGR -> RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted")

    h, w = frames[0].shape[:2]
    actual_fps = video_fps / frame_interval if frame_interval > 0 else video_fps
    return frames, actual_fps, (h, w)


_cached_da3_model = None
_cached_da3_model_name = None

def unload_da3_model():
    global _cached_da3_model, _cached_da3_model_name
    if _cached_da3_model is not None:
        print("[DA3] Unloading model to free VRAM")
        del _cached_da3_model
        _cached_da3_model = None
        _cached_da3_model_name = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_da3_depth(
    frames: list,
    model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    process_res: int = 504,
    device: str = "cuda",
    batch_size: int = 8,
    progress=None,
) -> np.ndarray:
    """
    Run DA3 inference on a list of RGB frames.

    Returns:
        depth: np.ndarray of shape (T, H_out, W_out)  -- note: may be at process_res resolution
    """
    global _cached_da3_model, _cached_da3_model_name
    
    # Lazy import: only load the heavy DA3 model (and any optional 3DGS code) when this function is actually called.
    # This saves VRAM and avoids pulling in gsplat until the user triggers depth estimation.
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.io.input_processor import InputProcessor

    if _cached_da3_model is None or _cached_da3_model_name != model_name:
        print(f"[DA3] Loading model: {model_name}")
        _cached_da3_model = DepthAnything3.from_pretrained(model_name).to(device).eval()
        _cached_da3_model_name = model_name
    
    model = _cached_da3_model

    depths = []
    total_batches = (len(frames) + batch_size - 1) // batch_size
    for idx, i in enumerate(tqdm(range(0, len(frames), batch_size), desc="DA3 depth inference")):
        if progress:
            try: progress(float(idx) / total_batches, desc=f"DA3 Depth Inference: batch {idx+1}/{total_batches}")
            except Exception: pass
        batch_frames = frames[i : i + batch_size]

        # Prepare batch for DA3 (list of paths or arrays)
        # The API accepts list of numpy arrays (RGB) or paths
        preds = model.inference(
            image=batch_frames,
            # Do not pass export_dir (or export_format) — we only want the in-memory Prediction
            process_res=process_res,
            process_res_method="upper_bound_resize",
            # For pure monocular video -> stereo we usually do NOT want multi-view pose
            # so single or independent frames. DA3 handles N frames gracefully.
        )

        # preds.depth is (N, H, W) at the processed resolution
        d = preds.depth.astype(np.float32)
        depths.append(d)

    depth = np.concatenate(depths, axis=0)  # (T, H_proc, W_proc)

    # Optional: upsample back to original video resolution?
    # For M2SVid warping we want depth at (or close to) the video resolution used by warping.
    # warping.py does cv2.resize internally to video size, so we can keep processed size
    # or upsample here for higher quality warping. For now, return as-is (user can control via process_res).
    return depth


def save_m2svid_compatible_npz(depth: np.ndarray, out_path: str):
    """Save with the single key 'depth' that m2svid/warping.py expects."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, depth=depth)
    print(f"[DA3->M2SVid] Saved compatible depth to {out_path}  shape={depth.shape}")


def main():
    parser = argparse.ArgumentParser(description="Prepare DA3 depth for M2SVid warping stage")
    parser.add_argument("--video", required=True, help="Input monocular video")
    parser.add_argument("--output", required=True, help="Output .npz path for M2SVid (will contain 'depth' key)")
    parser.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                        help="DA3 model preset or HF repo id")
    parser.add_argument("--process-res", type=int, default=504, help="DA3 processing resolution (controls speed/quality)")
    parser.add_argument("--fps", type=float, default=0.0, help="Target sampling FPS (0 = native / all frames)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("[DA3->M2SVid] Extracting frames...")
    frames, video_fps, (h, w) = load_video_frames(args.video, args.fps)
    print(f"  Loaded {len(frames)} frames @ ~{video_fps:.2f} fps, original res {w}x{h}")

    depth = run_da3_depth(
        frames,
        model_name=args.model,
        process_res=args.process_res,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Note: depth may be at lower res than video. warping.py will resize it.
    # For best quality we upsample depth to original video resolution here.
    if depth.shape[1:] != (h, w):
        print(f"[DA3->M2SVid] Upsampling depth from {depth.shape[1:]} to video resolution {h}x{w} ...")
        depth = np.stack([
            cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
            for d in depth
        ])

    # DA3 outputs high value = far by default.
    # M2SVid warping expects high value = near (same convention as DepthCrafter).
    # We always invert so the output .npz is directly usable.
    print("[DA3->M2SVid] Inverting depth (DA3 high=far -> M2SVid high=near convention)")
    depth = -depth

    save_m2svid_compatible_npz(depth, args.output)

    print("[OK] DA3 depth preparation complete. Next steps:")
    print("   1. Run warping.py with this depth file (same as legacy)")
    print("   2. Run inpaint_and_refine.py")



if __name__ == "__main__":
    main()
