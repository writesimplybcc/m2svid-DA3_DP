import os
import sys
import numpy as np
import torch
import cv2

# Ensure third_party/DepthCrafter is in sys.path
DEPTHCRAFTER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "DepthCrafter"))
if DEPTHCRAFTER_ROOT not in sys.path:
    sys.path.insert(0, DEPTHCRAFTER_ROOT)

_cached_depthcrafter_model = None

def get_depthcrafter_model(unet_path="tencent/DepthCrafter"):
    global _cached_depthcrafter_model
    if _cached_depthcrafter_model is None:
        from depthcrafter.inference import DepthCrafterInference
        print("[DepthCrafter] Loading model...")
        
        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            
        # On 20GB+ GPUs (RTX 3090, 4090, 5090, A100), keep model entirely in VRAM
        # to avoid PCIe weight transfer latency between CPU and GPU
        offload_strategy = None if vram_gb >= 20.0 else "model"
        if offload_strategy is None:
            print(f"[DepthCrafter] High VRAM detected ({vram_gb:.1f} GB). Keeping model in VRAM (cpu_offload=None) for maximum throughput.")
        else:
            print(f"[DepthCrafter] VRAM detected ({vram_gb:.1f} GB). Using cpu_offload='model' for safe memory management.")
            
        _cached_depthcrafter_model = DepthCrafterInference(
            unet_path=unet_path,
            pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
            cpu_offload=offload_strategy
        )
        try:
            if hasattr(_cached_depthcrafter_model.pipe.vae, "enable_slicing"):
                _cached_depthcrafter_model.pipe.vae.enable_slicing()
            if hasattr(_cached_depthcrafter_model.pipe.vae, "enable_tiling"):
                _cached_depthcrafter_model.pipe.vae.enable_tiling()
        except Exception as e:
            print(f"[DepthCrafter] VAE tiling notice: {e}")
    return _cached_depthcrafter_model

def run_depthcrafter_depth(video_path: str, process_res: int, guidance_scale: float = 1.0, num_inference_steps: int = 5, window_size: int = 30, overlap: int = 10, max_frames: int = -1, attn_slicing: str = "Auto (Adapts to GPU VRAM)", progress=None) -> np.ndarray:
    model = get_depthcrafter_model()
    if hasattr(model, "configure_attention_slicing"):
        model.configure_attention_slicing(mode=attn_slicing, window_size=window_size, process_res=process_res)
    
    from depthcrafter.utils import read_video_frames
    print(f"[DepthCrafter] Reading frames from {video_path}")
    frames, _ = read_video_frames(
        video_path,
        max_frames,
        -1,  # target_fps
        process_res, # max_res
        "open", # dataset
    )
    
    if progress:
        progress(0.1, desc="DepthCrafter: Processing video chunks...")
        
    import sys
    if getattr(sys.modules.get('__main__', None), 'GLOBAL_CANCEL', False):
        sys.modules['__main__'].GLOBAL_CANCEL = False
        raise RuntimeError("Stopped by user.")
        
    num_frames = len(frames)
    stride = window_size - overlap if window_size > overlap else 1
    total_windows = max(1, (num_frames - overlap + stride - 1) // stride)
    print(f"[DepthCrafter] Running inference on {num_frames} frames ({total_windows} sliding windows of {window_size} frames)...", flush=True)
    
    current_window = [0]
    
    def on_step_end(pipeline, step: int, timestep: int, callback_kwargs: dict):
        if step == 0:
            current_window[0] += 1
        w_idx = min(current_window[0], total_windows)
        
        # Print with flush=True so terminal updates immediately
        print(f"[DepthCrafter] 🪟 Window {w_idx}/{total_windows} | Step {step + 1}/{num_inference_steps}", flush=True)
        
        if progress:
            try:
                frac = min(0.95, 0.1 + 0.85 * ((w_idx - 1) * num_inference_steps + step + 1) / (total_windows * num_inference_steps))
                progress(frac, desc=f"DepthCrafter: Window {w_idx}/{total_windows} (Step {step + 1}/{num_inference_steps})")
            except Exception:
                pass
        return callback_kwargs

    vram_gb = 0.0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    adaptive_decode_chunk = 8 if vram_gb >= 20.0 else (4 if vram_gb >= 12.0 else 2)

    with torch.inference_mode():
        res = model.pipe(
            frames,
            height=frames.shape[1],
            width=frames.shape[2],
            output_type="np",
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            window_size=window_size,
            overlap=overlap,
            decode_chunk_size=adaptive_decode_chunk,
            track_time=False,
            callback_on_step_end=on_step_end,
        ).frames[0]
        
    res = res.sum(-1) / res.shape[-1]
    return res

def unload_depthcrafter_model():
    global _cached_depthcrafter_model
    if _cached_depthcrafter_model is not None:
        try:
            _cached_depthcrafter_model.clear_cache()
        except Exception:
            pass
        _cached_depthcrafter_model = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
