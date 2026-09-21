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
_cached_cpu_offload = None

def get_depthcrafter_model(unet_path="tencent/DepthCrafter", cpu_offload="Auto (Adapts to GPU VRAM)"):
    global _cached_depthcrafter_model, _cached_cpu_offload
    
    vram_gb = 0.0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        
    cpu_offload_clean = str(cpu_offload).lower()
    if "auto" in cpu_offload_clean:
        # At native 1080p+, UNet activations peak at 15-20GB.
        # Keeping VAE + text encoder in VRAM (~5GB) causes OOM on 24GB & 32GB GPUs (RTX 3090/4090/5090).
        # Use "model" offload for <45GB GPUs to free ~5GB VRAM and allow 80-110 frame windows safely.
        target_offload = None if vram_gb >= 45.0 else "model"
    elif "sequential" in cpu_offload_clean:
        target_offload = "sequential"
    elif "model" in cpu_offload_clean:
        target_offload = "model"
    else: # "none" or other
        target_offload = None

    if _cached_depthcrafter_model is not None and _cached_cpu_offload == target_offload:
        return _cached_depthcrafter_model
        
    if _cached_depthcrafter_model is not None:
        unload_depthcrafter_model()

    from depthcrafter.inference import DepthCrafterInference
    print(f"[DepthCrafter] Loading model with cpu_offload={target_offload} (VRAM: {vram_gb:.1f} GB)...")
    
    _cached_depthcrafter_model = DepthCrafterInference(
        unet_path=unet_path,
        pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
        cpu_offload=target_offload
    )
    _cached_cpu_offload = target_offload
    
    try:
        if hasattr(_cached_depthcrafter_model.pipe.vae, "enable_slicing"):
            _cached_depthcrafter_model.pipe.vae.enable_slicing()
        if hasattr(_cached_depthcrafter_model.pipe.vae, "enable_tiling"):
            _cached_depthcrafter_model.pipe.vae.enable_tiling()
    except Exception as e:
        print(f"[DepthCrafter] VAE tiling notice: {e}")
    return _cached_depthcrafter_model

def run_depthcrafter_depth(video_path: str, process_res: int, guidance_scale: float = 1.0, num_inference_steps: int = 5, window_size: int = 30, overlap: int = 10, max_frames: int = -1, attn_slicing: str = "Auto (Adapts to GPU VRAM)", cpu_offload: str = "Auto (Adapts to GPU VRAM)", progress=None) -> np.ndarray:
    model = get_depthcrafter_model(cpu_offload=cpu_offload)
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
    
    # At high native resolution (>=1536p), VAE decoding 8 frames consumes >10GB VRAM.
    # Scale decode chunk size safely to prevent VAE OOM spikes.
    if vram_gb >= 45.0:
        adaptive_decode_chunk = 8
    elif vram_gb >= 20.0:
        adaptive_decode_chunk = 4 if process_res >= 1536 else 8
    elif vram_gb >= 12.0:
        adaptive_decode_chunk = 2 if process_res >= 1536 else 4
    else:
        adaptive_decode_chunk = 1 if process_res >= 1536 else 2

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
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
