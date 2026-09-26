import os
import sys
import gc
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple

# Ensure Video-Depth-Anything is in sys.path
VDA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "Video-Depth-Anything"))
if VDA_ROOT not in sys.path:
    sys.path.insert(0, VDA_ROOT)

_cached_vda_model = None
_cached_vda_encoder = None
_cached_vda_metric = None

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'repo': 'depth-anything/Video-Depth-Anything-Small'},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'repo': 'depth-anything/Video-Depth-Anything-Base'},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'repo': 'depth-anything/Video-Depth-Anything-Large'},
}

def get_vda_model(encoder: str = 'vitl', metric: bool = False, device: str = 'cuda'):
    """Load and cache Video Depth Anything model on GPU."""
    global _cached_vda_model, _cached_vda_encoder, _cached_vda_metric
    
    clean_enc = encoder.lower()
    if 'large' in clean_enc or 'vitl' in clean_enc:
        clean_enc = 'vitl'
    elif 'base' in clean_enc or 'vitb' in clean_enc:
        clean_enc = 'vitb'
    elif 'small' in clean_enc or 'vits' in clean_enc:
        clean_enc = 'vits'
    else:
        clean_enc = 'vitl'
        
    if _cached_vda_model is not None and _cached_vda_encoder == clean_enc and _cached_vda_metric == metric:
        return _cached_vda_model
        
    if _cached_vda_model is not None:
        unload_vda_model()
        
    from video_depth_anything.video_depth import VideoDepthAnything
    from huggingface_hub import hf_hub_download
    
    cfg = MODEL_CONFIGS[clean_enc]
    repo_id = cfg['repo']
    ckpt_file = f"metric_video_depth_anything_{clean_enc}.pth" if metric else f"video_depth_anything_{clean_enc}.pth"
    
    print(f"[VDA] Fetching model weights for {repo_id}/{ckpt_file}...", flush=True)
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_file)
    print(f"[VDA] Loading {clean_enc.upper()} model into VRAM...", flush=True)
    
    model = VideoDepthAnything(
        encoder=clean_enc,
        features=cfg['features'],
        out_channels=cfg['out_channels'],
        metric=metric
    )
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)
    model = model.to(device).eval()
    
    _cached_vda_model = model
    _cached_vda_encoder = clean_enc
    _cached_vda_metric = metric
    print(f"[VDA] {clean_enc.upper()} model successfully loaded!", flush=True)
    return model


def unload_vda_model():
    """Unload Video Depth Anything model to free GPU VRAM."""
    global _cached_vda_model, _cached_vda_encoder, _cached_vda_metric
    if _cached_vda_model is not None:
        print("[VDA] Unloading Video Depth Anything model from VRAM...", flush=True)
        del _cached_vda_model
        _cached_vda_model = None
        _cached_vda_encoder = None
        _cached_vda_metric = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_vda_depth(
    video_path: str,
    encoder: str = 'vitl',
    input_size: int = 518,
    max_res: int = 1280,
    mode: str = 'sliding_window',
    target_fps: float = -1.0,
    max_len: int = -1,
    fp32: bool = False,
    global_norm: bool = True,
    progress=None
) -> Tuple[np.ndarray, float]:
    """
    Run Video Depth Anything inference on an input video.
    Returns:
        (depth_array, fps) where depth_array is (T, H, W) in float32 [0.0, 1.0]
        normalized disparity (1.0 = near, 0.0 = far) compatible with M2SVid warping.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_vda_model(encoder=encoder, metric=False, device=device)
    
    # Read video frames
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
        
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fps = orig_fps if target_fps <= 0 else target_fps
    stride = max(round(orig_fps / fps), 1)
    
    # Check resolution scaling
    scale_w, scale_h = orig_w, orig_h
    if max_res > 0 and max(orig_h, orig_w) > max_res:
        scale = max_res / max(orig_h, orig_w)
        scale_w = int(round(orig_w * scale))
        scale_h = int(round(orig_h * scale))
        # Ensure even dimensions for tensor operations
        scale_w = scale_w if scale_w % 2 == 0 else scale_w + 1
        scale_h = scale_h if scale_h % 2 == 0 else scale_h + 1

    print(f"[VDA] Reading frames from {Path(video_path).name} (orig: {orig_w}x{orig_h} @ {orig_fps:.2f}fps, target: {scale_w}x{scale_h})...", flush=True)
    if progress:
        try: progress(0.05, desc="VDA: Reading video frames...")
        except Exception: pass

    frames = []
    f_idx = 0
    while cap.isOpened():
        import sys
        if getattr(sys.modules.get('__main__', None), 'GLOBAL_CANCEL', False):
            sys.modules['__main__'].GLOBAL_CANCEL = False
            cap.release()
            raise RuntimeError("Stopped by user.")
            
        ret, frame = cap.read()
        if not ret or (max_len > 0 and len(frames) >= max_len):
            break
            
        if f_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if scale_w != orig_w or scale_h != orig_h:
                frame_rgb = cv2.resize(frame_rgb, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
            frames.append(frame_rgb)
        f_idx += 1
    cap.release()
    
    if not frames:
        raise RuntimeError("No frames could be extracted from video.")
        
    num_frames = len(frames)
    print(f"[VDA] Loaded {num_frames} frames. Running {encoder.upper()} inference (mode={mode}, input_size={input_size})...", flush=True)
    
    if progress:
        try: progress(0.15, desc=f"VDA: Running {encoder.upper()} depth estimation on {num_frames} frames...")
        except Exception: pass

    raw_depths = None
    if 'stream' in mode.lower():
        from video_depth_anything.video_depth_stream import VideoDepthAnything as VDAStream
        clean_enc = 'vitl' if 'large' in encoder.lower() or 'vitl' in encoder.lower() else ('vitb' if 'base' in encoder.lower() or 'vitb' in encoder.lower() else 'vits')
        stream_kwargs = {k: v for k, v in MODEL_CONFIGS[clean_enc].items() if k != 'repo'}
        stream_model = VDAStream(**stream_kwargs).to(device).eval()
        # Transfer weights from cached model
        stream_model.load_state_dict(model.state_dict(), strict=False)
        
        depths_list = []
        for i, f in enumerate(frames):
            import sys
            if getattr(sys.modules.get('__main__', None), 'GLOBAL_CANCEL', False):
                sys.modules['__main__'].GLOBAL_CANCEL = False
                raise RuntimeError("Stopped by user.")
                
            d = stream_model.infer_video_depth_one(f, input_size=input_size, device=device, fp32=fp32)
            depths_list.append(d)
            if progress and (i % 10 == 0 or i == num_frames - 1):
                try: progress(0.15 + 0.75 * (i + 1) / num_frames, desc=f"VDA Streaming: frame {i+1}/{num_frames}")
                except Exception: pass
        raw_depths = np.stack(depths_list, axis=0)
        del stream_model
    else:
        # Standard sliding-window mode (32-frame chunks with 10 overlap + keyframe alignment)
        frames_arr = np.array(frames)
        
        # We run the infer_video_depth logic
        raw_depths, _ = model.infer_video_depth(
            frames_arr,
            fps,
            input_size=input_size,
            device=device,
            fp32=fp32
        )

    # In VDA, values represent camera depth (smaller = near, larger = far).
    # M2SVid expects disparity/inverse-depth where larger = near (1.0) and smaller = far (0.0).
    disp = -raw_depths.astype(np.float32)
    
    if global_norm:
        # Global video normalization preserves global scale consistency across all frames
        d_min, d_max = float(disp.min()), float(disp.max())
        denom = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1.0
        norm_depth = (disp - d_min) / denom
    else:
        # Per-frame normalization
        norm_depth = np.empty_like(disp, dtype=np.float32)
        for i in range(num_frames):
            frame_d = disp[i]
            d_min, d_max = float(frame_d.min()), float(frame_d.max())
            denom = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1.0
            norm_depth[i] = (frame_d - d_min) / denom
            
    norm_depth = np.clip(norm_depth, 0.0, 1.0)
    
    # Ensure final depth resolution matches original video dimensions
    if norm_depth.shape[1:] != (orig_h, orig_w):
        print(f"[VDA] Upscaling depth from {norm_depth.shape[2]}x{norm_depth.shape[1]} to native {orig_w}x{orig_h}...", flush=True)
        norm_depth = np.stack([cv2.resize(d, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC) for d in norm_depth])
        
    print(f"[VDA] Depth estimation complete! Final depth shape: {norm_depth.shape}", flush=True)
    if progress:
        try: progress(0.95, desc="VDA: Depth estimation complete. Saving files...")
        except Exception: pass
        
    return norm_depth, fps
