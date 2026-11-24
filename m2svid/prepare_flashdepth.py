"""
FlashDepth Depth Estimation Module for StereoFaster
Streams video frames through Netflix Eyeline Studios' FlashDepth (Mamba State-Space Model).
Produces normalized disparity arrays compatible with Step 2 (M2SVid Warping & Inpainting).
"""

import os
import sys
import gc
import cv2
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple

SF_LOG = logging.getLogger("stereofaster")

# Ensure FlashDepth repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLASHDEPTH_ROOT = PROJECT_ROOT / "m2svid" / "third_party" / "FlashDepth"
if str(FLASHDEPTH_ROOT) not in sys.path:
    sys.path.insert(0, str(FLASHDEPTH_ROOT))

# Ensure bundled mamba package is in sys.path as fallback
MAMBA_DIR = FLASHDEPTH_ROOT / "mamba"
if str(MAMBA_DIR) not in sys.path:
    sys.path.insert(0, str(MAMBA_DIR))

_cached_fd_model = None
_cached_fd_checkpoint = None

FLASHDEPTH_MODELS = {
    "FlashDepth (Full) - 2K VFX Hybrid (Recommended)": {
        "config_path": FLASHDEPTH_ROOT / "configs" / "flashdepth" / "config.yaml",
        "ckpt_file": "flashdepth/iter_43002.pth",
        "local_ckpt": FLASHDEPTH_ROOT / "configs" / "flashdepth" / "iter_43002.pth",
        "teacher_file": "flashdepth-l/iter_10001.pth",
        "local_teacher": FLASHDEPTH_ROOT / "configs" / "flashdepth-l" / "iter_10001.pth",
        "code": "full"
    },
    "FlashDepth-L - ViT-Large Standalone (Best for 480p)": {
        "config_path": FLASHDEPTH_ROOT / "configs" / "flashdepth-l" / "config.yaml",
        "ckpt_file": "flashdepth-l/iter_10001.pth",
        "local_ckpt": FLASHDEPTH_ROOT / "configs" / "flashdepth-l" / "iter_10001.pth",
        "teacher_file": None,
        "local_teacher": None,
        "code": "vitl"
    },
    "FlashDepth-S - ViT-Small (Fastest / Low VRAM)": {
        "config_path": FLASHDEPTH_ROOT / "configs" / "flashdepth-s" / "config.yaml",
        "ckpt_file": "flashdepth-s/iter_14001.pth",
        "local_ckpt": FLASHDEPTH_ROOT / "configs" / "flashdepth-s" / "iter_14001.pth",
        "teacher_file": None,
        "local_teacher": None,
        "code": "vits"
    }
}


def ensure_checkpoint_downloaded(model_key: str) -> Tuple[Path, Optional[Path]]:
    """Download FlashDepth weights from Hugging Face if not already present."""
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    from huggingface_hub import hf_hub_download
    
    cfg = FLASHDEPTH_MODELS.get(model_key, FLASHDEPTH_MODELS["FlashDepth (Full) - 2K VFX Hybrid (Recommended)"])
    ckpt_path = cfg["local_ckpt"]
    teacher_path = cfg["local_teacher"]
    
    # 1. Download main checkpoint
    if not ckpt_path.exists():
        SF_LOG.info(f"[FlashDepth] Downloading {cfg['ckpt_file']} from Eyeline-Labs/FlashDepth...")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id="Eyeline-Labs/FlashDepth",
            filename=cfg["ckpt_file"],
            local_dir=str(FLASHDEPTH_ROOT / "configs"),
            local_dir_use_symlinks=False
        )
        if not ckpt_path.exists() and os.path.exists(downloaded):
            ckpt_path = Path(downloaded)

    # 2. Download teacher model if hybrid full model is selected
    if teacher_path is not None and not teacher_path.exists():
        SF_LOG.info(f"[FlashDepth] Downloading teacher model {cfg['teacher_file']}...")
        teacher_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_t = hf_hub_download(
            repo_id="Eyeline-Labs/FlashDepth",
            filename=cfg["teacher_file"],
            local_dir=str(FLASHDEPTH_ROOT / "configs"),
            local_dir_use_symlinks=False
        )
        if not teacher_path.exists() and os.path.exists(downloaded_t):
            teacher_path = Path(downloaded_t)

    return ckpt_path, teacher_path


def get_flashdepth_model(model_name: str = "FlashDepth (Full) - 2K VFX Hybrid (Recommended)", device: str = "cuda"):
    """Load and cache FlashDepth model on GPU."""
    global _cached_fd_model, _cached_fd_checkpoint
    
    cfg_meta = FLASHDEPTH_MODELS.get(model_name, FLASHDEPTH_MODELS["FlashDepth (Full) - 2K VFX Hybrid (Recommended)"])
    ckpt_code = cfg_meta["code"]
    
    if _cached_fd_model is not None and _cached_fd_checkpoint == ckpt_code:
        return _cached_fd_model
        
    if _cached_fd_model is not None:
        unload_flashdepth_model()
        
    import yaml
    from omegaconf import OmegaConf
    from flashdepth.model import FlashDepth
    
    ckpt_path, teacher_path = ensure_checkpoint_downloaded(model_name)
    config_yaml_path = cfg_meta["config_path"]
    
    if not config_yaml_path.exists():
        raise FileNotFoundError(f"FlashDepth config not found at: {config_yaml_path}")
        
    cfg = OmegaConf.load(str(config_yaml_path))
    cfg.inference = True
    cfg.load = str(ckpt_path)
    
    if teacher_path is not None and cfg.get("hybrid_configs") and cfg.hybrid_configs.get("use_hybrid"):
        cfg.hybrid_configs.teacher_model_path = str(teacher_path)
        
    SF_LOG.info(f"[FlashDepth] Building {ckpt_code} model architecture...")
    model = FlashDepth(
        batch_size=1,
        hybrid_configs=cfg.get("hybrid_configs"),
        training=False,
        **cfg.model
    )
    
    SF_LOG.info(f"[FlashDepth] Loading weights from {ckpt_path}...")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    
    # Load state dict with non-strict fallback for hybrid buffers
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        SF_LOG.debug(f"[FlashDepth] Missing keys: {len(missing)}")
    if unexpected:
        SF_LOG.debug(f"[FlashDepth] Unexpected keys: {len(unexpected)}")
        
    # If teacher model is needed for hybrid inference, load it
    if cfg.get("hybrid_configs") and cfg.hybrid_configs.get("use_hybrid") and teacher_path and teacher_path.exists():
        SF_LOG.info(f"[FlashDepth] Loading teacher model weights from {teacher_path}...")
        teacher_ckpt = torch.load(str(teacher_path), map_location="cpu")
        teacher_state = teacher_ckpt["model"] if "model" in teacher_ckpt else teacher_ckpt
        model.teacher_model.load_state_dict(teacher_state, strict=False)
        
    model = model.to(device).eval()
    
    _cached_fd_model = (model, cfg)
    _cached_fd_checkpoint = ckpt_code
    SF_LOG.info(f"[FlashDepth] {ckpt_code.upper()} model ready for real-time streaming!")
    return _cached_fd_model


def unload_flashdepth_model():
    """Unload FlashDepth from GPU memory."""
    global _cached_fd_model, _cached_fd_checkpoint
    if _cached_fd_model is not None:
        SF_LOG.info("[FlashDepth] Unloading model from GPU VRAM...")
        del _cached_fd_model
        _cached_fd_model = None
        _cached_fd_checkpoint = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_flashdepth(
    video_path: str,
    model_name: str = "FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
    max_res: int = 1280,
    target_fps: float = -1.0,
    max_len: int = -1,
    global_norm: bool = True,
    invert_depth: bool = False,
    progress=None
) -> Tuple[np.ndarray, float]:
    """
    Runs FlashDepth on input video stream and returns normalized disparity (T, H, W) in [0, 1].
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = get_flashdepth_model(model_name=model_name, device=device)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
        
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_fps = orig_fps if target_fps <= 0 else target_fps
    
    # Calculate target spatial resolution (must align with patch_size=14)
    patch_size = cfg.model.get("patch_size", 14)
    scale = 1.0
    if max_res > 0 and max(orig_h, orig_w) > max_res:
        scale = max_res / float(max(orig_h, orig_w))
        
    proc_h = int(round(orig_h * scale / patch_size) * patch_size)
    proc_w = int(round(orig_w * scale / patch_size) * patch_size)
    
    proc_h = max(patch_size, proc_h)
    proc_w = max(patch_size, proc_w)
    
    SF_LOG.info(f"[FlashDepth] Streaming frames: {orig_w}x{orig_h} -> {proc_w}x{proc_h} (patch {patch_size})")
    
    frames_bgr = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
        if max_len > 0 and len(frames_bgr) >= max_len:
            break
    cap.release()
    
    if len(frames_bgr) == 0:
        raise ValueError(f"No valid frames read from {video_path}")
        
    n_frames = len(frames_bgr)
    depth_maps = []
    
    # Initialize / reset Mamba temporal recurrent state for new sequence
    if hasattr(model, "mamba") and hasattr(model.mamba, "start_new_sequence"):
        model.mamba.start_new_sequence()
        
    use_mamba = cfg.model.get("use_mamba", True)
    
    mean_tensor = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_tensor = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16):
            for idx, frame_bgr in enumerate(frames_bgr):
                if progress is not None:
                    progress((idx + 1) / n_frames, desc=f"FlashDepth streaming frame {idx+1}/{n_frames} ({idx / max(1e-5, time.time() - t0):.1f} FPS)...")
                    
                # Resize and convert BGR -> RGB -> Tensor [1, C, H, W] with ImageNet norm
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frame_rgb.shape[0] != proc_h or frame_rgb.shape[1] != proc_w:
                    frame_rgb = cv2.resize(frame_rgb, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
                    
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                frame_tensor = (frame_tensor.to(device) - mean_tensor) / std_tensor
                
                B, C, H, W = frame_tensor.shape
                patch_h = H // patch_size
                patch_w = W // patch_size
                
                # Single-frame streaming forward pass
                dpt_features = model.get_dpt_features(frame_tensor, input_shape=(B, C, H, W))
                pred = model.final_head(dpt_features.float(), patch_h, patch_w)
                pred = torch.clip(pred, min=0)
                
                # Resize back to native aspect if resized
                if pred.shape[-2:] != (orig_h, orig_w):
                    pred = F.interpolate(pred.unsqueeze(1), size=(orig_h, orig_w), mode="bilinear", align_corners=False).squeeze(1)
                    
                pred_np = pred.squeeze().cpu().float().numpy()
                depth_maps.append(pred_np)
                
    elapsed = max(1e-5, time.time() - t0)
    actual_fps = n_frames / elapsed
    SF_LOG.info(f"[FlashDepth] Processed {n_frames} frames in {elapsed:.2f}s ({actual_fps:.1f} FPS)")
    
    depth_array = np.stack(depth_maps, axis=0) # (T, H, W)
    
    # Normalize depth to affine-invariant disparity [0, 1]
    if global_norm:
        q_low, q_high = np.percentile(depth_array, [1.0, 99.0])
        if q_high > q_low:
            depth_array = (depth_array - q_low) / (q_high - q_low)
        depth_array = np.clip(depth_array, 0.0, 1.0)
    else:
        for i in range(len(depth_array)):
            d = depth_array[i]
            q_low, q_high = np.percentile(d, [1.0, 99.0])
            if q_high > q_low:
                depth_array[i] = np.clip((d - q_low) / (q_high - q_low), 0.0, 1.0)
                
    if invert_depth:
        depth_array = 1.0 - depth_array
        
    return depth_array, out_fps

