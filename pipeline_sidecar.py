"""
StereoFaster Pipeline Sidecar & Benchmark Engine

Provides sidecar-driven configuration, library scanning, multi-sample benchmarking,
prioritized batch processing, and audio passthrough for DepthCrafter, Video Depth Anything (VDA),
and M2SVid stereography.
"""

import os
import sys

# Ensure UTF-8 stdout/stderr on Windows consoles to prevent charmap UnicodeEncodeErrors with emojis
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import gc
import json
import glob
import time
import shutil
import logging
import threading
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import torch
import cv2

logger = logging.getLogger("PipelineSidecar")
VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")

# Hardware Profiles with VRAM-optimized settings
HARDWARE_PROFILES = {
    "RTX Pro 6000 / A100 (48GB - 96GB VRAM)": {
        "dc_window": 110,
        "dc_overlap": 20,
        "dc_steps": 5,
        "dc_max_res": 1536,
        "dc_offload": "none",
        "dc_attn_slicing": "Disabled (Fastest / 12GB+ GPUs)",
        "icd_max_res": 1920,
        "icd_window": 81,
        "icd_overlap": 32,
        "icd_steps": 5,
        "icd_cfg_scale": 5.0,
        "icd_tiled": False,
        "fd_model": "FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
        "fd_max_res": 1540,
        "vda_model": "Large (vitl - Recommended)",
        "vda_input_size": 644,
        "vda_max_res": 1536,
        "m2svid_chunk": 14,
        "m2svid_batch": 3,
        "warp_batch": 32,
        "m2svid_res": "1024x576 (Optimal 12GB)",
        "description": "Enterprise workstation profile (48GB-96GB). Zero CPU offload, maximum window sizes, and parallel batch=3."
    },
    "RTX 5090 (32GB VRAM GDDR7)": {
        "dc_window": 90,
        "dc_overlap": 18,
        "dc_steps": 5,
        "dc_max_res": 1536,
        "dc_offload": "none",
        "dc_attn_slicing": "Disabled (Fastest / 12GB+ GPUs)",
        "icd_max_res": 1920,
        "icd_window": 81,
        "icd_overlap": 32,
        "icd_steps": 5,
        "icd_cfg_scale": 5.0,
        "icd_tiled": False,
        "fd_model": "FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
        "fd_max_res": 1540,
        "vda_model": "Large (vitl - Recommended)",
        "vda_input_size": 644,
        "vda_max_res": 1536,
        "m2svid_chunk": 14,
        "m2svid_batch": 2,
        "warp_batch": 32,
        "m2svid_res": "1024x576 (Optimal 12GB)",
        "description": "Blackwell 32GB GDDR7 profile. Zero offload, chunk=14, parallel chunk batch=2 (~20GB VRAM)."
    },
    "RTX 4090 / 3090 (24GB VRAM)": {
        "dc_window": 75,
        "dc_overlap": 15,
        "dc_steps": 5,
        "dc_max_res": 1536,
        "dc_offload": "none",
        "dc_attn_slicing": "Disabled (Fastest / 12GB+ GPUs)",
        "icd_max_res": 1024,
        "icd_window": 81,
        "icd_overlap": 32,
        "icd_steps": 5,
        "icd_cfg_scale": 5.0,
        "icd_tiled": False,
        "fd_model": "FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
        "fd_max_res": 1288,
        "vda_model": "Large (vitl - Recommended)",
        "vda_input_size": 644,
        "vda_max_res": 1280,
        "m2svid_chunk": 14,
        "m2svid_batch": 1,
        "warp_batch": 24,
        "m2svid_res": "1024x576 (Optimal 12GB)",
        "description": "Standard 24GB profile. Zero offload, chunk=14, sequential chunk execution."
    },
    "RTX 3060 / 4060 (12GB - 16GB VRAM)": {
        "dc_window": 30,
        "dc_overlap": 10,
        "dc_steps": 5,
        "dc_max_res": 1024,
        "dc_offload": "model",
        "dc_attn_slicing": "Auto (Adapts to GPU VRAM)",
        "icd_max_res": 720,
        "icd_window": 81,
        "icd_overlap": 32,
        "icd_steps": 5,
        "icd_cfg_scale": 5.0,
        "icd_tiled": False,
        "fd_model": "FlashDepth (Full) - 2K VFX Hybrid (Recommended)",
        "fd_max_res": 1288,
        "vda_model": "Large (vitl - Recommended)",
        "vda_input_size": 644,
        "vda_max_res": 1024,
        "m2svid_chunk": 5,
        "m2svid_batch": 1,
        "warp_batch": 8,
        "m2svid_res": "Native",
        "description": "12GB memory-safe profile. chunk_size=5 keeps VRAM under 9GB to prevent PCIe paging (10x faster)."
    }
}


def detect_gpu_hardware() -> str:
    """Detects primary CUDA GPU and returns best matching profile key."""
    if not torch.cuda.is_available():
        return "RTX 3060 / 4060 (12GB - 16GB VRAM)"
    try:
        gpu_name = torch.cuda.get_device_name(0).lower()
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if "pro 6000" in gpu_name or total_gb >= 45:
            return "RTX Pro 6000 / A100 (48GB - 96GB VRAM)"
        elif "5090" in gpu_name or total_gb >= 28:
            return "RTX 5090 (32GB VRAM GDDR7)"
        elif "4090" in gpu_name or "3090" in gpu_name or total_gb >= 20:
            return "RTX 4090 / 3090 (24GB VRAM)"
        else:
            return "RTX 3060 / 4060 (12GB - 16GB VRAM)"
    except Exception:
        return "RTX 3060 / 4060 (12GB - 16GB VRAM)"


def get_default_sidecar_config(profile_name: Optional[str] = None) -> Dict[str, Any]:
    """Generates a comprehensive, fully populated sidecar config dictionary."""
    if not profile_name or profile_name not in HARDWARE_PROFILES:
        profile_name = detect_gpu_hardware()
    prof = HARDWARE_PROFILES[profile_name]

    return {
        "version": "1.1",
        "hardware_profile": profile_name,
        "cloud_vast_rate_per_hour": 0.50,
        "stages_enabled": {
            "depthcrafter": True,
            "icdepth": False,
            "flashdepth": False,
            "vda": False,
            "m2svid": True
        },
        "depth_priority": "DepthCrafter", # 'DepthCrafter', 'ICDepth', 'FlashDepth', or 'VDA'
        "depthcrafter_settings": {
            "guidance_scale": 1.0,
            "inference_steps": prof.get("dc_steps", 5),
            "window_size": prof.get("dc_window", 30),
            "overlap": prof.get("dc_overlap", 10),
            "max_res": prof.get("dc_max_res", 1024),
            "attn_slicing": prof.get("dc_attn_slicing", "Auto (Adapts to GPU VRAM)"),
            "cpu_offload": prof.get("dc_offload", "model")
        },
        "icdepth_settings": {
            "max_res": prof.get("icd_max_res", 720),
            "window_size": prof.get("icd_window", 81),
            "overlap": prof.get("icd_overlap", 32),
            "inference_steps": prof.get("icd_steps", 5),
            "cfg_scale": prof.get("icd_cfg_scale", 5.0),
            "tiled": prof.get("icd_tiled", False)
        },
        "flashdepth_settings": {
            "model": prof.get("fd_model", "FlashDepth (Full) - 2K VFX Hybrid (Recommended)"),
            "max_res": prof.get("fd_max_res", 1288),
            "target_fps": -1.0,
            "max_len": -1,
            "global_norm": True,
            "invert_depth": False
        },
        "vda_settings": {
            "model": prof.get("vda_model", "Large (vitl - Recommended)"),
            "mode": "sliding_window",
            "input_size": prof.get("vda_input_size", 644),
            "max_res": prof.get("vda_max_res", 1024),
            "target_fps": -1,
            "max_len": -1,
            "fp32": False,
            "global_norm": True,
            "invert_depth": False
        },
        "m2svid_settings": {
            "processing_res": prof.get("m2svid_res", "Native"),
            "disparity_perc": 20.0,
            "convergence_point": 0.50,
            "closing_kernel": 11,
            "mask_antialias": False,
            "warping_batch_size": prof.get("warp_batch", 8),
            "gen_chunk_size": prof.get("m2svid_chunk", 5),
            "gen_batch_size": prof.get("m2svid_batch", 1)
        },
        "auto_crop": {
            "enabled": True,
            "force_preset": False,
            "manual_preset": "16:9 (None)"
        },
        "audio_passthrough": {
            "enabled": True
        }
    }


def load_sidecar_file(sidecar_path: Path) -> Dict[str, Any]:
    """Loads sidecar JSON config from disk, falling back to profile defaults if missing."""
    if sidecar_path and sidecar_path.is_file():
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded pipeline sidecar from {sidecar_path}")
            return data
        except Exception as e:
            logger.warning(f"Failed to parse sidecar file {sidecar_path}: {e}. Generating fresh defaults.")
    return get_default_sidecar_config()


def save_sidecar_file(sidecar_data: Dict[str, Any], sidecar_path: Path) -> str:
    """Saves sidecar dictionary to JSON file on disk."""
    try:
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_data, f, indent=2)
        logger.info(f"Saved pipeline sidecar to {sidecar_path}")
        return f"✅ Pipeline sidecar saved successfully to `{sidecar_path.name}`."
    except Exception as e:
        logger.error(f"Error saving sidecar {sidecar_path}: {e}")
        return f"❌ Failed to save pipeline sidecar: {e}"


def remux_audio_if_present(src_video: str, target_video: str) -> bool:
    """
    Copies audio stream from src_video into target_video if audio exists.
    Silently ignores if no audio stream is present.
    """
    if not os.path.isfile(src_video) or not os.path.isfile(target_video):
        return False
    try:
        probe_cmd = [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0", src_video
        ]
        res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "audio" in res.stdout.strip().lower():
            tmp_out = target_video + ".tmp_audio.mp4"
            remux_cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", target_video,
                "-i", src_video,
                "-c:v", "copy",
                "-c:a", "copy",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                tmp_out
            ]
            remux_res = subprocess.run(remux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if remux_res.returncode == 0 and os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 1024:
                shutil.move(tmp_out, target_video)
                logger.info(f"Remuxed audio from {os.path.basename(src_video)} -> {os.path.basename(target_video)}")
                return True
            elif os.path.exists(tmp_out):
                os.remove(tmp_out)
    except Exception as e:
        logger.warning(f"Audio remux failed on {target_video}: {e}")
    return False


def get_clip_metadata(video_path: str) -> Tuple[int, float, int, int, float]:
    """Returns (frames, fps, width, height, duration_sec) using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    cap.release()
    dur = total_frames / max(1.0, fps)
    return total_frames, fps, w, h, dur


def scan_source_library(source_dir: Path, depth_dir: Path, final_dir: Path) -> Tuple[list, str, dict]:
    """
    Scans source_dir, checks existing depth and stereo outputs.
    Returns: (table_rows, summary_markdown, stats_dict)
    """
    if not source_dir.exists():
        return [], "⚠️ Source directory does not exist.", {}

    all_vids = []
    for ext in VIDEO_EXTENSIONS:
        all_vids.extend(glob.glob(str(source_dir / ext)))
        all_vids.extend(glob.glob(str(source_dir / "**" / ext), recursive=True))
    all_vids = sorted(list(set(all_vids)))

    if not all_vids:
        return [], "⚠️ No video files found in `source_videos/`.", {}

    rows = []
    short_cnt, med_cnt, long_cnt = 0, 0, 0
    total_frames = 0
    total_sec = 0.0
    clip_records = []

    subdirs = [s for s in depth_dir.glob("*/") if s.is_dir()]

    for vp in all_vids:
        p = Path(vp)
        frames, fps, w, h, dur = get_clip_metadata(vp)
        total_frames += frames
        total_sec += dur

        if dur <= 6.0:
            category = "🟢 Short (<=6s)"
            short_cnt += 1
        elif dur <= 18.0:
            category = "🟡 Medium (6-18s)"
            med_cnt += 1
        else:
            category = f"🔴 Long ({dur:.1f}s)"
            long_cnt += 1

        stem = p.stem
        # Check existing depth outputs
        dc_done = (
            (depth_dir / f"{stem}_DC_depth.npz").exists() or 
            (depth_dir / f"{stem}_DC_depth.mp4").exists() or
            any((sub / f"{stem}_DC_depth.npz").exists() for sub in subdirs)
        )
        icd_done = (
            (depth_dir / f"{stem}_ICD_depth.npz").exists() or
            (depth_dir / f"{stem}_ICD_depth.mp4").exists() or
            any((sub / f"{stem}_ICD_depth.npz").exists() for sub in subdirs)
        )
        fd_done = (
            (depth_dir / f"{stem}_FDF_depth.npz").exists() or
            (depth_dir / f"{stem}_FDL_depth.npz").exists() or
            (depth_dir / f"{stem}_FDS_depth.npz").exists() or
            (depth_dir / f"{stem}_FD_depth.npz").exists() or
            any((sub / f"{stem}_FDF_depth.npz").exists() for sub in subdirs) or
            any((sub / f"{stem}_FDL_depth.npz").exists() for sub in subdirs)
        )
        vda_done = (
            (depth_dir / f"{stem}_VDAL_depth.npz").exists() or
            (depth_dir / f"{stem}_VDA_depth.npz").exists() or
            (depth_dir / f"{stem}_VDAB_depth.npz").exists() or
            (depth_dir / f"{stem}_depth.npz").exists() or
            any((sub / f"{stem}_VDAL_depth.npz").exists() for sub in subdirs) or
            any((sub / f"{stem}_VDA_depth.npz").exists() for sub in subdirs)
        )
        
        # Check existing stereo outputs
        stereo_folder = final_dir / f"{stem}_stereo"
        m2s_done = False
        if stereo_folder.is_dir():
            stereo_vids = list(stereo_folder.glob("*.mp4"))
            if any("_sbs" in v.name or "_anaglyph" in v.name or "_right" in v.name for v in stereo_vids):
                m2s_done = True

        status_tokens = [
            "DepthCrafter: ✅ OK" if dc_done else "DepthCrafter: --",
            "ICDepth: ✅ OK" if icd_done else "ICDepth: --",
        ]
        if fd_done:
            status_tokens.append("FlashDepth: ✅ OK")
        if vda_done:
            status_tokens.append("VDA: ✅ OK")
        status_tokens.append("M2SVid 3D: ✅ OK" if m2s_done else "M2SVid 3D: --")

        rows.append([
            p.name,
            f"{dur:.1f}s ({frames}f)",
            f"{w}x{h} @ {fps:.1f}fps",
            category,
            " | ".join(status_tokens)
        ])

        clip_records.append({
            "path": vp,
            "stem": stem,
            "frames": frames,
            "fps": fps,
            "width": w,
            "height": h,
            "duration": dur,
            "dc_done": dc_done,
            "icd_done": icd_done,
            "fd_done": fd_done,
            "vda_done": vda_done,
            "m2s_done": m2s_done
        })

    summary_md = (
        f"### 📊 Library Scan: `{len(all_vids)}` Clips in `source_videos/`\n"
        f"- **Total Volume**: `{total_frames:,}` frames | `{total_sec/60:.2f}` minutes of video\n"
        f"- **Duration Split**: `{short_cnt}` Short (<=6s) | `{med_cnt}` Medium (6-18s) | `{long_cnt}` Long (>18s)\n"
    )

    stats = {
        "clips": clip_records,
        "count": len(all_vids),
        "total_frames": total_frames,
        "total_seconds": total_sec,
        "short_count": short_cnt,
        "med_count": med_cnt,
        "long_count": long_cnt
    }
    return rows, summary_md, stats


def find_matching_depth_npz(stem: str, depth_dir: Path, priority: str = "DepthCrafter") -> Optional[str]:
    """Finds best matching depth .npz based on priority, checking root and subdirectories."""
    subdirs = [s for s in depth_dir.glob("*/") if s.is_dir()]
    dc_files = [depth_dir / f"{stem}_hybrid_depth.npz", depth_dir / f"{stem}_DC_depth.npz"]
    icd_files = [depth_dir / f"{stem}_ICD_depth.npz"]
    fd_files = [
        depth_dir / f"{stem}_FDF_depth.npz",
        depth_dir / f"{stem}_FDL_depth.npz",
        depth_dir / f"{stem}_FDS_depth.npz",
        depth_dir / f"{stem}_FD_depth.npz",
    ]
    vda_files = [
        depth_dir / f"{stem}_VDAL_depth.npz",
        depth_dir / f"{stem}_VDA_depth.npz",
        depth_dir / f"{stem}_VDAB_depth.npz",
        depth_dir / f"{stem}_depth.npz",
    ]

    if priority == "ICDepth":
        ordered_groups = [icd_files, dc_files, fd_files, vda_files]
    elif priority == "FlashDepth":
        ordered_groups = [fd_files, dc_files, icd_files, vda_files]
    elif priority == "VDA":
        ordered_groups = [vda_files, dc_files, icd_files, fd_files]
    else:  # Default DepthCrafter first
        ordered_groups = [dc_files, icd_files, fd_files, vda_files]

    candidates = [p for grp in ordered_groups for p in grp]
    for sub in subdirs:
        for p in [item for grp in ordered_groups for item in grp]:
            candidates.append(sub / p.name)

    for c in candidates:
        if c.exists() and c.stat().st_size > 1024:
            return str(c)
    matches = list(depth_dir.glob(f"**/{stem}_*_depth.npz"))
    if matches and matches[0].stat().st_size > 1024:
        return str(matches[0])
    return None


def validate_preflight(
    source_dir: Path,
    depth_dir: Path,
    do_dc: bool,
    do_vda: bool,
    do_m2svid: bool,
    do_icd: bool = False,
    do_fd: bool = False,
) -> Tuple[str, bool]:
    """Validates pipeline configuration before running benchmark or batch."""
    has_depth_stage = any([do_dc, do_icd, do_fd, do_vda])
    if not any([has_depth_stage, do_m2svid]):
        return (
            "<div style='padding:12px; border-radius:6px; background:#4a1e1e; color:#ffb3b3;'>"
            "⚠️ <b>No Processes Selected:</b> Please select at least one Step 1 Depth Engine (DepthCrafter, ICDepth, FlashDepth, VDA) or Step 2 M2SVid Stereography.</div>",
            False
        )

    if not source_dir.exists():
        return (
            "<div style='padding:12px; border-radius:6px; background:#4a1e1e; color:#ffb3b3;'>"
            "⚠️ <b>Source Folder Missing:</b> Directory `source_videos/` was not found.</div>",
            False
        )

    all_vids = []
    for ext in VIDEO_EXTENSIONS:
        all_vids.extend(glob.glob(str(source_dir / ext)))
    if not all_vids:
        return (
            "<div style='padding:12px; border-radius:6px; background:#4a1e1e; color:#ffb3b3;'>"
            "⚠️ <b>No Videos Found:</b> Please upload video clips to `source_videos/` via FileBrowser first.</div>",
            False
        )

    # Check if M2SVid is selected alone without any depth generator
    if do_m2svid and not has_depth_stage:
        missing_depths = []
        for vp in all_vids:
            stem = Path(vp).stem
            if not find_matching_depth_npz(stem, depth_dir):
                missing_depths.append(Path(vp).name)
        if missing_depths:
            example = ", ".join(missing_depths[:3])
            more = f" and {len(missing_depths)-3} more" if len(missing_depths) > 3 else ""
            return (
                f"<div style='padding:12px; border-radius:6px; background:#3d3618; color:#ffe885;'>"
                f"🛑 <b>Missing Depth Maps:</b> Step 2 (M2SVid) is selected alone, but {len(missing_depths)} clips "
                f"(e.g. <code>{example}{more}</code>) do not have matching depth maps in <code>depthmaps_videos/</code>. "
                f"Please enable <b>DepthCrafter</b> or <b>ICDepth</b> (or FlashDepth/VDA) in Step 1 to generate them automatically.</div>",
                False
            )

    depth_active = []
    if do_dc: depth_active.append("🎬 DepthCrafter")
    if do_icd: depth_active.append("💠 ICDepth")
    if do_fd: depth_active.append("⚡ FlashDepth")
    if do_vda: depth_active.append("⚡ VDA")

    stages_str_parts = []
    if depth_active:
        stages_str_parts.append(f"Step 1 ({' + '.join(depth_active)})")
    if do_m2svid:
        stages_str_parts.append("Step 2 (🎬 M2SVid 3D Stereography)")

    return (
        f"<div style='padding:12px; border-radius:6px; background:#1b3d22; color:#85ff9e;'>"
        f"✅ <b>Pre-Flight Passed:</b> Ready to process `{len(all_vids)}` clips through <b>{' ➔ '.join(stages_str_parts)}</b>.</div>",
        True
    )


def run_pipeline_benchmark(
    sidecar_config: Dict[str, Any],
    do_dc: bool,
    do_vda: bool,
    do_m2svid: bool,
    source_dir: Path,
    depth_dir: Path,
    final_dir: Path,
    dc_runner,
    vda_runner,
    m2svid_runner,
    stop_event: threading.Event,
    do_icd: bool = False,
    do_fd: bool = False,
    icd_runner=None,
    fd_runner=None,
    progress=None
):
    """
    Executes a 3-tier sample benchmark (Short, Medium, Long) using the exact sidecar settings.
    Outputs a detailed performance & cost projection report with a recommended timer.
    """
    stop_event.clear()
    yield "Initializing Multi-Sample Tiered Benchmark...", "Benchmarking...", ""

    all_vids = []
    for ext in VIDEO_EXTENSIONS:
        all_vids.extend(glob.glob(str(source_dir / ext)))
    all_vids = sorted(list(set(all_vids)))

    if not all_vids:
        yield "❌ No video clips found in `source_videos/`.", "Failed", ""
        return

    clip_info_list = []
    total_batch_frames = 0
    for c in all_vids:
        f_count, fps, w, h, dur = get_clip_metadata(c)
        total_batch_frames += f_count
        clip_info_list.append((c, f_count, fps, dur))

    total_batch_sec = sum(x[3] for x in clip_info_list)

    # 3-Tier Sample Selection
    target_bench_dur = 6.0
    test_samples = []
    
    # Tier 1: Short (~6s or shortest available)
    short_c = min(clip_info_list, key=lambda x: abs(x[3] - target_bench_dur))
    test_samples.append({"tier": f"Tier 1: Short (~{short_c[3]:.1f}s)", "path": short_c[0], "frames": short_c[1], "dur": short_c[3]})

    # Tier 2: Medium (~12-15s if available)
    med_candidates = [x for x in clip_info_list if x[0] != short_c[0]]
    if med_candidates:
        med_c = min(med_candidates, key=lambda x: abs(x[3] - 12.0))
        if abs(med_c[3] - short_c[3]) > 1.5:
            test_samples.append({"tier": f"Tier 2: Medium ({med_c[3]:.1f}s)", "path": med_c[0], "frames": med_c[1], "dur": med_c[3]})

    # Tier 3: Longest clip in library
    used_paths = {s["path"] for s in test_samples}
    long_candidates = [x for x in clip_info_list if x[0] not in used_paths]
    if long_candidates:
        long_c = max(long_candidates, key=lambda x: x[1])
        if long_c[1] > short_c[1]:
            test_samples.append({"tier": f"Tier 3: Long ({long_c[3]:.1f}s)", "path": long_c[0], "frames": long_c[1], "dur": long_c[3]})

    sample_results = []
    dc_cfg = sidecar_config.get("depthcrafter_settings", {})
    icd_cfg = sidecar_config.get("icdepth_settings", {})
    fd_cfg = sidecar_config.get("flashdepth_settings", {})
    vda_cfg = sidecar_config.get("vda_settings", {})
    m2s_cfg = sidecar_config.get("m2svid_settings", {})
    depth_prio = sidecar_config.get("depth_priority", "DepthCrafter")
    do_audio = sidecar_config.get("audio_passthrough", {}).get("enabled", True)
    vast_rate = float(sidecar_config.get("cloud_vast_rate_per_hour", 0.50))

    for idx, s in enumerate(test_samples):
        if stop_event.is_set():
            yield "🛑 Benchmark cancelled by user.", "Cancelled", ""
            return

        c_name = os.path.basename(s["path"])
        stem = Path(s["path"]).stem
        msg = f"Testing Sample {idx+1}/{len(test_samples)}: {s['tier']} -> {c_name}"
        yield msg, f"Testing {idx+1}/{len(test_samples)}: {c_name}", ""

        stage_times = {"dc": 0.0, "icd": 0.0, "fd": 0.0, "vda": 0.0, "m2svid": 0.0}

        # 1a. DepthCrafter
        if do_dc and not stop_event.is_set():
            t0 = time.perf_counter()
            dc_runner(
                video_path=s["path"],
                process_res=int(dc_cfg.get("max_res", 1024)),
                guidance_scale=float(dc_cfg.get("guidance_scale", 1.0)),
                inference_steps=int(dc_cfg.get("inference_steps", 5)),
                window_size=int(dc_cfg.get("window_size", 30)),
                overlap=int(dc_cfg.get("overlap", 10)),
                attn_slicing=dc_cfg.get("attn_slicing", "Auto (Adapts to GPU VRAM)"),
                cpu_offload=dc_cfg.get("cpu_offload", "model"),
            )
            stage_times["dc"] = time.perf_counter() - t0

        # 1b. ICDepth
        if do_icd and icd_runner and not stop_event.is_set():
            t0 = time.perf_counter()
            icd_runner(
                input_video=s["path"],
                max_res=int(icd_cfg.get("max_res", 720)),
                window_size=int(icd_cfg.get("window_size", 81)),
                overlap=int(icd_cfg.get("overlap", 32)),
                steps=int(icd_cfg.get("inference_steps", 5)),
                cfg_scale=float(icd_cfg.get("cfg_scale", 5.0)),
                tiled=bool(icd_cfg.get("tiled", False)),
            )
            stage_times["icd"] = time.perf_counter() - t0

        # 1c. FlashDepth
        if do_fd and fd_runner and not stop_event.is_set():
            t0 = time.perf_counter()
            fd_runner(
                input_video=s["path"],
                model_variant=fd_cfg.get("model", "FlashDepth (Full) - 2K VFX Hybrid (Recommended)"),
                max_res=int(fd_cfg.get("max_res", 1288)),
                target_fps=str(fd_cfg.get("target_fps", -1.0)),
                max_len=int(fd_cfg.get("max_len", -1)),
                global_norm=bool(fd_cfg.get("global_norm", True)),
                invert_depth=bool(fd_cfg.get("invert_depth", False)),
            )
            stage_times["fd"] = time.perf_counter() - t0

        # 1d. VDA
        if do_vda and not stop_event.is_set():
            t0 = time.perf_counter()
            vda_runner(
                input_video=s["path"],
                encoder=vda_cfg.get("model", "Large (vitl - Recommended)"),
                mode=vda_cfg.get("mode", "sliding_window"),
                input_size=int(vda_cfg.get("input_size", 644)),
                max_res=str(vda_cfg.get("max_res", 1024)),
                target_fps=str(vda_cfg.get("target_fps", -1)),
                max_len=int(vda_cfg.get("max_len", -1)),
                fp32=bool(vda_cfg.get("fp32", False)),
                global_norm=bool(vda_cfg.get("global_norm", True)),
                invert_depth=bool(vda_cfg.get("invert_depth", False)),
            )
            stage_times["vda"] = time.perf_counter() - t0

        # 2. M2SVid
        if do_m2svid and not stop_event.is_set():
            depth_npz = find_matching_depth_npz(stem, depth_dir, priority=depth_prio)
            if depth_npz:
                t0 = time.perf_counter()
                res = m2svid_runner(
                    depth_npz_path=depth_npz,
                    disparity_perc=float(m2s_cfg.get("disparity_perc", 20.0)),
                    convergence_point=float(m2s_cfg.get("convergence_point", 0.50)),
                    reprojected_closing_kernel=int(m2s_cfg.get("closing_kernel", 11)),
                    mask_antialias=bool(m2s_cfg.get("mask_antialias", False)),
                    m2svid_config="m2svid/configs/m2svid_no_fullatten.yaml",
                    m2svid_ckpt="m2svid/ckpts/m2svid_no_full_atten_weights.pt",
                    input_video_path=s["path"],
                    warping_batch_size=int(m2s_cfg.get("warping_batch_size", 8)),
                    gen_chunk_size=int(m2s_cfg.get("gen_chunk_size", 5)),
                    m2svid_process_res=m2s_cfg.get("processing_res", "Native"),
                    gen_batch_size=int(m2s_cfg.get("gen_batch_size", 1)),
                )
                stage_times["m2svid"] = time.perf_counter() - t0

                # Audio Remux if enabled
                if do_audio:
                    out_stereo_folder = final_dir / f"{stem}_stereo"
                    if out_stereo_folder.is_dir():
                        for vid in out_stereo_folder.glob("*.mp4"):
                            remux_audio_if_present(s["path"], str(vid))

        t_total = sum(stage_times.values())
        effective_fps = s["frames"] / max(0.001, t_total)
        ratio = t_total / max(0.001, s["dur"])
        sample_results.append({
            "tier": s["tier"],
            "name": c_name,
            "frames": s["frames"],
            "dur": s["dur"],
            "times": stage_times,
            "t_total": t_total,
            "fps": effective_fps,
            "ratio": ratio
        })

    # Extrapolate for remaining library
    tested_paths = {s["path"] for s in test_samples}
    remaining_clips = [x for x in clip_info_list if x[0] not in tested_paths]
    remaining_frames = sum(x[1] for x in remaining_clips)

    short_ref = sample_results[0]
    med_ref = sample_results[1] if len(sample_results) > 1 else short_ref
    long_ref = sample_results[-1] if len(sample_results) > 2 else med_ref

    predicted_rem_sec = 0.0
    for cp, cf, cfps, cdur in remaining_clips:
        ref = short_ref if cdur <= 6.0 else (med_ref if cdur <= 18.0 else long_ref)
        predicted_rem_sec += (cf / max(0.001, ref["fps"])) * 1.05

    total_project_sec = sum(r["t_total"] for r in sample_results) + predicted_rem_sec
    total_hours = total_project_sec / 3600.0
    total_cost = total_hours * vast_rate
    cost_per_clip = total_cost / max(1, len(all_vids))
    rem_minutes = predicted_rem_sec / 60.0
    timer_rec = int(rem_minutes + 1)

    # Format Markdown Report
    report = []
    report.append("## 🏆 StereoFaster Multi-Sample Benchmark & Cost Predictor")
    report.append(f"**Hardware Profile**: `{sidecar_config.get('hardware_profile', 'Auto')}` | **Cloud Rental Rate**: `${vast_rate:.3f}/hr`")
    report.append("")
    report.append("| Sample Tier | Clip | Frames | DC Depth | ICDepth | FlashDepth | VDA Depth | M2SVid 3D | Total Time | Speed | Ratio |")
    report.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in sample_results:
        dc_str = f"{r['times']['dc']:.1f}s" if do_dc else "—"
        icd_str = f"{r['times']['icd']:.1f}s" if do_icd else "—"
        fd_str = f"{r['times']['fd']:.1f}s" if do_fd else "—"
        vda_str = f"{r['times']['vda']:.1f}s" if do_vda else "—"
        m2s_str = f"{r['times']['m2svid']:.1f}s" if do_m2svid else "—"
        report.append(
            f"| **{r['tier']}** | `{r['name']}` | {r['frames']} | "
            f"{dc_str} | {icd_str} | {fd_str} | {vda_str} | {m2s_str} | "
            f"**{r['t_total']:.1f}s** | `{r['fps']:.2f} FPS` | `{r['ratio']:.1f}:1` |"
        )
    report.append("")
    report.append(f"> 💡 **Test Samples Auto-Promoted**: The `{len(test_samples)}` benchmarked clips have been saved directly to their output directories and will be **automatically skipped** during the subsequent batch run!")
    report.append("")
    report.append("### ⏱️ Full Batch Time & Cost Projections")
    report.append(f"- **Total Library**: `{len(all_vids)} clips` (`{total_batch_frames:,}` frames | `{total_batch_sec/60:.1f}` mins total runtime)")
    report.append(f"- **Remaining to Process**: `{len(remaining_clips)} clips` (`{remaining_frames:,}` frames)")
    report.append(f"- **ESTIMATED REMAINING TIME**: **`{predicted_rem_sec/3600.0:.2f} HOURS`** (`{rem_minutes:.1f}` minutes)")
    report.append(f"- **ESTIMATED TOTAL CLOUD COST**: **`${total_cost:.2f}`** (@ `${vast_rate:.3f}/hr`)")
    report.append(f"- **Cost Per Finished Clip**: `${cost_per_clip:.4f}` (`{cost_per_clip*100:.2f}¢ / clip`)")
    report.append("")
    report.append(f"### ⏰ Recommended Timer Setting")
    report.append(f"> ### 🔔 **Set your timer for `{timer_rec} MINUTES`** (or ~`{predicted_rem_sec/3600.0:.2f} hours`).")
    report.append("> Once the timer fires, your full batch queue will be finished and ready to download via FileBrowser!")

    yield "✅ Benchmark Complete!", "Benchmark Completed", "\n".join(report)


def run_pipeline_batch(
    sidecar_config: Dict[str, Any],
    do_dc: bool,
    do_vda: bool,
    do_m2svid: bool,
    source_dir: Path,
    depth_dir: Path,
    final_dir: Path,
    dc_runner,
    vda_runner,
    m2svid_runner,
    stop_event: threading.Event,
    do_icd: bool = False,
    do_fd: bool = False,
    icd_runner=None,
    fd_runner=None,
    progress=None
):
    """
    Executes the batch queue prioritized from Shortest to Longest clips with per-stage Auto-Skip.
    """
    stop_event.clear()
    yield "Starting Prioritized Batch Queue (Short ➔ Long)...", "Starting Batch..."

    all_vids = []
    for ext in VIDEO_EXTENSIONS:
        all_vids.extend(glob.glob(str(source_dir / ext)))
    all_vids = sorted(list(set(all_vids)))
    stems_in_dir = {Path(v).stem for v in all_vids}
    all_vids = [
        v for v in all_vids
        if not (not Path(v).stem.endswith("_cropped") and f"{Path(v).stem}_cropped" in stems_in_dir)
    ]

    if not all_vids:
        yield "❌ No video clips found in `source_videos/`.", "Failed"
        return

    clip_items = []
    for c in all_vids:
        f_count, fps, w, h, dur = get_clip_metadata(c)
        clip_items.append({"path": c, "dur": dur, "frames": f_count, "stem": Path(c).stem})

    # Prioritized Queue: Shortest clips first!
    clip_items.sort(key=lambda x: x["dur"])

    t_batch_start = time.perf_counter()
    total_clips = len(clip_items)
    dc_cfg = sidecar_config.get("depthcrafter_settings", {})
    icd_cfg = sidecar_config.get("icdepth_settings", {})
    fd_cfg = sidecar_config.get("flashdepth_settings", {})
    vda_cfg = sidecar_config.get("vda_settings", {})
    m2s_cfg = sidecar_config.get("m2svid_settings", {})
    depth_prio = sidecar_config.get("depth_priority", "DepthCrafter")
    do_audio = sidecar_config.get("audio_passthrough", {}).get("enabled", True)
    vast_rate = float(sidecar_config.get("cloud_vast_rate_per_hour", 0.50))

    yield f"Queued {total_clips} clips (Shortest to Longest). Executing...", "Running Batch..."

    for idx, c in enumerate(clip_items):
        if stop_event.is_set():
            yield f"🛑 Batch stopped by user at clip {idx+1}/{total_clips}.", "Stopped"
            return

        c_name = os.path.basename(c["path"])
        stem = c["stem"]
        prefix = f"[{idx+1}/{total_clips}] {c_name}"

        # 1a. DepthCrafter (with auto-skip)
        if do_dc and not stop_event.is_set():
            expected_npz = depth_dir / f"{stem}_DC_depth.npz"
            expected_mp4 = depth_dir / f"{stem}_DC_depth.mp4"
            if (expected_npz.exists() and expected_npz.stat().st_size > 1024) or (expected_mp4.exists() and expected_mp4.stat().st_size > 1024):
                logger.info(f"[AUTO-SKIP] DepthCrafter exists for {stem}")
            else:
                yield f"{prefix} ➔ Running DepthCrafter...", f"Clip {idx+1}/{total_clips}: DepthCrafter"
                dc_runner(
                    video_path=c["path"],
                    process_res=int(dc_cfg.get("max_res", 1024)),
                    guidance_scale=float(dc_cfg.get("guidance_scale", 1.0)),
                    inference_steps=int(dc_cfg.get("inference_steps", 5)),
                    window_size=int(dc_cfg.get("window_size", 30)),
                    overlap=int(dc_cfg.get("overlap", 10)),
                    attn_slicing=dc_cfg.get("attn_slicing", "Auto (Adapts to GPU VRAM)"),
                    cpu_offload=dc_cfg.get("cpu_offload", "model"),
                )

        # 1b. ICDepth (with auto-skip)
        if do_icd and icd_runner and not stop_event.is_set():
            expected_npz = depth_dir / f"{stem}_ICD_depth.npz"
            expected_mp4 = depth_dir / f"{stem}_ICD_depth.mp4"
            if (expected_npz.exists() and expected_npz.stat().st_size > 1024) or (expected_mp4.exists() and expected_mp4.stat().st_size > 1024):
                logger.info(f"[AUTO-SKIP] ICDepth exists for {stem}")
            else:
                yield f"{prefix} ➔ Running ICDepth...", f"Clip {idx+1}/{total_clips}: ICDepth"
                icd_runner(
                    input_video=c["path"],
                    max_res=int(icd_cfg.get("max_res", 720)),
                    window_size=int(icd_cfg.get("window_size", 81)),
                    overlap=int(icd_cfg.get("overlap", 32)),
                    steps=int(icd_cfg.get("inference_steps", 5)),
                    cfg_scale=float(icd_cfg.get("cfg_scale", 5.0)),
                    tiled=bool(icd_cfg.get("tiled", False)),
                )

        # 1c. FlashDepth (with auto-skip)
        if do_fd and fd_runner and not stop_event.is_set():
            fd_exists = any(
                ((depth_dir / f"{stem}_{sfx}_depth.npz").exists() and (depth_dir / f"{stem}_{sfx}_depth.npz").stat().st_size > 1024) or
                ((depth_dir / f"{stem}_{sfx}_depth.mp4").exists() and (depth_dir / f"{stem}_{sfx}_depth.mp4").stat().st_size > 1024)
                for sfx in ("FDF", "FDL", "FDS", "FD")
            )
            if fd_exists:
                logger.info(f"[AUTO-SKIP] FlashDepth exists for {stem}")
            else:
                yield f"{prefix} ➔ Running FlashDepth...", f"Clip {idx+1}/{total_clips}: FlashDepth"
                fd_runner(
                    input_video=c["path"],
                    model_variant=fd_cfg.get("model", "FlashDepth (Full) - 2K VFX Hybrid (Recommended)"),
                    max_res=int(fd_cfg.get("max_res", 1288)),
                    target_fps=str(fd_cfg.get("target_fps", -1.0)),
                    max_len=int(fd_cfg.get("max_len", -1)),
                    global_norm=bool(fd_cfg.get("global_norm", True)),
                    invert_depth=bool(fd_cfg.get("invert_depth", False)),
                )

        # 1d. VDA (with auto-skip)
        if do_vda and not stop_event.is_set():
            vda_exists = any(
                ((depth_dir / f"{stem}_{sfx}_depth.npz").exists() and (depth_dir / f"{stem}_{sfx}_depth.npz").stat().st_size > 1024) or
                ((depth_dir / f"{stem}_{sfx}_depth.mp4").exists() and (depth_dir / f"{stem}_{sfx}_depth.mp4").stat().st_size > 1024)
                for sfx in ("VDAL", "VDAB", "VDA")
            )
            if vda_exists:
                logger.info(f"[AUTO-SKIP] VDA depth exists for {stem}")
            else:
                yield f"{prefix} ➔ Running VDA Depth...", f"Clip {idx+1}/{total_clips}: VDA Depth"
                vda_runner(
                    input_video=c["path"],
                    encoder=vda_cfg.get("model", "Large (vitl - Recommended)"),
                    mode=vda_cfg.get("mode", "sliding_window"),
                    input_size=int(vda_cfg.get("input_size", 644)),
                    max_res=str(vda_cfg.get("max_res", 1024)),
                    target_fps=str(vda_cfg.get("target_fps", -1)),
                    max_len=int(vda_cfg.get("max_len", -1)),
                    fp32=bool(vda_cfg.get("fp32", False)),
                    global_norm=bool(vda_cfg.get("global_norm", True)),
                    invert_depth=bool(vda_cfg.get("invert_depth", False)),
                )

        # 3. M2SVid Stereography (with auto-skip)
        if do_m2svid and not stop_event.is_set():
            stereo_folder = final_dir / f"{stem}_stereo"
            already_done = False
            if stereo_folder.is_dir():
                sbs_vids = list(stereo_folder.glob("*.mp4"))
                if any("_sbs" in v.name or "_anaglyph" in v.name for v in sbs_vids):
                    already_done = True

            if already_done:
                logger.info(f"[AUTO-SKIP] M2SVid stereo outputs exist for {stem}")
            else:
                depth_npz = find_matching_depth_npz(stem, depth_dir, priority=depth_prio)
                if depth_npz:
                    yield f"{prefix} ➔ Running M2SVid Stereography...", f"Clip {idx+1}/{total_clips}: M2SVid 3D"
                    m2svid_runner(
                        depth_npz_path=depth_npz,
                        disparity_perc=float(m2s_cfg.get("disparity_perc", 20.0)),
                        convergence_point=float(m2s_cfg.get("convergence_point", 0.50)),
                        reprojected_closing_kernel=int(m2s_cfg.get("closing_kernel", 11)),
                        mask_antialias=bool(m2s_cfg.get("mask_antialias", False)),
                        m2svid_config="m2svid/configs/m2svid_no_fullatten.yaml",
                        m2svid_ckpt="m2svid/ckpts/m2svid_no_full_atten_weights.pt",
                        input_video_path=c["path"],
                        warping_batch_size=int(m2s_cfg.get("warping_batch_size", 8)),
                        gen_chunk_size=int(m2s_cfg.get("gen_chunk_size", 5)),
                        m2svid_process_res=m2s_cfg.get("processing_res", "Native"),
                        gen_batch_size=int(m2s_cfg.get("gen_batch_size", 1)),
                    )

                    # Audio Passthrough
                    if do_audio and stereo_folder.is_dir():
                        for vid in stereo_folder.glob("*.mp4"):
                            remux_audio_if_present(c["path"], str(vid))

    elapsed = time.perf_counter() - t_batch_start
    hours = elapsed / 3600.0
    cost = hours * vast_rate
    res_msg = (
        f"🎉 **Batch Complete!** Processed `{total_clips}` clips in `{hours:.2f} hours` "
        f"(`{elapsed/60:.1f} mins`). Estimated Cloud Cost: `${cost:.2f}`."
    )
    yield res_msg, "Batch Completed"

