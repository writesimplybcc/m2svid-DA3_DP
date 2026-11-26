"""
Depth Editor & Layer Hierarchy Module for StereoFaster
Allows interactive layer management, holdout matte protection, and multi-plane depth slicing
for complex surreal scenes (e.g. fish and dugong floating in the sky behind bridge trusses).

Workflow Architecture:
1. Layer Auto-Clustering: Discovers distinct depth planes from the base NPZ (Foremost Credits,
   Structural Midground/Road, Background Aperture).
2. Immutable Holdout Matte Protection: Foreground structures (Credits, Road, Trusses, Cars) are 100% locked.
3. Sub-Division of Background Aperture: Separates background into Floating Creatures, Cityscape, and Sky Floor.
4. Multi-Keyframe Painting: Records paint strokes across multiple user keyframes (t1, t2, t3...).
5. Holdout-Aware Tracking: Snaps to organic contours with Top-Hat filtering, tracks bidirectionally via
   optical flow, and subtracts locked trusses (Creatures = Painted \\ LockedTrusses).
6. Sidecar Persistence ({stem}.layers.json) for 100% reproducible batch runs.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2

SF_LOG = logging.getLogger("stereofaster")


def _merge_custom_layers_from_sidecar(
    layers: List[Dict[str, Any]],
    sidecar_layers: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Preserve custom layer additions, descriptions, depth values, and keyframe metadata from sidecar."""
    if not sidecar_layers:
        return layers

    base_ids = {l.get("id") for l in layers}
    for sl in sidecar_layers:
        sid = sl.get("id")
        if sid in base_ids:
            for l in layers:
                if l.get("id") == sid:
                    if sl.get("desc"):
                        l["desc"] = sl.get("desc")
                    if "depth_val" in sl:
                        try:
                            l["depth_val"] = float(sl["depth_val"])
                        except (ValueError, TypeError):
                            pass
                    if "depth_range" in sl and isinstance(sl["depth_range"], (list, tuple)) and len(sl["depth_range"]) == 2:
                        l["depth_min"] = float(sl["depth_range"][0])
                        l["depth_max"] = float(sl["depth_range"][1])
                        l["range"] = f"{l['depth_min']:.2f} – {l['depth_max']:.2f}"
                    if "keyframes" in sl:
                        l["keyframes"] = list(sl["keyframes"])
                    break
        else:
            custom_layer = dict(sl)
            if "name" not in custom_layer:
                custom_layer["name"] = f"Layer {sid}: Custom Layer"
            if "desc" not in custom_layer:
                custom_layer["desc"] = f"Custom Layer {sid}"
            if "range" not in custom_layer:
                dmin = custom_layer.get("depth_min", 0.10)
                dmax = custom_layer.get("depth_max", 0.25)
                custom_layer["range"] = f"{dmin:.2f} – {dmax:.2f}"
            if "locked" not in custom_layer:
                custom_layer["locked"] = True
            if "pixel_pct" not in custom_layer:
                custom_layer["pixel_pct"] = 0.0
            if "depth_val" not in custom_layer:
                custom_layer["depth_val"] = 0.18
            layers.append(custom_layer)

    return layers


def detect_scene_layer_hierarchy(
    depth_map: np.ndarray,
    sidecar_layers: Optional[List[Dict[str, Any]]] = None,
    credit_text_hint: str = "",
) -> List[Dict[str, Any]]:
    """Analyze the base depth map from NPZ to dynamically cluster semantic depth planes.

    Categorizes into sorted planes from closest to furthest:
    - When on-screen credits exist (d >= 0.82 or credit sidecar):
      - Layer 1: On-Screen Credits / Text (0.82 – 1.00)
      - Layer 2: Foreground Structure (Road / Bridge / Characters) (t_fg – 0.82)
      - Layer 3: Midground (Floating Creatures / Aperture) (t_bg – t_fg)
      - Layer 4: Deep Background (City & Sky Floor) (0.00 – t_bg)
    - Standard 3-plane scene (no credits):
      - Layer 1: Foreground (t_fg – 1.00)
      - Layer 2: Midground (t_bg – t_fg)
      - Layer 3: Background (0.00 – t_bg)

    BY DEFAULT, ALL LAYERS ARE LOCKED (locked = True).
    The user can unlock any layer for editing, and descriptions can be customized.
    """
    H, W = depth_map.shape[:2]
    total = float(H * W)
    d_clean = np.clip(np.nan_to_num(depth_map, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    flat = d_clean.ravel()

    # Detect if scene has on-screen credits / foremost title text
    has_credits = False
    if credit_text_hint and credit_text_hint.strip():
        has_credits = True
    elif float((d_clean > 0.80).mean()) > 0.0005 and float((d_clean[:int(H * 0.45)] > 0.80).mean()) > 0.0002:
        has_credits = True

    # Dynamic depth clustering
    sample_size = min(25000, len(flat))
    if sample_size > 0:
        samples = np.random.choice(flat, sample_size, replace=False)
    else:
        samples = flat

    if has_credits:
        t_credits = 0.82
        scene_samples = samples[samples < t_credits] if np.any(samples < t_credits) else samples
        if len(scene_samples) > 100:
            t_bg = float(np.percentile(scene_samples, 35.0))
            t_fg = float(np.percentile(scene_samples, 75.0))
        else:
            t_bg, t_fg = 0.12, 0.48

        t_bg = max(0.02, min(0.30, t_bg))
        t_fg = max(t_bg + 0.05, min(t_credits - 0.05, t_fg))

        p_cred = round(float(np.count_nonzero(d_clean >= t_credits) / total * 100.0), 1)
        p_fg = round(float(np.count_nonzero((d_clean >= t_fg) & (d_clean < t_credits)) / total * 100.0), 1)
        p_mid = round(float(np.count_nonzero((d_clean > t_bg) & (d_clean < t_fg)) / total * 100.0), 1)
        p_bg = round(float(np.count_nonzero(d_clean <= t_bg) / total * 100.0), 1)

        desc1 = "Credits / Top Overlay"
        desc2 = "Foreground"
        desc3 = "Midground"
        desc4 = "Background / Floor"

        if sidecar_layers:
            for sl in sidecar_layers:
                sid = sl.get("id")
                sdesc = sl.get("desc") or sl.get("description")
                if sid == 1 and sdesc: desc1 = sdesc
                elif sid == 2 and sdesc: desc2 = sdesc
                elif sid == 3 and sdesc: desc3 = sdesc
                elif sid == 4 and sdesc: desc4 = sdesc

        layers = [
            {
                "id": 1,
                "name": "Layer 1: Credits / Overlay",
                "desc": desc1,
                "range": f"{t_credits:.2f} – 1.00",
                "depth_min": round(t_credits, 2),
                "depth_max": 1.00,
                "depth_val": 0.95,
                "locked": True,
                "pixel_pct": p_cred,
            },
            {
                "id": 2,
                "name": "Layer 2: Foreground",
                "desc": desc2,
                "range": f"{t_fg:.2f} – {t_credits:.2f}",
                "depth_min": round(t_fg, 2),
                "depth_max": round(t_credits, 2),
                "depth_val": round((t_fg + t_credits) / 2.0, 2),
                "locked": True,
                "pixel_pct": p_fg,
            },
            {
                "id": 3,
                "name": "Layer 3: Midground",
                "desc": desc3,
                "range": f"{t_bg:.2f} – {t_fg:.2f}",
                "depth_min": round(t_bg, 2),
                "depth_max": round(t_fg, 2),
                "depth_val": round((t_bg + t_fg) / 2.0, 2),
                "locked": True,
                "pixel_pct": p_mid,
            },
            {
                "id": 4,
                "name": "Layer 4: Background / Floor",
                "desc": desc4,
                "range": f"0.00 – {t_bg:.2f}",
                "depth_min": 0.00,
                "depth_max": round(t_bg, 2),
                "depth_val": round(t_bg / 2.0, 2),
                "locked": True,
                "pixel_pct": p_bg,
            },
        ]
        return _merge_custom_layers_from_sidecar(layers, sidecar_layers)

    # 3-plane clustering fallback (no credits)
    try:
        from scipy.cluster.vq import kmeans
        centroids, _ = kmeans(samples, 3)
        centroids = sorted([float(c) for c in centroids])
        if len(centroids) == 3 and (centroids[2] - centroids[0]) > 0.05:
            t_bg = float((centroids[0] + centroids[1]) / 2.0)
            t_fg = float((centroids[1] + centroids[2]) / 2.0)
        else:
            t_bg = float(np.percentile(samples, 33.3))
            t_fg = float(np.percentile(samples, 66.6))
    except Exception:
        t_bg = float(np.percentile(samples, 33.3))
        t_fg = float(np.percentile(samples, 66.6))

    t_bg = max(0.02, min(0.40, t_bg))
    t_fg = max(t_bg + 0.05, min(0.90, t_fg))

    p_bg = round(float(np.count_nonzero(d_clean <= t_bg) / total * 100.0), 1)
    p_fg = round(float(np.count_nonzero(d_clean > t_fg) / total * 100.0), 1)
    p_mid = round(max(0.0, 100.0 - p_bg - p_fg), 1)

    desc1 = "Foreground"
    desc2 = "Midground"
    desc3 = "Background / Floor"

    if sidecar_layers:
        for sl in sidecar_layers:
            sid = sl.get("id")
            sdesc = sl.get("desc") or sl.get("description")
            if sid == 1 and sdesc:
                desc1 = sdesc
            elif sid == 2 and sdesc:
                desc2 = sdesc
            elif sid == 3 and sdesc:
                desc3 = sdesc

    layers = [
        {
            "id": 1,
            "name": "Layer 1: Foreground",
            "desc": desc1,
            "range": f"{t_fg:.2f} – 1.00",
            "depth_min": round(t_fg, 2),
            "depth_max": 1.00,
            "depth_val": round((t_fg + 1.0) / 2.0, 2),
            "locked": True,
            "pixel_pct": p_fg,
        },
        {
            "id": 2,
            "name": "Layer 2: Midground",
            "desc": desc2,
            "range": f"{t_bg:.2f} – {t_fg:.2f}",
            "depth_min": round(t_bg, 2),
            "depth_max": round(t_fg, 2),
            "depth_val": round((t_bg + t_fg) / 2.0, 2),
            "locked": True,
            "pixel_pct": p_mid,
        },
        {
            "id": 3,
            "name": "Layer 3: Background / Floor",
            "desc": desc3,
            "range": f"0.00 – {t_bg:.2f}",
            "depth_min": 0.00,
            "depth_max": round(t_bg, 2),
            "depth_val": round(t_bg / 2.0, 2),
            "locked": True,
            "pixel_pct": p_bg,
        },
    ]

    return _merge_custom_layers_from_sidecar(layers, sidecar_layers)


def layers_to_dataframe_data(
    layers: List[Dict[str, Any]],
    unlocked_id: Optional[int] = None,
) -> List[List[str]]:
    """Convert layer hierarchy to rows for an interactive gr.Dataframe."""
    rows = []
    for l in layers:
        is_unlocked = (unlocked_id is not None and l.get("id") == unlocked_id)
        status_str = "✏️ Unlocked (Active)" if is_unlocked else "🔒 Locked"
        lid = f"Layer {l.get('id', '')}"
        desc = str(l.get("desc", ""))
        drange = str(l.get("range", ""))
        pct = f"{l.get('pixel_pct', 0.0)}%"
        rows.append([lid, desc, drange, pct, status_str])
    return rows


def render_layer_table_markdown(
    layers: List[Dict[str, Any]],
    editable_layer_id: Optional[int] = None,
    sub_layers: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render a clean GitHub Flavored Markdown table of the scene's layer hierarchy."""
    md = "### 📍 Scene Depth Hierarchy & Holdout Matte Protection\n\n"
    md += "| Layer ID | Semantic Description | Depth Range | Status | Role |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"
    for l in layers:
        is_unlocked = (editable_layer_id is not None and l["id"] == editable_layer_id)
        lock_icon = "✏️ **Unlocked (Active)**" if is_unlocked else "🔒 **Locked**"
        action = f"**Active for Editing (Layer {l['id']})**" if is_unlocked else "*Protected Holdout Matte*"
        title = l.get("desc") or l.get("name", "")
        lid = f"Layer {l['id']}"
        md += f"| **{lid}** | {title} | `{l['range']}` ({l['pixel_pct']}%) | {lock_icon} | {action} |\n"

    if editable_layer_id is not None:
        md += "\n---\n"
        md += f"#### 📂 Active Editing Controls for Layer {editable_layer_id}:\n"
        if sub_layers:
            for sl in sub_layers:
                md += f"* {sl.get('icon', '🔹')} **{sl['name']}**: Target Depth = `{sl['depth']:.2f}` — *{sl.get('desc', '')}*\n"
        else:
            md += f"* ✏️ **Layer {editable_layer_id}**: Ready for object painting, optical flow tracking, and depth assignment.\n"
    else:
        md += "\n---\n*🔒 All layers are currently locked. Unlock a layer below to activate editing.*"

    return md


def analyze_scene_depth_anchors(
    rgb_frame: np.ndarray,
    depth_map: np.ndarray,
    sample_grid: int = 24,
) -> Dict[str, Any]:
    """Analyze a reference frame's depth map to identify distinct semantic depth layers/anchors."""
    H, W = depth_map.shape[:2]
    d_clean = np.nan_to_num(depth_map, nan=0.0, posinf=1.0, neginf=0.0)

    margin_y = int(H * 0.08)
    margin_x = int(W * 0.08)
    ys = np.linspace(margin_y, H - margin_y - 1, sample_grid, dtype=int)
    xs = np.linspace(margin_x, W - margin_x - 1, sample_grid, dtype=int)

    samples = []
    for y in ys:
        for x in xs:
            val = float(d_clean[y, x])
            samples.append((val, x, y))

    samples.sort(key=lambda s: s[0])
    vals = np.array([s[0] for s in samples])

    p5 = float(np.percentile(vals, 5))
    p50 = float(np.percentile(vals, 50))
    p90 = float(np.percentile(vals, 90))

    anchors = []

    # 1. Background Anchor (Sky / Distant City)
    bg_candidates = [s for s in samples if s[2] < int(H * 0.65) and s[0] <= p5 + 0.03]
    if bg_candidates:
        bg_cand = min(bg_candidates, key=lambda s: abs(s[1] - int(W * 0.50)) + abs(s[2] - int(H * 0.25)))
        anchors.append({
            "name": "Cityscape / Sky (Background)",
            "depth": round(bg_cand[0], 3),
            "x": int(bg_cand[1]),
            "y": int(bg_cand[2]),
            "tag": f"Cityscape/Sky: {bg_cand[0]:.2f}",
            "color": (255, 160, 0),
        })
    else:
        anchors.append({
            "name": "Distant Background",
            "depth": round(p5, 3),
            "x": W // 2,
            "y": int(H * 0.20),
            "tag": f"Background: {p5:.2f}",
            "color": (255, 160, 0),
        })

    # 2. Structural Midground / Road / Truss
    mid_candidates = [s for s in samples if 0.20 <= s[0] <= 0.75 and (s[2] > int(H * 0.50) or s[1] > int(W * 0.60))]
    if mid_candidates:
        mid_cand = min(mid_candidates, key=lambda s: abs(s[0] - max(0.35, p50)))
        anchors.append({
            "name": "Elevated Road / Structure",
            "depth": round(mid_cand[0], 3),
            "x": int(mid_cand[1]),
            "y": int(mid_cand[2]),
            "tag": f"Structure/Road: {mid_cand[0]:.2f}",
            "color": (0, 215, 255),
        })
    elif p90 > 0.30:
        anchors.append({
            "name": "Mid-Foreground",
            "depth": round(p90, 3),
            "x": int(W * 0.75),
            "y": int(H * 0.70),
            "tag": f"Mid-Foreground: {p90:.2f}",
            "color": (0, 215, 255),
        })

    # 3. Credits / Closest Foreground (if present above 0.85)
    credit_candidates = [s for s in samples if s[0] >= 0.85]
    if credit_candidates:
        cr_cand = credit_candidates[len(credit_candidates) // 2]
        anchors.append({
            "name": "On-Screen Credits (Foreground)",
            "depth": round(cr_cand[0], 3),
            "x": int(cr_cand[1]),
            "y": int(cr_cand[2]),
            "tag": f"Credits: {cr_cand[0]:.2f}",
            "color": (0, 255, 120),
        })

    bg_d = anchors[0]["depth"]
    mid_d = anchors[1]["depth"] if len(anchors) > 1 else max(0.40, p50)
    cr_d = anchors[2]["depth"] if len(anchors) > 2 else 1.00

    return {
        "anchors": anchors,
        "bg_depth": bg_d,
        "mid_depth": mid_d,
        "credits_depth": cr_d,
    }


def render_annotated_reference_frame(
    rgb_frame: np.ndarray,
    depth_map: np.ndarray,
    anchors_info: Dict[str, Any],
    user_points: Optional[List[Tuple[int, int]]] = None,
    object_mask: Optional[np.ndarray] = None,
    target_depth: Optional[float] = None,
) -> np.ndarray:
    """Render a clean HUD overlay on the reference frame showing depth anchor tags and user points."""
    H, W = rgb_frame.shape[:2]
    canvas = rgb_frame.copy()

    # Draw object mask if present
    if object_mask is not None and np.any(object_mask):
        mask_u8 = (object_mask * 255).astype(np.uint8) if object_mask.dtype != np.uint8 else object_mask
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = canvas.copy()
        cv2.drawContours(overlay, contours, -1, (255, 0, 255), -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)
        cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)

    # Draw anchor pins
    anchors = anchors_info.get("anchors", [])
    for anc in anchors:
        x, y = anc["x"], anc["y"]
        tag = anc["tag"]
        color = anc.get("color", (0, 215, 255))

        cv2.circle(canvas, (x, y), 6, color, -1)
        cv2.circle(canvas, (x, y), 8, (255, 255, 255), 1)

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bx0 = max(6, min(W - tw - 12, x - tw // 2))
        by0 = max(th + 8, min(H - 12, y - 12))

        cv2.rectangle(canvas, (bx0 - 4, by0 - th - 4), (bx0 + tw + 4, by0 + 4), (15, 15, 15), -1)
        cv2.rectangle(canvas, (bx0 - 4, by0 - th - 4), (bx0 + tw + 4, by0 + 4), color, 1)
        cv2.putText(canvas, tag, (bx0, by0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw user clicked points
    if user_points:
        for idx, (px, py) in enumerate(user_points):
            if 0 <= px < W and 0 <= py < H:
                val = float(depth_map[py, px])
                p_tag = f"P{idx+1} ({val:.2f})"
                cv2.circle(canvas, (px, py), 7, (0, 0, 255), -1)
                cv2.circle(canvas, (px, py), 9, (255, 255, 255), 2)

                (tw, th), _ = cv2.getTextSize(p_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                pbx = max(6, min(W - tw - 10, px + 10))
                pby = max(th + 6, min(H - 10, py - 4))
                cv2.rectangle(canvas, (pbx - 3, pby - th - 3), (pbx + tw + 3, pby + 3), (0, 0, 180), -1)
                cv2.putText(canvas, p_tag, (pbx, pby), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if target_depth is not None:
        info_text = f"Target Depth: {target_depth:.2f} [Cityscape: {anchors_info.get('bg_depth', 0.0):.2f} | Road: {anchors_info.get('mid_depth', 0.5):.2f}]"
        (tw, th), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (10, H - th - 18), (20 + tw, H - 8), (15, 15, 15), -1)
        cv2.rectangle(canvas, (10, H - th - 18), (20 + tw, H - 8), (0, 220, 255), 1)
        cv2.putText(canvas, info_text, (15, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas


def apply_direct_brush_fidelity_and_volume(
    user_mask: np.ndarray,
    target_shape: Optional[Tuple[int, int]] = None,
    feather_kernel: int = 5,
    relief_scale: float = 0.035,
) -> Tuple[np.ndarray, np.ndarray]:
    """Process user painted brush strokes with 100% direct contour fidelity and Euclidean 3D volume relief.

    Guarantees:
    1. Zero contrast-snapping distortion: The user's exact painted silhouette is preserved.
    2. Sub-pixel anti-aliasing: Gentle boundary roll-off to avoid hard stepped raster edges.
    3. Organic 3D Volumetric Relief: Computes Euclidean distance transform (fullest at body center,
       smoothly tapering to the boundaries) for natural 3D depth.
    """
    if user_mask is None:
        shape = target_shape or (480, 854)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    mask_f = user_mask.astype(np.float32)
    if target_shape is not None and mask_f.shape[:2] != target_shape[:2]:
        mask_f = cv2.resize(mask_f, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    H, W = mask_f.shape[:2]
    user_u8 = (mask_f > 0.05).astype(np.uint8)
    if not np.any(user_u8):
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    # Gentle morphological close to bridge tiny gaps within a single stroke while strictly preserving boundary
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_u8 = cv2.morphologyEx(user_u8, cv2.MORPH_CLOSE, kernel_small)

    # Anti-aliased / feathered boundary to prevent harsh aliasing
    fk = feather_kernel if feather_kernel % 2 == 1 else feather_kernel + 1
    feathered_mask = cv2.GaussianBlur(clean_u8.astype(np.float32), (fk, fk), 0)
    feathered_mask = np.clip(feathered_mask, 0.0, 1.0)

    # Compute organic 3D volumetric relief using Euclidean distance transform
    bin_m = (clean_u8 > 0).astype(np.uint8)
    if np.any(bin_m):
        dist = cv2.distanceTransform(bin_m, cv2.DIST_L2, 5)
        dist_max = float(np.percentile(dist[bin_m > 0], 90)) if np.any(bin_m) else 1.0
        vol_relief = np.clip(dist / max(1.0, dist_max), 0.0, 1.0) * relief_scale
    else:
        vol_relief = np.zeros((H, W), dtype=np.float32)

    return feathered_mask, vol_relief


def snap_mask_with_tophat_and_volume(
    rgb_frame: np.ndarray,
    user_mask: np.ndarray,
    kernel_size: int = 31,
    min_contrast: float = 16.0,
    relief_scale: float = 0.035,
) -> Tuple[np.ndarray, np.ndarray]:
    """Snap rough user painted strokes to precise object contours and compute 3D volume relief."""
    H, W = rgb_frame.shape[:2]
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY) if rgb_frame.ndim == 3 else rgb_frame.copy()

    user_u8 = (user_mask > 0.05).astype(np.uint8)
    if not np.any(user_u8):
        return np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32)

    # Top-Hat and Black-Hat morphological operators
    ks = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    response = np.maximum(th, bh)

    # Search window around user strokes
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    stroke_dil = cv2.dilate(user_u8, k_dil)

    cand = ((response >= min_contrast) & (stroke_dil > 0)).astype(np.uint8)
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel_c)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cand)
    snapped_mask = np.zeros((H, W), dtype=np.float32)
    matched = False
    for i in range(1, num_labels):
        comp = (labels == i)
        if np.any(comp & (user_u8 > 0)):
            snapped_mask += comp.astype(np.float32)
            matched = True

    if not matched or np.max(snapped_mask) < 0.1:
        snapped_mask = user_mask.astype(np.float32)

    # CRUCIAL: Always retain 100% of the user's painted interior. Top-hat only snaps contours outward;
    # it must never turn the object into a hollow static honeycomb.
    combined = np.maximum(user_u8.astype(np.float32), (snapped_mask > 0.5).astype(np.float32))
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined_u8 = cv2.morphologyEx((combined > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, kernel_fill)
    contours, _ = cv2.findContours(combined_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(combined_u8, contours, -1, 1, -1)

    snapped_mask = cv2.GaussianBlur(combined_u8.astype(np.float32), (5, 5), 0)
    snapped_mask = np.clip(snapped_mask, 0.0, 1.0)

    # Compute 3D volumetric relief
    bin_m = (snapped_mask > 0.3).astype(np.uint8)
    if np.any(bin_m):
        dist = cv2.distanceTransform(bin_m, cv2.DIST_L2, 5)
        dist_max = float(np.percentile(dist[bin_m > 0], 90)) if np.any(bin_m) else 1.0
        vol_relief = np.clip(dist / max(1.0, dist_max), 0.0, 1.0) * relief_scale
    else:
        vol_relief = np.zeros((H, W), dtype=np.float32)

    return snapped_mask, vol_relief


def split_skyline_and_horizon(
    rgb_frame: np.ndarray,
    depth_map: np.ndarray,
    sky_floor: float = 0.00,
    city_anchor: float = 0.05,
    road_floor: float = 0.30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Delineate horizon between sky and cityscape."""
    H, W = depth_map.shape[:2]
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY) if rgb_frame.ndim == 3 else rgb_frame.copy()

    y_min = int(H * 0.25)
    y_max = int(H * 0.60)
    if y_max <= y_min + 10:
        y_min, y_max = int(H * 0.15), int(H * 0.75)

    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    horizon_y = np.zeros(W, dtype=np.uint8)
    for x in range(W):
        col_grad = sobel_y[y_min:y_max, x]
        peak = y_min + int(np.argmin(col_grad))
        horizon_y[x] = np.clip(peak, 0, 255)

    k_smooth = max(15, (W // 40) | 1)
    horizon_smooth = cv2.medianBlur(horizon_y[:, None], k_smooth).ravel()

    sky_mask = np.zeros((H, W), dtype=bool)
    city_mask = np.zeros((H, W), dtype=bool)
    city_band_h = max(30, int(H * 0.14))

    for x in range(W):
        hy = int(horizon_smooth[x])
        sky_mask[:hy, x] = True
        city_mask[hy:min(H, hy + city_band_h), x] = True

    depth_out = depth_map.copy().astype(np.float32)
    depth_out[sky_mask] = float(sky_floor)
    depth_out[city_mask] = float(city_anchor)

    for x in range(W):
        hy = int(horizon_smooth[x])
        road_y = min(H, hy + city_band_h)
        if road_y < H:
            depth_out[road_y:, x] = np.maximum(depth_out[road_y:, x], float(road_floor))

    return depth_out, sky_mask, city_mask


def _warp_mask_flow(
    src_gray: np.ndarray,
    dst_gray: np.ndarray,
    src_mask: np.ndarray,
    dis_engine: Optional[Any] = None,
    preserve_contrast: bool = True,
) -> np.ndarray:
    """Warp src_mask to align with dst_gray using high-precision optical flow.

    To find where pixels in dst_gray came from in src_gray, we calculate flow from dst_gray
    to src_gray and sample src_mask at (grid + flow).
    """
    H, W = src_gray.shape[:2]
    h_grid, w_grid = np.indices((H, W), dtype=np.float32)

    if dis_engine is not None:
        flow = dis_engine.calc(dst_gray, src_gray, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(
            dst_gray, src_gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
    map_x = np.clip(w_grid + flow[..., 0], 0, W - 1).astype(np.float32)
    map_y = np.clip(h_grid + flow[..., 1], 0, H - 1).astype(np.float32)
    warped = cv2.remap(src_mask.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR)
    warped = np.clip(warped, 0.0, 1.0)

    if preserve_contrast:
        p_max = float(np.max(warped))
        if p_max > 0.05:
            # Re-normalize peak so repeated bilinear sampling never attenuates the mask core
            warped = np.clip(warped / p_max, 0.0, 1.0)
            # Gentle sigmoid roll-off to prevent boundary erosion across long frame sequences
            warped = 1.0 / (1.0 + np.exp(-12.0 * (warped - 0.35)))
            warped = np.where(warped > 0.15, warped, 0.0)
            warped = np.clip(warped, 0.0, 1.0)
        else:
            warped = np.zeros_like(warped)

    return warped


def track_object_across_video(
    rgb_frames: List[np.ndarray],
    keyframe_mask: Optional[np.ndarray] = None,
    keyframe_idx: int = 0,
    multi_keyframes: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    """Track segmented object mask(s) across all video frames using state-of-the-art optical flow.

    Supports:
    - Multi-keyframe: Piecewise optical flow interpolation between keyed frames.
    - Single keyframe: Bidirectional forward/backward optical flow tracking.
    """
    T = len(rgb_frames)
    H, W = rgb_frames[0].shape[:2]
    masks = np.zeros((T, H, W), dtype=np.float32)

    gray_frames = [
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 and f.shape[-1] == 3 else f
        for f in rgb_frames
    ]

    # Initialize high-speed, high-accuracy DIS dense inverse search flow engine if available
    dis_engine = None
    if hasattr(cv2, "DISOpticalFlow_create"):
        dis_engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    # Handle multi-keyframes interpolation
    valid_multi = {k: v for k, v in (multi_keyframes or {}).items() if 0 <= k < T and v is not None and np.any(v)}
    if valid_multi:
        kf_indices = sorted(valid_multi.keys())
        if len(kf_indices) == 1:
            k_idx = kf_indices[0]
            keyframe_mask = valid_multi[k_idx]
            keyframe_idx = k_idx
        else:
            for idx in kf_indices:
                masks[idx] = np.clip(valid_multi[idx].astype(np.float32), 0.0, 1.0)

            # Track backward from first keyframe
            first_kf = kf_indices[0]
            for t in range(first_kf - 1, -1, -1):
                masks[t] = _warp_mask_flow(gray_frames[t + 1], gray_frames[t], masks[t + 1], dis_engine)

            # Interpolate between consecutive keyframes
            for k in range(len(kf_indices) - 1):
                t_a, t_b = kf_indices[k], kf_indices[k + 1]
                span = t_b - t_a
                if span <= 0:
                    continue

                fwd = np.zeros((span + 1, H, W), dtype=np.float32)
                fwd[0] = masks[t_a]
                for step, t in enumerate(range(t_a + 1, t_b + 1)):
                    fwd[step + 1] = _warp_mask_flow(gray_frames[t - 1], gray_frames[t], fwd[step], dis_engine)

                bwd = np.zeros((span + 1, H, W), dtype=np.float32)
                bwd[-1] = masks[t_b]
                for step, t in enumerate(range(t_b - 1, t_a - 1, -1)):
                    idx_bwd = span - 1 - step
                    bwd[idx_bwd] = _warp_mask_flow(gray_frames[t + 1], gray_frames[t], bwd[idx_bwd + 1], dis_engine)

                for step, t in enumerate(range(t_a, t_b + 1)):
                    if t == t_a or t == t_b:
                        continue
                    w_b = step / float(span)
                    w_a = 1.0 - w_b
                    blended = np.clip(w_a * fwd[step] + w_b * bwd[step], 0.0, 1.0)
                    p_max = float(np.max(blended))
                    if p_max > 0.05:
                        blended = np.clip(blended / p_max, 0.0, 1.0)
                        blended = 1.0 / (1.0 + np.exp(-12.0 * (blended - 0.35)))
                        blended = np.where(blended > 0.15, blended, 0.0)
                    masks[t] = np.clip(blended, 0.0, 1.0)

            # Track forward from last keyframe
            last_kf = kf_indices[-1]
            for t in range(last_kf + 1, T):
                masks[t] = _warp_mask_flow(gray_frames[t - 1], gray_frames[t], masks[t - 1], dis_engine)

            return np.clip(masks, 0.0, 1.0)

    # Standard single keyframe bidirectional tracking
    if keyframe_mask is None:
        return masks

    k_idx = max(0, min(T - 1, keyframe_idx))
    masks[k_idx] = keyframe_mask.astype(np.float32)

    for t in range(k_idx - 1, -1, -1):
        masks[t] = _warp_mask_flow(gray_frames[t + 1], gray_frames[t], masks[t + 1], dis_engine)

    for t in range(k_idx + 1, T):
        masks[t] = _warp_mask_flow(gray_frames[t - 1], gray_frames[t], masks[t - 1], dis_engine)

    return masks


def get_layer_depth_bounds(l: Dict[str, Any]) -> Tuple[float, float]:
    """Robustly extract (depth_min, depth_max) from a layer dict.
    
    Handles parsed 'range' strings (e.g. '0.41 – 0.82'), explicit 'depth_min'/'depth_max',
    and ignores degenerate fallback ranges like [0.0, 1.0] when a specific range string exists.
    """
    r_str = str(l.get("range", "")).replace("–", "-")
    if "-" in r_str:
        parts = [p.strip() for p in r_str.split("-") if p.strip()]
        if len(parts) == 2:
            try:
                p0, p1 = float(parts[0]), float(parts[1])
                return min(p0, p1), max(p0, p1)
            except ValueError:
                pass
    if "depth_min" in l and "depth_max" in l:
        p0, p1 = float(l["depth_min"]), float(l["depth_max"])
        return min(p0, p1), max(p0, p1)
    dr = l.get("depth_range")
    if dr and len(dr) == 2 and not (float(dr[0]) == 0.0 and float(dr[1]) == 1.0):
        p0, p1 = float(dr[0]), float(dr[1])
        return min(p0, p1), max(p0, p1)
    return 0.0, 1.0


def apply_layer_depth_transfer(
    d_vals: np.ndarray,
    d_min: float,
    d_max: float,
    mode: str = "straight",
    amount: float = 0.0,
) -> np.ndarray:
    """Non-destructively transform depth values within [d_min, d_max].

    Modes:
    - 'straight' / 'linear': d_out = clip(d_in + amount, d_min, d_max)
    - 'curve_up' / 'lift': u = (d_in - d_min) / (d_max - d_min) in [0, 1]
                          gamma = 1.0 / (1.0 + amount * 3.0)  (gamma < 1.0)
                          u_out = u ** gamma
                          d_out = d_min + u_out * (d_max - d_min)
    - 'curve_down' / 'ramp': gamma = 1.0 + amount * 3.0  (gamma > 1.0)
                            u_out = u ** gamma
                            d_out = d_min + u_out * (d_max - d_min)
    """
    span = max(1e-6, float(d_max - d_min))
    mode_clean = str(mode).lower()

    if "straight" in mode_clean or "linear" in mode_clean:
        return np.clip(d_vals + float(amount), d_min, d_max)

    u = np.clip((d_vals - d_min) / span, 0.0, 1.0)

    if "up" in mode_clean or "lift" in mode_clean:
        gamma = 1.0 / max(0.05, 1.0 + float(amount) * 3.0)
        u_out = np.power(u, gamma)
    elif "down" in mode_clean or "ramp" in mode_clean:
        gamma = 1.0 + float(amount) * 3.0
        u_out = np.power(u, gamma)
    else:
        u_out = u

    return d_min + u_out * span


def apply_depth_editor_tracking(
    rgb_frames: List[np.ndarray],
    base_depth_seq: np.ndarray,
    keyframe_masks: Dict[int, np.ndarray],
    target_depth: float = 0.18,
    city_depth: float = 0.06,
    sky_depth: float = 0.00,
    separate_sky_city: bool = False,
    tophat_snap: bool = True,
    lock_threshold: float = 0.12,
    unlocked_id: Optional[int] = None,
    layers_state: Optional[List[Dict[str, Any]]] = None,
    enable_tracking: bool = True,
    boost_mode: str = "straight",
    boost_amount: float = 0.0,
    video_stem: Optional[str] = None,
    progress=None,
) -> np.ndarray:
    """Execute the Depth Editor tracking and holdout-matte layering pipeline.

    Guarantees:
    1. IMMUTABLE HOLDOUT: Locked layers (Credits and Foreground Structure)
       are 100% bitwise preserved. Zero modifications to their base geometry.
    2. HOLDOUT SUBTRACTION: The painted object mask subtracts the locked structures
       (Object = Painted \\ LockedStructures).
    3. SELECTIVE SCULPTING ONLY: Modifies ONLY the pixels inside the painted object.
       Never overwrites surrounding background unless explicitly requested.
    """
    T, H, W = base_depth_seq.shape
    out_depth = base_depth_seq.copy().astype(np.float32)

    # Auto-load keyframe masks from disk if available
    if video_stem:
        from pathlib import Path
        masks_path = Path("source_videos") / f"{video_stem}.masks.npz"
        if masks_path.exists():
            try:
                with np.load(str(masks_path)) as m_data:
                    for k in m_data.keys():
                        if not k.startswith("L") and k.isdigit():
                            idx = int(k)
                            if idx not in keyframe_masks or keyframe_masks[idx] is None or not np.any(keyframe_masks[idx]):
                                keyframe_masks[idx] = m_data[k].astype(np.float32)
            except Exception as e:
                SF_LOG.warning(f"Could not load disk masks from {masks_path}: {e}")

    # 1. Define immutable locked holdout matte strictly from depth geometry
    # Only layers whose depth plane is strictly closer to the camera than the active object can occlude it
    locked_mask = np.zeros((T, H, W), dtype=bool)
    active_d_max = float(np.clip(target_depth, 0.01, 0.99))
    if layers_state and unlocked_id:
        for l in layers_state:
            if l.get("id") == unlocked_id:
                _, l_max = get_layer_depth_bounds(l)
                if l_max > 0.0:
                    active_d_max = max(active_d_max, l_max)
                break
        for l in layers_state:
            if l.get("id") != unlocked_id and l.get("locked", True):
                l_min, l_max = get_layer_depth_bounds(l)
                # Only a layer strictly in front of the active layer can occlude it
                if l_min >= (active_d_max - 0.02):
                    locked_mask |= ((base_depth_seq >= l_min) & (base_depth_seq <= l_max))

    # Holdout occluders strictly closer to camera than active object depth
    # (protects bridge trusses, beams, and foreground structures >= target_depth + 0.02)
    fg_occlusion_thresh = max(0.20, active_d_max + 0.02)
    locked_mask |= (base_depth_seq >= fg_occlusion_thresh)

    # 2. Preprocess keyframe masks with selected contour snapping
    use_tophat = ("top-hat" in str(tophat_snap).lower()) or (isinstance(tophat_snap, bool) and tophat_snap)
    processed_keyframes = {}
    for k_idx, k_mask in keyframe_masks.items():
        if k_mask is None or not np.any(k_mask):
            continue
        if use_tophat:
            snapped, _ = snap_mask_with_tophat_and_volume(
                rgb_frames[k_idx], k_mask, kernel_size=31, min_contrast=16.0, relief_scale=0.035
            )
        else:
            snapped, _ = apply_direct_brush_fidelity_and_volume(
                k_mask, target_shape=(H, W), feather_kernel=3, relief_scale=0.0
            )
        processed_keyframes[k_idx] = snapped

    if not processed_keyframes:
        SF_LOG.warning("[DepthEditor] No valid keyframes provided; skipping creature injection.")
        creature_masks = np.zeros((T, H, W), dtype=np.float32)
    elif len(processed_keyframes) >= max(1, T - 2):
        # User has manually painted almost the entire video (e.g. 72 of 73 frames)
        # Use exact painted frames directly; do NOT warp phantom strokes onto deliberately unpainted frames
        if progress:
            progress(0.2, desc=f"Depth Editor: Applying {len(processed_keyframes)} manual keyframes directly (preserving occluded frames)...")
        creature_masks = np.zeros((T, H, W), dtype=np.float32)
        for k_idx, k_mask in processed_keyframes.items():
            if 0 <= k_idx < T:
                creature_masks[k_idx] = k_mask
    elif enable_tracking:
        if progress:
            progress(0.2, desc=f"Depth Editor: Optical flow tracking across {len(processed_keyframes)} keyframes...")
        # Optical flow multi-keyframe tracking
        creature_masks = track_object_across_video(
            rgb_frames,
            multi_keyframes=processed_keyframes,
        )
    else:
        if progress:
            progress(0.2, desc=f"Depth Editor: Applying strictly to {len(processed_keyframes)} keyframes (tracking disabled)...")
        creature_masks = np.zeros((T, H, W), dtype=np.float32)
        for k_idx, k_mask in processed_keyframes.items():
            if 0 <= k_idx < T:
                creature_masks[k_idx] = k_mask

    # Subtract locked foreground structures on ALL frames (including keyframes)
    # The painted creature is strictly tucked BEHIND the bridge beams and pillars
    for t in range(T):
        creature_masks[t] = np.where(locked_mask[t], 0.0, creature_masks[t])

    # 3. Non-Destructive Layer Depth Transfer Function (Boost / Lift / Curve)
    # Applied strictly to the unlocked layer within its bounds [unlocked_min, unlocked_max]
    if unlocked_id is not None and layers_state and abs(boost_amount) > 1e-4:
        for l in layers_state:
            if l.get("id") == unlocked_id:
                u_min, u_max = get_layer_depth_bounds(l)
                if u_max > u_min:
                    if progress:
                        progress(0.5, desc=f"Applying {boost_mode} depth transfer to Layer {unlocked_id}...")
                    for t in range(T):
                        layer_pix = (base_depth_seq[t] >= u_min) & (base_depth_seq[t] <= u_max) & (~locked_mask[t])
                        if np.any(layer_pix):
                            d_orig = base_depth_seq[t][layer_pix]
                            out_depth[t][layer_pix] = apply_layer_depth_transfer(
                                d_orig, u_min, u_max, mode=boost_mode, amount=boost_amount
                            )
                break
    elif separate_sky_city:
        if progress:
            progress(0.5, desc="Depth Editor: Separating Cityscape and Sky floor in aperture...")
        aperture_mask = ~locked_mask
        for t in range(T):
            gray = cv2.cvtColor(rgb_frames[t], cv2.COLOR_BGR2GRAY) if rgb_frames[t].ndim == 3 else rgb_frames[t]
            sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            y_min, y_max = int(H * 0.25), int(H * 0.60)
            horizon_y = np.zeros(W, dtype=np.uint8)
            for x in range(W):
                col_grad = sobel_y[y_min:y_max, x]
                peak = y_min + int(np.argmin(col_grad))
                horizon_y[x] = np.clip(peak, 0, 255)
            horizon_smooth = cv2.medianBlur(horizon_y[:, None], 31).ravel()

            c_mask = creature_masks[t]
            for x in range(W):
                hy = int(horizon_smooth[x])
                ap_col = aperture_mask[t, :, x]
                sky_pixels = ap_col & (np.arange(H) < hy) & (c_mask[:, x] < 0.20)
                city_pixels = ap_col & (np.arange(H) >= hy) & (c_mask[:, x] < 0.20)
                out_depth[t, sky_pixels, x] = float(sky_depth)
                out_depth[t, city_pixels, x] = float(city_depth)

    # 4. Inject sculpted creatures
    if progress:
        progress(0.75, desc=f"Depth Editor: Injecting sculpted creatures at d = {target_depth:.2f}...")

    t_depth = float(np.clip(target_depth, 0.01, 0.99))
    for t in range(T):
        m = creature_masks[t]
        if np.max(m) < 0.05:
            continue

        if use_tophat:
            bin_m = (m > 0.30).astype(np.uint8)
            if np.any(bin_m):
                dist = cv2.distanceTransform(bin_m, cv2.DIST_L2, 5)
                dist_max = float(np.percentile(dist[bin_m > 0], 90)) if np.any(bin_m) else 1.0
                vol_relief = np.clip(dist / max(1.0, dist_max), 0.0, 1.0) * 0.025
            else:
                vol_relief = np.zeros((H, W), dtype=np.float32)

            c_depth = t_depth + vol_relief
            alpha = np.clip(m, 0.0, 1.0)

            # Blend creature into depth map
            curr_d = out_depth[t]
            blended = curr_d * (1.0 - alpha) + c_depth * alpha
            inject_mask = (m > 0.01) & (~locked_mask[t])
            out_depth[t] = np.where(inject_mask, blended, curr_d)
        else:
            # Solid uniform depth: 100% solid target depth, zero dark rim, zero inner gradient
            curr_d = out_depth[t]
            solid_mask = (m > 0.35) & (~locked_mask[t])
            edge_mask = (m > 0.05) & (m <= 0.35) & (~locked_mask[t])
            # Direct solid insertion
            out_depth[t] = np.where(solid_mask, t_depth, curr_d)
            if np.any(edge_mask):
                norm_alpha = (m[edge_mask] - 0.05) / 0.30
                out_depth[t][edge_mask] = curr_d[edge_mask] * (1.0 - norm_alpha) + t_depth * norm_alpha

    # 5. ABSOLUTE BITWISE GUARANTEE: Re-assert locked layers
    # Any pixel in a locked foreground layer is 100% bitwise identical to base depth
    for t in range(T):
        out_depth[t, locked_mask[t]] = base_depth_seq[t, locked_mask[t]]

    # 6. Re-assert Credit / Typography Overlay on top if credits file exists
    if video_stem:
        from pathlib import Path
        credits_npz = Path("depthmaps_videos") / f"{video_stem}_credits.npz"
        if credits_npz.exists():
            try:
                with np.load(str(credits_npz)) as c_data:
                    if "mask" in c_data:
                        c_mask = c_data["mask"]
                        for t in range(T):
                            c_t = c_mask[min(t, len(c_mask) - 1)]
                            if np.any(c_t > 0.5):
                                out_depth[t, c_t > 0.5] = 0.95
            except Exception as e:
                SF_LOG.warning(f"Could not re-assert credits from {credits_npz}: {e}")

    return out_depth


def apply_object_layer_to_depth_seq(
    depth_seq: np.ndarray,
    object_masks: np.ndarray,
    target_depth: float,
    edge_feather: int = 5,
    relief_scale: float = 0.035,
    apply_skyline_split: bool = False,
    rgb_frames: Optional[List[np.ndarray]] = None,
    sky_floor: float = 0.00,
    city_anchor: float = 0.05,
    road_floor: float = 0.30,
) -> np.ndarray:
    """Legacy compatibility wrapper for prepare_icdepth.py."""
    if apply_skyline_split and rgb_frames is not None:
        k_masks = {0: object_masks[0]} if len(object_masks) > 0 else {}
        return apply_depth_editor_tracking(
            rgb_frames=rgb_frames,
            base_depth_seq=depth_seq,
            keyframe_masks=k_masks,
            target_depth=target_depth,
            city_depth=city_anchor,
            sky_depth=sky_floor,
            separate_sky_city=True,
            tophat_snap=True,
        )

    T, H, W = depth_seq.shape
    out_depth = depth_seq.copy().astype(np.float32)
    target_d = float(np.clip(target_depth, 0.01, 0.99))

    for t in range(T):
        m = object_masks[t]
        if np.max(m) < 0.05:
            continue
        alpha = np.clip(m, 0.0, 1.0)
        curr_d = out_depth[t]
        out_depth[t] = np.where(curr_d > (target_d + 0.04), curr_d, curr_d * (1.0 - alpha) + target_d * alpha)

    return out_depth


def segment_object_keyframe(
    rgb_frame: np.ndarray,
    point_clicks: Optional[List[Tuple[int, int]]] = None,
    text_prompt: str = "",
) -> np.ndarray:
    """Segment target object on the keyframe using point clicks and/or text guidance."""
    H, W = rgb_frame.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    points = [(int(p[0]), int(p[1])) for p in (point_clicks or []) if 0 <= p[0] < W and 0 <= p[1] < H]
    if points:
        for (cx, cy) in points:
            rx, ry = 55, 65
            x0, y0 = max(0, cx - rx), max(0, cy - ry)
            x1, y1 = min(W, cx + rx), min(H, cy + ry)
            bw, bh = x1 - x0, y1 - y0

            sub_mask = np.zeros((H, W), dtype=np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect = (x0, y0, bw, bh)

            try:
                cv2.grabCut(rgb_frame, sub_mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_RECT)
                sub_mask[max(0, cy - 8):min(H, cy + 8), max(0, cx - 8):min(W, cx + 8)] = cv2.GC_FGD
                cv2.grabCut(rgb_frame, sub_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
                m = np.where((sub_mask == cv2.GC_FGD) | (sub_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
                mask |= m
            except Exception as e:
                SF_LOG.warning(f"[ObjectLayer] GrabCut point ({cx}, {cy}) failed: {e}")

    text_lower = (text_prompt or "").lower().strip()
    if not points or "fish" in text_lower or "whale" in text_lower or "animal" in text_lower:
        y_top, y_bot = int(H * 0.12), int(H * 0.45)
        x_left, x_right = int(W * 0.20), int(W * 0.65)
        roi = rgb_frame[y_top:y_bot, x_left:x_right]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (25, 25), 0)
        diff = cv2.absdiff(gray, blur)
        _, thresh = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if 300 < area < 30000:
                comp = (labels == i).astype(np.uint8)
                if not points:
                    mask[y_top:y_bot, x_left:x_right] |= comp

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clean)
    return mask


def load_layers_sidecar(video_path: str) -> Optional[Dict[str, Any]]:
    """Load {stem}.layers.json sidecar file if present."""
    if not video_path:
        return None
    vp = Path(video_path)
    parent = vp.parent
    stem = vp.stem
    if stem.endswith("_cropped"):
        stem = stem[:-8]
    sidecar = parent / f"{stem}.layers.json"
    if sidecar.is_file():
        try:
            with open(sidecar, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            SF_LOG.warning(f"[ObjectLayer] Could not read sidecar {sidecar}: {e}")
    return None


def save_layers_sidecar(
    video_path: str,
    layers_data: Dict[str, Any],
) -> Optional[Path]:
    """Save {stem}.layers.json sidecar file next to the source video."""
    if not video_path:
        return None
    vp = Path(video_path)
    parent = vp.parent
    stem = vp.stem
    if stem.endswith("_cropped"):
        stem = stem[:-8]
    sidecar = parent / f"{stem}.layers.json"
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(layers_data, f, indent=2)
        SF_LOG.info(f"[ObjectLayer] Saved layer sidecar: {sidecar}")
        return sidecar
    except Exception as e:
        SF_LOG.warning(f"[ObjectLayer] Could not write layer sidecar {sidecar}: {e}")
    return None


def prepare_cavity_frames(
    rgb_frames: List[np.ndarray],
    locked_mask: np.ndarray,
    mode: str = "inpaint",
    dilation_px: int = 5,
) -> Tuple[List[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Prepare occluder-free cavity frames for ICDepth depth estimation.

    Args:
        rgb_frames: List of BGR/RGB uint8 frames.
        locked_mask: (T, H, W) boolean mask where True indicates locked foreground occluders.
        mode: "inpaint" (fill locked occluders using multi-scale pyramid fill) or
              "crop" (crop frames to bounding box of unlocked cavity).
        dilation_px: Number of pixels to dilate the locked mask to prevent seam bleeding.

    Returns:
        cavity_frames: List of cavity RGB frames ready for ICDepth.
        crop_bbox: (ymin, ymax, xmin, xmax) if mode == "crop", else None.
    """
    T = len(rgb_frames)
    H, W = rgb_frames[0].shape[:2]

    if mode == "crop":
        unlocked_2d = (~locked_mask).any(axis=0)
        y_indices, x_indices = np.where(unlocked_2d)
        if len(y_indices) > 0 and len(x_indices) > 0:
            ymin, ymax = int(y_indices.min()), int(y_indices.max())
            xmin, xmax = int(x_indices.min()), int(x_indices.max())
            pad_y = int((ymax - ymin) * 0.08)
            pad_x = int((xmax - xmin) * 0.08)
            ymin = max(0, ymin - pad_y)
            ymax = min(H, ymax + pad_y + 1)
            xmin = max(0, xmin - pad_x)
            xmax = min(W, xmax + pad_x + 1)

            # Ensure dimensions are divisible by 2 for video encoding
            if (ymax - ymin) % 2 != 0:
                ymax = min(H, ymax + 1)
            if (xmax - xmin) % 2 != 0:
                xmax = min(W, xmax + 1)

            cropped_frames = [f[ymin:ymax, xmin:xmax].copy() for f in rgb_frames]
            return cropped_frames, (ymin, ymax, xmin, xmax)

    # Default: "inpaint" (Multi-scale pyramid fill)
    out_frames = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_px, dilation_px)) if dilation_px > 1 else None
    for t in range(T):
        frame = rgb_frames[t]
        locked_t = locked_mask[t]
        if kernel is not None and np.any(locked_t):
            locked_t = cv2.dilate(locked_t.astype(np.uint8), kernel).astype(bool)

        unlocked_rgb = frame.copy().astype(np.float32)
        unlocked_rgb[locked_t] = 0.0

        pyr = [unlocked_rgb]
        m_pyr = [(~locked_t).astype(np.float32)[:, :, None]]

        for _ in range(4):
            pyr.append(cv2.pyrDown(pyr[-1]))
            down_m = cv2.pyrDown(m_pyr[-1])
            if down_m.ndim == 2:
                down_m = down_m[:, :, None]
            m_pyr.append(down_m)

        top_img = pyr[-1]
        top_mask = m_pyr[-1]
        safe_top_mask = np.maximum(top_mask, 1e-4)
        curr = np.clip(top_img / safe_top_mask, 0.0, 255.0)

        for i in range(len(pyr) - 2, -1, -1):
            up = cv2.pyrUp(curr, dstsize=(pyr[i].shape[1], pyr[i].shape[0]))
            if up.ndim == 2:
                up = up[:, :, None]
            m = m_pyr[i]
            curr = np.where(m > 0.05, pyr[i], up)

        out_frames.append(curr.astype(np.uint8))

    return out_frames, None


def run_icdepth_on_frames(
    frames: List[np.ndarray],
    fps: float = 24.0,
    max_res: int = 512,
    num_inference_steps: int = 5,
    cfg_scale: float = 1.0,
    engine: str = "ICDepth",
    progress=None,
) -> np.ndarray:
    """Run specified depth engine (ICDepth or DepthCrafter) on a list of BGR numpy frames."""
    import tempfile

    T = len(frames)
    H, W = frames[0].shape[:2]

    fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (W, H))
        for f in frames:
            writer.write(f)
        writer.release()

        if "depthcrafter" in engine.lower():
            from m2svid.prepare_depthcrafter import run_depthcrafter_depth
            depth_array = run_depthcrafter_depth(
                temp_video_path,
                process_res=max_res,
                guidance_scale=cfg_scale,
                num_inference_steps=num_inference_steps,
                progress=progress,
            )
            d_min = float(depth_array.min())
            d_max = float(depth_array.max())
            denom = max(1e-5, d_max - d_min)
            depth_array = np.clip((depth_array - d_min) / denom, 0.0, 1.0).astype(np.float32)
            if depth_array.shape[1:] != (H, W):
                depth_array = np.stack([
                    cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
                    for d in depth_array
                ]).astype(np.float32)
            return depth_array
        else:
            from m2svid.prepare_icdepth import run_icdepth_depth
            depth_array, _ = run_icdepth_depth(
                temp_video_path,
                max_res=max_res,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                void_title_recovery=False,
                volumetric_splash_recovery=False,
                credit_pop=False,
                progress=progress,
            )
            if depth_array.shape[1:] != (H, W):
                depth_array = np.stack([
                    cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
                    for d in depth_array
                ]).astype(np.float32)
            return depth_array
    finally:
        try:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except Exception:
            pass


def apply_option_b_cavity_pass(
    rgb_frames: List[np.ndarray],
    base_depth_seq: np.ndarray,
    unlocked_id: int = 3,
    layers_state: Optional[List[Dict[str, Any]]] = None,
    cavity_mode: str = "inpaint",
    engine: str = "DepthCrafter",
    icd_steps: int = 5,
    icd_cfg: float = 1.0,
    icd_res: int = 512,
    target_depth: Optional[float] = None,
    fps: float = 24.0,
    progress=None,
) -> np.ndarray:
    """Option B: Occluder-Removed Cavity Pass (DepthCrafter / ICDepth Cavity).

    1. Removes all locked foreground occluders (road, bridge trusses, cars) using
       neutral background pyramid inpainting or cavity bounding box cropping.
    2. Runs DepthCrafter or ICDepth on the occluder-free cavity video with full dynamic range.
    3. Normalizes the resulting depth and maps it strictly into the unlocked layer's
       depth bounds [depth_min, depth_max].
    4. ABSOLUTE HOLDOUT: Re-asserts locked layers bitwise (diff == 0.0).
    """
    T, H, W = base_depth_seq.shape

    # Determine unlocked layer depth bounds
    depth_min = 0.00
    depth_max = 0.21
    if layers_state:
        for l in layers_state:
            if l.get("id") == unlocked_id:
                depth_min, depth_max = get_layer_depth_bounds(l)
                break

    # Identify unlocked pixels vs locked occluder pixels
    unlocked_mask = (base_depth_seq >= depth_min) & (base_depth_seq <= depth_max)
    # Only layers strictly in front of the active layer are occluders
    locked_mask = np.zeros((T, H, W), dtype=bool)
    if layers_state:
        for l in layers_state:
            if l.get("id") != unlocked_id and l.get("locked", True):
                l_min, l_max = get_layer_depth_bounds(l)
                if l_min >= (depth_max - 0.02):
                    locked_mask |= ((base_depth_seq >= l_min) & (base_depth_seq <= l_max))
    locked_mask |= (base_depth_seq >= max(0.40, min(depth_max, 0.70)))

    if progress:
        progress(0.1, desc=f"Option B: Removing locked occluders for Layer {unlocked_id} ({cavity_mode})...")

    cavity_frames, crop_bbox = prepare_cavity_frames(
        rgb_frames=rgb_frames,
        locked_mask=locked_mask,
        mode="crop" if "crop" in cavity_mode.lower() else "inpaint",
        dilation_px=5,
    )

    engine_name = "DepthCrafter" if "depthcrafter" in engine.lower() else "ICDepth"
    if progress:
        progress(0.2, desc=f"Option B: Running {engine_name} on occluder-free cavity ({len(cavity_frames)} frames)...")

    raw_cavity_depth = run_icdepth_on_frames(
        frames=cavity_frames,
        fps=fps,
        max_res=icd_res,
        num_inference_steps=icd_steps,
        cfg_scale=icd_cfg,
        engine=engine,
        progress=progress,
    )

    # If cropped, embed cavity depth back to full frame size
    if crop_bbox is not None:
        ymin, ymax, xmin, xmax = crop_bbox
        full_cavity_depth = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            c_slice = cv2.resize(raw_cavity_depth[t], (xmax - xmin, ymax - ymin), interpolation=cv2.INTER_LINEAR)
            full_cavity_depth[t, ymin:ymax, xmin:xmax] = c_slice
        raw_cavity_depth = full_cavity_depth
    elif raw_cavity_depth.shape[1:] != (H, W):
        interp = cv2.INTER_AREA if (W <= raw_cavity_depth.shape[2] and H <= raw_cavity_depth.shape[1]) else cv2.INTER_LINEAR
        raw_cavity_depth = np.stack([
            cv2.resize(raw_cavity_depth[t], (W, H), interpolation=interp)
            for t in range(len(raw_cavity_depth))
        ]).astype(np.float32)

    # Normalize depth within the unlocked cavity region
    cav_vals = raw_cavity_depth[unlocked_mask]
    if len(cav_vals) > 0:
        p2 = float(np.percentile(cav_vals, 2.0))
        p98 = float(np.percentile(cav_vals, 98.0))
        denom = max(1e-4, p98 - p2)
        norm_d = np.clip((raw_cavity_depth - p2) / denom, 0.0, 1.0)
    else:
        norm_d = np.clip(raw_cavity_depth, 0.0, 1.0)

    # Remap into unlocked layer depth interval [depth_min, depth_max]
    mapped_d = depth_min + norm_d * (depth_max - depth_min)

    out_depth = base_depth_seq.copy().astype(np.float32)
    out_depth[unlocked_mask] = mapped_d[unlocked_mask]

    # ABSOLUTE BITWISE HOLDOUT GUARANTEE:
    # Not a single pixel outside the unlocked layer is modified
    out_depth[locked_mask] = base_depth_seq[locked_mask]

    return out_depth


def apply_option_a_keyframe_propagation(
    rgb_frames: List[np.ndarray],
    base_depth_seq: np.ndarray,
    keyframe_masks: Dict[int, np.ndarray],
    unlocked_id: int = 3,
    layers_state: Optional[List[Dict[str, Any]]] = None,
    target_depth: float = 0.18,
    tophat_snap: bool = True,
    icd_steps: int = 5,
    icd_cfg: float = 1.0,
    icd_res: int = 512,
    fps: float = 24.0,
    progress=None,
) -> np.ndarray:
    """Option A: In-Context Keyframe Propagation (ICDepth DiT).

    1. Preprocesses user keyframe masks with Top-Hat contour snapping and volumetric relief.
    2. Tracks object contours across frames using optical flow.
    3. Runs ICDepth DiT inference on the sequence to extract temporally coherent depth structures.
    4. Blends the user keyframe relief with the DiT depth inside the unlocked layer bounds [depth_min, depth_max].
    5. ABSOLUTE HOLDOUT: Re-asserts locked layers bitwise (diff == 0.0).
    """
    T, H, W = base_depth_seq.shape

    # Determine unlocked layer depth bounds
    depth_min = 0.00
    depth_max = 0.21
    if layers_state:
        for l in layers_state:
            if l.get("id") == unlocked_id:
                depth_min, depth_max = get_layer_depth_bounds(l)
                break

    unlocked_mask = (base_depth_seq >= depth_min) & (base_depth_seq <= depth_max)
    # Only layers strictly in front of the active layer are occluders
    locked_mask = np.zeros((T, H, W), dtype=bool)
    if layers_state:
        for l in layers_state:
            if l.get("id") != unlocked_id and l.get("locked", True):
                l_min, l_max = get_layer_depth_bounds(l)
                if l_min >= (depth_max - 0.02):
                    locked_mask |= ((base_depth_seq >= l_min) & (base_depth_seq <= l_max))
    locked_mask |= (base_depth_seq >= max(0.40, min(depth_max, 0.70)))

    # Process keyframes with selected contour snapping & holdout subtraction
    use_tophat = ("top-hat" in str(tophat_snap).lower()) or (isinstance(tophat_snap, bool) and tophat_snap)
    processed_keyframes = {}
    for k_idx, k_mask in keyframe_masks.items():
        if k_mask is None or not np.any(k_mask):
            continue
        if use_tophat:
            snapped, _ = snap_mask_with_tophat_and_volume(
                rgb_frames[k_idx], k_mask, kernel_size=31, min_contrast=16.0, relief_scale=0.035
            )
        else:
            snapped, _ = apply_direct_brush_fidelity_and_volume(
                k_mask, target_shape=(H, W), feather_kernel=5, relief_scale=0.035
            )
        # Holdout subtraction
        snapped = np.where(locked_mask[k_idx], 0.0, snapped)
        processed_keyframes[k_idx] = snapped

    # Track object masks across frames
    if processed_keyframes:
        if progress:
            progress(0.15, desc=f"Option A: Tracking keyframe masks across {T} frames...")
        tracked_masks = track_object_across_video(rgb_frames, multi_keyframes=processed_keyframes)
    else:
        tracked_masks = np.zeros((T, H, W), dtype=np.float32)

    # Holdout subtraction across all frames
    for t in range(T):
        tracked_masks[t] = np.where(locked_mask[t], 0.0, tracked_masks[t])

    # Run ICDepth DiT inference for temporal depth structures
    if progress:
        progress(0.3, desc=f"Option A: Running ICDepth DiT propagation ({icd_steps} steps, res={icd_res})...")

    dit_depth = run_icdepth_on_frames(
        frames=rgb_frames,
        fps=fps,
        max_res=icd_res,
        num_inference_steps=icd_steps,
        cfg_scale=icd_cfg,
        progress=progress,
    )
    if dit_depth.shape[1:] != (H, W):
        interp = cv2.INTER_AREA if (W <= dit_depth.shape[2] and H <= dit_depth.shape[1]) else cv2.INTER_LINEAR
        dit_depth = np.stack([
            cv2.resize(dit_depth[t], (W, H), interpolation=interp)
            for t in range(len(dit_depth))
        ]).astype(np.float32)

    if progress:
        progress(0.85, desc="Option A: Fusing keyframe relief and DiT depth into unlocked layer...")

    # Normalize DiT depth within unlocked region
    unlocked_dit_vals = dit_depth[unlocked_mask]
    if len(unlocked_dit_vals) > 0:
        p2 = float(np.percentile(unlocked_dit_vals, 2.0))
        p98 = float(np.percentile(unlocked_dit_vals, 98.0))
        denom = max(1e-4, p98 - p2)
        norm_dit = np.clip((dit_depth - p2) / denom, 0.0, 1.0)
    else:
        norm_dit = np.clip(dit_depth, 0.0, 1.0)

    # Base cavity depth mapped to [depth_min, target_depth]
    t_d = float(np.clip(target_depth, depth_min, depth_max))
    bg_mapped = depth_min + norm_dit * (t_d - depth_min)

    out_depth = base_depth_seq.copy().astype(np.float32)
    out_depth[unlocked_mask] = bg_mapped[unlocked_mask]

    # Inject sculpted objects with volumetric relief and DiT structure fusion
    for t in range(T):
        m = tracked_masks[t]
        if np.max(m) < 0.05:
            continue

        bin_m = (m > 0.30).astype(np.uint8)
        if np.any(bin_m):
            dist = cv2.distanceTransform(bin_m, cv2.DIST_L2, 5)
            dist_max = float(np.percentile(dist[bin_m > 0], 90)) if np.any(bin_m) else 1.0
            vol_relief = np.clip(dist / max(1.0, dist_max), 0.0, 1.0) * 0.035
        else:
            vol_relief = np.zeros((H, W), dtype=np.float32)

        # Fuse volumetric relief with local DiT surface detail
        dit_local = (norm_dit[t] - float(norm_dit[t, m > 0.3].mean() if np.any(m > 0.3) else 0.5)) * 0.02
        creature_depth = t_d + vol_relief + dit_local
        creature_depth = np.clip(creature_depth, depth_min, depth_max)

        alpha = np.clip(m, 0.0, 1.0)
        curr_d = out_depth[t]
        blended = curr_d * (1.0 - alpha) + creature_depth * alpha
        out_depth[t] = np.where(m > 0.01, blended, curr_d)

    # ABSOLUTE BITWISE HOLDOUT GUARANTEE
    out_depth[locked_mask] = base_depth_seq[locked_mask]

    return out_depth
