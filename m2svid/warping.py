"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from m2svid.utils.video_utils import open_ffmpeg_process, read_frames_in_batches_ffmpeg, get_video_fps
from m2svid.warping.warping import scatter_image

import numpy as np
import tqdm
from pathlib import Path
import ffmpeg
import os
import cv2
import concurrent.futures


def process_video_with_depth(
    video_path,
    depth_path,
    output_path_reprojected=None,
    output_path_mask=None,
    disparity_scale=None,
    disparity_perc=None,
    batch_size=10,
    convergence_point=0.5,
    return_in_memory=False,
):
    relative_depth_data = np.load(depth_path)
    relative_depth = relative_depth_data['depth']
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    if disparity_perc is not None:
        eff_perc = disparity_perc / 1000.0 if disparity_perc > 1.0 else disparity_perc
        disparity_scale = int(width * eff_perc)

    fps = get_video_fps(video_path, probe)

    ffmpeg_process_reprojected = None
    ffmpeg_process_mask = None

    all_left = [] if return_in_memory else None
    all_reproj = [] if return_in_memory else None
    all_masks = [] if return_in_memory else None

    # Number of worker threads for parallel frame warping
    max_workers = min(max(batch_size, 1), os.cpu_count() or 4)

    def _warp_single(pair):
        lf, disp = pair
        reproj_img, inpaint_mask, _ = scatter_image(
            lf, disp, direction=-1, scale_factor=1, reproject_depth=False
        )
        return reproj_img, inpaint_mask

    for i, left_frames in enumerate(
        tqdm.tqdm(
            read_frames_in_batches_ffmpeg(video_path, batch_size, width, height),
            total=int(relative_depth.shape[0] // batch_size),
        )
    ):
        depth_batch = relative_depth[i * batch_size : (i + 1) * batch_size]
        depth_batch = np.array([
            cv2.resize(depth_frame, (width, height), interpolation=cv2.INTER_CUBIC)
            for depth_frame in depth_batch
        ])

        # Introduce a Convergence Point (Zero Parallax Setting)
        # Puts the main subject closer to the screen plane, treating the screen like a window
        disparities = (depth_batch - convergence_point) * disparity_scale

        # Multi-threaded warping across frames in batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_warp_single, zip(left_frames, disparities)))

        reprojected_right_videos = np.stack([r[0] for r in results], axis=0)
        reprojected_right_masks = np.stack([r[1] for r in results], axis=0)

        if return_in_memory:
            all_left.append(np.array(left_frames))
            all_reproj.append(reprojected_right_videos)
            all_masks.append(reprojected_right_masks)

        if output_path_reprojected is not None and output_path_mask is not None:
            if ffmpeg_process_reprojected is None:
                _, h_out, w_out, _ = reprojected_right_videos.shape
                ffmpeg_process_reprojected = open_ffmpeg_process(
                    output_path_reprojected, w_out, h_out, fps
                )
                ffmpeg_process_mask = open_ffmpeg_process(
                    output_path_mask, w_out, h_out, fps, grayscale=True, no_compression=True
                )

            for reprojected_frame, mask_frame in zip(
                reprojected_right_videos, reprojected_right_masks
            ):
                ffmpeg_process_reprojected.stdin.write(reprojected_frame.tobytes())
                ffmpeg_process_mask.stdin.write(mask_frame.tobytes())

    if ffmpeg_process_reprojected is not None:
        ffmpeg_process_reprojected.stdin.close()
        ret1 = ffmpeg_process_reprojected.wait()
        ffmpeg_process_mask.stdin.close()
        ret2 = ffmpeg_process_mask.wait()
        if ret1 != 0 or ret2 != 0:
            raise RuntimeError(f"FFmpeg warping process failed (reprojected={ret1}, mask={ret2})")

    if return_in_memory and all_reproj:
        reproj_arr = np.concatenate(all_reproj, axis=0)
        masks_arr = np.concatenate(all_masks, axis=0)
        left_arr = np.concatenate(all_left, axis=0)
        return reproj_arr, masks_arr, left_arr, fps

    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames with depth data to generate reprojected videos and masks.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--depth_path", type=str, required=True, help="Path to the depth numpy file.")
    parser.add_argument("--output_path_reprojected", type=str, required=True, help="Path to save the reprojected output video.")
    parser.add_argument("--output_path_mask", type=str, required=True, help="Path to save the mask output video.")
    parser.add_argument("--disparity_scale", type=float, default=None, help="List of disparity scales to apply.")
    parser.add_argument("--disparity_perc", type=float, default=None, help="List of disparity scales to apply.")

    args = parser.parse_args()

    assert (args.disparity_scale is None) or (args.disparity_perc is None)
    assert (args.disparity_scale is not None) or (args.disparity_perc is not None)

    os.makedirs(os.path.dirname(args.output_path_reprojected), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_path_mask), exist_ok=True)

    process_video_with_depth(
        args.video_path,
        args.depth_path,
        args.output_path_reprojected,
        args.output_path_mask,
        disparity_scale=args.disparity_scale,
        disparity_perc=args.disparity_perc,
    )
