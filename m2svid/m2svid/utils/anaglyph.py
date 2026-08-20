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

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import imageio
import torch


def make_anaglyph_video(left_video, right_video, unnormalized_videos=False):
    """
    Fast, vectorized generation of Optimized Red/Cyan Anaglyph to prevent retinal rivalry
    on highly saturated colors (e.g. pure red or pure blue objects).
    """
    if unnormalized_videos:
        device = left_video.device
        # Convert [-1, 1] tensor to [0, 255] numpy [T, H, W, C]
        left_np = left_video.cpu().numpy().transpose(1, 2, 3, 0)
        right_np = right_video.cpu().numpy().transpose(1, 2, 3, 0)
        left_np = (((left_np + 1) / 2).clip(0, 1) * 255).astype(np.float32)
        right_np = (((right_np + 1) / 2).clip(0, 1) * 255).astype(np.float32)
    else:
        left_np = np.array(left_video, dtype=np.float32)
        right_np = np.array(right_video, dtype=np.float32)
        
    # Optimized Anaglyph formula to reduce retinal rivalry:
    # Left Eye (Red) = Luminance of Left View
    # Right Eye (Cyan) = Green and Blue of Right View
    
    r1, g1, b1 = left_np[..., 0], left_np[..., 1], left_np[..., 2]
    r2, g2, b2 = right_np[..., 0], right_np[..., 1], right_np[..., 2]
    
    # Calculate luminance for the left view (Red channel output)
    out_r = (0.299 * r1 + 0.587 * g1 + 0.114 * b1).clip(0, 255)
    
    # Keep right view colors for the right eye (Green/Blue channels output)
    out_g = g2
    out_b = b2
    
    output_frames = np.stack([out_r, out_g, out_b], axis=-1)

    if unnormalized_videos:
        # Scale back to [-1, 1]
        output_frames = (output_frames / 255.0) * 2.0 - 1.0
        output_frames = torch.from_numpy(output_frames.transpose(3, 0, 1, 2)).to(device)

    return output_frames

