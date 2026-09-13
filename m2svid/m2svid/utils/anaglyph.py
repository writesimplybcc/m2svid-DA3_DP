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


def to_pil(image):
    image = Image.fromarray(image)
    return image


def make_anaglyph_image(left, right):
    if isinstance(left, Image.Image):
        left = np.array(left)
    if isinstance(right, Image.Image):
        right = np.array(right)
        
    # Standard Red-Cyan Anaglyph: Left Eye Red, Right Eye Green & Blue
    anaglyph = np.empty_like(left)
    anaglyph[..., 0] = left[..., 0]   # Red from Left Eye
    anaglyph[..., 1] = right[..., 1]  # Green from Right Eye
    anaglyph[..., 2] = right[..., 2]  # Blue from Right Eye
    return Image.fromarray(anaglyph)


def make_anaglyph_video(left_video, right_video, unnormalized_videos=False):
    # If tensors [C, T, H, W]
    if isinstance(left_video, torch.Tensor):
        if unnormalized_videos:
            left_video = (((left_video + 1.0) / 2.0).clamp(0, 1) * 255.0).to(torch.uint8)
            right_video = (((right_video + 1.0) / 2.0).clamp(0, 1) * 255.0).to(torch.uint8)
        else:
            left_video = left_video.to(torch.uint8)
            right_video = right_video.to(torch.uint8)
            
        anaglyph = torch.empty_like(left_video)
        anaglyph[0] = left_video[0]   # Red from Left
        anaglyph[1] = right_video[1]  # Green from Right
        anaglyph[2] = right_video[2]  # Blue from Right
        return anaglyph.float() / 127.5 - 1.0 if unnormalized_videos else anaglyph

    if unnormalized_videos:
        left_video = (((left_video + 1) / 2).clip(0, 1) * 255).astype(np.uint8)
        right_video = (((right_video + 1) / 2).clip(0, 1) * 255).astype(np.uint8)

    anaglyph = np.empty_like(left_video)
    anaglyph[..., 0] = left_video[..., 0]
    anaglyph[..., 1] = right_video[..., 1]
    anaglyph[..., 2] = right_video[..., 2]
    return anaglyph

