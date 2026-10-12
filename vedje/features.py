"""Frozen VideoPrism-B patch features, computed once per video at indexing time.

    frames = load_frames("video.mp4")                 # (16, 3, 288, 288) in [0, 1]
    backbone = load_backbone()
    local_patches, v_global = extract_patch_features(backbone, frames)
    # local_patches: (1, 16 * 256, 768) patch tokens X_1..X_T, frame-major
    # v_global:      (1, 768) L2-normalized mean of the patch tokens

scripts/extract_features.py stores both in bf16, one file per video.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from vedje.video import load_video_frames

DEFAULT_BACKBONE = "MHRDYN7/videoprism-base-f16r288"
FRAME_SIZE = 288
NUM_FRAMES = 16
PATCHES_PER_FRAME = 256


def load_backbone(path: str = DEFAULT_BACKBONE, attn_implementation: str = "eager",
                  device=None):
    """The frozen VideoPrism-B encoder (stock transformers >= 5.13), in eval mode.

    device defaults to cuda when available, otherwise cpu.
    """
    from transformers import VideoPrismVisionModel
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = VideoPrismVisionModel.from_pretrained(
        path, attn_implementation=attn_implementation,
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def load_frames(source: str, num_frames: int = NUM_FRAMES) -> torch.Tensor:
    """Frames as scripts/extract_features.py reads them: (T, 3, 288, 288) in [0, 1].

    source is a video file or a folder of frame images (sorted by name).
    """
    return load_video_frames(source, num_frames, size=FRAME_SIZE, normalize=False)


@torch.inference_mode()
def extract_patch_features(backbone, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Patch features and global feature of a batch of videos.

    Args:
        backbone: VideoPrism encoder from load_backbone
        frames: (B, T, 3, 288, 288) or (T, 3, 288, 288) in [0, 1]
    Returns:
        local_patches: (B, T*256, 768) float32, frame-major
        v_global: (B, 768) float32, the L2-normalized mean of the patch tokens
    """
    if frames.dim() == 4:
        frames = frames.unsqueeze(0)
    device = next(backbone.parameters()).device
    patch_tokens = backbone(pixel_values_videos=frames.to(device).float()).last_hidden_state
    v_global = F.normalize(patch_tokens.mean(dim=1).float(), dim=-1)
    return patch_tokens, v_global
