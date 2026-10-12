from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vedje.features import extract_patch_features, load_backbone


class _StandInBackbone(torch.nn.Module):
    """Maps (B, T, 3, H, W) frames to (B, T*4, 6) tokens, like VideoPrism's frame-major output."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 6)

    def forward(self, pixel_values_videos):
        B, T = pixel_values_videos.shape[:2]
        pooled = pixel_values_videos.mean(dim=(-1, -2))  # (B, T, 3)
        tokens = self.proj(pooled).unsqueeze(2).expand(B, T, 4, 6).reshape(B, T * 4, 6)
        return SimpleNamespace(last_hidden_state=tokens)


def test_extract_patch_features_outputs():
    backbone = _StandInBackbone()
    frames = torch.rand(5, 3, 8, 8)  # one video without a batch dimension
    local_patches, v_global = extract_patch_features(backbone, frames)
    assert local_patches.shape == (1, 5 * 4, 6)
    torch.testing.assert_close(v_global, F.normalize(local_patches.mean(dim=1), dim=-1))
    torch.testing.assert_close(v_global.norm(dim=-1), torch.ones(1))


@pytest.mark.network
def test_videoprism_backbone_features():
    backbone = load_backbone()
    local_patches, v_global = extract_patch_features(backbone, torch.rand(1, 16, 3, 288, 288))
    assert local_patches.shape == (1, 16 * 256, 768)
    assert v_global.shape == (1, 768)
    assert torch.isfinite(local_patches).all()
