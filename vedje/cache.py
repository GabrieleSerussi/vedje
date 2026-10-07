"""Frame-indexed cache compressor (Section 3.2).

FramewiseCrossAttentionCompressor: M learned queries per frame attend to that
frame's patches only (Eq. 1); frames are concatenated in temporal order to form
the cache Z(v) (Eq. 2).

cache_bytes: tensor payload of a cache, 2 * T * M * d bytes in BF16.
"""

import torch
from torch import nn


class _CrossAttentionPooling(nn.Module):
    """Learnable queries cross-attending to patch tokens (2-layer decoder)."""

    def __init__(self, num_queries, vision_dim, bert_dim, num_heads=8,
                 num_layers=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, bert_dim) * 0.02)
        self.kv_proj = nn.Linear(vision_dim, bert_dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=bert_dim, nhead=num_heads,
                dim_feedforward=bert_dim * 4, dropout=dropout,
                activation="gelu", batch_first=True,  # Appendix A.2
            )
            for _ in range(num_layers)
        ])

    def forward(self, patch_tokens):
        B = patch_tokens.shape[0]
        kv = self.kv_proj(patch_tokens)
        q = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, kv)
        return q


class FramewiseCrossAttentionCompressor(nn.Module):
    """Per-frame cross-attention compression.

    Each frame is compressed independently to num_queries_per_frame (M) tokens;
    the output is flattened to (B, T * M, bert_dim) in temporal order, so the
    tokens of frame t occupy rows t * M to t * M + M - 1.
    """

    def __init__(
        self,
        num_queries_per_frame: int = 4,
        vision_dim: int = 768,
        bert_dim: int = 384,
        patches_per_frame: int = 256,
        max_frames: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries_per_frame = num_queries_per_frame
        self.num_queries = max_frames * num_queries_per_frame
        self.patches_per_frame = patches_per_frame
        self.max_frames = max_frames

        self._inner = _CrossAttentionPooling(
            num_queries=num_queries_per_frame,
            vision_dim=vision_dim,
            bert_dim=bert_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, T*P, vision_dim), frame-major
        Returns:
            (B, T*M, bert_dim): the cache Z(v)
        """
        B = patch_tokens.shape[0]
        P = self.patches_per_frame
        T = patch_tokens.shape[1] // P
        per_frame = patch_tokens.view(B * T, P, -1)
        compressed = self._inner(per_frame)  # (B*T, M, bert_dim)
        return compressed.view(B, T * self.num_queries_per_frame, -1)


def cache_bytes(num_frames: int, tokens_per_frame: int, dim: int,
                bytes_per_element: int = 2) -> int:
    """Tensor payload in bytes of num_frames x tokens_per_frame vectors of size dim.

    With the BF16 default of two bytes per element this is B_cache = 2 T M d_l
    (Section 3.2): 49,152 bytes (48 KiB) for 16 x 4 tokens at d_l = 384 and
    12,288 bytes (12 KiB) for 16 x 1. File-container and index metadata are not
    counted.
    """
    return num_frames * tokens_per_frame * dim * bytes_per_element
