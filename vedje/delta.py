"""Feature-change supervision for the cache (Section 3.3).

FixedQueryFutureFeaturePredictor: training-only predictor F_omega (Eq. 3).
compute_delta_loss: L_delta (Eq. 4).
valid_pairs, delta_targets: the (t, h) pairs and the targets X_{t+h} - X_t that
compute_delta_loss uses, with zero-based frame indices t.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class FixedQueryFutureFeaturePredictor(nn.Module):
    """F_omega: predict patch-level deltas X_{t+h} - X_t from Z_t.

    Queries are fixed sinusoidal vectors indexed by (h, p): the first half of
    bert_dim encodes the horizon h, the second half the patch position p. A
    2-layer transformer decoder cross-attends to one frame's cached tokens and
    a linear layer projects each output to vision_dim.
    """

    def __init__(
        self,
        horizons: Sequence[int] = (3,),
        patches_per_frame: int = 256,
        bert_dim: int = 384,
        vision_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizons = tuple(int(h) for h in horizons)
        self.patches_per_frame = patches_per_frame
        self.vision_dim = vision_dim

        queries = self._build_sinusoidal_queries(self.horizons, patches_per_frame, bert_dim)
        self.register_buffer('fixed_queries', queries.unsqueeze(0))  # (1, |H|*P, bert_dim)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=bert_dim,
                nhead=num_heads,
                dim_feedforward=bert_dim * 4,
                dropout=dropout,
                activation="gelu",  # Appendix A.2
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(bert_dim)
        self.output_proj = nn.Linear(bert_dim, vision_dim)

    @staticmethod
    def _build_sinusoidal_queries(
        horizons: Sequence[int], patches_per_frame: int, bert_dim: int
    ) -> torch.Tensor:
        half_dim = bert_dim // 2
        queries = torch.zeros(len(horizons) * patches_per_frame, bert_dim)
        for hi, h in enumerate(horizons):
            for p in range(patches_per_frame):
                i = hi * patches_per_frame + p
                for j in range(half_dim):
                    denom = 10000.0 ** (2 * (j // 2) / half_dim)
                    fn = math.sin if j % 2 == 0 else math.cos
                    queries[i, j] = fn(h / denom)
                    queries[i, half_dim + j] = fn(p / denom)
        return queries

    def forward(self, compressed_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            compressed_frame: (B, M, bert_dim), the cached tokens Z_t of one frame
        Returns:
            (B, |H|, patches_per_frame, vision_dim)
        """
        B = compressed_frame.shape[0]
        q = self.fixed_queries.expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, compressed_frame)
        out = self.output_proj(self.output_norm(q))
        return out.view(B, len(self.horizons), self.patches_per_frame, self.vision_dim)


def valid_pairs(num_frames: int, horizons: Sequence[int]) -> List[Tuple[int, int]]:
    """The valid pairs Omega of Eq. 4 with zero-based frame indices.

    Omega = {(t, h) : t + h <= T, h in H} with frames numbered 1 to T is the set
    of zero-based pairs with t + h < T. The order (t first, then the horizons in
    the given order) is the order in which compute_delta_loss visits them.
    """
    return [(t, int(h)) for t in range(num_frames) for h in horizons if t + int(h) < num_frames]


def delta_targets(
    features: torch.Tensor,
    num_frames: int,
    patches_per_frame: int,
    horizons: Sequence[int],
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Frozen-feature targets Delta_{t,h} = X_{t+h} - X_t for every valid pair.

    Args:
        features: (B, T*P, vision_dim) frozen backbone patch features, frame-major
    Returns:
        {(t, h): (B, P, vision_dim)} with zero-based t, the targets of compute_delta_loss
    """
    B = features.shape[0]
    raw = features.view(B, num_frames, patches_per_frame, -1).detach()
    return {(t, h): raw[:, t + h] - raw[:, t] for t, h in valid_pairs(num_frames, horizons)}


def compute_delta_loss(
    predictor: FixedQueryFutureFeaturePredictor,
    per_frame_compressed: torch.Tensor,
    raw_patches: torch.Tensor,
    num_frames: int,
    patches_per_frame: int,
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    """L_delta (Eq. 4).

    Mean squared error between the predicted delta and the frozen-backbone
    target Delta_{t,h} = X_{t+h} - X_t, averaged over patches and the visual
    feature dimension, then over the valid pairs (valid_pairs: t + h < T for
    zero-based t).

    Args:
        per_frame_compressed: (B, T, M, bert_dim) cached tokens per frame
        raw_patches: (B, T*P, vision_dim) frozen backbone patch features
    Returns:
        (loss, {"delta_h<h>": mean pair loss per horizon})
    """
    B = per_frame_compressed.shape[0]
    T, P = num_frames, patches_per_frame
    device = per_frame_compressed.device
    raw = raw_patches.view(B, T, P, -1).detach()

    total_loss = torch.zeros((), device=device)
    num_valid = 0
    per_h = {h: [] for h in predictor.horizons}

    for t in range(T):
        valid = [(hi, h) for hi, h in enumerate(predictor.horizons) if t + h < T]
        if not valid:
            continue
        pred = predictor(per_frame_compressed[:, t])  # (B, |H|, P, vision_dim)
        for hi, h in valid:
            target = raw[:, t + h] - raw[:, t]
            pair_loss = F.mse_loss(pred[:, hi].float(), target.float())
            total_loss = total_loss + pair_loss
            num_valid += 1
            per_h[h].append(pair_loss.item())

    if num_valid > 0:
        total_loss = total_loss / num_valid
    diag = {f"delta_h{h}": sum(v) / len(v) for h, v in per_h.items() if v}
    return total_loss, diag
