import torch
import torch.nn.functional as F

from vedje.delta import FixedQueryFutureFeaturePredictor, compute_delta_loss, delta_targets, valid_pairs


def test_valid_pairs():
    assert len(valid_pairs(16, [3])) == 13
    assert valid_pairs(16, [3])[0] == (0, 3) and valid_pairs(16, [3])[-1] == (12, 3)
    assert valid_pairs(4, [1, 2]) == [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1)]
    assert valid_pairs(3, [3]) == []
    assert all(t + h < 16 for t, h in valid_pairs(16, [2, 3, 4]))
    assert len(valid_pairs(16, [2, 3, 4])) == 14 + 13 + 12


def test_delta_targets_are_future_minus_current():
    T, P, D = 6, 4, 5
    x = torch.randn(2, T * P, D)
    targets = delta_targets(x, num_frames=T, patches_per_frame=P, horizons=[1, 3])
    assert list(targets) == valid_pairs(T, [1, 3])
    frames = x.view(2, T, P, D)
    for (t, h), target in targets.items():
        assert target.shape == (2, P, D)
        torch.testing.assert_close(target, frames[:, t + h] - frames[:, t])
    # frame t occupies rows t*P to t*P+P-1 of the flattened features
    torch.testing.assert_close(targets[(0, 1)], x[:, P:2 * P] - x[:, 0:P])


def test_compute_delta_loss_matches_manual_computation():
    torch.manual_seed(0)
    T, M, P, D, d = 7, 2, 4, 6, 16
    horizons = (2, 3)
    predictor = FixedQueryFutureFeaturePredictor(
        horizons=horizons, patches_per_frame=P, bert_dim=d, vision_dim=D, num_heads=4,
    ).eval()
    compressed = torch.randn(3, T, M, d)
    raw = torch.randn(3, T * P, D)

    loss, diag = compute_delta_loss(predictor, compressed, raw, num_frames=T, patches_per_frame=P)

    targets = delta_targets(raw, T, P, horizons)
    pair_losses = {}
    with torch.no_grad():
        for (t, h), target in targets.items():
            pred = predictor(compressed[:, t])[:, horizons.index(h)]
            pair_losses[(t, h)] = ((pred.float() - target.float()) ** 2).mean()
    manual = torch.stack(list(pair_losses.values())).mean()
    torch.testing.assert_close(loss, manual)
    for h in horizons:
        per_h = [v.item() for (t, hh), v in pair_losses.items() if hh == h]
        assert abs(diag[f"delta_h{h}"] - sum(per_h) / len(per_h)) < 1e-5


def test_predictor_output_shape_and_queries():
    predictor = FixedQueryFutureFeaturePredictor(
        horizons=(3,), patches_per_frame=8, bert_dim=16, vision_dim=10, num_heads=4,
    )
    out = predictor(torch.randn(2, 4, 16))
    assert out.shape == (2, 1, 8, 10)
    q = predictor.fixed_queries[0]
    assert q.shape == (8, 16)
    # first half encodes the horizon (shared by all patches), second half the patch index
    torch.testing.assert_close(q[:, :8], q[:1, :8].expand(8, -1))
    assert not torch.allclose(q[0, 8:], q[1, 8:])


def test_loss_is_zero_when_prediction_is_exact():
    T, P, D = 5, 3, 4
    raw = torch.randn(1, T * P, D)

    class Oracle(torch.nn.Module):
        horizons = (1,)

        def forward(self, z):  # z carries the frame index in its first value
            t = int(z[0, 0, 0])
            frames = raw.view(1, T, P, D)
            nxt = frames[:, min(t + 1, T - 1)] - frames[:, t]
            return nxt.unsqueeze(1)

    compressed = torch.arange(T, dtype=torch.float32).view(1, T, 1, 1).expand(1, T, 2, 3).clone()
    loss, diag = compute_delta_loss(Oracle(), compressed, raw, num_frames=T, patches_per_frame=P)
    assert loss.item() == 0.0
    assert diag == {"delta_h1": 0.0}


def test_predictor_layers_use_gelu():
    # Appendix A.2: two decoder layers, FFN width 4 d, GELU
    predictor = FixedQueryFutureFeaturePredictor(
        horizons=(3,), patches_per_frame=4, bert_dim=16, vision_dim=6, num_heads=4,
    )
    assert len(predictor.layers) == 2
    for layer in predictor.layers:
        assert layer.activation is F.gelu and layer.linear1.out_features == 4 * 16
