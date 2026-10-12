from pathlib import Path

import pytest
import torch

from vedje.model import VideoPretrainModel, build_model, parameter_counts

FRAMES, M, P, VISION_DIM = 2, 2, 256, 768  # the "videoprism" encoder config: 256 patches of 768


def _tiny_model(tiny_lm, **kwargs):
    torch.manual_seed(0)
    kwargs.setdefault("delta_enabled", False)
    return VideoPretrainModel(
        language_model_path=tiny_lm, num_frames=FRAMES, num_queries_per_frame=M,
        num_attention_heads=4, delta_num_heads=4, **kwargs,
    )


def test_residual_prior_enters_before_the_score_head(tiny_lm):
    model = _tiny_model(tiny_lm).eval()
    captions = ["a man is cooking", "two dogs run"]
    tokens = torch.randn(2, FRAMES * M, 32)
    rho = torch.tensor([0.7, -1.3])

    seen = {}
    hook = model.vtm_head.register_forward_pre_hook(lambda mod, inp: seen.update(head_input=inp[0]))
    with torch.no_grad():
        scores = model.compute_logits(captions, vision_projected_tokens=tokens, clip_scores=rho)
        cls = model._get_hidden_states(model._prepare_lm_inputs(captions, tokens))[:, 0]
        prior = model.clip_score_proj(rho.unsqueeze(-1))
        expected = model.vtm_head(cls + prior).view(-1)
    hook.remove()

    # the score head reads c(q, v) + e_rho(rho): Eq. 5
    torch.testing.assert_close(seen["head_input"], cls + prior)
    torch.testing.assert_close(scores, expected)
    assert scores.shape == (2,)


def test_prior_changes_the_score_through_the_nonlinear_head(tiny_lm):
    model = _tiny_model(tiny_lm).eval()
    captions = ["a man is cooking"] * 3
    tokens = torch.randn(1, FRAMES * M, 32).expand(3, -1, -1)
    with torch.no_grad():
        without = model.compute_logits(captions, vision_projected_tokens=tokens)
        cls = model._get_hidden_states(model._prepare_lm_inputs(captions, tokens))[:, 0]
        torch.testing.assert_close(without, model.vtm_head(cls).view(-1))
        with_prior = model.compute_logits(captions, vision_projected_tokens=tokens,
                                          clip_scores=torch.tensor([-1.0, 0.0, 1.0]))
    assert not torch.allclose(with_prior, without)


def test_clip_injection_off_ignores_the_stage1_score(tiny_lm):
    model = _tiny_model(tiny_lm, clip_injection="off").eval()
    captions = ["two dogs run", "a video of the kitchen"]
    tokens = torch.randn(2, FRAMES * M, 32)
    with torch.no_grad():
        a = model.compute_logits(captions, vision_projected_tokens=tokens, clip_scores=torch.tensor([3.0, -3.0]))
        b = model.compute_logits(captions, vision_projected_tokens=tokens)
    torch.testing.assert_close(a, b)


def test_training_step_runs_on_cpu(tiny_lm):
    model = _tiny_model(tiny_lm, delta_enabled=True, delta_horizons=(1,)).train()
    B, K = 2, 3
    patches = torch.randn(B, FRAMES * P, VISION_DIM)
    losses = model(
        captions=["a man is cooking", "two dogs run"],
        vid_feat=torch.randn(B, 768),
        vision_embeds=patches,
        video_ids=[0, 1],
        neg_vision_embeds=torch.randn(B, K, FRAMES * P, VISION_DIM),
        neg_vid_feats=torch.randn(B, K, 768),
        neg_captions=[["the kitchen", "two dogs", "a video"], ["a man", "is cooking", "the video"]],
        precomputed_clip_scores=torch.randn(B, 1 + 2 * K),
    )
    assert len(losses) == 4 and all(torch.isfinite(l) for l in losses)
    assert losses[3].item() > 0  # L_delta is active in training mode
    sum(losses).backward()
    assert model.vision_projection._inner.queries.grad is not None
    assert model.clip_score_proj[0].weight.grad is not None


def test_parameter_counts_cover_the_model(tiny_lm):
    model = _tiny_model(tiny_lm, delta_enabled=True, delta_horizons=(3,))
    counts = parameter_counts(model)
    assert set(counts) == {"joint_encoder", "compressor", "score_head", "prior", "vtc_projection",
                           "mlm_head", "future_predictor"}
    assert counts["score_head"] == 32 * 32 + 32 + 32 + 1
    assert counts["prior"] == 1 * 64 + 64 + 64 * 32 + 32
    assert counts["vtc_projection"] == 32 * 768 + 768
    assert sum(counts.values()) == sum(p.numel() for p in model.parameters())
    assert parameter_counts(_tiny_model(tiny_lm))["future_predictor"] == 0


@pytest.mark.network
def test_default_model_parameter_counts():
    from vedje.config import load_config
    config = load_config(Path(__file__).resolve().parent.parent / "configs" / "vedje_vp_msrvtt.yaml")
    counts = parameter_counts(build_model(config, training=False))
    assert round(counts["joint_encoder"] / 1e6) == 33
    assert counts["score_head"] == 148225
    assert counts["prior"] + counts["score_head"] < 200_000
    assert counts["future_predictor"] == 0


def test_joint_encoder_reads_at_most_64_text_tokens(tiny_lm):
    # Section 4.1 and Appendix A.2: 64 text tokens, [CLS] and [SEP] included; the cache is never cut
    model = _tiny_model(tiny_lm)
    tokens = torch.zeros(1, FRAMES * M, 32)
    ids = model._tokenize_text_vision_pair([" ".join(["video"] * 200)], tokens)["input_ids"][0].tolist()
    assert ids.index(model.tokenizer.sep_token_id) + 1 == 64
    assert ids.count(model.tokenizer.image_token_id) == FRAMES * M
    assert len(ids) == 64 + FRAMES * M + 1


def test_masked_language_modelling_follows_bert(tiny_lm, monkeypatch):
    # Appendix A.3: standard MLM, 15% of the text tokens; 80% of them become [MASK]
    model = _tiny_model(tiny_lm)
    captions = [" ".join(["a man is cooking in the kitchen"] * 8)] * 64
    tokens = torch.zeros(64, FRAMES * M, 32)
    original = model._tokenize_text_vision_pair(captions, tokens)["input_ids"]
    seen, real = {}, model._compute_inputs_embeds

    def capture(ids, vision):  # records the ids after masking
        seen["ids"] = ids
        return real(ids, vision)

    monkeypatch.setattr(model, "_compute_inputs_embeds", capture)
    torch.manual_seed(0)
    _, labels = model._prepare_mlm_inputs_labels(captions, tokens)
    text = (original != model.tokenizer.image_token_id) & (original != model.tokenizer.pad_token_id) \
        & (original != model.tokenizer.cls_token_id) & (original != model.tokenizer.sep_token_id)
    selected = labels != -100
    assert not (selected & ~text).any()
    assert 0.12 < selected.sum() / text.sum() < 0.18
    assert 0.7 < (seen["ids"][selected] == model.tokenizer.mask_token_id).float().mean() < 0.9
    assert (seen["ids"][~text] == original[~text]).all()  # the cache and special tokens stay


def test_masked_language_loss_is_zero_when_nothing_is_masked(tiny_lm, monkeypatch):
    import vedje.model as vm
    monkeypatch.setattr(vm, "MLM_PROBABILITY", 0.0)
    model = _tiny_model(tiny_lm)
    loss = model.compute_mlm_loss(["two dogs run"], torch.zeros(1, FRAMES * M, 32))
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()  # the zero loss keeps the graph, so a training step still runs

