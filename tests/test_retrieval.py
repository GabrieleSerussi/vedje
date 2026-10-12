import torch
import torch.nn.functional as F

from vedje.retrieval import evaluation_t2v, evaluation_v2t, recall_at_k


def test_recall_at_k_single_targets():
    rankings = [[3, 1, 2], [0, 2, 1], [2, 1, 0], [1, 0, 2]]
    targets = [3, 2, 0, 9]
    assert recall_at_k(rankings, targets, ks=(1, 2, 3)) == {"R@1": 0.25, "R@2": 0.5, "R@3": 0.75}


def test_recall_at_k_multiple_targets_and_tensors():
    rankings = torch.tensor([[4, 5, 6], [7, 8, 9]])
    targets = [[6, 1], torch.tensor([8, 9])]
    assert recall_at_k(rankings, targets, ks=(1, 2, 3)) == {"R@1": 0.0, "R@2": 0.5, "R@3": 1.0}


def test_recall_at_k_rounds_to_three_decimals():
    rankings = [[0], [1], [2]]
    assert recall_at_k(rankings, [0, 0, 0], ks=(1,)) == {"R@1": 0.333}


class _ToyDataset:
    """Six videos, two captions each; caption i describes video txt2vid[i]."""

    def __init__(self):
        self.video = [f"v{i}.mp4" for i in range(6)]
        self.text = [f"t{i % 6}" for i in range(12)]
        self.txt2vid = {i: i % 6 for i in range(12)}
        self.vid2txt = {v: [v, v + 6] for v in range(6)}


class _OracleReranker:
    """Scores 10 for the correct pair and 0 otherwise; the video index is stored in the cache."""

    def _chunked_logits(self, captions, vision_tokens, clip_scores=None):
        vids = vision_tokens[:, 0, 0].round().long().tolist()
        return torch.tensor([10.0 if int(c[1:]) == v else 0.0 for c, v in zip(captions, vids)])


def _toy_features(seed=0):
    g = torch.Generator().manual_seed(seed)
    vid = F.normalize(torch.randn(6, 8, generator=g), dim=-1)
    txt = F.normalize(torch.cat([vid, vid]) + 0.9 * torch.randn(12, 8, generator=g), dim=-1)
    tokens = torch.arange(6, dtype=torch.float32).view(6, 1, 1).expand(6, 4, 3).clone()
    return vid, txt, tokens


def test_two_stage_evaluation_with_an_oracle_reranker():
    ds, model = _ToyDataset(), _OracleReranker()
    vid, txt, tokens = _toy_features()
    config = {"k": 3}

    clip_t2v, reranked_t2v = evaluation_t2v(model, ds, "cpu", config, vid, txt, tokens)
    top3 = (txt @ vid.T).argsort(dim=1, descending=True)[:, :3]  # stage-1 candidates
    targets = [ds.txt2vid[i] for i in range(12)]
    assert clip_t2v == recall_at_k(top3, targets)
    # the oracle moves the correct video to the top whenever stage 1 retrieved it
    retrieved = recall_at_k(top3, targets, ks=(3,))["R@3"]
    assert reranked_t2v == {"R@1": retrieved, "R@5": retrieved, "R@10": retrieved}
    assert reranked_t2v["R@1"] >= clip_t2v["R@1"]

    clip_v2t, reranked_v2t = evaluation_v2t(model, ds, "cpu", config, vid, txt, tokens)
    top3 = (vid @ txt.T).argsort(dim=1, descending=True)[:, :3]
    targets = [ds.vid2txt[v] for v in range(6)]
    assert clip_v2t == recall_at_k(top3, targets)
    retrieved = recall_at_k(top3, targets, ks=(3,))["R@3"]
    assert reranked_v2t == {"R@1": retrieved, "R@5": retrieved, "R@10": retrieved}
