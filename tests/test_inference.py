import torch
import torch.nn.functional as F

from vedje.inference import Index, list_videos, rank


def test_list_videos_keeps_video_files_in_name_order(tmp_path):
    for name in ("b.mp4", "a.webm", "notes.txt", "c.MOV"):
        (tmp_path / name).write_bytes(b"")
    assert [p.rsplit("/", 1)[-1] for p in list_videos(tmp_path)] == ["a.webm", "b.mp4", "c.MOV"]
    assert list_videos(["x.mp4", "y.mp4"]) == ["x.mp4", "y.mp4"]


def test_index_saves_and_loads(tmp_path):
    index = Index(videos=["a.mp4", "b.mp4"], caches=torch.randn(2, 64, 8).bfloat16(),
                  stage1=F.normalize(torch.randn(2, 4), dim=-1), checkpoint="ckpt.pth", config={"k": 20})
    index.save(tmp_path / "index.pt")
    loaded = Index.load(tmp_path / "index.pt")
    assert loaded.videos == index.videos and loaded.checkpoint == "ckpt.pth" and loaded.config == {"k": 20}
    torch.testing.assert_close(loaded.caches, index.caches)


class _Reranker:
    """Scores each candidate by the value stored in its cache, and records what it was given."""

    def _chunked_logits(self, captions, vision_tokens, clip_scores=None):
        self.captions, self.prior = captions, clip_scores
        return vision_tokens[:, 0, 0]


def test_vedje_reranks_the_first_stage_candidates():
    # six videos; the first stage prefers videos 0, 1, 2 for the query, VEDJE prefers the third of them
    stage1 = F.normalize(torch.tensor([[1.0, 0.0], [0.9, 0.3], [0.8, 0.5], [0.1, 1.0], [0.0, 1.0], [-1.0, 0.0]]), dim=-1)
    caches = torch.zeros(6, 4, 3)
    caches[:, 0, 0] = torch.tensor([0.2, 0.1, 0.9, 5.0, 5.0, 5.0])   # high values outside the candidates never count
    index = Index(videos=[f"v{i}.mp4" for i in range(6)], caches=caches, stage1=stage1)
    model = _Reranker()
    hits = rank(model, "A person walks!", torch.tensor([1.0, 0.0]), index, top=3, candidates=3)
    assert [h.video for h in hits] == ["v2.mp4", "v0.mp4", "v1.mp4"]
    assert [h.first_stage_rank for h in hits] == [3, 1, 2] and [h.rank for h in hits] == [1, 2, 3]
    assert model.captions == ["a person walks"] * 3                    # the joint encoder reads the cleaned query
    assert abs(float(model.prior.mean())) < 1e-6 and abs(float(model.prior.std()) - 1) < 1e-5   # standardised prior
