import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from ruamel.yaml import YAML
from torch.utils.data import DataLoader

from vedje.backbone import Backbone, prepare
from vedje.cli import main, plan
from vedje.config import load_config
from vedje.data import create_eval_dataset, create_train_dataset
from vedje.data.utils import load_precomputed_features
from vedje.model import build_model
from vedje.video import load_video_frames

ROOT = Path(__file__).resolve().parent.parent
T = 4  # frames per video


class TinyBackbone(Backbone):
    """2 x 2 patches of 4 x 4 pixels per 8 x 8 frame; the first stage embeds the colours and the letters."""
    name = "tiny"
    frame_size = 8
    patches_per_frame = 4
    patch_dim = 6

    def __init__(self):
        self.patch_calls = 0

    def patches(self, frames):  # each patch is its mean colour and that colour squared
        self.patch_calls += 1
        colour = frames.unfold(2, 4, 4).unfold(3, 4, 4).mean(dim=(-1, -2)).flatten(2).transpose(1, 2)  # (T, 4, 3)
        return torch.cat([colour, colour ** 2], dim=-1)

    def embed_video(self, frames):
        return torch.cat([frames.mean(dim=(0, 2, 3)), torch.ones(1)])

    def embed_texts(self, captions):
        return torch.tensor([[len(c), c.count("a"), c.count("o"), 1.0] for c in captions])


def _write_video(path, seed):
    """Six random 24 x 32 frames, stored losslessly."""
    av = pytest.importorskip("av")
    rng = np.random.default_rng(seed)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("png", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "rgb24"
        for _ in range(6):
            frame = av.VideoFrame.from_ndarray(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """configs/vedje_vp_msrvtt.yaml on six tiny videos (four for training, two captions each), prepared once."""
    root = tmp_path_factory.mktemp("msrvtt")
    (root / "videos").mkdir()
    for i in range(6):
        _write_video(root / "videos" / f"video{i}.mp4", seed=i)
    train = [{"video_id": f"video{i}", "video": f"video{i}.mp4", "caption": [f"A man cooks {i}.", "two dogs" + " run" * i]}
             for i in range(4)]
    test = [{"video": f"video{i}.mp4", "caption": f"A video of the kitchen, number {i}!"} for i in (5, 4)]
    (root / "train.json").write_text(json.dumps(train))
    (root / "test.json").write_text(json.dumps(test))
    config = load_config(ROOT / "configs" / "vedje_vp_msrvtt.yaml")
    config.update(num_frames=T, num_hard_negatives=2, msrvtt_videos=str(root / "videos"),
                  msrvtt_train_ann=str(root / "train.json"), msrvtt_test_ann=str(root / "test.json"))
    YAML().dump(config, root / "base.yaml")
    backbone = TinyBackbone()
    path = prepare(backbone, str(root / "base.yaml"), str(root / "out" / "tiny"), num_workers=0)
    return backbone, str(root / "base.yaml"), path


def test_prepare_writes_the_files_the_loaders_read(prepared):
    backbone, base, path = prepared
    config, videos = load_config(path), load_config(base)["msrvtt_videos"]
    assert path.endswith("tiny.yaml") and config["vision_encoder"] == "tiny"
    assert (config["vision_dim"], config["clip_dim"], config["patches_per_frame"]) == (6, 4, 4)
    assert not {"vision_encoder_path", "clip_model_path"} & set(config)  # no VideoPrism or LvT checkpoint to load

    def frames(video):
        return load_video_frames(f"{videos}/{video}", T, size=8, normalize=False)

    def video_embedding(video):
        return F.normalize(backbone.embed_video(frames(video)), dim=0)

    # one feature file per video: frame t in rows t*P to t*P+P-1, and the first-stage embedding of the video
    for i in range(6):
        patches, v_global = load_precomputed_features(f"{config['precomputed_features_dir']}/video{i}.pt")
        torch.testing.assert_close(patches, backbone.patches(frames(f"video{i}.mp4")).reshape(T * 4, 6).bfloat16())
        torch.testing.assert_close(v_global, video_embedding(f"video{i}.mp4").bfloat16())

    # the training file: one embedding per video and per caption, in the order of the training samples
    stage1, train_ds = torch.load(config["lvt_embeds_path"]), create_train_dataset(config)
    assert stage1["video_ids"] == [f"video{i}" for i in range(4)]
    assert stage1["caption_to_video_idx"] == [0, 0, 1, 1, 2, 2, 3, 3]
    for i, sample in enumerate(train_ds.samples):
        torch.testing.assert_close(stage1["text_embeds"][i],
                                   F.normalize(backbone.embed_texts([sample["caption"]]), dim=-1)[0])
        torch.testing.assert_close(train_ds[i][1], video_embedding(sample["video"]).bfloat16())  # the contrastive target

    # the test file: in the order in which the evaluation reads the test videos and captions
    stage1, test_ds = torch.load(config["lvt_test_features_path"]), create_eval_dataset(config)
    torch.testing.assert_close(stage1["vid_feats"], torch.stack([video_embedding(v) for v in test_ds.video]))
    torch.testing.assert_close(stage1["text_feats"], F.normalize(backbone.embed_texts(test_ds.raw_text), dim=-1))


def test_a_prepared_config_trains(prepared, tiny_lm):
    # as in vedje train: the real script mines the hard negatives, then a training step reads the prepared files
    _, _, path = prepared
    spec = importlib.util.spec_from_file_location("mine_hard_negatives", ROOT / "scripts" / "mine_hard_negatives.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.main(script.parse_args(["--config", path, "--top_k", "3", "--device", "cpu"]))

    config = load_config(path)
    model = build_model({**config, "language_model_path": tiny_lm}).train()
    assert (model.vision_dim, model.clip_dim, model.patches_per_frame) == (6, 4, 4)
    patches, vid_feat, captions, video_ids, _, neg_patches, neg_feats, neg_captions, scores = next(iter(
        DataLoader(create_train_dataset(config), batch_size=2)))
    losses = model(captions=list(captions), vid_feat=vid_feat.float(), vision_embeds=patches.float(),
                   video_ids=video_ids.tolist(), neg_vision_embeds=neg_patches.float(), neg_vid_feats=neg_feats.float(),
                   neg_captions=[list(c) for c in zip(*neg_captions)], precomputed_clip_scores=scores)
    assert all(torch.isfinite(loss) for loss in losses) and losses[3] > 0  # L_delta reads the backbone's patches


def test_prepare_resumes_and_the_vedje_prepare_command(prepared, tmp_path, monkeypatch):
    backbone, base, path = prepared
    calls = backbone.patch_calls
    prepare(backbone, base, str(Path(path).parent), num_workers=0)
    assert backbone.patch_calls == calls  # every video has its features, so they are not computed again

    # vedje prepare imports the class from the current folder; vedje train then skips the first-stage steps
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path[:])
    (tmp_path / "my_backbone.py").write_text("from test_backbone import TinyBackbone as MyBackbone\n")
    main(["prepare", "my_backbone:MyBackbone", base, "--out", "runs/mine", "--num_workers", "0"])
    config, steps = plan("runs/mine/mine.yaml")
    assert [s[1] for s in steps[:2]] == [["mine_hard_negatives.py"], ["train.py", "--output_dir", "output/mine"]]
    assert steps[2][1][0] == "evaluate.py"
    assert all(Path(config[k]).exists() for k in ("precomputed_features_dir", "lvt_embeds_path", "lvt_test_features_path"))


def test_the_config_gives_the_backbone_dimensions(tiny_lm):
    # without them, the shipped configs keep those of their built-in backbones
    small = {"language_model_path": tiny_lm, "num_frames": 2, "num_queries_per_frame": 2, "num_attention_heads": 4}
    for name, dims in (("vedje_vp_msrvtt", (768, 768, 256)), ("vedje_vclip_msrvtt", (1024, 768, 256))):
        model = build_model({**load_config(ROOT / "configs" / f"{name}.yaml"), **small}, training=False)
        assert (model.vision_dim, model.clip_dim, model.patches_per_frame) == dims
    # the config's own values win over those of a built-in backbone
    model = build_model({**small, "vision_encoder": "vcxl", "vision_dim": 6, "clip_dim": 4, "patches_per_frame": 9},
                        training=False)
    assert model.vision_projection(torch.randn(1, 2 * 9, 6)).shape == (1, 2 * 2, 32)
    assert model.text_projection.out_features == 4
    with pytest.raises(ValueError, match="vision_dim, clip_dim and patches_per_frame"):
        build_model({**small, "vision_encoder": "mine"}, training=False)


def test_vedje_indexes_and_searches_with_the_backbone(prepared, tiny_lm, tmp_path, capsys):
    from vedje.inference import VEDJE

    backbone, base, path = prepared
    config = {**load_config(path), "language_model_path": tiny_lm}
    assert config["backbone"] == f"{Path(__file__).resolve()}:TinyBackbone"  # prepare records where the class lives
    model = build_model(config, training=False).eval()
    torch.save({"config": config, "model": model.state_dict()}, tmp_path / "ckpt.pth")

    # the checkpoint names its backbone, so VEDJE loads it by itself
    vedje = VEDJE.from_checkpoint(tmp_path / "ckpt.pth", device="cpu")
    assert type(vedje.backbone).__name__ == "TinyBackbone"
    videos = load_config(base)["msrvtt_videos"]
    index = vedje.index(videos, progress=False)
    assert len(index) == 6 and index.stage1.shape == (6, 4)

    # each cache is the compressor's reading of the patch features in bf16, as training reads them
    frames = load_video_frames(f"{videos}/video0.mp4", T, size=8, normalize=False)
    with torch.no_grad():
        expected = model.vision_projection(backbone.patches(frames).reshape(1, T * 4, 6).bfloat16().float())
    torch.testing.assert_close(index.caches[:1].float(), expected.bfloat16().float())
    torch.testing.assert_close(index.stage1[0], F.normalize(backbone.embed_video(frames), dim=0))
    hits = vedje.search("A man cooks 0.", index, top=3, candidates=4)
    assert [h.rank for h in hits] == [1, 2, 3] and {h.video for h in hits} <= set(index.videos)

    # and so do the commands
    main(["index", str(tmp_path / "ckpt.pth"), videos, "--out", str(tmp_path / "index.pt"), "--device", "cpu"])
    main(["search", str(tmp_path / "index.pt"), "two dogs run", "--top", "2", "--device", "cpu"])
    assert capsys.readouterr().out.strip().splitlines()[-1].endswith(".mp4")

    # a checkpoint that does not name its backbone asks for it
    torch.save({"config": {k: v for k, v in config.items() if k != "backbone"}, "model": model.state_dict()},
               tmp_path / "unnamed.pth")
    with pytest.raises(ValueError, match="backbone=MyBackbone"):
        VEDJE.from_checkpoint(tmp_path / "unnamed.pth", device="cpu").index(videos, progress=False)
