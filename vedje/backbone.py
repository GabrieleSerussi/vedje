"""Use VEDJE with another frozen backbone: write one small class, then prepare a config's dataset with it.

    from vedje.backbone import Backbone, prepare

    class MyBackbone(Backbone):
        name = "my_backbone"
        frame_size = 224                  # frames are resized to frame_size x frame_size
        patches_per_frame = 256           # P
        patch_dim = 768                   # D

        def patches(self, frames):        # (T, 3, H, W) in [0, 1] -> (T, P, D) frozen patch features
            ...
        def embed_video(self, frames):    # the same frames -> (C,) first-stage embedding of the video
            ...
        def embed_texts(self, captions):  # N captions -> (N, C) first-stage embeddings of the captions
            ...

    if __name__ == "__main__":            # the worker processes that decode the videos import this file
        prepare(MyBackbone(), "configs/vedje_vp_msrvtt.yaml", "output/my_backbone")

prepare encodes the videos and captions of the config's dataset once and writes, in the formats of the dataset
loaders and the step scripts, the features of each video, the first-stage embeddings of the training and test sets
and a config that reads them. The backbone's embeddings are the first stage: they give the candidates, the score
prior, the hard negatives and the contrastive targets. From the command line, with the class in my_backbone.py:

    vedje prepare my_backbone:MyBackbone configs/vedje_vp_msrvtt.yaml --out output/my_backbone
    vedje train output/my_backbone/my_backbone.yaml    # hard negatives, training and evaluation

The config records where the class lives (`backbone: /path/to/my_backbone.py:MyBackbone`), and training copies it
into every checkpoint, so `vedje index` and `vedje search` load the backbone again by themselves.
"""

import functools
import importlib
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from vedje.video import load_video_frames


class Backbone:
    """A frozen visual backbone with its first stage: subclass it, set the attributes and write the three methods.

    The frames of a video are the config's num_frames (T) frames, sampled uniformly and resized to frame_size x
    frame_size: a (T, 3, H, W) float tensor in [0, 1] on the CPU. The methods move them to their device and
    normalise them as their model expects; prepare L2-normalises the embeddings.
    """
    name: str               # the vision_encoder of the config
    frame_size: int = 224   # H = W
    patches_per_frame: int  # P
    patch_dim: int          # D

    def patches(self, frames: torch.Tensor) -> torch.Tensor:
        """(T, 3, H, W) frames -> (T, P, D) frozen patch features, the P patches of every frame in the same order."""
        raise NotImplementedError

    def embed_video(self, frames: torch.Tensor) -> torch.Tensor:
        """(T, 3, H, W) frames -> (C,) first-stage embedding of the video."""
        raise NotImplementedError

    def embed_texts(self, captions: List[str]) -> torch.Tensor:
        """N captions, as written -> (N, C) first-stage embeddings, compared with those of embed_video by cosine."""
        raise NotImplementedError


def load(spec: str) -> Backbone:
    """A backbone from `module:Class` (a module in the current folder) or `path/to/module.py:Class`."""
    module, _, name = spec.rpartition(":")
    folder = os.getcwd()
    if module.endswith(".py"):
        folder, module = os.path.split(os.path.abspath(module))
        module = module[:-3]
    if folder not in sys.path:
        sys.path.insert(0, folder)
    return getattr(importlib.import_module(module), name)()


def spec_of(backbone: Backbone) -> Optional[str]:
    """`path/to/module.py:Class` for the class of `backbone`, or None for a class without a source file."""
    cls = type(backbone)
    path = getattr(sys.modules.get(cls.__module__), "__file__", None)
    return f"{os.path.abspath(path)}:{cls.__qualname__}" if path else None


def _embed_texts(backbone, captions, batch_size, split):
    """(N, C) L2-normalised first-stage embeddings of the captions, batch_size captions at a time."""
    return torch.cat([F.normalize(backbone.embed_texts(captions[i:i + batch_size]).float(), dim=-1).cpu()
                      for i in tqdm(range(0, len(captions), batch_size), desc=f"Embedding the {split} captions")])


@torch.inference_mode()
def prepare(backbone: Backbone, config: str, out_dir: str, batch_size: int = 256, num_workers: int = 4) -> str:
    """Encode the dataset of a config with `backbone` and write what `vedje train` reads; returns the new config.

    Writes to out_dir: features/ (one file per video), stage1_train.pt and stage1_test.pt (the first-stage
    embeddings of the training and test sets) and <out_dir name>.yaml, the config with the backbone's name,
    its dimensions and these paths. A video whose feature file exists is not encoded again, so an interrupted run
    resumes; after a change to the backbone, write to a new out_dir. embed_texts receives batch_size captions at a
    time, and num_workers processes decode the videos (0 decodes them in this process).
    """
    from ruamel.yaml import YAML

    from vedje.config import load_config
    from vedje.data import DATASET_REGISTRY, create_eval_dataset, feature_files, stage1_layout
    from vedje.data.utils import save_precomputed_features

    base, config = config, load_config(config)
    reg = DATASET_REGISTRY[config["train_dataset"]]
    out, features_dir = Path(out_dir), Path(out_dir) / "features"
    shape = (config.get("num_frames", 16), backbone.patches_per_frame, backbone.patch_dim)  # (T, P, D)
    print(f"Preparing {config['train_dataset']} for the {backbone.name} backbone in {out}")

    # The training captions in the order of the training datasets, and the test set read without features
    with open(config[reg["train_ann_key"]]) as f:
        train_videos, train_captions, layout = stage1_layout(json.load(f))
    test_set = config.get("test_set", "msrvtt")
    test_ds = create_eval_dataset({**config, "precomputed_features_dir": "",
                                   DATASET_REGISTRY[test_set]["features_key"]: ""}, test_set)

    # Each video is decoded once: its features are written unless they exist, and its embedding is kept
    files = feature_files(config)  # {feature file name: video path}
    paths = [os.path.join(config[reg["video_key"]], rel) for rel in files.values()]
    frames_of = functools.partial(load_video_frames, num_frames=shape[0], size=backbone.frame_size, normalize=False)
    loader = DataLoader(paths, batch_size=None, collate_fn=frames_of, num_workers=num_workers)
    features_dir.mkdir(parents=True, exist_ok=True)
    video_embeds = {}
    for (name, rel), frames in tqdm(zip(files.items(), loader), total=len(files), desc="Encoding videos"):
        video_embeds[rel] = F.normalize(backbone.embed_video(frames).float().flatten(), dim=0).cpu()
        if not (features_dir / name).exists():
            patches = backbone.patches(frames)
            if tuple(patches.shape) != shape:
                raise ValueError(f"patches gave {tuple(patches.shape)} for {rel}, expected (T, P, D) = {shape}")
            save_precomputed_features(str(features_dir / name), patches.reshape(-1, shape[2]), video_embeds[rel])

    train_embeds = torch.stack([video_embeds[v] for v in train_videos])
    torch.save({"video_embeds": train_embeds,
                "text_embeds": _embed_texts(backbone, train_captions, batch_size, "training"), **layout},
               out / "stage1_train.pt")
    torch.save({"vid_feats": torch.stack([video_embeds[v] for v in test_ds.video]),
                "text_feats": _embed_texts(backbone, test_ds.raw_text, batch_size, "test")}, out / "stage1_test.pt")

    # The backbone's name, class and dimensions, then the base config without its VideoPrism and LvT checkpoints
    drop = ("vision_encoder", "backbone", "vision_encoder_path", "vp_attn_implementation", "clip_model_path")
    spec = spec_of(backbone)
    config = {"vision_encoder": backbone.name, **({"backbone": spec} if spec else {}), "vision_dim": shape[2],
              "clip_dim": train_embeds.shape[1], "patches_per_frame": shape[1],
              **{k: v for k, v in config.items() if k not in drop}}
    config.update({
        "use_precomputed_features": True,
        "precomputed_features_dir": str(features_dir),
        reg["features_key"]: str(features_dir),
        "lvt_embeds_path": str(out / "stage1_train.pt"),
        "lvt_test_features_path": str(out / "stage1_test.pt"),
        "hard_negatives_path": str(out / "hard_negatives.json"),
        "experiment": {**config.get("experiment", {}), "run_name": out.resolve().name},
    })
    # Named after out_dir, so that `vedje train` writes its checkpoints to output/<out_dir name>
    config_path = out / f"{out.resolve().name}.yaml"
    with open(config_path, "w") as f:
        f.write(f"# VEDJE with the {backbone.name} backbone, written by vedje.backbone.prepare from {base}\n")
        yaml = YAML()
        yaml.width = 4096  # one line per path
        yaml.dump(config, f)
    print(f"Wrote {config_path}. Train VEDJE with: vedje train {config_path}")
    return str(config_path)
