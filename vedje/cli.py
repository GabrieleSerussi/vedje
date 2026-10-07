"""The vedje command: train VEDJE, index your videos, search them with text.

    vedje train configs/vedje_vp_msrvtt.yaml
    vedje index output/vedje_vp_msrvtt/checkpoint_03.pth my_videos/ --out my_index.pt
    vedje search my_index.pt "a person walks across a field"
    vedje prepare my_backbone:MyBackbone configs/vedje_vp_msrvtt.yaml --out output/my_backbone

`vedje train` runs the step scripts in scripts/ in order (index the dataset's videos, prepare the first stage,
train, rerank and evaluate) and skips the steps whose outputs exist, so running it again resumes where it stopped.
`vedje index` and `vedje search` use a trained checkpoint through vedje.inference.
`vedje prepare` encodes a config's dataset with another backbone (vedje.backbone) and writes a config for `vedje train`;
`vedje index` and `vedje search` then load that backbone from the checkpoint, or from --backbone.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
BACKBONE_HELP = "another backbone as module:Class or path/to/module.py:Class (default: the one the checkpoint names)"


def plan(config_path, output_dir=None):
    """The config and its training steps: (label, script and extra arguments, the file that marks it done or None)."""
    from vedje.config import load_config
    config = load_config(config_path)
    out = output_dir or os.path.join("output", Path(config_path).stem)
    checkpoint = os.path.join(out, f"checkpoint_{config['max_epoch'] - 1:02d}.pth")
    own_features = config.get("vision_encoder", "videoprism") == "videoprism"
    steps = []
    if own_features:
        steps.append(("1. index the videos", ["extract_features.py"], None))   # skips the videos already indexed
        steps.append(("2a. first-stage embeddings of the training set", ["stage1_train_embeddings.py"], config.get("lvt_embeds_path")))
    steps.append(("2b. hard negatives", ["mine_hard_negatives.py"], config.get("hard_negatives_path")))
    if own_features:
        steps.append(("3. first-stage embeddings of the test set", ["stage1_test_features.py"], config.get("lvt_test_features_path")))
    steps.append(("4. train", ["train.py", "--output_dir", out], checkpoint))
    steps.append(("5. rerank and evaluate", ["evaluate.py", "--checkpoint", checkpoint], None))
    return config, steps


def train(args):
    if not SCRIPTS.is_dir():
        sys.exit("vedje train runs the scripts of the repository: install it from a clone with `pip install -e .`")
    config, steps = plan(args.config, args.output_dir)
    if config.get("vision_encoder", "videoprism") != "videoprism" and not args.dry_run:
        given = [config.get(k) for k in ("precomputed_features_dir", "lvt_embeds_path", "lvt_test_features_path")]
        missing = [p for p in given if p and not os.path.exists(p)]
        if missing:
            sys.exit("This config reads precomputed inputs (see the comments in the config). Missing: " + ", ".join(missing))
    for label, (script, *extra), done in steps:
        if done and os.path.exists(done):
            print(f"== {label}: done, {done} exists")
            continue
        print(f"== {label}\n   python scripts/{script} --config {args.config} {' '.join(extra)}".rstrip(), flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, str(SCRIPTS / script), "--config", args.config, *extra], check=True)


def prepare(args):
    from vedje.backbone import load, prepare as prepare_backbone
    prepare_backbone(load(args.backbone), args.config, args.out, batch_size=args.batch_size, num_workers=args.num_workers)


def index(args):
    from vedje.inference import VEDJE
    vedje = VEDJE.from_checkpoint(args.checkpoint, device=args.device, backbone=args.backbone)
    idx = vedje.index(args.videos if len(args.videos) > 1 else args.videos[0], batch_size=args.batch_size)
    idx.save(args.out)
    print(f"Indexed {len(idx)} videos into {args.out}")


def search(args):
    from vedje.inference import VEDJE, Index
    idx = Index.load(args.index)
    vedje = VEDJE.from_checkpoint(args.checkpoint or idx.checkpoint, device=args.device, backbone=args.backbone)
    hits = vedje.search(args.query, idx, top=args.top, candidates=args.candidates)
    print(f"{'rank':>4}  {'score':>7}  {'first stage':>11}  video")
    for h in hits:
        print(f"{h.rank:>4}  {h.score:>7.3f}  {h.first_stage_rank:>11}  {h.video}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vedje", description="Train VEDJE, index your videos and search them with text.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="run the whole training pipeline for a config, resuming where it stopped")
    p.add_argument("config", help="for example configs/vedje_vp_msrvtt.yaml")
    p.add_argument("--output_dir", default=None, help="default: output/<config name>")
    p.add_argument("--dry_run", action="store_true", help="print the steps without running them")
    p.set_defaults(func=train)

    p = sub.add_parser("prepare", help="encode a config's dataset with another backbone and write a config for it")
    p.add_argument("backbone", help="the Backbone subclass as module:Class, for example my_backbone:MyBackbone")
    p.add_argument("config", help="the config whose dataset to encode, for example configs/vedje_vp_msrvtt.yaml")
    p.add_argument("--out", required=True, help="the folder to write to, for example output/my_backbone")
    p.add_argument("--batch_size", type=int, default=256, help="captions per embed_texts call (default: 256)")
    p.add_argument("--num_workers", type=int, default=4, help="processes that decode the videos (default: 4)")
    p.set_defaults(func=prepare)

    p = sub.add_parser("index", help="index video files once with a trained checkpoint")
    p.add_argument("checkpoint", help="a checkpoint written by vedje train")
    p.add_argument("videos", nargs="+", help="a folder of video files, or the files themselves")
    p.add_argument("--out", default="vedje_index.pt", help="where to save the index (default: vedje_index.pt)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", default=None, help="default: cuda when available, otherwise cpu")
    p.add_argument("--backbone", default=None, help=BACKBONE_HELP)
    p.set_defaults(func=index)

    p = sub.add_parser("search", help="search an index with a text query")
    p.add_argument("index", help="an index written by vedje index")
    p.add_argument("query", help="the text query, in quotes")
    p.add_argument("--top", type=int, default=5, help="how many videos to print (default: 5)")
    p.add_argument("--candidates", type=int, default=20, help="first-stage candidates that VEDJE reranks (default: 20)")
    p.add_argument("--checkpoint", default=None, help="default: the checkpoint that wrote the index")
    p.add_argument("--device", default=None, help="default: cuda when available, otherwise cpu")
    p.add_argument("--backbone", default=None, help=BACKBONE_HELP)
    p.set_defaults(func=search)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
