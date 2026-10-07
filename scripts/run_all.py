"""Run VEDJE end to end for one config: index the videos, prepare the first stage, train, rerank and evaluate.

    python scripts/run_all.py --config configs/vedje_vp_msrvtt.yaml

It runs the step scripts next to this one in order, with the paths of the config. A step whose output already
exists is skipped, so running the command again resumes where it stopped. --dry_run prints the commands without
running them. Configs that read precomputed features (VideoCLIP-XL) skip the steps that compute them and expect
those files (REPRODUCING.md, "Inputs you provide").
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from vedje.config import load_config  # noqa: E402


def plan(config_path, output_dir=None):
    """The config and its steps: (label, script and extra arguments, the file that marks the step as done or None)."""
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run VEDJE end to end for one config, resuming where it stopped.")
    parser.add_argument("--config", required=True, help="for example configs/vedje_vp_msrvtt.yaml")
    parser.add_argument("--output_dir", default=None, help="default: output/<config name>")
    parser.add_argument("--dry_run", action="store_true", help="print the commands without running them")
    args = parser.parse_args(argv)

    config, steps = plan(args.config, args.output_dir)
    if config.get("vision_encoder", "videoprism") != "videoprism" and not args.dry_run:
        given = [config.get(k) for k in ("precomputed_features_dir", "lvt_embeds_path", "lvt_test_features_path")]
        missing = [p for p in given if p and not os.path.exists(p)]
        if missing:
            sys.exit("This config reads precomputed inputs (REPRODUCING.md, \"Inputs you provide\"). Missing: " + ", ".join(missing))

    for label, (script, *extra), done in steps:
        if done and os.path.exists(done):
            print(f"== {label}: done, {done} exists")
            continue
        command = [sys.executable, str(HERE / script), "--config", args.config, *extra]
        print(f"== {label}\n   python {os.path.relpath(HERE / script)} --config {args.config} {' '.join(extra)}".rstrip(), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
