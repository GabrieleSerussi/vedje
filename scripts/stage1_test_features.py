"""Step 3. Stage-1 (VideoPrism-LvT) embeddings of the test videos and captions.

Writes {"vid_feats": (N_videos, 768), "text_feats": (N_texts, 768)}, the
stage-1 embeddings that scripts/train.py and scripts/evaluate.py read through
the config's `lvt_test_features_path`. Test videos are read with mediapy,
which needs FFmpeg on the PATH.

Usage:
    python scripts/stage1_test_features.py --config configs/vedje_vp_msrvtt.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_CONFIG = "configs/vedje_vp_msrvtt.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage-1 (VideoPrism-LvT) embeddings of the test set.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default="",
                        help="Output .pt (default: the config's lvt_test_features_path)")
    parser.add_argument("--device", default=None,
                        help="default: cuda when available, otherwise cpu")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for LvT video feature extraction (lower = less VRAM)")
    return parser.parse_args(argv)


def main(args):
    import torch

    from vedje.config import load_config
    from vedje.data import create_eval_dataset
    from vedje.lvt import DEFAULT_LVT_MODEL, load_lvt_model_fixed
    from vedje.retrieval import compute_lvt_text_features, compute_lvt_video_features

    config = load_config(args.config)
    output = args.output or config.get("lvt_test_features_path")
    if not output:
        raise ValueError("pass --output or set lvt_test_features_path in the config")

    # Override eval batch size to avoid OOM on LvT full-video model
    config['eval_batch_size'] = args.batch_size

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load LvT model
    clip_model_path = config.get("clip_model_path", DEFAULT_LVT_MODEL)
    print(f"Loading LvT CLIP model from {clip_model_path}...")
    lvt_model, lvt_tokenizer = load_lvt_model_fixed(clip_model_path, device=device, dtype=torch.float32)

    print("LvT model loaded.")

    ds_name = config.get("test_set", "msrvtt")
    test_ds = create_eval_dataset(config, ds_name)
    print(f"Test dataset: {len(test_ds)} videos, {len(test_ds.text)} captions")

    # Compute features
    print("Computing LvT video features...")
    vid_feats = compute_lvt_video_features(lvt_model, test_ds, device, config, dataset_name=ds_name)
    print(f"  vid_feats: {vid_feats.shape}")

    print("Computing LvT text features...")
    # The first stage reads the captions as written (raw_text), as in step 2a: the
    # SentencePiece tokenizer of LvT is case-sensitive.
    lvt_texts = getattr(test_ds, "raw_text", test_ds.text)
    text_feats = compute_lvt_text_features(lvt_model, lvt_tokenizer, lvt_texts, device)
    print(f"  text_feats: {text_feats.shape}")

    # Save
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "vid_feats": vid_feats.cpu(),
        "text_feats": text_feats.cpu(),
    }, output)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main(parse_args())
