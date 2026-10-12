"""Step 2b. Stage-1 hard negatives, mined from the stage-1 embeddings of step 2a.

For each training caption, finds the top-K most similar videos (excluding the
positive) by stage-1 cosine similarity. This matches the evaluation, where the
first stage retrieves the top-K videos for each text query. The embeddings are
those of the config's first stage: VideoPrism-LvT from step 2a, or VideoCLIP-XL
embeddings in the same format for configs/vedje_vclip_msrvtt.yaml.

Output format: {caption_index: [video_id1, video_id2, ...]}

Usage:
    python scripts/mine_hard_negatives.py --config configs/vedje_vp_msrvtt.yaml
With --config, the embeddings (lvt_embeds_path) and the output file
(hard_negatives_path) come from the config; explicit arguments override them.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm


@torch.inference_mode()
def mine(text_embeds, video_embeds, caption_to_video_idx, video_ids, top_k=50, device="cpu",
         chunk_size=1000):
    """{caption index: its top_k most similar wrong videos}, from L2-normalised embeddings."""
    video_embeds = video_embeds.float().to(device)
    hard_negatives = {}
    for start in tqdm(range(0, len(text_embeds), chunk_size), desc="Mining"):
        sims = text_embeds[start:start + chunk_size].float().to(device) @ video_embeds.T
        positives = torch.as_tensor(caption_to_video_idx[start:start + chunk_size], device=device)
        sims[torch.arange(len(positives), device=device), positives] = -float("inf")
        for offset, idxs in enumerate(sims.topk(top_k, dim=1).indices.tolist()):
            hard_negatives[str(start + offset)] = [video_ids[i] for i in idxs]
    return hard_negatives


def resolve_paths(args):
    """Fill the arguments left unset from --config, then from the MSR-VTT defaults."""
    embeddings = output_path = None
    if args.config:
        from vedje.config import load_config
        config = load_config(args.config)
        embeddings = config.get("lvt_embeds_path") or None
        output_path = config.get("hard_negatives_path") or None
    args.embeddings = args.embeddings or embeddings or "./data_root/msrvtt/lvt_clip_embeds_train.pt"
    args.output_path = args.output_path or output_path or \
        "./data_root/msrvtt/hard_negatives_text2vid_top50.json"
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Mine text-to-video hard negatives for training from the stage-1 embeddings.")
    parser.add_argument("--config", default=None,
                        help="config whose paths fill the arguments below")
    parser.add_argument("--embeddings", default=None,
                        help="stage-1 embeddings of the training set written by step 2a (default: the "
                             "config's lvt_embeds_path, otherwise ./data_root/msrvtt/lvt_clip_embeds_train.pt)")
    parser.add_argument("--output_path", default=None,
                        help="default: the config's hard_negatives_path, otherwise "
                             "./data_root/msrvtt/hard_negatives_text2vid_top50.json")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", default=None,
                        help="default: cuda when available, otherwise cpu")
    return parser.parse_args(argv)


def main(args):
    args = resolve_paths(args)
    data = torch.load(args.embeddings, map_location="cpu")
    print(f"Videos: {len(data['video_ids'])}, Captions: {len(data['text_embeds'])}")

    hard_negatives = mine(data["text_embeds"], data["video_embeds"], data["caption_to_video_idx"],
                          data["video_ids"], top_k=args.top_k, device=args.device)

    Path(os.path.dirname(args.output_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(hard_negatives, f)
    print(f"Saved {len(hard_negatives)} entries to {args.output_path}")


if __name__ == "__main__":
    main(parse_args())
