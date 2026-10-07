"""Step 2a. Stage-1 (VideoPrism-LvT) embeddings of the training videos and captions.

Saves L2-normalized embeddings, so that cosine similarity is a dot product.
During training, the stage-1 scores that enter the residual prior are computed
from these embeddings as embed_text . embed_video, and the video embeddings are
the contrastive targets (Table 7). Step 2b mines the hard negatives from this file.

Usage:
    python scripts/stage1_train_embeddings.py --config configs/vedje_vp_msrvtt.yaml
With --config, the video root, the training annotations and the output file
(lvt_embeds_path) come from the config; explicit arguments override them.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vedje.lvt import DEFAULT_LVT_MODEL, text_embeddings, video_embeddings
from vedje.video import load_video_frames


class RawVideoDataset(Dataset):
    def __init__(self, video_paths, video_root, num_frames=16, frame_size=288):
        self.video_paths = video_paths
        self.video_root = video_root
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        vid_path = os.path.join(self.video_root, self.video_paths[index])
        frames = load_video_frames(
            vid_path, self.num_frames, size=self.frame_size, normalize=False,
        )
        return frames, index


@torch.inference_mode()
def compute_video_features(lvt_model, video_paths, video_root, device,
                           num_frames=16, batch_size=4):
    dataset = RawVideoDataset(video_paths, video_root, num_frames=num_frames)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_feats = []
    for frames, indices in tqdm(loader, desc="Computing LvT video features"):
        frames = frames.to(device, dtype=torch.float32, non_blocking=True)
        vout = lvt_model.video_model(pixel_values_videos=frames)
        pooled = video_embeddings(vout)  # (B, 1, D)
        v_global = pooled.mean(dim=1)
        v_global = F.normalize(v_global, dim=-1)
        all_feats.append(v_global.cpu())

    return torch.cat(all_feats, dim=0)


@torch.inference_mode()
def compute_text_features(lvt_model, lvt_tokenizer, texts, device, batch_size=64):
    all_feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing LvT text features"):
        batch_texts = texts[i: i + batch_size]
        inputs = lvt_tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=64, return_tensors="pt",
        ).to(device)
        text_out = lvt_model.text_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        text_feats = text_embeddings(text_out)
        text_feats = F.normalize(text_feats.float(), dim=-1)
        all_feats.append(text_feats.cpu())
    return torch.cat(all_feats, dim=0)


def resolve_paths(args):
    """Fill the arguments left unset from --config, then from the MSR-VTT defaults."""
    video_root = train_ann = output_dir = output_name = None
    if args.config:
        from vedje.config import load_config
        from vedje.data import DATASET_REGISTRY
        config = load_config(args.config)
        reg = DATASET_REGISTRY[config.get("train_dataset", "msrvtt")]
        video_root = config.get(reg["video_key"])
        train_ann = config.get(reg["train_ann_key"])
        if config.get("lvt_embeds_path"):
            output_dir, output_name = os.path.split(config["lvt_embeds_path"])
        if args.lvt_model_path is None:
            args.lvt_model_path = config.get("clip_model_path") or None
    args.video_root = args.video_root or video_root or "./data_root/msrvtt/videos/"
    args.train_ann = args.train_ann or train_ann or "./data_root/msrvtt/msrvtt_train_9k.json"
    args.output_dir = args.output_dir or output_dir or "./data_root/msrvtt/"
    args.output_name = args.output_name or output_name or "lvt_clip_embeds_train.pt"
    args.lvt_model_path = args.lvt_model_path or DEFAULT_LVT_MODEL
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage-1 (VideoPrism-LvT) embeddings of the training set, for the "
                    "training-time prior.")
    parser.add_argument("--config", default=None,
                        help="config whose paths fill the arguments below")
    parser.add_argument("--lvt_model_path", default=None,
                        help=f"stage-1 model (default: the config's clip_model_path, "
                             f"otherwise {DEFAULT_LVT_MODEL})")
    parser.add_argument("--video_root", default=None,
                        help="default: the config's video root, otherwise ./data_root/msrvtt/videos/")
    parser.add_argument("--train_ann", default=None,
                        help="default: the config's training annotations, otherwise "
                             "./data_root/msrvtt/msrvtt_train_9k.json")
    parser.add_argument("--output_dir", default=None,
                        help="default: the folder of the config's lvt_embeds_path, otherwise ./data_root/msrvtt/")
    parser.add_argument("--output_name", default=None,
                        help="default: the file name of the config's lvt_embeds_path, otherwise "
                             "lvt_clip_embeds_train.pt")
    parser.add_argument("--device", default=None,
                        help="default: cuda when available, otherwise cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=16,
                        help="frames per video for the stage-1 model")
    return parser.parse_args(argv)


def main(args):
    args = resolve_paths(args)
    device = torch.device(args.device)

    # Load training annotations
    with open(args.train_ann) as f:
        annotations = json.load(f)

    # Collect unique videos and all captions (flattened in same order as dataset)
    unique_videos = []
    video_id_to_idx = {}
    all_captions = []
    caption_to_video_idx = []
    vid_to_caption_indices = {}  # video_id -> list of global caption indices

    for ann in annotations:
        vid = ann.get("video_id", ann["video"])
        video_file = ann["video"]
        if vid not in video_id_to_idx:
            video_id_to_idx[vid] = len(unique_videos)
            unique_videos.append(video_file)
            vid_to_caption_indices[vid] = []

        captions = ann["caption"] if isinstance(ann["caption"], list) else [ann["caption"]]
        for cap in captions:
            cap_idx = len(all_captions)
            all_captions.append(cap)
            caption_to_video_idx.append(video_id_to_idx[vid])
            vid_to_caption_indices[vid].append(cap_idx)

    print(f"Unique videos: {len(unique_videos)}")
    print(f"Total captions: {len(all_captions)}")

    # Load LvT model
    print(f"Loading LvT CLIP model from {args.lvt_model_path}...")
    from vedje.lvt import load_lvt_model_fixed
    lvt_model, lvt_tokenizer = load_lvt_model_fixed(args.lvt_model_path, device=device, dtype=torch.float32)

    # Compute video features
    vid_feats = compute_video_features(
        lvt_model, unique_videos, args.video_root, device,
        num_frames=args.num_frames, batch_size=args.batch_size,
    )
    print(f"Video features: {vid_feats.shape}")

    # Compute text features
    text_feats = compute_text_features(
        lvt_model, lvt_tokenizer, all_captions, device,
    )
    print(f"Text features: {text_feats.shape}")

    del lvt_model, lvt_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save the embeddings; a full score matrix would be too large.
    output = {
        "video_embeds": vid_feats,             # (num_unique_videos, 768)
        "text_embeds": text_feats,             # (num_total_captions, 768)
        "video_ids": list(video_id_to_idx.keys()),
        "video_id_to_idx": video_id_to_idx,
        "caption_to_video_idx": caption_to_video_idx,
        "vid_to_caption_indices": vid_to_caption_indices,
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    torch.save(output, output_path)
    print(f"Saved to {output_path}")

    # Print some stats
    pos_scores = []
    for cap_idx, vid_idx in enumerate(caption_to_video_idx):
        score = (text_feats[cap_idx].float() @ vid_feats[vid_idx].float()).item()
        pos_scores.append(score)
    pos_scores = torch.tensor(pos_scores)
    print(f"\nPositive pair scores: mean={pos_scores.mean():.4f}, std={pos_scores.std():.4f}, "
          f"min={pos_scores.min():.4f}, max={pos_scores.max():.4f}")

    # Random negative scores
    neg_scores = []
    for cap_idx in range(min(1000, len(all_captions))):
        vid_idx = caption_to_video_idx[cap_idx]
        rand_vid = random.randint(0, len(unique_videos) - 1)
        if rand_vid != vid_idx:
            neg_scores.append((text_feats[cap_idx].float() @ vid_feats[rand_vid].float()).item())
    neg_scores = torch.tensor(neg_scores)
    print(f"Random neg scores:   mean={neg_scores.mean():.4f}, std={neg_scores.std():.4f}, "
          f"min={neg_scores.min():.4f}, max={neg_scores.max():.4f}")


if __name__ == "__main__":
    main(parse_args())
