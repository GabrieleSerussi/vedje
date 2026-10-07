"""Step 1. Index videos once: frozen VideoPrism-B patch features.

The videos are listed from the config's train and test annotations. Each video
is saved as <name>.pt with
    local_patches: (T*256, 768) bf16 patch tokens X_1..X_T
    v_global:      (768,)       bf16 L2-normalized mean of the patch tokens
Files that already exist are skipped, so an interrupted run can be resumed.

Usage (one process, or one process per GPU under torchrun):
    python scripts/extract_features.py --config configs/vedje_vp_msrvtt.yaml
    torchrun --nproc_per_node=N scripts/extract_features.py \
        --config configs/vedje_vp_msrvtt.yaml --batch_size 16
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import Dataset

from vedje.features import load_frames


class VideoListDataset(Dataset):
    def __init__(self, rel_paths, out_names, video_root, num_frames, frame_size):
        self.rel_paths = rel_paths
        self.out_names = out_names
        self.video_root = video_root
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, idx):
        vpath = os.path.join(self.video_root, self.rel_paths[idx])
        try:
            frames = load_frames(vpath, self.num_frames)  # (T, 3, 288, 288) in [0, 1]
            return frames, self.out_names[idx], True
        except Exception:
            dummy = torch.zeros(self.num_frames, 3, self.frame_size, self.frame_size)
            return dummy, self.out_names[idx], False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract frozen VideoPrism-B patch features once per video (indexing).")
    parser.add_argument("--config", required=True, help="a VideoPrism config from configs/")
    parser.add_argument("--output_dir", default=None,
                        help="where to write the .pt files (default: the config's precomputed "
                             "features directory for its dataset)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None,
                        help="device without torchrun (default: cuda when available, otherwise cpu)")
    return parser.parse_args(argv)


def main(args):
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from tqdm import tqdm

    from vedje.config import load_config
    from vedje.data import DATASET_REGISTRY
    from vedje.features import extract_patch_features, load_backbone

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        use_cuda = torch.cuda.is_available()
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
        if use_cuda:
            torch.cuda.set_device(device)
    else:
        rank, world_size = 0, 1
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    config = load_config(args.config)
    if config.get("vision_encoder", "videoprism") != "videoprism":
        raise ValueError("extract_features.py extracts VideoPrism features only")

    num_frames = config.get("num_frames", 16)
    dataset_kind = config["train_dataset"]
    reg = DATASET_REGISTRY[dataset_kind]
    video_root = config[reg["video_key"]]
    ann_keys = [reg["train_ann_key"], reg["test_ann_key"]]

    output_dir = args.output_dir or config.get(reg["features_key"],
                                               config.get("precomputed_features_dir", ""))
    if not output_dir:
        raise ValueError("pass --output_dir or set precomputed_features_dir in the config")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Unique videos across train + test: output .pt name -> relative input path
    video_entries = {}
    for ann_key in ann_keys:
        ann_path = config.get(ann_key, "")
        if not (ann_path and os.path.exists(ann_path)):
            continue
        with open(ann_path) as f:
            for item in json.load(f):
                rel = item["video"]
                if dataset_kind == "activitynet":
                    out_name = f"{item['video_id']}.pt"
                else:
                    out_name = os.path.splitext(os.path.basename(rel))[0] + ".pt"
                video_entries.setdefault(out_name, rel)

    existing = set(os.listdir(output_dir))
    todo = [(o, r) for o, r in sorted(video_entries.items()) if o not in existing]
    if rank == 0:
        print(f"Total videos: {len(video_entries)}, remaining: {len(todo)}, "
              f"processes: {world_size}, device: {device}, batch_size: {args.batch_size}")
    if not todo:
        if distributed:
            dist.destroy_process_group()
        return

    backbone = load_backbone(
        config["vision_encoder_path"],
        attn_implementation=config.get("vp_attn_implementation", "eager"),
        device=device,
    )

    dataset = VideoListDataset(
        [r for _, r in todo], [o for o, _ in todo], video_root, num_frames, frame_size=288,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=4, pin_memory=True)

    extracted = 0
    for frames, out_names, valid in tqdm(loader, desc=f"rank {rank}", disable=(rank != 0)):
        patch_tokens, v_global = extract_patch_features(backbone, frames)
        for j, out_name in enumerate(out_names):
            if not valid[j]:
                continue
            torch.save({
                "local_patches": patch_tokens[j].to(torch.bfloat16).cpu(),
                "v_global": v_global[j].to(torch.bfloat16).cpu(),
            }, os.path.join(output_dir, out_name))
            extracted += 1

    if distributed:
        dist.barrier()
    if rank == 0:
        print(f"Done: {len(os.listdir(output_dir))} feature files in {output_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
