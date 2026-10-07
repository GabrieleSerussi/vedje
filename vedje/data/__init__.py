import json
import os

import torch
from torch.utils.data import DataLoader, Dataset

from vedje.config import _activitynet_paths_for_mode, _resolve_activitynet_path  # noqa: F401
from vedje.data.msrvtt_dataset import msrvtt_train, msrvtt_retrieval_eval
from vedje.data.msvd_dataset import msvd_train, msvd_retrieval_eval
from vedje.data.didemo_dataset import didemo_train, didemo_retrieval_eval
from vedje.data.activitynet_dataset import activitynet_train, activitynet_retrieval_eval

DATASET_REGISTRY = {
    "msrvtt": {
        "train": msrvtt_train,
        "eval": msrvtt_retrieval_eval,
        "video_key": "msrvtt_videos",
        "train_ann_key": "msrvtt_train_ann",
        "test_ann_key": "msrvtt_test_ann",
        "features_key": "msrvtt_precomputed_features_dir",
    },
    "activitynet": {
        "train": activitynet_train,
        "eval": activitynet_retrieval_eval,
        "video_key": "activitynet_videos_dir",
        "train_ann_key": "activitynet_train_ann",
        "test_ann_key": "activitynet_test_ann",
        "features_key": "activitynet_precomputed_features_dir",
    },
    "msvd": {
        "train": msvd_train,
        "eval": msvd_retrieval_eval,
        "video_key": "msvd_videos",
        "train_ann_key": "msvd_train_ann",
        "test_ann_key": "msvd_test_ann",
        "features_key": "msvd_precomputed_features_dir",
    },
    "didemo": {
        "train": didemo_train,
        "eval": didemo_retrieval_eval,
        "video_key": "didemo_videos",
        "train_ann_key": "didemo_train_ann",
        "test_ann_key": "didemo_test_ann",
        "features_key": "didemo_precomputed_features_dir",
    },
}


def feature_files(config: dict) -> dict:
    """{feature file name: video path} for the unique videos of the config's training and test annotations.

    One .pt file per video, named as the dataset loaders read it: <video_id>.pt for ActivityNet, otherwise the
    video's file name with .pt. The video paths are relative to the dataset's video folder.
    """
    dataset_kind = config["train_dataset"]
    reg = DATASET_REGISTRY[dataset_kind]
    video_entries = {}
    for ann_key in (reg["train_ann_key"], reg["test_ann_key"]):
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
    return video_entries


def stage1_layout(annotations: list) -> tuple:
    """Training videos and captions in the order of the stage-1 training file (lvt_embeds_path).

    Captions are flattened in annotation order, as the training datasets number their samples, and videos are
    keyed by video_id, or by the video path where there is none. Returns the unique video paths, the captions as
    written and the file's index: video_ids, video_id_to_idx, caption_to_video_idx and vid_to_caption_indices.
    """
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

    return unique_videos, all_captions, {
        "video_ids": list(video_id_to_idx.keys()),
        "video_id_to_idx": video_id_to_idx,
        "caption_to_video_idx": caption_to_video_idx,
        "vid_to_caption_indices": vid_to_caption_indices,
    }


def _vision_encoder_params(config: dict) -> dict:
    """Derive frame_size and normalize from vision_encoder config."""
    encoder = config.get("vision_encoder", "videoprism")
    is_vp = encoder == "videoprism"
    return {
        "frame_size": 288 if is_vp else 224,
        "normalize": not is_vp,
    }


def _build_single_train_dataset(name: str, config: dict, ve_params: dict) -> Dataset:
    """Build one registered train dataset, resolving its paths."""
    reg = DATASET_REGISTRY[name]

    if name == "activitynet":
        ann_path = _resolve_activitynet_path(config, "activitynet_train_ann")
        hn_path = config.get("activitynet_hard_negatives_path") or \
            _resolve_activitynet_path(config, "hard_negatives_path")
        lvt_path = config.get("activitynet_lvt_embeds_path") or \
            _resolve_activitynet_path(config, "lvt_embeds_path")
    else:
        ann_path = config[reg["train_ann_key"]]
        hn_path = config.get(f"{name}_hard_negatives_path",
                             config.get("hard_negatives_path", ""))
        lvt_path = config.get(f"{name}_lvt_embeds_path",
                              config.get("lvt_embeds_path", ""))

    precomputed_dir = config.get(reg["features_key"],
                                 config.get("precomputed_features_dir", ""))

    return reg["train"](
        video_root=config[reg["video_key"]],
        ann_path=ann_path,
        num_frames=config.get("num_frames", 8),
        precomputed_dir=precomputed_dir,
        hard_negatives_path=hn_path,
        num_hard_negatives=config.get("num_hard_negatives", 3),
        hard_neg_pool_size=config.get("hard_neg_pool_size", 50),
        lvt_embeds_path=lvt_path,
        **ve_params,
    )


def create_train_dataset(config: dict) -> Dataset:
    """Create the training dataset named by config['train_dataset']."""
    name = config.get("train_dataset", "msrvtt")
    return _build_single_train_dataset(name, config, _vision_encoder_params(config))


def create_eval_dataset(config: dict, dataset_name: str = None) -> Dataset:
    """Create an evaluation dataset."""
    name = dataset_name or config.get("test_set", "msrvtt")
    reg = DATASET_REGISTRY[name]
    ve_params = _vision_encoder_params(config)
    # Per-dataset feature dir takes priority over global precomputed_features_dir
    features_dir = config.get(reg["features_key"], config.get("precomputed_features_dir", ""))
    if name == "activitynet":
        ann_path = _resolve_activitynet_path(config, "activitynet_test_ann")
    else:
        ann_path = config[reg["test_ann_key"]]
    return reg["eval"](
        video_root=config[reg["video_key"]],
        ann_path=ann_path,
        num_frames=config.get("num_frames", 8),
        precomputed_dir=features_dir,
        **ve_params,
    )


def create_sampler(datasets, shuffles, num_replicas, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_replicas, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns,
                  worker_init_fn=None):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        drop_last = is_train
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            persistent_workers=n_worker > 0,
            prefetch_factor=4 if n_worker > 0 else None,
        )
        loaders.append(loader)
    return loaders
