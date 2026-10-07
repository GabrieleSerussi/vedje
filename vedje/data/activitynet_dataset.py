"""ActivityNet Captions dataset for video-text retrieval.

Closely mirrors vedje/data/msrvtt_dataset.py, with three changes:
  - The `video` field in annotations is a path relative to the video root
    (`activitynet_videos_dir`, e.g. "v1-3/train_val/v_XXX.mp4").
  - Precomputed features are keyed by the bare video_id (".pt" filename).
  - The annotation files follow `activitynet_retrieval_mode`: "paragraph"
    (default) or "sentence".
"""
import os
import json
import random

import torch
from torch.utils.data import Dataset

from vedje.data.utils import pre_caption, load_video_frames, load_precomputed_features, vtc_target


def _feature_pt_name(video_id: str) -> str:
    """Map a video_id to the .pt filename used for precomputed features.

    MSR-VTT uses "<video_id>.pt" keyed by the mp4 filename stem. For
    ActivityNet videos are "v_XXXX.mp4" so the video_id IS the stem.
    """
    return f"{video_id}.pt"


class activitynet_train(Dataset):
    """ActivityNet training dataset for video-text retrieval.

    Annotation format (paragraph mode):
        {"video_id": "v_XXXX",
         "video": "v1-3/train_val/v_XXXX.mp4",  (relative to video_root)
         "caption": ["paragraph"],              (list for train)
         "source": "ActivityNet"}

    In sentence mode `caption` is a list of individual sentences.
    """

    def __init__(self, video_root: str, ann_path: str, num_frames: int = 8,
                 precomputed_dir: str = "", frame_size: int = 224,
                 normalize: bool = True,
                 hard_negatives_path: str = "",
                 num_hard_negatives: int = 3,
                 hard_neg_pool_size: int = 50,
                 lvt_embeds_path: str = ""):
        self.video_root = video_root
        self.num_frames = num_frames
        self.precomputed_dir = precomputed_dir
        self.use_precomputed = bool(precomputed_dir)
        self.frame_size = frame_size
        self.normalize = normalize
        self.num_hard_negatives = num_hard_negatives
        self.hard_neg_pool_size = hard_neg_pool_size

        with open(ann_path, "r") as f:
            self.annotation = json.load(f)

        # Flatten: one entry per (video, caption) pair for training
        self.samples = []
        self.video_ids = {}
        self._vid_to_captions = {}  # video_id -> list of captions
        self._vid_to_video_path = {}  # video_id -> relative mp4 path
        n = 0
        for ann in self.annotation:
            vid = ann["video_id"]
            if vid not in self.video_ids:
                self.video_ids[vid] = n
                n += 1
                self._vid_to_video_path[vid] = ann["video"]
            captions = ann["caption"] if isinstance(ann["caption"], list) else [ann["caption"]]
            self._vid_to_captions[vid] = [pre_caption(c) for c in captions]
            for cap in captions:
                self.samples.append({
                    "video": ann["video"],
                    "video_id": vid,
                    "caption": cap,
                })

        # Precomputed LvT CLIP embeddings for train/eval consistent CLIP scores
        self._lvt_video_embeds = None
        self._lvt_text_embeds = None
        self._lvt_vid_to_cap_indices = None
        if lvt_embeds_path and os.path.exists(lvt_embeds_path):
            lvt_data = torch.load(lvt_embeds_path, map_location="cpu")
            self._lvt_video_embeds = lvt_data["video_embeds"].float()
            self._lvt_text_embeds = lvt_data["text_embeds"].float()
            self._lvt_vid_to_cap_indices = lvt_data["vid_to_caption_indices"]
            self._lvt_video_id_to_idx = lvt_data["video_id_to_idx"]
            print(f"[ActivityNet] Loaded LvT CLIP embeddings: "
                  f"{self._lvt_video_embeds.shape[0]} videos, "
                  f"{self._lvt_text_embeds.shape[0]} captions")

        # CLIP-mined hard negatives
        self.hard_negatives = {}
        self._hard_neg_per_caption = False
        if hard_negatives_path and os.path.exists(hard_negatives_path):
            with open(hard_negatives_path, "r") as f:
                raw_negs = json.load(f)
            first_key = next(iter(raw_negs))
            if first_key in self.video_ids:
                # Per-video
                self.hard_negatives = {
                    k: [v for v in vs if v in self.video_ids]
                    for k, vs in raw_negs.items()
                    if k in self.video_ids
                }
                print(f"[ActivityNet] Loaded per-video hard negatives for "
                      f"{len(self.hard_negatives)} videos, K={num_hard_negatives}")
            else:
                # Per-caption
                self._hard_neg_per_caption = True
                self.hard_negatives = {
                    k: [v for v in vs if v in self.video_ids]
                    for k, vs in raw_negs.items()
                }
                print(f"[ActivityNet] Loaded per-caption hard negatives for "
                      f"{len(self.hard_negatives)} captions, K={num_hard_negatives}")

        # Pre-load all features into memory (eliminates per-step disk I/O)
        if self.use_precomputed and os.environ.get("VEDJE_DISABLE_PRELOAD", "") != "1":
            self._preload_all_features()

    def _preload_all_features(self):
        import time
        t0 = time.time()
        self._feature_cache = {}
        # Keyed by video_id (since multiple sample entries share the same video)
        unique_vids = sorted(self._vid_to_video_path.keys())
        for i, vid in enumerate(unique_vids):
            pt_path = os.path.join(self.precomputed_dir, _feature_pt_name(vid))
            self._feature_cache[vid] = load_precomputed_features(pt_path)
            if (i + 1) % 1000 == 0:
                print(f"  [Pre-load] {i+1}/{len(unique_vids)} videos ({time.time()-t0:.0f}s)")
        print(f"  [Pre-load] Done: {len(unique_vids)} videos in {time.time()-t0:.0f}s")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        caption = pre_caption(sample["caption"])
        video_id = sample["video_id"]
        video_id_idx = self.video_ids[video_id]

        if self.use_precomputed:
            if hasattr(self, '_feature_cache') and video_id in self._feature_cache:
                patch_tokens, vid_feat = self._feature_cache[video_id]
            else:
                pt_path = os.path.join(self.precomputed_dir, _feature_pt_name(video_id))
                patch_tokens, vid_feat = load_precomputed_features(pt_path)

            vid_feat = vtc_target(self, video_id, vid_feat)

            if self.hard_negatives:
                neg_patches, neg_vid_feats, neg_captions, clip_scores = self._load_hard_negatives(
                    video_id, index
                )
                return (patch_tokens, vid_feat, caption, video_id_idx, index,
                        neg_patches, neg_vid_feats, neg_captions, clip_scores)

            return patch_tokens, vid_feat, caption, video_id_idx, index
        else:
            video_path = os.path.join(self.video_root, sample["video"])
            frames = load_video_frames(
                video_path, self.num_frames,
                size=self.frame_size, normalize=self.normalize,
            )
            return frames, caption, video_id_idx, index

    def _load_hard_negatives(self, video_id: str, sample_idx: int = 0):
        K = self.num_hard_negatives
        if self._hard_neg_per_caption:
            neg_video_ids = self.hard_negatives.get(str(sample_idx), [])
        else:
            neg_video_ids = self.hard_negatives.get(video_id, [])

        if len(neg_video_ids) < K:
            all_vids = list(self.video_ids.keys())
            while len(neg_video_ids) < K:
                rand_vid = random.choice(all_vids)
                if rand_vid != video_id:
                    neg_video_ids.append(rand_vid)

        pool = neg_video_ids[:self.hard_neg_pool_size]
        K_hard = K // 2
        K_random = K - K_hard
        hard_selected = random.sample(pool, min(K_hard, len(pool)))
        all_vids = list(self.video_ids.keys())
        random_selected = []
        hard_set = set(hard_selected)
        while len(random_selected) < K_random:
            v = random.choice(all_vids)
            if v != video_id and v not in hard_set:
                random_selected.append(v)
                hard_set.add(v)
        selected = hard_selected + random_selected

        neg_patches_list = []
        neg_vid_feats_list = []
        neg_captions_list = []
        neg_clip_scores_v = []
        neg_clip_scores_t = []

        has_lvt = self._lvt_video_embeds is not None
        pos_vid_lvt_idx = self._lvt_video_id_to_idx.get(video_id) if has_lvt else None

        for neg_vid in selected:
            if hasattr(self, '_feature_cache') and neg_vid in self._feature_cache:
                neg_patch, neg_vf = self._feature_cache[neg_vid]
            else:
                pt_path = os.path.join(self.precomputed_dir, _feature_pt_name(neg_vid))
                neg_patch, neg_vf = load_precomputed_features(pt_path)
            neg_patches_list.append(neg_patch)
            neg_vid_feats_list.append(neg_vf)

            if has_lvt and neg_vid in self._lvt_vid_to_cap_indices:
                neg_cap_indices = self._lvt_vid_to_cap_indices[neg_vid]
                cap_local_idx = random.randrange(len(neg_cap_indices))
                neg_captions_list.append(self._vid_to_captions[neg_vid][cap_local_idx])

                neg_vid_lvt_idx = self._lvt_video_id_to_idx[neg_vid]
                neg_clip_scores_v.append(
                    (self._lvt_text_embeds[sample_idx] @ self._lvt_video_embeds[neg_vid_lvt_idx]).item()
                )
                neg_global_cap_idx = neg_cap_indices[cap_local_idx]
                neg_clip_scores_t.append(
                    (self._lvt_text_embeds[neg_global_cap_idx] @ self._lvt_video_embeds[pos_vid_lvt_idx]).item()
                )
            else:
                neg_captions_list.append(random.choice(self._vid_to_captions[neg_vid]))
                neg_clip_scores_v.append(0.0)
                neg_clip_scores_t.append(0.0)

        neg_patches = torch.stack(neg_patches_list)
        neg_vid_feats = torch.stack(neg_vid_feats_list)

        if has_lvt:
            pos_clip_score = (self._lvt_text_embeds[sample_idx] @ self._lvt_video_embeds[pos_vid_lvt_idx]).item()
        else:
            pos_clip_score = 0.0

        clip_scores = torch.tensor(
            [pos_clip_score] + neg_clip_scores_v + neg_clip_scores_t,
            dtype=torch.float32,
        )
        clip_std = clip_scores.std()
        if clip_std > 1e-8:
            clip_scores = (clip_scores - clip_scores.mean()) / clip_std

        return neg_patches, neg_vid_feats, tuple(neg_captions_list), clip_scores


class activitynet_retrieval_eval(Dataset):
    """ActivityNet evaluation dataset for video-text retrieval.

    Annotation format (paragraph mode):
        {"video_id": "v_XXXX",
         "video": "v1-3/train_val/v_XXXX.mp4",
         "caption": "<paragraph>",
         "all_captions": ["<paragraph>"]}

    In sentence mode each sentence is its own entry (flattened).
    Builds text/video lists and mapping dicts for recall computation.
    """

    def __init__(self, video_root: str, ann_path: str, num_frames: int = 8,
                 precomputed_dir: str = "", frame_size: int = 224,
                 normalize: bool = True):
        self.video_root = video_root
        self.num_frames = num_frames
        self.precomputed_dir = precomputed_dir
        self.use_precomputed = bool(precomputed_dir)
        self.frame_size = frame_size
        self.normalize = normalize

        with open(ann_path, "r") as f:
            raw = json.load(f)

        # De-duplicate videos (in sentence mode multiple rows have the same video_id)
        self.video = []              # relative mp4 paths (unique videos)
        self.video_ids = []          # video_id list parallel to self.video
        self._vid_to_idx = {}        # video_id -> index in self.video
        self.text = []               # flattened caption strings, cleaned for the joint encoder
        self.raw_text = []           # the same captions as written, for the first stage
        self.txt2vid = {}            # txt_id -> vid_id
        self.vid2txt = {}            # vid_id -> [txt_ids]

        txt_id = 0
        for ann in raw:
            vid = ann["video_id"]
            if vid not in self._vid_to_idx:
                self._vid_to_idx[vid] = len(self.video)
                self.video.append(ann["video"])
                self.video_ids.append(vid)
                self.vid2txt[self._vid_to_idx[vid]] = []
            vid_idx = self._vid_to_idx[vid]
            caption = ann["caption"]
            # Single caption string per eval entry (this is the text query)
            self.text.append(pre_caption(caption))
            self.raw_text.append(caption)
            self.vid2txt[vid_idx].append(txt_id)
            self.txt2vid[txt_id] = vid_idx
            txt_id += 1

        # Keep self.annotation for compatibility (one entry per unique video)
        self.annotation = [
            {"video_id": self.video_ids[i], "video": self.video[i]}
            for i in range(len(self.video))
        ]

        if self.use_precomputed and os.environ.get("VEDJE_DISABLE_PRELOAD", "") != "1":
            self._preload_all_features()

    def _preload_all_features(self):
        import time
        t0 = time.time()
        self._feature_cache = {}
        for i, vid in enumerate(self.video_ids):
            pt_path = os.path.join(self.precomputed_dir, _feature_pt_name(vid))
            self._feature_cache[vid] = load_precomputed_features(pt_path)
            if (i + 1) % 1000 == 0:
                print(f"  [Pre-load eval] {i+1}/{len(self.video_ids)} videos ({time.time()-t0:.0f}s)")
        print(f"  [Pre-load eval] Done: {len(self.video_ids)} videos in {time.time()-t0:.0f}s")

    def __len__(self):
        return len(self.video)

    def __getitem__(self, index):
        vid = self.video_ids[index]
        if self.use_precomputed:
            if hasattr(self, '_feature_cache') and vid in self._feature_cache:
                patch_tokens, vid_feat = self._feature_cache[vid]
            else:
                pt_path = os.path.join(self.precomputed_dir, _feature_pt_name(vid))
                patch_tokens, vid_feat = load_precomputed_features(pt_path)
            return patch_tokens, vid_feat, index
        else:
            video_path = os.path.join(self.video_root, self.video[index])
            frames = load_video_frames(
                video_path, self.num_frames,
                size=self.frame_size, normalize=self.normalize,
            )
            return frames, index
