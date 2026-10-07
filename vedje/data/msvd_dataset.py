import os
import json
import random

import torch
from torch.utils.data import Dataset

from vedje.data.utils import pre_caption, load_video_frames, load_precomputed_features, vtc_target


class msvd_train(Dataset):
    """MSVD training dataset for video-text retrieval.

    Annotation format:
        {"video_id": "...", "video": "....avi", "caption": ["desc1", ...]}
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

        self.samples = []
        self.video_ids = {}
        self._vid_to_captions = {}
        n = 0
        for ann in self.annotation:
            vid = ann["video_id"]
            if vid not in self.video_ids:
                self.video_ids[vid] = n
                n += 1
            captions = ann["caption"] if isinstance(ann["caption"], list) else [ann["caption"]]
            self._vid_to_captions[vid] = [pre_caption(c) for c in captions]
            for cap in captions:
                self.samples.append({
                    "video": ann["video"],
                    "video_id": vid,
                    "caption": cap,
                })

        # Precomputed LvT CLIP embeddings
        self._lvt_video_embeds = None
        self._lvt_text_embeds = None
        self._lvt_vid_to_cap_indices = None
        if lvt_embeds_path and os.path.exists(lvt_embeds_path):
            lvt_data = torch.load(lvt_embeds_path, map_location="cpu")
            self._lvt_video_embeds = lvt_data["video_embeds"].float()
            self._lvt_text_embeds = lvt_data["text_embeds"].float()
            self._lvt_vid_to_cap_indices = lvt_data["vid_to_caption_indices"]
            self._lvt_video_id_to_idx = lvt_data["video_id_to_idx"]
            print(f"[MSVD] Loaded LvT CLIP embeddings: "
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
                self.hard_negatives = {
                    k: [v for v in vs if v in self.video_ids]
                    for k, vs in raw_negs.items()
                    if k in self.video_ids
                }
                print(f"[MSVD] Loaded per-video hard negatives for "
                      f"{len(self.hard_negatives)} videos, K={num_hard_negatives}")
            else:
                self._hard_neg_per_caption = True
                self.hard_negatives = {
                    k: [v for v in vs if v in self.video_ids]
                    for k, vs in raw_negs.items()
                }
                print(f"[MSVD] Loaded per-caption hard negatives for "
                      f"{len(self.hard_negatives)} captions, K={num_hard_negatives}")

        # Pre-load all features into memory (eliminates per-step disk I/O)
        if self.use_precomputed and os.environ.get("VEDJE_DISABLE_PRELOAD", "") != "1":
            self._preload_all_features()

    def _preload_all_features(self):
        """Pre-load ALL precomputed features into RAM to avoid per-step I/O."""
        import time
        t0 = time.time()
        self._feature_cache = {}            # keyed by video filename (e.g. "xyz.avi")
        self._feature_cache_by_id = {}      # keyed by video_id (e.g. "xyz")
        vid_to_file = {}
        for s in self.samples:
            vid_to_file[s["video_id"]] = s["video"]
        for i, (vid_id, vid_file) in enumerate(sorted(vid_to_file.items())):
            pt_path = os.path.join(self.precomputed_dir, self._video_to_pt(vid_file))
            feats = load_precomputed_features(pt_path)
            self._feature_cache[vid_file] = feats
            self._feature_cache_by_id[vid_id] = feats
            if (i + 1) % 500 == 0:
                print(f"  [MSVD Pre-load] {i+1}/{len(vid_to_file)} videos ({time.time()-t0:.0f}s)")
        print(f"  [MSVD Pre-load] Done: {len(vid_to_file)} videos in {time.time()-t0:.0f}s")

    def __len__(self):
        return len(self.samples)

    def _video_to_pt(self, video_name):
        """Convert video filename to .pt feature path."""
        return os.path.splitext(video_name)[0] + ".pt"

    def __getitem__(self, index):
        sample = self.samples[index]
        caption = pre_caption(sample["caption"])
        video_id_idx = self.video_ids[sample["video_id"]]

        if self.use_precomputed:
            if hasattr(self, '_feature_cache') and sample["video"] in self._feature_cache:
                patch_tokens, vid_feat = self._feature_cache[sample["video"]]
            else:
                pt_path = os.path.join(
                    self.precomputed_dir, self._video_to_pt(sample["video"]),
                )
                patch_tokens, vid_feat = load_precomputed_features(pt_path)

            vid_feat = vtc_target(self, sample["video_id"], vid_feat)

            if self.hard_negatives:
                neg_patches, neg_vid_feats, neg_captions, clip_scores = self._load_hard_negatives(
                    sample["video_id"], index
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
        """Load K hard negative video features, captions, and LvT CLIP scores."""
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

        # Mix hard and random negatives (50/50)
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
            if hasattr(self, '_feature_cache_by_id') and neg_vid in self._feature_cache_by_id:
                neg_patch, neg_vf = self._feature_cache_by_id[neg_vid]
            else:
                # MSVD: video_id matches feature filename (e.g. "xyz_0_10" -> "xyz_0_10.pt")
                pt_path = os.path.join(self.precomputed_dir, f"{neg_vid}.pt")
                neg_patch, neg_vf = load_precomputed_features(pt_path)
            neg_patches_list.append(neg_patch)
            neg_vid_feats_list.append(neg_vf)

            if has_lvt and neg_vid in self._lvt_vid_to_cap_indices:
                neg_lvt_idx = self._lvt_video_id_to_idx[neg_vid]
                neg_cap_idxs = self._lvt_vid_to_cap_indices[neg_vid]
                neg_cap_idx = random.choice(neg_cap_idxs)
                neg_clip_scores_v.append(
                    (self._lvt_text_embeds[sample_idx] @ self._lvt_video_embeds[neg_lvt_idx]).item()
                )
                if pos_vid_lvt_idx is not None:
                    neg_clip_scores_t.append(
                        (self._lvt_text_embeds[neg_cap_idx] @ self._lvt_video_embeds[pos_vid_lvt_idx]).item()
                    )
                else:
                    neg_clip_scores_t.append(0.0)
                neg_captions_list.append(pre_caption(
                    self._vid_to_captions.get(neg_vid, [""])[0]
                ))
            else:
                neg_captions_list.append(pre_caption(
                    self._vid_to_captions.get(neg_vid, [""])[0]
                ))
                neg_clip_scores_v.append(0.0)
                neg_clip_scores_t.append(0.0)

        neg_patches = torch.stack(neg_patches_list)
        neg_vid_feats = torch.stack(neg_vid_feats_list)
        # Pack CLIP scores: [pos_clip_v, neg_clip_v..., neg_clip_t...]
        if has_lvt and pos_vid_lvt_idx is not None:
            pos_clip = (self._lvt_text_embeds[sample_idx] @ self._lvt_video_embeds[pos_vid_lvt_idx]).item()
        else:
            pos_clip = 0.0
        clip_scores = torch.tensor(
            [pos_clip] + neg_clip_scores_v + neg_clip_scores_t,
            dtype=torch.float32,
        )

        return neg_patches, neg_vid_feats, neg_captions_list, clip_scores


class msvd_retrieval_eval(Dataset):
    """MSVD evaluation dataset for video-text retrieval."""

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
            self.annotation = json.load(f)

        self.text = []
        self.raw_text = []
        self.video = []
        self.txt2vid = {}
        self.vid2txt = {}

        txt_id = 0
        for vid_id, ann in enumerate(self.annotation):
            self.video.append(ann["video"])
            self.vid2txt[vid_id] = []
            captions = ann["caption"] if isinstance(ann["caption"], list) else [ann["caption"]]
            for cap in captions:
                self.text.append(pre_caption(cap))
                self.raw_text.append(cap)
                self.vid2txt[vid_id].append(txt_id)
                self.txt2vid[txt_id] = vid_id
                txt_id += 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        if self.use_precomputed:
            basename = os.path.splitext(ann["video"])[0] + ".pt"
            pt_path = os.path.join(self.precomputed_dir, basename)
            patch_tokens, vid_feat = load_precomputed_features(pt_path)
            return patch_tokens, vid_feat, index
        else:
            video_path = os.path.join(self.video_root, ann["video"])
            frames = load_video_frames(
                video_path, self.num_frames,
                size=self.frame_size, normalize=self.normalize,
            )
            return frames, index
