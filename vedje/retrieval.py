"""
Two-stage video retrieval evaluation.

Stage 1: VideoPrism-LvT (or precomputed stage-1) cosine similarity -> top-K candidates
Stage 2: VEDJE joint reranker over the cached video tokens of the top-K

Reports R@1, R@5, R@10 for both T2V and V2T.
"""

import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vedje.lvt import text_embeddings, video_embeddings
from vedje.model import VideoPretrainModel

K_LIST = [1, 5, 10]


# ------------------------------------------------------------------
# Recall@K
# ------------------------------------------------------------------

def _target_set(target) -> set:
    if torch.is_tensor(target):
        target = target.tolist()
    if isinstance(target, (int, np.integer)):
        return {int(target)}
    return {int(t) for t in target}


def _hit(ranked: Sequence[int], targets: set, k: int) -> int:
    return int(any(r in targets for r in ranked[:k]))


def recall_at_k(rankings: Iterable[Sequence[int]], targets: Iterable,
                ks: Sequence[int] = K_LIST) -> dict:
    """Recall@K over queries, rounded to three decimals.

    Args:
        rankings: one ranked list of item indices per query (a list of lists or a 2D tensor)
        targets: per query, the correct item index or a collection of correct indices
        ks: cutoffs
    Returns:
        {"R@k": fraction of queries with a correct item among their first k items}
    """
    hits = {f"R@{k}": 0 for k in ks}
    num = 0
    for ranked, target in zip(rankings, targets):
        ranked = [int(r) for r in (ranked.tolist() if torch.is_tensor(ranked) else ranked)]
        target_set = _target_set(target)
        for k in ks:
            hits[f"R@{k}"] += _hit(ranked, target_set, k)
        num += 1
    return {key: round(v / num, 3) for key, v in hits.items()}


# ------------------------------------------------------------------
# Stage 1: CLIP-style features (LvT or learned VTC)
# ------------------------------------------------------------------

class _RawVideoDataset(Dataset):
    """Test videos read the way the public VideoPrism evaluation reads them."""

    def __init__(self, paths, root, num_frames, frame_size):
        self.paths = paths
        self.root = root
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # mediapy-based loading matches the public VideoPrism eval pipeline
        # (read_video + center-crop + resize_video + endpoint=False sampling).
        import mediapy
        video_path = os.path.join(self.root, self.paths[idx])
        frames = mediapy.read_video(video_path)
        n = len(frames)
        idx_np = np.linspace(0, n, num=self.num_frames, endpoint=False, dtype=np.int32)
        frames = np.array([frames[i] for i in idx_np])
        h, w = frames.shape[1], frames.shape[2]
        s = min(h, w)
        y, x = (h - s) // 2, (w - s) // 2
        frames = frames[:, y:y + s, x:x + s, :]
        frames = mediapy.resize_video(frames, shape=(self.frame_size, self.frame_size))
        frames = mediapy.to_float01(frames).astype(np.float32)
        # (T,H,W,C) -> (T,C,H,W)
        frames = np.transpose(frames, (0, 3, 1, 2))
        return torch.from_numpy(frames), idx


@torch.inference_mode()
def compute_lvt_video_features(
    lvt_model,
    test_dataset: Dataset,
    device: torch.device,
    config: dict,
    dataset_name: str = None,
) -> torch.Tensor:
    """Compute LvT video embeddings for stage 1 CLIP retrieval.

    Returns:
        vid_feats: (N_videos, clip_dim) L2-normed CLIP features
    """
    # An eval dataset with precomputed features provides no raw frames, so the
    # LvT features are computed from a raw-video loader.
    dataset_uses_precomputed = getattr(test_dataset, 'use_precomputed', config.get('use_precomputed_features', False))
    if dataset_uses_precomputed:
        from vedje.data import DATASET_REGISTRY
        ds_name = dataset_name or config.get('test_set', 'msrvtt')
        if isinstance(ds_name, list):
            ds_name = ds_name[0]
        reg = DATASET_REGISTRY[ds_name]
        video_root = config[reg['video_key']]
        video_paths = test_dataset.video

        raw_dataset = _RawVideoDataset(
            video_paths, video_root,
            num_frames=config.get('num_frames', 16),
            frame_size=288,
        )
        loader = DataLoader(
            raw_dataset,
            batch_size=config.get('eval_batch_size', 16),
            shuffle=False, drop_last=False, num_workers=4,
        )
    else:
        loader = DataLoader(
            test_dataset,
            batch_size=config.get('eval_batch_size', 16),
            shuffle=False, drop_last=False, num_workers=4,
        )

    all_vid_feats = []
    for batch in tqdm(loader, desc="Computing LvT video features"):
        frames, indices = batch[:2]
        frames = frames.to(device, dtype=torch.float32, non_blocking=True)
        video_out = lvt_model.video_model(pixel_values_videos=frames)
        pooled = video_embeddings(video_out)
        # If pooled (B, 1, D), squeeze; else (B, T, D) -> mean pool
        if pooled.dim() == 3 and pooled.shape[1] == 1:
            vid_feat = pooled.squeeze(1)
        else:
            vid_feat = pooled.mean(dim=1)
        vid_feat = F.normalize(vid_feat.float(), dim=-1)
        all_vid_feats.append(vid_feat.cpu())

    return torch.cat(all_vid_feats, dim=0)


@torch.inference_mode()
def compute_lvt_text_features(
    lvt_model,
    lvt_tokenizer,
    texts: list,
    device: torch.device,
    batch_size: int = 64,
    prompt_template: str = "",
) -> torch.Tensor:
    """Compute LvT text embeddings for stage 1 CLIP retrieval.

    Returns:
        text_feats: (N_texts, clip_dim) L2-normed CLIP features
    """
    if prompt_template:
        texts = [prompt_template.format(t) for t in texts]
    all_feats = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        inputs = lvt_tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        text_out = lvt_model.text_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )
        text_feats = text_embeddings(text_out)  # (B, clip_dim), L2-normed
        all_feats.append(text_feats.cpu())

    return torch.cat(all_feats, dim=0)


# ------------------------------------------------------------------
# Pretrain model features (for cross-encoder reranking)
# ------------------------------------------------------------------

@torch.inference_mode()
def compute_video_features(
    model: VideoPretrainModel,
    test_dataset: Dataset,
    device: torch.device,
    config: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract video global features and patch token features.

    Returns:
        vid_feats: (N_videos, clip_dim) L2-normed global features
        vision_tokens: (N_videos, num_queries, bert_dim) projected tokens (the caches)
    """
    use_precomputed = getattr(test_dataset, 'use_precomputed', config.get('use_precomputed_features', False))
    loader = DataLoader(
        test_dataset,
        batch_size=config.get('eval_batch_size', 16),
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    all_vid_feats = []
    all_vision_tokens = []

    for batch in tqdm(loader, desc="Computing video features"):
        if use_precomputed:
            patch_tokens, vid_feat, indices = batch
            patch_tokens = patch_tokens.to(device, non_blocking=True)
            vid_feat = vid_feat.to(device, non_blocking=True)
        else:
            frames, indices = batch
            frames = frames.to(device, non_blocking=True)
            patch_tokens, vid_feat = model._encode_vision_online(frames)

        vision_tokens = model.vision_projection(patch_tokens)

        all_vid_feats.append(F.normalize(vid_feat, dim=-1).cpu())
        all_vision_tokens.append(vision_tokens.cpu())

    vid_feats = torch.cat(all_vid_feats, dim=0)
    vision_tokens = torch.cat(all_vision_tokens, dim=0)
    return vid_feats, vision_tokens


@torch.inference_mode()
def compute_text_features_vtc(
    model,
    texts: list,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Compute text features using learned VTC text_projection (fallback)."""
    all_feats = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        text_inputs = model.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        outputs = model.language_model(**text_inputs, output_hidden_states=True)
        cls_hidden = outputs.hidden_states[-1][:, 0, :]
        text_feats = model.text_projection(cls_hidden)
        text_feats = F.normalize(text_feats, dim=-1)
        all_feats.append(text_feats.cpu())
    return torch.cat(all_feats, dim=0)


def get_top_k_with_scores(query_feats: torch.Tensor, gallery_feats: torch.Tensor, k: int):
    """Return top-K indices and their cosine similarity scores."""
    scores = query_feats @ gallery_feats.T
    topk = torch.topk(scores, k=k, dim=1)
    return topk.indices, topk.values


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.inference_mode()
def evaluation_t2v(
    model: VideoPretrainModel,
    test_dataset: Dataset,
    device: torch.device,
    config: dict,
    vid_feats: Optional[torch.Tensor] = None,
    text_feats: Optional[torch.Tensor] = None,
    vision_tokens: Optional[torch.Tensor] = None,
) -> Tuple[dict, dict]:
    """Text-to-Video retrieval: for each caption, find the correct video.

    Returns:
        (clip_metrics, reranked_metrics), each a dict with R@1, R@5, R@10.
    """
    k = config.get('k', 10)

    # Compute features if not provided
    if vision_tokens is None:
        vid_feats_computed, vision_tokens = compute_video_features(model, test_dataset, device, config)
        if vid_feats is None:
            vid_feats = vid_feats_computed
    if vid_feats is None:
        vid_feats, _ = compute_video_features(model, test_dataset, device, config)
    if text_feats is None:
        text_feats = compute_text_features_vtc(model, test_dataset.text, device)

    top_k_indices, top_k_scores = get_top_k_with_scores(text_feats, vid_feats, k)

    stage1_rankings, reranked_rankings, targets = [], [], []
    stage1_hits1 = reranked_hits1 = 0

    pbar = tqdm(total=len(test_dataset.text), desc="T2V Eval")

    for txt_idx in range(len(test_dataset.text)):
        target_vid_idx = test_dataset.txt2vid[txt_idx]
        candidate_idxs = top_k_indices[txt_idx]
        clip_scores = top_k_scores[txt_idx].float()
        # Z-score normalize CLIP scores within candidates (matches training normalization)
        cs_std = clip_scores.std()
        if cs_std > 1e-8:
            clip_scores = (clip_scores - clip_scores.mean()) / cs_std

        # Stage 2: rerank with CLIP scores as input
        candidate_tokens = vision_tokens[candidate_idxs].to(device)
        text = test_dataset.text[txt_idx]
        reranker_scores = model._chunked_logits(
            captions=[text] * candidate_tokens.shape[0],
            vision_tokens=candidate_tokens,
            clip_scores=clip_scores.to(device),
        ).cpu().float()

        # Algorithm 1, line 10: the candidates are sorted by s_theta
        reranked_order = [
            idx.item() for idx, _ in
            sorted(zip(candidate_idxs, reranker_scores), key=lambda x: x[1], reverse=True)
        ]

        stage1_order = candidate_idxs.tolist()
        stage1_rankings.append(stage1_order)
        reranked_rankings.append(reranked_order)
        targets.append(target_vid_idx)
        stage1_hits1 += _hit(stage1_order, {target_vid_idx}, 1)
        reranked_hits1 += _hit(reranked_order, {target_vid_idx}, 1)

        pbar.set_description(
            f"CLIP R@1: {stage1_hits1/(txt_idx+1):.4f} | "
            f"Reranked R@1: {reranked_hits1/(txt_idx+1):.4f}"
        )
        pbar.update(1)

    pbar.close()

    clip_metrics = recall_at_k(stage1_rankings, targets, K_LIST)
    reranked_metrics = recall_at_k(reranked_rankings, targets, K_LIST)
    print(f"T2V CLIP Metrics: {clip_metrics}")
    print(f"T2V Reranked Metrics: {reranked_metrics}")
    return clip_metrics, reranked_metrics


@torch.inference_mode()
def evaluation_v2t(
    model: VideoPretrainModel,
    test_dataset: Dataset,
    device: torch.device,
    config: dict,
    vid_feats: Optional[torch.Tensor] = None,
    text_feats: Optional[torch.Tensor] = None,
    vision_tokens: Optional[torch.Tensor] = None,
) -> Tuple[dict, dict]:
    """Video-to-Text retrieval: for each video, find the correct captions.

    Returns:
        (clip_metrics, reranked_metrics), each a dict with R@1, R@5, R@10.
    """
    k = config.get('k', 10)

    # Compute features if not provided
    if vision_tokens is None:
        vid_feats_computed, vision_tokens = compute_video_features(model, test_dataset, device, config)
        if vid_feats is None:
            vid_feats = vid_feats_computed
    if vid_feats is None:
        vid_feats, _ = compute_video_features(model, test_dataset, device, config)
    if text_feats is None:
        text_feats = compute_text_features_vtc(model, test_dataset.text, device)

    top_k_indices, top_k_scores = get_top_k_with_scores(vid_feats, text_feats, k)

    stage1_rankings, reranked_rankings, targets = [], [], []
    stage1_hits1 = reranked_hits1 = 0

    pbar = tqdm(total=len(test_dataset.video), desc="V2T Eval")

    for vid_idx in range(len(test_dataset.video)):
        target_txt_idxs = test_dataset.vid2txt[vid_idx]
        candidate_txt_idxs = top_k_indices[vid_idx]
        clip_scores = top_k_scores[vid_idx].float()
        # Z-score normalize CLIP scores within candidates (matches training normalization)
        cs_std = clip_scores.std()
        if cs_std > 1e-8:
            clip_scores = (clip_scores - clip_scores.mean()) / cs_std

        # Stage 2: rerank with CLIP scores as input
        current_vision = vision_tokens[vid_idx:vid_idx + 1].to(device)
        captions_to_rerank = [test_dataset.text[ci] for ci in candidate_txt_idxs]
        reranker_scores = model._chunked_logits(
            captions=captions_to_rerank,
            vision_tokens=current_vision.expand(len(captions_to_rerank), -1, -1),
            clip_scores=clip_scores.to(device),
        ).cpu().float()

        # Algorithm 1, line 10: the candidates are sorted by s_theta
        reranked_order = [
            idx.item() for idx, _ in
            sorted(zip(candidate_txt_idxs, reranker_scores), key=lambda x: x[1], reverse=True)
        ]

        stage1_order = candidate_txt_idxs.tolist()
        target_set = _target_set(target_txt_idxs)
        stage1_rankings.append(stage1_order)
        reranked_rankings.append(reranked_order)
        targets.append(target_txt_idxs)
        stage1_hits1 += _hit(stage1_order, target_set, 1)
        reranked_hits1 += _hit(reranked_order, target_set, 1)

        pbar.set_description(
            f"CLIP R@1: {stage1_hits1/(vid_idx+1):.4f} | "
            f"Reranked R@1: {reranked_hits1/(vid_idx+1):.4f}"
        )
        pbar.update(1)

    pbar.close()

    clip_metrics = recall_at_k(stage1_rankings, targets, K_LIST)
    reranked_metrics = recall_at_k(reranked_rankings, targets, K_LIST)
    print(f"V2T CLIP Metrics: {clip_metrics}")
    print(f"V2T Reranked Metrics: {reranked_metrics}")
    return clip_metrics, reranked_metrics
