"""Caption normalisation and cached-feature loading shared by the dataset loaders."""

import torch

from vedje.video import load_video_frames  # noqa: F401  (used by the loaders)

try:
    from open_clip.tokenizer import canonicalize_text, basic_clean
except ImportError:
    def basic_clean(text):
        return text.strip()

    def canonicalize_text(text):
        return text.strip().lower()


def pre_caption(caption):
    return canonicalize_text(basic_clean(caption))


def load_precomputed_features(
    pt_path: str,
) -> tuple:
    """Load one video's cached backbone features written by scripts/extract_features.py.

    Format: {"local_patches": (T*P, vision_dim), "v_global": (clip_dim,)}
    Returns (patch_tokens, vid_feat) in bf16.
    """
    feats = torch.load(pt_path, map_location="cpu")
    patch_tokens = feats["local_patches"]
    vid_feat = feats["v_global"]
    if patch_tokens.dtype == torch.float32:
        patch_tokens = patch_tokens.to(torch.bfloat16)
        vid_feat = vid_feat.to(torch.bfloat16)
    return patch_tokens, vid_feat


def vtc_target(dataset, video_key, vid_feat):
    """The contrastive target of a training video: the first stage's embedding of it (Table 7).

    The embedding comes from the stage-1 file of step 2a (lvt_embeds_path) when the dataset has
    loaded it; without that file, the global feature of the cached backbone features is kept.
    """
    if dataset._lvt_video_embeds is None:
        return vid_feat
    return dataset._lvt_video_embeds[dataset._lvt_video_id_to_idx[video_key]].to(vid_feat.dtype)
