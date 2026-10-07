"""
Step 5. Two-stage retrieval evaluation: stage-1 candidates, reranked by VEDJE
from the cached video tokens. Prints R@1, R@5 and R@10 in both directions
(text-to-video and video-to-text), for stage 1 and after reranking.

Usage:
    python scripts/evaluate.py --config configs/vedje_vp_msrvtt.yaml \
        --checkpoint ./output/vp_msrvtt/checkpoint_03.pth
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Two-stage VEDJE evaluation: R@1/5/10, text-to-video and video-to-text.")
    parser.add_argument('--config', default='./configs/vedje_vp_msrvtt.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default=None,
                        help='default: cuda when available, otherwise cpu')
    args = parser.parse_args(argv)
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


def main(args):
    from vedje.config import load_config
    from vedje.data import create_eval_dataset
    from vedje.model import build_model
    from vedje.retrieval import (
        compute_lvt_text_features, compute_lvt_video_features,
        compute_text_features_vtc, compute_video_features,
        evaluation_t2v, evaluation_v2t,
    )

    torch.set_default_dtype(torch.bfloat16)

    # ActivityNet: load_config derives the data paths from `activitynet_retrieval_mode` unless set.
    config = load_config(args.config)

    model = build_model(config, training=False)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    # strict=False tolerates the training-only future predictor; report any
    # missing trainable weights so a bad load is never mistaken for a bad result.
    trainable = {n for n, _ in model.named_parameters()}
    missing_trainable = [m for m in missing if m in trainable]
    print(f"[ckpt] loaded {args.checkpoint}")
    print(f"[ckpt] missing={len(missing)} (trainable: {len(missing_trainable)}), unexpected={len(unexpected)}")
    if missing_trainable:
        print(f"[ckpt] WARNING first missing trainable keys: {missing_trainable[:10]}")

    device = torch.device(args.device)
    model = model.to(device).eval()

    # Load LvT model for stage 1 CLIP retrieval (if using videoprism)
    lvt_model = None
    lvt_tokenizer = None
    clip_model_path = config.get('clip_model_path', '')
    if clip_model_path:
        print(f"Loading LvT CLIP model from {clip_model_path}...")
        try:
            from vedje.lvt import load_lvt_model_fixed
            lvt_model, lvt_tokenizer = load_lvt_model_fixed(
                clip_model_path, device=device, dtype=torch.float32)
            print("LvT CLIP model loaded.")
        except (ValueError, ImportError, KeyError) as e:
            print(f"Warning: Could not load LvT CLIP model: {e}")
            print("Will use precomputed LvT test features if available.")

    test_set = config.get('test_set', 'msrvtt')
    if isinstance(test_set, str):
        test_set = [test_set]

    for ds_name in test_set:
        test_dataset = create_eval_dataset(config, ds_name)
        print(f"\n{'='*60}")
        print(f"Evaluating on {ds_name} ({len(test_dataset)} videos, {len(test_dataset.text)} captions)")
        print(f"{'='*60}")

        # Compute vision tokens for cross-encoder (stage 2)
        _, vision_tokens = compute_video_features(model, test_dataset, device, config)

        # Compute CLIP features for stage 1.
        # Precomputed test features are preferred when provided (they work for any
        # backbone, including those the LvT video tower cannot read, e.g.
        # VideoCLIP-XL).
        lvt_test_path = config.get('lvt_test_features_path', '')
        if lvt_test_path and os.path.isfile(lvt_test_path):
            print(f"\nUsing precomputed LvT test features from {lvt_test_path}")
            precomputed = torch.load(lvt_test_path, map_location='cpu', weights_only=True)
            vid_feats = precomputed['vid_feats']
            text_feats = precomputed['text_feats']
        elif lvt_model is not None:
            print("\nUsing LvT model for stage 1 CLIP retrieval")
            vid_feats = compute_lvt_video_features(lvt_model, test_dataset, device, config, dataset_name=ds_name)
            # The first stage reads the captions as written (raw_text), as in step 2a:
            # the c4_en SentencePiece tokenizer of LvT is case-sensitive.
            lvt_texts = getattr(test_dataset, 'raw_text', test_dataset.text)
            text_feats = compute_lvt_text_features(
                lvt_model, lvt_tokenizer, lvt_texts, device,
                prompt_template=config.get('lvt_prompt_template', ''),
            )
        else:
            print("\nUsing learned VTC for stage 1 retrieval")
            vid_feats, _ = compute_video_features(model, test_dataset, device, config)
            text_feats = compute_text_features_vtc(model, test_dataset.text, device)

        print("\n--- Text-to-Video ---")
        t2v_clip, t2v_reranked = evaluation_t2v(
            model, test_dataset, device, config,
            vid_feats, text_feats, vision_tokens,
        )

        print("\n--- Video-to-Text ---")
        v2t_clip, v2t_reranked = evaluation_v2t(
            model, test_dataset, device, config,
            vid_feats, text_feats, vision_tokens,
        )

        print(f"\n{'='*60}")
        print(f"  {ds_name} T2V CLIP Metrics:     {t2v_clip}")
        print(f"  {ds_name} T2V Reranked Metrics: {t2v_reranked}")
        print(f"  {ds_name} V2T CLIP Metrics:     {v2t_clip}")
        print(f"  {ds_name} V2T Reranked Metrics: {v2t_reranked}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main(parse_args())
