"""
Step 4. Train the compressor, the joint reranker and the auxiliary heads:
L = L_vtm + L_vtc + L_mlm + L_delta (Eq. 6), with a per-epoch two-stage R@K
evaluation.

Usage:
    python scripts/train.py --config configs/vedje_vp_msrvtt.yaml --output_dir ./output/vp_msrvtt
    torchrun --nproc_per_node=1 scripts/train.py --config configs/vedje_vp_msrvtt.yaml \
        --output_dir ./output/vp_msrvtt
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from vedje import utils
from vedje.config import load_config
from vedje.data import create_train_dataset, create_eval_dataset, create_sampler, create_loader
from vedje.exp_logger import ExperimentLogger
from vedje.model import build_model
from vedje.retrieval import (
    evaluation_t2v, evaluation_v2t,
    compute_video_features, compute_lvt_video_features, compute_lvt_text_features,
)
from vedje.utils import warmup_lr_schedule, step_lr_schedule


def train(model, data_loader, optimizer, epoch, device, config,
          logger=None, global_step=0, output_dir=None, model_without_ddp=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    for name in ('loss_mlm', 'loss_vtm', 'loss_vtc', 'loss_delta'):
        metric_logger.add_meter(name, utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    use_precomputed = config.get('use_precomputed_features', True)
    grad_accum_steps = config.get('gradient_accumulation_steps', 1)
    w_mlm = config.get('loss_weight_mlm', 1.0)
    w_vtc = config.get('loss_weight_vtc', 1.0)
    w_vtm = config.get('loss_weight_vtm', 1.0)
    w_delta = config.get('loss_weight_delta', 1.0)

    data_loader.sampler.set_epoch(epoch)

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        neg_vision_embeds = None
        neg_captions_list = None
        neg_vid_feats = None
        precomputed_clip_scores = None
        if use_precomputed:
            if len(batch) >= 8:
                if len(batch) == 9:
                    # patches, global, caption, video id, index, hard negatives, stage-1 scores
                    patch_tokens, vid_feat, captions, video_ids, _, neg_patches, neg_vf, neg_caps_raw, precomputed_clip_scores = batch
                    precomputed_clip_scores = precomputed_clip_scores.to(device, non_blocking=True)
                else:
                    patch_tokens, vid_feat, captions, video_ids, _, neg_patches, neg_vf, neg_caps_raw = batch
                neg_vision_embeds = neg_patches.to(device, non_blocking=True)
                neg_vid_feats = neg_vf.to(device, non_blocking=True)
                # K tuples of B strings -> B lists of K strings
                neg_captions_list = [list(x) for x in zip(*neg_caps_raw)]
            else:
                patch_tokens, vid_feat, captions, video_ids, _ = batch
            patch_tokens = patch_tokens.to(device, non_blocking=True)
            vid_feat = vid_feat.to(device, non_blocking=True)
            frames = None
        else:
            frames, captions, video_ids, _ = batch
            frames = frames.to(device, non_blocking=True)
            patch_tokens = None
            model_ref = model.module if hasattr(model, 'module') else model
            vid_feat = torch.zeros(len(captions), model_ref.clip_dim, device=device)

        if isinstance(video_ids, torch.Tensor):
            video_ids = video_ids.tolist()
        if isinstance(captions, (list, tuple)) and isinstance(captions[0], (list, tuple)):
            captions = [c[0] if isinstance(c, (list, tuple)) else c for c in captions]

        if epoch == 0 and i < config['warmup_steps']:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'],
                               config['warmup_lr'], config['init_lr'])

        if i % grad_accum_steps == 0:
            optimizer.zero_grad()

        loss_mlm, loss_vtc, loss_vtm, loss_delta = model(
            captions=captions,
            vid_feat=vid_feat,
            vision_embeds=patch_tokens,
            frames=frames,
            video_ids=video_ids,
            neg_vision_embeds=neg_vision_embeds,
            neg_vid_feats=neg_vid_feats,
            neg_captions=neg_captions_list,
            precomputed_clip_scores=precomputed_clip_scores,
        )
        loss = w_mlm * loss_mlm + w_vtc * loss_vtc + w_vtm * loss_vtm + w_delta * loss_delta
        (loss / grad_accum_steps).backward()

        grad_norm_val = 0.0
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(data_loader):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config.get('max_grad_norm', 1.0),
            )
            grad_norm_val = float(grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        global_step += 1

        vals = {
            'loss_mlm': loss_mlm.item(), 'loss_vtm': loss_vtm.item(),
            'loss_vtc': loss_vtc.item(), 'loss_delta': float(loss_delta),
            'lr': optimizer.param_groups[0]["lr"], 'grad_norm': grad_norm_val,
        }
        metric_logger.update(**vals)
        if logger and utils.is_main_process() and global_step % 50 == 0:
            for k, v in vals.items():
                logger.log_metric(f'train_{k}', v, step=global_step)

        temp_checkpoint_steps = config.get('temp_checkpoint_steps', 1000)
        if (global_step % temp_checkpoint_steps == 0
                and utils.is_main_process()
                and output_dir is not None
                and model_without_ddp is not None):
            temp_path = os.path.join(output_dir, 'temp_checkpoint.pth')
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'global_step': global_step,
            }
            torch.save(save_obj, temp_path)
            print(f"\n==> Saved temp checkpoint at step {global_step} to {temp_path}\n")

        max_steps = config.get('max_steps', 0)
        if max_steps and global_step >= max_steps:
            print(f"\n==> Reached max_steps={max_steps}, stopping epoch early.\n")
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())

    if logger and utils.is_main_process():
        for metric_name, meter in metric_logger.meters.items():
            logger.log_metric(f'train_{metric_name}', meter.global_avg, step=epoch)

    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, global_step


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    exp_logger = None
    if utils.is_main_process():
        exp_logger = ExperimentLogger(config)

    # Dataset
    print("Creating dataset")
    train_dataset = create_train_dataset(config)
    print(f"Number of training samples: {len(train_dataset)}")
    # Evaluation datasets
    test_set_name = config.get('test_set', 'msrvtt')
    test_datasets = {}
    if isinstance(test_set_name, str):
        test_set_name = [test_set_name]
    for ds_name in test_set_name:
        try:
            test_datasets[ds_name] = create_eval_dataset(config, ds_name)
            print(f"Created {ds_name} eval dataset: {len(test_datasets[ds_name])} videos")
        except Exception as e:
            print(f"Warning: could not create {ds_name} eval dataset: {e}")

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    samplers = create_sampler([train_dataset], [True], num_tasks, global_rank)
    data_loader = create_loader(
        [train_dataset], samplers,
        batch_size=[config['batch_size']],
        num_workers=[config.get('num_workers', 4)],
        is_trains=[True],
        collate_fns=[None]
    )[0]

    # Model
    print("Creating model")
    model = build_model(config, training=True).to(device)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config['init_lr'],
        weight_decay=config['weight_decay'],
    )

    start_epoch = 0
    global_step = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = {k: v for k, v in checkpoint['model'].items()
                      if not k.startswith('videoprism.')}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {args.checkpoint}: missing={len(missing)}, unexpected={len(unexpected)}")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f'Could not load optimizer state: {e}')
        start_epoch = checkpoint.get('epoch', -1) + 1
        global_step = checkpoint.get('global_step', 0)
        print(f'Resuming from epoch {start_epoch}, step {global_step}')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # Load LvT model for CLIP retrieval in evaluation (if configured)
    lvt_model = None
    lvt_tokenizer = None
    clip_model_path = config.get('clip_model_path', '')
    if clip_model_path and not args.skip_eval and utils.is_main_process():
        print(f"Loading LvT CLIP model from {clip_model_path}...")
        try:
            from vedje.lvt import load_lvt_model_fixed
            lvt_model, lvt_tokenizer = load_lvt_model_fixed(
                clip_model_path, device=device, dtype=torch.float32)
            print("LvT CLIP model loaded.")
        except (ValueError, ImportError, KeyError) as e:
            print(f"Warning: Could not load LvT CLIP model: {e}")
            print("LvT-based evaluation (Stage 1 retrieval) will be skipped.")
            lvt_model = None
            lvt_tokenizer = None

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, config['max_epoch']):
        step_lr_schedule(optimizer, epoch, config['init_lr'],
                         config['min_lr'], config['lr_decay_rate'])
        train_stats, global_step = train(
            model, data_loader, optimizer, epoch, device, config,
            exp_logger, global_step, args.output_dir, model_without_ddp,
        )

        model_without_ddp.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if utils.is_main_process():
            all_metrics = {}

            for ds_name, test_ds in test_datasets.items():
                if not args.skip_eval:
                    # Compute vision tokens for reranker (stage 2)
                    _, vision_tokens = compute_video_features(
                        model_without_ddp, test_ds, device, config
                    )

                    # Compute CLIP features for stage 1.
                    # Precomputed test features are preferred when provided (they work for
                    # any backbone, including those the LvT video tower cannot read,
                    # e.g. VideoCLIP-XL).
                    lvt_test_path = config.get('lvt_test_features_path', '')
                    if lvt_test_path and os.path.isfile(lvt_test_path):
                        precomputed = torch.load(lvt_test_path, map_location='cpu')
                        vid_feats = precomputed['vid_feats']
                        text_feats = precomputed['text_feats']
                        if epoch == start_epoch:
                            print(f"Using precomputed LvT test features from {lvt_test_path}")
                    elif lvt_model is not None:
                        vid_feats = compute_lvt_video_features(
                            lvt_model, test_ds, device, config
                        )
                        text_feats = compute_lvt_text_features(
                            lvt_model, lvt_tokenizer,
                            getattr(test_ds, 'raw_text', test_ds.text), device,
                            prompt_template=config.get('lvt_prompt_template', ''),
                        )
                    else:
                        vid_feats = None
                        text_feats = None

                    print(f"\n=== Evaluating {ds_name} Text-to-Video ===")
                    t2v_clip, t2v_reranked = evaluation_t2v(
                        model_without_ddp, test_ds, device, config,
                        vid_feats, text_feats, vision_tokens,
                    )
                    print(f"\n=== Evaluating {ds_name} Video-to-Text ===")
                    v2t_clip, v2t_reranked = evaluation_v2t(
                        model_without_ddp, test_ds, device, config,
                        vid_feats, text_feats, vision_tokens,
                    )

                    print(f"\n{'='*60}")
                    print(f"  {ds_name} T2V CLIP Metrics:     {t2v_clip}")
                    print(f"  {ds_name} T2V Reranked Metrics: {t2v_reranked}")
                    print(f"  {ds_name} V2T CLIP Metrics:     {v2t_clip}")
                    print(f"  {ds_name} V2T Reranked Metrics: {v2t_reranked}")
                    print(f"{'='*60}")

                    all_metrics.update({f'{ds_name}_t2v_clip_{k}': v for k, v in t2v_clip.items()})
                    all_metrics.update({f'{ds_name}_t2v_reranked_{k}': v for k, v in t2v_reranked.items()})
                    all_metrics.update({f'{ds_name}_v2t_clip_{k}': v for k, v in v2t_clip.items()})
                    all_metrics.update({f'{ds_name}_v2t_reranked_{k}': v for k, v in v2t_reranked.items()})

            model_without_ddp.train()

            if exp_logger:
                for metric, score in all_metrics.items():
                    exp_logger.log_metric(f'eval_{metric}', score, step=epoch)

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'eval_{k}': v for k, v in all_metrics.items()},
                'epoch': epoch,
                'global_step': global_step,
            }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'global_step': global_step,
            }
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_{epoch:02d}.pth')
            torch.save(save_obj, checkpoint_path)

            if exp_logger:
                exp_logger.log_checkpoint(checkpoint_path, epoch=epoch)

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.distributed:
            dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')

    if utils.is_main_process() and exp_logger:
        exp_logger.log_metric('total_training_time_seconds', total_time)
        exp_logger.end()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train VEDJE: the frame-indexed compressor, the joint reranker, the prior "
                    "embedding, the score head and the training-only heads.")
    parser.add_argument('--config', default='./configs/vedje_vp_msrvtt.yaml')
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--checkpoint', default='', help='resume from checkpoint')
    parser.add_argument('--device', default=None,
                        help='default: cuda when available, otherwise cpu')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--max_steps', default=0, type=int,
                        help='Stop training after N steps (0 = full epoch). For debug runs.')
    args = parser.parse_args(argv)
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.set_default_dtype(torch.bfloat16)
    from dotenv import load_dotenv
    load_dotenv()

    # ActivityNet: load_config derives the data paths from `activitynet_retrieval_mode` unless set.
    config = load_config(args.config)
    if args.max_steps:
        config['max_steps'] = args.max_steps

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)
