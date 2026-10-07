# Reproducing the paper

## Installation

```bash
git clone https://github.com/GabrieleSerussi/vedje.git
cd vedje
pip install -e .            # ".[dev]" adds pytest, ".[comet]" adds Comet logging
```

- Python 3.10 or newer. Stock `transformers` (5.13 or newer) provides the VideoPrism models.
- The code downloads three checkpoints from the Hugging Face Hub on first use: VideoPrism-B, VideoPrism-LvT-B and MiniLM-L12-H384 ([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists them with their licences).
- Every script runs on a CPU for small inputs. Training at full scale runs on a CUDA GPU. The paper used one 48 GB NVIDIA L40S, with about 16 to 24 hours for a full training run (Appendix A.5).
- Videos are decoded with decord when it is installed (`pip install decord`, Linux) and with PyAV otherwise. Both read the same frame indices, and a folder of frame images works as well.
- The stage-1 test embeddings read the test videos with mediapy, which needs FFmpeg on the PATH.

## The CPU check

```bash
python reproduce/cpu_check.py
```

The check recomputes these numbers from the package's own modules and prints the paper value, the computed value and OK or FAIL for each line. It exits with status 1 on any FAIL and takes about ten seconds once MiniLM is cached.

| Check | Paper | Source |
| --- | --- | --- |
| Cache payload, 16 frames x 4 tokens at d = 384 in BF16, measured on compressor outputs | 49,152 bytes (48 KiB) | Section 3.2, Appendix A.2 |
| Cache payload, 16 frames x 1 token | 12,288 bytes (12 KiB) | Section 3.2, Appendix A.2 |
| Frame-and-patch payload, 16 x 257 x 768 x 2 bytes | 6.02 MiB | Table 9 |
| Storage ratio of the frame-and-patch features to the 64-token cache | 128.5 | Section 4.3, Table 9 |
| Cache traffic for 20 candidates, 64 and 16 tokens | 0.983 MB and 0.246 MB | Section 4.7 |
| Valid (t, h) pairs of L_delta for T = 16 and H = {3} | 13, derived from the definition of Omega | Eq. 4 |
| Parameters of the joint encoder, counted on the model the package builds | about 33M | Appendix A.2 |
| Parameters of the score head | about 0.15M | Appendix A.2 |
| Parameters of the prior embedding and the score head | fewer than 0.2M | Appendix A.2 |

The last three lines download MiniLM-L12-H384 (about 130 MB) on the first run. With `VEDJE_OFFLINE=1`, or without network access, they are skipped with a message.

## From the paper to the commands

Each row runs the pipeline below with the named config. A change means a copy of that config with the given keys set. The evaluation reads `k`, so one checkpoint can be evaluated with several values.

| Paper | Setting | Config | Change |
| --- | --- | --- | --- |
| Table 1b; Tables 12 to 15 | VideoPrism first stage, MSR-VTT | `configs/vedje_vp_msrvtt.yaml` | none |
| Tables 12 to 15 | VideoPrism first stage, MSVD | `configs/vedje_vp_msvd.yaml` | none |
| Tables 12 to 15 | VideoPrism first stage, DiDeMo | `configs/vedje_vp_didemo.yaml` | none |
| Tables 12 to 15 | VideoPrism first stage, ActivityNet paragraph retrieval | `configs/vedje_vp_activitynet.yaml` | none |
| Table 1b; Tables 12 to 15 | zero-shot VideoCLIP-XL first stage, MSR-VTT | `configs/vedje_vclip_msrvtt.yaml` | VideoCLIP-XL features and stage-1 embeddings that you provide (see the inputs below) |
| Table 3, 64 tokens; Table 4, separate frame groups; Table 5, joint encoder | default cache and scorer | `configs/vedje_vp_msrvtt.yaml` | none, and `loss_weight_delta: 0` for the rows without L_delta |
| Table 3, 16 tokens; Table 17 | 12 KiB cache | `configs/vedje_vp_msrvtt.yaml` | `num_queries_per_frame: 1`, and `loss_weight_delta: 0` for the rows without L_delta |
| Table 6, recall columns; Figure 3, VideoPrism curves and the MSR-VTT VideoCLIP-XL curve | 20, 50, 100 or 200 candidates | the config of the trained model | `k: 20`, `k: 50`, `k: 100` or `k: 200` |
| Table 11, joint stream only | no stage-1 prior | `configs/vedje_vp_msrvtt.yaml` | `clip_injection: "off"` |
| Table 19 | loss stack | `configs/vedje_vp_msrvtt.yaml` | L_vtm only: `loss_weight_mlm: 0`, `loss_weight_vtc: 0`, `loss_weight_delta: 0`; adding L_mlm: `loss_weight_vtc: 0`, `loss_weight_delta: 0`; adding L_vtc: `loss_weight_delta: 0` |
| Table 20 | prediction horizon | `configs/vedje_vp_msrvtt.yaml` | `delta_horizons: [2]` or `delta_horizons: [4]` |
| Table 21 | BERT-base joint encoder | `configs/vedje_vp_msrvtt.yaml` | `language_model_path` set to a BERT-base checkpoint, for example `"google-bert/bert-base-uncased"` |

Tables 3 and 4 report means over three training runs; `scripts/train.py --seed` sets the seed of a run (42 by default).

## The pipeline

With `CFG` set to a config, the steps run in this order (the numbers match the scripts' docstrings and the README):

- **Step 1, index the videos.** `python scripts/extract_features.py --config $CFG` writes one `.pt` file per video of the train and test annotations into the config's features folder. Each file holds `local_patches`, 4096 patch tokens of dimension 768 (16 frames of 256 patches), and `v_global` (768). Files that already exist are skipped. `torchrun --nproc_per_node=N scripts/extract_features.py --config $CFG` shares the videos among N GPUs.
- **Step 2a, stage-1 embeddings of the training set.** `python scripts/stage1_train_embeddings.py --config $CFG` writes the VideoPrism-LvT embeddings of the training videos and captions to the config's `lvt_embeds_path`. Training turns them into the stage-1 scores of the residual prior, and the video embeddings are the contrastive targets (Table 7).
- **Step 2b, stage-1 hard negatives.** `python scripts/mine_hard_negatives.py --config $CFG` reads the step 2a embeddings and writes, for each training caption, its 50 most similar wrong videos to `hard_negatives_path`.
- **Step 3, stage-1 embeddings of the test set.** `python scripts/stage1_test_features.py --config $CFG` writes `{"vid_feats", "text_feats"}` for the test videos and captions to `lvt_test_features_path`.
- **Step 4, train.** `python scripts/train.py --config $CFG --output_dir output/run` trains the compressor, the joint reranker and the auxiliary heads, and writes `checkpoint_XX.pth`, `temp_checkpoint.pth` and `log.txt`. It runs the two-stage evaluation after each pass over the training data unless `--skip_eval` is set. `--checkpoint` resumes a run, `--max_steps N` stops early, and `torchrun` runs data-parallel training.
- **Step 5, rerank and evaluate.** `python scripts/evaluate.py --config $CFG --checkpoint output/run/checkpoint_03.pth` prints R@1, R@5 and R@10 of the first stage and of VEDJE, text-to-video and video-to-text.

Every script has `--help`. Steps 2a and 2b also accept explicit paths, which override the config. Setting `VEDJE_DISABLE_PRELOAD=1` makes the dataset loaders read the feature files from disk at each step instead of loading them all into memory first. Comet logging is off by default. It needs `experiment.backend: "comet"` in the config and the `COMET_API_KEY` environment variable, which `scripts/train.py` also reads from a local `.env` file.

## Inputs you provide

The configs expect everything under `./data_root/<dataset>/`, relative to the folder the scripts run from; edit the config to use other paths. Each annotation file is a JSON list, and video paths are relative to the config's video folder.

| Dataset | Config keys | Training entry | Test entry | Feature file |
| --- | --- | --- | --- | --- |
| MSR-VTT | `msrvtt_videos`, `msrvtt_train_ann`, `msrvtt_test_ann` | `{"video_id": "video0", "video": "video0.mp4", "caption": ["...", "..."]}` | `{"video": "video7020.mp4", "caption": "..."}`, with a caption string or list | `video0.pt` |
| MSVD | `msvd_videos`, `msvd_train_ann`, `msvd_test_ann` | `{"video_id": "<name>", "video": "<name>.avi", "caption": ["...", "..."]}` | `{"video": "<name>.avi", "caption": ["...", "..."]}` | `<name>.pt` |
| DiDeMo | `didemo_videos`, `didemo_train_ann`, `didemo_test_ann` | `{"video": "train/<name>.mp4", "caption": "<descriptions, concatenated>"}` | same form, one entry per caption | `<name>.pt` |
| ActivityNet | `activitynet_videos_dir`; the annotation files follow `activitynet_retrieval_mode` | `{"video_id": "v_XXXX", "video": "v1-3/train_val/v_XXXX.mp4", "caption": ["<paragraph>"]}` | `{"video_id": "v_XXXX", "video": "v1-3/train_val/v_XXXX.mp4", "caption": "<paragraph>"}` | `v_XXXX.pt` |

The first stage reads the captions as written, since its tokenizer is case-sensitive, and the joint encoder reads them lower-cased without punctuation.

For ActivityNet, the mode `paragraph` (default) or `sentence` selects `activitynet_train_<mode>.json`, `activitynet_test_<mode>.json` and the matching stage-1 and hard-negative files under `./data_root/activitynet/`, unless the config sets them.

`configs/vedje_vclip_msrvtt.yaml` reads precomputed VideoCLIP-XL inputs, so it skips steps 1, 2a and 3:

- One feature file per video in `./data_root/msrvtt/vclip_features_16f/`, `{"local_patches": (16 * 256, 1024), "v_global": (768,)}`, from 16 uniformly sampled frames.
- Stage-1 embeddings of the training set in `./data_root/msrvtt/vclip_stage1_train_embeds.pt`, in the format of step 2a: `{"video_embeds": (N_videos, 768), "text_embeds": (N_captions, 768), "video_ids", "video_id_to_idx", "caption_to_video_idx", "vid_to_caption_indices"}`, L2-normalised, in the order of the training annotation file. Step 2b mines the hard negatives from them.
- Stage-1 test embeddings in `./data_root/msrvtt/vclip_stage1_test_features.pt`, `{"vid_feats": (N_videos, D), "text_feats": (N_texts, D)}`, L2-normalised, with the videos in the order of the test annotation file and the captions in file order.

## What this release does not contain

- Trained checkpoints, extracted features, stage-1 embeddings, hard-negative files, datasets and annotation files.
- Code that computes VideoCLIP-XL features or VideoCLIP-XL stage-1 embeddings; `configs/vedje_vclip_msrvtt.yaml` reads them precomputed.
- The fine-tuned VideoCLIP-XL first stage (Table 1b, Table 22 and Figure 1) and VideoCLIP-XL configurations for MSVD, DiDeMo and ActivityNet (Tables 12 to 15).
- The PE-Core-B configuration (Table 1b, Tables 13 to 15 and Table 22).
- The CLIP ViT-B/16 control (Table 2) and the evaluation on the released EERCF candidates (Table 10).
- The pooled caches of Table 4 and the trained MaxSim scorer of Table 5.
- The prior-only control of Table 11, the reconstruction targets of Table 18 and the single-frame EDJE-style reference of Appendix A.7.
- The latency and I/O measurements (Table 6, latency columns; Table 8; Appendix A.6) and the cache quantization of Table 16.
- Scripts that average several training runs.
