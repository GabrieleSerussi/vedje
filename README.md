<div align="center">

<h1>VEDJE: Video-Efficient Discriminative Joint Encoder for Scalable Video-Text Retrieval</h1>

<p><b>VEDJE moves video encoding offline and keeps joint matching online.</b></p>

<p>
<a href="https://scholar.google.com/citations?user=yp4EiKQAAAAJ&amp;hl=en">Shahaf Wagner</a><sup>1*</sup>&nbsp;&nbsp;
<a href="https://scholar.google.com/citations?user=GJ19YUEAAAAJ&amp;hl=en">Gabriele Serussi</a><sup>1,3*</sup>&nbsp;&nbsp;
<a href="https://scholar.google.com/citations?user=OmIy5cgAAAAJ">Dan Ben Ami</a><sup>1</sup>&nbsp;&nbsp;
<a href="https://scholar.google.com/citations?user=ut_ISVIAAAAJ">Tomer Galanti</a><sup>2</sup>&nbsp;&nbsp;
<a href="https://chaimbaskin.bgu.ac.il/">Chaim Baskin</a><sup>1,3</sup>
</p>

<p>
<sup>1</sup>INSIGHT Lab, Ben-Gurion University of the Negev&nbsp;&nbsp;&nbsp;
<sup>2</sup>Texas A&amp;M University&nbsp;&nbsp;&nbsp;
<sup>3</sup>Decart AI<br>
<sup>*</sup>Equal contribution
</p>

<p>
<a href="https://gabrieleserussi.github.io/vedje/"><img src="https://img.shields.io/badge/Project-Page-1f6feb" alt="Project page"></a>
<a href="https://colab.research.google.com/github/GabrieleSerussi/vedje/blob/main/colab/vedje_tour.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

<img src="docs/assets/figures/fig1-frontier.png" width="760" alt="Scatter plot of MSR-VTT text-to-video R@1 in percent against online model parameters on a logarithmic axis from 63M to 7.6B. VEDJE reaches 59.8 at about 157M parameters. CLIP4Clip at 46.4, X-CLIP at 49.3, EERCF at 49.9 and Video-ColBERT at 51.5 lie between 63M and 110M parameters. LamRA, reproduced, at 59.7 and CaRe-DPO at 64.1 lie at 7.6B.">

</div>

VEDJE reranks first-stage video candidates with a 33M-parameter joint encoder that reads a compact cache written once per video. The cache keeps a few learned tokens per sampled frame, and feature-change supervision during training improves retrieval from it without adding query-time work. This repository holds the VEDJE package and its `vedje` command, one script per method step, five of the paper's configurations and the tests.

## Quick start

1. **Install** (Python 3.10 or newer):

   ```bash
   git clone https://github.com/GabrieleSerussi/vedje.git
   cd vedje
   pip install -e .
   ```

2. **Train** on a GPU. Put your videos and annotation files under `./data_root/msrvtt/`, in [the format below](#your-data), then run:

   ```bash
   vedje train configs/vedje_vp_msrvtt.yaml
   ```

   It indexes the videos, prepares the first stage, trains VEDJE and reports R@1, R@5 and R@10 on the test set. Running it again skips the steps already done.

3. **Search your own videos** with the trained model:

   ```bash
   vedje index output/vedje_vp_msrvtt/checkpoint_03.pth my_videos/ --out my_index.pt
   vedje search my_index.pt "a person walks across a field"
   ```

   Indexing runs the frozen video encoder once per video. Each search reads only the caches of the first stage's candidates, with no visual encoding.

## From Python

```python
from vedje.inference import VEDJE

vedje = VEDJE.from_checkpoint("output/vedje_vp_msrvtt/checkpoint_03.pth")
index = vedje.index("my_videos/")            # once per collection
index.save("my_index.pt")
for hit in vedje.search("a person walks across a field", index, top=5):
    print(hit.rank, hit.video, round(hit.score, 3))
```

`Index.load("my_index.pt")` reopens a saved index, and each hit also gives the video's first-stage rank and score. Indexing new videos needs a checkpoint trained with a [VideoPrism](https://proceedings.mlr.press/v235/zhao24f.html) config or with your own backbone ([below](#using-vedje-with-another-backbone)).

## Your data

Each config reads its data from `./data_root/<dataset>/`; edit the paths in the config to use others. Annotation files are JSON lists, and video paths are relative to the config's video folder.

| Dataset | Config | Training entry | Test entry |
| --- | --- | --- | --- |
| [MSR-VTT](https://openaccess.thecvf.com/content_cvpr_2016/html/Xu_MSR-VTT_A_Large_CVPR_2016_paper.html) | `vedje_vp_msrvtt.yaml` | `{"video_id": "video0", "video": "video0.mp4", "caption": ["...", "..."]}` | `{"video": "video7020.mp4", "caption": "..."}` |
| [MSVD](https://aclanthology.org/P11-1020/) | `vedje_vp_msvd.yaml` | `{"video_id": "<name>", "video": "<name>.avi", "caption": ["...", "..."]}` | `{"video": "<name>.avi", "caption": ["...", "..."]}` |
| [DiDeMo](https://arxiv.org/abs/1708.01641) | `vedje_vp_didemo.yaml` | `{"video": "train/<name>.mp4", "caption": "<descriptions, concatenated>"}` | the same form, one entry per caption |
| [ActivityNet](https://arxiv.org/abs/1705.00754) | `vedje_vp_activitynet.yaml` | `{"video_id": "v_XXXX", "video": "v1-3/train_val/v_XXXX.mp4", "caption": ["<paragraph>"]}` | `{"video_id": "v_XXXX", "video": "v1-3/train_val/v_XXXX.mp4", "caption": "<paragraph>"}` |

The first stage reads the captions as written, and the joint encoder reads them lower-cased without punctuation. [`configs/vedje_vclip_msrvtt.yaml`](configs/vedje_vclip_msrvtt.yaml) reads precomputed [VideoCLIP-XL](https://aclanthology.org/2024.emnlp-main.898/) features and first-stage embeddings instead of videos, and its comments list their formats. Each step of `vedje train` is also a script in [`scripts/`](scripts) with `--help`, and `scripts/train.py` runs data-parallel under `torchrun`.

## How it works

<p align="center">
<img src="docs/assets/figures/fig2-pipeline.png" width="760" alt="VEDJE pipeline in three panels. (A) Offline indexing: a frozen visual encoder reads the frames f1 to fT, a frame-indexed compressor turns the features of each frame into a few tokens Z1 to ZT, and the ordered cache Z(v) is stored per video. (B) Training only: a future-delta head predicts feature changes from Zt, and L_delta compares them with the target X(t+h) minus X(t). (C) Online reranking: a joint encoder reads the query q with the cache Z(v), the embedded stage-1 score is added to its output c(q, v), and a score head gives the score s(q, v).">
</p>

Offline, a frozen backbone encodes each video once, and a shared compressor writes a few tokens per sampled frame in temporal order. At query time, a 33M-parameter joint encoder reads the query together with the cache of each candidate. The first-stage score is embedded and added to the encoder's output before the score head. During training, a future-delta head predicts how the frozen features change between sampled frames; it is discarded afterwards.

| Step | What it does | Script |
| --- | --- | --- |
| 1. Index videos | Encodes each video once with the frozen [VideoPrism-B](https://proceedings.mlr.press/v235/zhao24f.html) backbone and stores its frame-indexed patch features. | [`scripts/extract_features.py`](scripts/extract_features.py) |
| 2 and 3. Stage-1 embeddings and hard negatives | Embeds the training and test videos and captions with the first-stage retriever (VideoPrism-LvT) and mines the closest wrong videos for each training caption. | [`scripts/stage1_train_embeddings.py`](scripts/stage1_train_embeddings.py), [`scripts/mine_hard_negatives.py`](scripts/mine_hard_negatives.py), [`scripts/stage1_test_features.py`](scripts/stage1_test_features.py) |
| 4. Train | Trains the frame-indexed compressor, the joint reranker, the prior embedding, the score head and the training-only heads. | [`scripts/train.py`](scripts/train.py) |
| 5. Rerank and evaluate | Retrieves candidates with the first stage, reranks them from their caches and reports R@1, R@5 and R@10 in both directions. | [`scripts/evaluate.py`](scripts/evaluate.py) |

`vedje train` runs these steps in order.

## Results

- VEDJE reaches **59.8 [MSR-VTT](https://openaccess.thecvf.com/content_cvpr_2016/html/Xu_MSR-VTT_A_Large_CVPR_2016_paper.html) text-to-video R@1 with about 157M online parameters**, using a fine-tuned [VideoCLIP-XL](https://aclanthology.org/2024.emnlp-main.898/) first stage. The [LamRA](https://arxiv.org/abs/2412.01720) reproduction reaches 59.7 with a 7.6B base decoder, **about 49 times that count** (Figure 1, Table 22 and Appendix B).
- VEDJE improves R@1 over each matched first stage on MSR-VTT by **3.6 to 7.2 points, in both directions** (Table 1b). The first stages are [VideoPrism](https://proceedings.mlr.press/v235/zhao24f.html), [PE-Core-B](https://proceedings.neurips.cc/paper_files/paper/2025/hash/57bc0a850255e2041341bf74c7e2b9fa-Abstract-Conference.html) and VideoCLIP-XL, zero-shot and fine-tuned.
- With the zero-shot VideoCLIP-XL first stage, VEDJE improves text-to-video R@1 by **4.9 to 8.9 points on MSR-VTT, [MSVD](https://aclanthology.org/P11-1020/), [DiDeMo](https://arxiv.org/abs/1708.01641) and [ActivityNet](https://arxiv.org/abs/1705.00754)** (Table 12).
- The default cache stores **48 KiB per video, 128.5 times less** than the frame-and-patch features of the same backbone (Section 4.3 and Table 9).
- In the VideoPrism configuration on MSR-VTT, the **12 KiB cache keeps text-to-video R@1 within 0.2 points** of the 48 KiB cache (Table 3).

## Repository layout

```
vedje/        the package: cache, delta, model, features, video, lvt, retrieval, data, config,
              inference (index and search), backbone (another backbone) and cli (the vedje command)
scripts/      one script per method step, which vedje train runs in order
configs/      five configurations of the paper
tests/        the test suite (pytest, CPU only)
docs/         the project page
colab/        the Colab tour
```

## Using VEDJE with another backbone

To use another backbone, write one small class with three methods. They give the patch features of a video's sampled frames and the first-stage embeddings of videos and captions. `vedje prepare` runs the class once over a config's dataset ([Your data](#your-data)) and writes everything `vedje train` needs.

```python
# my_backbone.py
from vedje.backbone import Backbone

class MyBackbone(Backbone):
    name = "my_backbone"
    frame_size = 224                  # the T frames of a video (num_frames in the config) are 224 x 224
    patches_per_frame = 256           # P
    patch_dim = 768                   # D

    def patches(self, frames):        # (T, 3, H, W) in [0, 1] -> (T, P, D) frozen patch features
        ...

    def embed_video(self, frames):    # the same frames -> (C,) first-stage embedding of the video
        ...

    def embed_texts(self, captions):  # N captions -> (N, C) first-stage embeddings of the captions
        ...
```

```bash
vedje prepare my_backbone:MyBackbone configs/vedje_vp_msrvtt.yaml --out output/my_backbone
vedje train output/my_backbone/my_backbone.yaml
```

`vedje train` then mines the hard negatives, trains VEDJE and evaluates it. `vedje index` and `vedje search` work as in the [Quick start](#quick-start): the checkpoint names the backbone class, so they load it by themselves (`--backbone module:Class` overrides it). From Python, `vedje.backbone.prepare` does the same as `vedje prepare`, and `VEDJE.from_checkpoint` takes `backbone=`.

<details><summary>File formats</summary>

To use features computed elsewhere, write these files and a config that reads them. [`configs/vedje_vclip_msrvtt.yaml`](configs/vedje_vclip_msrvtt.yaml) does this for [VideoCLIP-XL](https://aclanthology.org/2024.emnlp-main.898/) features.

- **Features.** `precomputed_features_dir` holds one `.pt` file per video, named after the video file (`video0.mp4` gives `video0.pt`, and [ActivityNet](https://arxiv.org/abs/1705.00754) uses `<video_id>.pt`), with two tensors. `local_patches` holds the `(T*P, D)` patch features in frame order, with frame t in rows t*P to t*P+P-1. `v_global` is a `(C,)` global feature of the video, which training uses only without first-stage embeddings.
- **Training set.** `lvt_embeds_path` holds the first-stage embeddings `{"video_embeds": (N_videos, C), "text_embeds": (N_captions, C), "video_ids", "video_id_to_idx", "caption_to_video_idx", "vid_to_caption_indices"}`, L2-normalised, with the captions in the order of the training annotations, as [`scripts/stage1_train_embeddings.py`](scripts/stage1_train_embeddings.py) writes them. They give the score prior, the hard negatives and the contrastive targets.
- **Test set.** `lvt_test_features_path` holds the first-stage embeddings `{"vid_feats": (N_videos, C), "text_feats": (N_captions, C)}`, L2-normalised, in the order of the test annotations, as [`scripts/stage1_test_features.py`](scripts/stage1_test_features.py) writes them.
- **Config.** `vision_encoder` names the backbone, and `vision_dim` (D), `clip_dim` (C) and `patches_per_frame` (P) give its dimensions. `num_frames` (T) and `num_queries_per_frame` (M) set the cache to T x M tokens, and `hard_negatives_path` is where `vedje train` writes the hard negatives.

</details>

## Citation

```bibtex
@misc{wagner2026vedje,
  title={VEDJE: Video-Efficient Discriminative Joint Encoder for Scalable Video-Text Retrieval},
  author={Shahaf Wagner and Gabriele Serussi and Dan Ben Ami and Tomer Galanti and Chaim Baskin},
  year={2026},
  note={Preprint},
}
```

## License

The code is released under the MIT License ([LICENSE](LICENSE)). [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists the third-party code and the model checkpoints that the code downloads, with their licences.
