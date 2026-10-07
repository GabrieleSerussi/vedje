"""Search your own videos with a trained VEDJE checkpoint: index them once, then query them with text.

    from vedje.inference import VEDJE

    vedje = VEDJE.from_checkpoint("output/vedje_vp_msrvtt/checkpoint_03.pth")
    index = vedje.index("my_videos/")            # once per collection
    index.save("my_index.pt")
    for hit in vedje.search("a person walks across a field", index, top=5):
        print(hit.rank, hit.video, round(hit.score, 3))

Indexing runs the frozen VideoPrism-B encoder once per video, writes its cache with the trained compressor and
keeps its first-stage (VideoPrism-LvT) embedding. Searching runs no visual encoder: the first stage keeps the
most similar videos, and VEDJE reranks them from their caches, with the first-stage score as its prior (Eq. 5).
A checkpoint trained with another backbone (vedje.backbone) indexes and searches with that backbone instead,
loaded from the class its config names or given as `backbone=`. The `vedje index` and `vedje search` commands
wrap this module.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Union

import torch
import torch.nn.functional as F

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")


class Hit(NamedTuple):
    rank: int                   # rank after reranking, from 1
    video: str                  # the video's path, as indexed
    score: float                # VEDJE's score s_theta(q, v)
    first_stage_rank: int       # the video's rank in the first stage, from 1
    first_stage_score: float    # the first-stage cosine similarity


@dataclass
class Index:
    """The searchable form of a collection: one cache and one first-stage embedding per video."""
    videos: List[str]
    caches: torch.Tensor        # (N, T*M, d) cached tokens, bf16
    stage1: torch.Tensor        # (N, D) L2-normalised first-stage video embeddings
    checkpoint: str = ""        # the checkpoint whose compressor wrote the caches
    config: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.videos)

    def save(self, path: Union[str, Path]) -> None:
        torch.save({"videos": self.videos, "caches": self.caches, "stage1": self.stage1,
                    "checkpoint": self.checkpoint, "config": self.config}, str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Index":
        return cls(**torch.load(str(path), map_location="cpu"))


def list_videos(videos: Union[str, Path, Sequence[Union[str, Path]]]) -> List[str]:
    """Video files from a folder (sorted by name), a single file, or a list of files."""
    if isinstance(videos, (str, Path)):
        p = Path(videos)
        if p.is_dir():
            files = sorted(str(f) for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS)
            if not files:
                raise FileNotFoundError(f"no video files ({', '.join(VIDEO_EXTENSIONS)}) in {p}")
            return files
        return [str(p)]
    return [str(v) for v in videos]


@torch.inference_mode()
def rank(model, query: str, query_stage1: torch.Tensor, index: Index, top: int = 10, candidates: int = 20,
         device: Union[str, torch.device] = "cpu") -> List[Hit]:
    """Two-stage search over an index with the query's first-stage embedding.

    The first stage keeps the `candidates` most similar videos; VEDJE scores each of them from its cache with the
    first-stage scores standardised over the candidates as its prior, as in training and evaluation, and the
    candidates are sorted by that score (Algorithm 1).
    """
    from vedje.data.utils import pre_caption

    first = index.stage1.float() @ query_stage1.float().view(-1)
    k = min(candidates, len(index))
    s1, cand = torch.topk(first, k)
    prior = s1.clone()
    if k > 1 and prior.std() > 1e-8:
        prior = (prior - prior.mean()) / prior.std()
    scores = model._chunked_logits(
        captions=[pre_caption(query)] * k,
        vision_tokens=index.caches[cand].to(device=device, dtype=torch.float32),
        clip_scores=prior.to(device),
    ).float().cpu()
    order = torch.argsort(scores, descending=True)[:top].tolist()
    return [Hit(rank=r + 1, video=index.videos[int(cand[i])], score=float(scores[i]),
                first_stage_rank=i + 1, first_stage_score=float(s1[i])) for r, i in enumerate(order)]


class VEDJE:
    """A trained VEDJE reranker with the frozen encoders it needs to index and search."""

    def __init__(self, model, config: dict, checkpoint: str = "", device: Optional[str] = None, backbone=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device, dtype=torch.float32).eval()
        self.config = config
        self.checkpoint = checkpoint
        self.backbone = backbone   # a vedje.backbone.Backbone for a checkpoint trained with another backbone
        self._videoprism = None
        self._lvt = None

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], device: Optional[str] = None, backbone=None) -> "VEDJE":
        """Load a checkpoint written by `vedje train` (scripts/train.py); it carries its own config.

        backbone: for a checkpoint trained with another backbone, a Backbone or its `module:Class`; by default the
        class that its config names (written by vedje.backbone.prepare).
        """
        from vedje.model import build_model
        ckpt = torch.load(str(path), map_location="cpu")
        config = dict(ckpt["config"])
        model = build_model(config, training=False)
        missing, _ = model.load_state_dict(ckpt["model"], strict=False)
        trainable = {n for n, _ in model.named_parameters()}
        lost = [m for m in missing if m in trainable]
        if lost:
            raise RuntimeError(f"{path} lacks {len(lost)} trained weights, for example {lost[:3]}")
        backbone = backbone or config.get("backbone")
        if isinstance(backbone, str):
            from vedje.backbone import load
            backbone = load(backbone)
        return cls(model, config, checkpoint=str(path), device=device, backbone=backbone)

    def _check_encoders(self):
        if self.backbone is None and self.config.get("vision_encoder", "videoprism") != "videoprism":
            raise ValueError(f"this checkpoint was trained with the {self.config['vision_encoder']} backbone: give it "
                             "as backbone=MyBackbone() or --backbone module:Class (see vedje.backbone)")

    def _first_stage(self):
        if self._lvt is None:
            from vedje.lvt import DEFAULT_LVT_MODEL, load_lvt_model_fixed
            self._lvt = load_lvt_model_fixed(self.config.get("clip_model_path") or DEFAULT_LVT_MODEL,
                                             device=self.device, dtype=torch.float32)
        return self._lvt

    def _encoder(self):
        if self._videoprism is None:
            from vedje.features import DEFAULT_BACKBONE, load_backbone
            self._videoprism = load_backbone(self.config.get("vision_encoder_path") or DEFAULT_BACKBONE,
                                           attn_implementation=self.config.get("vp_attn_implementation", "eager"),
                                           device=self.device)
        return self._videoprism

    @torch.inference_mode()
    def index(self, videos, batch_size: int = 4, progress: bool = True) -> Index:
        """Index video files (a folder, a file or a list): one cache and one first-stage embedding per video."""
        from tqdm import tqdm
        from vedje.features import extract_patch_features
        from vedje.lvt import video_embeddings
        from vedje.video import load_video_frames

        self._check_encoders()
        paths = list_videos(videos)
        num_frames = self.config.get("num_frames", 16)
        if self.backbone is not None:
            return self._index_with_backbone(paths, num_frames, progress)
        encoder, (lvt, _) = self._encoder(), self._first_stage()
        caches, stage1 = [], []
        for start in tqdm(range(0, len(paths), batch_size), desc="Indexing", disable=not progress):
            frames = torch.stack([load_video_frames(p, num_frames, size=288, normalize=False)
                                  for p in paths[start:start + batch_size]])        # (B, T, 3, 288, 288) in [0, 1]
            patches, _ = extract_patch_features(encoder, frames)                    # (B, T*256, 768), computed once
            caches.append(self.model.vision_projection(patches.float()).to(torch.bfloat16).cpu())
            pooled = video_embeddings(lvt.video_model(pixel_values_videos=frames.to(self.device, torch.float32)))
            pooled = pooled.mean(dim=1) if pooled.dim() == 3 else pooled
            stage1.append(F.normalize(pooled.float(), dim=-1).cpu())
        return Index(videos=paths, caches=torch.cat(caches), stage1=torch.cat(stage1),
                     checkpoint=self.checkpoint, config=self.config)

    def _index_with_backbone(self, paths: List[str], num_frames: int, progress: bool) -> Index:
        """Index with another backbone, reading each video as vedje.backbone.prepare did for training."""
        from tqdm import tqdm
        from vedje.video import load_video_frames

        backbone, caches, stage1 = self.backbone, [], []
        for p in tqdm(paths, desc="Indexing", disable=not progress):
            frames = load_video_frames(p, num_frames, size=backbone.frame_size, normalize=False)    # (T, 3, H, W)
            patches = backbone.patches(frames).reshape(1, -1, backbone.patch_dim)                  # (1, T*P, D)
            # the patch features in bf16, as training reads them, then the cache written by the trained compressor
            tokens = self.model.vision_projection(patches.to(torch.bfloat16).float().to(self.device))
            caches.append(tokens.to(torch.bfloat16).cpu())
            stage1.append(F.normalize(backbone.embed_video(frames).float().flatten(), dim=0).cpu())
        return Index(videos=paths, caches=torch.cat(caches), stage1=torch.stack(stage1),
                     checkpoint=self.checkpoint, config=self.config)

    @torch.inference_mode()
    def search(self, query: str, index: Index, top: int = 10, candidates: int = 20) -> List[Hit]:
        """The `top` best videos for a text query: first-stage candidates, reranked by VEDJE from their caches."""
        self._check_encoders()
        if self.backbone is not None:
            q = F.normalize(self.backbone.embed_texts([query]).float(), dim=-1)[0].cpu()
        else:
            from vedje.lvt import text_embeddings
            lvt, tokenizer = self._first_stage()
            inputs = tokenizer([query], padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
            q = text_embeddings(lvt.text_model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask))
            q = F.normalize(q.float(), dim=-1)[0].cpu()
        return rank(self.model, query, q, index, top=top, candidates=candidates, device=self.device)
