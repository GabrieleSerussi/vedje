"""
VEDJE: cached joint reranker for video-text retrieval.

    Indexing (once per video): frozen backbone g_phi -> patch features X_t ->
        frame-local compressor -> cache Z(v) in R^{TM x d_l}   (Eqs. 1 and 2)
    Query time: MiniLM joint encoder reads [tokens(q); Z(v)]; the stage-1
        score rho(q, v) is added to the pooled CLS as a residual prior
        e_rho(rho), and a small head produces s_theta(q, v)       (Eq. 5)
    Training: L = L_vtm + L_vtc + L_mlm + L_delta               (Eq. 6)
"""

import copy
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, BatchEncoding

from vedje.cache import FramewiseCrossAttentionCompressor
from vedje.delta import FixedQueryFutureFeaturePredictor, compute_delta_loss
from vedje.utils import get_world_size, all_gather_object_safe, get_rank, concat_all_gather

IMAGE_TOKEN = "<image>"
# The joint encoder reads at most 64 text tokens, [CLS] and [SEP] included (Section 4.1, Appendix A.2).
MAX_TEXT_CONTEXT_LENGTH = 64
# Standard masked-language modelling (Appendix A.3): BERT's masking rate.
MLM_PROBABILITY = 0.15

ENCODER_CONFIGS = {
    # VideoPrism-B, 16 frames x 16x16 patches
    "videoprism": {"vision_dim": 768, "clip_dim": 768, "patches_per_frame": 256},
    # VideoCLIP-XL (ViT-L/14), precomputed features only
    "vcxl": {"vision_dim": 1024, "clip_dim": 768, "patches_per_frame": 256},
}


class VideoPretrainModel(nn.Module):
    """VEDJE reranker.

    Two input modes:
        - Precomputed (default): pre-extracted frozen patch tokens + global feature
        - Online: frozen VideoPrism runs in the forward pass (feature extraction)
    """

    def __init__(
        self,
        language_model_path: str,
        vision_encoder: str = "videoprism",
        vision_encoder_path: Optional[str] = None,
        use_precomputed_features: bool = True,
        num_queries_per_frame: int = 4,
        num_attention_heads: int = 8,
        num_adapter_layers: int = 2,
        num_frames: int = 16,
        num_negatives_per_sample: int = 1,
        vp_attn_implementation: str = "eager",
        vtm_temperature: float = 1.0,
        vtc_logit_scale: float = 20.0,
        clip_injection: str = "post_only",
        delta_enabled: bool = True,
        delta_horizons: Sequence[int] = (3,),
        delta_num_layers: int = 2,
        delta_num_heads: int = 8,
    ):
        super().__init__()
        if clip_injection not in ("post_only", "off"):
            raise ValueError(f"clip_injection must be 'post_only' or 'off', got {clip_injection!r}")
        self.num_negatives_per_sample = num_negatives_per_sample
        self.vtm_temperature = vtm_temperature
        self.vtc_logit_scale = vtc_logit_scale
        self.clip_injection = clip_injection
        self.use_precomputed_features = use_precomputed_features
        self.num_frames = num_frames
        self.vision_encoder_name = vision_encoder
        self.vp_attn_implementation = vp_attn_implementation

        enc_cfg = ENCODER_CONFIGS[vision_encoder]
        self.vision_dim = enc_cfg["vision_dim"]
        self.clip_dim = enc_cfg["clip_dim"]
        patches_per_frame = enc_cfg["patches_per_frame"]
        self.patches_per_frame = patches_per_frame

        # Joint encoder (MiniLM-L12-H384)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_path)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [IMAGE_TOKEN]}
        )
        self.tokenizer.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        _orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.language_model = AutoModelForMaskedLM.from_pretrained(
            language_model_path, trust_remote_code=True, attn_implementation="eager",
        )
        torch.set_default_dtype(_orig_dtype)
        self.language_model.to(dtype=_orig_dtype)
        self._reinit_non_persistent_buffers()
        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.text_embeddings = self.language_model.get_input_embeddings()
        bert_dim = self.language_model.config.hidden_size

        # Base encoder without the MLM head (used for VTM scoring)
        self._base_encoder = None
        for attr in ('bert', 'roberta', 'new', 'model'):
            if hasattr(self.language_model, attr):
                self._base_encoder = getattr(self.language_model, attr)
                break

        # Frame-local compressor: T frames x M queries -> cache Z(v)
        self.vision_projection = FramewiseCrossAttentionCompressor(
            num_queries_per_frame=num_queries_per_frame,
            vision_dim=self.vision_dim,
            bert_dim=bert_dim,
            patches_per_frame=patches_per_frame,
            max_frames=num_frames,
            num_heads=num_attention_heads,
            num_layers=num_adapter_layers,
        )
        self.max_vision_context_length = self.vision_projection.num_queries
        # [CLS] text [SEP] cache [SEP]: the text budget, the cache tokens and the closing [SEP]
        self.max_context_length = MAX_TEXT_CONTEXT_LENGTH + self.max_vision_context_length + 1
        max_pos = self.language_model.config.max_position_embeddings
        if self.max_context_length > max_pos:
            raise ValueError(
                f"text ({MAX_TEXT_CONTEXT_LENGTH}) + cache ({self.max_vision_context_length}) "
                f"tokens exceed the joint encoder's {max_pos} positions"
            )

        # Score head: CLS -> s_theta
        self.vtm_head = nn.Sequential(
            nn.Linear(bert_dim, bert_dim),
            nn.GELU(),
            nn.Linear(bert_dim, 1),
        )
        # Residual prior e_rho: scalar stage-1 score -> bert_dim, added to CLS (Eq. 5)
        self.clip_score_proj = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(), nn.Linear(64, bert_dim)
        )
        # VTC text projection: CLS -> first-stage embedding space
        self.text_projection = nn.Linear(bert_dim, self.clip_dim)

        self.loss_fct = nn.CrossEntropyLoss()

        # Online mode: frozen VideoPrism
        self.videoprism = None
        if not use_precomputed_features and vision_encoder_path:
            if vision_encoder != "videoprism":
                raise ValueError("Online mode is only supported for VideoPrism")
            self._load_videoprism(vision_encoder_path)

        # Future-delta predictor F_omega (training only)
        self.delta_enabled = delta_enabled
        if delta_enabled:
            self.future_predictor = FixedQueryFutureFeaturePredictor(
                horizons=delta_horizons,
                patches_per_frame=patches_per_frame,
                bert_dim=bert_dim, vision_dim=self.vision_dim,
                num_heads=delta_num_heads,
                num_layers=delta_num_layers,
            )

        print(
            f"VideoPretrainModel: encoder={vision_encoder}, "
            f"vision_dim={self.vision_dim}, bert_dim={bert_dim}, "
            f"cache_tokens={self.max_vision_context_length} "
            f"({num_frames} frames x {num_queries_per_frame}), "
            f"max_seq={self.max_context_length}, precomputed={use_precomputed_features}, "
            f"clip_injection={clip_injection}, delta_horizons="
            f"{tuple(delta_horizons) if delta_enabled else None}"
        )

    def _load_videoprism(self, path: str) -> None:
        """Load and freeze VideoPrism encoder for online mode (stock transformers >= 5.13)."""
        from transformers import VideoPrismVisionModel
        self.videoprism = VideoPrismVisionModel.from_pretrained(
            path, attn_implementation=self.vp_attn_implementation,
        ).eval()
        for p in self.videoprism.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        captions: List[str],
        vid_feat: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        frames: Optional[torch.Tensor] = None,
        video_ids: Optional[List[int]] = None,
        neg_vision_embeds: Optional[torch.Tensor] = None,
        neg_vid_feats: Optional[torch.Tensor] = None,
        neg_captions: Optional[List[List[str]]] = None,
        precomputed_clip_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            captions: list of B caption strings
            vid_feat: (B, clip_dim) global video features (VTC target)
            vision_embeds: (B, T*P, vision_dim) precomputed patch tokens
            frames: (B, T, C, H, W) raw frames (online mode only)
            video_ids: optional video IDs (in-batch negative de-duplication)
            neg_vision_embeds: (B, K, T*P, vision_dim) stage-1 hard-negative videos
            neg_vid_feats: (B, K, clip_dim) their global features
            neg_captions: B lists of K hard-negative captions
            precomputed_clip_scores: (B, 1+2K) stage-1 scores [pos, neg videos, neg texts]
        Returns:
            (mlm_loss, vtc_loss, vtm_loss, delta_loss)
        """
        device = vid_feat.device

        if vision_embeds is None and frames is not None:
            vision_embeds, vid_feat_online = self._encode_vision_online(frames)
            vision_embeds = vision_embeds.clone()
            vid_feat = vid_feat_online.clone()

        vision_tokens = self.vision_projection(vision_embeds)  # Z(v): (B, T*M, bert_dim)

        mlm_loss = self.compute_mlm_loss(captions, vision_tokens)
        vtc_loss = self.compute_vtc_loss(captions, vid_feat, device=device)
        vtm_loss = self.compute_vtm_loss(
            captions, vision_tokens, vid_feat, video_ids,
            neg_vision_embeds=neg_vision_embeds,
            neg_vid_feats=neg_vid_feats,
            neg_captions=neg_captions,
            precomputed_clip_scores=precomputed_clip_scores,
        )

        if self.delta_enabled and self.training:
            B = vision_embeds.shape[0]
            per_frame_compressed = vision_tokens.view(
                B, self.num_frames, self.vision_projection.num_queries_per_frame, -1
            )
            delta_loss, _ = compute_delta_loss(
                predictor=self.future_predictor,
                per_frame_compressed=per_frame_compressed,
                raw_patches=vision_embeds,
                num_frames=self.num_frames,
                patches_per_frame=vision_embeds.shape[1] // self.num_frames,
            )
        else:
            delta_loss = torch.zeros((), device=device)

        return mlm_loss, vtc_loss, vtm_loss, delta_loss

    @torch.inference_mode()
    def _encode_vision_online(
        self, frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """VideoPrism: (B,T,C,288,288) -> patches (B,T*256,768), global (B,768).

        The global feature is the L2-normalized mean of the patch tokens.
        """
        assert self.videoprism is not None, "VideoPrism not loaded for online mode"
        outputs = self.videoprism(pixel_values_videos=frames)
        patch_tokens = outputs.last_hidden_state.to(self.text_projection.weight.dtype)
        v_global = F.normalize(patch_tokens.mean(dim=1), dim=-1)
        return patch_tokens, v_global

    # ------------------------------------------------------------------
    # VTM (Video-Text Matching)
    # ------------------------------------------------------------------

    def compute_vtm_loss(
        self,
        captions: List[str],
        vision_tokens: torch.Tensor,
        v_global: torch.Tensor,
        video_ids: Optional[List[int]] = None,
        neg_vision_embeds: Optional[torch.Tensor] = None,
        neg_vid_feats: Optional[torch.Tensor] = None,
        neg_captions: Optional[List[List[str]]] = None,
        precomputed_clip_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if neg_vision_embeds is not None:
            return self._compute_vtm_hard_negatives(
                captions, vision_tokens, v_global,
                neg_vision_embeds, neg_vid_feats, neg_captions,
                precomputed_clip_scores=precomputed_clip_scores,
            )

        # Fallback when no stage-1 hard negatives are provided: in-batch negatives
        pos_logits = self.compute_logits(
            captions, vision_projected_tokens=vision_tokens
        ).view(-1, 1)
        bs = pos_logits.shape[0]
        device = pos_logits.device

        all_captions = [None] * get_world_size()
        all_gather_object_safe(all_captions, captions)
        all_captions = sum((list(c) for c in all_captions), [])

        if video_ids is not None:
            all_video_ids = [None] * get_world_size()
            all_gather_object_safe(all_video_ids, video_ids)
            all_video_ids = sum((list(c) for c in all_video_ids), [])
        else:
            all_video_ids = None

        all_vision_tokens = concat_all_gather(vision_tokens)

        neg_video_idxs, neg_text_idxs = self.sample_hard_negatives(
            v_global, all_captions, all_video_ids, self.num_negatives_per_sample
        )

        neg_vision_tokens = all_vision_tokens[neg_video_idxs.clone().view(-1)]
        repeated_captions = list(np.repeat(captions, self.num_negatives_per_sample))
        neg_logits_videos = self._chunked_logits(
            repeated_captions, neg_vision_tokens
        ).view(bs, -1)

        neg_captions_list = [all_captions[idx] for idx in neg_text_idxs.clone().view(-1)]
        repeated_vision = vision_tokens.repeat_interleave(
            self.num_negatives_per_sample, dim=0
        )
        neg_logits_texts = self._chunked_logits(
            neg_captions_list, repeated_vision
        ).view(bs, -1)

        vtm_logits_v = torch.cat((pos_logits, neg_logits_videos), dim=1)
        vtm_logits_v = vtm_logits_v / self.vtm_temperature
        vtm_loss_v = F.cross_entropy(
            vtm_logits_v.float(),
            torch.zeros(bs, dtype=torch.long, device=device),
        )
        vtm_logits_t = torch.cat((pos_logits, neg_logits_texts), dim=1)
        vtm_logits_t = vtm_logits_t / self.vtm_temperature
        vtm_loss_t = F.cross_entropy(
            vtm_logits_t.float(),
            torch.zeros(bs, dtype=torch.long, device=device),
        )

        vtm_loss = (vtm_loss_v + vtm_loss_t) / 2
        self._vtm_loss_v = vtm_loss_v.item()
        self._vtm_loss_t = vtm_loss_t.item()
        return vtm_loss

    @torch.no_grad()
    def _get_text_embeds_for_clip(self, captions, device):
        """Get L2-normalized text embeddings for CLIP score computation (no grad)."""
        text_inputs = self.tokenizer(
            captions, padding=True, truncation=True,
            max_length=MAX_TEXT_CONTEXT_LENGTH, return_tensors="pt",
        ).to(device)
        if self._base_encoder is not None:
            text_out = self._base_encoder(**text_inputs)
            text_cls = text_out.last_hidden_state[:, 0]
        else:
            text_out = self.language_model(**text_inputs, output_hidden_states=True)
            text_cls = text_out.hidden_states[-1][:, 0]
        text_embed = self.text_projection(text_cls)
        return F.normalize(text_embed.float(), dim=-1)

    def _compute_vtm_hard_negatives(
        self,
        captions: List[str],
        vision_tokens: torch.Tensor,
        v_global: torch.Tensor,
        neg_vision_embeds: torch.Tensor,
        neg_vid_feats: Optional[torch.Tensor],
        neg_captions: List[List[str]],
        precomputed_clip_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """VTM loss over stage-1 hard negatives with the residual stage-1 prior.

        precomputed_clip_scores: (B, 1+K+K) tensor from dataset containing
            [pos_clip, neg_clip_v_1..K, neg_clip_t_1..K], the stage-1 (LvT) scores.
        """
        B, K = neg_vision_embeds.shape[:2]
        device = vision_tokens.device

        # Extract CLIP scores: either precomputed (LvT) or proxy (VTC)
        if precomputed_clip_scores is not None:
            # Z-score normalize over the candidate set, matching evaluation.
            scores_raw = precomputed_clip_scores  # (B, 1+2K)
            s_std = scores_raw.std(dim=-1, keepdim=True).clamp(min=1e-8)
            scores_norm = (scores_raw - scores_raw.mean(dim=-1, keepdim=True)) / s_std
            pos_clip_scores = scores_norm[:, 0]                          # (B,)
            neg_clip_scores_v = scores_norm[:, 1:1+K].reshape(B * K)     # (B*K,)
            neg_clip_scores_t = scores_norm[:, 1+K:].reshape(B * K)      # (B*K,)
        else:
            # Fallback: compute proxy CLIP scores from VTC embeddings
            text_embed = self._get_text_embeds_for_clip(captions, device)
            v_norm = F.normalize(v_global.float(), dim=-1)
            pos_clip_scores = (text_embed * v_norm).sum(dim=-1)
            neg_clip_scores_v = None
            neg_clip_scores_t = None
            if neg_vid_feats is not None:
                neg_vf_norm = F.normalize(neg_vid_feats.float(), dim=-1)
                neg_clip_scores_v = torch.einsum('bd,bkd->bk', text_embed, neg_vf_norm).reshape(B * K)
                neg_caps_flat_tmp = []
                for caps_list in neg_captions:
                    neg_caps_flat_tmp.extend(caps_list)
                neg_text_embed = self._get_text_embeds_for_clip(neg_caps_flat_tmp, device)
                v_repeated = v_norm.repeat_interleave(K, dim=0)
                neg_clip_scores_t = (neg_text_embed * v_repeated).sum(dim=-1)

        pos_logits = self.compute_logits(
            captions, vision_projected_tokens=vision_tokens,
            clip_scores=pos_clip_scores,
        ).view(-1, 1)

        # Project negative vision embeddings
        neg_flat = neg_vision_embeds.view(B * K, *neg_vision_embeds.shape[2:])
        neg_vision_tokens = self.vision_projection(neg_flat)

        repeated_caps = []
        for cap in captions:
            repeated_caps.extend([cap] * K)
        neg_logits_v = self._chunked_logits(
            repeated_caps, neg_vision_tokens,
            clip_scores=neg_clip_scores_v,
        ).view(B, K)

        # Negative texts: same video, wrong text
        neg_caps_flat = []
        for caps_list in neg_captions:
            neg_caps_flat.extend(caps_list)

        repeated_vision = vision_tokens.repeat_interleave(K, dim=0)
        neg_logits_t = self._chunked_logits(
            neg_caps_flat, repeated_vision,
            clip_scores=neg_clip_scores_t,
        ).view(B, K)

        vtm_logits_v = torch.cat((pos_logits, neg_logits_v), dim=1)
        vtm_logits_v = vtm_logits_v / self.vtm_temperature
        vtm_loss_v = F.cross_entropy(
            vtm_logits_v.float(),
            torch.zeros(B, dtype=torch.long, device=device),
        )
        vtm_logits_t = torch.cat((pos_logits, neg_logits_t), dim=1)
        vtm_logits_t = vtm_logits_t / self.vtm_temperature
        vtm_loss_t = F.cross_entropy(
            vtm_logits_t.float(),
            torch.zeros(B, dtype=torch.long, device=device),
        )

        vtm_loss = (vtm_loss_v + vtm_loss_t) / 2
        self._vtm_loss_v = vtm_loss_v.item()
        self._vtm_loss_t = vtm_loss_t.item()
        return vtm_loss

    @torch.inference_mode()
    def sample_hard_negatives(
        self,
        v_global: torch.Tensor,
        all_captions: List[str],
        all_video_ids: Optional[List[int]],
        num_negatives: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """In-batch hard negative mining using vision encoder global features."""
        if all_video_ids is None:
            all_video_ids = list(range(len(all_captions)))

        all_v_global = concat_all_gather(v_global)

        bs = v_global.shape[0]
        rank = get_rank()
        device = v_global.device

        sim = v_global @ all_v_global.T

        eq_vid = self._get_equal_objects_mask(all_video_ids)
        eq_vid = eq_vid[bs * rank: bs * (rank + 1)].to(device)
        neg_v_weights = sim.masked_fill(eq_vid, torch.finfo(sim.dtype).min)
        neg_v_weights = F.softmax(neg_v_weights, dim=-1)
        neg_video_idxs = torch.topk(neg_v_weights, k=num_negatives, dim=-1).indices

        eq_cap = self._get_equal_objects_mask(all_captions)
        eq_cap = eq_cap[bs * rank: bs * (rank + 1)].to(device)
        neg_t_weights = sim.masked_fill(eq_cap, torch.finfo(sim.dtype).min)
        neg_t_weights = F.softmax(neg_t_weights, dim=-1)
        neg_text_idxs = torch.topk(neg_t_weights, k=num_negatives, dim=-1).indices

        return neg_video_idxs, neg_text_idxs

    # ------------------------------------------------------------------
    # MLM (Masked Language Modeling)
    # ------------------------------------------------------------------

    def compute_mlm_loss(
        self, captions: List[str], vision_tokens: torch.Tensor
    ) -> torch.Tensor:
        text_inputs, labels = self._prepare_mlm_inputs_labels(captions, vision_tokens)
        outputs = self.language_model(**text_inputs)
        logits = outputs.logits
        if not (labels != -100).any():
            # a small batch of short captions can have no masked token at the 15% rate; the loss is then zero, not 0/0
            return logits.sum() * 0.0
        return self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))


    # ------------------------------------------------------------------
    # VTC (Video-Text Contrastive)
    # ------------------------------------------------------------------

    def compute_vtc_loss(
        self,
        captions: List[str],
        vid_feat: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Contrastive alignment of BERT CLS text features with video global features.

        Uses InfoNCE (symmetric cross-entropy) over in-batch text-video pairs.
        """
        if vid_feat is None:
            return torch.zeros(1, device=device, requires_grad=True)

        text_inputs = self.tokenizer(
            captions,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_CONTEXT_LENGTH,
            return_tensors="pt",
        ).to(device)

        outputs = self.language_model(**text_inputs, output_hidden_states=True)
        cls_hidden = outputs.hidden_states[-1][:, 0, :]
        text_embeds = F.normalize(self.text_projection(cls_hidden), dim=-1)
        vid_embeds = F.normalize(vid_feat.detach(), dim=-1)

        # InfoNCE: symmetric cross-entropy over similarity matrix
        logits = text_embeds @ vid_embeds.T * self.vtc_logit_scale
        labels = torch.arange(len(logits), device=device)
        loss_t2v = F.cross_entropy(logits, labels)
        loss_v2t = F.cross_entropy(logits.T, labels)
        return (loss_t2v + loss_v2t) / 2

    # ------------------------------------------------------------------
    # Logits
    # ------------------------------------------------------------------

    def compute_logits(
        self,
        captions: List[str],
        vision_projected_tokens: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        clip_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """s_theta(q, v) = head(c(q, v) + e_rho(rho))   (Eq. 5).

        Args:
            clip_scores: (B,) stage-1 scores rho (z-normalized over the candidate
                         set); None or clip_injection="off" skips the prior.
        """
        if vision_projected_tokens is None and vision_embeds is not None:
            vision_projected_tokens = self.vision_projection(vision_embeds)

        inputs = self._prepare_lm_inputs(captions, vision_projected_tokens)
        hidden_states = self._get_hidden_states(inputs)
        cls_hidden = hidden_states[:, 0]

        if clip_scores is not None and self.clip_injection == "post_only":
            cs_input = clip_scores.unsqueeze(-1).to(cls_hidden.dtype)
            cls_hidden = cls_hidden + self.clip_score_proj(cs_input)
        return self.vtm_head(cls_hidden).view(-1)

    def _get_hidden_states(self, inputs: dict) -> torch.Tensor:
        """Last hidden states of the base encoder (skips the MLM head).

        The encoder runs under fp32 autocast; bf16 matmuls inside MiniLM were
        numerically unstable for VTM on some GPU/cuDNN versions.
        """
        with torch.autocast(device_type="cuda", dtype=torch.float32,
                            enabled=torch.cuda.is_available()):
            if self._base_encoder is not None:
                outputs = self._base_encoder(**inputs, output_hidden_states=True)
                hidden = outputs.last_hidden_state
            else:
                outputs = self.language_model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
        return hidden.to(torch.get_default_dtype())

    def _chunked_logits(
        self,
        captions: List[str],
        vision_tokens: torch.Tensor,
        chunk_size: Optional[int] = None,
        clip_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n = len(captions)
        if chunk_size is None:
            # Scale chunk size inversely with vision token count to avoid OOM
            chunk_size = max(1, 24 * 256 // max(vision_tokens.shape[1], 1))
        if n <= chunk_size:
            return self.compute_logits(
                captions, vision_projected_tokens=vision_tokens,
                clip_scores=clip_scores,
            )
        logits_parts = []
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            cs = clip_scores[i:end] if clip_scores is not None else None
            part = self.compute_logits(
                captions[i:end],
                vision_projected_tokens=vision_tokens[i:end],
                clip_scores=cs,
            )
            logits_parts.append(part)
        return torch.cat(logits_parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_mlm_inputs_labels(
        self, captions: List[str], vision_embeds: torch.Tensor
    ) -> Tuple[dict, torch.LongTensor]:
        target_inputs = self._tokenize_text_vision_pair(captions, vision_embeds)
        target_inputs = dict(target_inputs)
        labels = target_inputs["input_ids"]

        text_inputs = copy.deepcopy(target_inputs)
        input_ids = text_inputs["input_ids"]

        special_tokens_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(
                    seq.tolist(), already_has_special_tokens=True
                )
                for seq in input_ids
            ],
            dtype=torch.bool,
            device=input_ids.device,
        )
        random_mask = (torch.rand(input_ids.shape) < MLM_PROBABILITY).to(input_ids.device)
        final_mask = (
            (input_ids != self.tokenizer.image_token_id)
            & ~special_tokens_mask
            & random_mask
        )

        # As in BERT, 80% of the selected tokens become [MASK], 10% a random word and 10% stay unchanged.
        masked_input_ids = input_ids.clone()
        to_mask = final_mask & (torch.rand(input_ids.shape) < 0.8).to(input_ids.device)
        to_random = final_mask & ~to_mask & (torch.rand(input_ids.shape) < 0.5).to(input_ids.device)
        masked_input_ids[to_mask] = self.tokenizer.mask_token_id
        random_words = torch.randint(self.tokenizer.vocab_size, input_ids.shape).to(input_ids.device)
        masked_input_ids[to_random] = random_words[to_random]
        text_inputs["input_ids"] = masked_input_ids
        labels[~final_mask] = -100

        inputs_embeds = self._compute_inputs_embeds(text_inputs["input_ids"], vision_embeds)
        text_inputs["inputs_embeds"] = inputs_embeds
        text_inputs.pop("input_ids")

        return text_inputs, labels

    def _tokenize_text_vision_pair(
        self, captions: List[str], vision_embeds: torch.Tensor
    ) -> BatchEncoding:
        bs, vision_seq_len, _ = vision_embeds.shape
        image_placeholder = "".join([IMAGE_TOKEN] * vision_seq_len)
        text_inputs = self.tokenizer(
            captions,
            [image_placeholder] * bs,
            padding=True,
            truncation="only_first",
            max_length=self.max_context_length,
            return_tensors="pt",
        ).to(vision_embeds.device)
        return text_inputs

    def _compute_inputs_embeds(
        self, input_ids: torch.LongTensor, vision_embeds: torch.Tensor
    ) -> torch.Tensor:
        inputs_embeds = self.text_embeddings(input_ids)
        mask = input_ids == self.tokenizer.image_token_id
        mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
        vision_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(mask_expanded, vision_embeds)
        return inputs_embeds

    def _prepare_lm_inputs(
        self, captions: List[str], vision_embeds: torch.Tensor
    ) -> dict:
        tokenized_pair = self._tokenize_text_vision_pair(captions, vision_embeds)
        tokenized_pair = dict(tokenized_pair)
        inputs_embeds = self._compute_inputs_embeds(
            input_ids=tokenized_pair.pop("input_ids"),
            vision_embeds=vision_embeds,
        )
        tokenized_pair["inputs_embeds"] = inputs_embeds
        return tokenized_pair

    def _reinit_non_persistent_buffers(self) -> None:
        """Reinitialize non-persistent buffers (e.g. position_ids, inv_freq) that may
        contain garbage after lazy weight loading in transformers v5."""
        for module in self.language_model.modules():
            if hasattr(module, 'position_ids'):
                max_pos = getattr(self.language_model.config, 'max_position_embeddings', 8192)
                module.position_ids = torch.arange(max_pos).expand((1, -1))
            # Fix RotaryEmbedding inv_freq (also non-persistent)
            if hasattr(module, 'inv_freq') and hasattr(module, 'dim'):
                base = getattr(module, 'base', 10000.0)
                inv_freq = 1.0 / (base ** (torch.arange(0, module.dim, 2).float() / module.dim))
                module.inv_freq = inv_freq
                # Recompute cos/sin cache with corrected inv_freq
                if hasattr(module, '_set_cos_sin_cache'):
                    max_pos = getattr(module, 'max_position_embeddings', 8192)
                    module._set_cos_sin_cache(
                        seq_len=max_pos, device=inv_freq.device, dtype=torch.float32
                    )

    @staticmethod
    def _get_equal_objects_mask(objects: list) -> torch.BoolTensor:
        object_to_id = {obj: idx for idx, obj in enumerate(set(objects))}
        objects_ids = torch.tensor([object_to_id[obj] for obj in objects])
        return objects_ids[:, None] == objects_ids[None, :]


def build_model(config: dict, training: bool = True) -> VideoPretrainModel:
    """Construct VideoPretrainModel from a config dict (train / eval share this)."""
    return VideoPretrainModel(
        language_model_path=config['language_model_path'],
        vision_encoder=config.get('vision_encoder', 'videoprism'),
        vision_encoder_path=config.get('vision_encoder_path'),
        use_precomputed_features=config.get('use_precomputed_features', True),
        num_queries_per_frame=config.get('num_queries_per_frame', 4),
        num_attention_heads=config.get('num_attention_heads', 8),
        num_adapter_layers=config.get('num_adapter_layers', 2),
        num_frames=config.get('num_frames', 16),
        num_negatives_per_sample=config.get('num_negatives_per_sample', 1),
        vp_attn_implementation=config.get('vp_attn_implementation', 'eager'),
        vtm_temperature=config.get('vtm_temperature', 1.0),
        vtc_logit_scale=config.get('vtc_logit_scale', 20.0),
        clip_injection=config.get('clip_injection', 'post_only'),
        delta_enabled=training and config.get('loss_weight_delta', 1.0) > 0,
        delta_horizons=config.get('delta_horizons', [3]),
        delta_num_layers=config.get('delta_num_layers', 2),
        delta_num_heads=config.get('delta_num_heads', 8),
    )


def parameter_counts(model: VideoPretrainModel) -> dict:
    """Parameters of each part of a VideoPretrainModel.

    joint_encoder is the base encoder of the language model (MiniLM-L12-H384 by
    default) with its embeddings; score_head is h_psi (vtm_head) and prior is
    e_rho (clip_score_proj) from Eq. 5. compressor, vtc_projection, mlm_head and
    future_predictor are the remaining trained parts; future_predictor is 0 when
    the model was built without L_delta (build_model with training=False).
    """
    def count(module) -> int:
        return sum(p.numel() for p in module.parameters()) if module is not None else 0

    base = model._base_encoder if model._base_encoder is not None else model.language_model
    return {
        "joint_encoder": count(base),
        "compressor": count(model.vision_projection),
        "score_head": count(model.vtm_head),
        "prior": count(model.clip_score_proj),
        "vtc_projection": count(model.text_projection),
        "mlm_head": count(model.language_model) - count(base),
        "future_predictor": count(getattr(model, "future_predictor", None)),
    }
