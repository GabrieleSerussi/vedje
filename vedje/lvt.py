"""The VideoPrism-LvT first-stage model (stage 1).

load_lvt_model_fixed(model_name, device) returns (model, tokenizer). With stock
transformers (>= 5.13) the checkpoint loads directly into VideoPrismClipModel.
Older builds of the VideoPrism transformers port use a nested BERT-style layout
(text_model.text_encoder.layer.{i}, video_model.backbone.spatial_encoder.layer.{i},
contrastive_vision_pooler); for those, the checkpoint's flat names
(text_model.layers.{i}, video_model.vision_model.spatial_layers.{i},
video_model.head, ...) are mapped onto the model's names.

The two layouts return the embeddings under different output fields;
text_embeddings and video_embeddings read them from either.
"""
import math

import torch
from transformers import AutoModel, AutoTokenizer
from safetensors import safe_open
from huggingface_hub import hf_hub_download

DEFAULT_LVT_MODEL = "MHRDYN7/videoprism-lvt-base-f16r288"


def text_embeddings(text_out) -> torch.Tensor:
    """(B, D) L2-normalised caption embeddings from a text_model output.

    Stock transformers returns them as pooler_output (last_hidden_state holds
    the token states); the nested layout returns them as last_hidden_state.
    """
    pooled = getattr(text_out, "pooler_output", None)
    return text_out.last_hidden_state if pooled is None else pooled


def video_embeddings(video_out) -> torch.Tensor:
    """(B, 1, D) L2-normalised pooled video embeddings from a video_model output.

    Stock transformers returns them as pooler_output (last_hidden_state holds
    the patch tokens); the nested layout returns them as video_last_hidden_state.
    """
    pooled = getattr(video_out, "video_last_hidden_state", None)
    return video_out.pooler_output if pooled is None else pooled


def _intra_layer_attn_map():
    return {
        "attention.q_proj.weight": "attention.attention.query.weight",
        "attention.q_proj.bias":   "attention.attention.query.bias",
        "attention.k_proj.weight": "attention.attention.key.weight",
        "attention.k_proj.bias":   "attention.attention.key.bias",
        "attention.v_proj.weight": "attention.attention.value.weight",
        "attention.v_proj.bias":   "attention.attention.value.bias",
        "attention.o_proj.weight": "attention.output.dense.weight",
        "attention.o_proj.bias":   "attention.output.dense.bias",
        "mlp.fc1.weight":          "intermediate.dense.weight",
        "mlp.fc1.bias":            "intermediate.dense.bias",
        "mlp.fc2.weight":          "output.dense.weight",
        "mlp.fc2.bias":            "output.dense.bias",
        "layernorm_before.weight": "layernorm_before.weight",
        "layernorm_before.bias":   "layernorm_before.bias",
        "layernorm_after.weight":  "layernorm_after.weight",
        "layernorm_after.bias":    "layernorm_after.bias",
    }


def _build_rename_map():
    """Return dict mapping every checkpoint key to the nested-layout model key."""
    rename = {}

    # -------- text_model --------
    rename["text_model.embeddings.token_embedding.weight"] = "text_model.token_embeddings.weight"
    rename["text_model.embeddings.position_embedding"]     = "text_model.position_embeddings"
    rename["text_model.embeddings.cls_emb"]                = "text_model.cls_emb"
    rename["text_model.layernorm.weight"] = "text_model.layernorm.weight"
    rename["text_model.layernorm.bias"]   = "text_model.layernorm.bias"
    layer_map = _intra_layer_attn_map()
    for i in range(12):
        for k, v in layer_map.items():
            rename[f"text_model.layers.{i}.{k}"] = f"text_model.text_encoder.layer.{i}.{v}"

    # -------- video_model.vision_model -> video_model.backbone --------
    # Top-level layernorms + embeddings
    rename["video_model.vision_model.layernorm1.weight"] = "video_model.backbone.layernorm1.weight"
    rename["video_model.vision_model.layernorm1.bias"]   = "video_model.backbone.layernorm1.bias"
    rename["video_model.vision_model.layernorm2.weight"] = "video_model.backbone.layernorm2.weight"
    rename["video_model.vision_model.layernorm2.bias"]   = "video_model.backbone.layernorm2.bias"
    rename["video_model.vision_model.spatial_embeddings.patch_embeddings.projection.weight"] = \
        "video_model.backbone.spatial_embeddings.patch_embeddings.projection.weight"
    rename["video_model.vision_model.spatial_embeddings.patch_embeddings.projection.bias"] = \
        "video_model.backbone.spatial_embeddings.patch_embeddings.projection.bias"
    rename["video_model.vision_model.spatial_embeddings.position_embeddings"] = \
        "video_model.backbone.spatial_embeddings.position_embeddings"
    rename["video_model.vision_model.temporal_embeddings.position_embeddings"] = \
        "video_model.backbone.temporal_embeddings.position_embeddings"

    # 12 spatial layers
    for i in range(12):
        for k, v in layer_map.items():
            rename[f"video_model.vision_model.spatial_layers.{i}.{k}"] = \
                f"video_model.backbone.spatial_encoder.layer.{i}.{v}"
    # 4 temporal layers
    for i in range(4):
        for k, v in layer_map.items():
            rename[f"video_model.vision_model.temporal_layers.{i}.{k}"] = \
                f"video_model.backbone.temporal_encoder.layer.{i}.{v}"

    # -------- video_model.auxiliary_layers -> video_model.auxiliary_encoder.layer --------
    for i in range(2):
        for k, v in layer_map.items():
            rename[f"video_model.auxiliary_layers.{i}.{k}"] = \
                f"video_model.auxiliary_encoder.layer.{i}.{v}"

    # -------- video_model.head -> video_model.contrastive_vision_pooler --------
    # checkpoint head has: q,k,v,o_proj + per_dim_scale + pooling_attention_query
    # model pooler has:    query, key, value, projection + per_dim_scale + pooling_attention_query + scale + layernorm
    rename["video_model.head.q_proj.weight"]              = "video_model.contrastive_vision_pooler.query.weight"
    rename["video_model.head.q_proj.bias"]                = "video_model.contrastive_vision_pooler.query.bias"
    rename["video_model.head.k_proj.weight"]              = "video_model.contrastive_vision_pooler.key.weight"
    rename["video_model.head.k_proj.bias"]                = "video_model.contrastive_vision_pooler.key.bias"
    rename["video_model.head.v_proj.weight"]              = "video_model.contrastive_vision_pooler.value.weight"
    rename["video_model.head.v_proj.bias"]                = "video_model.contrastive_vision_pooler.value.bias"
    rename["video_model.head.o_proj.weight"]              = "video_model.contrastive_vision_pooler.projection.weight"
    rename["video_model.head.o_proj.bias"]                = "video_model.contrastive_vision_pooler.projection.bias"
    rename["video_model.head.per_dim_scale"]              = "video_model.contrastive_vision_pooler.per_dim_scale"
    rename["video_model.head.pooling_attention_query"]    = "video_model.contrastive_vision_pooler.pooling_attention_query"
    rename["video_model.head_layernorm.weight"]           = "video_model.contrastive_vision_pooler.layernorm.weight"
    rename["video_model.head_layernorm.bias"]             = "video_model.contrastive_vision_pooler.layernorm.bias"
    # The pooler's 'scale' buffer is absent from the checkpoint; it is recomputed below.
    return rename


def load_lvt_state_dict(model_name=DEFAULT_LVT_MODEL):
    """Load the checkpoint with the nested-layout renaming applied."""
    sf = hf_hub_download(model_name, "model.safetensors")
    rename = _build_rename_map()
    state = {}
    with safe_open(sf, framework="pt") as f:
        for k in f.keys():
            tensor = f.get_tensor(k)
            if k in rename:
                state[rename[k]] = tensor
            elif k.endswith("position_ids"):
                continue  # runtime buffer
            else:
                state[k] = tensor
    return state


def _report(missing, unexpected):
    t_miss = [k for k in missing if k.startswith("text_model")]
    v_miss = [k for k in missing if k.startswith("video_model")]
    t_unxp = [k for k in unexpected if k.startswith("text_model")]
    v_unxp = [k for k in unexpected if k.startswith("video_model")]
    print(f"[lvt] text_model: missing={len(t_miss)} unexpected={len(t_unxp)}")
    for k in t_miss[:5]: print(f"    text miss: {k}")
    for k in t_unxp[:5]: print(f"    text unxp: {k}")
    print(f"[lvt] video_model: missing={len(v_miss)} unexpected={len(v_unxp)}")
    for k in v_miss[:5]: print(f"    vid miss: {k}")
    for k in v_unxp[:5]: print(f"    vid unxp: {k}")


def load_lvt_model_fixed(model_name=DEFAULT_LVT_MODEL,
                         device=None, dtype=torch.float32):
    """Load and initialize the LvT model. Returns (model, tokenizer).

    device defaults to cuda when available, otherwise cpu.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = AutoModel.from_pretrained(
        model_name, dtype=dtype, attn_implementation="eager",
        output_loading_info=True,
    )
    model = model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)

    if not hasattr(model.video_model, "contrastive_vision_pooler"):
        # Stock layout: from_pretrained has loaded every checkpoint tensor.
        _report(info.get("missing_keys", []), info.get("unexpected_keys", []))
        return model.to(device), tok

    # Nested layout: load the renamed checkpoint.
    state = load_lvt_state_dict(model_name)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # Manually set the pooler's `scale` buffer to its formula-derived default.
    # The model class registers it as a buffer in __init__ via:
    #   r_softplus_0 = 1.442695041
    #   scale = r_softplus_0 / sqrt(dim)
    # where dim = intermediate_size / num_attention_heads = 256 for base.
    # The lazy-init path leaves it as uninitialized memory, so it is recomputed.
    pooler = model.video_model.contrastive_vision_pooler
    r_softplus_0 = 1.442695041
    dim = pooler.dim
    pooler.scale = torch.full_like(pooler.scale, r_softplus_0 / math.sqrt(dim))
    _report(missing, unexpected)
    return model.to(device), tok


if __name__ == "__main__":
    import torch.nn.functional as F

    rename = _build_rename_map()
    print(f"rename map size: {len(rename)}")
    m, tok = load_lvt_model_fixed()
    device = next(m.parameters()).device
    # Quick text encoder sanity (5 captions, two identical)
    captions = [
        "a video of a man surfing on a wave.",
        "a video of a woman cooking pasta in a kitchen.",
        "a video of children playing soccer on a field.",
        "a video of a dog running in the park.",
        "a video of a man surfing on a wave.",  # same as [0]
    ]
    with torch.inference_mode():
        enc = tok(captions, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        out = m.text_model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        emb = text_embeddings(out)
    print("text emb shape:", emb.shape)
    en = F.normalize(emb.float(), dim=-1)
    print("text-text sim matrix:")
    print((en @ en.T).cpu())
