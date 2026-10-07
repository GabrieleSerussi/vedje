from pathlib import Path

import pytest

from vedje.config import ACTIVITYNET_KEYS, load_config
from vedje.data import DATASET_REGISTRY

CONFIGS = sorted((Path(__file__).resolve().parent.parent / "configs").glob("*.yaml"))
REQUIRED = ("language_model_path", "num_frames", "num_queries_per_frame", "loss_weight_vtm",
            "loss_weight_vtc", "loss_weight_mlm", "loss_weight_delta", "delta_horizons", "init_lr",
            "warmup_steps", "max_epoch", "batch_size", "k", "train_dataset", "test_set")


def test_five_configs_are_present():
    assert [c.name for c in CONFIGS] == [
        "vedje_vclip_msrvtt.yaml", "vedje_vp_activitynet.yaml", "vedje_vp_didemo.yaml",
        "vedje_vp_msrvtt.yaml", "vedje_vp_msvd.yaml",
    ]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads(path):
    config = load_config(path)
    for key in REQUIRED:
        assert key in config, key
    assert config["train_dataset"] in DATASET_REGISTRY and config["test_set"] in DATASET_REGISTRY
    assert config["experiment"]["backend"] == "none"  # logging is off by default
    for key, value in config.items():
        if isinstance(value, str) and ("/" in value) and not value.startswith(("MHRDYN7/", "microsoft/")):
            assert value.startswith("./data_root/"), (key, value)
    # the cache holds 64 tokens in every shipped config
    assert config["num_frames"] * config["num_queries_per_frame"] == 64


def test_activitynet_paths_follow_the_retrieval_mode():
    config = load_config(Path(__file__).resolve().parent.parent / "configs" / "vedje_vp_activitynet.yaml")
    for key in ACTIVITYNET_KEYS:
        assert config[key].startswith("./data_root/activitynet/"), key
    assert config["activitynet_train_ann"].endswith("activitynet_train_paragraph.json")
    assert config["hard_negatives_path"].endswith("hard_negatives_text2vid_top50_paragraph.json")
    raw = load_config(Path(__file__).resolve().parent.parent / "configs" / "vedje_vp_activitynet.yaml",
                      resolve_activitynet=False)
    assert "activitynet_train_ann" not in raw
