"""Configuration loading.

load_config reads a YAML config and, for ActivityNet, fills the data paths that
depend on `activitynet_retrieval_mode`.

ActivityNet supports two retrieval modes: "paragraph" (default, all sentences
per video concatenated into one query) and "sentence" (each sentence is a
separate query). Flipping `activitynet_retrieval_mode` in the config selects
all five dependent paths (annotation files, hard negatives, stage-1 embeddings).
An explicit config key always wins over the mode-derived default.
"""

from ruamel.yaml import YAML

_ACTIVITYNET_DATA_ROOT = "./data_root/activitynet"

ACTIVITYNET_KEYS = (
    "activitynet_train_ann",
    "activitynet_test_ann",
    "hard_negatives_path",
    "lvt_embeds_path",
    "lvt_test_features_path",
)


def _activitynet_paths_for_mode(mode: str) -> dict:
    return {
        "activitynet_train_ann":
            f"{_ACTIVITYNET_DATA_ROOT}/activitynet_train_{mode}.json",
        "activitynet_test_ann":
            f"{_ACTIVITYNET_DATA_ROOT}/activitynet_test_{mode}.json",
        "hard_negatives_path":
            f"{_ACTIVITYNET_DATA_ROOT}/hard_negatives_text2vid_top50_{mode}.json",
        "lvt_embeds_path":
            f"{_ACTIVITYNET_DATA_ROOT}/lvt_clip_embeds_train_{mode}.pt",
        "lvt_test_features_path":
            f"{_ACTIVITYNET_DATA_ROOT}/lvt_clip_embeds_test_{mode}.pt",
    }


def _resolve_activitynet_path(config: dict, key: str) -> str:
    """Return explicit config value if set, else the mode-derived default."""
    if key in config and config[key]:
        return config[key]
    mode = config.get("activitynet_retrieval_mode", "paragraph")
    return _activitynet_paths_for_mode(mode)[key]


def uses_activitynet(config: dict) -> bool:
    return (config.get("train_dataset") == "activitynet"
            or config.get("test_set") == "activitynet")


def resolve_activitynet_paths(config: dict) -> dict:
    """Fill the ActivityNet data paths that the config leaves unset (in place)."""
    if uses_activitynet(config):
        for key in ACTIVITYNET_KEYS:
            if not config.get(key):
                config[key] = _resolve_activitynet_path(config, key)
    return config


def load_config(path, resolve_activitynet: bool = True) -> dict:
    """Load a YAML config as a dict.

    ActivityNet configs derive their data paths from `activitynet_retrieval_mode`
    unless the config sets them.
    """
    with open(path, "r") as f:
        config = YAML(typ="safe").load(f)
    if resolve_activitynet:
        resolve_activitynet_paths(config)
    return config
