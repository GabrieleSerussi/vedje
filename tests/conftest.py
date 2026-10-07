import os
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _online() -> bool:
    if os.environ.get("VEDJE_OFFLINE") == "1":
        return False
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except OSError:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "network: downloads model checkpoints from the Hugging Face Hub")


def pytest_collection_modifyitems(config, items):
    if any("network" in item.keywords for item in items) and not _online():
        reason = "VEDJE_OFFLINE=1" if os.environ.get("VEDJE_OFFLINE") == "1" else "no network"
        marker = pytest.mark.skip(reason=f"{reason}: the test downloads checkpoints")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(marker)


@pytest.fixture(scope="session")
def tiny_lm(tmp_path_factory):
    """A tiny BERT masked language model and tokenizer saved locally (no download)."""
    from transformers import BertConfig, BertForMaskedLM, BertTokenizer

    path = tmp_path_factory.mktemp("tiny_lm")
    words = ["a", "man", "is", "cooking", "two", "dogs", "run", "video", "of", "the", "in", "kitchen",
             "t0", "t1", "t2", "t3", "t4", "t5"]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizer.from_pretrained(str(path)).save_pretrained(str(path))
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=512)
    BertForMaskedLM(config).save_pretrained(str(path))
    return str(path)
