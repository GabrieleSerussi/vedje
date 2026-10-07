import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["extract_features.py", "stage1_train_embeddings.py", "mine_hard_negatives.py",
           "stage1_test_features.py", "train.py", "evaluate.py"]


def test_every_script_help_exits_0_on_cpu():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    procs = {name: subprocess.Popen([sys.executable, str(ROOT / "scripts" / name), "--help"],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=ROOT)
             for name in SCRIPTS}  # run in parallel to keep the test fast
    for name, proc in procs.items():
        out, err = proc.communicate(timeout=300)
        assert proc.returncode == 0, (name, err.decode()[-2000:])
        assert out.decode().startswith("usage:"), name


def test_cpu_check_and_paper_results_exist():
    assert (ROOT / "reproduce" / "cpu_check.py").is_file()
    from vedje.paper import load_results
    results = load_results()
    for block in ("table_1b", "table_3", "table_4", "table_5", "table_6", "table_9", "table_12",
                  "figure_1_table_22"):
        assert results[block]["source"], block


def test_hard_negatives_are_the_most_similar_wrong_videos():
    spec = importlib.util.spec_from_file_location("mine_hard_negatives", ROOT / "scripts" / "mine_hard_negatives.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    videos = torch.eye(4)
    captions = torch.tensor([[0.9, 0.5, 0.1, 0.0], [0.2, 0.8, 0.0, 0.6]])  # captions of videos 0 and 1
    negatives = script.mine(captions, videos, [0, 1], ["v0", "v1", "v2", "v3"], top_k=2)
    assert negatives == {"0": ["v1", "v2"], "1": ["v3", "v0"]}
