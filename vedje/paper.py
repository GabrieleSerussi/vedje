"""The paper's reported numbers, as printed in its tables and figures.

    from vedje.paper import load_results
    results = load_results()
    results["table_1b"]["rows"][0]   # VideoPrism first stage, MSR-VTT

The numbers live in artifacts/paper_results.json (or a gzipped copy next to it);
each block names its table or figure.
"""

import gzip
import json
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "paper_results.json"


def load_results(path=None) -> dict:
    """Load the paper's numbers from artifacts/paper_results.json (.json or .json.gz)."""
    path = Path(path) if path is not None else RESULTS_PATH
    if not path.exists():
        gz = path.with_name(path.name + ".gz")
        if gz.exists():
            path = gz
        else:
            raise FileNotFoundError(
                f"{path} not found; load_results reads the artifacts/ folder of a repository clone")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)
