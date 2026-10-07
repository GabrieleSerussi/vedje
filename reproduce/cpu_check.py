"""Reproduce numbers of the paper on a CPU in under a minute.

Every line prints the paper's value, the value computed from this package and
OK or FAIL. The parameter counts build the reranker that scripts/train.py
trains, which downloads MiniLM-L12-H384 (about 130 MB) on the first run; with
VEDJE_OFFLINE=1, or when the download fails, those lines are skipped.

    python reproduce/cpu_check.py

Exit status 1 on any FAIL.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# keep the output to the checks: no weight-loading report, hub notices or progress bars
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch  # noqa: E402

from vedje.cache import FramewiseCrossAttentionCompressor, cache_bytes  # noqa: E402
from vedje.delta import delta_targets, valid_pairs  # noqa: E402

KIB, MIB = 1024, 1024 ** 2
rows = []


def check(name, source, paper, computed, ok):
    rows.append((name, source, paper, computed, "OK" if ok else "FAIL"))
    print(f"{'OK  ' if ok else 'FAIL'}  {name}  [{source}]\n      paper: {paper}\n      computed: {computed}")


def skip(name, source, reason):
    rows.append((name, source, "", "", "SKIP"))
    print(f"SKIP  {name}  [{source}]\n      {reason}")


def measured_cache_bytes(tokens_per_frame, num_frames=16, dim=384):
    """Bytes of a real compressor output in BF16 for one video of random features."""
    torch.manual_seed(0)
    compressor = FramewiseCrossAttentionCompressor(
        num_queries_per_frame=tokens_per_frame, vision_dim=768, bert_dim=dim,
        patches_per_frame=256, max_frames=num_frames,
    ).to(torch.bfloat16).eval()
    patches = torch.randn(1, num_frames * 256, 768, dtype=torch.bfloat16)
    with torch.inference_mode():
        cache = compressor(patches)[0]  # (T*M, d), the cache of one video
    return cache.numel() * cache.element_size(), tuple(cache.shape)


def main():
    start = time.time()

    # Section 3.2 and Appendix A.2: cache payload, measured on compressor outputs.
    b64, shape64 = measured_cache_bytes(4)
    check("cache payload, 16 frames x 4 tokens, d = 384, BF16", "Section 3.2, Appendix A.2",
          "49,152 bytes (48 KiB)", f"{b64:,} bytes ({b64 / KIB:g} KiB), cache tensor {shape64}",
          b64 == 49152 == cache_bytes(16, 4, 384))
    b16, shape16 = measured_cache_bytes(1)
    check("cache payload, 16 frames x 1 token, d = 384, BF16", "Section 3.2, Appendix A.2",
          "12,288 bytes (12 KiB)", f"{b16:,} bytes ({b16 / KIB:g} KiB), cache tensor {shape16}",
          b16 == 12288 == cache_bytes(16, 1, 384))

    # Table 9: the frame-and-patch representation and the storage ratio.
    full = cache_bytes(16, 257, 768)
    check("frame-and-patch payload, 16 x 257 x 768 x 2 bytes", "Table 9",
          "6.02 MiB", f"{full:,} bytes = {full / MIB:.2f} MiB", f"{full / MIB:.2f}" == "6.02")
    ratio = full / b64
    check("storage ratio, frame-and-patch over the 64-token cache", "Section 4.3, Table 9",
          "128.5", f"{ratio:.1f}", f"{ratio:.1f}" == "128.5")

    # Section 4.7: cache payload read per query for 20 candidates.
    read64, read16 = 20 * b64, 20 * b16
    check("video payload read for 20 candidates, 64 and 16 tokens", "Section 4.7",
          "0.983 MB and 0.246 MB", f"{read64 / 1e6:.3f} MB and {read16 / 1e6:.3f} MB",
          (f"{read64 / 1e6:.3f}", f"{read16 / 1e6:.3f}") == ("0.983", "0.246"))

    # Eq. 4: valid (t, h) pairs for T = 16 and H = {3}.
    pairs = valid_pairs(16, [3])
    targets = delta_targets(torch.randn(1, 16 * 4, 8), num_frames=16, patches_per_frame=4, horizons=[3])
    check("valid (t, h) pairs of L_delta, T = 16, H = {3}", "Eq. 4",
          "13, derived from the definition of Omega = {(t, h) : 1 <= t, t + h <= T}",
          f"{len(pairs)} pairs, t = {pairs[0][0] + 1} to {pairs[-1][0] + 1} (one-based); "
          f"{len(targets)} targets",
          len(pairs) == 13 == len(targets))

    # Appendix A.2: parameter counts of the model this package builds.
    source = "Appendix A.2"
    names = ("joint encoder", "score head", "prior embedding plus score head")
    if os.environ.get("VEDJE_OFFLINE") == "1":
        for n in names:
            skip(f"parameters, {n}", source, "VEDJE_OFFLINE=1: the MiniLM download is skipped")
    else:
        try:
            from vedje.config import load_config
            from vedje.model import build_model, parameter_counts
            config = load_config(ROOT / "configs" / "vedje_vp_msrvtt.yaml")
            counts = parameter_counts(build_model(config, training=True))
        except Exception as e:  # no network and no cached checkpoint
            counts = None
            for n in names:
                skip(f"parameters, {n}", source, f"could not build the model ({type(e).__name__}: {e})")
        if counts is not None:
            joint, head, prior = counts["joint_encoder"], counts["score_head"], counts["prior"]
            check("parameters, joint encoder (MiniLM-L12-H384 with embeddings)", source,
                  "about 33M", f"{joint:,} = {joint / 1e6:.1f}M", round(joint / 1e6) == 33)
            check("parameters, score head h_psi", source,
                  "about 0.15M", f"{head:,} = {head / 1e6:.2f}M", round(head / 1e6, 2) == 0.15)
            check("parameters, prior embedding e_rho plus score head", source,
                  "fewer than 0.2M", f"{prior:,} + {head:,} = {(prior + head) / 1e6:.2f}M",
                  prior + head < 200_000)

    failed = sum(r[4] == "FAIL" for r in rows)
    skipped = sum(r[4] == "SKIP" for r in rows)
    print(f"\n{len(rows) - failed - skipped} OK, {failed} FAIL, {skipped} SKIP "
          f"in {time.time() - start:.1f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
