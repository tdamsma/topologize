"""
Quality benchmark: output complexity at various buffer distances.

Reports: chains, total points, mean/max chain length.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from topologize import topologize
from topologize.helpers import load_svg
from inputs import SIMPLE_CURVES, SIMPLE_BUFFERS, SVG_PATH, SVG_BUFFERS


def _chain_stats(chains):
    if not chains:
        return 0, 0, 0.0, 0
    lengths = [len(c) for c in chains]
    return len(chains), sum(lengths), sum(lengths) / len(lengths), max(lengths)


def run_dataset(label, curves, buffer_distances):
    rows = []
    for bd in buffer_distances:
        try:
            chains = topologize(curves, bd)
            n_chains, total_pts, mean_len, max_len = _chain_stats(chains)
        except Exception:
            n_chains, total_pts, mean_len, max_len = -1, -1, -1.0, -1
        rows.append({
            "dataset": label,
            "buf": bd,
            "chains": n_chains,
            "total_pts": total_pts,
            "mean_len": mean_len,
            "max_len": max_len,
        })
    return rows


def run():
    rows = run_dataset("simple", SIMPLE_CURVES, SIMPLE_BUFFERS)
    if SVG_PATH.exists():
        svg_curves = load_svg(str(SVG_PATH))
        rows += run_dataset("svg", svg_curves, SVG_BUFFERS)
    else:
        print(f"  [skip] SVG not found: {SVG_PATH}")
    return rows


if __name__ == "__main__":
    rows = run()
    HDR = f"{'dataset':<8}  {'buf':>6}  {'chains':>6}  {'total_pts':>9}  {'mean_len':>8}  {'max_len':>7}"
    SEP = "-" * len(HDR)
    print(HDR)
    print(SEP)
    prev_buf = None
    for r in rows:
        if prev_buf and r["buf"] != prev_buf:
            print()
        prev_buf = r["buf"]
        print(
            f"{r['dataset']:<8}  {r['buf']:>6.2f}  {r['chains']:>6d}  "
            f"{r['total_pts']:>9d}  {r['mean_len']:>8.1f}  {r['max_len']:>7d}"
        )
    print(SEP)
