"""
Timing benchmark: raw CPU timing at various buffer distances (3 runs, midpoint backend).
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from topologize import topologize
from topologize.helpers import load_svg
from inputs import SIMPLE_CURVES, SIMPLE_BUFFERS, SVG_PATH, SVG_BUFFERS

N_RUNS = 3


def _time_run(curves, buffer_distance, n_runs):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        topologize(curves, buffer_distance)
        times.append((time.perf_counter() - t0) * 1000)
    return {"min_ms": min(times), "mean_ms": sum(times) / len(times), "max_ms": max(times)}


def run_dataset(label, curves, buffer_distances):
    rows = []
    for bd in buffer_distances:
        try:
            t = _time_run(curves, bd, N_RUNS)
        except Exception:
            t = {"min_ms": -1, "mean_ms": -1, "max_ms": -1}
        rows.append({"dataset": label, "buf": bd, **t})
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
    HDR = f"{'dataset':<8}  {'buf':>6}  {'min_ms':>8}  {'mean_ms':>8}  {'max_ms':>8}"
    SEP = "-" * len(HDR)
    print(HDR)
    print(SEP)
    prev_buf = None
    for r in rows:
        if prev_buf and r["buf"] != prev_buf:
            print()
        prev_buf = r["buf"]
        print(
            f"{r['dataset']:<8}  {r['buf']:>6.2f}  "
            f"{r['min_ms']:>8.1f}  {r['mean_ms']:>8.1f}  {r['max_ms']:>8.1f}"
        )
    print(SEP)
