# %%
"""
SVG centerline extraction example.

Runs three skeletonization methods side-by-side and plots them as single
colour-coded layers on one figure:
  - input strokes (light gray, wide)
  - python  — shapely + triangle prototype
  - midpoint — Rust CDT midpoint skeleton
  - voronoi  — Rust Boost Voronoi skeleton

Usage:
    uv run python/examples/svg_centerline.py python/examples/data/input.svg --buffer 10

Requires plotly (dev dependency):
    uv add --dev plotly
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from topologize import topologize
from topologize.helpers import load_svg

# %%


def _chains_to_xy(chains):
    """Concatenate chains with NaN separators → (x, y) arrays for one trace."""
    NAN = np.full((1, 2), np.nan)
    pts = np.concatenate([np.vstack([c, NAN]) for c in chains])
    return pts[:, 0], pts[:, 1]


def _run(label, fn):
    t0 = time.perf_counter()
    result = fn()
    ms = (time.perf_counter() - t0) * 1000
    n_chains = len(result)
    n_pts = sum(len(c) for c in result)
    lengths = sorted((len(c) for c in result), reverse=True)
    median = lengths[len(lengths) // 2] if lengths else 0
    print(f"  {label}: {n_chains} chains, {n_pts} pts  "
          f"[longest={lengths[0] if lengths else 0}, median={median}]  {ms:.0f} ms")
    return result


def plot(curves, results, buffer_distance):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — run: uv add --dev plotly")
        sys.exit(1)

    NAN = np.full((1, 2), np.nan)
    METHOD_COLORS = {
        "python":        "#377eb8",
        "midpoint-spade": "#e41a1c",
        "midpoint-cdt":  "#ff7f00",
        "voronoi":       "#4daf4a",
    }

    traces = []

    # Input strokes — single trace
    input_pts = np.concatenate([np.vstack([c, NAN]) for c in curves])
    traces.append(go.Scatter(
        x=input_pts[:, 0], y=input_pts[:, 1], mode="lines",
        name="input",
        line=dict(color="rgba(180,180,180,0.6)", width=5),
    ))

    # One trace per method
    for label, chains in results.items():
        if not chains:
            continue
        x, y = _chains_to_xy(chains)
        traces.append(go.Scatter(
            x=x, y=y, mode="lines",
            name=label,
            line=dict(color=METHOD_COLORS.get(label, "#333"), width=1.5),
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Centerline comparison — buf={buffer_distance}  "
              + "  |  ".join(
                  f"{lbl}: {sum(len(c) for c in ch)} pts"
                  for lbl, ch in results.items() if ch
              ),
        yaxis_scaleanchor="x",
        width=1200,
        height=750,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    fig.write_html(tmp.name)
    subprocess.Popen(["open", tmp.name])


# %%

def main():
    parser = argparse.ArgumentParser(description="Compare SVG centerline methods.")
    parser.add_argument("svg", nargs="?", help="Path to input SVG file")
    parser.add_argument("--buffer", type=float, default=20.0, help="Buffer distance (default: 20)")
    args = parser.parse_args()

    if not args.svg:
        parser.print_help()
        sys.exit(1)

    svg_path = Path(args.svg)
    if not svg_path.exists():
        print(f"File not found: {svg_path}")
        sys.exit(1)

    t0 = time.perf_counter()
    curves = load_svg(str(svg_path))
    print(f"Loaded {svg_path}: {len(curves)} curves, "
          f"{sum(len(c) for c in curves)} pts  "
          f"{(time.perf_counter()-t0)*1000:.0f} ms")

    buf = args.buffer
    results = {}

    # Python prototype (shapely + triangle)
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from prototype import extract_centerline_python
        results["python"] = _run("python", lambda: extract_centerline_python(curves, buf))
    except ImportError as e:
        print(f"  python method unavailable ({e})")

    # Rust midpoint-spade
    results["midpoint-spade"] = _run("midpoint-spade", lambda: topologize(curves, buf, method="midpoint-spade"))

    # Rust midpoint-cdt
    results["midpoint-cdt"] = _run("midpoint-cdt", lambda: topologize(curves, buf, method="midpoint-cdt"))

    # Rust voronoi
    try:
        results["voronoi"] = _run("voronoi", lambda: topologize(curves, buf, method="voronoi"))
    except BaseException as e:
        print(f"  voronoi: failed ({type(e).__name__}: {e})")

    plot(curves, results, buf)


# %%

if __name__ == "__main__":
    main()
