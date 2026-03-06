# %%
"""
SVG centerline extraction example.

Extracts the centerline skeleton from an SVG file and plots it.

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


def plot(curves, chains, buffer_distance):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — run: uv add --dev plotly")
        sys.exit(1)

    NAN = np.full((1, 2), np.nan)

    input_pts = np.concatenate([np.vstack([c, NAN]) for c in curves])
    x, y = _chains_to_xy(chains)

    fig = go.Figure([
        go.Scatter(
            x=input_pts[:, 0], y=input_pts[:, 1], mode="lines",
            name="input",
            line=dict(color="rgba(180,180,180,0.6)", width=5),
        ),
        go.Scatter(
            x=x, y=y, mode="lines",
            name="centerline",
            line=dict(color="#e41a1c", width=1.5),
        ),
    ])
    fig.update_layout(
        title=f"Centerline — buf={buffer_distance}  |  {len(chains)} chains, {sum(len(c) for c in chains)} pts",
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
    parser = argparse.ArgumentParser(description="SVG centerline extraction.")
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

    t0 = time.perf_counter()
    chains = topologize(curves, args.buffer)
    ms = (time.perf_counter() - t0) * 1000
    print(f"topologize: {len(chains)} chains, {sum(len(c) for c in chains)} pts  {ms:.0f} ms")

    plot(curves, chains, args.buffer)


# %%

if __name__ == "__main__":
    main()
