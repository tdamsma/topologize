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

from topologize import inflate, triangulate, topologize


def load_svg(path, sample_distance=5.0):
    try:
        from svgpathtools import svg2paths2
    except ImportError:
        print("svgpathtools not installed — run: uv add --dev svgpathtools")
        sys.exit(1)
    paths, *_ = svg2paths2(path)
    curves = []
    for svg_path in paths:
        for subpath in svg_path.continuous_subpaths():
            pts = []
            for seg in subpath:
                n = max(1, int(seg.length() / sample_distance))
                if not pts:
                    pts.append(seg.point(0))
                for i in range(1, n + 1):
                    pts.append(seg.point(i / n))
            if len(pts) >= 2:
                curves.append(np.array([(p.real, p.imag) for p in pts]))
    return curves

# %%


def _chains_to_xy(chains):
    """Concatenate chains with NaN separators → (x, y) arrays for one trace."""
    NAN = np.full((1, 2), np.nan)
    pts = np.concatenate([np.vstack([c, NAN]) for c in chains])
    return pts[:, 0], pts[:, 1]


def _buffer_rings(curves, buffer_distance):
    """Return NaN-separated (x, y) arrays for all buffer polygon rings."""
    NAN = np.full((1, 2), np.nan)
    polygons = inflate(curves, buffer_distance)
    rings = [r for outer, holes in polygons for r in [outer, *holes]]
    if not rings:
        return None, None
    buf_pts = np.concatenate([np.vstack([r, NAN]) for r in rings])
    return buf_pts[:, 0], buf_pts[:, 1]


def _triangles_to_xy(triangles):
    """Convert list of (a, b, c) vertex tuples → NaN-separated closed loops."""
    NAN = np.array([[np.nan, np.nan]])
    parts = []
    for (ax, ay), (bx, by), (cx, cy) in triangles:
        parts.append(np.array([[ax, ay], [bx, by], [cx, cy], [ax, ay]]))
        parts.append(NAN)
    if not parts:
        return None, None
    pts = np.concatenate(parts)
    return pts[:, 0], pts[:, 1]


def plot(curves, chains, buffer_distance, show_cdt=False):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — run: uv add --dev plotly")
        sys.exit(1)

    NAN = np.full((1, 2), np.nan)

    input_pts = np.concatenate([np.vstack([c, NAN]) for c in curves])
    x, y = _chains_to_xy(chains)
    bx, by = _buffer_rings(curves, buffer_distance)

    traces = [
        go.Scatter(
            x=input_pts[:, 0], y=input_pts[:, 1], mode="lines",
            name="input",
            line=dict(color="rgba(180,180,180,0.6)", width=5),
        ),
    ]
    if bx is not None:
        traces.append(go.Scatter(
            x=bx, y=by, mode="lines",
            name="buffer",
            line=dict(color="rgba(80,120,200,0.45)", width=1, dash="dot"),
        ))
    if show_cdt:
        tris = triangulate(curves, buffer_distance)
        tx, ty = _triangles_to_xy(tris)
        if tx is not None:
            traces.append(go.Scatter(
                x=tx, y=ty, mode="lines",
                name="CDT",
                line=dict(color="rgba(0,160,80,0.35)", width=0.5),
            ))
    traces.append(go.Scatter(
        x=x, y=y, mode="lines",
        name="centerline",
        line=dict(color="#e41a1c", width=1.5),
    ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Centerline — buf={buffer_distance}  |  {len(chains)} chains, {sum(len(c) for c in chains)} pts",
        yaxis_scaleanchor="x",
        yaxis_autorange="reversed",
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
    parser.add_argument("--cdt", action="store_true", help="Overlay the CDT triangulation")
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
    result = topologize(curves, args.buffer)
    chains = result.chains
    ms = (time.perf_counter() - t0) * 1000
    print(f"topologize: {len(chains)} chains, {sum(len(c) for c in chains)} pts  {ms:.0f} ms")

    plot(curves, chains, args.buffer, show_cdt=args.cdt)


# %%

if __name__ == "__main__":
    main()
