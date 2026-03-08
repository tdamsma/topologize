# %%
"""
SVG centerline extraction example.

Extracts the centerline skeleton from an SVG file and plots it.

Usage:
    uv run python/examples/svg_centerline.py python/examples/data/input.svg --buffer 10

Requires plotly (dev dependency):
    uv sync --group dev
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
    """Load SVG paths with full group-transform support."""
    try:
        from svgpathtools import parse_path, Path, Line, CubicBezier, QuadraticBezier
    except ImportError:
        print("svgpathtools not installed — run: uv sync")
        sys.exit(1)
    import re
    import xml.etree.ElementTree as ET

    def parse_transform(t):
        """SVG transform string → 3×3 numpy affine matrix."""
        mat = np.eye(3)
        if not t:
            return mat
        for m in re.finditer(r'(\w+)\(([^)]*)\)', t):
            func = m.group(1)
            args = [float(x) for x in re.split(r'[\s,]+', m.group(2).strip()) if x]
            if func == 'matrix' and len(args) == 6:
                a, b, c, d, e, f = args
                mat = mat @ np.array([[a, c, e], [b, d, f], [0, 0, 1]])
            elif func == 'translate':
                tx, ty = args[0], (args[1] if len(args) > 1 else 0)
                mat = mat @ np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            elif func == 'scale':
                sx = args[0]
                sy = args[1] if len(args) > 1 else sx
                mat = mat @ np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
            elif func == 'rotate':
                angle = np.radians(args[0])
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
                if len(args) == 3:
                    cx, cy = args[1], args[2]
                    T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
                    mat = mat @ T @ R @ np.linalg.inv(T)
                else:
                    mat = mat @ R
        return mat

    def xform(z, mat):
        """Apply affine matrix to complex coordinate."""
        xn = mat[0, 0] * z.real + mat[0, 1] * z.imag + mat[0, 2]
        yn = mat[1, 0] * z.real + mat[1, 1] * z.imag + mat[1, 2]
        return complex(xn, yn)

    def transform_path(svg_path, mat):
        segs = []
        for seg in svg_path:
            if isinstance(seg, Line):
                segs.append(Line(xform(seg.start, mat), xform(seg.end, mat)))
            elif isinstance(seg, CubicBezier):
                segs.append(CubicBezier(
                    xform(seg.start, mat), xform(seg.control1, mat),
                    xform(seg.control2, mat), xform(seg.end, mat),
                ))
            elif isinstance(seg, QuadraticBezier):
                segs.append(QuadraticBezier(
                    xform(seg.start, mat), xform(seg.control, mat), xform(seg.end, mat),
                ))
            else:
                # Arc or unknown: sample densely into lines
                try:
                    n = max(8, int(seg.length() / sample_distance))
                except Exception:
                    n = 16
                pts_z = [seg.point(i / n) for i in range(n + 1)]
                for i in range(len(pts_z) - 1):
                    segs.append(Line(xform(pts_z[i], mat), xform(pts_z[i + 1], mat)))
        return Path(*segs)

    def walk(element, parent_mat):
        mat = parent_mat @ parse_transform(element.get('transform', ''))
        tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        if tag == 'path':
            d = element.get('d', '')
            if d:
                try:
                    yield transform_path(parse_path(d), mat)
                except Exception:
                    pass
        for child in element:
            yield from walk(child, mat)

    tree = ET.parse(path)
    root = tree.getroot()
    curves = []
    for svg_path in walk(root, np.eye(3)):
        for subpath in svg_path.continuous_subpaths():
            pts = []
            for seg in subpath:
                try:
                    n = max(1, int(seg.length() / sample_distance))
                except Exception:
                    n = 4
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
        print("plotly not installed — run: uv sync --group dev")
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
    parser.add_argument("--junction-merge-fraction", type=float, default=None,
                        help="Junction merge fraction (default: 1.5; set 0 to disable)")
    args = parser.parse_args()

    if not args.svg:
        parser.print_help()
        sys.exit(1)

    svg_path = Path(args.svg)
    if not svg_path.exists():
        print(f"File not found: {svg_path}")
        sys.exit(1)

    t0 = time.perf_counter()
    curves = load_svg(str(svg_path), sample_distance=args.buffer / 5)
    print(f"Loaded {svg_path}: {len(curves)} curves, "
          f"{sum(len(c) for c in curves)} pts  "
          f"{(time.perf_counter()-t0)*1000:.0f} ms")

    polygons = inflate(curves, args.buffer)
    n_holes = sum(len(holes) for _, holes in polygons)
    print(f"inflate: {len(polygons)} exterior rings, {n_holes} interior rings")

    t0 = time.perf_counter()
    kwargs = {}
    if args.junction_merge_fraction is not None:
        kwargs["junction_merge_fraction"] = args.junction_merge_fraction
    result = topologize(curves, args.buffer, **kwargs)
    chains = result.chains
    ms = (time.perf_counter() - t0) * 1000
    print(f"topologize: {len(chains)} chains, {sum(len(c) for c in chains)} pts  {ms:.0f} ms")

    plot(curves, chains, args.buffer, show_cdt=args.cdt)


# %%

if __name__ == "__main__":
    main()
