"""Generate docs/example_topologize.png for the README.

Run:
    uv run python scripts/generate_readme_image.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from topologize import inflate, topologize

SVG = Path(__file__).parent.parent / "python/examples/data/topologize.svg"
BUFFER = 0.47
OUT = Path(__file__).parent.parent / "docs/example_topologize.png"


def load_svg(path, sample_distance=5.0):
    try:
        from svgpathtools import parse_path, Path, Line, CubicBezier, QuadraticBezier
    except ImportError:
        print("svgpathtools not installed — run: uv sync")
        sys.exit(1)
    import re
    import xml.etree.ElementTree as ET

    def parse_transform(t):
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


def draw_buffer(ax, polygons, color, alpha):
    patches = []
    for outer, holes in polygons:
        patches.append(MplPolygon(outer, closed=True))
    col = PatchCollection(patches, facecolor=color, edgecolor="none", alpha=alpha, zorder=1)
    ax.add_collection(col)
    # Draw holes as white on top
    hole_patches = []
    for outer, holes in polygons:
        for hole in holes:
            hole_patches.append(MplPolygon(hole, closed=True))
    if hole_patches:
        hcol = PatchCollection(hole_patches, facecolor="white", edgecolor="none", zorder=2)
        ax.add_collection(hcol)


def main():
    print(f"Loading {SVG} ...")
    curves = load_svg(str(SVG), sample_distance=BUFFER / 2)
    print(f"  {len(curves)} curves, {sum(len(c) for c in curves)} pts")

    polygons = inflate(curves, BUFFER)
    n_holes = sum(len(h) for _, h in polygons)
    print(f"  inflate: {len(polygons)} exterior, {n_holes} interior rings")

    result = topologize(curves, BUFFER)
    chains = result.chains
    print(f"  topologize: {len(chains)} chains, {sum(len(c) for c in chains)} pts")

    # Auto-fit bounds from input curves
    all_pts = np.concatenate(curves)
    xmin, ymin = all_pts.min(axis=0) - BUFFER * 4
    xmax, ymax = all_pts.max(axis=0) + BUFFER * 4

    BUF_COLOR = "#cfe0f5"
    INPUT_COLOR = "#4878cf"
    CHAIN_COLOR = "#c0392b"

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)  # invert y for SVG coordinates
        ax.axis("off")

    # --- Left: Input ---
    ax = axes[0]
    ax.set_title("Input — overlapping / bundled polylines",
                 fontsize=13, fontweight="bold", pad=10)
    draw_buffer(ax, polygons, BUF_COLOR, alpha=0.7)
    for c in curves:
        ax.plot(c[:, 0], c[:, 1], color=INPUT_COLOR, lw=1.2, solid_capstyle="round",
                solid_joinstyle="round", zorder=3)

    # --- Right: Output ---
    ax = axes[1]
    ax.set_title("Output — clean centerline chains",
                 fontsize=13, fontweight="bold", pad=10)
    for chain in chains:
        ax.plot(chain[:, 0], chain[:, 1], color=CHAIN_COLOR, lw=1.5,
                solid_capstyle="round", solid_joinstyle="round", zorder=3)

    plt.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
