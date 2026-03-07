"""CLI: topologize an SVG file and print summary statistics."""
import argparse
import sys
import time

import numpy as np

from topologize import inflate, topologize


def load_svg(path, sample_distance=5.0):
    try:
        from svgpathtools import svg2paths2
    except ImportError:
        print("svgpathtools not installed — run: uv sync")
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


def main():
    parser = argparse.ArgumentParser(description="Topologize an SVG file.")
    parser.add_argument("svg", help="Path to input SVG file")
    parser.add_argument("--buffer", type=float, default=10.0, help="Buffer distance (default: 10)")
    args = parser.parse_args()

    curves = load_svg(args.svg)

    polygons = inflate(curves, args.buffer)
    n_holes = sum(len(holes) for _, holes in polygons)

    t0 = time.perf_counter()
    result = topologize(curves, args.buffer)
    chains = result.chains
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"input curves  : {len(curves)}")
    print(f"buffer        : {args.buffer}")
    print(f"exterior rings: {len(polygons)}")
    print(f"interior rings: {n_holes}")
    print(f"output chains : {len(chains)}")
    print(f"total points  : {sum(len(c) for c in chains)}")
    print(f"elapsed       : {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()
