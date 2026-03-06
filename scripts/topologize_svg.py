"""CLI: topologize an SVG file and print summary statistics."""
import argparse
import time

from topologize import topologize
from topologize.helpers import load_svg


def main():
    parser = argparse.ArgumentParser(description="Topologize an SVG file.")
    parser.add_argument("svg", help="Path to input SVG file")
    parser.add_argument("--buffer", type=float, default=10.0, help="Buffer distance (default: 10)")
    args = parser.parse_args()

    curves = load_svg(args.svg)

    t0 = time.perf_counter()
    chains = topologize(curves, args.buffer)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"input curves : {len(curves)}")
    print(f"buffer       : {args.buffer}")
    print(f"output chains: {len(chains)}")
    print(f"total points : {sum(len(c) for c in chains)}")
    print(f"elapsed      : {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()
