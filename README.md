# topologize

Convert messy, overlapping polylines into clean centerline chains.

Given a set of curves (open or closed, possibly intersecting), `topologize` inflates them into a region, skeletonizes that region, and returns a list of maximal non-branching polylines — one per continuous segment of the medial axis.

## Use case

The typical input is a bundle of strokes that approximate the same underlying path (e.g. vector artwork, GPS traces, hand-drawn curves). The output is a single clean skeleton of the shared shape.

## Installation

Requires a Rust toolchain and `maturin`.

```bash
uv run maturin develop --release
```

Runtime dependency: `numpy` only.

## Usage

```python
import numpy as np
from topologize import topologize

curves = [
    np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]]),
    np.array([[0.1, 0.1], [9.9, 0.1], [5.0, 4.9]]),
]

chains = topologize(curves, buffer_distance=0.5)
# chains: list of (N, 2) numpy arrays
```

| Parameter | Type | Description |
|---|---|---|
| `curves` | `list[np.ndarray]` | Input polylines, each an `(N, 2)` array. Closed curves should repeat the first point at the end. |
| `buffer_distance` | `float` | Inflation radius. Use roughly half the typical gap between nearby strokes. |

### SVG input

```python
from topologize.helpers import load_svg

curves = load_svg("path/to/file.svg")
chains = topologize(curves, buffer_distance=10.0)
```

The SVG parser handles nested groups, transforms (`matrix`, `translate`, `scale`), and path commands `M L H V Q C Z` (absolute and relative). Beziers are discretized at a fixed sample distance.

## Examples

All examples are `# %%` cell-delimited Python files — run directly or open as Jupyter notebooks.

```bash
# Interactive SVG centerline plot (requires plotly: uv add --dev plotly)
uv run python/examples/svg_centerline.py python/examples/data/input.svg --buffer 20

# CLI summary (no visualisation)
uv run python scripts/topologize_svg.py python/examples/data/input.svg --buffer 20
```

## Benchmarks

```bash
uv run python benchmarks/run_all.py
```

Datasets:
- **simple**: four synthetic curves (two open polylines, a circle, a star) at buffer distances 0.3, 0.6, 1.2
- **svg**: paths extracted from a real SVG file at buffer distances 5, 10, 20, 50

## Project structure

```
src/                      Rust extension (PyO3 / maturin)
  lib.rs                  pymodule entry point (_internal)
  python.rs               Python-facing bindings
  inflate.rs              Clipper2-based polygon inflation
  skeleton.rs             CDT midpoint-graph skeletonizer
  graph.rs                Endpoint snapping + chain extraction

python/
  topologize/             Python package
    __init__.py           Public API
    helpers/
      svg_parser.py       SVG file parser
  examples/
    svg_centerline.py     SVG → centerline chains → Plotly visualisation
    prototype.py          Algorithm exploration notebook

scripts/
  topologize_svg.py       CLI: process an SVG, print summary

benchmarks/
  inputs.py               Shared test datasets
  bench_quality.py        Output complexity measurements
  bench_timing.py         CPU timing measurements
  run_all.py              Benchmark runner

Cargo.toml                Rust manifest
pyproject.toml            Python build config (maturin)
```

## Algorithm overview

See [algorithm.md](algorithm.md) for a detailed description of the three pipeline stages: inflate, skeletonize, and extract chains.
