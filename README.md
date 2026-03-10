# topologize

**Convert messy, overlapping polylines into a clean topological skeleton.**

Given a set of curves — open or closed, possibly intersecting or bundled — `topologize` inflates them into a region, skeletonizes that region via constrained Delaunay triangulation, and returns a list of maximal non-branching polylines tracing the medial axis.

![Before and after on topologize.svg](https://raw.githubusercontent.com/tdamsma/topologize/main/docs/example_topologize.png)

## What it does

The three-stage pipeline:

1. **Inflate** — buffer all input curves by `inflation_radius` and union the results into one or more polygons (Clipper2)
2. **Skeletonize** — constrained Delaunay triangulation of the polygon interior; midpoints of internal edges form the skeleton graph
3. **Extract chains** — snap nearby endpoints, then traverse the graph to extract maximal non-branching polylines

The output chains share junction points, so the result is a proper topological graph: you can traverse it, measure it, and match it to other representations.

## When to use it

Use `topologize` when you have *geometry that approximates a graph* and need *an actual graph* — a set of polylines with shared junction points you can traverse, measure, or match to other data.

Concrete examples:

- Vector artwork or scanned drawings converted to machine toolpaths (laser, pen plotter, CNC, printing)
- Road, river, or network centerline extraction from polygon or buffered line data
- GPS or sensor traces where the same route was recorded multiple times

## Installation

```bash
pip install topologize
```

Runtime dependency: `numpy` only.

To build from source (requires a Rust toolchain):

```bash
git clone https://github.com/tdamsma/topologize
cd topologize
uv run maturin develop --release
```

## Quick start

```python
import numpy as np
from topologize import topologize

# Any collection of (N, 2) numpy arrays — open or closed, overlapping is fine
curves = [
    np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]]),
    np.array([[0.1, 0.1], [9.9, 0.1], [5.1, 4.9]]),  # near-duplicate stroke
]

result = topologize(curves, inflation_radius=0.5)
result.chains          # list of (M, 2) arrays — one per non-branching segment
result.nodes           # (K, 2) array of unique junction/endpoint positions
result.chain_node_ids  # list of (start_id, end_id) per chain
```

`inflation_radius` is the main tuning parameter. Use roughly **half the typical gap between nearby strokes** — small enough to keep distinct paths separate, large enough to merge strokes that belong together.

For variable-width inflation, pass a list of per-vertex radius arrays:

```python
widths_a = np.linspace(0.3, 0.1, len(curve_a))
widths_b = np.linspace(0.1, 0.5, len(curve_b))
result = topologize([curve_a, curve_b], inflation_radius=[widths_a, widths_b])
```

| Parameter | Type | Description |
|---|---|---|
| `curves` | `list[np.ndarray]` | Input polylines, each `(N, 2)` or `(N, 3)` with per-vertex widths. Closed curves should repeat the first point at the end. |
| `inflation_radius` | `float \| list` | Inflation radius. A single float for uniform width, or a list of per-vertex radius arrays for variable width. |
| `feature_size` | `float \| None` | Scale parameter for all derived thresholds. Defaults to `inflation_radius` (float) or `median(all widths)` (list). |
| `simplification` | `float \| None` | RDP tolerance on output chains (default: `feature_size / 10`). Set to `0` to disable. |
| `min_tip_length` | `float \| None` | Prune terminal chains shorter than this (default: `feature_size * 2`). Set to `0` to disable. |
| `junction_merge_fraction` | `float \| None` | Merge nearby junctions within `fraction × feature_size` (default: `1.5`). Set to `0` to disable. |
| `compute_widths` | `bool` | If `True`, populate `result.chain_widths` with estimated contour width at each chain point. |

### Visualization

`TopologizeResult` has a built-in `plot()` method (requires `plotly`):

```python
result.plot(curves, inflation_radius=0.5)                    # basic
result.plot(curves, inflation_radius=0.5, show_triangulation=True)  # + CDT overlay
```

When `compute_widths=True`, the plot also shows bead-width envelopes around each chain.

### Batch processing

For workloads with many independent curve-sets (e.g. per-layer toolpath slicing), `topologize_batch` processes them in parallel via Rayon with the GIL released. Each job carries its own parameters:

```python
from topologize import topologize_batch, TopologizeJob

jobs = [
    TopologizeJob(curves_layer_1, inflation_radius=0.5),
    TopologizeJob(curves_layer_2, inflation_radius=1.0, simplification=0.0),
    # ...
]
results = topologize_batch(jobs)
# returns list[TopologizeResult], one per job
```

On multi-core machines this is significantly faster than a Python loop.

### Async-style processing with ThreadPoolExecutor

Single `topologize` releases the GIL during Rust computation, so you can also use Python's `ThreadPoolExecutor` for async-style processing:

```python
from concurrent.futures import ThreadPoolExecutor
from topologize import topologize

with ThreadPoolExecutor() as pool:
    futures = [pool.submit(topologize, cs, inflation_radius=0.5) for cs in curve_sets]
    results = [f.result() for f in futures]
```

This is useful when jobs arrive one at a time (e.g. from a queue) rather than all at once.

## Examples

All examples are `# %%` cell-delimited Python files — run directly or open as Jupyter notebooks.

```bash
uv sync --group dev  # install plotly, svgpathtools, etc.

# Getting started — basic usage and inflation_radius tuning
uv run python/examples/getting_started.py

# SVG centerline extraction
uv run python/examples/svg_centerline.py python/examples/data/topologize.svg --buffer 0.47

# Variable-width per-vertex inflation
uv run python/examples/variable_width_demo.py

# Parallel / batch processing benchmarks
uv run python/examples/parallel_processing.py
```

## Algorithm

See [algorithm.md](algorithm.md) for a detailed description of all three pipeline stages, the boundary preprocessing steps (RDP simplification + subdivision), the post-processing applied to output chains (projection smoothing, endpoint straightening, RDP), and the rationale for the CDT midpoint approach over alternatives (Voronoi, Python prototype).

## Project structure

```
src/
  lib.rs             pymodule entry point (_internal)
  python.rs          Python-facing bindings
  inflate.rs         Clipper2-based polygon inflation + boundary prep
  skeleton_cdt.rs    CDT midpoint-graph skeletonizer
  graph.rs           Endpoint snapping + chain extraction

python/
  topologize/
    __init__.py      Public API (TopologizeResult, topologize, topologize_batch, inflate, triangulate)
  examples/
    getting_started.py
    svg_centerline.py
    variable_width_demo.py
    compare_methods.py
    parallel_processing.py

tests/
  test_topologize.py
  test_batch.py

Cargo.toml           Rust manifest
pyproject.toml       Python build config (maturin)
```
