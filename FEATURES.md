# Planned features

## Feature 1: Variable buffer width per point

### User-facing change

Input curves change from `(N, 2)` to `(N, 3)` arrays, where the third column is the per-point inflate radius. Curves with shape `(N, 2)` continue to use the global `buffer_distance`. Mixed input is supported.

```python
# Uniform (existing):
curves = [np.array([[x, y], ...])]
result = topologize(curves, buffer_distance=0.5)

# Variable:
curves = [np.array([[x, y, width], ...])]
result = topologize(curves, buffer_distance=0.5)
# buffer_distance still controls snap_tol, tip pruning, etc.
```

### Implementation

`clipper2-rust 1.0.1` exposes `ClipperOffset::execute_with_callback(cb, &mut paths)` where `cb: Box<dyn Fn(&Path64, &PathD, curr_idx, prev_idx) -> f64>`. The `curr_idx` maps 1:1 to the input vertex index, so the callback simply returns `widths[curr_idx]`.

- **`src/inflate.rs`**: add `inflate_curve_variable(pts, widths, arc_tol, decimal_places)` using `ClipperOffset` + delta callback. Update `inflate_curves` to accept `Option<&[Vec<f64>]>` per-curve widths; branch to variable path when `Some`, keep existing `inflate_paths_d` path when `None`.
- **`src/python.rs`**: add `per_curve_widths: Option<Vec<Vec<f64>>>` parameter to the `topologize` pyfunction.
- **`python/topologize/__init__.py`**: split `(N, 3)` arrays into `(N, 2)` coords + `(N,)` width column before calling into Rust; pass widths separately.

`buffer_distance` remains required and controls all algorithmic thresholds (snap_tol, tip pruning, subdivision, junction merging). Per-point widths only override the inflate radius.

---

## Feature 2: Local width at each medial axis point

### User-facing change

`TopologizeResult` gains `chain_widths: list[np.ndarray]` — one `(M,)` float array per chain, giving the estimated contour width at each chain point.

```python
result = topologize(curves, buffer_distance=0.5)
result.chain_widths        # list of (M,) arrays, same length as result.chains
result.chain_widths[0]     # width at each point of the first chain
```

For uniform-width inputs the values should be ≈ 2 × `buffer_distance` everywhere (useful sanity check).

### Implementation

Post-processing step in `src/python.rs`, after all inflation is complete:

1. Collect every boundary vertex (outer rings + holes of all inflated polygons) into a flat list `boundary_pts`.
2. After final chain assembly (post-smoothing/simplification), for each chain point compute:
   ```
   width = 2 × min_distance(point, boundary_pts)
   ```
3. Return as a 4th element of the Rust return tuple; unpack in `__init__.py` and store as numpy arrays.

The nearest boundary vertex is one of the two nearest "wall" points, so 2× that distance is a good approximation of the local tube diameter. No extra geometry is needed — the boundary polygon is already in memory from the inflation step. Cost: O(S × B) brute-force nearest-neighbour, which is fine for typical input sizes.

---

## Feature 3: Input curve provenance per output chain

### User-facing change

Each input curve can be annotated with an ID. The result reports, for each output chain, which input curve IDs contributed to it.

```python
result = topologize(curves, buffer_distance=0.5, curve_ids=[1, 2, 3, ...])
result.chain_source_ids    # list[frozenset[int]], one set per output chain
```

If `curve_ids` is omitted, `chain_source_ids` is `None`.

### Definition of "contributed"

Input curve `i` contributed to output chain `c` if any point on `c` is within `buffer_distance` of at least one point on curve `i` — i.e. the chain passes through the inflated region of curve `i`. This is computed as a polyline-distance check (cheaper than point-in-polygon, and the semantic is intuitive).

### Implementation

- **`src/python.rs`**: accept `curve_ids: Option<Vec<i64>>`. After final chain extraction, for each chain point compute distance to each original input curve; if distance ≤ `buffer_distance` (or local per-point width), record that curve's ID as a contributor. Return `Vec<Vec<i64>>` (sorted) as a 5th return element.
- **`python/topologize/__init__.py`**: convert to `list[frozenset[int]]` and store on `TopologizeResult`.

### Open question

Should "contributed" be determined by:
- **A)** Point-in-polygon against the per-curve pre-union inflated polygons (geometrically exact)
- **B)** Polyline distance to the original raw input curve ≤ `buffer_distance` (cheaper, equally intuitive)

Option B is recommended.
