# Algorithm

The pipeline has three stages: **inflate**, **skeletonize**, **extract chains**.

---

## Stage 1 — Inflate

Each input curve (open or closed polyline) is expanded into a polygon by
buffering it outward by `buffer_distance` on all sides. All resulting polygons
are then unioned into a single region, which may have holes if input curves
enclose empty space.

Uses the [Clipper2](https://github.com/ange-yaghi/clipper2) library
(`clipper2-rust` crate) with square join and round end cap.

**Boundary preprocessing** (after inflate, before CDT):

1. **RDP simplification** (ε = 0.15 × buffer): the parallel-offset boundary
   inherits one point per input vertex on each side, easily 20k+ points.
   The CDT skeleton can't resolve features below `buffer_distance`, so
   near-collinear boundary points are pure triangulation overhead. RDP reduces
   a typical boundary from ~27k to ~1k points.

2. **Subdivision** (max edge = 1.5 × buffer): after RDP, some boundary edges
   are very long (straight sections collapse to two endpoints). Long edges
   produce elongated CDT triangles whose midpoints don't land on the true
   centerline. Subdivision re-densifies to a maximum of 1.5 × buffer per edge,
   keeping triangles compact without re-introducing the excess from step 1.

The net effect: CDT input is reduced from ~29k to ~14k boundary points on the
benchmark input, cutting skeleton time from ~1 s to ~40 ms total.

---

## Stage 2 — Skeletonize

### Midpoint skeleton (production)

Classic "medial axis via CDT midpoints" approach, first described by Sherbrooke
et al. (1995) and widely used since.

**Steps:**

1. **Constrained Delaunay triangulation**: all boundary vertices (outer ring +
   holes) are inserted as points; all boundary edges are added as constraints.
   Uses the [`cdt`](https://crates.io/crates/cdt) crate (Formlabs sweep-line
   implementation).

2. **Edge classification**: for each CDT edge, count adjacent interior
   triangles (those returned by `triangulate_contours` are already interior-
   only — no centroid filter needed):
   - **Boundary edge**: only 1 adjacent triangle → discard
   - **Short edge**: length < 1.9 × `buffer_distance` → discard
   - **Internal edge**: 2 adjacent triangles, long enough → keep

3. **Skeleton segment generation**: for each triangle, examine its internal
   edges:
   - **2 internal edges**: connect their midpoints (one segment)
   - **3 internal edges**: connect each midpoint to the triangle centroid
     (three segments, forming a Y-junction)
   - **0 or 1 internal edges**: skip

**Post-processing** (applied to output chains):

1. **3× projection smoothing**: each interior point moves halfway toward its
   projection on the line through its two neighbours. This cancels the
   left-right staircase artifact from alternating CDT triangle orientations on
   near-straight corridor sections.

2. **Terminal endpoint straightening**: degree-1 (terminal) chain endpoints are
   projected halfway onto the extrapolated chain direction. At the tip of a
   pointed shape the last CDT midpoint is off-axis; this pulls it back.
   Junction endpoints are left fixed to preserve chain connectivity.

3. **RDP simplification** (ε = buffer / 10 by default): removes redundant
   points on near-straight runs after smoothing.

---

## Stage 3 — Extract chains

Raw skeleton edges are assembled into maximal non-branching polylines.

1. **Endpoint snapping**: nearby endpoints are merged with a grid-cell hash map
   at tolerance `snap_tol = buffer_distance / 20`. Closes small gaps from
   floating-point rounding across adjacent triangles.

2. **Graph construction**: undirected adjacency list; self-loops and duplicate
   edges discarded.

3. **Chain traversal**:
   - Start from all junction/terminal nodes (degree ≠ 2); walk along degree-2
     nodes until the next junction. Mark edges visited.
   - Second pass picks up any remaining unvisited edges (pure cycles).
   - Each chain records whether its endpoints are terminal (degree 1) or
     junction (degree > 2), used by the endpoint straightening step above.

---

## Design decisions

### Why CDT midpoints?

Three skeletonization approaches were prototyped and compared on real SVG input
at several buffer sizes:

| Method | Speed | Output quality | Tuneability |
|---|---|---|---|
| **CDT midpoints** (chosen) | fast | good, predictable | excellent |
| Voronoi diagram | slower | many short fragments | poor |
| Python prototype (shapely + triangle) | medium | good | impractical |

**Voronoi** (via `centerline` / `boostvoronoi` crates): produces a Voronoi
diagram of the boundary points. Theoretically the correct medial axis dual, but
in practice every minor boundary feature generates a Voronoi vertex that
becomes a short skeleton branch. The result is extremely fragmented — 10× more
chains than midpoint — and no simple filter cleans it up without also eating
real topology. Abandoned.

**Python prototype** (shapely + `triangle` library): performance was
surprisingly competitive (~600 ms vs ~40 ms Rust on the benchmark input), but
the gap widens with input size. More importantly, the post-processing needed to
produce clean output — smoothing, endpoint straightening, RDP, boundary
simplification — involves per-point conditional logic that is natural to write
as imperative Rust but painful in Python. Vectorising those operations with
numpy requires mental gymnastics (rolling windows, masked arrays, recursive
RDP) that obscure correctness and make tuning harder. The Rust version is easier
to reason about and faster to iterate on.

### Why the `cdt` crate over `spade`?

Both implement constrained Delaunay triangulation. On standard benchmarks
(coastline shapefiles with many small separate contours) the two are within 50%
of each other. On our input — a **single large merged polygon** from the union
of all inflated curves — spade degrades to O(n²):

| boundary pts | spade | cdt |
|---|---|---|
| 28 500 (buf=5) | 957 ms | — |
| 14 200 (buf=10) | 255 ms | 11 ms |
|  7 000 (buf=20) |  69 ms |  — |

Halving the point count quadruples spade's time, confirming O(n²). The root
cause: spade's incremental constraint insertion walks the triangulation to find
and flip crossing edges — O(n) per constraint — whereas `cdt` uses a
sweep-line algorithm with O(log n) per constraint.

The output of both implementations is identical. spade was removed entirely.

---

## Parameter guidance

**`buffer_distance`**: the single most important parameter. Too small → nearby
curves don't merge and the skeleton fragments. Too large → fine detail is lost
and distinct paths merge incorrectly. A good starting point is roughly half the
typical gap between nearby strokes.

**`simplification`**: RDP tolerance on the output chains (default buffer/10).
Increase for fewer points on straight runs; set to 0 to disable and inspect the
smoothed-but-unsimplified skeleton.

**Short-edge threshold** (internal, 1.9 × buffer): prunes skeleton branches
from minor boundary features. Increasing it gives a smoother skeleton with
fewer spurious branches; decreasing it retains more detail at the cost of noise.
