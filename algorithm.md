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

**Input preprocessing** (before inflate): split subpaths that share an endpoint
(degree-2 only; real junctions are preserved) are **merged** into single
polylines, controlled by `merge_tolerance`. Contours authored as many separate
subpaths — common in CAD/SVG exports — otherwise get a square end cap at every
interior junction, which roughens the buffer boundary and fragments the skeleton.

**Boundary preprocessing** (after inflate, before CDT):

1. **RDP simplification** (ε = 0.05 × `feature_size`, `boundary_simplification`):
   the parallel-offset boundary inherits one point per input vertex on each side,
   easily 20k+ points. The CDT skeleton can't resolve features below
   `feature_size`, so near-collinear boundary points are pure triangulation
   overhead; RDP also strips the square-join micro-jank. This is a
   denoising/performance step only — connectivity is handled by the
   topology-aware pruning in Stage 3, so `boundary_simplification = 0` (a true
   pass-through) still yields a connected skeleton; it just leaves more vertices
   for the CDT.

2. **Densification** — one of two modes:
   - **Subdivision** (default, max edge = 1.5 × buffer): splits long edges only,
     re-densifying straight sections so CDT triangles stay compact.
   - **Curvature-adaptive resample** (when `resample` is set): redistributes each
     ring to a base arc-length spacing, tightening *smoothly* through curves.
     Spacing follows a constant chord-error (sagitta) law, `s ∝ √R`: it stays at
     the base spacing until the local radius of curvature drops below ~10 × buffer,
     then eases down — so gentle bends (radius 5–10 × buffer) pick up extra
     samples, not only sharp turns — bottoming out at `subdivision_ratio × buffer`.
     Curvature refinement is folded into the resample — every sample is taken *on*
     the offset boundary (never on a post-hoc chord), and density varies
     continuously. The radius is estimated cross-ring, so opposing channel walls
     get matched density (the convex side otherwise carries several times more
     vertices than the concave side).

The net effect: CDT input is reduced from ~29k to ~14k boundary points on the
benchmark input, cutting skeleton time from ~1 s to ~40 ms total.

![Curvature-adaptive resampling](https://raw.githubusercontent.com/tdamsma/topologize/main/docs/resample_comparison.png)

*Left: default subdivision over-samples the concave wall and fans triangles to
the convex side. Right: `resample` balances the density, with every vertex on
the offset boundary. Regenerate with `python/examples/resample_comparison.py`.*

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

2. **Edge classification**: for each CDT edge, decide whether it is a
   *cross-edge* (spans the polygon width — its midpoint lies on the medial
   axis) using purely topological criteria:
   - **Boundary edge**: only 1 adjacent triangle → discard
   - **Same-side edge**: both endpoints on the same boundary ring and within 2
     hops along it → discard (connects nearby vertices on one wall, not across)
   - **Cross-edge**: 2 adjacent triangles, not same-side → keep

   Edges are **not** filtered by length here. A length cut at this stage is not
   topology-aware: it severs legitimate interior cross-edges, collapsing T/X
   junctions and breaking the skeleton across tight bends (most visibly with
   little or no boundary simplification, where cross-edges sit right at the
   nominal `2 × buffer_distance` width). Short spurs are removed later, in Stage
   3, where the graph topology is known.

3. **Skeleton segment generation**: for each triangle, examine its cross-edges:
   - **2 cross-edges**: connect their midpoints (one segment)
   - **3 cross-edges**: connect each midpoint to the triangle centroid
     (three segments, forming a Y-junction)
   - **0 or 1 cross-edges**: skip

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

3. **Prune short tips**: iteratively remove spurs whose total arc length is
   below `min_tip_fraction × feature_size`. A "tip" is walked from a degree-1
   node *through* degree-2 nodes (contracting them into one chain) up to the
   first junction; only then, knowing the whole spur length, is it culled. This
   is the only edge-culling step — it acts solely on chains attached to a
   degree-1 node, so it can never sever an interior cross-edge. It removes the
   square-endcap "snake tongue" spurs that the (now-removed) length filter used
   to target, without the collateral damage of breaking junctions and bends.

4. **Merge close junctions**: contract short degree≥3-to-degree≥3 bridges (below
   `junction_merge_fraction × feature_size`) so a steep crossing rendered as two
   adjacent T-junctions becomes one X-junction. Runs after pruning so spurs
   don't leave behind spurious junctions.

5. **Chain traversal**:
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

**`min_tip_fraction`** (default 2.0 × `feature_size`): the spur-pruning length.
A degree-1 tip whose total arc length (walked through degree-2 nodes to the
first junction) is below this is removed — this is the sole edge-culling step
and acts only on chains attached to a degree-1 node, so it never severs interior
structure. Increasing it gives a smoother skeleton with fewer spurious branches;
decreasing it retains more detail at the cost of noise.
