from __future__ import annotations

import numpy as np


def topologize(
    curves: list[np.ndarray],
    buffer_distance: float,
    *,
    method: str = "midpoint",
    cos_angle: float = 0.0,
    simplification: float | None = None,
) -> list[np.ndarray]:
    """
    Clean and topologize line input via inflate-skeletonize.

    Inflates all input curves by `buffer_distance`, unions the resulting
    polygons, skeletonizes each polygon, and returns the medial axis as
    maximal non-branching polylines.

    Parameters
    ----------
    curves : list of (N, 2) numpy arrays
        Input polylines. Closed curves should repeat the first point at the end.
    buffer_distance : float
        Inflation radius. Use roughly half the typical gap between nearby strokes.
    method : "midpoint" (default) | "voronoi"
        Skeletonization algorithm.
        "midpoint" — constrained Delaunay triangulation, midpoint graph.
        "voronoi"  — Boost Voronoi diagram via the `centerline` crate.
    cos_angle : float, default 0.0
        Voronoi only. Cosine of the minimum acceptable angle between a Voronoi
        edge and the nearest input segment. 0.0 keeps all edges; values toward
        1.0 prune shallow-angle branches progressively.
    simplification : float or None, default None (= buffer_distance / 10)
        RDP (Ramer-Douglas-Peucker) tolerance applied to output polylines
        (in input units). For "midpoint": applied after projection smoothing.
        For "voronoi": applied internally by the skeletonizer.
        Larger values produce fewer output points; 0.0 disables.

    Returns
    -------
    list of (M, 2) numpy arrays, one per continuous non-branching segment.
    """
    from topologize._internal import topologize as _topologize

    kwargs = {}
    if simplification is not None:
        kwargs["simplification"] = float(simplification)

    raw = _topologize(
        [[tuple(float(v) for v in pt) for pt in curve] for curve in curves],
        buffer_distance,
        method,
        cos_angle,
        **kwargs,
    )
    return [np.array(chain) for chain in raw]
