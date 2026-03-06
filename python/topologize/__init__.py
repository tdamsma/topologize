from __future__ import annotations

import numpy as np


def topologize(
    curves: list[np.ndarray],
    buffer_distance: float,
    *,
    simplification: float | None = None,
) -> list[np.ndarray]:
    """
    Clean and topologize line input via inflate-skeletonize.

    Inflates all input curves by `buffer_distance`, unions the resulting
    polygons, skeletonizes each polygon via constrained Delaunay triangulation
    midpoints, and returns the medial axis as maximal non-branching polylines.

    Parameters
    ----------
    curves : list of (N, 2) numpy arrays
        Input polylines. Closed curves should repeat the first point at the end.
    buffer_distance : float
        Inflation radius. Use roughly half the typical gap between nearby strokes.
    simplification : float or None, default None (= buffer_distance / 10)
        RDP (Ramer-Douglas-Peucker) tolerance applied to output polylines
        (in input units), applied after projection smoothing.
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
        **kwargs,
    )
    return [np.array(chain) for chain in raw]
