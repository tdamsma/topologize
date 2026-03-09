from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TopologizeResult:
    """Result of :func:`topologize`.

    Attributes
    ----------
    chains : list of (M, 2) numpy arrays
        Maximal non-branching centerline segments.
    nodes : (K, 2) numpy array
        Unique chain-endpoint positions.
    chain_node_ids : list of (start_id, end_id) tuples
        Indices into ``nodes`` for each chain's endpoints.
    """

    chains: list[np.ndarray]
    nodes: np.ndarray
    chain_node_ids: list[tuple[int, int]]

    @property
    def node_degree(self) -> np.ndarray:
        """(K,) integer array — number of chains incident on each node."""
        degree = np.zeros(len(self.nodes), dtype=int)
        for s, e in self.chain_node_ids:
            degree[s] += 1
            degree[e] += 1
        return degree

    def chains_at_node(self, node_id: int) -> list[int]:
        """Return indices of chains whose start or end node is *node_id*."""
        return [i for i, (s, e) in enumerate(self.chain_node_ids) if s == node_id or e == node_id]


def triangulate(
    curves: list[np.ndarray],
    buffer_distance: float,
) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    """
    Return the CDT triangles used internally by :func:`topologize`.

    Applies the same boundary preprocessing (RDP + subdivision) as the main
    pipeline. Useful for visualising and debugging the triangulation step.

    Parameters
    ----------
    curves : list of (N, 2) numpy arrays
    buffer_distance : float

    Returns
    -------
    list of ((x0,y0),(x1,y1),(x2,y2)) tuples — one per triangle.
    """
    from topologize._internal import triangulate_curves as _tri
    return _tri(
        [[tuple(float(v) for v in pt) for pt in curve] for curve in curves],
        buffer_distance,
    )


def inflate(
    curves: list[np.ndarray],
    buffer_distance: float,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """
    Inflate polylines and return the buffer polygons.

    Parameters
    ----------
    curves : list of (N, 2) numpy arrays
    buffer_distance : float

    Returns
    -------
    list of (outer, holes) where:
        outer : (N, 2) array — outer ring
        holes : list of (M, 2) arrays — hole rings (may be empty)
    """
    from topologize._internal import inflate_curves as _inflate

    raw = _inflate(
        [[tuple(float(v) for v in pt) for pt in curve] for curve in curves],
        buffer_distance,
    )
    return [(np.array(outer), [np.array(h) for h in holes]) for outer, holes in raw]


def topologize(
    curves: list[np.ndarray],
    buffer_distance: float,
    *,
    simplification: float | None = None,
    min_tip_length: float | None = None,
    junction_merge_fraction: float | None = None,
    per_curve_widths: list[np.ndarray] | None = None,
) -> TopologizeResult:
    """
    Clean and topologize line input via inflate-skeletonize.

    Inflates all input curves by `buffer_distance`, unions the resulting
    polygons, skeletonizes each polygon via constrained Delaunay triangulation
    midpoints, and returns the medial axis as maximal non-branching polylines.

    Parameters
    ----------
    curves : list of (N, 2) or (N, 3) numpy arrays
        Input polylines. Closed curves should repeat the first point at the end.
        If an array has shape (N, 3), the third column is interpreted as a
        per-vertex buffer radius, overriding ``buffer_distance`` for that curve.
    buffer_distance : float
        Inflation radius. Use roughly half the typical gap between nearby strokes.
        Used as the default radius for curves without per-vertex widths.
    simplification : float or None, default None (= buffer_distance / 10)
        RDP (Ramer-Douglas-Peucker) tolerance applied to output polylines
        (in input units), applied after projection smoothing.
        Larger values produce fewer output points; 0.0 disables.
    min_tip_length : float or None, default None (= buffer_distance * 2)
        Terminal chains shorter than this are pruned before chain extraction.
        Set to 0.0 to disable pruning.
    junction_merge_fraction : float or None, default None (= 1.5)
        Contract short edges between junction nodes (degree ≥ 3) at crossings.
        Threshold = fraction × buffer_distance. Merges 70–90° crossings with
        the default; set to 0.0 to preserve two separate T-junctions.
    per_curve_widths : list of array-like or None, default None
        Explicit per-vertex radii for each curve, as a list with one entry per
        curve. Each entry is a 1-D array of radii (one per vertex). Takes
        precedence over widths embedded in (N, 3) curve arrays. Pass an empty
        list entry (or ``[]``) for a curve that should use ``buffer_distance``.

    Returns
    -------
    TopologizeResult
        ``.chains``        — list of (M, 2) arrays, one per non-branching segment
        ``.nodes``         — (K, 2) array of unique chain-endpoint positions
        ``.chain_node_ids``— list of (start_id, end_id) per chain
    """
    from topologize._internal import topologize as _topologize

    # Pre-process curves: split (N, 3) arrays into xy + width columns.
    curves_xy = []
    auto_widths = []
    for c in curves:
        arr = np.asarray(c, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            curves_xy.append(arr[:, :2])
            auto_widths.append(arr[:, 2].tolist())
        else:
            curves_xy.append(arr)
            auto_widths.append([])

    # Use explicit per_curve_widths if provided, else auto_widths (if any curve had 3 cols).
    has_auto = any(len(w) > 0 for w in auto_widths)
    widths_to_pass = per_curve_widths if per_curve_widths is not None else (auto_widths if has_auto else None)

    kwargs = {}
    if simplification is not None:
        kwargs["simplification"] = float(simplification)
    if min_tip_length is not None:
        kwargs["min_tip_length"] = float(min_tip_length)
    if junction_merge_fraction is not None:
        kwargs["junction_merge_fraction"] = float(junction_merge_fraction)
    if widths_to_pass is not None:
        kwargs["per_curve_widths"] = [
            [float(v) for v in w] for w in widths_to_pass
        ]

    raw_chains, raw_nodes, raw_chain_node_ids = _topologize(
        [[tuple(float(v) for v in pt) for pt in curve] for curve in curves_xy],
        buffer_distance,
        **kwargs,
    )

    chains = [np.array(chain) for chain in raw_chains]
    nodes = np.array(raw_nodes).reshape(-1, 2)
    chain_node_ids = [tuple(pair) for pair in raw_chain_node_ids]

    return TopologizeResult(chains=chains, nodes=nodes, chain_node_ids=chain_node_ids)
