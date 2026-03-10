from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TopologizeJob:
    """Input specification for a single :func:`topologize_batch` job.

    Attributes
    ----------
    curves : list of (N, 2) numpy arrays
        Input polylines for this job.
    buffer_distance : float
        Inflation radius.
    simplification : float or None
    min_tip_length : float or None
    junction_merge_fraction : float or None
    """

    curves: list[np.ndarray]
    buffer_distance: float
    simplification: float | None = None
    min_tip_length: float | None = None
    junction_merge_fraction: float | None = None


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
    chain_widths : list of (M,) numpy arrays or None
        Estimated contour width at each chain point (2 × distance to nearest
        inflated polygon boundary vertex). Only populated when
        ``compute_widths=True`` is passed to :func:`topologize`; otherwise None.
    """

    chains: list[np.ndarray]
    nodes: np.ndarray
    chain_node_ids: list[tuple[int, int]]
    chain_widths: list[np.ndarray] | None = None

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

    def plot(
        self,
        curves: list[np.ndarray] | None = None,
        buffer_distance: float | None = None,
        *,
        per_curve_widths: list | None = None,
        show_triangulation: bool = False,
        title: str = "",
    ):
        """Interactive plotly visualization of the topologize result.

        Parameters
        ----------
        curves : list of (N, 2) arrays, optional
            Original input curves (shown as gray lines).
        buffer_distance : float, optional
            Buffer distance used; needed to show the inflated boundary.
        per_curve_widths : list, optional
            Per-vertex widths (passed to :func:`inflate` for the boundary layer).
        show_triangulation : bool, default False
            If True (requires *curves* and *buffer_distance*), overlay the CDT
            triangulation used internally by the skeleton step.
        title : str
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for plot(). Install it with: pip install plotly"
            )

        COLORS = [
            "#e41a1c", "#377eb8", "#4daf4a", "#ff7f00",
            "#984ea3", "#a65628", "#f781bf", "#333333",
        ]

        fig = go.Figure()

        # --- Input curves ---
        if curves is not None:
            for i, c in enumerate(c_arr := [np.asarray(c) for c in curves]):
                fig.add_trace(go.Scatter(
                    x=c[:, 0], y=c[:, 1], mode="lines",
                    line=dict(color="gray", width=1),
                    legendgroup="input",
                    showlegend=(i == 0), name="Input curves",
                    hoverinfo="skip",
                ))

        # --- Input buffer boundary ---
        if curves is not None and (buffer_distance is not None or per_curve_widths is not None):
            polys = inflate(curves, buffer_distance, per_curve_widths=per_curve_widths)
            first_buf = True
            for outer, holes in polys:
                for ring in [outer] + holes:
                    fig.add_trace(go.Scatter(
                        x=np.append(ring[:, 0], ring[0, 0]),
                        y=np.append(ring[:, 1], ring[0, 1]),
                        mode="lines",
                        line=dict(color="gray", width=1, dash="dot"),
                        legendgroup="buffer",
                        showlegend=first_buf, name="Buffer boundary",
                        hoverinfo="skip",
                    ))
                    first_buf = False

        # --- CDT triangulation ---
        if show_triangulation and curves is not None and (buffer_distance is not None or per_curve_widths is not None):
            tris = triangulate(curves, buffer_distance, per_curve_widths=per_curve_widths)
            xs, ys = [], []
            for (x0, y0), (x1, y1), (x2, y2) in tris:
                xs.extend([x0, x1, x2, x0, None])
                ys.extend([y0, y1, y2, y0, None])
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="lightblue", width=0.5),
                legendgroup="cdt",
                showlegend=True, name="CDT triangles",
                hoverinfo="skip",
            ))

        # --- Bead width envelopes ---
        if self.chain_widths is not None:
            first_bead = True
            for i, (chain, widths) in enumerate(zip(self.chains, self.chain_widths)):
                color = COLORS[i % len(COLORS)]
                # Compute perpendicular offsets
                d = np.diff(chain, axis=0)
                d = np.vstack([d, d[-1:]])  # repeat last direction
                lengths = np.linalg.norm(d, axis=1, keepdims=True)
                lengths = np.where(lengths == 0, 1, lengths)
                d = d / lengths
                perp = np.column_stack([-d[:, 1], d[:, 0]])
                half_w = (widths / 2)[:, np.newaxis]
                left = chain + perp * half_w
                right = chain - perp * half_w
                # Closed polygon: left forward, right reversed
                poly_x = np.concatenate([left[:, 0], right[::-1, 0], left[:1, 0]])
                poly_y = np.concatenate([left[:, 1], right[::-1, 1], left[:1, 1]])
                fig.add_trace(go.Scatter(
                    x=poly_x, y=poly_y, mode="lines",
                    fill="toself", fillcolor=color, opacity=0.15,
                    line=dict(color=color, width=0.5),
                    legendgroup="widths",
                    showlegend=first_bead, name="Bead width",
                    hoverinfo="skip",
                ))
                first_bead = False

        # --- Centerline chains ---
        for i, chain in enumerate(self.chains):
            color = COLORS[i % len(COLORS)]
            fig.add_trace(go.Scatter(
                x=chain[:, 0], y=chain[:, 1], mode="lines",
                line=dict(color=color, width=2.5),
                legendgroup="chains",
                showlegend=(i == 0), name="Chains",
                text=[f"chain {i}"] * len(chain), hoverinfo="text",
            ))

        # --- Nodes ---
        degree = self.node_degree
        fig.add_trace(go.Scatter(
            x=self.nodes[:, 0], y=self.nodes[:, 1], mode="markers",
            marker=dict(size=7, color="black"),
            name="Nodes",
            text=[f"node {j}, degree {degree[j]}" for j in range(len(self.nodes))],
            hoverinfo="text",
        ))

        fig.update_layout(
            title=title, yaxis_scaleanchor="x",
            template="plotly_white", width=900, height=650,
        )
        return fig


def triangulate(
    curves: list[np.ndarray],
    buffer_distance: float | None = None,
    *,
    per_curve_widths: list[list[float]] | None = None,
) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    """
    Return the CDT triangles used internally by :func:`topologize`.

    Applies the same boundary preprocessing (RDP + subdivision) as the main
    pipeline. Useful for visualising and debugging the triangulation step.

    Parameters
    ----------
    curves : list of (N, 2) or (N, 3) numpy arrays
    buffer_distance : float or None
        Derived from the median of *per_curve_widths* when ``None``.
    per_curve_widths : list of list[float] or None
        Per-vertex radii (same semantics as :func:`topologize`).

    Returns
    -------
    list of ((x0,y0),(x1,y1),(x2,y2)) tuples — one per triangle.
    """
    from topologize._internal import triangulate_curves as _tri

    curves_xy, widths_to_pass = _extract_widths(curves, per_curve_widths)
    bd = _resolve_buffer_distance(buffer_distance, widths_to_pass)

    kwargs = {}
    if widths_to_pass is not None:
        kwargs["per_curve_widths"] = [[float(v) for v in w] for w in widths_to_pass]

    return _tri(_convert_curves(curves_xy), bd, **kwargs)


def inflate(
    curves: list[np.ndarray],
    buffer_distance: float | None = None,
    *,
    per_curve_widths: list[list[float]] | None = None,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """
    Inflate polylines and return the buffer polygons.

    Parameters
    ----------
    curves : list of (N, 2) or (N, 3) numpy arrays
        If an array has shape (N, 3), column 2 is used as per-vertex buffer radius.
    buffer_distance : float or None
        Derived from the median of *per_curve_widths* when ``None``.
    per_curve_widths : list of list[float] or None
        Explicit per-vertex radii; takes precedence over (N, 3) column.

    Returns
    -------
    list of (outer, holes) where:
        outer : (N, 2) array — outer ring
        holes : list of (M, 2) arrays — hole rings (may be empty)
    """
    from topologize._internal import inflate_curves as _inflate

    curves_xy, widths_to_pass = _extract_widths(curves, per_curve_widths)
    bd = _resolve_buffer_distance(buffer_distance, widths_to_pass)

    kwargs = {}
    if widths_to_pass is not None:
        kwargs["per_curve_widths"] = [[float(v) for v in w] for w in widths_to_pass]

    raw = _inflate(_convert_curves(curves_xy), bd, **kwargs)
    return [(np.array(outer), [np.array(h) for h in holes]) for outer, holes in raw]


def topologize(
    curves: list[np.ndarray],
    buffer_distance: float | None = None,
    *,
    simplification: float | None = None,
    min_tip_length: float | None = None,
    junction_merge_fraction: float | None = None,
    per_curve_widths: list[np.ndarray] | None = None,
    compute_widths: bool = False,
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
    buffer_distance : float or None
        Inflation radius. Use roughly half the typical gap between nearby strokes.
        When ``None`` and *per_curve_widths* is provided, derived automatically
        from the median of all per-vertex widths.
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
    compute_widths : bool, default False
        If True, populate ``chain_widths`` with the estimated contour width at
        each chain point (2 × distance to the nearest inflated polygon boundary
        vertex). Disabled by default to avoid the O(S × B) scan overhead.

    Returns
    -------
    TopologizeResult
        ``.chains``        — list of (M, 2) arrays, one per non-branching segment
        ``.nodes``         — (K, 2) array of unique chain-endpoint positions
        ``.chain_node_ids``— list of (start_id, end_id) per chain
        ``.chain_widths``  — list of (M,) arrays when ``compute_widths=True``,
                             else None
    """
    from topologize._internal import topologize as _topologize

    curves_xy, widths_to_pass = _extract_widths(curves, per_curve_widths)
    bd = _resolve_buffer_distance(buffer_distance, widths_to_pass)

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
    if compute_widths:
        kwargs["compute_widths"] = True

    raw = _topologize(_convert_curves(curves_xy), bd, **kwargs)
    return _unpack_result_with_widths(*raw, compute_widths=compute_widths)


def _convert_curves(curves: list[np.ndarray]) -> list[list[tuple[float, float]]]:
    """Convert numpy arrays to list-of-tuples for Rust."""
    return [[tuple(float(v) for v in pt) for pt in curve] for curve in curves]


def _extract_widths(
    curves: list[np.ndarray],
    per_curve_widths: list | None,
) -> tuple[list[np.ndarray], list | None]:
    """Split (N,3) arrays into xy + widths; validate explicit per_curve_widths.

    Returns (curves_xy, widths_to_pass) where widths_to_pass is None when no
    per-vertex widths are needed.
    """
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

    has_auto = any(len(w) > 0 for w in auto_widths)
    widths_to_pass = per_curve_widths if per_curve_widths is not None else (auto_widths if has_auto else None)

    if per_curve_widths is not None:
        if len(per_curve_widths) > len(curves_xy):
            raise ValueError(
                f"per_curve_widths has {len(per_curve_widths)} entries but only "
                f"{len(curves_xy)} curves were provided; lengths must match"
            )
        for i, (w, c) in enumerate(zip(per_curve_widths, curves_xy)):
            w_len = len(w)
            if w_len > 0 and w_len != len(c):
                raise ValueError(
                    f"per_curve_widths[{i}] has {w_len} entries but curve {i} "
                    f"has {len(c)} points; lengths must match (or pass [] to use "
                    "buffer_distance for that curve)"
                )

    return curves_xy, widths_to_pass


def _resolve_buffer_distance(
    buffer_distance: float | None,
    widths_to_pass: list | None,
) -> float:
    """Return an effective buffer_distance.

    When *buffer_distance* is ``None``, derive it from the median of all
    per-vertex widths. Raises if no width information is available.
    """
    if buffer_distance is not None:
        return float(buffer_distance)
    if widths_to_pass is None:
        raise ValueError(
            "buffer_distance is required when per_curve_widths is not provided"
        )
    all_w = [v for w in widths_to_pass for v in w]
    if not all_w:
        raise ValueError(
            "buffer_distance is required when all per_curve_widths entries are empty"
        )
    return float(np.median(all_w))


def _unpack_result(raw_chains, raw_nodes, raw_chain_node_ids) -> TopologizeResult:
    """Unpack a raw Rust result tuple into TopologizeResult."""
    chains = [np.array(chain) for chain in raw_chains]
    nodes = np.array(raw_nodes).reshape(-1, 2)
    chain_node_ids = [tuple(pair) for pair in raw_chain_node_ids]
    return TopologizeResult(chains=chains, nodes=nodes, chain_node_ids=chain_node_ids)


def _unpack_result_with_widths(raw_chains, raw_nodes, raw_chain_node_ids, raw_chain_widths, *, compute_widths=False) -> TopologizeResult:
    """Unpack a raw Rust result tuple (4-element, with chain_widths) into TopologizeResult."""
    chains = [np.array(chain) for chain in raw_chains]
    nodes = np.array(raw_nodes).reshape(-1, 2)
    chain_node_ids = [tuple(pair) for pair in raw_chain_node_ids]
    return TopologizeResult(
        chains=chains,
        nodes=nodes,
        chain_node_ids=chain_node_ids,
        chain_widths=[np.array(w) for w in raw_chain_widths] if compute_widths else None,
    )


def topologize_batch(
    jobs: list[TopologizeJob],
) -> list[TopologizeResult]:
    """
    Process multiple independent curve-sets in parallel.

    Each :class:`TopologizeJob` bundles its own curves and parameters.
    Processing runs in parallel via Rayon with the GIL released.

    Parameters
    ----------
    jobs : list of TopologizeJob
        One job per independent topologize invocation.

    Returns
    -------
    list of TopologizeResult — one per input job, in the same order.
    """
    from topologize._internal import topologize_batch as _batch

    packed = [
        (
            _convert_curves(job.curves),
            job.buffer_distance,
            job.simplification,
            job.min_tip_length,
            job.junction_merge_fraction,
        )
        for job in jobs
    ]
    raw_results = _batch(packed)
    return [_unpack_result(*r) for r in raw_results]
