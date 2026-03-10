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
    inflation_radius : float or list of array-like
        Uniform radius (float) or per-vertex radii (list of arrays).
    feature_size : float or None
        Scale parameter for derived thresholds. Defaults to
        ``inflation_radius`` (float) or ``median(all widths)`` (list).
    simplification : float or None
    min_tip_length : float or None
    junction_merge_fraction : float or None
    """

    curves: list[np.ndarray]
    inflation_radius: float | list[np.ndarray]
    feature_size: float | None = None
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
        inflation_radius: float | list | None = None,
        *,
        feature_size: float | None = None,
        show_triangulation: bool = False,
        title: str = "",
    ):
        """Interactive plotly visualization of the topologize result.

        Parameters
        ----------
        curves : list of (N, 2) arrays, optional
            Original input curves (shown as gray lines).
        inflation_radius : float or list, optional
            Inflation radius used; needed to show the inflated boundary.
            Float for uniform, list of arrays for per-vertex widths.
        feature_size : float, optional
            Override the feature size used for inflate boundary display.
        show_triangulation : bool, default False
            If True (requires *curves* and *inflation_radius*), overlay the CDT
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
            for i, c in enumerate([np.asarray(c) for c in curves]):
                fig.add_trace(go.Scatter(
                    x=c[:, 0], y=c[:, 1], mode="lines",
                    line=dict(color="gray", width=1),
                    legendgroup="input",
                    showlegend=(i == 0), name="Input curves",
                    hoverinfo="skip",
                ))

        # --- Input buffer boundary ---
        if curves is not None and inflation_radius is not None:
            polys = inflate(curves, inflation_radius, feature_size=feature_size)
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
        if show_triangulation and curves is not None and inflation_radius is not None:
            tris = triangulate(curves, inflation_radius, feature_size=feature_size)
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


def _resolve_inflation_radius(
    inflation_radius: float | list,
    curves: list[np.ndarray],
) -> tuple[list[np.ndarray], float, list[list[float]] | None]:
    """Resolve inflation_radius into (curves_xy, buffer_distance, per_curve_widths).

    Also handles (N,3) curve arrays by splitting off the width column.

    Returns
    -------
    curves_xy : list of (N,2) arrays
    buffer_distance : float — the uniform radius for inflate_paths_d calls
    per_curve_widths : list of list[float] or None
    """
    if isinstance(inflation_radius, (int, float)):
        # Uniform mode: also handle (N,3) arrays for backward compat
        curves_xy, widths = _extract_widths(curves, None)
        return curves_xy, float(inflation_radius), widths
    elif isinstance(inflation_radius, list):
        # Per-vertex mode
        curves_xy, _ = _extract_widths(curves, None)
        if len(inflation_radius) != len(curves_xy):
            raise ValueError(
                f"inflation_radius has {len(inflation_radius)} entries but "
                f"{len(curves_xy)} curves were provided; lengths must match"
            )
        pcw: list[list[float]] = []
        for i, w in enumerate(inflation_radius):
            w_arr = np.asarray(w, dtype=float).ravel()
            if len(w_arr) > 0 and i < len(curves_xy) and len(w_arr) != len(curves_xy[i]):
                raise ValueError(
                    f"inflation_radius[{i}] has {len(w_arr)} entries but curve {i} "
                    f"has {len(curves_xy[i])} points; lengths must match"
                )
            pcw.append(w_arr.tolist())
        # Derive a uniform buffer_distance from median of all widths
        all_w = [v for w in pcw for v in w]
        if not all_w:
            raise ValueError("inflation_radius list has no width values")
        bd = float(np.median(all_w))
        return curves_xy, bd, pcw
    else:
        raise TypeError(
            f"inflation_radius must be a float or list of arrays, got {type(inflation_radius).__name__}"
        )


def _resolve_feature_size(
    feature_size: float | None,
    inflation_radius: float | list,
    per_curve_widths: list[list[float]] | None,
) -> float:
    """Compute the effective feature_size."""
    if feature_size is not None:
        return float(feature_size)
    if isinstance(inflation_radius, (int, float)):
        return float(inflation_radius)
    # Per-vertex mode: median of all widths
    if per_curve_widths is not None:
        all_w = [v for w in per_curve_widths for v in w]
        if all_w:
            return float(np.median(all_w))
    raise ValueError("feature_size is required when inflation_radius contains no width values")


def triangulate(
    curves: list[np.ndarray],
    inflation_radius: float | list[np.ndarray],
    *,
    feature_size: float | None = None,
) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    """
    Return the CDT triangles used internally by :func:`topologize`.

    Applies the same boundary preprocessing (RDP + subdivision) as the main
    pipeline. Useful for visualising and debugging the triangulation step.

    Parameters
    ----------
    curves : list of (N, 2) or (N, 3) numpy arrays
    inflation_radius : float or list of array-like
        Uniform radius (float) or per-vertex radii (list of arrays).
    feature_size : float, optional
        Scale parameter for derived thresholds. Defaults to
        ``inflation_radius`` (float) or ``median(all widths)`` (list).

    Returns
    -------
    list of ((x0,y0),(x1,y1),(x2,y2)) tuples — one per triangle.
    """
    from topologize._internal import triangulate_curves as _tri

    curves_xy, bd, pcw = _resolve_inflation_radius(inflation_radius, curves)
    fs = _resolve_feature_size(feature_size, inflation_radius, pcw)

    kwargs = {}
    if pcw is not None:
        kwargs["per_curve_widths"] = [[float(v) for v in w] for w in pcw]

    return _tri(_convert_curves(curves_xy), bd, fs, **kwargs)


def inflate(
    curves: list[np.ndarray],
    inflation_radius: float | list[np.ndarray],
    *,
    feature_size: float | None = None,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """
    Inflate polylines and return the buffer polygons.

    Parameters
    ----------
    curves : list of (N, 2) or (N, 3) numpy arrays
        If an array has shape (N, 3), column 2 is used as per-vertex buffer radius.
    inflation_radius : float or list of array-like
        Uniform radius (float) or per-vertex radii (list of arrays).
    feature_size : float, optional
        Scale parameter for derived thresholds. Defaults to
        ``inflation_radius`` (float) or ``median(all widths)`` (list).

    Returns
    -------
    list of (outer, holes) where:
        outer : (N, 2) array — outer ring
        holes : list of (M, 2) arrays — hole rings (may be empty)
    """
    from topologize._internal import inflate_curves as _inflate

    curves_xy, bd, pcw = _resolve_inflation_radius(inflation_radius, curves)
    fs = _resolve_feature_size(feature_size, inflation_radius, pcw)

    kwargs = {}
    if pcw is not None:
        kwargs["per_curve_widths"] = [[float(v) for v in w] for w in pcw]

    raw = _inflate(_convert_curves(curves_xy), bd, fs, **kwargs)
    return [(np.array(outer), [np.array(h) for h in holes]) for outer, holes in raw]


def topologize(
    curves: list[np.ndarray],
    inflation_radius: float | list[np.ndarray],
    *,
    feature_size: float | None = None,
    simplification: float | None = None,
    min_tip_length: float | None = None,
    junction_merge_fraction: float | None = None,
    compute_widths: bool = False,
) -> TopologizeResult:
    """
    Clean and topologize line input via inflate-skeletonize.

    Inflates all input curves by ``inflation_radius``, unions the resulting
    polygons, skeletonizes each polygon via constrained Delaunay triangulation
    midpoints, and returns the medial axis as maximal non-branching polylines.

    Parameters
    ----------
    curves : list of (N, 2) or (N, 3) numpy arrays
        Input polylines. Closed curves should repeat the first point at the end.
        If an array has shape (N, 3), the third column is interpreted as a
        per-vertex buffer radius.
    inflation_radius : float or list of array-like
        Inflation radius. Either a single float for uniform width, or a list of
        per-vertex radius arrays (one per curve) for variable width.
    feature_size : float or None, default None
        Scale parameter for all derived thresholds (simplification, tip pruning,
        junction merging, etc.). Defaults to ``inflation_radius`` (when float) or
        ``median(all widths)`` (when list).
    simplification : float or None, default None (= feature_size / 10)
        RDP (Ramer-Douglas-Peucker) tolerance applied to output polylines
        (in input units), applied after projection smoothing.
        Larger values produce fewer output points; 0.0 disables.
    min_tip_length : float or None, default None (= feature_size * 2)
        Terminal chains shorter than this are pruned before chain extraction.
        Set to 0.0 to disable pruning.
    junction_merge_fraction : float or None, default None (= 1.5)
        Contract short edges between junction nodes (degree >= 3) at crossings.
        Threshold = fraction x feature_size. Merges 70-90 deg crossings with
        the default; set to 0.0 to preserve two separate T-junctions.
    compute_widths : bool, default False
        If True, populate ``chain_widths`` with the estimated contour width at
        each chain point (2 x distance to the nearest inflated polygon boundary
        vertex). Disabled by default to avoid the O(S x B) scan overhead.

    Returns
    -------
    TopologizeResult
        ``.chains``        -- list of (M, 2) arrays, one per non-branching segment
        ``.nodes``         -- (K, 2) array of unique chain-endpoint positions
        ``.chain_node_ids``-- list of (start_id, end_id) per chain
        ``.chain_widths``  -- list of (M,) arrays when ``compute_widths=True``,
                             else None
    """
    from topologize._internal import topologize as _topologize

    curves_xy, bd, pcw = _resolve_inflation_radius(inflation_radius, curves)
    fs = _resolve_feature_size(feature_size, inflation_radius, pcw)

    kwargs = {}
    if simplification is not None:
        kwargs["simplification"] = float(simplification)
    if min_tip_length is not None:
        kwargs["min_tip_length"] = float(min_tip_length)
    if junction_merge_fraction is not None:
        kwargs["junction_merge_fraction"] = float(junction_merge_fraction)
    if pcw is not None:
        kwargs["per_curve_widths"] = [
            [float(v) for v in w] for w in pcw
        ]
    if compute_widths:
        kwargs["compute_widths"] = True

    raw = _topologize(_convert_curves(curves_xy), bd, fs, **kwargs)
    return _unpack_result_with_widths(*raw, compute_widths=compute_widths)


def _convert_curves(curves: list[np.ndarray]) -> list[list[tuple[float, float]]]:
    """Convert numpy arrays to list-of-tuples for Rust."""
    return [[tuple(float(v) for v in pt) for pt in curve] for curve in curves]


def _extract_widths(
    curves: list[np.ndarray],
    per_curve_widths: list | None,
) -> tuple[list[np.ndarray], list[list[float]] | None]:
    """Split (N,3) arrays into xy + widths.

    Returns (curves_xy, widths) where widths is None when no
    per-vertex widths are available.
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

    if per_curve_widths is not None:
        return curves_xy, per_curve_widths
    elif has_auto:
        return curves_xy, auto_widths
    else:
        return curves_xy, None


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

    packed = []
    for job in jobs:
        curves_xy, bd, pcw = _resolve_inflation_radius(job.inflation_radius, job.curves)
        fs = _resolve_feature_size(job.feature_size, job.inflation_radius, pcw)
        packed.append((
            _convert_curves(curves_xy),
            bd,
            fs,
            job.simplification,
            job.min_tip_length,
            job.junction_merge_fraction,
        ))
    raw_results = _batch(packed)
    return [_unpack_result(*r) for r in raw_results]
