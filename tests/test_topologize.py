import numpy as np
import pytest
from topologize import inflate, topologize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def circle(cx, cy, r, n=64):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.vstack([pts, pts[:1]])  # close the ring


def pair(y_offset=0.3):
    """Two close parallel horizontal lines — guaranteed to produce skeleton output."""
    return [
        np.array([[0.0, y_offset], [10.0, y_offset]]),
        np.array([[0.0, 0.0],      [10.0, 0.0]]),
    ]


def crossing_curves():
    return [
        np.array([[-5.0, 0.1], [5.0, 0.1]]),
        np.array([[-5.0, 0.0], [5.0, 0.0]]),
        np.array([[0.1, -5.0], [0.1, 5.0]]),
        np.array([[0.0, -5.0], [0.0, 5.0]]),
    ]


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_returns_result_with_chains():
    result = topologize(pair(), buffer_distance=0.5)
    assert isinstance(result.chains, list)
    for chain in result.chains:
        assert isinstance(chain, np.ndarray)
        assert chain.ndim == 2
        assert chain.shape[1] == 2


def test_returns_result_with_nodes():
    result = topologize(pair(), buffer_distance=0.5)
    assert isinstance(result.nodes, np.ndarray)
    assert result.nodes.ndim == 2
    assert result.nodes.shape[1] == 2


def test_returns_result_with_chain_node_ids():
    result = topologize(pair(), buffer_distance=0.5)
    assert len(result.chain_node_ids) == len(result.chains)
    for s, e in result.chain_node_ids:
        assert 0 <= s < len(result.nodes)
        assert 0 <= e < len(result.nodes)


def test_empty_input_returns_empty():
    result = topologize([], buffer_distance=1.0)
    assert result.chains == []
    assert result.nodes.shape == (0, 2)
    assert result.chain_node_ids == []


def test_degenerate_curve_skipped():
    # Single-point curve — too short to buffer meaningfully
    result = topologize([np.array([[5.0, 5.0]])], buffer_distance=1.0)
    assert isinstance(result.chains, list)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_two_parallel_lines_produce_output():
    """Two close parallel lines should produce at least one chain."""
    result = topologize(pair(), buffer_distance=0.5)
    chains = result.chains
    assert len(chains) >= 1
    assert sum(len(c) for c in chains) >= 2


def test_two_parallel_lines_merge():
    """Two parallel lines close together should not produce one chain per input stroke."""
    result = topologize(pair(y_offset=0.3), buffer_distance=0.5)
    assert len(result.chains) <= 5


def test_closed_circle():
    """A closed ring should produce at least one chain."""
    result = topologize([circle(0.0, 0.0, 5.0)], buffer_distance=0.5)
    chains = result.chains
    assert len(chains) >= 1
    assert sum(len(c) for c in chains) >= 4


def test_crossing_lines_produce_junction():
    """Two crossing pairs of parallel lines should produce multiple chains (arms at junction)."""
    result = topologize(crossing_curves(), buffer_distance=0.4)
    assert len(result.chains) >= 4


# ---------------------------------------------------------------------------
# Buffer distance sensitivity
# ---------------------------------------------------------------------------

def test_merging_at_larger_buffer():
    """Two separate line pairs: stay independent at small buffer, join into H at large buffer."""
    curves = [
        # pair 1 at y=0
        np.array([[0.0, 0.0],  [10.0, 0.0]]),
        np.array([[0.0, 0.3],  [10.0, 0.3]]),
        # pair 2 at y=10
        np.array([[0.0, 10.0], [10.0, 10.0]]),
        np.array([[0.0, 10.3], [10.0, 10.3]]),
    ]
    result_small = topologize(curves, buffer_distance=0.5)
    # Disable tip pruning for large buffer: default min_tip_length = 12.0 would
    # prune the entire H-shape (all arms are ~5–10 units, below the threshold).
    result_large = topologize(curves, buffer_distance=6.0, min_tip_length=0.0)

    chains_small = result_small.chains
    chains_large = result_large.chains

    # Separate: one chain per pair
    assert len(chains_small) == 2

    # Merged H-shape: top, bottom, and 3 bridge arms at the junction
    assert len(chains_large) >= 3

    # The merged skeleton spans the full y-range (some chain has pts near both y=0 and y=10)
    y_ranges = [c[:, 1].max() - c[:, 1].min() for c in chains_large]
    assert max(y_ranges) > 5.0


def test_three_parallel_pairs_merge_progressively():
    """More buffer → fewer total chains as three regions merge into one."""
    curves = [
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[0.0, 0.3], [10.0, 0.3]]),
        np.array([[0.0, 1.5], [10.0, 1.5]]),
        np.array([[0.0, 1.8], [10.0, 1.8]]),
        np.array([[0.0, 3.0], [10.0, 3.0]]),
        np.array([[0.0, 3.3], [10.0, 3.3]]),
    ]
    result_small = topologize(curves, buffer_distance=0.5)
    result_large = topologize(curves, buffer_distance=2.0)
    assert len(result_large.chains) <= len(result_small.chains)


# ---------------------------------------------------------------------------
# Output values
# ---------------------------------------------------------------------------

def test_output_points_are_finite():
    curves = [
        np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]]),
        np.array([[0.1, 0.1], [9.9, 0.1], [5.0, 4.9]]),
    ]
    result = topologize(curves, buffer_distance=0.5)
    for chain in result.chains:
        assert np.all(np.isfinite(chain))


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def test_crossing_lines_junction_degree():
    """Crossing lines should produce junction nodes (degree >= 3)."""
    result = topologize(crossing_curves(), buffer_distance=0.4)
    # CDT skeleton may produce two degree-3 junctions connected by a bridge
    # rather than a single degree-4 node — either is valid crossing topology.
    assert np.any(result.node_degree >= 3)


def test_crossing_lines_junction_chains():
    """Junction nodes should have chains_at_node consistent with their degree."""
    result = topologize(crossing_curves(), buffer_distance=0.4)
    for node_id, deg in enumerate(result.node_degree):
        if deg >= 3:
            assert len(result.chains_at_node(node_id)) == deg


def test_chain_endpoints_match_nodes():
    """chain[0] and chain[-1] must match the corresponding node positions."""
    result = topologize(crossing_curves(), buffer_distance=0.4)
    for i, (chain, (s, e)) in enumerate(zip(result.chains, result.chain_node_ids)):
        assert np.allclose(chain[0], result.nodes[s], atol=1e-6), \
            f"chain {i} start mismatch: chain[0]={chain[0]}, nodes[{s}]={result.nodes[s]}"
        assert np.allclose(chain[-1], result.nodes[e], atol=1e-6), \
            f"chain {i} end mismatch: chain[-1]={chain[-1]}, nodes[{e}]={result.nodes[e]}"


def test_node_degree_property():
    """node_degree shape and values are consistent with chain_node_ids."""
    result = topologize(pair(), buffer_distance=0.5)
    deg = result.node_degree
    assert deg.shape == (len(result.nodes),)
    assert deg.dtype == int
    # Every node must have degree >= 1
    assert np.all(deg >= 1)
    # Sum of degrees == 2 * number of chains
    assert deg.sum() == 2 * len(result.chains)


def test_chains_at_node_covers_all():
    """Every chain index appears in chains_at_node for both its endpoints."""
    result = topologize(crossing_curves(), buffer_distance=0.4)
    for i, (s, e) in enumerate(result.chain_node_ids):
        assert i in result.chains_at_node(s)
        assert i in result.chains_at_node(e)


# ---------------------------------------------------------------------------
# Variable per-vertex widths
# ---------------------------------------------------------------------------

def test_inflate_backward_compatible_2col():
    """Standard (N,2) curves inflate without per-vertex widths (backward compat)."""
    curves = [np.array([[0.0, 0.0], [5.0, 0.0]])]
    result = inflate(curves, buffer_distance=0.5)
    assert len(result) >= 1
    outer, holes = result[0]
    assert isinstance(outer, np.ndarray)
    assert outer.ndim == 2 and outer.shape[1] == 2


def test_inflate_n3_auto_widths():
    """(N,3) curves: column 2 is used as per-vertex buffer radius."""
    # Curve with per-vertex widths embedded in column 2.
    curve = np.array([[0.0, 0.0, 1.0], [5.0, 0.0, 2.0]])
    result = inflate([curve], buffer_distance=0.5)
    assert len(result) >= 1
    outer, _ = result[0]
    assert isinstance(outer, np.ndarray)
    assert outer.shape[1] == 2  # output is always (N,2)


def test_inflate_explicit_per_curve_widths():
    """Explicit per_curve_widths overrides (N,3) column."""
    curves = [np.array([[0.0, 0.0], [5.0, 0.0]])]
    widths = [[0.8, 1.5]]
    result = inflate(curves, buffer_distance=0.5, per_curve_widths=widths)
    assert len(result) >= 1
    outer, _ = result[0]
    assert isinstance(outer, np.ndarray)
    assert outer.shape[1] == 2


def test_inflate_mixed_widths_some_empty():
    """Mixed input: one curve with widths, one using buffer_distance (empty list)."""
    curves = [
        np.array([[0.0, 0.0], [5.0, 0.0]]),
        np.array([[0.0, 3.0], [5.0, 3.0]]),
    ]
    widths = [[1.0, 1.0], []]  # second curve uses buffer_distance
    result = inflate(curves, buffer_distance=0.5, per_curve_widths=widths)
    assert len(result) >= 1


def test_inflate_per_curve_widths_length_mismatch_raises():
    """per_curve_widths with wrong vertex count raises ValueError."""
    curves = [np.array([[0.0, 0.0], [5.0, 0.0]])]
    widths = [[1.0, 2.0, 3.0]]  # 3 entries for a 2-point curve
    with pytest.raises(ValueError, match="per_curve_widths"):
        inflate(curves, buffer_distance=0.5, per_curve_widths=widths)


def test_inflate_per_curve_widths_too_many_entries_raises():
    """More per_curve_widths entries than curves raises ValueError."""
    curves = [np.array([[0.0, 0.0], [5.0, 0.0]])]
    widths = [[1.0, 1.0], [2.0, 2.0]]  # 2 width entries for 1 curve
    with pytest.raises(ValueError, match="per_curve_widths"):
        inflate(curves, buffer_distance=0.5, per_curve_widths=widths)


def test_topologize_with_n3_curves():
    """topologize accepts (N,3) curves and produces valid output."""
    curve1 = np.array([[0.0, 0.0, 0.5], [10.0, 0.0, 0.5]])
    curve2 = np.array([[0.0, 0.3, 0.5], [10.0, 0.3, 0.5]])
    result = topologize([curve1, curve2], buffer_distance=0.5)
    assert isinstance(result.chains, list)
    assert len(result.chains) >= 1


def test_topologize_explicit_per_curve_widths():
    """topologize accepts explicit per_curve_widths."""
    curves = pair()
    widths = [[0.5, 0.5], [0.5, 0.5]]
    result = topologize(curves, buffer_distance=0.5, per_curve_widths=widths)
    assert isinstance(result.chains, list)
    assert len(result.chains) >= 1


def test_topologize_per_curve_widths_too_many_raises():
    """topologize rejects per_curve_widths with more entries than curves."""
    curves = pair()  # 2 curves
    widths = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]  # 3 entries
    with pytest.raises(ValueError, match="per_curve_widths"):
        topologize(curves, buffer_distance=0.5, per_curve_widths=widths)


def test_topologize_per_curve_widths_vertex_mismatch_raises():
    """topologize rejects per_curve_widths with wrong vertex count."""
    curves = pair()  # each curve has 2 points
    widths = [[0.5, 0.5, 0.5], [0.5, 0.5]]  # first has 3 entries for 2-point curve
    with pytest.raises(ValueError, match="per_curve_widths"):
        topologize(curves, buffer_distance=0.5, per_curve_widths=widths)


# ---------------------------------------------------------------------------
# Junction merging
# ---------------------------------------------------------------------------

def test_perpendicular_crossing_merges_to_x_junction():
    """Default junction_merge_fraction should collapse two T-junctions into a degree-4 node."""
    result = topologize(crossing_curves(), buffer_distance=0.4)
    assert np.any(result.node_degree >= 4), (
        f"Expected a degree-4 X-junction after merging, got degrees: {result.node_degree}"
    )


def test_perpendicular_crossing_no_merge_preserves_t_junctions():
    """junction_merge_fraction=0.0 should keep two separate degree-3 T-junctions."""
    result = topologize(crossing_curves(), buffer_distance=0.4, junction_merge_fraction=0.0)
    assert not np.any(result.node_degree >= 4), (
        f"Expected no degree-4 nodes with merging disabled, got degrees: {result.node_degree}"
    )
    assert np.sum(result.node_degree >= 3) >= 2, (
        f"Expected at least two degree-3 T-junctions, got degrees: {result.node_degree}"
    )
