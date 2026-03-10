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
    result = topologize(pair(), inflation_radius=0.5)
    assert isinstance(result.chains, list)
    for chain in result.chains:
        assert isinstance(chain, np.ndarray)
        assert chain.ndim == 2
        assert chain.shape[1] == 2


def test_returns_result_with_nodes():
    result = topologize(pair(), inflation_radius=0.5)
    assert isinstance(result.nodes, np.ndarray)
    assert result.nodes.ndim == 2
    assert result.nodes.shape[1] == 2


def test_returns_result_with_chain_node_ids():
    result = topologize(pair(), inflation_radius=0.5)
    assert len(result.chain_node_ids) == len(result.chains)
    for s, e in result.chain_node_ids:
        assert 0 <= s < len(result.nodes)
        assert 0 <= e < len(result.nodes)


def test_chain_widths_none_by_default():
    result = topologize(pair(), inflation_radius=0.5)
    assert result.chain_widths is None


def test_returns_result_with_chain_widths():
    result = topologize(pair(), inflation_radius=0.5, compute_widths=True)
    assert isinstance(result.chain_widths, list)
    assert len(result.chain_widths) == len(result.chains)
    for w, chain in zip(result.chain_widths, result.chains):
        assert isinstance(w, np.ndarray)
        assert w.ndim == 1
        assert len(w) == len(chain)


def test_chain_widths_are_positive():
    result = topologize(pair(), inflation_radius=0.5, compute_widths=True)
    for w in result.chain_widths:
        assert np.all(w > 0), "all width values must be positive"


def test_chain_widths_approximately_two_times_buffer():
    """Interior width estimates should be close to 2 × inflation_radius."""
    ir = 0.5
    result = topologize(pair(), inflation_radius=ir, compute_widths=True)
    for w in result.chain_widths:
        if len(w) < 3:
            continue
        interior = w[1:-1]
        assert np.all(interior > 0), "interior widths must be positive"
        assert np.all(interior < 4 * ir), (
            f"interior widths unexpectedly large: {interior}"
        )


def test_empty_input_returns_empty():
    result = topologize([], inflation_radius=1.0)
    assert result.chains == []
    assert result.nodes.shape == (0, 2)
    assert result.chain_node_ids == []
    assert result.chain_widths is None


def test_degenerate_curve_skipped():
    # Single-point curve — too short to buffer meaningfully
    result = topologize([np.array([[5.0, 5.0]])], inflation_radius=1.0)
    assert isinstance(result.chains, list)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_two_parallel_lines_produce_output():
    """Two close parallel lines should produce at least one chain."""
    result = topologize(pair(), inflation_radius=0.5)
    chains = result.chains
    assert len(chains) >= 1
    assert sum(len(c) for c in chains) >= 2


def test_two_parallel_lines_merge():
    """Two parallel lines close together should not produce one chain per input stroke."""
    result = topologize(pair(y_offset=0.3), inflation_radius=0.5)
    assert len(result.chains) <= 5


def test_closed_circle():
    """A closed ring should produce at least one chain."""
    result = topologize([circle(0.0, 0.0, 5.0)], inflation_radius=0.5)
    chains = result.chains
    assert len(chains) >= 1
    assert sum(len(c) for c in chains) >= 4


def test_crossing_lines_produce_junction():
    """Two crossing pairs of parallel lines should produce multiple chains (arms at junction)."""
    result = topologize(crossing_curves(), inflation_radius=0.4)
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
    result_small = topologize(curves, inflation_radius=0.5)
    # Disable tip pruning for large buffer: default min_tip_length = 12.0 would
    # prune the entire H-shape (all arms are ~5–10 units, below the threshold).
    result_large = topologize(curves, inflation_radius=6.0, min_tip_length=0.0)

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
    result_small = topologize(curves, inflation_radius=0.5)
    result_large = topologize(curves, inflation_radius=2.0)
    assert len(result_large.chains) <= len(result_small.chains)


# ---------------------------------------------------------------------------
# Output values
# ---------------------------------------------------------------------------

def test_output_points_are_finite():
    curves = [
        np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]]),
        np.array([[0.1, 0.1], [9.9, 0.1], [5.0, 4.9]]),
    ]
    result = topologize(curves, inflation_radius=0.5)
    for chain in result.chains:
        assert np.all(np.isfinite(chain))


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def test_crossing_lines_junction_degree():
    """Crossing lines should produce junction nodes (degree >= 3)."""
    result = topologize(crossing_curves(), inflation_radius=0.4)
    assert np.any(result.node_degree >= 3)


def test_crossing_lines_junction_chains():
    """Junction nodes should have chains_at_node consistent with their degree."""
    result = topologize(crossing_curves(), inflation_radius=0.4)
    for node_id, deg in enumerate(result.node_degree):
        if deg >= 3:
            assert len(result.chains_at_node(node_id)) == deg


def test_chain_endpoints_match_nodes():
    """chain[0] and chain[-1] must match the corresponding node positions."""
    result = topologize(crossing_curves(), inflation_radius=0.4)
    for i, (chain, (s, e)) in enumerate(zip(result.chains, result.chain_node_ids)):
        assert np.allclose(chain[0], result.nodes[s], atol=1e-6), \
            f"chain {i} start mismatch: chain[0]={chain[0]}, nodes[{s}]={result.nodes[s]}"
        assert np.allclose(chain[-1], result.nodes[e], atol=1e-6), \
            f"chain {i} end mismatch: chain[-1]={chain[-1]}, nodes[{e}]={result.nodes[e]}"


def test_node_degree_property():
    """node_degree shape and values are consistent with chain_node_ids."""
    result = topologize(pair(), inflation_radius=0.5)
    deg = result.node_degree
    assert deg.shape == (len(result.nodes),)
    assert deg.dtype == int
    assert np.all(deg >= 1)
    assert deg.sum() == 2 * len(result.chains)


def test_chains_at_node_covers_all():
    """Every chain index appears in chains_at_node for both its endpoints."""
    result = topologize(crossing_curves(), inflation_radius=0.4)
    for i, (s, e) in enumerate(result.chain_node_ids):
        assert i in result.chains_at_node(s)
        assert i in result.chains_at_node(e)


# ---------------------------------------------------------------------------
# Variable per-vertex widths
# ---------------------------------------------------------------------------

def test_inflate_uniform():
    """Standard (N,2) curves inflate with uniform radius."""
    curves = [np.array([[0.0, 0.0], [5.0, 0.0]])]
    result = inflate(curves, inflation_radius=0.5)
    assert len(result) >= 1
    outer, holes = result[0]
    assert isinstance(outer, np.ndarray)
    assert outer.ndim == 2 and outer.shape[1] == 2


def test_inflate_n3_auto_widths():
    """(N,3) curves: column 2 is used as per-vertex buffer radius."""
    curve = np.array([[0.0, 0.0, 1.0], [5.0, 0.0, 2.0]])
    result = inflate([curve], inflation_radius=0.5)
    assert len(result) >= 1
    outer, _ = result[0]
    assert isinstance(outer, np.ndarray)
    assert outer.shape[1] == 2


def test_inflate_per_vertex_widths():
    """Per-vertex widths via inflation_radius list."""
    curves = [np.array([[0.0, 0.0], [5.0, 0.0]])]
    widths = [np.array([0.8, 1.5])]
    result = inflate(curves, inflation_radius=widths)
    assert len(result) >= 1
    outer, _ = result[0]
    assert isinstance(outer, np.ndarray)
    assert outer.shape[1] == 2


def test_inflate_per_vertex_length_mismatch_raises():
    """Per-vertex widths with wrong vertex count raises ValueError."""
    curves = [np.array([[0.0, 0.0], [5.0, 0.0]])]
    widths = [np.array([1.0, 2.0, 3.0])]  # 3 entries for a 2-point curve
    with pytest.raises(ValueError, match="inflation_radius"):
        inflate(curves, inflation_radius=widths)


def test_inflate_too_many_entries_raises():
    """More inflation_radius entries than curves raises ValueError."""
    curves = [np.array([[0.0, 0.0], [5.0, 0.0]])]
    widths = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]  # 2 entries for 1 curve
    with pytest.raises(ValueError, match="inflation_radius"):
        inflate(curves, inflation_radius=widths)


def test_inflate_fewer_entries_raises():
    """Fewer inflation_radius entries than curves raises ValueError."""
    curves = [
        np.array([[0.0, 0.0], [5.0, 0.0]]),
        np.array([[0.0, 3.0], [5.0, 3.0]]),
    ]
    widths = [np.array([1.0, 1.0])]  # 1 entry for 2 curves
    with pytest.raises(ValueError, match="inflation_radius"):
        inflate(curves, inflation_radius=widths)


def test_topologize_with_n3_curves():
    """topologize accepts (N,3) curves and produces valid output."""
    curve1 = np.array([[0.0, 0.0, 0.5], [10.0, 0.0, 0.5]])
    curve2 = np.array([[0.0, 0.3, 0.5], [10.0, 0.3, 0.5]])
    result = topologize([curve1, curve2], inflation_radius=0.5)
    assert isinstance(result.chains, list)
    assert len(result.chains) >= 1


def test_topologize_per_vertex_widths():
    """topologize accepts per-vertex widths via inflation_radius list."""
    curves = pair()
    widths = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    result = topologize(curves, inflation_radius=widths)
    assert isinstance(result.chains, list)
    assert len(result.chains) >= 1


def test_topologize_per_vertex_too_many_raises():
    """topologize rejects inflation_radius with more entries than curves."""
    curves = pair()  # 2 curves
    widths = [np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.5])]  # 3 entries
    with pytest.raises(ValueError, match="inflation_radius"):
        topologize(curves, inflation_radius=widths)


def test_topologize_per_vertex_mismatch_raises():
    """topologize rejects inflation_radius with wrong vertex count."""
    curves = pair()  # each curve has 2 points
    widths = [np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5])]  # first has 3 entries for 2-point curve
    with pytest.raises(ValueError, match="inflation_radius"):
        topologize(curves, inflation_radius=widths)


def test_feature_size_override():
    """Explicit feature_size should be used for thresholds."""
    curves = pair()
    # Small inflation_radius but large feature_size => more aggressive pruning
    result = topologize(curves, inflation_radius=0.5, feature_size=5.0)
    assert isinstance(result.chains, list)


def test_feature_size_auto_derived_from_per_vertex_widths():
    """When inflation_radius is a list, feature_size defaults to median of all widths."""
    curves = pair()
    widths = [np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    # Should not raise — feature_size is auto-derived as median(all widths) = 0.5
    result = topologize(curves, inflation_radius=widths)
    assert isinstance(result.chains, list)
    assert len(result.chains) >= 1


def test_topologize_per_vertex_no_widths_raises():
    """inflation_radius as an empty list raises ValueError."""
    curves = pair()
    with pytest.raises(ValueError, match="no width values"):
        topologize(curves, inflation_radius=[np.array([]), np.array([])])


# ---------------------------------------------------------------------------
# Junction merging
# ---------------------------------------------------------------------------

def test_perpendicular_crossing_merges_to_x_junction():
    """Default junction_merge_fraction should collapse two T-junctions into a degree-4 node."""
    result = topologize(crossing_curves(), inflation_radius=0.4)
    assert np.any(result.node_degree >= 4), (
        f"Expected a degree-4 X-junction after merging, got degrees: {result.node_degree}"
    )


def test_perpendicular_crossing_no_merge_preserves_t_junctions():
    """junction_merge_fraction=0.0 should keep two separate degree-3 T-junctions."""
    result = topologize(crossing_curves(), inflation_radius=0.4, junction_merge_fraction=0.0)
    assert not np.any(result.node_degree >= 4), (
        f"Expected no degree-4 nodes with merging disabled, got degrees: {result.node_degree}"
    )
    assert np.sum(result.node_degree >= 3) >= 2, (
        f"Expected at least two degree-3 T-junctions, got degrees: {result.node_degree}"
    )


# ---------------------------------------------------------------------------
# curve_ids / chain_source_ids provenance
# ---------------------------------------------------------------------------

def test_chain_source_ids_none_when_omitted():
    """chain_source_ids must be None when curve_ids is not provided."""
    result = topologize(pair(), buffer_distance=0.5)
    assert result.chain_source_ids is None


def test_chain_source_ids_list_when_provided():
    """chain_source_ids must be a list of frozensets when curve_ids is provided."""
    curves = pair()
    result = topologize(curves, buffer_distance=0.5, curve_ids=[10, 20])
    assert result.chain_source_ids is not None
    assert isinstance(result.chain_source_ids, list)
    assert len(result.chain_source_ids) == len(result.chains)
    for s in result.chain_source_ids:
        assert isinstance(s, frozenset)


def test_chain_source_ids_contains_expected_ids():
    """Two close parallel lines with distinct IDs: the merged chain should reference both IDs."""
    curves = pair()
    result = topologize(curves, buffer_distance=0.5, curve_ids=[10, 20])
    assert result.chain_source_ids is not None
    # The two input curves are close together and merge into one skeleton chain;
    # both IDs should appear in at least one chain's source set.
    all_ids = set().union(*result.chain_source_ids)
    assert 10 in all_ids
    assert 20 in all_ids


def test_chain_source_ids_single_curve():
    """A single input curve: its ID should appear in all non-empty source sets."""
    curve = np.array([[0.0, 0.0], [10.0, 0.0]])
    result = topologize([curve, np.array([[0.0, 0.2], [10.0, 0.2]])],
                        buffer_distance=0.5, curve_ids=[42, 43])
    assert result.chain_source_ids is not None
    all_ids = set().union(*result.chain_source_ids)
    assert 42 in all_ids


def test_chain_source_ids_crossing_lines():
    """Crossing lines: each arm of the X should reference at least one input ID."""
    curves = crossing_curves()
    ids = [1, 2, 3, 4]
    result = topologize(curves, buffer_distance=0.4, curve_ids=ids)
    assert result.chain_source_ids is not None
    assert len(result.chain_source_ids) == len(result.chains)
    # Each chain must have at least one contributing source curve
    for source_set in result.chain_source_ids:
        assert len(source_set) >= 1
    # All four IDs should appear across all chains
    all_ids = set().union(*result.chain_source_ids)
    for i in ids:
        assert i in all_ids


def test_chain_source_ids_length_mismatch_raises():
    """Passing curve_ids with wrong length must raise ValueError."""
    import pytest
    curves = pair()  # 2 curves
    with pytest.raises(ValueError, match="curve_ids"):
        topologize(curves, buffer_distance=0.5, curve_ids=[1, 2, 3])
