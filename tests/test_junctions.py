"""Topology preservation in the skeletonizer.

These guard against a non-topological short-cross-edge filter that used to cull
edges purely by length at triangulation time. That severed legitimate interior
structure: it collapsed T/X junctions and broke the skeleton across tight bends,
worst of all at ``boundary_simplification=0`` (no margin), where a clean
crossing could vanish entirely. Culling now happens only downstream, on chains
attached to a degree-1 node (``prune_short_tips``).
"""
from collections import Counter

import numpy as np

from topologize import topologize


def _degrees(result):
    deg = Counter()
    for s, e in result.chain_node_ids:
        deg[s] += 1
        deg[e] += 1
    terminals = sum(1 for d in deg.values() if d == 1)
    junctions = sum(1 for d in deg.values() if d >= 3)
    return terminals, junctions


def test_t_junction_preserved_at_zero_simplification():
    """A T of two strokes must keep its junction: 3 arms, one degree-3 node."""
    curves = [
        np.array([[0.0, 0.0], [200.0, 0.0]]),
        np.array([[100.0, 0.0], [100.0, 150.0]]),
    ]
    res = topologize(curves, 15.0, boundary_simplification=0.0, simplification=0.0)
    terminals, junctions = _degrees(res)
    assert junctions == 1, f"expected one junction, got {junctions}"
    assert terminals == 3, f"expected three terminals, got {terminals}"


def test_x_crossing_not_collapsed_at_zero_simplification():
    """Two crossing strokes must yield a real crossing (one junction, 4 arms),
    not an empty skeleton. This collapsed to zero chains before the fix."""
    curves = [
        np.array([[0.0, 100.0], [200.0, 100.0]]),
        np.array([[100.0, 0.0], [100.0, 200.0]]),
    ]
    res = topologize(curves, 15.0, boundary_simplification=0.0, simplification=0.0)
    assert len(res.chains) >= 4, f"crossing collapsed: {len(res.chains)} chains"
    terminals, junctions = _degrees(res)
    assert junctions == 1, f"expected one crossing node, got {junctions}"
    assert terminals == 4, f"expected four arms, got {terminals}"


def test_closed_loop_stays_connected_across_resample():
    """A closed buffered loop must skeletonize to a single cycle (no terminal
    nodes) regardless of resample spacing. Coarse resampling used to place a
    cross-edge exactly at the 2*buffer width at a bend, which the length filter
    culled, breaking the loop open."""
    a = np.linspace(0.0, 2.0 * np.pi, 240)
    # rounded square so the loop has gentle bends and no sharp corners
    loop = np.column_stack([
        120.0 * np.sign(np.cos(a)) * np.abs(np.cos(a)) ** 0.5,
        120.0 * np.sign(np.sin(a)) * np.abs(np.sin(a)) ** 0.5,
    ])
    loop[-1] = loop[0]
    for spacing in (20.0, 35.0, 50.0):
        res = topologize([loop], 30.0, resample=spacing,
                         boundary_simplification=0.0, simplification=0.0)
        terminals, _ = _degrees(res)
        assert terminals == 0, f"resample={spacing}: loop broke open ({terminals} terminals)"
