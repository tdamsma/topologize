"""Tests for topologize_batch parallel processing."""

import numpy as np
import pytest
from topologize import topologize, topologize_batch


def line(x0, y0, x1, y1, n=10):
    return np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)])


def make_curve_sets():
    """Three independent curve-sets."""
    return [
        [line(0, 0, 10, 0), line(0, 1, 10, 1)],   # parallel horizontal
        [line(0, 0, 0, 10), line(1, 0, 1, 10)],   # parallel vertical
        [line(0, 0, 5, 5), line(5, 0, 0, 5)],     # crossing diagonals
    ]


def test_batch_matches_sequential():
    curve_sets = make_curve_sets()
    bd = 0.5
    batch_results = topologize_batch(curve_sets, bd)
    for cs, br in zip(curve_sets, batch_results):
        sr = topologize(cs, buffer_distance=bd)
        assert len(br.chains) == len(sr.chains)
        assert br.nodes.shape == sr.nodes.shape
        assert len(br.chain_node_ids) == len(sr.chain_node_ids)
        for bc, sc in zip(br.chains, sr.chains):
            np.testing.assert_allclose(bc, sc)


def test_batch_empty_input():
    results = topologize_batch([], buffer_distance=1.0)
    assert results == []


def test_batch_single_item():
    curves = [line(0, 0, 10, 0)]
    batch = topologize_batch([curves], buffer_distance=0.5)
    single = topologize(curves, buffer_distance=0.5)
    assert len(batch) == 1
    assert len(batch[0].chains) == len(single.chains)


def test_batch_kwargs_forwarded():
    curve_sets = make_curve_sets()
    results = topologize_batch(
        curve_sets,
        buffer_distance=0.5,
        simplification=0.0,
        min_tip_length=0.0,
        junction_merge_fraction=0.0,
    )
    assert len(results) == 3
    for r in results:
        assert len(r.chains) > 0
