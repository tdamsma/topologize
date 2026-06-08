"""Tests for endpoint-merging of split input subpaths before offsetting."""
import numpy as np
import pytest

from topologize import inflate, topologize


def _outer_area(polys):
    """Total shoelace area of all outer rings."""
    total = 0.0
    for outer, _holes in polys:
        x, y = outer[:, 0], outer[:, 1]
        total += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return total


def test_merge_reconstructs_split_polyline():
    """A bent polyline split at its corner inflates identically to the unsplit one
    when merging is on; with merging off the interior square caps change the area."""
    single = [np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]])]
    split = [
        np.array([[0.0, 0.0], [50.0, 0.0]]),
        np.array([[50.0, 0.0], [50.0, 50.0]]),
    ]
    r = 5.0
    a_single = _outer_area(inflate(single, r))
    a_merged = _outer_area(inflate(split, r))          # merge on by default
    a_unmerged = _outer_area(inflate(split, r, merge_tolerance=0.0))

    assert a_merged == pytest.approx(a_single, rel=1e-9)
    assert a_unmerged > a_single + 1.0  # interior end-caps add area


def test_merge_orientation_handles_reversed_subpath():
    """Subpaths that connect tail-to-tail (one reversed) still merge cleanly."""
    single = [np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]])]
    # second subpath reversed so the shared point (50,0) is *its* end, not start
    split = [
        np.array([[0.0, 0.0], [50.0, 0.0]]),
        np.array([[50.0, 50.0], [50.0, 0.0]]),
    ]
    r = 5.0
    assert _outer_area(inflate(split, r)) == pytest.approx(
        _outer_area(inflate(single, r)), rel=1e-9
    )


def test_merge_single_chain_for_split_contour():
    """A connected contour authored as separate subpaths yields one chain."""
    pts = [[0, 0], [40, 0], [80, 25], [120, 0], [160, 25]]
    split = [np.array([pts[i], pts[i + 1]], float) for i in range(len(pts) - 1)]
    res = topologize(split, inflation_radius=10.0)  # merge on by default
    assert len(res.chains) == 1


def test_merge_preserves_true_junction():
    """A 4-way star (degree-4 shared point) must not be merged through, so the
    result is identical with merging on vs off."""
    star = [
        np.array([[0.0, 0.0], [50.0, 0.0]]),
        np.array([[0.0, 0.0], [-50.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 50.0]]),
        np.array([[0.0, 0.0], [0.0, -50.0]]),
    ]
    on = topologize(star, inflation_radius=8.0)
    off = topologize(star, inflation_radius=8.0, merge_tolerance=0.0)
    assert len(on.chains) == len(off.chains)
    assert len(on.nodes) == len(off.nodes)


def test_merge_disabled_keeps_caps():
    """merge_tolerance=0 disables merging (sanity: differs from default on a split)."""
    split = [
        np.array([[0.0, 0.0], [50.0, 0.0]]),
        np.array([[50.0, 0.0], [50.0, 50.0]]),
    ]
    r = 5.0
    assert _outer_area(inflate(split, r, merge_tolerance=0.0)) != pytest.approx(
        _outer_area(inflate(split, r)), rel=1e-6
    )
