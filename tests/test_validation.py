"""Tests for input validation: feature_size and max_nodes guards."""

import json
from pathlib import Path

import numpy as np
import pytest
from topologize import inflate, topologize, topologize_batch, triangulate, TopologizeJob


CURVES = [np.array([(0, 0), (10, 0), (10, 10)], dtype=float)]
BD = 1.0


# ---------------------------------------------------------------------------
# feature_size validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_fs", [0.0, -1.0, -0.001, float("inf"), float("-inf"), float("nan")])
class TestFeatureSizeValidation:
    def test_topologize(self, bad_fs):
        with pytest.raises(ValueError, match="feature_size"):
            topologize(CURVES, inflation_radius=BD, feature_size=bad_fs)

    def test_topologize_batch(self, bad_fs):
        with pytest.raises(ValueError, match="feature_size"):
            topologize_batch([TopologizeJob(CURVES, BD, feature_size=bad_fs)])

    def test_inflate(self, bad_fs):
        with pytest.raises(ValueError, match="feature_size"):
            inflate(CURVES, inflation_radius=BD, feature_size=bad_fs)

    def test_triangulate(self, bad_fs):
        with pytest.raises(ValueError, match="feature_size"):
            triangulate(CURVES, inflation_radius=BD, feature_size=bad_fs)


# ---------------------------------------------------------------------------
# max_nodes validation
# ---------------------------------------------------------------------------

def test_max_nodes_exceeded():
    """A tight max_nodes limit should raise ValueError."""
    with pytest.raises(ValueError, match="exceeding max_nodes"):
        topologize(CURVES, inflation_radius=BD, feature_size=0.1, max_nodes=5)


def test_max_nodes_negative():
    """Negative max_nodes should raise ValueError."""
    with pytest.raises(ValueError, match="max_nodes must be a positive integer"):
        topologize(CURVES, inflation_radius=BD, max_nodes=-1)


def test_max_nodes_none_disables_limit():
    """max_nodes=None should disable the limit entirely."""
    result = topologize(CURVES, inflation_radius=BD, feature_size=0.1, max_nodes=None)
    assert len(result.nodes) > 0


def test_max_nodes_generous():
    """A generous limit should not interfere."""
    result = topologize(CURVES, inflation_radius=BD, feature_size=0.1, max_nodes=100_000)
    assert len(result.nodes) > 0


def test_max_nodes_batch_includes_job_index():
    """Batch max_nodes error should identify which job failed."""
    jobs = [
        TopologizeJob(CURVES, BD),
        TopologizeJob(CURVES, BD, feature_size=0.1, max_nodes=5),
    ]
    with pytest.raises(ValueError, match="job 1"):
        topologize_batch(jobs)


# ---------------------------------------------------------------------------
# per_curve_widths validation
#
# A wrong-length width list used to panic the whole process via an assert_eq!
# deep in inflate_curve_variable. It must surface as a clean ValueError at every
# Rust entry point. (These hit the _internal bindings directly: the Python
# wrappers derive widths from (N, 3) curves, so a mismatch can't occur there.)
# ---------------------------------------------------------------------------

_GOOD_CURVE = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]


def test_topologize_internal_rejects_mismatched_widths():
    from topologize._internal import topologize as _t
    with pytest.raises(ValueError, match="per_curve_widths"):
        _t([_GOOD_CURVE], 5.0, 5.0, per_curve_widths=[[8.0, 8.0]])


def test_triangulate_internal_rejects_mismatched_widths():
    from topologize._internal import triangulate_curves as _tri
    with pytest.raises(ValueError, match="per_curve_widths"):
        _tri([_GOOD_CURVE], 5.0, 5.0, per_curve_widths=[[8.0, 8.0]])


def test_inflate_internal_rejects_mismatched_widths():
    from topologize._internal import inflate_curves as _inf
    with pytest.raises(ValueError, match="per_curve_widths"):
        _inf([_GOOD_CURVE], 5.0, 5.0, per_curve_widths=[[8.0, 8.0]])


def test_variable_width_close_points_does_not_panic():
    """Variable-width curves with vertices closer than the decimation step used
    to panic: points were decimated but widths were not, so they fell out of
    sync inside inflate_curve_variable. They must now topologize cleanly."""
    n = 20
    curve = np.column_stack([np.linspace(0, 19, n), np.zeros(n)])
    widths = [np.full(n, 8.0)]  # one radius per vertex
    result = topologize([curve], inflation_radius=widths)
    assert len(result.chains) >= 1


def test_large_coordinates_recover_from_cdt_wedge_escape():
    # This boat layer hits cdt's WedgeEscape when offset to 1e6 coordinates.
    data = Path(__file__).parents[1] / "python/examples/data/topo_variability_input.json"
    layer = next(x for x in json.loads(data.read_text())["layers"] if x["z"] == 1196.0)
    curves = [np.asarray(c, dtype=float) for c in layer["curves"]]
    expected = topologize(curves, inflation_radius=7.9)
    shifted = topologize([c + 1e6 for c in curves], inflation_radius=7.9)
    assert len(shifted.chains) == len(expected.chains) > 0


@pytest.mark.parametrize("resample", [None, 0.6])
def test_boundary_preprocessing_does_not_cross_narrow_neck(resample):
    # Simplifying a narrow neck in this boundary used to make a ring cross itself.
    curves = [
        np.array([[-5.34, -2.14], [-4.86, -1.9], [-7.26, 0.92]]),
        np.array([[0.07, 1.81], [-2.52, 0.11], [-2.81, -2.0], [-4.69, -9.29]]),
    ]
    result = topologize(curves, inflation_radius=1.0, resample=resample)
    assert len(result.chains) > 0
