"""Tests for input validation: feature_size and max_nodes guards."""

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


def test_max_nodes_none_is_unlimited():
    """Default max_nodes=None should not limit output."""
    result = topologize(CURVES, inflation_radius=BD, feature_size=0.1)
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
