"""Tests for topologize_batch parallel processing."""

import time

import numpy as np
from topologize import topologize, topologize_batch, TopologizeJob


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
    batch_results = topologize_batch([TopologizeJob(cs, bd) for cs in curve_sets])
    assert len(batch_results) == len(curve_sets)
    for cs, br in zip(curve_sets, batch_results):
        sr = topologize(cs, inflation_radius=bd)
        assert len(br.chains) == len(sr.chains)
        assert br.nodes.shape == sr.nodes.shape
        assert len(br.chain_node_ids) == len(sr.chain_node_ids)
        for bc, sc in zip(br.chains, sr.chains):
            np.testing.assert_allclose(bc, sc)


def test_batch_empty_input():
    results = topologize_batch([])
    assert results == []


def test_batch_single_item():
    curves = [line(0, 0, 10, 0)]
    batch = topologize_batch([TopologizeJob(curves, 0.5)])
    single = topologize(curves, inflation_radius=0.5)
    assert len(batch) == 1
    assert len(batch[0].chains) == len(single.chains)


def test_batch_faster_than_sequential():
    """Batch should be faster than a Python loop over the same inputs."""
    rng = np.random.default_rng(42)
    curve_sets = []
    for _ in range(50):
        n_curves = rng.integers(2, 5)
        curves = []
        for _ in range(n_curves):
            pts = rng.uniform(0, 100, size=(20, 2))
            pts = np.cumsum(pts, axis=0)
            curves.append(pts)
        curve_sets.append(curves)

    bd = 2.0
    jobs = [TopologizeJob(cs, bd) for cs in curve_sets]

    # Warm up (first call may include one-time rayon thread-pool init)
    topologize_batch(jobs[:2])

    t0 = time.perf_counter()
    for cs in curve_sets:
        topologize(cs, inflation_radius=bd)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    topologize_batch(jobs)
    t_batch = time.perf_counter() - t0

    speedup = t_seq / t_batch
    print(f"sequential: {t_seq:.3f}s, batch: {t_batch:.3f}s, speedup: {speedup:.1f}x")
    assert speedup >= 1.5, (
        f"batch should be at least 1.5x faster, got {speedup:.1f}x "
        f"(sequential {t_seq:.3f}s, batch {t_batch:.3f}s)"
    )


def test_batch_per_job_params():
    """Each job can have its own parameters, verified against sequential calls."""
    curve_sets = make_curve_sets()
    jobs = [
        TopologizeJob(curve_sets[0], 0.5, simplification=0.0),
        TopologizeJob(curve_sets[1], 0.5, min_tip_length=0.0),
        TopologizeJob(curve_sets[2], 0.5, junction_merge_fraction=0.0),
    ]
    batch_results = topologize_batch(jobs)
    assert len(batch_results) == len(jobs)
    for job, br in zip(jobs, batch_results):
        sr = topologize(
            job.curves,
            inflation_radius=job.inflation_radius,
            simplification=job.simplification,
            min_tip_length=job.min_tip_length,
            junction_merge_fraction=job.junction_merge_fraction,
        )
        assert len(br.chains) == len(sr.chains)
        assert br.nodes.shape == sr.nodes.shape
        for bc, sc in zip(br.chains, sr.chains):
            np.testing.assert_allclose(bc, sc)
