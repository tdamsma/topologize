"""Tests for the curvature-adaptive boundary resampling.

The resample step folds curvature refinement in by *choosing where to sample
the offset boundary* (denser through curves), so every triangulation vertex
lands on the true offset instead of on a chord added afterwards.
"""
import numpy as np
import pytest

from topologize import inflate, triangulate


def _arc(cx, cy, r, a0, a1, n=80):
    a = np.linspace(a0, a1, n)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def _boundary_segments(polys):
    A, B = [], []
    for outer, holes in polys:
        for ring in [outer] + list(holes):
            r = np.vstack([ring, ring[:1]])
            A.append(r[:-1])
            B.append(r[1:])
    return np.vstack(A), np.vstack(B)


def _max_dist_to_boundary(points, polys):
    A, B = _boundary_segments(polys)
    AB = B - A
    denom = np.maximum((AB * AB).sum(1), 1e-12)
    worst = 0.0
    for p in points:
        t = np.clip(((p - A) * AB).sum(1) / denom, 0.0, 1.0)
        proj = A + t[:, None] * AB
        worst = max(worst, np.hypot(*(proj - p).T).min())
    return worst


def test_resample_vertices_lie_on_boundary():
    """With RDP off, every CDT vertex must sit on the offset boundary, even on a
    tightly curved buffer and even at a coarse resample spacing. This guards
    against the old behaviour where curvature refinement split coarse chords and
    pushed vertices ~1 unit inside the smooth arc."""
    curve = _arc(0.0, 0.0, 60.0, -0.6 * np.pi, 0.6 * np.pi)
    buf = 30.0

    polys = inflate([curve], buf)
    for spacing in (8.0, 20.0):
        tris = triangulate([curve], buf, resample=spacing, boundary_simplification=0.0)
        verts = np.array(sorted({(round(p[0], 6), round(p[1], 6))
                                 for t in tris for p in t}))
        worst = _max_dist_to_boundary(verts, polys)
        assert worst < 1e-6, f"spacing={spacing}: vertex {worst:.3f} off boundary"


def test_resample_denser_through_curves():
    """Curvature refinement is folded into the resample: a curved stretch ends up
    with shorter boundary edges than the requested base spacing, while straight
    stretches stay near it (so the density genuinely adapts)."""
    curve = _arc(0.0, 0.0, 60.0, -0.6 * np.pi, 0.6 * np.pi)
    buf = 30.0
    spacing = 20.0
    tris = triangulate([curve], buf, resample=spacing, boundary_simplification=0.0)

    from collections import Counter
    key = lambda p: (round(p[0], 4), round(p[1], 4))
    ec = Counter()
    for a, b, c in tris:
        for p, q in [(a, b), (b, c), (c, a)]:
            ec[tuple(sorted([key(p), key(q)]))] += 1
    bedges = [e for e, n in ec.items() if n == 1]
    L = np.array([np.hypot(e[0][0] - e[1][0], e[0][1] - e[1][1]) for e in bedges])

    # The curved side carries edges well below the base spacing...
    assert L.min() < 0.9 * spacing
    # ...but nothing is forced far denser than the curvature floor (~ratio*buf).
    assert L.min() > 0.3 * spacing


def test_resample_with_holes_produces_skeleton():
    """A closed loop inflates to an outer ring plus a hole. Resampling must not
    collapse either ring to a degenerate (<3 pt) contour, which previously made
    the CDT return nothing (0 chains) for hole-bearing geometry."""
    loop = [np.array([[0.0, 0.0], [120.0, 0.0], [120.0, 120.0],
                      [0.0, 120.0], [0.0, 0.0]])]
    polys = inflate(loop, 20.0)
    assert sum(len(h) for _, h in polys) >= 1, "expected a hole"

    from topologize import topologize
    for spacing in (20.0, 50.0):
        res = topologize(loop, 20.0, resample=spacing, boundary_simplification=0.0)
        assert len(res.chains) >= 1, f"resample={spacing} produced no skeleton"


def test_resample_no_subdivision_quantization():
    """Density varies smoothly: boundary edges should not cluster at the discrete
    base/n values (20, 10, 6.67, ...) that the old ceil()-based subdivision
    produced."""
    curve = _arc(0.0, 0.0, 60.0, -0.6 * np.pi, 0.6 * np.pi)
    buf = 30.0
    spacing = 20.0
    tris = triangulate([curve], buf, resample=spacing, boundary_simplification=0.0)

    from collections import Counter
    key = lambda p: (round(p[0], 4), round(p[1], 4))
    ec = Counter()
    for a, b, c in tris:
        for p, q in [(a, b), (b, c), (c, a)]:
            ec[tuple(sorted([key(p), key(q)]))] += 1
    L = np.array([np.hypot(e[0][0] - e[1][0], e[0][1] - e[1][1])
                  for e, n in ec.items() if n == 1])

    # Almost no edges should land exactly on base/2 = 10 (the old artifact).
    near_half = int((np.abs(L - spacing / 2) < 0.2).sum())
    assert near_half <= 2, f"{near_half} edges quantized to base/2 (subdivision jump)"
