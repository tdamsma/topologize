# %% [markdown]
# # topologize — getting started
#
# `topologize` takes a set of polylines, inflates them by a `buffer_distance`,
# and returns the medial axis as maximal non-branching chains.
#
# **Typical use case:** clean up hand-drawn or over-sampled strokes into a
# single-pixel-wide centerline graph.

# %%
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from topologize import topologize

# %% [markdown]
# ## Example 1: a few simple polylines
#
# Four curves — two open polylines, a circle, and a star — placed close enough
# that their inflated buffers overlap and merge.

# %%
# open polyline
p1 = np.array([[0.0, 2.0], [10.0, 0.0], [5.0, 0.1], [5.0, 5.0]])

# open polyline resampled to 24 points
_p2 = np.array([[0.0, 6.0], [1.0, 4.0], [2.0, 0.0]])
_p2_dists = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(_p2, axis=0), axis=1))])
_t = np.linspace(0, _p2_dists[-1], 24)
p2 = np.column_stack([np.interp(_t, _p2_dists, _p2[:, 0]), np.interp(_t, _p2_dists, _p2[:, 1])])

# circle (closed ring)
_cc, _cr = np.array([13.0, 1.0]), np.linalg.norm(np.array([13.0, 1.0]) - [10.0, 0.0])
_a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
_circle = _cc + _cr * np.column_stack([np.cos(_a), np.sin(_a)])
circle = np.vstack([_circle, _circle[:1]])

# 5-point star (closed ring)
_sc = np.array([5.0, -5.0])
_sa = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 10, endpoint=False)
_sr = np.tile([8.0, 1.8], 5)
_star = _sc + np.column_stack([_sr * np.cos(_sa), _sr * np.sin(_sa)])
star = np.vstack([_star, _star[:1]])

curves = [p1, p2, circle, star]

# %%
buffer_distance = 0.6
result = topologize(curves, buffer_distance)
print(f"{len(result.chains)} chains, {sum(len(c) for c in result.chains)} total points")
result.plot(curves, buffer_distance, title="Simple shapes")

# %% [markdown]
# ## Example 2: effect of buffer_distance
#
# A larger `buffer_distance` merges nearby strokes more aggressively,
# producing fewer but smoother chains.

# %%
for bd in [0.3, 0.6, 1.2]:
    r = topologize(curves, bd)
    r.plot(curves, bd, title=f"buffer_distance={bd}  ({len(r.chains)} chains)").show()

# %% [markdown]
# ## Example 3: parallel lines merging into one
#
# Two nearly-parallel lines become a single centerline when the buffer
# is large enough to bridge the gap between them.

# %%
line_a = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
line_b = np.column_stack([np.linspace(0, 10, 50), np.ones(50) * 0.8])

for bd in [0.3, 0.6]:
    r = topologize([line_a, line_b], bd)
    r.plot([line_a, line_b], bd, title=f"buffer_distance={bd}  ({len(r.chains)} chain{'s' if len(r.chains) != 1 else ''})").show()

# %% [markdown]
# ## Example 4: inspecting chain geometry
#
# Each chain is a plain `(N, 2)` numpy array — easy to work with downstream.

# %%
chains = topologize(curves, buffer_distance=0.6).chains

for i, chain in enumerate(chains):
    length = np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1))
    print(f"chain {i:2d}:  {len(chain):4d} pts,  length={length:.3f}")