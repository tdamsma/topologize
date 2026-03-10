# %%
"""
Centerline visualization with buffer overlay.

Usage:
    uv run python/examples/compare_methods.py
"""
import subprocess
import tempfile

import numpy as np

from topologize import topologize

# %%  --- input geometry (same as getting_started.py) ---

p_pts = np.array([[0.0, 2.0], [10.0, 0.0], [5.0, 0.1], [5.0, 5.0]])

t = np.linspace(0, 1, 24)
p2_pts = np.column_stack([
    np.interp(t, [0, 0.5, 1], [0, 1, 2]),
    np.interp(t, [0, 0.5, 1], [6, 4, 0]),
])

_cc, _cr = np.array([13.0, 1.0]), np.linalg.norm(np.array([13.0, 1.0]) - [10.0, 0.0])
_a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
_circle_open = _cc + _cr * np.column_stack([np.cos(_a), np.sin(_a)])
circle_pts = np.vstack([_circle_open, _circle_open[:1]])

_sc = np.array([5.0, -5.0])
_sa = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 10, endpoint=False)
_sr = np.tile([8.0, 1.8], 5)
_star_open = _sc + np.column_stack([_sr * np.cos(_sa), _sr * np.sin(_sa)])
star_pts = np.vstack([_star_open, _star_open[:1]])

curves = [p_pts, p2_pts, circle_pts, star_pts]
buffer_distance = 0.6

# %% --- run ---

result = topologize(curves, buffer_distance)
print(f"topologize: {len(result.chains)} chains, {sum(len(c) for c in result.chains)} pts")

# %% --- plot ---

fig = result.plot(curves, buffer_distance, title=f"Centerline — buf={buffer_distance}")
tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
fig.write_html(tmp.name)
subprocess.Popen(["open", tmp.name])
print(f"Plot written to {tmp.name}")
