# %%
"""
Compare midpoint, voronoi, and python centerline methods on the same input geometry.

Legend entries: input, buffer, midpoint, voronoi, python

Usage:
    uv run python/examples/compare_methods.py
"""
import subprocess
import sys
import tempfile

import numpy as np
import shapely
import plotly.graph_objects as go

from topologize import topologize

# %%  --- input geometry (same as prototype.py) ---

p_pts = np.array([[0.0, 2.0], [10.0, 0.0], [5.0, 0.1], [5.0, 5.0]])

_p2 = shapely.LineString([[0, 6], [1, 4], [2, 0]])
p2_pts = np.array(shapely.LineString(
    _p2.interpolate(np.linspace(0, 1, 24), normalized=True)
).coords)

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

# %% --- run all three methods ---

chains_midpoint = topologize(curves, buffer_distance, method="midpoint")
chains_voronoi  = topologize(curves, buffer_distance, method="voronoi")

# python method (shapely + triangle CDT, same algorithm as prototype.py)
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from examples.prototype import extract_centerline_python  # noqa: E402
chains_python = extract_centerline_python(curves, buffer_distance)

print(f"midpoint : {len(chains_midpoint)} chains")
print(f"voronoi  : {len(chains_voronoi)} chains")
print(f"python   : {len(chains_python)} chains")

# %% --- build NaN-separated arrays ---

NAN = np.full((1, 2), np.nan)


def nan_stack(chains):
    if not chains:
        return np.empty((0, 2))
    return np.concatenate([np.vstack([c, NAN]) for c in chains])


mid_pts = nan_stack(chains_midpoint)
vor_pts = nan_stack(chains_voronoi)
py_pts  = nan_stack(chains_python)

# input strokes
input_pts = np.concatenate([np.vstack([c, NAN]) for c in curves])

# buffer boundary via shapely
def _to_geom(c):
    if np.allclose(c[0], c[-1], atol=1e-8):
        return shapely.LinearRing(c)
    return shapely.LineString(c)

buffered = shapely.unary_union([_to_geom(c) for c in curves]).buffer(
    buffer_distance, join_style="round", cap_style="square"
)
polys = list(buffered.geoms) if buffered.geom_type == "MultiPolygon" else [buffered]
rings = [r for p in polys for r in [p.exterior, *p.interiors]]
buf_pts = np.concatenate([np.vstack([np.array(r.coords), NAN]) for r in rings])

# %% --- plot ---

traces = [
    go.Scatter(
        x=input_pts[:, 0], y=input_pts[:, 1], mode="lines",
        name="input",
        line=dict(color="rgba(180,180,180,0.6)", width=5),
    ),
    go.Scatter(
        x=buf_pts[:, 0], y=buf_pts[:, 1], mode="lines",
        name="buffer",
        line=dict(color="rgba(120,120,120,0.35)", width=1, dash="dot"),
    ),
    go.Scatter(
        x=mid_pts[:, 0], y=mid_pts[:, 1], mode="lines",
        name="midpoint",
        line=dict(color="#e41a1c", width=2),
    ),
    go.Scatter(
        x=vor_pts[:, 0], y=vor_pts[:, 1], mode="lines",
        name="voronoi",
        line=dict(color="#377eb8", width=2),
    ),
    go.Scatter(
        x=py_pts[:, 0], y=py_pts[:, 1], mode="lines",
        name="python",
        line=dict(color="#4daf4a", width=2, dash="dash"),
    ),
]

fig = go.Figure(traces)
fig.update_layout(
    title=f"Centerline comparison — buf={buffer_distance} | "
          f"midpoint={len(chains_midpoint)} voronoi={len(chains_voronoi)} python={len(chains_python)} chains",
    yaxis_scaleanchor="x",
    width=1100,
    height=700,
    margin=dict(l=20, r=20, t=50, b=20),
)

tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
fig.write_html(tmp.name)
subprocess.Popen(["open", tmp.name])
print(f"Plot written to {tmp.name}")
