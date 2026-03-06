# %%
"""
Centerline visualization with buffer overlay.

Usage:
    uv run python/examples/compare_methods.py
"""
import subprocess
import tempfile

import numpy as np
import shapely
import plotly.graph_objects as go

from topologize import topologize

# %%  --- input geometry (same as getting_started.py) ---

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

# %% --- run ---

chains = topologize(curves, buffer_distance)
print(f"topologize: {len(chains)} chains, {sum(len(c) for c in chains)} pts")

# %% --- build NaN-separated arrays ---

NAN = np.full((1, 2), np.nan)


def nan_stack(chains):
    if not chains:
        return np.empty((0, 2))
    return np.concatenate([np.vstack([c, NAN]) for c in chains])


chain_pts = nan_stack(chains)
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

fig = go.Figure([
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
        x=chain_pts[:, 0], y=chain_pts[:, 1], mode="lines",
        name="centerline",
        line=dict(color="#e41a1c", width=2),
    ),
])

fig.update_layout(
    title=f"Centerline — buf={buffer_distance} | {len(chains)} chains",
    yaxis_scaleanchor="x",
    width=1100,
    height=700,
    margin=dict(l=20, r=20, t=50, b=20),
)

tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
fig.write_html(tmp.name)
subprocess.Popen(["open", tmp.name])
print(f"Plot written to {tmp.name}")
