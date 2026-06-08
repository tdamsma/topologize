# %% [markdown]
# # Curvature-adaptive boundary resampling
#
# The CDT skeleton is only as good as the boundary it triangulates. On a
# tightly-curved buffer the parallel offset puts many more vertices on the
# convex (outer) side than the concave (inner) side, which produces skewed
# "fan" triangles whose midpoints wander off the true centerline.
#
# Setting `resample` redistributes each boundary ring to a roughly uniform
# arc-length spacing, tightening *smoothly* through curves. Crucially, every
# sample is taken **on** the offset boundary, so triangulation vertices never
# sit on a chord cutting across the arc.
#
# Run this file to regenerate `docs/resample_comparison.png`.

# %%
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from topologize import inflate, triangulate

# %% [markdown]
# ## A tightly-curved input
#
# A single circular arc (radius 60) inflated by 30 gives an inner wall of
# radius 30 and an outer wall of radius 90 — the outer wall is three times
# longer, so a naive offset over-samples it.

# %%
R, BUF = 60.0, 30.0
a = np.linspace(-0.95, 0.95, 160)
arc = np.column_stack([R * np.cos(a), R * np.sin(a)])
curves = [arc]


# %%
def draw(ax, title, **kw):
    tris = triangulate(curves, BUF, boundary_simplification=0.0, **kw)
    xs, ys = [], []
    for p, q, r in tris:
        for u, v in [(p, q), (q, r), (r, p)]:
            xs += [u[0], v[0], None]
            ys += [u[1], v[1], None]
    ax.plot(xs, ys, color="#8fc0e8", lw=0.5, zorder=1)

    for outer, holes in inflate(curves, BUF):
        for ring in [outer, *holes]:
            ring = np.vstack([ring, ring[:1]])
            ax.plot(ring[:, 0], ring[:, 1], color="0.35", ls=(0, (1, 1)), lw=1.0, zorder=2)

    verts = np.array(sorted({(round(p[0], 4), round(p[1], 4)) for t in tris for p in t}))
    ax.scatter(verts[:, 0], verts[:, 1], s=9, c="#d6336c", zorder=3)
    ax.set_title(f"{title}  ({len(verts)} boundary vertices)")
    ax.set_aspect("equal")
    ax.axis("off")


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 6))
draw(ax0, "resample off (subdivide)")
draw(ax1, "resample = 18", resample=18.0)
fig.suptitle("Curvature-adaptive boundary resampling — even density, vertices on the offset",
             fontsize=12)
fig.tight_layout()

out = Path(__file__).parent.parent.parent / "docs" / "resample_comparison.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"wrote {out}")
