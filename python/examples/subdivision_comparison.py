# %% [markdown]
# # Curvature-adaptive subdivision comparison
#
# Demonstrates the `subdivision_ratio` parameter by running topologize
# on near-identical inputs from adjacent Z-layers and comparing the
# output stability across different settings.

# %%
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))
from topologize import topologize

# %% [markdown]
# ## Load test data
#
# Five near-identical cross-sections from adjacent Z-layers of a boat hull.
# The inputs differ only by ~4mm in slicing height.

# %%
data = json.load(open(Path(__file__).parent / "data" / "topo_variability_input.json"))
layers = data["layers"]
print(f"{len(layers)} layers, Z = {[lay['z'] for lay in layers]}")

# %% [markdown]
# ## Compare subdivision settings
#
# - **off**: `subdivision_ratio=100` effectively minimizes curvature refinement
# - **default**: no overrides — uses `ratio=0.5` (at 90°, max seg = 0.5×bd)
# - **tight**: `subdivision_ratio=0.3` — stronger refinement at corners

# %%
settings = [
    ("off (ratio=100)", {"subdivision_ratio": 100}),
    ("default (ratio=0.5)", {}),
    ("tight (ratio=0.3)", {"subdivision_ratio": 0.3}),
]

LAYER_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00", "#984ea3"]

columns = [("input", None)] + settings
n_cols = len(columns)

fig = make_subplots(
    rows=1, cols=n_cols,
    subplot_titles=[c[0] for c in columns],
    shared_xaxes=True, shared_yaxes=True,
    horizontal_spacing=0.02,
)

# First column: raw input curves
for li, layer in enumerate(layers):
    curves = [np.array(c) for c in layer["curves"]]
    color = LAYER_COLORS[li]
    for ci, curve in enumerate(curves):
        fig.add_trace(go.Scatter(
            x=curve[:, 0], y=curve[:, 1], mode="lines+markers",
            line=dict(color=color, width=1.5),
            marker=dict(size=3, color=color),
            legendgroup=f"z{layer['z']}",
            showlegend=(ci == 0),
            name=f"Z={layer['z']}",
            hoverinfo="text",
            text=[f"Z={layer['z']} curve {ci}"] * len(curve),
        ), row=1, col=1)

# Remaining columns: topologize results per setting
for col_idx, (label, kwargs) in enumerate(settings):
    col = col_idx + 2  # offset by 1 for the input column
    for li, layer in enumerate(layers):
        curves = [np.array(c) for c in layer["curves"]]
        p = layer["params"]
        bd = p["inflation_radius"] * 0.5  # halved buffer distance
        r = topologize(
            curves,
            inflation_radius=bd,
            min_tip_length=p["min_tip_length"],
            simplification=p["simplification"],
            **kwargs,
        )
        color = LAYER_COLORS[li]
        for ci, chain in enumerate(r.chains):
            fig.add_trace(go.Scatter(
                x=chain[:, 0], y=chain[:, 1], mode="lines",
                line=dict(color=color, width=2),
                legendgroup=f"z{layer['z']}",
                showlegend=False,
                hoverinfo="text",
                text=[f"Z={layer['z']} chain {ci}"] * len(chain),
            ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=r.nodes[:, 0], y=r.nodes[:, 1], mode="markers",
            marker=dict(size=5, color=color, symbol="circle"),
            legendgroup=f"z{layer['z']}",
            showlegend=False,
            hoverinfo="text",
            text=[
                f"Z={layer['z']} node {j} deg={r.node_degree[j]}"
                for j in range(len(r.nodes))
            ],
        ), row=1, col=col)

for i in range(1, n_cols + 1):
    fig.update_xaxes(scaleanchor=f"y{i}" if i > 1 else "y", row=1, col=i)

fig.update_layout(
    title="Curvature-adaptive subdivision: 5 layers overlaid per setting (bd halved)",
    template="plotly_white",
    width=350 * n_cols,
    height=600,
)
fig.show()

# %% [markdown]
# ## Summary table

# %%
print(f"{'Setting':<30} {'Nodes':<25} {'Chains':<25}")
print("-" * 80)
for label, kwargs in settings:
    nl, cl = [], []
    for layer in layers:
        curves = [np.array(c) for c in layer["curves"]]
        p = layer["params"]
        bd = p["inflation_radius"] * 0.5
        r = topologize(
            curves,
            inflation_radius=bd,
            min_tip_length=p["min_tip_length"],
            simplification=p["simplification"],
            **kwargs,
        )
        nl.append(len(r.nodes))
        cl.append(len(r.chains))
    print(f"{label:<30} {str(nl):<25} {str(cl):<25}")

# %%
