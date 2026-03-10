# %% [markdown]
# # Variable-width demo
#
# Demonstrates per-vertex buffer widths and chain width estimation.
# Each input curve tapers from thick to thin, and the output chains
# carry estimated bead widths that can be visualized with `result.plot()`.

# %%
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from topologize import topologize

# %% [markdown]
# ## Create synthetic curves with variable widths

# %%
# Two crossing lines
t = np.linspace(0, 10, 30)
curve_a = np.column_stack([t, np.sin(t) * 2])
curve_b = np.column_stack([t, -np.sin(t**2/10) * 2 + 2])

# Per-vertex widths: taper from thick (1.5) to thin (0.3)
widths_a = np.linspace(0.3, 0.1, len(curve_a))
widths_b = np.linspace(0.1, 0.5, len(curve_b))

# %%
result = topologize(
    [curve_a, curve_b],
    per_curve_widths=[widths_a, widths_b],
    compute_widths=True,
)

print(f"{len(result.chains)} chains, widths populated: {result.chain_widths is not None}")

# %%
result.plot(
    [curve_a, curve_b],
    per_curve_widths=[widths_a, widths_b],
    show_triangulation=True,
    title="Variable-width tapering curves",
)


# %%