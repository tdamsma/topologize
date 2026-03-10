# %% [markdown]
# # Parallel processing
#
# Both `topologize` and `topologize_batch` release the GIL, enabling real
# parallelism from Python. Two options:
#
# - **`topologize_batch`** — best when all jobs are available upfront (e.g.
#   processing layers of a sliced model). Single call, Rayon work-stealing,
#   minimal Python overhead.
# - **`ThreadPoolExecutor` + `topologize`** — best when jobs arrive
#   incrementally (e.g. from a queue or generator). Each thread calls single
#   `topologize`; the GIL release lets them run in parallel.
#
# Both achieve similar speedups over a sequential loop.

# %%
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from topologize import TopologizeJob, topologize, topologize_batch

# %% Generate 100 random-walk curve-sets
rng = np.random.default_rng(42)
curve_sets = [
    [np.cumsum(rng.uniform(-50, 50, (100, 2)), axis=0) for _ in range(rng.integers(2, 5))]
    for _ in range(200)
]
bd = 2.0

# Warm up (initialises Rayon thread pool)
topologize_batch([TopologizeJob(curve_sets[0], bd)])

# %% Sequential
t0 = time.perf_counter()
[topologize(cs, buffer_distance=bd) for cs in curve_sets]
t_seq = time.perf_counter() - t0

# %% ThreadPoolExecutor (GIL released in single topologize)
t0 = time.perf_counter()
with ThreadPoolExecutor() as pool:
    list(pool.map(lambda cs: topologize(cs, buffer_distance=bd), curve_sets))
t_threads = time.perf_counter() - t0

# %% topologize_batch (Rayon parallel)
t0 = time.perf_counter()
topologize_batch([TopologizeJob(cs, bd) for cs in curve_sets])
t_batch = time.perf_counter() - t0

# %% Summary
print(f"{'Method':<25} {'Time':>8} {'Speedup':>8}")
print("-" * 43)
print(f"{'Sequential loop':<25} {t_seq:>7.3f}s {1.0:>7.1f}x")
print(f"{'ThreadPoolExecutor':<25} {t_threads:>7.3f}s {t_seq / t_threads:>7.1f}x")
print(f"{'topologize_batch':<25} {t_batch:>7.3f}s {t_seq / t_batch:>7.1f}x")

# %% show result of one job with plotly
import plotly.graph_objects as go  # noqa: E402
cs = curve_sets[0]
result = topologize(cs, buffer_distance=bd)
fig = go.Figure()
for c in cs:
    fig.add_trace(go.Scatter(x=c[:, 0], y=c[:, 1], mode="lines", line=dict(color="lightgray", width=6), name="input"))
for chain in result.chains:
    fig.add_trace(go.Scatter(x=chain[:, 0], y=chain[:, 1], mode="lines", name="chain"))
fig.update_layout(title="Example curve-set and its topology", yaxis_scaleanchor="x")
fig.show()

# %%