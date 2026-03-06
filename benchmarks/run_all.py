"""Run all benchmarks and print results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import bench_quality
import bench_timing


print("=== OUTPUT QUALITY / COMPLEXITY ===")
print()
rows = bench_quality.run()
HDR = f"{'dataset':<8}  {'buf':>6}  {'chains':>6}  {'total_pts':>9}  {'mean_len':>8}  {'max_len':>7}"
SEP = "-" * len(HDR)
print(HDR)
print(SEP)
prev_buf = None
for r in rows:
    if prev_buf and r["buf"] != prev_buf:
        print()
    prev_buf = r["buf"]
    print(
        f"{r['dataset']:<8}  {r['buf']:>6.2f}  {r['chains']:>6d}  "
        f"{r['total_pts']:>9d}  {r['mean_len']:>8.1f}  {r['max_len']:>7d}"
    )
print(SEP)


print()
print()
print("=== RAW CPU TIMINGS ===")
print()
rows = bench_timing.run()
HDR = f"{'dataset':<8}  {'buf':>6}  {'min_ms':>8}  {'mean_ms':>8}  {'max_ms':>8}"
SEP = "-" * len(HDR)
print(HDR)
print(SEP)
prev_buf = None
for r in rows:
    if prev_buf and r["buf"] != prev_buf:
        print()
    prev_buf = r["buf"]
    print(
        f"{r['dataset']:<8}  {r['buf']:>6.2f}  "
        f"{r['min_ms']:>8.1f}  {r['mean_ms']:>8.1f}  {r['max_ms']:>8.1f}"
    )
print(SEP)
