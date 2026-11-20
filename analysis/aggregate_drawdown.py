#!/usr/bin/env python3
"""Summarise total and drawdown-relative errors across seeds (Experiments A and B).

Reads results_{A,B}_seed{s}.json from --dir (default: logs/seeds_drawdown) written by
pod_vs_ae_eta_dd.py, and prints, for each condition and n:
  mean +- SD (ddof=1) of eta_POD, eta_AE, gap, eta_POD_drawdown, eta_AE_drawdown
plus a check that gap computed from the drawdown errors equals the total-error gap.

Usage: python3 aggregate_drawdown.py <DATA_ROOT>/logs/seeds_drawdown
"""
import argparse, glob, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
args = ap.parse_args()

rows = {}  # (exp, n) -> list of dicts
for f in sorted(glob.glob(os.path.join(args.dir, "results_*_seed*.json"))):
    m = re.search(r"results_([AB])_seed(\d+)\.json$", f)
    if not m:
        continue
    exp = m.group(1)
    d = json.load(open(f))
    for r in d["results"]:
        if "eta_pod_drawdown" not in r:
            raise SystemExit(f"{f}: no drawdown fields -- wrong script version?")
        rows.setdefault((exp, r["n"]), []).append(r)

def ms(x):
    x = np.asarray(x, float)
    return x.mean(), (x.std(ddof=1) if len(x) > 1 else 0.0)

max_mismatch = 0.0
print(f"{'exp':>3} {'n':>3} {'k':>2} {'eta_POD':>22} {'eta_AE':>22} {'gap %':>16} "
      f"{'eta_POD_dd':>22} {'eta_AE_dd':>22}")
for (exp, n) in sorted(rows, key=lambda t: (t[0], t[1])):
    R = rows[(exp, n)]
    gap_from_dd = [100 * (r["eta_pod_drawdown"] - r["eta_ae_drawdown"]) / r["eta_pod_drawdown"] for r in R]
    for a, b in zip(gap_from_dd, [r["gap_pct"] for r in R]):
        max_mismatch = max(max_mismatch, abs(a - b))
    cols = [ms([r[k] for r in R]) for k in ("eta_pod", "eta_ae", "gap_pct",
                                              "eta_pod_drawdown", "eta_ae_drawdown")]
    fmt = lambda c, s=1e3: f"{c[0]*s:9.4f} +- {c[1]*s:7.4f}"
    print(f"{exp:>3} {n:>3} {len(R):>2} {fmt(cols[0])} {fmt(cols[1])} "
          f"{cols[2][0]:8.1f} +- {cols[2][1]:5.1f} {fmt(cols[3])} {fmt(cols[4])}")

print(f"\nmax |gap_dd - gap_total| over all rows = {max_mismatch:.2e} percentage points")
print("(eta values shown x1e3; drawdown errors use x0 = P_RES_KIN, same for both models)")
