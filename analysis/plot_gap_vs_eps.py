#!/usr/bin/env python3
"""Relative gap against the POD truncation error eps_n, for all 28 (condition, n) pairs.

eps_n (seed 0) comes from analysis/plot_singular_values.py run on each condition:
  A: ldo_data_v3_fixedwell; B: ldo_data_v3_reconfirm; dose levels: ldo_data_v3_r{R}.
gap (mean over seeds, %) comes from the paper's Table (tab:full-ab, tab:full-dose).
Each list is ordered n = 8, 16, 32, 64.

Output: spectrum_gap.pdf (figure fig:diagnostic) and a printed summary:
Spearman correlation, sign-agreement threshold check.
"""
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = {  # seed-0 truncation errors, n = 8, 16, 32, 64
    "A (R=0)":   [0.0237, 0.0186, 0.0145, 0.0107],
    "B":         [0.3152, 0.1971, 0.1114, 0.0553],
    "R/l=0.21":  [0.0719, 0.0253, 0.0154, 0.0111],
    "R/l=0.42":  [0.1121, 0.0518, 0.0230, 0.0113],
    "R/l=0.83":  [0.1858, 0.0770, 0.0413, 0.0215],
    "R/l=1.25":  [0.2254, 0.1097, 0.0488, 0.0297],
    "R/l=2.5":   [0.3097, 0.1872, 0.0994, 0.0437],
}
GAP = {  # mean gap (%) over seeds
    "A (R=0)":   [-2.5, -18.8, -20.3, -41.8],
    "B":         [89.2, 85.1, 76.1, 58.1],
    "R/l=0.21":  [69.6, 25.6, -5.1, -29.1],
    "R/l=0.42":  [80.9, 63.6, 27.0, -30.6],
    "R/l=0.83":  [88.6, 76.0, 60.9, 34.7],
    "R/l=1.25":  [90.4, 84.1, 67.7, 54.9],
    "R/l=2.5":   [92.3, 90.4, 84.7, 70.0],
}
NS = [8, 16, 32, 64]

e = np.array([EPS[k][j] for k in EPS for j in range(4)])
g = np.array([GAP[k][j] for k in GAP for j in range(4)])
rho, p = spearmanr(e, g)
print(f"n points = {len(e)}, Spearman rho = {rho:.3f} (p = {p:.1e})")

for thr in np.arange(0.0200, 0.0245, 0.0005):
    pred_pos = e >= thr
    acc = np.mean(pred_pos == (g > 0))
    print(f"threshold {thr:.4f}: sign agreement {np.sum(pred_pos == (g > 0))}/{len(e)}")

fig, ax = plt.subplots(figsize=(5.2, 4.0))
markers = {"A (R=0)": "o", "B": "s"}
for k in EPS:
    m = markers.get(k, "^")
    filled = k != "B"
    ax.plot(EPS[k], GAP[k], marker=m, linestyle="-", markersize=5,
            markerfacecolor=None if not filled else None,
            fillstyle="full" if filled else "none", label=k)
ax.axhline(0, color="k", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"POD truncation error $\varepsilon_n$ (training snapshots)")
ax.set_ylabel("relative gap (%)  [positive favours AE]")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7, ncol=2, frameon=False)
fig.tight_layout()
fig.savefig("spectrum_gap.pdf")
print("saved spectrum_gap.pdf")
