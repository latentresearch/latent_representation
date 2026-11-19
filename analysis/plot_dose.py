"""Dose-response analysis for Experiment C.

Combines Experiment A (R = 0, from the multi-seed run) with the five dose
levels, and optionally marks the original Experiment B for reference.

Outputs (in --out):
  dose_gap.pdf/png    relative gap vs R/ell, one curve per latent size n
  dose_eta.pdf/png    eta_POD and eta_AE vs R/ell (log scale), per n
  dose_summary.csv    mean and std per (R, n)
and prints the crossover R/ell for each n (linear interpolation of the mean gap).

Usage:
  python3 plot_dose.py --seeds ~/seeds_500ep --dose /ddnB/work/jing/logs/dose \
                       --ell 1200 --out ~/figs_dose
"""
import argparse, glob, json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_dir(directory):
    rows = []
    for path in glob.glob(os.path.join(os.path.expanduser(directory), "results_*_seed*.json")):
        m = re.search(r"results_(\w+?)_seed(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        d = json.load(open(path))
        for r in d["results"]:
            gap = r.get("gap_pct")
            if gap is None:
                gap = 100 * (r["eta_pod"] - r["eta_ae"]) / r["eta_pod"]
            rows.append(dict(exp=m.group(1), seed=int(m.group(2)), n=int(r["n"]),
                             eta_pod=r["eta_pod"], eta_ae=r["eta_ae"], gap=gap))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="dir with results_A/B_seed*.json")
    ap.add_argument("--dose", required=True, help="dir with results_r*_seed*.json")
    ap.add_argument("--ell", type=float, default=1200.0, help="drawdown radius (m)")
    ap.add_argument("--out", default="figs_dose")
    a = ap.parse_args()
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    out = os.path.expanduser(a.out)

    df = pd.concat([load_dir(a.seeds), load_dir(a.dose)], ignore_index=True)
    df["R"] = df["exp"].map(lambda e: 0.0 if e == "A" else
                            (float(e[1:]) if e.startswith("r") else np.nan))
    base = df[df["exp"] != "B"].copy()           # A + dose levels
    origB = df[df["exp"] == "B"].copy()           # original Experiment B (reference)

    agg = (base.groupby(["R", "n"])
               .agg(gap_mean=("gap", "mean"), gap_std=("gap", "std"),
                    pod_mean=("eta_pod", "mean"), pod_std=("eta_pod", "std"),
                    ae_mean=("eta_ae", "mean"), ae_std=("eta_ae", "std"),
                    seeds=("seed", "nunique"))
               .reset_index())
    agg["R_over_ell"] = agg["R"] / a.ell
    agg.to_csv(os.path.join(out, "dose_summary.csv"), index=False)

    ns = sorted(agg["n"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(ns)))

    # ---- figure 1: gap vs R/ell -------------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for c, n in zip(colors, ns):
        g = agg[agg["n"] == n].sort_values("R")
        ax.errorbar(g["R_over_ell"], g["gap_mean"], yerr=g["gap_std"], marker="o",
                    ms=4, capsize=2, color=c, label=f"$n={n}$")
        if len(origB):
            b = origB[origB["n"] == n]["gap"]
            ax.plot([3000 / a.ell], [b.mean()], marker="x", ms=7, color=c, ls="none")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"translation range $R/\ell$")
    ax.set_ylabel(r"relative gap $(\eta_{\mathrm{POD}}-\eta_{\mathrm{AE}})/\eta_{\mathrm{POD}}$ (%)")
    ax.legend(frameon=False, fontsize=8)
    if len(origB):
        ax.text(0.98, 0.04, "x: original Experiment B", transform=ax.transAxes,
                ha="right", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"dose_gap.{ext}"), dpi=200)

    # ---- figure 2: eta_POD and eta_AE vs R/ell -----------------------------
    fig, axes = plt.subplots(1, len(ns), figsize=(2.4 * len(ns), 2.8), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), ns):
        g = agg[agg["n"] == n].sort_values("R")
        ax.errorbar(g["R_over_ell"], g["pod_mean"], yerr=g["pod_std"], marker="s",
                    ms=3, capsize=2, label="POD")
        ax.errorbar(g["R_over_ell"], g["ae_mean"], yerr=g["ae_std"], marker="o",
                    ms=3, capsize=2, label="AE")
        ax.set_yscale("log")
        ax.set_title(f"$n={n}$", fontsize=9)
        ax.set_xlabel(r"$R/\ell$")
    np.atleast_1d(axes)[0].set_ylabel(r"$\eta$ (held-out)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"dose_eta.{ext}"), dpi=200)

    # ---- printed summary ----------------------------------------------------
    pd.set_option("display.width", 120)
    print(agg[["R", "R_over_ell", "n", "gap_mean", "gap_std", "pod_mean", "ae_mean", "seeds"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nGrowth of eta from R=0 to the largest R (ratio of means):")
    Rmax = agg["R"].max()
    for n in ns:
        g0 = agg[(agg["n"] == n) & (agg["R"] == 0)]
        g1 = agg[(agg["n"] == n) & (agg["R"] == Rmax)]
        if len(g0) and len(g1):
            print(f"  n={n:>2}: POD x{g1.pod_mean.iloc[0] / g0.pod_mean.iloc[0]:.1f}, "
                  f"AE x{g1.ae_mean.iloc[0] / g0.ae_mean.iloc[0]:.1f}")
    print("\nCrossover R/ell (first sign change of the mean gap, linear interpolation):")
    for n in ns:
        g = agg[agg["n"] == n].sort_values("R")
        x, y = g["R_over_ell"].values, g["gap_mean"].values
        k = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
        if len(k):
            i = k[0]
            xc = x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            print(f"  n={n:>2}: between {x[i]:.2f} and {x[i+1]:.2f}  (~{xc:.2f})")
        else:
            print(f"  n={n:>2}: no sign change in the tested range")
        mono = np.all(np.diff(y) > 0)
        print(f"         gap monotone increasing in R: {mono}")
    print(f"\nfigures and dose_summary.csv written to {out}")


if __name__ == "__main__":
    main()
