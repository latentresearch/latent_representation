"""Aggregate multi-seed POD vs AE results.

Reads the JSON files written by pod_vs_ae_eta.py, named
results_{A|B}_seed{s}.json, each containing
  {"results": [{"n":..., "eta_pod":..., "eta_ae":...}, ...],
   "ae_diagnostics": {"8": {"best_epoch":..., "n_epochs_run":...}, ...}}

Usage:  python3 aggregate_seeds.py --dir <DATA_ROOT>/logs/seeds --out table_seeds.tex
"""
import json
import argparse, glob, os, re
import numpy as np
import pandas as pd


def load(directory):
    rows = []
    for path in glob.glob(os.path.join(directory, "results_*_seed*.json")):
        m = re.search(r"results_(\w+?)_seed(\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        d = json.load(open(path))
        df = pd.DataFrame(d["results"])
        if "gap_pct" not in df:
            df["gap_pct"] = float("nan")
        df = df[["n", "eta_pod", "eta_ae", "gap_pct"]]
        diag = d.get("ae_diagnostics", {})
        df["best_epoch"] = [diag.get(str(n), {}).get("best_epoch") for n in df["n"]]
        df["epochs_run"] = [diag.get(str(n), {}).get("n_epochs_run") for n in df["n"]]
        df["exp"], df["seed"] = m.group(1), int(m.group(2))
        rows.append(df)
    if not rows:
        raise SystemExit(f"No result files found in {directory}")
    df = pd.concat(rows, ignore_index=True)
    # paired relative gap per seed: positive favors AE, negative favors POD.
    # Prefer the gap stored by pod_vs_ae_eta.py (computed from unrounded etas).
    recomputed = (df["eta_pod"] - df["eta_ae"]) / df["eta_pod"]
    df["gap"] = (df["gap_pct"] / 100.0).fillna(recomputed)
    return df


def summarize(df):
    out = []
    for (exp, n), g in df.groupby(["exp", "n"]):
        k = len(g)
        sign = np.sign(g["gap"].mean())
        consistent = int((np.sign(g["gap"]) == sign).sum())
        out.append(dict(
            exp=exp, n=int(n), seeds=k,
            pod_mean=g["eta_pod"].mean(), pod_std=g["eta_pod"].std(ddof=1),
            ae_mean=g["eta_ae"].mean(), ae_std=g["eta_ae"].std(ddof=1),
            gap_mean=100 * g["gap"].mean(), gap_std=100 * g["gap"].std(ddof=1),
            consistent=consistent,
            # pre-registered rule: a "win" requires the same sign in every seed
            verdict=("tie" if consistent < k else ("AE" if sign > 0 else "POD")),
        ))
    return pd.DataFrame(out).sort_values(["exp", "n"])


def pm(m, s, fmt):
    return f"${m:{fmt}} \\pm {s:{fmt}}$"


def to_latex(summary):
    lines = [
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Exp. & $n$ & $\eta_{\mathrm{POD}}$ & $\eta_{\mathrm{AE}}$ & gap (\%) & sign \\",
        r"\midrule",
    ]
    for exp, block in summary.groupby("exp"):
        for _, r in block.iterrows():
            lines.append(
                f"{exp} & {r.n} & {pm(r.pod_mean, r.pod_std, '.5f')} & "
                f"{pm(r.ae_mean, r.ae_std, '.5f')} & "
                f"{pm(r.gap_mean, r.gap_std, '.1f')} & "
                f"{r.consistent}/{r.seeds} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--out", default="table_seeds.tex")
    ap.add_argument("--patience", type=int, default=20)
    a = ap.parse_args()
    df = load(a.dir)
    # convergence check: with early stopping (patience P), a run that stopped on its
    # own has best_epoch == epochs_run - P; a smaller gap means the epoch cap ended it.
    late = df[(df["epochs_run"] - df["best_epoch"]) < a.patience]
    if len(late):
        print(f"WARNING: training ended by the epoch cap, not early stopping "
              f"(epochs_run - best_epoch < {a.patience}); AE may be undertrained:")
        print(late[["exp", "seed", "n", "best_epoch", "epochs_run"]].to_string(index=False))
        print()
    summary = summarize(df)
    print(summary[["exp", "n", "gap_mean", "gap_std", "consistent", "seeds", "verdict"]]
          .to_string(index=False))
    with open(a.out, "w") as f:
        f.write(to_latex(summary) + "\n")
    print(f"\nLaTeX table written to {a.out}")
