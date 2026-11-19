#!/usr/bin/env python3
"""POD singular-value decay and the truncation error eps_n for Experiments A and B.

Uses the same well-level split as analysis/pod_vs_ae_eta.py (well_split):
rng = default_rng(seed); permutation of the well IDs; first
int(n_wells * held_out_frac) wells are held out. Only training snapshots enter
the SVD.

Saves to --npz (default singular_values.npz), for each analysed condition X:
  sigma_raw_X  : top n_sv singular values of the centred training matrix (unnormalized)
  total_sq_X   : ||X_tr - mean||_F^2 over ALL training snapshots (the denominator of eq. epsn)
  sigma_norm_X : sigma_raw_X / sigma_raw_X[0]
  eps_X        : epsilon_n for n = 8, 16, 32, 64, from eq. epsn
and prints eps_n and sigma_n/sigma_1 at those n.

Usage:
  python3 plot_singular_values.py --dir-A <DATA_ROOT>/ldo_data_v3_fixedwell \
      --dir-B <DATA_ROOT>/ldo_data_v3_reconfirm --seed 0 --out singular_values.pdf
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NX, NY, NZ_TOTAL = 40, 40, 20
N_CELLS = NX * NY * NZ_TOTAL
N_REPORT = (8, 16, 32, 64)


def train_matrix(data_dir, max_samples, held_out_frac, seed):
    files = sorted(Path(data_dir).glob("sample_*_p.npy"))
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise SystemExit(f"no sample_*_p.npy files in {data_dir}")

    lengths = [np.load(f, mmap_mode="r").shape[0] for f in files]
    well_id = np.repeat(np.arange(len(files)), lengths)

    # identical to well_split() in pod_vs_ae_eta.py
    rng = np.random.default_rng(seed)
    n_wells = len(files)
    perm = rng.permutation(n_wells)
    n_held = int(n_wells * held_out_frac)
    held = set(perm[:n_held].tolist())
    is_test = np.isin(well_id, list(held))

    n_train = int((~is_test).sum())
    X = np.empty((n_train, N_CELLS), dtype=np.float32)
    row, start = 0, 0
    for f, n in zip(files, lengths):
        p = np.load(f)
        keep = ~is_test[start:start + n]
        block = p[keep].astype(np.float32)
        X[row:row + block.shape[0]] = block
        row += block.shape[0]
        start += n
    assert row == n_train, (row, n_train)
    print(f"  {n_wells} wells, {len(held)} held out, {n_train} training snapshots",
          flush=True)
    return X


def analyse(X, n_sv, seed):
    mean = X.mean(axis=0, keepdims=True)
    X -= mean                                   # centre in place
    total_sq = 0.0
    for i in range(0, X.shape[0], 5000):         # chunked, float64 accumulation
        blk = X[i:i + 5000].astype(np.float64)
        total_sq += float(np.sum(blk * blk))
    _, S, _ = randomized_svd(X, n_components=n_sv, random_state=seed)
    eps = {n: float(np.sqrt(max(0.0, 1.0 - np.sum(S[:n] ** 2) / total_sq)))
           for n in N_REPORT}
    return S, total_sq, eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-A", required=True)
    ap.add_argument("--dir-B", default=None,
                    help="omit to analyse only --dir-A (e.g. one dose level)")
    ap.add_argument("--npz", default="singular_values.npz")
    ap.add_argument("--n-sv", type=int, default=64)
    ap.add_argument("--held-out-frac", type=float, default=0.15)
    ap.add_argument("--max-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="singular_values.pdf")
    args = ap.parse_args()

    conds = [("A", args.dir_A)] + ([("B", args.dir_B)] if args.dir_B else [])
    out = {}
    for label, d in conds:
        print(f"Experiment {label}: {d}", flush=True)
        X = train_matrix(d, args.max_samples, args.held_out_frac, args.seed)
        print("  computing SVD ...", flush=True)
        S, total_sq, eps = analyse(X, args.n_sv, args.seed)
        del X
        out[label] = dict(sigma_raw=S, total_sq=total_sq, sigma_norm=S / S[0], eps=eps)

    save = {}
    for label, o in out.items():
        save[f"sigma_raw_{label}"] = o["sigma_raw"]
        save[f"total_sq_{label}"] = o["total_sq"]
        save[f"sigma_norm_{label}"] = o["sigma_norm"]
        save[f"eps_{label}"] = np.array([o["eps"][n] for n in N_REPORT])
    save["n_report"] = np.array(N_REPORT)
    np.savez(args.npz, **save)

    sA = out["A"]["sigma_norm"]
    idx = np.arange(1, len(sA) + 1)
    plt.figure(figsize=(5, 4))
    plt.semilogy(idx, sA, "o-", label="Experiment A (fixed well)", markersize=4)
    if "B" in out:
        plt.semilogy(idx, out["B"]["sigma_norm"], "s-",
                     label="Experiment B (variable well)", markersize=4)
    plt.xlabel("index i")
    plt.ylabel("normalized singular value")
    plt.title("POD singular-value decay")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Saved -> " + args.out + " and " + args.npz, flush=True)

    for label in out:
        o = out[label]
        for n in N_REPORT:
            print(f"Exp {label}: n={n:2d}  eps_n={o['eps'][n]:.4f}  "
                  f"sigma_n/sigma_1={o['sigma_norm'][n-1]:.4e}", flush=True)
        print(f"Exp {label}: total_sq={o['total_sq']:.6e}", flush=True)


if __name__ == "__main__":
    main()
