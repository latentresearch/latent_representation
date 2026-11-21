#!/usr/bin/env python3
"""Compare 500-epoch and 1500-epoch AE fits for the capped cases.

Usage:
  python3 -I compare_caps.py --seeds-dir <DATA_ROOT>/logs/seeds --cap-dir <DATA_ROOT>/logs/cap1500

Prints (1) per-fit gaps at 500 and 1500 epochs, and (2) a mixed table: 1500-epoch
results where a fit was capped, 500-epoch results elsewhere. The mixed table
gives mean, SD, and the sign verdict (AE / POD / tie) per (experiment, n).
The POD error for capped fits is taken from the 1500-epoch run, so the two
gaps use the same denominator.
"""
import argparse, glob, json, os, re
import numpy as np


def load(directory, pattern, rx):
    out = {}
    for p in glob.glob(os.path.join(directory, pattern)):
        m = re.search(rx, os.path.basename(p))
        if not m:
            continue
        j = json.load(open(p))
        out[(m.group(1), int(m.group(2)))] = {
            r["n"]: (r["eta_pod"], r["eta_ae"]) for r in j["results"]}
    return out


def gap(pod, ae):
    return 100.0 * (pod - ae) / pod


def verdict(g):
    if np.all(g > 0):
        return "AE"
    if np.all(g < 0):
        return "POD"
    return "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-dir", required=True)
    ap.add_argument("--cap-dir", required=True)
    a = ap.parse_args()

    base = load(a.seeds_dir, "results_*_seed*.json", r"results_(\w)_seed(\d+)\.json$")
    cap = load(a.cap_dir, "results_*_seed*_cap1500.json", r"results_(\w)_seed(\d+)_cap1500\.json$")

    print("Capped fits: AE gap at 500 vs 1500 epochs (eta_POD from the 1500-epoch run)")
    print(f"{'exp':<4}{'seed':<6}{'n':<5}{'gap_500':>9}{'gap_1500':>10}{'change':>9}")
    for key in sorted(cap):
        for n, (pod1500, ae1500) in sorted(cap[key].items()):
            if key not in base or n not in base[key]:
                print(f"{key[0]:<4}{key[1]:<6}{n:<5}  missing 500-epoch result")
                continue
            ae500 = base[key][n][1]
            g500, g1500 = gap(pod1500, ae500), gap(pod1500, ae1500)
            print(f"{key[0]:<4}{key[1]:<6}{n:<5}{g500:9.1f}{g1500:10.1f}{g1500 - g500:9.1f}")

    print("\nPer (experiment, n): 500-epoch only vs mixed (1500 where capped)")
    print(f"{'exp':<4}{'n':<5}{'500 mean±SD':>18}{'500 verdict':>13}{'mixed mean±SD':>20}{'mixed verdict':>15}")
    for exp in ("A", "B"):
        for n in (8, 16, 32, 64):
            g500, gmix = [], []
            for seed in range(5):
                key = (exp, seed)
                if key not in base or n not in base[key]:
                    continue
                pod, ae = base[key][n]
                g500.append(gap(pod, ae))
                if key in cap and n in cap[key]:
                    pod, ae = cap[key][n]
                gmix.append(gap(pod, ae))
            if len(g500) < 2:
                continue
            g5, gm = np.array(g500), np.array(gmix)
            print(f"{exp:<4}{n:<5}{g5.mean():>10.1f}±{g5.std(ddof=1):<6.1f}"
                  f"{verdict(g5):>13}{gm.mean():>12.1f}±{gm.std(ddof=1):<6.1f}"
                  f"{verdict(gm):>15}")


if __name__ == "__main__":
    main()
