#!/usr/bin/env python3
"""Classify each AE fit in a set of job logs as early-stopped or cap-reached.

Usage:
  python3 -I check_convergence.py <logs_dir> <cap> [glob]
  e.g. python3 -I check_convergence.py <DATA_ROOT>/logs 500 "*_podae_seeds.out"
       python3 -I check_convergence.py <DATA_ROOT>/logs 1500 "*_cap1500.out"

Best epoch:
  - Early-stopped fits: the log line "early stop at epoch S (no val improvement in P epochs)"
    gives the exact best epoch as S - P. The best@ values printed every 50 epochs are
    stale for these fits, so they are not used.
  - Cap-reached fits: the last printed line is at the cap, so its best@ is exact.
"""
import re, sys, pathlib

logdir = pathlib.Path(sys.argv[1])
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 500
PATTERN = sys.argv[3] if len(sys.argv) > 3 else "*_podae_seeds.out"

rows = []
for f in sorted(logdir.glob(PATTERN)):
    text = f.read_text(errors="replace").splitlines()
    if not text:
        continue
    m = re.search(r"exp=(\w) seed=(\d+)", text[0])
    if not m:
        continue
    exp, seed = m.group(1), int(m.group(2))
    cur_n = None
    state = {}
    for line in text:
        mn = re.search(r"\bn=\s*(\d+)\b", line)
        if mn:
            cur_n = int(mn.group(1))
            state.setdefault(cur_n, {"last_epoch": 0, "best_printed": 0,
                                     "stop": None, "patience": None})
        if cur_n is None:
            continue
        s = state[cur_n]
        me = re.search(r"epoch (\d+)/", line)
        if me:
            s["last_epoch"] = int(me.group(1))
        mb = re.search(r"best@(\d+)", line)
        if mb:
            s["best_printed"] = max(s["best_printed"], int(mb.group(1)))
        ms = re.search(r"early stop at epoch (\d+) \(no val improvement in (\d+) epochs\)", line)
        if ms:
            s["stop"] = int(ms.group(1))
            s["patience"] = int(ms.group(2))
    for n, s in sorted(state.items()):
        if s["stop"] is not None:
            status, last = "early-stop", s["stop"]
            best = s["stop"] - s["patience"]          # exact: last improvement
        else:
            status = "CAP" if s["last_epoch"] >= CAP else "unknown"
            last = s["last_epoch"]
            best = s["best_printed"]                  # exact at the cap line
        rows.append((exp, seed, n, status, last, best, last - best))

print(f"{'exp':<4}{'seed':<6}{'n':<5}{'status':<12}{'last_ep':<9}{'best_ep':<9}{'gap_ep':<7}")
for r in rows:
    print(f"{r[0]:<4}{r[1]:<6}{r[2]:<5}{r[3]:<12}{r[4]:<9}{r[5]:<9}{r[6]:<7}")
n_cap = sum(1 for r in rows if r[3] == "CAP")
n_stop = sum(1 for r in rows if r[3] == "early-stop")
print(f"\nfits: {len(rows)}  early-stop: {n_stop}  cap-reached: {n_cap}")
near = [r for r in rows if r[3] == "CAP" and r[6] <= 50]
print(f"cap-reached with best within 50 epochs of the cap: {len(near)}")
if rows and n_stop:
    bests = [r[5] for r in rows if r[3] == "early-stop"]
    print(f"early-stop best_ep range: {min(bests)} to {max(bests)}")
