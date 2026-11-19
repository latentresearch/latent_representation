"""Rebuild the full per-seed error tables of Chapter 3 / the paper.

Experiments A and B (500 epochs): the result JSONs were not kept, so eta values are read
from the job logs logs/1136521_{0..9}_podae_seeds.out (tasks 0-4: A seeds 0-4; 5-9: B).
Experiment C: read from the JSONs in logs/dose.
Outputs: tables/table_500ep.tex, tables/table_dose.tex, and logs/seeds/results_{A,B}_seed*.json
(same structure as the dose JSONs, so that plot_dose.py works). The extracted gaps are checked
against the published means before anything is written.
"""
import copy, glob, json, os, re, sys
import numpy as np

W = "/work/jing"
NS = (8, 16, 32, 64)
LEVELS = (250, 500, 1000, 1500, 3000)
ELL = 1200.0
PUB_AB = {"A": (-2.5, -18.8, -20.3, -41.9), "B": (89.2, 85.1, 76.1, 58.1)}
PUB_DOSE = {250: (69.6, 25.6, -5.1, -29.1), 500: (80.9, 63.6, 27.0, -30.6),
            1000: (88.6, 76.0, 60.9, 34.7), 1500: (90.4, 84.1, 67.7, 54.9),
            3000: (92.3, 90.4, 84.7, 70.0)}

# ---------- generic access to the dose JSON structure ----------
def find_eta(obj):
    """Return {n: (eta_pod, eta_ae, path_pod, path_ae)} from an arbitrary result JSON."""
    found = {}
    def walk(o, path, n_ctx):
        if isinstance(o, dict):
            n_here = n_ctx
            for k in ("n", "latent_dim", "latent", "n_latent"):
                if k in o and isinstance(o[k], (int, float)) and int(o[k]) in NS:
                    n_here = int(o[k])
            for k, v in o.items():
                nk = n_here
                m = re.fullmatch(r"(?:n[=_]?)?(\d+)", str(k))
                if m and int(m.group(1)) in NS:
                    nk = int(m.group(1))
                walk(v, path + (k,), nk)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, path + (i,), n_ctx)
        elif isinstance(o, (int, float)) and not isinstance(o, bool) and n_ctx is not None and path:
            kind = None
            for k in reversed(path):
                k = str(k).lower()
                if any(t in k for t in ("gap", "time", "epoch", "param", "loss", "sec")):
                    break
                if "pod" in k:
                    kind = "pod"; break
                if re.search(r"(^|_)ae($|_)", k) or k == "eta_ae":
                    kind = "ae"; break
            if kind:
                d = found.setdefault(n_ctx, {})
                d.setdefault(kind, (float(o), path))
    walk(obj, (), None)
    out = {}
    for n, d in found.items():
        if "pod" in d and "ae" in d:
            out[n] = (d["pod"][0], d["ae"][0], d["pod"][1], d["ae"][1])
    return out

def find_gap(obj):
    """Return {n: path} of the relative-gap fields in a result JSON."""
    found = {}
    def walk(o, path, n_ctx):
        if isinstance(o, dict):
            n_here = n_ctx
            for k in ("n", "latent_dim", "latent", "n_latent"):
                if k in o and isinstance(o[k], (int, float)) and int(o[k]) in NS:
                    n_here = int(o[k])
            for k, v in o.items():
                nk = n_here
                m = re.fullmatch(r"(?:n[=_]?)?(\d+)", str(k))
                if m and int(m.group(1)) in NS:
                    nk = int(m.group(1))
                walk(v, path + (k,), nk)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, path + (i,), n_ctx)
        elif isinstance(o, (int, float)) and not isinstance(o, bool) and n_ctx is not None and path:
            if any("gap" in str(k).lower() for k in path):
                found.setdefault(n_ctx, []).append((path, float(o)))
    walk(obj, (), None)
    return found

def set_path(obj, path, value):
    for k in path[:-1]:
        obj = obj[k]
    obj[path[-1]] = value

# ---------- Experiment C from the JSONs ----------
dose = {}
template = None
for R in LEVELS:
    for s in (0, 1, 2):
        p = f"{W}/logs/dose/results_r{R}_seed{s}.json"
        d = json.load(open(p))
        e = find_eta(d)
        if set(e) != set(NS):
            sys.exit(f"cannot locate eta values in {p}; found n={sorted(e)}.\n"
                     f"Structure: {json.dumps(d)[:800]}")
        dose[(R, s)] = {n: e[n][:2] for n in NS}
        if template is None:
            template, tpaths = d, {n: e[n][2:] for n in NS}
            gpaths = {}
            for n, lst in find_gap(d).items():
                g_true = 100.0 * (e[n][0] - e[n][1]) / e[n][0]
                for path, v in lst:
                    if abs(v - g_true) < 0.05: gpaths.setdefault(n, []).append((path, 1.0))
                    elif abs(100 * v - g_true) < 0.05: gpaths.setdefault(n, []).append((path, 0.01))
            print("gap fields in the dose JSONs:", {n: [p for p, _ in v] for n, v in gpaths.items()})

# ---------- Experiments A and B from the logs ----------
rx_pod = re.compile(r"n=\s*(\d+)\s+eta_POD\s*=\s*([0-9.eE+-]+)")
rx_ae = re.compile(r"n=\s*(\d+)\s+eta_AE\s*=\s*([0-9.eE+-]+)")
rx_row = re.compile(r"^\s*(8|16|32|64)\s+([0-9.]+(?:e-?\d+)?)\s+([0-9.]+(?:e-?\d+)?)\s+(-?[0-9.]+)\s*$")
ab, ab_gap_log = {}, {}
for task in range(10):
    exp, s = ("A", task) if task < 5 else ("B", task - 5)
    p = f"{W}/logs/1136521_{task}_podae_seeds.out"
    txt = open(p).read()
    pod = {int(n): float(v) for n, v in rx_pod.findall(txt)}
    ae = {int(n): float(v) for n, v in rx_ae.findall(txt)}
    rows = {}
    for line in txt.splitlines():
        m = rx_row.match(line)
        if m:
            rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    if not all(n in pod and n in ae for n in NS):
        if all(n in rows for n in NS):
            pod = {n: rows[n][0] for n in NS}; ae = {n: rows[n][1] for n in NS}
        else:
            sys.exit(f"cannot parse eta values in {p}; last lines:\n" + "\n".join(txt.splitlines()[-15:]))
    ab[(exp, s)] = {n: (pod[n], ae[n]) for n in NS}
    if all(n in rows for n in NS):
        ab_gap_log[(exp, s)] = {n: rows[n][2] for n in NS}

# ---------- gaps, statistics, check against published means ----------
def gap(pa): return 100.0 * (pa[0] - pa[1]) / pa[0]
def stats(cond, seeds, use_log=False):
    out = {}
    for n in NS:
        pods = [cond[s][n][0] for s in seeds]; aes = [cond[s][n][1] for s in seeds]
        if use_log and all(k in ab_gap_log for k in seeds):
            gaps = [ab_gap_log[k][n] for k in seeds]
        else:
            gaps = [gap(cond[s][n]) for s in seeds]
        out[n] = (np.mean(pods), np.std(pods), np.mean(aes), np.std(aes), np.mean(gaps), np.std(gaps),
                  np.std(gaps, ddof=1))
    return out

ok = True
S_ab = {}
for exp in "AB":
    keys = [(exp, s) for s in range(5)]
    st = stats(ab, keys, use_log=True)
    S_ab[exp] = st
    for n, pub in zip(NS, PUB_AB[exp]):
        if abs(st[n][4] - pub) > 0.6: ok = False
        print(f"check {exp} n={n:2d}: gap {st[n][4]:7.2f} (published {pub:6.1f})"
              f"  std ddof0={st[n][5]:.2f} ddof1={st[n][6]:.2f}")
S_dose = {}
for R in LEVELS:
    st = stats(dose, [(R, s) for s in (0, 1, 2)])
    S_dose[R] = st
    for n, pub in zip(NS, PUB_DOSE[R]):
        if abs(st[n][4] - pub) > 0.6: ok = False
    print(f"check R={R:4d}: gaps " + " ".join(f"{st[n][4]:6.1f}" for n in NS)
          + "  published " + " ".join(f"{v:6.1f}" for v in PUB_DOSE[R]))
if not ok:
    sys.exit("extracted gaps do not reproduce the published means; nothing written")

# standard deviation convention: the one that reproduces the published A, n=8 value (3.1)
ddof = 0 if abs(S_ab["A"][8][5] - 3.1) <= abs(S_ab["A"][8][6] - 3.1) else 1
print(f"standard deviations reported with ddof={ddof}")
def sd_gap(st, n): return st[n][5] if ddof == 0 else st[n][6]
def sd_eta(vals): return np.std(vals, ddof=ddof)

# ---------- LaTeX tables (eta in units of 1e-3) ----------
os.makedirs(f"{W}/tables", exist_ok=True)
def f3(m, s): return f"${1e3*m:.3f}\\pm{1e3*s:.3f}$"
def fg(m, s): return f"${m:.1f}\\pm{s:.1f}$"

lines = [r"\begin{tabular}{c ccc c ccc}", r"\toprule",
         r"& \multicolumn{3}{c}{Experiment~A (fixed well)} & & \multicolumn{3}{c}{Experiment~B (variable well)} \\",
         r"\cmidrule(lr){2-4} \cmidrule(lr){6-8}",
         r"$n$ & $\eta_{\mathrm{POD}}\times10^{3}$ & $\eta_{\mathrm{AE}}\times10^{3}$ & gap (\%) & & "
         r"$\eta_{\mathrm{POD}}\times10^{3}$ & $\eta_{\mathrm{AE}}\times10^{3}$ & gap (\%) \\", r"\midrule"]
for n in NS:
    row = [str(n)]
    for exp in "AB":
        keys = [(exp, s) for s in range(5)]
        pods = [ab[k][n][0] for k in keys]; aes = [ab[k][n][1] for k in keys]
        row += [f3(np.mean(pods), sd_eta(pods)), f3(np.mean(aes), sd_eta(aes)),
                fg(S_ab[exp][n][4], sd_gap(S_ab[exp], n))]
        if exp == "A": row.append("")
    lines.append(" & ".join(row) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{W}/tables/table_500ep.tex", "w").write("\n".join(lines) + "\n")

lines = [r"\begin{tabular}{cc ccc}", r"\toprule",
         r"$R/\ell$ & $n$ & $\eta_{\mathrm{POD}}\times10^{3}$ & $\eta_{\mathrm{AE}}\times10^{3}$ & gap (\%) \\",
         r"\midrule"]
for i, R in enumerate(LEVELS):
    for j, n in enumerate(NS):
        keys = [(R, s) for s in (0, 1, 2)]
        pods = [dose[k][n][0] for k in keys]; aes = [dose[k][n][1] for k in keys]
        lab = f"${R/ELL:.2f}$" if j == 0 else ""
        lines.append(f"{lab} & {n} & {f3(np.mean(pods), sd_eta(pods))} & {f3(np.mean(aes), sd_eta(aes))}"
                     f" & {fg(S_dose[R][n][4], sd_gap(S_dose[R], n))} \\\\")
    if i < len(LEVELS) - 1: lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{W}/tables/table_dose.tex", "w").write("\n".join(lines) + "\n")

if set(gpaths) != set(NS) and find_gap(template):
    sys.exit("gap fields present in the dose JSONs but not matched to the eta values; "
             f"found {find_gap(template)}")
# ---------- JSONs for Experiments A and B in the dose structure (for plot_dose.py) ----------
os.makedirs(f"{W}/logs/seeds", exist_ok=True)
for (exp, s), vals in ab.items():
    d = copy.deepcopy(template)
    for n in NS:
        set_path(d, tpaths[n][0], vals[n][0]); set_path(d, tpaths[n][1], vals[n][1])
        g = ab_gap_log[(exp, s)][n] if (exp, s) in ab_gap_log else gap(vals[n])
        for path, scale in gpaths.get(n, []):
            set_path(d, path, g * scale)
    d["rebuilt_from_log"] = f"logs/1136521_{s if exp == 'A' else s + 5}_podae_seeds.out"
    out = f"{W}/logs/seeds/results_{exp}_seed{s}.json"
    json.dump(d, open(out, "w"), indent=1)
    back = json.load(open(out))
    for n, lst in find_gap(back).items():
        for path, v in lst:
            if any(path == q for q, _ in gpaths.get(n, [])):
                scale = [sc for q, sc in gpaths[n] if q == path][0]
                assert abs(v / scale - (ab_gap_log[(exp, s)][n] if (exp, s) in ab_gap_log else gap(vals[n]))) < 1e-6
            elif abs(v) > 0 and n in gpaths:
                pass
    e_back = find_eta(back)
    assert all(abs(e_back[n][0] - vals[n][0]) < 1e-12 and abs(e_back[n][1] - vals[n][1]) < 1e-12 for n in NS)

print("wrote tables/table_500ep.tex, tables/table_dose.tex and logs/seeds/results_{A,B}_seed*.json")
