"""Measure the characteristic drawdown radius ell of the fixed-well ensemble.

ell = radius at which the depth-averaged drawdown falls to half its value in
the innermost occupied radial bin (the four cells nearest the well, about
177 m from it). The far-field reference is the maximum of the radial mean
profile (draw = prof.max() - prof), and ell is linearly interpolated between
125 m bins. Because the cone grows by diffusion, ell is reported at several
snapshot times.

Assumptions (checked by the sanity test printed first):
  * 40 x 40 x 20 cells, 250 m x 250 m in plane, OpenFOAM blockMesh ordering
    (x fastest, then y, then z), i.e. a snapshot reshapes to (20, 40, 40).
  * the well centre is read from each sample's meta file.

Usage:  python3 measure_ell.py --data <DATA_ROOT>/ldo_data_v3_fixedwell --n 200
"""
import argparse, glob, os
import numpy as np

NX, NY, NZ, DX = 40, 40, 20, 250.0


def load_sample(path_p):
    p = np.load(path_p)                                   # (n_snap, 32000)
    meta = np.load(path_p.replace("_p.npy", "_meta.npy"), allow_pickle=True).item()
    return p.reshape(p.shape[0], NZ, NY, NX), meta


def radial_profile(field2d, cx, cy, edges):
    xc = (np.arange(NX) + 0.5) * DX
    yc = (np.arange(NY) + 0.5) * DX
    X, Y = np.meshgrid(xc, yc)                            # (NY, NX)
    r = np.hypot(X - cx, Y - cy)
    idx = np.digitize(r.ravel(), edges) - 1
    prof = np.array([field2d.ravel()[idx == k].mean() if np.any(idx == k) else np.nan
                     for k in range(len(edges) - 1)])
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200, help="number of samples to average")
    ap.add_argument("--times", type=int, nargs="+", default=[0, 9, 49, 99],
                    help="snapshot indices (0 = day 5, 99 = day 500)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data, "sample_*_p.npy")))[: a.n]
    if not files:
        raise SystemExit(f"no *_p.npy files in {a.data}")

    # ---- sanity check: minimum pressure should sit at the well ------------
    p, meta = load_sample(files[0])
    dav = p[-1].mean(axis=0)                              # depth average, last time
    j, i = np.unravel_index(np.argmin(dav), dav.shape)
    print(f"sanity: min pressure at cell (i={i}, j={j}) = "
          f"({(i+0.5)*DX:.0f} m, {(j+0.5)*DX:.0f} m); "
          f"well centre in meta = ({meta['well_cx_m']:.0f} m, {meta['well_cy_m']:.0f} m)")
    print("        if these disagree, the reshape order assumption is wrong -- stop here.\n")

    edges = np.arange(0, 7000 + DX / 2, DX / 2)           # 125 m radial bins
    centres = 0.5 * (edges[1:] + edges[:-1])
    profs = {t: [] for t in a.times}
    for f in files:
        p, meta = load_sample(f)
        cx, cy = meta["well_cx_m"], meta["well_cy_m"]
        for t in a.times:
            profs[t].append(radial_profile(p[t].mean(axis=0), cx, cy, edges))

    print(f"averaged over {len(files)} samples")
    print(f"{'day':>5} {'ell (m)':>9} {'ell (cells)':>12}")
    for t in a.times:
        stack = np.array(profs[t])
        ok = ~np.all(np.isnan(stack), axis=0)             # bins with at least one cell
        prof = np.nanmean(stack[:, ok], axis=0)
        rc = centres[ok]
        draw = prof.max() - prof                          # drawdown w.r.t. far field
        half = 0.5 * draw[0]                              # innermost occupied bin
        k = int(np.argmax(draw < half))                   # first bin below half
        ell = np.interp(half, [draw[k], draw[k - 1]], [rc[k], rc[k - 1]])
        print(f"{meta['snap_times_d'][t]:>5} {ell:>9.0f} {ell / DX:>12.1f}")


if __name__ == "__main__":
    main()
