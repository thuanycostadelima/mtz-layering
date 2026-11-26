#!/usr/bin/env python3
"""
Build two-layer MTZ velocity models from the uniform (T, f) .tvel grid.

Usage
-----
  python splice_layers.py MODELS_DIR OUT_DIR [--zc 522.97] [--w 0 40]
         [--T1 1300 1900 100] [--T2 1300 1900 100]
         [--f1 0 0.2 0.4 0.6] [--f2 0 0.2 0.4 0.6]

MODELS_DIR is the folder holding MM0.0/ ... MM1.0/ (the Perple_X grid).
Each output is a .tvel with the same header convention as the inputs:
  line 1: label   (e.g. MM0.2-0.6_1400-1600_w40)
  line 2: T1      (kept numeric so TauP readers that parse it still work)
then depth Vp Vs rho rows.

w = 0 reproduces a plain join: upper profile to zc, lower from zc,
depth row duplicated at zc (first-order interface).
w > 0 blends Vp, Vs, rho linearly between zc - w/2 and zc + w/2 on the
union of both depth grids, so the interface is a gradient zone of width w.

Validation: with --T1 1400 --T2 1400 --f1 0 --f2 0.6 --w 0 the output must
match twolayer/MM0.0-0.6_1400.tvel row for row (check_against() does this).
"""
import argparse
import os
import sys
import numpy as np


def read_tvel(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    hdr = lines[:2]
    data = np.loadtxt(lines[2:])
    return hdr, data  # data columns: depth Vp Vs rho


def write_tvel(path, label, T1, data):
    with open(path, "w") as fh:
        fh.write(f"{label}\n{T1:.1f}\n")
        np.savetxt(fh, data, fmt="%10.4f")


def uniform_path(models_dir, f, T):
    return os.path.join(models_dir, f"MM{f:.1f}", f"MM{f:.1f}_{int(T)}.tvel")


def splice(upper, lower, zc, w):
    """upper/lower: (n,4) arrays depth Vp Vs rho. Returns spliced array."""
    if w <= 0:
        up = upper[upper[:, 0] <= zc]
        lo = lower[lower[:, 0] >= zc]
        # Interpolate both onto zc so the interface sits exactly at zc even
        # if zc is not a grid node; duplicate the depth row.
        up_zc = np.array([zc, *[np.interp(zc, upper[:, 0], upper[:, k]) for k in (1, 2, 3)]])
        lo_zc = np.array([zc, *[np.interp(zc, lower[:, 0], lower[:, k]) for k in (1, 2, 3)]])
        up = up[up[:, 0] < zc]
        lo = lo[lo[:, 0] > zc]
        return np.vstack([up, up_zc, lo_zc, lo])

    z0, z1 = zc - w / 2, zc + w / 2
    # union of grids within the blend zone, plus the edges
    zu = np.unique(np.concatenate([
        upper[(upper[:, 0] > z0) & (upper[:, 0] < z1), 0],
        lower[(lower[:, 0] > z0) & (lower[:, 0] < z1), 0],
        [z0, z1],
    ]))
    frac = (zu - z0) / w
    blend = np.column_stack([zu] + [
        (1 - frac) * np.interp(zu, upper[:, 0], upper[:, k])
        + frac * np.interp(zu, lower[:, 0], lower[:, k])
        for k in (1, 2, 3)
    ])
    up = upper[upper[:, 0] < z0]
    lo = lower[lower[:, 0] > z1]
    return np.vstack([up, blend, lo])


def build(models_dir, out_dir, T1, f1, T2, f2, zc, w):
    _, upper = read_tvel(uniform_path(models_dir, f1, T1))
    _, lower = read_tvel(uniform_path(models_dir, f2, T2))
    data = splice(upper, lower, zc, w)
    # two-layer reference files use Vs = 0.001 in the fluid outer core (not 0);
    # keep that convention so the files are drop-in for the same synthetics code.
    data[:, 2] = np.where(data[:, 2] <= 0, 0.001, data[:, 2])
    label = f"MM{f1:.1f}-{f2:.1f}_{int(T1)}-{int(T2)}_zc{int(round(zc))}_w{int(w)}"
    out = os.path.join(out_dir, label + ".tvel")
    write_tvel(out, label, T1, data)
    return out


def check_against(models_dir, ref_path, T1, f1, T2, f2, zc):
    """Row-for-row comparison with a two-layer reference file."""
    _, ref = read_tvel(ref_path)
    _, upper = read_tvel(uniform_path(models_dir, f1, T1))
    _, lower = read_tvel(uniform_path(models_dir, f2, T2))
    mine = splice(upper, lower, zc, 0)
    mine[:, 2] = np.where(mine[:, 2] <= 0, 0.001, mine[:, 2])
    if mine.shape != ref.shape:
        return False, f"shape {mine.shape} vs {ref.shape}"
    d = np.abs(mine - ref).max()
    return d < 1e-3, f"max abs diff {d:.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--zc", type=float, default=522.9705)
    ap.add_argument("--w", type=float, nargs="+", default=[0.0, 40.0])
    ap.add_argument("--T1", type=float, nargs=3, default=[1300, 1900, 100])
    ap.add_argument("--T2", type=float, nargs=3, default=[1300, 1900, 100])
    ap.add_argument("--f1", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6])
    ap.add_argument("--f2", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6])
    ap.add_argument("--check", help="path to a reference twolayer .tvel to validate the join")
    a = ap.parse_args()

    if a.check:
        base = os.path.basename(a.check)            # MM0.0-0.6_1400.tvel
        ff, T = base[2:].replace(".tvel", "").split("_")
        f1, f2 = (float(x) for x in ff.split("-"))
        ok, msg = check_against(a.models_dir, a.check, float(T), f1, float(T), f2, a.zc)
        print(("PASS " if ok else "FAIL ") + msg)
        if not ok:
            sys.exit(1)

    os.makedirs(a.out_dir, exist_ok=True)
    T1s = np.arange(a.T1[0], a.T1[1] + 1, a.T1[2])
    T2s = np.arange(a.T2[0], a.T2[1] + 1, a.T2[2])
    n = 0
    for T1 in T1s:
        for T2 in T2s:
            for f1 in a.f1:
                for f2 in a.f2:
                    for w in a.w:
                        build(a.models_dir, a.out_dir, T1, f1, T2, f2, a.zc, w)
                        n += 1
    print(f"wrote {n} models to {a.out_dir}")


if __name__ == "__main__":
    main()
