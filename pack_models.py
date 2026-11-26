#!/usr/bin/env python3
"""
Pack, for every model in the synthetic library, its single-source traces at all distances
already shifted onto the reference-distance slowness curve, so that a bin's synthetic stack
is a weighted sum over distances:  stack_bin(t) = sum_D W_bin(D) * S_model(D, t).

For each RUNS_DIR/<model>/post/xsec_<D>.txt (D = 100..170 step 1):
    S(D, t) = x_D(t + p(t) * (D - ref))      on t = -340..-70 s step 1 s
where p(t) is the relative slowness curve of the observed cross-sections (read from OBS_FILE).

Usage:
  python pack_models.py OBS_FILE OUT_PREFIX RUNS_DIR [RUNS_DIR ...] [--ref 124] [--group-size 400]
Output: OUT_PREFIX_<k>.npz with  names [n], dist [71], t [nt], S [n, 71, nt] float16
"""
import argparse, glob, os, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obs"); ap.add_argument("out_prefix"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--ref", type=float, default=124.0); ap.add_argument("--group-size", type=int, default=400)
    ap.add_argument("--dmin", type=int, default=100); ap.add_argument("--dmax", type=int, default=170)
    ap.add_argument("--tmin", type=float, default=-340.0); ap.add_argument("--tmax", type=float, default=-70.0, help="time window (s rel. main phase); PP at 105 deg: -200 -30")
    a = ap.parse_args()
    od = np.loadtxt(a.obs); tg = np.arange(a.tmin, a.tmax + 1e-9, 1.0)
    p = np.interp(tg, od[:, 0], od[:, 1])
    dists = np.arange(a.dmin, a.dmax + 1)
    models = []
    for rd in a.runs:
        for d in sorted(glob.glob(os.path.join(rd, "MM*"))):
            if os.path.exists(os.path.join(d, "post", f"xsec_{a.ref:.1f}.txt")):
                models.append(d)
    print(len(models), "models")
    names, S = [], []
    k = 0
    for i, d in enumerate(models):
        arr = np.zeros((len(dists), len(tg)), np.float32)
        okm = True
        for j, D in enumerate(dists):
            fn = os.path.join(d, "post", f"xsec_{D:.1f}.txt")
            if not os.path.exists(fn):
                okm = False; break
            x = np.loadtxt(fn)
            arr[j] = np.interp(tg + p * (D - a.ref), x[:, 0], x[:, 2], left=0.0, right=0.0)
        if not okm:
            print("incomplete:", os.path.basename(d)); continue
        names.append(os.path.basename(d)); S.append(arr.astype(np.float16))
        if len(names) == a.group_size or i == len(models) - 1:
            np.savez_compressed(f"{a.out_prefix}_{k}.npz", names=np.array(names), dist=dists, t=tg, S=np.array(S))
            print(f"wrote {a.out_prefix}_{k}.npz ({len(names)} models)", flush=True); names, S = [], []; k += 1


if __name__ == "__main__":
    main()
