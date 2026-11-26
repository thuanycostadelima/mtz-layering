#!/usr/bin/env python3
"""
Per-bin source-mechanism/depth correction filters.

For each bin, a short FIR filter h (2L+1 taps at 1 s) is fitted by ridge-regularised least squares
so that  h * stack_fixed  ~  stack_realCMT  over the precursor window, where both stacks are of the
same reference model: stack_fixed uses a fixed source (M_rphi, 10 km) for every record and
stack_realCMT uses each record's GCMT tensor and depth (stack_bins.py with/without --fixed-source).
The filter captures what the real source distribution does to the wavelet beyond the distance
distribution (which the per-model exact stacks already contain).

Usage: python mech_filter.py STACKS_REAL STACKS_FIXED out.npz [--L 15] [--lam 1e-2]
"""
import argparse, glob, os
import numpy as np
from scipy.linalg import solve

TG = np.arange(-340.0, -70.0 + 1e-9, 1.0)
WIN = (TG >= -300) & (TG <= -110)


def set_window(tmin, tmax, wmin, wmax):
    global TG, WIN
    TG = np.arange(tmin, tmax + 1e-9, 1.0); WIN = (TG >= wmin) & (TG <= wmax)


def fit(x, y, L, lam):
    X = np.array([np.roll(x, k) for k in range(-L, L + 1)]).T
    Xw, yw = X[WIN], y[WIN]
    G = Xw.T @ Xw; G += lam * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
    h = solve(G, Xw.T @ yw)
    r = X @ h
    return h, 1 - np.sum((r[WIN] - yw) ** 2) / np.sum(yw ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("real"); ap.add_argument("fixed"); ap.add_argument("out")
    ap.add_argument("--L", type=int, default=15); ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--window", type=float, nargs=4, default=None, metavar=("TMIN", "TMAX", "WMIN", "WMAX"),
                    help="trace span and fit window (s rel. main phase); PP at 105: -200 -30 -150 -60")
    a = ap.parse_args()
    if a.window: set_window(*a.window)
    names, H, R2 = [], [], []
    for f in sorted(glob.glob(os.path.join(a.fixed, "*.txt"))):
        nm = os.path.basename(f)
        if nm == "summary.txt" or not os.path.exists(os.path.join(a.real, nm)):
            continue
        fx = np.loadtxt(f); fx = np.interp(TG, fx[:, 0], fx[:, 2])
        rc = np.loadtxt(os.path.join(a.real, nm)); rc = np.interp(TG, rc[:, 0], rc[:, 2])
        h, r2 = fit(fx, rc, a.L, a.lam)
        names.append(nm[:-4]); H.append(h); R2.append(r2)
    np.savez(a.out, names=np.array(names), h=np.array(H), r2=np.array(R2), L=a.L)
    R2 = np.array(R2)
    print(f"{len(names)} bins; variance explained by filter: median {np.median(R2):.4f}, min {R2.min():.4f}")


if __name__ == "__main__":
    main()
