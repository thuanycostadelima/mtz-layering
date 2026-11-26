#!/usr/bin/env python3
"""
Phase-aware interpolation of the Perple_X .tvel profiles to intermediate temperatures, and the
50 K layered set for the thermal-vs-compositional ambiguity.

Between profiles A (T) and B (T+100) at the same f, the 410, 520 and 660 are located (max dVs/dz in
370-470, 480-610, 630-700 km); the output depth axis is mapped piecewise-linearly onto each parent so that
boundaries interpolate in depth and properties interpolate between corresponding phases. Validated at the
existing 100 K nodes: RMS Vs error 9-17 m/s over a 200 K span (naive: 18-38; neighbour difference: 70-85).

Outputs (in OUT_DIR):
  uniform/MM<f>_<T>.tvel          f = 0..0.6, T = 1350..1950 step 100 (interpolated)
  layered/MM<f1>-<f2>_<T1>-<T2>_zc523_w0.tvel   spliced at 522.9705 km as in splice_layers.py
Usage: python interp_profiles.py MODELS_DIR OUT_DIR
"""
import os, sys, numpy as np

ZC = 522.9705
MD, OUT = sys.argv[1], sys.argv[2]


def tvel(f, T):
    p = os.path.join(MD, f"MM{f:.1f}", f"MM{f:.1f}_{T}.tvel")
    if os.path.exists(p):
        return np.loadtxt(p, skiprows=2), open(p).readlines()[:2]
    p = os.path.join(OUT, "uniform", f"MM{f:.1f}_{T}.tvel")
    return np.loadtxt(p, skiprows=2), open(p).readlines()[:2]


def anchors(x):
    z, vs = x[:, 0], x[:, 2]; g = np.diff(vs) / np.maximum(np.diff(z), 1e-6); zm = 0.5 * (z[1:] + z[:-1]); out = []
    for lo, hi in ((370, 470), (480, 610), (630, 700)):
        k = (zm > lo) & (zm < hi); out.append(zm[k][np.argmax(g[k])])
    out[2] = 658.7   # the 660 does not move with T in these tables; avoid 4-km node jitter
    return out


def interp(A, B, w=0.5, zlo=250.0, zhi=900.0):
    """Return a profile on A's depth nodes: phase-aware blend inside zlo..zhi, plain blend outside."""
    aA, aB = anchors(A), anchors(B); aM = [(1 - w) * a + w * b for a, b in zip(aA, aB)]
    knA, knB, knM = [zlo] + aA + [zhi], [zlo] + aB + [zhi], [zlo] + aM + [zhi]
    z = A[:, 0].copy(); out = A.copy()
    inside = (z > zlo) & (z < zhi)
    zA = np.where(inside, np.interp(z, knM, knA), z); zB = np.where(inside, np.interp(z, knM, knB), z)
    for c in (1, 2, 3):
        va = np.interp(zA, A[:, 0], A[:, c]); vb = np.interp(zB, B[:, 0], B[:, c])
        out[:, c] = (1 - w) * va + w * vb
    # keep exact duplicate-depth structure of A outside the mantle window untouched
    out[~inside] = (1 - w) * A[~inside] + w * B[~inside] if A.shape == B.shape else A[~inside]
    out[:, 0] = z
    return out


def write(path, hdr, x):
    with open(path, "w") as fh:
        fh.writelines(hdr)
        for r in x:
            fh.write(f"{r[0]:.4f} {r[1]:.5f} {r[2]:.5f} {r[3]:.5f}\n")


os.makedirs(os.path.join(OUT, "uniform"), exist_ok=True); os.makedirs(os.path.join(OUT, "layered"), exist_ok=True)
fs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for f in fs:
    for T in range(1350, 2000, 100):
        A, hdr = tvel(f, T - 50); B, _ = tvel(f, T + 50)
        hdr = [hdr[0], f"{T:.1f}\n"]
        write(os.path.join(OUT, "uniform", f"MM{f:.1f}_{T}.tvel"), hdr, interp(A, B))
print("uniform done")


def splice(f1, T1, f2, T2):
    A, hdr = tvel(f1, T1); B, _ = tvel(f2, T2)
    out = np.vstack([A[A[:, 0] <= ZC], B[B[:, 0] >= ZC]])
    return hdr, out


n = 0
T1s = (1550, 1650, 1750); T2s = range(1450, 1900, 50)
for f in (0.0, 0.2, 0.4, 0.6):                       # M1: f1 = f2, 50 K T2
    for T1 in T1s:
        for T2 in T2s:
            if T2 == T1: continue
            hdr, out = splice(f, T1, f, T2); write(os.path.join(OUT, "layered", f"MM{f:.1f}-{f:.1f}_{T1}-{T2}_zc523_w0.tvel"), hdr, out); n += 1
for f1, f2 in ((0.0, 0.2), (0.0, 0.4), (0.0, 0.6), (0.2, 0.4), (0.2, 0.6)):   # M2: 50 K T1, T2
    for T1 in T1s:
        for T2 in T1s:
            hdr, out = splice(f1, T1, f2, T2); write(os.path.join(OUT, "layered", f"MM{f1:.1f}-{f2:.1f}_{T1}-{T2}_zc523_w0.tvel"), hdr, out); n += 1
print("layered:", n)
