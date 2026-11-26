#!/usr/bin/env python3
"""
Width of the 410 and 660 pulses in the data compared with the best-fitting synthetic of each class (same run, same A and tau).
Width = full width at half maximum of the positive pulse, in seconds, measured on the data (bootstrap-smoothed only by the stack)
and on the synthetic; the ratio data/synthetic > 1 means the observed pulse is broader than the model's.
    python pulse_widths.py OBS_DIR INPUTS_DIR RUN_DIR --lib ... --mech mech_filters.npz
"""
import argparse, glob, json, numpy as np
from invert_bins2 import load_lib, bin_weights, TW, DT

ap = argparse.ArgumentParser(); ap.add_argument("obs"); ap.add_argument("inputs"); ap.add_argument("run"); ap.add_argument("--lib", nargs="+", required=True); ap.add_argument("--mech", required=True)
a = ap.parse_args()
names, S, meta, tg, dist = load_lib(a.lib); M = dict(zip(names, range(len(names)))); mech = dict(zip(*[np.load(a.mech)[k] for k in ("names", "h")]))
tobs = np.arange(TW[0], TW[1] + 1e-9, DT); i0 = int(round(TW[0] - tg[0])); nt = tobs.size


def fwhm(x, t, lo, hi):
    w = (t >= lo) & (t <= hi); xx = x[w]; tt = t[w]; k = np.argmax(xx); pk = xx[k]
    if pk <= 0: return np.nan
    l = k
    while l > 0 and xx[l] > pk / 2: l -= 1
    r = k
    while r < len(xx) - 1 and xx[r] > pk / 2: r += 1
    if l == 0 or r == len(xx) - 1: return np.nan
    tl = np.interp(pk / 2, [xx[l], xx[l + 1]], [tt[l], tt[l + 1]]); tr = np.interp(pk / 2, [xx[r], xx[r - 1]], [tt[r], tt[r - 1]])
    return tr - tl


rows = []
for f in sorted(glob.glob(f"{a.run}/*.json")):
    J = json.load(open(f)); name = J["bin"]
    if name not in mech: continue
    W, _ = bin_weights(f"{a.inputs}/{name}.txt", dist); t, p, x, sg = np.loadtxt(f"{a.obs}/{name}.txt", unpack=True); d = np.interp(tobs, t, x)
    cls = 2 if J["lnB21"] > 3 else (1 if J["lnB10"] > 3 else 0); row = [cls]
    for c in ("M0", "M2"):
        b = J["classes"][c]; g = np.convolve(W @ S[M[b["best"]]], mech[name], mode="same"); tau = int(b["best_tau"]); g = b["best_A"] * g[i0 + tau:i0 + tau + nt]
        for lo, hi in ((-260, -210), (-190, -140)):          # 660, 410
            row += [fwhm(d, tobs, lo, hi), fwhm(g, tobs, lo, hi)]
    rows.append(row)
R = np.array(rows, float)
print(f"{len(R)} caps. FWHM in seconds (median over caps) and ratio data/synthetic:")
print(f"{'':14s} {'660 data':>9s} {'660 syn':>8s} {'ratio':>6s} {'>1 in':>6s} | {'410 data':>9s} {'410 syn':>8s} {'ratio':>6s} {'>1 in':>6s}")
for lab, m in (("all caps", np.ones(len(R), bool)), ("M2 caps", R[:, 0] == 2), ("M0 caps", R[:, 0] == 0)):
    for c, j in (("best M0", 1), ("best M2", 5)):
        r6 = R[m, j] / R[m, j + 1]; r4 = R[m, j + 2] / R[m, j + 3]
        print(f"{lab + ', ' + c:22s} {np.nanmedian(R[m, j]):7.1f} {np.nanmedian(R[m, j + 1]):8.1f} {np.nanmedian(r6):6.2f} {np.nanmean(r6 > 1) * 100:5.0f}% | {np.nanmedian(R[m, j + 2]):7.1f} {np.nanmedian(R[m, j + 3]):8.1f} {np.nanmedian(r4):6.2f} {np.nanmean(r4 > 1) * 100:5.0f}%")
np.savetxt(f"{a.run}_pulse_widths.csv", R, delimiter=",", header="class,w660_data_M0,w660_syn_M0,w410_data_M0,w410_syn_M0,w660_data_M2,w660_syn_M2,w410_data_M2,w410_syn_M2", comments="")
