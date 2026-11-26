#!/usr/bin/env python3
"""
Relative amplitude of the 660 and 410 in the data against the best-fitting synthetic of each class (same run, same A and tau).
For every cap: peak amplitude of the 410 and 660 pulses in the data and in the synthetic, the 660/410 ratio of each, and the
ratio of ratios (data 660/410) / (synthetic 660/410).  A library whose 660 is too weak and 410 too strong relative to the Earth
by 20 % each would give a ratio of ratios of about 1.5 (Earth 660 stronger relative to the 410 than the library); the reverse
mismatch 0.67.  Printed by class of the cap and by record-count quartile.
    python amp_ratio.py OBS_DIR INPUTS_DIR RUN_DIR --lib ... --mech mech_filters.npz
"""
import argparse, glob, json, numpy as np
from invert_bins2 import load_lib, bin_weights, TW, DT

ap = argparse.ArgumentParser(); ap.add_argument("obs"); ap.add_argument("inputs"); ap.add_argument("run"); ap.add_argument("--lib", nargs="+", required=True); ap.add_argument("--mech", required=True)
a = ap.parse_args()
names, S, meta, tg, dist = load_lib(a.lib); M = dict(zip(names, range(len(names)))); mech = dict(zip(*[np.load(a.mech)[k] for k in ("names", "h")]))
tobs = np.arange(TW[0], TW[1] + 1e-9, DT); i0 = int(round(TW[0] - tg[0])); nt = tobs.size
W660 = (tobs >= -265) & (tobs <= -210); W410 = (tobs >= -190) & (tobs <= -140)
rows = []
for f in sorted(glob.glob(f"{a.run}/*.json")):
    J = json.load(open(f)); name = J["bin"]
    if name not in mech: continue
    W, _ = bin_weights(f"{a.inputs}/{name}.txt", dist); t, p, x, sg = np.loadtxt(f"{a.obs}/{name}.txt", unpack=True); d = np.interp(tobs, t, x)
    cls = 2 if J["lnB21"] > 3 else (1 if J["lnB10"] > 3 else 0); row = [J["n_records"], cls, d[W660].max(), d[W410].max()]
    for c in ("M0", "M1", "M2"):
        b = J["classes"][c]; g = np.convolve(W @ S[M[b["best"]]], mech[name], mode="same"); tau = int(b["best_tau"]); g = b["best_A"] * g[i0 + tau:i0 + tau + nt]
        row += [g[W660].max(), g[W410].max()]
    rows.append(row)
R = np.array(rows, float); n = len(R)
rr = {c: (R[:, 2] / R[:, 3]) / (R[:, 4 + 2 * k] / R[:, 5 + 2 * k]) for k, c in enumerate(("M0", "M1", "M2"))}   # ratio of ratios
a660 = {c: R[:, 2] / R[:, 4 + 2 * k] for k, c in enumerate(("M0", "M1", "M2"))}; a410 = {c: R[:, 3] / R[:, 5 + 2 * k] for k, c in enumerate(("M0", "M1", "M2"))}
q = np.percentile(R[:, 0], [0, 25, 50, 75, 100]); iq = np.clip(np.digitize(R[:, 0], q) - 1, 0, 3)
print(f"{n} caps. Observed 660/410 amplitude ratio: median {np.median(R[:, 2] / R[:, 3]):.2f} (interquartile {np.percentile(R[:, 2] / R[:, 3], 25):.2f}-{np.percentile(R[:, 2] / R[:, 3], 75):.2f})")
print("Ratio of ratios (data 660/410) / (best synthetic 660/410): median and interquartile range; and data/synthetic amplitude of each pulse at the fitted A")
print(f"{'caps':12s} {'n':>4s} | {'best M0: 660/410':>17s} {'660':>6s} {'410':>6s} | {'best M1: 660/410':>17s} {'660':>6s} {'410':>6s} | {'best M2: 660/410':>17s} {'660':>6s} {'410':>6s}")
def line(lab, m):
    s = f"{lab:12s} {m.sum():4d}"
    for c in ("M0", "M1", "M2"):
        v = rr[c][m]; s += f" | {np.median(v):6.2f} ({np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f}) {np.median(a660[c][m]):6.2f} {np.median(a410[c][m]):6.2f}"
    print(s)
line("all caps", np.ones(n, bool))
for c, lab in ((0, "M0 caps"), (1, "M1 caps"), (2, "M2 caps")): line(lab, R[:, 1] == c)
for k in range(4): line(f"Q{k + 1} records", iq == k)
np.savetxt(f"{a.run}_amp_ratio.csv", R, delimiter=",", header="nrec,class,d660,d410,M0_660,M0_410,M1_660,M1_410,M2_660,M2_410", comments="")
