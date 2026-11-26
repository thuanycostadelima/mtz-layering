#!/usr/bin/env python3
"""
Where does the M2 improvement come from?  For every cap of a run, take the best model of M1 and of M2 (with their A and tau),
compute the whitened squared residual sum((d - A g)^2 / sigma^2) in three sub-windows of the fit window:
   410 region  -185 .. -145 s      520 region  -225 .. -185 s      660 region  -265 .. -225 s   (124 deg; 410 ~ -166 s, 520 ~ -204 s, 660 ~ -237 s)
and report the misfit reduction of M2 relative to M1 in each, for the M2 caps and for the rest.
    python misfit_by_window.py OBS_DIR INPUTS_DIR RUN_DIR --lib ... --mech mech_filters.npz
"""
import argparse, glob, json, os, numpy as np
from invert_bins2 import load_lib, bin_weights, TW, DT

ap = argparse.ArgumentParser(); ap.add_argument("obs"); ap.add_argument("inputs"); ap.add_argument("run"); ap.add_argument("--lib", nargs="+", required=True); ap.add_argument("--mech", required=True)
a = ap.parse_args()
names, S, meta, tg, dist = load_lib(a.lib); M = dict(zip(names, range(len(names)))); mech = dict(zip(*[np.load(a.mech)[k] for k in ("names", "h")]))
tobs = np.arange(TW[0], TW[1] + 1e-9, DT); i0 = int(round(TW[0] - tg[0])); nt = tobs.size
wins = {"660": (-265, -225), "520": (-225, -185), "410": (-185, -145)}
rows = []
for f in sorted(glob.glob(f"{a.run}/*.json")):
    J = json.load(open(f)); name = J["bin"]
    if name not in mech: continue
    W, _ = bin_weights(f"{a.inputs}/{name}.txt", dist)
    t, p, x, sg = np.loadtxt(f"{a.obs}/{name}.txt", unpack=True); d = np.interp(tobs, t, x); s = np.interp(tobs, t, sg) * np.sqrt(J["noise_scale"])
    chi = {}
    for c in ("M0", "M1", "M2"):
        b = J["classes"][c]; g = np.convolve(W @ S[M[b["best"]]], mech[name], mode="same"); tau = int(b["best_tau"]); g = g[i0 + tau:i0 + tau + nt]
        r = (d - b["best_A"] * g) / s
        chi[c] = {k: float(np.sum(r[(tobs >= lo) & (tobs < hi)] ** 2)) for k, (lo, hi) in wins.items()}
    cls = 2 if J["lnB21"] > 3 else (1 if J["lnB10"] > 3 else 0)
    rows.append((name, cls, J["lnB21"], *[chi["M1"][k] - chi["M2"][k] for k in wins], *[chi["M0"][k] - chi["M2"][k] for k in wins]))
R = np.array([r[1:] for r in rows], float)
print(f"{len(rows)} caps.  Reduction of the whitened squared misfit, best M2 relative to best M1 (and to best M0), summed per window (mean over caps):")
print(f"{'caps':22s} {'n':>4s} | {'M1-M2 660':>10s} {'520':>8s} {'410':>8s} | {'M0-M2 660':>10s} {'520':>8s} {'410':>8s} | share of the M1->M2 reduction in the 520 window")
for lab, m in (("M2 caps", R[:, 0] == 2), ("M1 caps", R[:, 0] == 1), ("M0 caps", R[:, 0] == 0)):
    r = R[m]; tot = r[:, 2:5].sum(1); share = np.where(tot > 0, r[:, 3] / np.where(tot > 0, tot, 1), np.nan)
    print(f"{lab:22s} {m.sum():4d} | {r[:,2].mean():10.1f} {r[:,3].mean():8.1f} {r[:,4].mean():8.1f} | {r[:,5].mean():10.1f} {r[:,6].mean():8.1f} {r[:,7].mean():8.1f} | median {np.nanmedian(share)*100:.0f} %, >50 % in {np.nanmean(share > 0.5)*100:.0f} % of caps")
m = R[:, 0] == 2; r = R[m]
print("\nM2 caps: fraction where the largest M1->M2 reduction is in the 520 window: %.0f %%; in the 660 window: %.0f %%; in the 410 window: %.0f %%"
      % tuple(np.mean(np.argmax(r[:, 2:5], 1) == k) * 100 for k in (1, 0, 2)))
np.savetxt(f"{a.run}_misfit_by_window.csv", R, delimiter=",", header="class,lnB21,dM1M2_660,dM1M2_520,dM1M2_410,dM0M2_660,dM0M2_520,dM0M2_410", comments="")
