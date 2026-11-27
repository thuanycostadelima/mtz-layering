#!/usr/bin/env python3
"""
Is the 660 of the M2 caps broadened or weakened relative to the 410, as incoherent stacking of 660 topography would do?
For every cap: full width at half maximum of the 410 and 660 in the data and in the best uniform (M0) synthetic (same A, tau),
the 660/410 amplitude ratio of data and synthetic, and the bootstrap scatter of the data around each pulse (mean sigma over the
pulse window divided by the peak amplitude, a proxy for within-cap timing and amplitude scatter).  Reported by class of the
cap and by tercile of the uniform-class temperature (cold / intermediate / warm), with the correlation of each quantity
with P(M2).
    python pulse_by_class.py OBS_DIR INPUTS_DIR RUN_DIR --lib ... --mech mech_filters.npz
"""
import argparse, glob, json, numpy as np
from invert_bins2 import load_lib, bin_weights, TW, DT

ap = argparse.ArgumentParser(); ap.add_argument("obs"); ap.add_argument("inputs"); ap.add_argument("run"); ap.add_argument("--lib", nargs="+", required=True); ap.add_argument("--mech", required=True)
a = ap.parse_args()
names, S, meta, tg, dist = load_lib(a.lib); M = dict(zip(names, range(len(names)))); mech = dict(zip(*[np.load(a.mech)[k] for k in ("names", "h")]))
tobs = np.arange(TW[0], TW[1] + 1e-9, DT); i0 = int(round(TW[0] - tg[0])); nt = tobs.size
WIN = {"660": (-265, -210), "410": (-190, -140)}


def fwhm(x, t, lo, hi):
    w = (t >= lo) & (t <= hi); xx = x[w]; tt = t[w]; k = np.argmax(xx); pk = xx[k]
    if pk <= 0: return np.nan
    l = k
    while l > 0 and xx[l] > pk / 2: l -= 1
    r = k
    while r < len(xx) - 1 and xx[r] > pk / 2: r += 1
    if l == 0 or r == len(xx) - 1: return np.nan
    return np.interp(pk / 2, [xx[r], xx[r - 1]], [tt[r], tt[r - 1]]) - np.interp(pk / 2, [xx[l], xx[l + 1]], [tt[l], tt[l + 1]])


rows = []
for f in sorted(glob.glob(f"{a.run}/*.json")):
    J = json.load(open(f)); name = J["bin"]
    if name not in mech: continue
    W, _ = bin_weights(f"{a.inputs}/{name}.txt", dist); t, p, x, sg = np.loadtxt(f"{a.obs}/{name}.txt", unpack=True); d = np.interp(tobs, t, x); s = np.interp(tobs, t, sg)
    c = J["classes"]; cls = 2 if J["lnB21"] > 3 else (1 if J["lnB10"] > 3 else 0)
    P2 = np.exp(c["M2"]["logZ"]) / sum(np.exp(c[k]["logZ"]) for k in ("M0", "M1", "M2"))
    b = c["M0"]; g = np.convolve(W @ S[M[b["best"]]], mech[name], mode="same"); tau = int(b["best_tau"]); g = b["best_A"] * g[i0 + tau:i0 + tau + nt]
    r = [cls, P2, b["mean_T1"]]
    for k in ("660", "410"):
        lo, hi = WIN[k]; w = (tobs >= lo) & (tobs <= hi)
        r += [fwhm(d, tobs, lo, hi), fwhm(g, tobs, lo, hi), d[w].max(), g[w].max(), s[w].mean() / d[w].max()]
    rows.append(r)
R = np.array(rows, float); n = len(R)
# columns: 0 cls, 1 P2, 2 T_M0, 3 w660_d, 4 w660_s, 5 a660_d, 6 a660_s, 7 n660, 8 w410_d, 9 w410_s, 10 a410_d, 11 a410_s, 12 n410
Q = {"660 width data/syn": R[:, 3] / R[:, 4], "410 width data/syn": R[:, 8] / R[:, 9],
     "660/410 ratio data/syn": (R[:, 5] / R[:, 10]) / (R[:, 6] / R[:, 11]),
     "660 data FWHM (s)": R[:, 3], "410 data FWHM (s)": R[:, 8],
     "660 scatter sigma/peak": R[:, 7], "410 scatter sigma/peak": R[:, 12], "scatter 660/410": R[:, 7] / R[:, 12]}
tT = np.percentile(R[:, 2], [100 / 3, 200 / 3])
groups = [("all caps", np.ones(n, bool)), ("M0 caps", R[:, 0] == 0), ("M1 caps", R[:, 0] == 1), ("M2 caps", R[:, 0] == 2),
          (f"cold third (T_M0 < {tT[0]:.0f} K)", R[:, 2] < tT[0]), ("middle third", (R[:, 2] >= tT[0]) & (R[:, 2] < tT[1])), (f"warm third (T_M0 >= {tT[1]:.0f} K)", R[:, 2] >= tT[1])]
print(f"{n} caps. Medians (interquartile range) of each quantity by group; last column: correlation with P(M2) over all caps")
print(f"{'quantity':26s}" + "".join(f"{g[0][:18]:>20s}" for g in groups) + f"{'r(P(M2))':>10s}")
for lab, v in Q.items():
    ok = np.isfinite(v)
    line = f"{lab:26s}"
    for _, m in groups:
        vv = v[m & ok]; line += f"{np.nanmedian(vv):8.2f} ({np.percentile(vv, 25):.2f}-{np.percentile(vv, 75):.2f})"[:20].rjust(20)
    print(line + f"{np.corrcoef(v[ok], R[ok, 1])[0, 1]:10.2f}")
m2 = R[:, 0] == 2; m0 = R[:, 0] == 0
print(f"\nM2 against M0 caps: 660 width data/syn {np.nanmedian(Q['660 width data/syn'][m2]):.2f} vs {np.nanmedian(Q['660 width data/syn'][m0]):.2f}; "
      f"410 width data/syn {np.nanmedian(Q['410 width data/syn'][m2]):.2f} vs {np.nanmedian(Q['410 width data/syn'][m0]):.2f}; "
      f"660 scatter sigma/peak {np.nanmedian(Q['660 scatter sigma/peak'][m2]):.3f} vs {np.nanmedian(Q['660 scatter sigma/peak'][m0]):.3f}")
np.savetxt(f"{a.run}_pulse_by_class.csv", R, delimiter=",", header="class,P_M2,T_M0,w660_data,w660_syn,a660_data,a660_syn,noise660,w410_data,w410_syn,a410_data,a410_syn,noise410", comments="")
