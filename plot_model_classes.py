#!/usr/bin/env python3
"""
Illustrate the three model classes (M0 uniform, M1 thermal layering, M2 thermochemical
layering): Vs profiles from the Perple_X .tvel grid (top) and the Yspec SS synthetic
at 124 deg, plotted vs PREM-converted depth (bottom).

Usage: python plot_model_classes.py MODELS_DIR RUNS_UNIFORM RUNS_LAYERED TABLE.npz out.png
"""
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

models_dir, runs_u, runs_l, table, out = sys.argv[1:6]
pp = sys.argv[6:9] if len(sys.argv) >= 9 else None   # runs_pp_uniform runs_pp_layered tableP
PPD = float(sys.argv[9]) if len(sys.argv) > 9 else 110.0   # PP reference distance (PKIKP sits in the window for >110)
ZC = 522.9705


def tvel(f, T):
    d = np.loadtxt(os.path.join(models_dir, f"MM{f:.1f}", f"MM{f:.1f}_{T}.tvel"), skiprows=2)
    return d[:, 0], d[:, 2]


def spliced(f1, T1, f2, T2):
    z1, v1 = tvel(f1, T1); z2, v2 = tvel(f2, T2)
    m1 = z1 <= ZC; m2 = z2 >= ZC
    return np.r_[z1[m1], ZC, ZC, z2[m2]], np.r_[v1[m1], np.interp(ZC, z1, v1), np.interp(ZC, z2, v2), v2[m2]]


def trace(path, tmin=-340, tmax=-110):
    d = np.loadtxt(path)
    t, a = d[:, 0], d[:, 2]
    m = (t > tmin) & (t < tmax)
    return t[m], a[m]


tab = np.load(table)
j = np.argmin(np.abs(tab["dist"] - 124.0))
dt = tab["dt"][:, j]; ok = np.isfinite(dt)
def depth(t): return np.interp(t, dt[ok][::-1], tab["depth"][ok][::-1])
if pp:
    tabP = np.load(pp[2]); jP = np.argmin(np.abs(tabP["dist"] - PPD)); dtP = tabP["dt"][:, jP]; okP = np.isfinite(dtP)
    def depthP(t): return np.interp(t, dtP[okP][::-1], tabP["depth"][okP][::-1])

cases = [
    ("M0  uniform MTZ\n(f = 0.2, one T)", [(("M0", 0.2, T), f"T = {T} K", c) for T, c in ((1500, "C0"), (1700, "k"), (1900, "C3"))]),
    ("M1  thermal layering at 523 km\n(f = 0.2 both layers, T1 = 1700 K)",
     [(("M1", 0.2, 1700, 0.2, T2), f"T2 = {T2} K", c) for T2, c in ((1500, "C0"), (1700, "k"), (1900, "C3"))]),
    ("M2  thermochemical layering at 523 km\n(T = 1700 K both layers, f1 = 0.2)",
     [(("M2", 0.2, 1700, f2, 1700), f"f2 = {f2}", c) for f2, c in ((0.0, "C0"), (0.2, "k"), (0.4, "C1"), (0.6, "C3"))]),
]

nrow = 3 if pp else 2
fig, axs = plt.subplots(nrow, 3, figsize=(13, 4.5 * nrow), gridspec_kw=dict(height_ratios=[1.1] + [1] * (nrow - 1)))
for col, (title, items) in enumerate(cases):
    axp, axt = axs[0, col], axs[1, col]
    for spec, lab, c in items:
        if spec[0] == "M0":
            _, f, T = spec; z, v = tvel(f, T)
            run = os.path.join(runs_u, f"MM{f:.1f}_{T}", "post", "xsec_124.0.txt")
        else:
            _, f1, T1, f2, T2 = spec; z, v = spliced(f1, T1, f2, T2)
            run = os.path.join(runs_l, f"MM{f1:.1f}-{f2:.1f}_{T1}-{T2}_zc523_w0", "post", "xsec_124.0.txt")
        lw = 1.8 if c == "k" else 1.2
        axp.plot(v, z, color=c, lw=lw, label=lab)
        t, a = trace(run)
        axt.plot(a, depth(t), color=c, lw=lw, label=lab)
        if pp:
            runp = run.replace(runs_u, pp[0]).replace(runs_l, pp[1]).replace("xsec_124.0", f"xsec_{PPD:.1f}")
            t, a = trace(runp, -180, -50)
            axs[2, col].plot(a, depthP(t), color=c, lw=lw, label=lab)
    axp.set_ylim(800, 300); axp.set_xlim(4.4, 5.9); axp.set_title(title, fontsize=10)
    axp.axhline(ZC, color="0.6", ls=":", lw=0.8)
    axp.set_xlabel("Vs (km/s)"); axp.legend(fontsize=8, loc="lower left"); axp.grid(alpha=0.3)
    axt.set_ylim(800, 350); axt.axvline(0, color="0.7", lw=0.6)
    for zz in (410, 520, 660): axt.axhline(zz, color="0.75", ls=":", lw=0.8)
    axt.set_xlabel("SdS / SS amplitude"); axt.grid(alpha=0.3)
    axt.set_title("Yspec synthetic, SS at 124°, 15–50 s", fontsize=9)
    if pp:
        axp2 = axs[2, col]; axp2.set_ylim(800, 350); axp2.axvline(0, color="0.7", lw=0.6)
        for zz in (410, 520, 660): axp2.axhline(zz, color="0.75", ls=":", lw=0.8)
        axp2.set_xlabel("PdP / PP amplitude"); axp2.grid(alpha=0.3); axp2.set_title(f"Yspec synthetic, PP at {PPD:.0f}°, 8–50 s", fontsize=9)
axs[0, 0].set_ylabel("depth (km)"); axs[1, 0].set_ylabel("PREM depth (km)")
if pp: axs[2, 0].set_ylabel("PREM depth (km)")

fig.tight_layout(); fig.savefig(out, dpi=140); print("saved", out)
