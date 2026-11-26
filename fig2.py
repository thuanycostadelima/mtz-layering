#!/usr/bin/env python3
"""
Figure 2 — forward modelling (Nature Communications double column, 183 mm).

  (a–c) the three model classes: Vs profiles (left) and SS synthetic stacks at 124° (right) for M0 / M1 / M2
  (d) one cap: observed stack vs the single 124° synthetic vs the synthetic stacked with the cap's real geometry
  (e) 660/410 amplitude ratio of the real-geometry stack relative to the single trace, all caps
  (f) effect of the real source mechanisms and depths, before and after the per-cap filter, SS and PP

Run in the mtz520 env (no cartopy needed):  python fig2.py   (from mtzsyn/ or mtzsyn/code/)  ->  figs/fig2.png / .pdf
"""
import os, sys, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "prem_sds_S.npz")) else os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, ROOT)

CFG = dict(models="models", runs_u="runs_uniform", runs_l="runs_layered", table_S="prem_sds_S.npz",
           obs_ss="crosssections_2024_SNR/SS/10deg_caps", inputs_ss="crosssections_2024_SNR/SSinputs/10deg_caps",
           stacks_real="calib_stacks_v7/SS10", stacks_fixed="calib_stacks_v7/SS10_fixed", mech="mech_filters_SS10_v7.npz",
           single="runs_uniform/MM0.2_1600/post/xsec_124.0.txt", cap="0.0_27.0", out="figs/fig2")
ZC = 522.9705; C_SS = "#17405f"; C_PP = "#c8641e"; C_CLASS = {"M0": "#4c78a8", "M1": "#3c9a5a", "M2": "#c8322b"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "font.family": "sans-serif"})


def depth_fn(table, dist):
    tab = np.load(table); j = np.argmin(np.abs(tab["dist"] - dist)); dt = tab["dt"][:, j]; ok = np.isfinite(dt)
    return lambda t: np.interp(t, dt[ok][::-1], tab["depth"][ok][::-1])


def tvel(f, T):
    d = np.loadtxt(os.path.join(CFG["models"], f"MM{f:.1f}", f"MM{f:.1f}_{T}.tvel"), skiprows=2); return d[:, 0], d[:, 2]


def spliced(f1, T1, f2, T2):
    z1, v1 = tvel(f1, T1); z2, v2 = tvel(f2, T2); m1 = z1 <= ZC; m2 = z2 >= ZC
    return np.r_[z1[m1], ZC, ZC, z2[m2]], np.r_[v1[m1], np.interp(ZC, z1, v1), np.interp(ZC, z2, v2), v2[m2]]


def pk(x, t, lo, hi):
    w = (t > lo) & (t < hi); k = np.argmax(x[w]); return x[w][k], t[w][k]


def mech_residuals(real, fixed, filt, tg, win):
    z = np.load(filt); L = int(z["L"]); H = dict(zip(z["names"], z["h"])); out = []
    for f in sorted(glob.glob(os.path.join(fixed, "*.txt"))):
        nm = os.path.basename(f)
        if nm == "summary.txt" or nm[:-4] not in H or not os.path.exists(os.path.join(real, nm)): continue
        fx = np.loadtxt(f); fx = np.interp(tg, fx[:, 0], fx[:, 2]); rc = np.loadtxt(os.path.join(real, nm)); rc = np.interp(tg, rc[:, 0], rc[:, 2])
        X = np.array([np.roll(fx, k) for k in range(-L, L + 1)]).T; v = lambda y: np.sum((y[win] - rc[win]) ** 2) / np.sum(rc[win] ** 2)
        out.append((v(fx), v(X @ H[nm[:-4]])))
    return np.array(out) * 100


def find(path, *cands):
    for p in (path,) + cands:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"none of {(path,) + cands} found in {ROOT}")


def main():
    dep = depth_fn(CFG["table_S"], 124.0)
    CFG["mech"] = find(CFG["mech"], "mech_filters.npz", "mech_filters_10.npz", "mech_filters_SS10deg.npz")
    fig = plt.figure(figsize=(7.2, 8.2))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.1, 1.0], hspace=0.5, wspace=0.36, left=0.075, right=0.98, top=0.965, bottom=0.19)

    # ---------------------------------------------------------------- (a–c) model classes
    cases = [("M0", "a   M0: uniform MTZ", "f = 0.2", [((0.2, T), f"{T} K", c) for T, c in ((1500, "#9ecae1"), (1700, "#2b6ca3"), (1900, "#08306b"))]),
             ("M1", "b   M1: thermal layering", "f = 0.2, T$_1$ = 1700 K, T$_2$ =",
              [((0.2, 1700, 0.2, T2), f"{T2} K", c) for T2, c in ((1500, "#a1d99b"), (1700, "#2e8b57"), (1900, "#00441b"))]),
             ("M2", "c   M2: thermochemical", "T = 1700 K, f$_1$ = 0.2, f$_2$ =",
              [((0.2, 1700, f2, 1700), f"{f2}", c) for f2, c in ((0.0, "#fcae91"), (0.2, "#e34a33"), (0.4, "#b30000"), (0.6, "#67000d"))])]
    for col, (cls, title, sub, items) in enumerate(cases):
        sg = gs[0, col].subgridspec(1, 2, width_ratios=[0.5, 1], wspace=0.08); axv = fig.add_subplot(sg[0]); ax = fig.add_subplot(sg[1], sharey=axv)
        for spec, lab, c in items:
            if cls == "M0":
                f, T = spec; z, v = tvel(f, T); run = os.path.join(CFG["runs_u"], f"MM{f:.1f}_{T}", "post", "xsec_124.0.txt")
            else:
                f1, T1, f2, T2 = spec; z, v = spliced(f1, T1, f2, T2); run = os.path.join(CFG["runs_l"], f"MM{f1:.1f}-{f2:.1f}_{T1}-{T2}_zc523_w0", "post", "xsec_124.0.txt")
            d = np.loadtxt(run); t, a = d[:, 0], d[:, 2]; m = (t > -330) & (t < -100)
            ax.plot(a[m], dep(t[m]), color=c, lw=1.0, label=lab); axv.plot(v, z, color=c, lw=1.0)
        axv.set_ylim(720, 350); axv.set_xlim(4.5, 5.9); axv.set_xticks([4.5, 5.0, 5.5]); axv.set_xlabel("V$_s$ (km/s)", labelpad=1); axv.grid(alpha=0.15, lw=0.4)
        for zz in (410, 520, 660): axv.axhline(zz, color="0.55", ls=(0, (2, 2)), lw=0.6); ax.axhline(zz, color="0.55", ls=(0, (2, 2)), lw=0.6)
        if cls != "M0": axv.axhline(ZC, color=C_CLASS[cls], lw=1.2, alpha=0.6)
        ax.axvline(0, color="0.6", lw=0.5); ax.set_xlim(-0.05, 0.08); ax.set_xticks([-0.04, 0, 0.04, 0.08]); ax.tick_params(labelleft=False); ax.grid(alpha=0.15, lw=0.4)
        ax.set_xlabel("synthetic S$d$S / SS"); axv.set_title(title, loc="left", fontweight="bold")
        ax.legend(loc="upper center", bbox_to_anchor=(0.2, -0.2), ncol=len(items), title=sub, title_fontsize=7.5, handlelength=1.1, columnspacing=0.8, frameon=False)
        if col == 0: axv.set_ylabel("depth (km)")

    # ---------------------------------------------------------------- (d) one cap: single trace vs real-geometry stack vs data
    cap = CFG["cap"]; st = np.loadtxt(f"{CFG['stacks_real']}/{cap}.txt"); ob = np.loadtxt(f"{CFG['obs_ss']}/{cap}.txt"); ref = np.loadtxt(CFG["single"])
    m = (st[:, 0] > -330) & (st[:, 0] < -100); mr = (ref[:, 0] > -330) & (ref[:, 0] < -100); nrec = sum(1 for _ in open(f"{CFG['inputs_ss']}/{cap}.txt"))
    ax = fig.add_subplot(gs[1, 0])
    ax.fill_betweenx(dep(ob[m, 0]), ob[m, 2] - 2 * ob[m, 3], ob[m, 2] + 2 * ob[m, 3], color="0.85", lw=0); ax.plot(ob[m, 2], dep(ob[m, 0]), "k", lw=1.4, label="observed stack ±2σ")
    ax.plot(ref[mr, 2], dep(ref[mr, 0]), color="#7f7f7f", lw=1.1, ls="--", label="synthetic, single trace at 124°")
    ax.plot(st[m, 2], dep(st[m, 0]), color=C_SS, lw=1.2, label=f"synthetic stacked over the\ncap's {nrec} records")
    for zz in (410, 520, 660): ax.axhline(zz, color="0.55", ls=(0, (2, 2)), lw=0.6)
    ax.axvline(0, color="0.6", lw=0.5); ax.set_ylim(720, 350); ax.set_xlabel("S$d$S / SS"); ax.set_ylabel("depth (km), PREM at 124°"); ax.grid(alpha=0.15, lw=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, handlelength=1.6); la, lo = cap.replace("n", "-").split("_")
    ax.set_title(f"d   geometry: one cap", loc="left", fontweight="bold")

    # ---------------------------------------------------------------- (e) 410 and 660 amplitude ratios, all caps
    r410, _ = pk(ref[:, 2], ref[:, 0], -175, -140); r660, _ = pk(ref[:, 2], ref[:, 0], -240, -195); A4, A6 = [], []
    for f in sorted(glob.glob(f"{CFG['stacks_real']}/*.txt")):
        if "summary" in f: continue
        d = np.loadtxt(f); A4.append(pk(d[:, 2], d[:, 0], -175, -140)[0] / r410); A6.append(pk(d[:, 2], d[:, 0], -240, -195)[0] / r660)
    A4, A6 = np.array(A4), np.array(A6); ax = fig.add_subplot(gs[1, 1]); b = np.linspace(0.6, 1.6, 41)
    ax.hist(A4, bins=b, color="#9ecae1", lw=0, label=f"410  (median {np.median(A4):.2f})"); ax.hist(A6, bins=b, color=C_SS, alpha=0.75, lw=0, label=f"660  (median {np.median(A6):.2f})")
    ax.hist(A6 / A4, bins=b, histtype="step", color="k", lw=1.2, label=f"660/410 ratio  (median {np.median(A6 / A4):.2f},\nrange {np.min(A6 / A4):.2f}–{np.max(A6 / A4):.2f})")
    ax.axvline(1, color="0.3", lw=0.6); ax.set_xlabel("amplitude ratio:\nreal-geometry stack / single trace"); ax.set_ylabel("caps"); ax.grid(alpha=0.15, lw=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, handlelength=1.6); ax.set_title(f"e   geometry: all caps", loc="left", fontweight="bold")

    # ---------------------------------------------------------------- (f) mechanism effect and its correction
    tg = np.arange(-340.0, -70.0 + 1e-9, 1.0); win = (tg >= -300) & (tg <= -110); Rs = mech_residuals(CFG["stacks_real"], CFG["stacks_fixed"], CFG["mech"], tg, win)
    # (PP window would be -200..-30 s with fit -150..-60 s)
    Rp = None   # PP is not inverted and its filter is not used; SS only in panel f
    ax = fig.add_subplot(gs[1, 2]); b = np.linspace(0, 15, 31)
    ax.hist(Rs[:, 0], bins=b, color=C_SS, alpha=0.55, lw=0, label=f"SS, uncorrected  (median {np.median(Rs[:, 0]):.1f}%)"); ax.hist(Rs[:, 1], bins=b, histtype="step", color=C_SS, lw=1.4, label=f"SS, after filter  ({np.median(Rs[:, 1]):.1f}%)")
    if Rp is not None:
        ax.hist(Rp[:, 0], bins=b, color=C_PP, alpha=0.55, lw=0, label=f"PP, uncorrected  ({np.median(Rp[:, 0]):.1f}%)"); ax.hist(Rp[:, 1], bins=b, histtype="step", color=C_PP, lw=1.4, label=f"PP, after filter  ({np.median(Rp[:, 1]):.1f}%)")
    ax.set_xlabel("fixed vs real source\n(misfit, % of signal variance)"); ax.set_ylabel("caps"); ax.grid(alpha=0.15, lw=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, handlelength=1.6, ncol=1); ax.set_title("f   source mechanisms", loc="left", fontweight="bold")

    os.makedirs(os.path.dirname(CFG["out"]), exist_ok=True)
    fig.savefig(CFG["out"] + ".png", dpi=300); fig.savefig(CFG["out"] + ".pdf"); print("saved", CFG["out"])


if __name__ == "__main__":
    main()
