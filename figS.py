#!/usr/bin/env python3
"""
Supplementary figures S2, S3, S4 at print size (180 mm wide, 8-pt text) — print-size versions.

  python figS.py S2        Vs profiles of the model library            needs models/            -> figs/figS2.png/.pdf
  python figS.py S3        synthetic catalogue vs temperature          needs catalogue/catalogue.csv, table_SS.npz
  python figS.py S4        pick-sensitivity figure from the CSV         needs figs/figS4_picks.csv (made by suppfigs.py figS4_picks)
  python figS.py all

Run from mtzsyn/ (or mtzsyn/code/), any env with numpy + matplotlib (+ pandas for S3).
"""
import os, sys, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "prem_sds_S.npz")) else os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, ROOT)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "font.family": "sans-serif"})
CFG = dict(models="models", catalogue="catalogue/catalogue.csv", table="prem_sds_S.npz", picks="figs/figS4_picks.csv", zc=522.9705)
C = {"M0": "#4c78a8", "M1": "#3c9a5a", "M2": "#c8322b"}


def save(name, fig):
    os.makedirs("figs", exist_ok=True); fig.savefig(f"figs/{name}.png", dpi=300); fig.savefig(f"figs/{name}.pdf"); print("saved figs/" + name); plt.close(fig)


# ----------------------------------------------------------------------------- S2: the model library
def tvel(f, T):
    return np.loadtxt(os.path.join(CFG["models"], f"MM{f:.1f}", f"MM{f:.1f}_{T}.tvel"), skiprows=2)   # z, Vp, Vs, rho


def layered(f1, T1, f2, T2, w=0):
    from splice_layers import splice
    d = splice(tvel(f1, T1), tvel(f2, T2), CFG["zc"], w); return d[:, 0], d[:, 2]


def S2():
    Ts = list(range(1300, 2001, 100)); cmT = plt.get_cmap("coolwarm"); fs = [0.0, 0.2, 0.4, 0.6]
    fig = plt.figure(figsize=(7.2, 6.0)); gs = gridspec.GridSpec(2, 12, figure=fig, left=0.07, right=0.985, top=0.95, bottom=0.07, hspace=0.4, wspace=1.2)
    # row 1: uniform models, one panel per basalt fraction, coloured by temperature
    for k, f in enumerate(fs):
        ax = fig.add_subplot(gs[0, 3 * k:3 * k + 3])
        for T in Ts:
            d = tvel(f, T); ax.plot(d[:, 2], d[:, 0], color=cmT((T - 1300) / 700), lw=0.9)
        ax.set_ylim(800, 300); ax.set_xlim(4.4, 6.1); ax.grid(alpha=0.15, lw=0.4); ax.set_title(f"{'abcd'[k]}   uniform, f = {f:.1f}", loc="left", fontweight="bold")
        ax.set_xlabel("V$_s$ (km s$^{-1}$)"); ax.set_ylabel("depth (km)" if k == 0 else ""); ax.tick_params(labelleft=k == 0)
        for z in (410, 523, 660): ax.axhline(z, color="0.6", lw=0.5, ls=":")
    sm = plt.cm.ScalarMappable(cmap=cmT, norm=plt.Normalize(1300, 2000)); cax = ax.inset_axes([0.07, 0.1, 0.45, 0.03]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_ticks([]); cax.set_title("T$_p$ = 1300 – 2000 K", fontsize=6.5, pad=2)
    # row 2: layered models
    row2 = []
    ax = fig.add_subplot(gs[1, 0:4]); row2.append(ax); cm = plt.get_cmap("Greens")
    for i, T2 in enumerate((1500, 1600, 1700, 1800, 1900)):
        z, v = layered(0.2, 1700, 0.2, T2); ax.plot(v, z, color=cm(0.35 + 0.15 * i), lw=1.0, label=f"T$_2$ = {T2} K")
    ax.set_title("e   M1: f = 0.2, T$_1$ = 1700 K", loc="left", fontweight="bold", color=C["M1"]); ax.legend(frameon=False, loc="lower left", handlelength=1.2)
    ax = fig.add_subplot(gs[1, 4:8]); row2.append(ax); cm = plt.get_cmap("Reds")
    for i, f2 in enumerate((0.0, 0.2, 0.4, 0.6)):
        z, v = layered(0.2, 1700, f2, 1700); ax.plot(v, z, color=cm(0.35 + 0.2 * i), lw=1.0, label=f"f$_2$ = {f2:.1f}")
    ax.set_title("f   M2: T = 1700 K, f$_1$ = 0.2", loc="left", fontweight="bold", color=C["M2"]); ax.legend(frameon=False, loc="lower left", handlelength=1.2)
    ax = fig.add_subplot(gs[1, 8:12]); row2.append(ax)
    z, v = layered(0.2, 1700, 0.5, 1700, 0); ax.plot(v, z, color=C["M2"], lw=1.0, label="sharp interface")
    z, v = layered(0.2, 1700, 0.5, 1700, 40); ax.plot(v, z, color=C["M2"], lw=1.0, ls="--", label="40-km gradient")
    ax.set_title("g   interface width (M2, f$_2$ = 0.5)", loc="left", fontweight="bold"); ax.legend(frameon=False, loc="lower right", handlelength=1.6)
    for k, ax in enumerate(row2):
        ax.set_ylim(800, 300); ax.set_xlim(4.4, 6.1)
        if k == 2: ax.set_ylim(640, 440); ax.set_xlim(5.0, 5.7)
        ax.grid(alpha=0.15, lw=0.4); ax.set_xlabel("V$_s$ (km s$^{-1}$)"); ax.set_ylabel("depth (km)" if k != 1 else ""); ax.tick_params(labelleft=k != 1)
        for z in (410, 523, 660): ax.axhline(z, color="0.6", lw=0.5, ls=":")
    save("figS2", fig)


# ----------------------------------------------------------------------------- S3: synthetic catalogue
def depth_fn(table, dist):
    tab = np.load(table); j = np.argmin(np.abs(tab["dist"] - dist)); dt = tab["dt"][:, j]; ok = np.isfinite(dt)
    return lambda t: np.interp(t, dt[ok][::-1], tab["depth"][ok][::-1])


def S3():
    import pandas as pd
    cat = pd.read_csv(CFG["catalogue"]); cat = cat[(cat.f <= 0.6) & (cat["T"] >= 1300) & (cat["T"] <= 2000)]; dep = depth_fn(CFG["table"], 124)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7)); fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.3, wspace=0.45)
    cols = plt.cm.viridis(np.linspace(0, 0.95, 7))
    for f, c in zip(sorted(cat.f.unique()), cols):
        s = cat[cat.f == f].sort_values("T")
        axs[0].plot(s["T"], dep(s.t410.values), "-o", ms=2.5, lw=0.9, color=c, label=f"f = {f:.1f}"); axs[0].plot(s["T"], dep(s.t660.values), "-s", ms=2.5, lw=0.9, color=c)
        axs[1].plot(s["T"], dep(s.t520.values), "-o", ms=2.5, lw=0.9, color=c); axs[2].plot(s["T"], s.a660 / s.a410, "-o", ms=2.5, lw=0.9, color=c)
    for ax, t in zip(axs, ("a   410 (circles), 660 (squares)", "b   mid-MTZ arrival", "c   660/410 amplitude ratio")):
        ax.set_title(t, loc="left", fontweight="bold"); ax.set_xlabel("potential temperature (K)"); ax.grid(alpha=0.15, lw=0.4); ax.set_xticks([1300, 1500, 1700, 1900])
    axs[0].axhline(413, color="k", ls=":", lw=0.8); axs[0].axhline(658, color="k", ls=":", lw=0.8); axs[0].set_ylabel("apparent depth at 124° (km)"); axs[0].invert_yaxis()
    axs[1].axhline(528, color="k", ls=":", lw=0.8); axs[1].set_ylabel("apparent depth at 124° (km)"); axs[1].invert_yaxis(); axs[2].set_ylabel("a$_{660}$ / a$_{410}$")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=1.5)
    save("figS3", fig)


# ----------------------------------------------------------------------------- S4: pick sensitivity (from the CSV written by pick_sensitivity.py)
def S4():
    import csv
    rows = list(csv.DictReader(open(CFG["picks"])))
    def col(k): return np.array([float(r[k]) for r in rows])
    def scol(k): return np.array([r[k] for r in rows])
    fac = [("search window", scol("window"), lambda v: v.split()[-1] + " km"), ("minimum width", col("width_km"), lambda v: f"{v:g} km"), ("significance", col("snr"), lambda v: f"{v:g} σ"),
           ("polarity", col("both_polarities"), lambda v: "both" if v else "positive"), ("prominence", col("prominence"), lambda v: f"{v:g}")]
    ycols = [("frac_any", "a   caps with a mid-MTZ arrival"), ("frac_split", "b   caps with two arrivals (splitting)"), ("frac_negative", "c   caps with a negative arrival")]
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True); fig.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.12, hspace=0.35)
    xs, labs, groups = [], [], []; x = 0
    for name, v, fmt in fac:
        levels = sorted(set(v)); groups.append((x, x + len(levels) - 1, name))
        for lv in levels: xs.append((x, v == lv)); labs.append(fmt(lv)); x += 1
        x += 1
    for ax, (yk, t) in zip(axs, ycols):
        y = col(yk); n = len(y)
        for (xi, m) in xs:
            yy = y[m]; ax.plot([xi, xi], [yy.min(), yy.max()], color="0.75", lw=5, solid_capstyle="butt"); ax.plot([xi - 0.3, xi + 0.3], [np.median(yy)] * 2, color="k", lw=1.4)
        ax.set_ylim(0, 1); ax.set_ylabel("fraction of 10° caps"); ax.grid(alpha=0.15, lw=0.4, axis="y"); ax.set_title(f"{t}: {y.min() * 100:.0f} – {y.max() * 100:.0f} % over the {n} settings", loc="left", fontweight="bold")
    ax = axs[-1]; ax.set_xticks([xi for xi, _ in xs]); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7); ax.set_xlim(-0.8, x - 1.2)
    for x0, x1, name in groups:
        ax.annotate(name, xy=((x0 + x1) / 2, -0.42), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=7.5, fontweight="bold", annotation_clip=False)
        ax.plot([x0 - 0.4, x1 + 0.4], [-0.36, -0.36], color="0.4", lw=0.8, transform=ax.get_xaxis_transform(), clip_on=False)
    axs[0].text(0.99, 0.06, "bar: range over all other settings;  tick: median", transform=axs[0].transAxes, ha="right", fontsize=7, color="0.3")
    save("figS4", fig)


if __name__ == "__main__":
    for a in (sys.argv[1:] or ["all"]):
        [S2, S3, S4][int(a[1]) - 2]() if a != "all" else [f() for f in (S2, S3, S4)]
