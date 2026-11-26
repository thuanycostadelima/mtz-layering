#!/usr/bin/env python3
"""
Figure 5 — the result at every cap radius, and why the fraction of caps requiring layering is a lower bound.

  (a) P(M2) at 5, 7.5, 10, 12.5, 15, 20 deg (smooth fields, same colour scale)
  (b) fraction of caps assigned M0 / M1 / M2 and mean P(M2) as a function of cap radius
  (c) fraction of caps assigned M2 as a function of the number of records in the cap (quartiles), at each radius

Run in the pygmt env:  python fig5.py [--no-coast]  (from mtzsyn/ or mtzsyn/code/)  ->  figs/fig5.png / .pdf
Inputs: v7_SS<r>_fields.csv and v7_SS<r>_results.csv from export_fields.py.
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "prem_sds_S.npz")) else os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, ROOT)
from maps import add_trenches, add_paleo_trenches

CFG = dict(prefix="v7_SS", radii=[5, 7.5, 10, 12.5, 15, 17.5, 20], map_radii=[5, 7.5, 10, 12.5, 15, 20], thr=3.0, out="figs/fig5")
C_CLASS = {"M0": "#4c78a8", "M1": "#3c9a5a", "M2": "#c8322b"}; PROJ = ccrs.Robinson(central_longitude=180); PC = ccrs.PlateCarree()
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "font.family": "sans-serif"})


def gridfield(lat, lon, val, width, maxdist):
    glo = np.arange(-180, 181, 1.0); gla = np.arange(-86, 87, 1.0); GLO, GLA = np.meshgrid(glo, gla)
    la1 = np.radians(GLA)[..., None]; lo1 = np.radians(GLO)[..., None]; la2 = np.radians(lat); lo2 = np.radians(lon)
    d = np.degrees(np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1)))
    W = np.exp(-0.5 * (d / width) ** 2); F = (W * val).sum(-1) / np.maximum(W.sum(-1), 1e-12)
    return GLO, GLA, np.ma.masked_where(d.min(-1) > maxdist, F)


def load(r):
    F = np.genfromtxt(f"{CFG['prefix']}{r:g}_fields.csv", delimiter=",", names=True); R = np.genfromtxt(f"{CFG['prefix']}{r:g}_results.csv", delimiter=",", names=True, dtype=None, encoding=None)
    cls = np.where(R["lnB21"] > CFG["thr"], 2, np.where(R["lnB10"] > CFG["thr"], 1, 0)); return F, R, cls


def main():
    data = {r: load(r) for r in CFG["radii"]}
    fig = plt.figure(figsize=(7.2, 6.4)); gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.18, wspace=0.03, left=0.01, right=0.99, top=0.97, bottom=0.56)
    # (a) maps
    pc = None
    for k, r in enumerate(CFG["map_radii"]):
        F, R, cls = data[r]; ax = fig.add_subplot(gs[k // 3, k % 3], projection=PROJ); ax.set_global(); ax.gridlines(lw=0.15, ls="--", color="0.85")
        if "--no-coast" not in sys.argv: ax.coastlines(lw=0.3, color="0.4", zorder=6)
        GLO, GLA, Z = gridfield(F["lat"], F["lon"], F["P_M2"], 0.6 * r, r); pc = ax.pcolormesh(GLO, GLA, Z, cmap="YlOrRd", vmin=0, vmax=1, shading="auto", transform=PC, zorder=3, rasterized=True)
        ax.scatter(F["lon"], F["lat"], s=0.4, c="k", alpha=0.25, lw=0, transform=PC, zorder=7); add_paleo_trenches(ax, lw=0.35, zorder=6.5); add_trenches(ax, lw=0.6, color="0.1"); ax.set_title(f"{'a   ' if k == 0 else ''}{r:g}° ({len(F)} caps): M2 in {np.mean(cls == 2) * 100:.0f}%", loc="left", fontweight="bold" if k == 0 else "normal", fontsize=8)
    cax = fig.add_axes([0.36, 0.53, 0.28, 0.013]); cb = fig.colorbar(pc, cax=cax, orientation="horizontal"); cb.set_label("P(M2): compositional layering required", labelpad=2); cb.ax.tick_params(labelsize=7, length=2)
    # (b) class fractions vs radius
    rs = CFG["radii"]
    ax = fig.add_axes([0.08, 0.13, 0.36, 0.3])
    fr = np.array([[np.mean(data[r][2] == c) * 100 for c in (0, 1, 2)] for r in rs]); mp = [data[r][0]["P_M2"].mean() for r in rs]
    for c, lab in enumerate(("M0", "M1", "M2")): ax.plot(rs, fr[:, c], "o-", color=C_CLASS[lab], lw=1.3, ms=4, label=f"{lab} assigned")
    ax.set_xlabel("cap radius (°)"); ax.set_ylabel("caps (%)"); ax.set_ylim(0, 100); ax.grid(alpha=0.15, lw=0.4); ax.set_xticks(rs); ax.set_xticklabels([f"{r:g}" for r in rs])
    ax2 = ax.twinx(); ax2.plot(rs, mp, "s--", color="k", lw=1, ms=4, label="mean P(M2)"); ax2.set_ylim(0, 1); ax2.set_ylabel("mean P(M2)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False, ncol=2, columnspacing=1.5, handlelength=1.5)
    ax.set_title("b   classification versus cap radius", loc="left", fontweight="bold")
    # (c) M2 fraction vs records per cap
    ax = fig.add_axes([0.6, 0.13, 0.37, 0.3]); cmap = plt.get_cmap("viridis", len(rs))
    for i, r in enumerate(rs):
        F, R, cls = data[r]; n = R["n_records"]; q = np.percentile(n, [0, 25, 50, 75, 100]); idx = np.clip(np.digitize(n, q) - 1, 0, 3)
        x = [np.median(n[idx == k]) for k in range(4)]; y = [np.mean(cls[idx == k] == 2) * 100 for k in range(4)]; ax.plot(x, y, "o-", color=cmap(i), lw=1.2, ms=4, label=f"{r:g}°")
    ax.set_xscale("log"); ax.set_xlabel("records per cap (median of each quartile)"); ax.set_ylabel("caps with M2 required (%)"); ax.grid(alpha=0.15, lw=0.4, which="both")
    ax.legend(title="cap radius", ncol=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), columnspacing=0.6, handlelength=1.2, handletextpad=0.3); ax.set_ylim(0, 60)
    ax.set_title("c   detection versus records per cap", loc="left", fontweight="bold")
    os.makedirs(os.path.dirname(CFG["out"]), exist_ok=True); fig.savefig(CFG["out"] + ".png", dpi=300); fig.savefig(CFG["out"] + ".pdf"); print("saved", CFG["out"])


if __name__ == "__main__":
    main()
