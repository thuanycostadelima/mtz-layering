#!/usr/bin/env python3
"""
Figure 4 — the result at 10°: smooth maps of the class probabilities and of the posterior parameters.

  (a) P(M2): probability that compositional layering is required     (b) P(M0): a uniform transition zone suffices
  (c) expected basalt contrast P(M2)·Δf across 523 km                 (d) transition-zone temperature (M0 posterior mean)

Fields are interpolated on a 1° grid with Gaussian weights (width 0.6 × cap radius) and masked beyond one radius from any cap centre.
Run in the pygmt env (cartopy):  python fig4.py [--no-coast]   (from mtzsyn/ or mtzsyn/code/)  ->  figs/fig4.png / .pdf
Input: the fields CSV from export_fields.py (lat, lon, P_M2, P_M1, dT_M1, df_M2, E_df, T_M0, f_M0).
"""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "prem_sds_S.npz")) else os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, ROOT)
from maps import add_trenches, add_paleo_trenches

CFG = dict(fields=["v7_SS10_fields.csv"], results=["v7_SS10_results.csv"], radius=10.0, out="figs/fig4")
C_CLASS = {"M0": "#4c78a8", "M1": "#3c9a5a", "M2": "#c8322b"}
PROJ = ccrs.Robinson(central_longitude=180); PC = ccrs.PlateCarree()
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "font.family": "sans-serif"})


def gridfield(lat, lon, val, width, maxdist):
    glo = np.arange(-180, 181, 1.0); gla = np.arange(-86, 87, 1.0); GLO, GLA = np.meshgrid(glo, gla)
    la1 = np.radians(GLA)[..., None]; lo1 = np.radians(GLO)[..., None]; la2 = np.radians(lat); lo2 = np.radians(lon)
    d = np.degrees(np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1)))
    W = np.exp(-0.5 * (d / width) ** 2); F = (W * val).sum(-1) / np.maximum(W.sum(-1), 1e-12)
    return GLO, GLA, np.ma.masked_where(d.min(-1) > maxdist, F)


def panel(fig, ax, df, col, r, cmap, vmin, vmax, label, title):
    ax.set_global(); ax.set_facecolor("white"); ax.gridlines(lw=0.2, ls="--", color="0.85")
    if "--no-coast" not in sys.argv: ax.coastlines(lw=0.35, color="0.35", zorder=6)
    GLO, GLA, F = gridfield(df["lat"], df["lon"], df[col], 0.6 * r, r)
    pc = ax.pcolormesh(GLO, GLA, F, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", transform=PC, zorder=3, rasterized=True)
    ax.scatter(df["lon"], df["lat"], s=1.2, c="k", alpha=0.3, transform=PC, zorder=7, lw=0); add_trenches(ax, lw=0.8, color="0.1")
    cb = fig.colorbar(pc, ax=ax, orientation="horizontal", shrink=0.55, pad=0.03, aspect=35); cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7, length=2)
    ax.set_title(title, loc="left", fontweight="bold")
    return cb


def main():
    path = next(p for p in CFG["fields"] if os.path.exists(p)); print("fields:", path)
    df = np.genfromtxt(path, delimiter=",", names=True); r = CFG["radius"]
    df = {k: df[k] for k in df.dtype.names}; df["P_M0"] = 1 - df["P_M1"] - df["P_M2"]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), subplot_kw=dict(projection=PROJ)); fig.subplots_adjust(left=0.01, right=0.91, top=0.96, bottom=0.03, wspace=0.04, hspace=0.0)
    panel(fig, axs[0, 0], df, "P_M2", r, "YlOrRd", 0, 1, "P(M2): compositional layering required", "a   compositional layering across 523 km")
    # (b) assigned class per cap (Bayes-factor thresholds, as in the text)
    ax = axs[0, 1]; ax.set_global(); ax.set_facecolor("white"); ax.gridlines(lw=0.2, ls="--", color="0.85")
    if "--no-coast" not in sys.argv: ax.coastlines(lw=0.35, color="0.35", zorder=6)
    sm = add_paleo_trenches(ax, lw=0.6, zorder=6.5); add_trenches(ax, lw=0.8, color="0.1"); rp = next((p for p in CFG["results"] if os.path.exists(p)), None)
    cax = ax.inset_axes([1.015, 0.3, 0.018, 0.4]); cb = fig.colorbar(sm, cax=cax); cb.ax.tick_params(labelsize=6, length=2); cb.set_ticks([0, 100, 200]); cb.set_ticklabels(["0", "100", "200 Ma"])
    if rp:
        R = np.genfromtxt(rp, delimiter=",", names=True, dtype=None, encoding=None); cls = R["class"] if "class" in R.dtype.names else np.where(R["lnB21"] > 3, "M2", np.where(R["lnB10"] > 3, "M1", "M0"))
        for c in ("M0", "M1", "M2"):
            k = cls == c; ax.scatter(R["lon"][k], R["lat"][k], s=9, c=C_CLASS[c], edgecolors="k", linewidths=0.15, transform=PC, zorder=8, label=f"{c}  ({k.mean() * 100:.0f}%)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.2, markerscale=1.5)
    ax.set_title("b   class assigned to each cap", loc="left", fontweight="bold")
    panel(fig, axs[1, 0], df, "E_df", r, "YlOrBr", 0, 0.3, "P(M2) · Δf: expected basalt-fraction contrast (f$_2$ − f$_1$)", "c   basalt enrichment of the lower MTZ")
    panel(fig, axs[1, 1], df, "T_M0", r, "coolwarm", 1500, 1900, "potential temperature (K), uniform-MTZ posterior mean", "d   MTZ temperature")
    os.makedirs(os.path.dirname(CFG["out"]), exist_ok=True)
    fig.savefig(CFG["out"] + ".png", dpi=300); fig.savefig(CFG["out"] + ".pdf"); print("saved", CFG["out"])


def main_m1():
    """--m1: the thermal-layering class, same layout as Fig. 4a and c: P(M1) and the expected thermal contrast P(M1)·ΔT (T2 − T1)"""
    path = next(p for p in CFG["fields"] if os.path.exists(p)); df = np.genfromtxt(path, delimiter=",", names=True); r = CFG["radius"]
    df = {k: df[k] for k in df.dtype.names}; df["E_dT"] = df["P_M1"] * df["dT_M1"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), subplot_kw=dict(projection=PROJ)); fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.03, wspace=0.04)
    panel(fig, axs[0], df, "P_M1", r, "YlGnBu", 0, 1, "P(M1): thermal layering required", "a   thermal layering across 523 km")
    panel(fig, axs[1], df, "E_dT", r, "coolwarm", -150, 150, "P(M1) · ΔT: expected temperature contrast (T$_2$ − T$_1$, K)", "b   thermal contrast across 523 km")
    for ax in axs: add_trenches(ax, lw=0.8, color="0.1")
    out = CFG["out"] + "_M1"; fig.savefig(out + ".png", dpi=300); fig.savefig(out + ".pdf"); print("saved", out)


if __name__ == "__main__":
    if "--m1" in sys.argv: main_m1(); sys.exit()
    main()
