#!/usr/bin/env python3
"""
Map helpers used by fig1.py, fig4.py and fig5.py (add_trenches, add_paleo_trenches, load_gplates, add_paleo), and a
stand-alone four-panel diagnostic map of one inversion run (Mollweide, central meridian 180):
  (a) model class per cap (M0 uniform, M1 thermal layering, M2 thermochemical layering) over the T22 basalt
      fraction, with trenches and paleo-subduction zones
  (b) ln B(M2/M1): evidence for compositional vs thermal layering
  (c) thermal contrast across 523 km, dT = T2 - T1 (M1 posterior mean)
  (d) basalt enrichment of the lower MTZ, df = f2 - f1 (M2 posterior mean) where M2 is required

Requires: the results CSV from export_fields.py, SSall.txt (T22), trench_gmt.txt and trenches_<age>Ma.txt (make_trenches.py);
cartopy, and gplately for the paleo lines (skipped if unavailable).

Usage:  python maps.py v7_SS10_results.csv figs/maps_10deg [--no-paleo] [--radius 10]
        python maps.py v7_SS10_results.csv v7_SS15_results.csv figs/maps_all   (several radii, marker size by radius)
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings; warnings.filterwarnings("ignore")

PROJ = ccrs.Mollweide(central_longitude=180)
PC = ccrs.PlateCarree()
FS = 10
CLS_COL = {"M0": "#3b76b5", "M1": "#3a9d3a", "M2": "#c8322c"}
CLS_LAB = {"M0": "uniform MTZ adequate", "M1": "thermal layering", "M2": "thermochemical layering"}
SIZE = {5: 10, 7.5: 16, 10: 26, 12.5: 34, 15: 44, 17.5: 54, 20: 64}

PALEO_TIMES = [
    (  0, '#7ec8e3', 2.00, 0.98, 'solid'), ( 20, '#4aaac8', 1.75, 0.93, 'solid'), ( 40, '#2080aa', 1.55, 0.88, 'solid'),
    ( 60, '#0a5c8a', 1.35, 0.84, 'solid'), ( 80, '#084878', 1.15, 0.80, (0, (6, 3))), (100, '#063870', 1.00, 0.76, (0, (6, 3))),
    (120, '#052a60', 0.90, 0.72, (0, (5, 4))), (140, '#041e50', 0.80, 0.68, (0, (5, 4))), (160, '#031440', 0.72, 0.64, (0, (4, 5))),
    (180, '#020c30', 0.65, 0.60, (0, (4, 5))), (200, '#010620', 0.58, 0.56, (0, (3, 6))),
]


def load_gplates():
    try:
        import gplately
        g = gplately.DataServer("Muller2022")
        rot, topo, static = g.get_plate_reconstruction_files()
        return gplately.PlateReconstruction(rot, topo, static)
    except Exception as e:
        print(f"GPlates not available ({e}); paleo-subduction skipped"); return None


def add_paleo(ax, model, times=PALEO_TIMES):
    if model is None:
        return
    for age, color, lw, alpha, ls in reversed(times):
        try:
            pts = model.tessellate_subduction_zones(age, tessellation_threshold_radians=np.radians(1.0))
        except Exception as e:
            print(f"  {age} Ma failed: {e}"); continue
        if pts is None or len(pts) == 0:
            continue
        lons, lats = pts[:, 0], pts[:, 1]
        brk = np.where(np.hypot(np.diff(lons), np.diff(lats)) > 2.0)[0] + 1
        for seg in np.split(np.arange(len(lons)), brk):
            if len(seg) < 2:
                continue
            wrap = np.where(np.abs(np.diff(lons[seg])) > 180)[0] + 1
            for part in np.split(seg, wrap):
                if len(part) > 1:
                    ax.plot(lons[part], lats[part], color=color, lw=lw * 0.8, alpha=alpha, linestyle=ls, zorder=5, transform=PC)


def add_paleo_trenches(ax, ages=range(20, 201, 20), cmap="Blues_r", lw=0.7, zorder=9):
    """reconstructed subduction zones from trenches_<age>Ma.txt (make_trenches.py), light (old) to dark (young); returns a ScalarMappable for a colourbar"""
    import matplotlib.pyplot as plt, matplotlib.colors as mcolors
    ages = list(ages); cm = plt.get_cmap(cmap); norm = mcolors.Normalize(0, max(ages))
    for age in sorted(ages, reverse=True):
        add_trenches(ax, path=f"trenches_{age}Ma.txt", color=cm(norm(age)), lw=lw, zorder=zorder)
    return plt.cm.ScalarMappable(cmap=cm, norm=norm)


def add_trenches(ax, path="trench_gmt.txt", color="#111111", lw=1.2, zorder=10):
    if not os.path.exists(path):
        return
    seg = []
    def draw(s):
        if len(s) < 2: return
        s = np.array(s); brk = np.where(np.abs(np.diff(s[:, 0])) > 180)[0] + 1
        for part in np.split(s, brk):
            if len(part) > 1: ax.plot(part[:, 0], part[:, 1], color=color, lw=lw, alpha=0.9, zorder=zorder, transform=PC)
    for line in open(path):
        line = line.strip()
        if not line: continue
        if line.startswith(">"): draw(seg); seg = []
        else:
            p = line.split()
            try: seg.append([float(p[0]), float(p[1])])
            except ValueError: pass
    draw(seg)


def background(ax):
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f2f2f2", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="#e8eef4", zorder=0)
    ax.coastlines(linewidth=0.35, color="#777777", zorder=2)
    ax.gridlines(draw_labels=False, linestyle="--", linewidth=0.2, color="#bbbbbb", zorder=2)


def t22_basalt(ax, alpha):
    if not os.path.exists("SSall.txt"):
        return None
    d = pd.read_csv("SSall.txt", sep=r"\s+", header=None,
                    names=["lat", "lon", "temp", "basalt_frac", "d410", "a410", "d520", "a520", "d660", "a660"])
    return ax.scatter(d["lon"], d["lat"], c=d["basalt_frac"], cmap="pink_r", vmin=0, vmax=0.6, s=3, alpha=alpha, zorder=3,
                      transform=PC, rasterized=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+"); ap.add_argument("out")
    ap.add_argument("--no-paleo", action="store_true"); ap.add_argument("--lnb", type=float, default=3.0)
    a = ap.parse_args()
    frames = []
    for f in a.csv:
        d = pd.read_csv(f)
        r = float(os.path.basename(f).split("_")[-1].replace("deg.csv", "")) if "deg.csv" in f else 10.0
        d["radius"] = r; frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["cls"] = np.where(df["lnB21"] > a.lnb, "M2", np.where(df["lnB10"] > a.lnb, "M1", "M0"))
    size = df["radius"].map(SIZE).fillna(26).values
    model = None if a.no_paleo else load_gplates()

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), subplot_kw={"projection": PROJ})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.08, wspace=0.05, hspace=0.18)
    # (a) classes
    ax = axes[0, 0]; background(ax); t22_basalt(ax, 0.18); add_trenches(ax); add_paleo(ax, model)
    for c in ("M0", "M1", "M2"):
        m = df["cls"] == c
        ax.scatter(df["lon"][m], df["lat"][m], s=size[m], c=CLS_COL[c], edgecolors="k", linewidths=0.3, zorder=8, transform=PC,
                   label=f"{CLS_LAB[c]} ({m.sum()})")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9, bbox_to_anchor=(0.0, -0.13), ncol=3)
    ax.set_title(f"(a) preferred model class (ln B > {a.lnb:g}); background: T22 basalt fraction; lines: trenches and paleo-subduction", fontsize=FS)
    # (b) lnB21
    ax = axes[0, 1]; background(ax); add_trenches(ax)
    sc = ax.scatter(df["lon"], df["lat"], c=np.clip(df["lnB21"], -15, 15), cmap="RdBu_r", vmin=-15, vmax=15, s=size, edgecolors="k",
                    linewidths=0.3, zorder=8, transform=PC)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.5, pad=0.03); cb.set_label("ln B(M2/M1)", fontsize=FS)
    ax.set_title("(b) evidence for compositional vs thermal layering", fontsize=FS)
    # (c) dT
    ax = axes[1, 0]; background(ax); add_trenches(ax)
    sc = ax.scatter(df["lon"], df["lat"], c=df["dT_M1"], cmap="coolwarm", vmin=-300, vmax=300, s=size, edgecolors="k", linewidths=0.3,
                    zorder=8, transform=PC)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.5, pad=0.03); cb.set_label("ΔT = T2 − T1 across 523 km (K), M1 posterior mean", fontsize=FS)
    ax.set_title("(c) thermal contrast across 523 km", fontsize=FS)
    # (d) df where M2
    ax = axes[1, 1]; background(ax); bg = t22_basalt(ax, 0.35); add_trenches(ax)
    m = df["cls"] == "M2"
    sc = ax.scatter(df["lon"][m], df["lat"][m], c=df["df_M2"][m], cmap="YlOrBr", vmin=0, vmax=0.6, s=size[m] * 1.2, edgecolors="k",
                    linewidths=0.4, zorder=9, transform=PC)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.5, pad=0.03); cb.set_label("Δf = f2 − f1 (M2 posterior mean) where M2 is required", fontsize=FS)
    ax.set_title("(d) basalt enrichment of the lower MTZ; background: T22 basalt fraction", fontsize=FS)
    fig.savefig(a.out + ".pdf", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(a.out + ".png", dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", a.out + ".png/.pdf", "|", len(df), "caps:", df["cls"].value_counts().to_dict())


if __name__ == "__main__":
    main()
