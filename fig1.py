#!/usr/bin/env python3
"""
Figure 1 — the data (double column, 183 mm).

  (a) SS coverage: cap centres coloured by number of records        (b) PP coverage, same colour scale
  (c) global average SS and PP precursor stacks (all caps)           (d–f) three example caps, SS and PP on the same depth axis

Run in the pygmt env (cartopy):  python fig1.py   (from mtzsyn/ or mtzsyn/code/)  ->  figs/fig1.png / .pdf
Options: --no-coast (offline test). Edit CFG for file locations; "caps" = None picks the examples from the inversion run
(strongest M2 cap, a clear M0 cap, the best-sampled cap), or give three cap names e.g. ["57.3_176.3", "n11.6_299.1", "28.2_158.0"].
"""
import os, sys, glob, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec, matplotlib.ticker as mticker, matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "prem_sds_S.npz")) else os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, ROOT)
# (no trenches on this figure)

CFG = dict(obs_ss="crosssections_2024_SNR/SS/10deg_caps", obs_pp="crosssections_2024_SNR/PP/10deg_caps",
           inputs_ss="crosssections_2024_SNR/SSinputs/10deg_caps", inputs_pp="crosssections_2024_SNR/PPinputs/10deg_caps",
           run=["inv_v7_SS10"],                      # used only to choose the example caps
           table_S="prem_sds_S.npz", table_P="prem_sds_P.npz", pp_ref=105.0, caps=None, out="figs/fig1")
C_SS = "#17405f"; C_PP = "#c8641e"; C_MARK = "#ff2bd6"   # colour of the example-cap circles on the maps (drawn over a white ring so it shows on any dot colour)
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "font.family": "sans-serif"})


def depth_fn(table, dist):
    tab = np.load(table); j = np.argmin(np.abs(tab["dist"] - dist)); dt = tab["dt"][:, j]; ok = np.isfinite(dt)
    return (lambda t: np.interp(t, dt[ok][::-1], tab["depth"][ok][::-1])), (lambda z: np.interp(z, tab["depth"][ok], dt[ok]))


def parse_cap(name):
    la, lo = name.replace("n", "-").split("_"); la, lo = float(la), float(lo); return la, ((lo + 180) % 360) - 180


def coverage(obs, inputs):
    """lat, lon, n_records for every cap in obs (n from the inputs file; 1 if no inputs)."""
    rows = []
    for f in sorted(glob.glob(f"{obs}/*.txt")):
        nm = os.path.basename(f)[:-4]
        if nm.startswith("."): continue
        la, lo = parse_cap(nm); ip = f"{inputs}/{nm}.txt"; n = sum(1 for _ in open(ip)) if os.path.exists(ip) else np.nan
        rows.append((la, lo, n))
    return np.array(rows)


def load_stack(path, tg):
    d = np.loadtxt(path); return np.interp(tg, d[:, 0], d[:, 2]), np.interp(tg, d[:, 0], d[:, 3])


def global_mean(obs, tg):
    """mean over caps, and the 16th/84th percentiles across caps (the spread between caps, not the error of the mean)"""
    X = [load_stack(f, tg)[0] for f in sorted(glob.glob(f"{obs}/*.txt")) if not os.path.basename(f).startswith(".")]
    X = np.array(X); return X.mean(0), np.percentile(X, [16, 84], axis=0), len(X)


def all_records(inputs):
    """epicentral distance (col 8) and source depth (col 4) of every UNIQUE record in a cap directory.
    A seismogram belongs to every cap that contains its bounce point, so the per-cap files overlap; records are de-duplicated by their ID (col 1)."""
    seen = set(); D, Z = [], []
    for f in sorted(glob.glob(f"{inputs}/*.txt")):
        if os.path.basename(f).startswith("."): continue
        for line in open(f):
            p = line.split()
            if len(p) < 8 or p[0] in seen: continue
            seen.add(p[0]); Z.append(float(p[3])); D.append(float(p[7]))
    return np.array(D), np.array(Z)


def n_unique(inputs):
    seen = set()
    for f in glob.glob(f"{inputs}/*.txt"):
        if os.path.basename(f).startswith("."): continue
        seen.update(line.split(" ", 1)[0] for line in open(f))
    return len(seen)


def pick_caps(runs):
    for run in runs:
        fs = sorted(glob.glob(f"{run}/*.json"))
        if not fs: continue
        J = [json.load(open(f)) for f in fs]; good = [j for j in J if j["chi2_red_best"] < 2]
        m2 = max((j for j in good if j["lnB21"] > 3), key=lambda j: j["lnB21"]); m0 = min((j for j in good if j["lnB10"] < -3), key=lambda j: j["lnB10"])
        big = max(good, key=lambda j: j["n_records"]); print("example caps from", run, ":", big["bin"], m2["bin"], m0["bin"])
        return [big["bin"], m2["bin"], m0["bin"]], ["best-sampled cap", "cap requiring compositional layering", "cap fitted by a uniform MTZ"]
    return ["57.3_176.3", "n11.6_299.1", "28.2_158.0"], ["", "", ""]


def map_panel(ax, cov, label, title, norm, cmap, marks):
    PC = ccrs.PlateCarree(); ax.set_global(); ax.set_facecolor("white"); ax.gridlines(lw=0.2, ls="--", color="0.8")
    if "--no-coast" not in sys.argv: ax.coastlines(lw=0.3, color="0.45")
    c = np.where(np.isfinite(cov[:, 2]), cov[:, 2], norm.vmin)
    sc = ax.scatter(cov[:, 1], cov[:, 0], c=c, s=13, cmap=cmap, norm=norm, edgecolors="k", linewidths=0.15, transform=PC, zorder=8)
    for (la, lo), letter in marks:
        ax.scatter([lo], [la], marker="o", s=120, facecolors="none", edgecolors="white", linewidths=3.0, transform=PC, zorder=9)
        ax.scatter([lo], [la], marker="o", s=120, facecolors="none", edgecolors=C_MARK, linewidths=1.4, transform=PC, zorder=9)
        ax.text(lo + 4, la + 4, letter, transform=PC, fontsize=8, fontweight="bold", color=C_MARK, zorder=10, ha="left", va="bottom", path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_title(f"{label}   {title}", loc="left", fontweight="bold")
    return sc


def main():
    depS, timeS = depth_fn(CFG["table_S"], 124.0); depP, _ = depth_fn(CFG["table_P"], CFG["pp_ref"])
    tS = np.arange(-330.0, -60.0, 0.5); tP = np.arange(-200.0, -30.0, 0.5)
    covS = coverage(CFG["obs_ss"], CFG["inputs_ss"]); covP = coverage(CFG["obs_pp"], CFG["inputs_pp"]) if os.path.isdir(CFG["obs_pp"]) else None
    caps, descr = (CFG["caps"], ["", "", ""]) if CFG["caps"] else pick_caps(CFG["run"])
    letters = ["f", "g", "h"]; marks = [(parse_cap(c), L) for c, L in zip(caps, letters)]

    fig = plt.figure(figsize=(7.2, 7.6))
    gst = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08, left=0.03, right=0.97, top=0.96, bottom=0.76)            # maps: row height matched to the Robinson aspect
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[0.42, 1.2], hspace=0.5, wspace=0.4, left=0.075, right=0.93, top=0.665, bottom=0.1)
    allc = np.concatenate([covS[:, 2], covP[:, 2]]) if covP is not None else covS[:, 2]; allc = allc[np.isfinite(allc)]
    norm = LogNorm(vmin=max(10, allc.min()), vmax=allc.max())

    # (a) SS map, (b) PP map
    ax = fig.add_subplot(gst[0], projection=ccrs.Robinson(central_longitude=180)); sc = map_panel(ax, covS, "a", "SS precursors", norm, "viridis", marks)
    ax = fig.add_subplot(gst[1], projection=ccrs.Robinson(central_longitude=180))
    if covP is not None: map_panel(ax, covP, "b", "PP precursors", norm, "viridis", marks)
    else: ax.axis("off"); ax.text(0.5, 0.5, f"PP cross-sections not found in {CFG['obs_pp']}", ha="center", va="center", transform=ax.transAxes)
    cax = fig.add_axes([0.37, 0.735, 0.26, 0.012]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    tk = [t for t in (10, 30, 100, 300, 1000, 3000, 10000, 30000) if norm.vmin <= t <= norm.vmax]; cb.set_ticks(tk); cb.set_ticklabels([f"{t:,}" for t in tk]); cb.set_label("seismograms per 10°-radius cap", labelpad=2); cb.ax.tick_params(labelsize=6, length=2)

    # (c, d) record geometry: epicentral distance and source depth, SS and PP
    dS, zS = all_records(CFG["inputs_ss"]); dP, zP = all_records(CFG["inputs_pp"]) if os.path.isdir(CFG["inputs_pp"]) else (np.array([]), np.array([]))
    ax = fig.add_subplot(gs[0, :2]); b = np.arange(90, 181, 2)
    ax.hist(dS, bins=b, color=C_SS, alpha=0.85, lw=0, zorder=3, label=f"SS: {len(covS)} caps, {len(dS):,} seismograms")   # zorder=3: SS drawn on top of PP
    
    if len(dP): ax.hist(dP, bins=b, color=C_PP, alpha=0.75, lw=0, zorder=2, label=f"PP: {len(covP)} caps, {len(dP):,} seismograms")
    ax.axvline(124, color=C_SS, ls=":", lw=0.9); ax.axvline(CFG["pp_ref"], color=C_PP, ls=":", lw=0.9)
    ax.text(124.6, 0.04, "124°", color=C_SS, fontsize=6, transform=ax.get_xaxis_transform(), va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5)); ax.text(CFG["pp_ref"] + 0.6, 0.04, f"{CFG['pp_ref']:.0f}°", color=C_PP, fontsize=6, transform=ax.get_xaxis_transform(), va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
    ax.set_xlim(90, 180); ax.set_xlabel("epicentral distance (°)"); ax.set_ylabel("seismograms (×10³)"); ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x / 1e3:g}")); ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="upper right"); ax.grid(alpha=0.15, lw=0.4)
    ax.set_title("c   epicentral distances", loc="left", fontweight="bold")
    ax = fig.add_subplot(gs[0, 2:]); b = np.arange(0, 101, 5)
    ax.hist(zS, bins=b, color=C_SS, alpha=0.85, lw=0, zorder=3, label="SS")
    
    if len(zP): ax.hist(zP, bins=b, color=C_PP, alpha=0.75, lw=0, zorder=2, label="PP")
    zmax = max(zS.max(), zP.max() if len(zP) else 0); ax.axvline(75, color="0.2", ls="--", lw=0.8); ax.text(73.5, 0.95, f"75 km selection limit\n(deepest record {zmax:.0f} km)", transform=ax.get_xaxis_transform(), fontsize=6, va="top", ha="right", color="0.2")
    ax.set_xlim(0, 100); ax.set_xlabel("source depth (km)"); ax.set_ylabel("seismograms (×10³)"); ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x / 1e3:g}")); ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="center right"); ax.grid(alpha=0.15, lw=0.4)
    ax.set_title("d   source depths", loc="left", fontweight="bold")

    # (c) global average stacks
    ax = fig.add_subplot(gs[1, 0]); mS, eS, nS = global_mean(CFG["obs_ss"], tS)
    ax.fill_betweenx(depS(tS), eS[0], eS[1], color=C_SS, alpha=0.18, lw=0); ax.plot(mS, depS(tS), color=C_SS, lw=1.4, label=f"SS, mean of {nS} caps")
    if covP is not None:
        mP, eP, nP = global_mean(CFG["obs_pp"], tP); ax.fill_betweenx(depP(tP), eP[0], eP[1], color=C_PP, alpha=0.18, lw=0); ax.plot(mP, depP(tP), color=C_PP, lw=1.4, label=f"PP, mean of {nP} caps")
    ax.axvline(0, color="0.6", lw=0.5); ax.set_ylim(720, 350); ax.set_ylabel("depth (km), PREM at the reference distance")
    for zz in (410, 520, 660): ax.axhline(zz, color="0.55", ls=(0, (2, 2)), lw=0.6); ax.text(0.98, zz - 4, str(zz), transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6, color="0.35")
    ax.set_xlabel("amplitude / main phase"); ax.grid(alpha=0.15, lw=0.4); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncol=1, handlelength=1.4, borderpad=0.2)
    ax.set_title("e   global average", loc="left", fontweight="bold")

    # (d–f) example caps
    for k, (cap, L, dsc) in enumerate(zip(caps, letters, descr)):
        ax = fig.add_subplot(gs[1, k + 1]); la, lo = parse_cap(cap); nS_ = sum(1 for _ in open(f"{CFG['inputs_ss']}/{cap}.txt")) if os.path.exists(f"{CFG['inputs_ss']}/{cap}.txt") else None
        x, s = load_stack(f"{CFG['obs_ss']}/{cap}.txt", tS); ax.fill_betweenx(depS(tS), x - 2 * s, x + 2 * s, color=C_SS, alpha=0.2, lw=0); ax.plot(x, depS(tS), color=C_SS, lw=1.3, label="SS" + (f" ({nS_} seismograms)" if nS_ else ""))
        pf = f"{CFG['obs_pp']}/{cap}.txt"
        if os.path.exists(pf):
            nP_ = sum(1 for _ in open(f"{CFG['inputs_pp']}/{cap}.txt")) if os.path.exists(f"{CFG['inputs_pp']}/{cap}.txt") else None
            x, s = load_stack(pf, tP); ax.fill_betweenx(depP(tP), x - 2 * s, x + 2 * s, color=C_PP, alpha=0.2, lw=0); ax.plot(x, depP(tP), color=C_PP, lw=1.3, label="PP" + (f" ({nP_} seismograms)" if nP_ else ""))
        ax.axvline(0, color="0.6", lw=0.5); ax.set_ylim(720, 350); ax.tick_params(labelleft=False)
        for zz in (410, 520, 660): ax.axhline(zz, color="0.55", ls=(0, (2, 2)), lw=0.6)
        ax.set_xlabel("amplitude / main phase"); ax.grid(alpha=0.15, lw=0.4); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncol=1, handlelength=1.4, borderpad=0.2)
        ax.set_title(f"{L}   {abs(la):.0f}°{'N' if la >= 0 else 'S'}, {abs(lo):.0f}°{'E' if lo >= 0 else 'W'}", loc="left", fontweight="bold")
        if k == 2:
            ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim()); yt = [400, 500, 600, 700]; ax2.set_yticks(yt); ax2.set_yticklabels([f"{timeS(z):.0f}" for z in yt]); ax2.tick_params(labelsize=6, length=2); ax2.set_ylabel("time before the main phase (s), SS at 124°", fontsize=6)
    os.makedirs(os.path.dirname(CFG["out"]), exist_ok=True)
    fig.savefig(CFG["out"] + ".png", dpi=300); fig.savefig(CFG["out"] + ".pdf"); print("saved", CFG["out"])


def suppl_coverage(radii=(5, 7.5, 10, 12.5, 15, 17.5, 20)):
    """Fig. S1: SS (and PP) cap coverage at every radius.  python fig1.py --suppl"""
    rows = [("SS", "crosssections_2024_SNR/SS/{r}deg_caps", "crosssections_2024_SNR/SSinputs/{r}deg_caps"), ("PP", "crosssections_2024_SNR/PP/{r}deg_caps", "crosssections_2024_SNR/PPinputs/{r}deg_caps")]
    rows = [r for r in rows if any(os.path.isdir(r[1].format(r=f"{x:g}")) for x in radii)]
    nc = 3; nr = int(np.ceil(len(radii) / nc))
    fig, axs = plt.subplots(nr * len(rows), nc, figsize=(7.2, 1.3 * nr * len(rows) + 0.3), subplot_kw=dict(projection=ccrs.Robinson(central_longitude=180)), squeeze=False)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.02, wspace=0.02, hspace=0.22); sc = None
    for ax in axs.ravel(): ax.axis("off")
    for i, (ph, obs, inp) in enumerate(rows):
        for j, r in enumerate(radii):
            ax = axs[i * nr + j // nc, j % nc]; o, p = obs.format(r=f"{r:g}"), inp.format(r=f"{r:g}")
            if not os.path.isdir(o): continue
            ax.axis("on"); cov = coverage(o, p); ax.set_global(); ax.gridlines(lw=0.15, ls="--", color="0.85")
            if "--no-coast" not in sys.argv: ax.coastlines(lw=0.25, color="0.5")
            sc = ax.scatter(cov[:, 1], cov[:, 0], c=np.where(np.isfinite(cov[:, 2]), cov[:, 2], 10), s=5, cmap="viridis", norm=LogNorm(10, 30000), edgecolors="none", transform=ccrs.PlateCarree())
            ax.set_title(f"{ph}, {r:g}° radius: {len(cov)} caps", fontsize=7.5, pad=3)
    p = axs[-1, 0].get_position(); cax = fig.add_axes([0.45, p.y0 + 0.45 * p.height, 0.3, 0.012]) if len(radii) % nc else fig.add_axes([0.37, 0.03, 0.26, 0.01]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_ticks([10, 100, 1000, 10000]); cb.set_ticklabels(["10", "100", "1,000", "10,000"]); cb.set_label("seismograms per cap", fontsize=8, labelpad=2); cb.ax.tick_params(labelsize=7, length=2)
    os.makedirs("figs", exist_ok=True); fig.savefig("figs/figS_coverage.png", dpi=300); fig.savefig("figs/figS_coverage.pdf"); print("saved figs/figS_coverage")


if __name__ == "__main__":
    suppl_coverage() if "--suppl" in sys.argv else main()
