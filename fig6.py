#!/usr/bin/env python3
"""
Figure 6 — what the layering is, and where it is: posteriors of the M2 caps and comparison with tomography and T22.

  (a) mean P(M2) by quintile of TX2011 shear-velocity anomaly at 500–600 km      (b) transition-zone temperature (M0 posterior mean) vs Tauzin et al. 2022
  (c) posterior distributions of T1 (above 523 km) and T2 (below) over the M2 caps   (d) posterior distributions of f1 and f2

Run in the mtz520 env (needs netCDF4):  python fig6.py   (from mtzsyn/ or mtzsyn/code/)  ->  figs/fig6.png / .pdf
Inputs: inv_v7_SS10/ (per-cap JSON), v7_SS10_fields.csv, TX2011_percent.nc, SSall.txt (T22: lat lon T f ...).
"""
import os, sys, glob, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "prem_sds_S.npz")) else os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, ROOT)

CFG = dict(run=["inv_v7_SS10"], fields=["v7_SS10_fields.csv"], tomo="TX2011_percent.nc", t22="SSall.txt", thr=3.0, radius=10.0, out="figs/fig6")
C = {"M0": "#4c78a8", "M1": "#3c9a5a", "M2": "#c8322b"}; C_T1, C_T2 = "#e08a2e", "#2e8b57"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "font.family": "sans-serif"})


def first(paths):
    for p in paths:
        if os.path.exists(p) and (os.path.isfile(p) or glob.glob(f"{p}/*.json")): return p
    raise FileNotFoundError(paths)


def gcd(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return np.degrees(np.arccos(np.clip(np.sin(la1) * np.sin(la2) + np.cos(la1) * np.cos(la2) * np.cos(lo1 - lo2), -1, 1)))


def capmean(lat, lon, glat, glon, gval, r):
    return np.array([np.average(gval[k], weights=np.cos(np.radians(glat[k]))) if k.any() else np.nan for k in (gcd(la, lo, glat, glon) <= r for la, lo in zip(lat, lon))])


def main():
    F = np.genfromtxt(first(CFG["fields"]), delimiter=",", names=True); lat, lon, P = F["lat"], F["lon"], F["P_M2"]
    J = {j["bin"]: j for j in (json.load(open(f)) for f in sorted(glob.glob(f"{first(CFG['run'])}/*.json")))}
    m2 = [j for j in J.values() if j["lnB21"] > CFG["thr"]]; n2 = len(m2)
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.0)); fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.09, wspace=0.3, hspace=0.4)

    # (a) tomography
    import netCDF4 as nc
    d = nc.Dataset(CFG["tomo"]); dep = d.variables["depth"][:]; tl = d.variables["latitude"][:]; tn = d.variables["longitude"][:]; dvs = np.array(d.variables["dvs"][:])
    L = np.nanmean(dvs[(dep >= 500) & (dep <= 600)], axis=0); LAT, LON = np.meshgrid(tl, tn, indexing="ij"); v = capmean(lat, lon, LAT.ravel(), LON.ravel(), L.ravel(), CFG["radius"])
    e = np.percentile(v, [0, 20, 40, 60, 80, 100]); idx = np.clip(np.digitize(v, e) - 1, 0, 4)
    m = [P[idx == k].mean() for k in range(5)]; se = [P[idx == k].std() / np.sqrt((idx == k).sum()) for k in range(5)]; ax = axs[0, 0]
    ax.bar(range(5), m, yerr=se, color=C["M2"], alpha=0.8, capsize=2, lw=0); ax.set_xticks(range(5)); ax.set_xticklabels([f"{e[k]:+.1f}\nto {e[k+1]:+.1f}" for k in range(5)], fontsize=6.5)
    ax.set_xlabel("TX2011 δVs at 500–600 km (%), cap mean, quintiles"); ax.set_ylabel("mean P(M2)"); ax.set_ylim(0, 0.7); ax.grid(alpha=0.15, lw=0.4, axis="y")
    ax.text(0.03, 0.93, f"r = {np.corrcoef(v, P)[0, 1]:.2f}  ({len(P)} caps)", transform=ax.transAxes); ax.set_title("a   P(M2) vs shear velocity at 500–600 km", loc="left", fontweight="bold")
    # (b) temperature vs T22
    t = np.loadtxt(CFG["t22"]); T22 = capmean(lat, lon, t[:, 0], t[:, 1], t[:, 2], CFG["radius"]); T0 = F["T_M0"]; ok = np.isfinite(T22); ax = axs[0, 1]
    ax.scatter(T22[ok], T0[ok], s=7, c=P[ok], cmap="YlOrRd", vmin=0, vmax=1, edgecolors="0.3", linewidths=0.2); lim = (1450, 1950); ax.plot(lim, lim, "k:", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("T22 potential temperature (K), cap mean"); ax.set_ylabel("this study: uniform-MTZ posterior mean (K)"); ax.grid(alpha=0.15, lw=0.4)
    ax.text(0.03, 0.93, f"r = {np.corrcoef(T22[ok], T0[ok])[0, 1]:.2f}, mean difference {np.mean(T0[ok] - T22[ok]):+.0f} K", transform=ax.transAxes); ax.set_title("b   temperature vs Tauzin et al. (2022)", loc="left", fontweight="bold")
    sc = ax.collections[0]; cax = ax.inset_axes([0.62, 0.08, 0.33, 0.035]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label("P(M2)", fontsize=7, labelpad=1); cb.ax.tick_params(labelsize=6.5, length=2)
    # (c) T posteriors, (d) f posteriors over the M2 caps
    def stack(key):
        D = {}
        for j in m2:
            for k, vv in j["classes"]["M2"][key].items(): D[float(k)] = D.get(float(k), 0) + vv
        ks = sorted(D); return np.array(ks), np.array([D[k] / n2 for k in ks])
    kT1, pT1 = stack("post_T1"); kT2, pT2 = stack("post_T2"); ax = axs[1, 0]; w = 20
    ax.bar(kT1 - w / 2, pT1, w, color=C_T1, label="T$_1$ (above 523 km)"); ax.bar(kT2 + w / 2, pT2, w, color=C_T2, label="T$_2$ (below 523 km)")
    ax.set_xlabel("potential temperature (K)"); ax.set_ylabel("mean posterior probability"); ax.legend(frameon=False); ax.grid(alpha=0.15, lw=0.4, axis="y"); ax.set_xlim(1280, 2020)
    mT1 = (kT1 * pT1).sum() / pT1.sum(); mT2 = (kT2 * pT2).sum() / pT2.sum(); ax.set_title(f"c   temperature posteriors, {n2} M2 caps", loc="left", fontweight="bold"); ax.text(0.03, 0.72, f"⟨T$_2$ − T$_1$⟩ = {mT2 - mT1:+.0f} K", transform=ax.transAxes)
    # f1 posterior is not stored directly: reconstruct from f2 and mean_df? post_f2 only; use f1 marginal via classes if present
    kf2, pf2 = stack("post_f2"); ax = axs[1, 1]; w = 0.08
    if "post_f1" in m2[0]["classes"]["M2"]:
        kf1, pf1 = stack("post_f1"); ax.bar(kf1 - w / 2, pf1, w, color=C_T1, label="f$_1$ (above 523 km)")
    else:
        f1m = np.mean([j["classes"]["M2"]["mean_f1"] for j in m2]); ax.axvline(f1m, color=C_T1, lw=1.5, ls="--", label=f"⟨f$_1$⟩ = {f1m:.2f} (above 523 km)")
    ax.bar(kf2 + w / 2, pf2, w, color=C["M2"], alpha=0.85, label="f$_2$ (below 523 km)"); dfs = [j["classes"]["M2"]["mean_df"] for j in m2]
    ax.set_xlabel("basalt fraction"); ax.set_ylabel("mean posterior probability"); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.06, 0.78)); ax.grid(alpha=0.15, lw=0.4, axis="y"); ax.set_xticks([0, 0.2, 0.4, 0.6]); ax.set_ylim(0, 0.55)
    ax.set_title(f"d   composition posteriors, {n2} M2 caps", loc="left", fontweight="bold"); ax.text(0.97, 0.93, f"Δf = f$_2$ − f$_1$: median {np.median(dfs):.2f}\nf$_2$ > f$_1$ in {np.mean(np.array(dfs) > 0) * 100:.0f}% of caps", transform=ax.transAxes, ha="right", va="top")
    os.makedirs(os.path.dirname(CFG["out"]), exist_ok=True); fig.savefig(CFG["out"] + ".png", dpi=300); fig.savefig(CFG["out"] + ".pdf"); print("saved", CFG["out"])


if __name__ == "__main__":
    main()
