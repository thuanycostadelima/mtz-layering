#!/usr/bin/env python3
"""Observed vs best M0/M1/M2 synthetic (per-cap geometry) for every cap of a run, 12 caps per page, multi-page PDF.
Usage: python plot_all_fits.py RUN_DIR OBS_DIR INPUTS_DIR out.pdf --lib lib_shift_*.npz --mech mech_filters.npz [--sort lon|lat|lnB21]"""
import argparse, glob, json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from invert_bins2 import load_lib, bin_weights, apply_d660, TW, DT
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("obs_dir"); ap.add_argument("inputs_dir"); ap.add_argument("out")
ap.add_argument("--lib", nargs="+", required=True); ap.add_argument("--mech", default=None); ap.add_argument("--table", default="prem_sds_S.npz"); ap.add_argument("--sort", default="lon"); ap.add_argument("--subset", type=int, default=0, help="one page with N caps spanning the full range of P(M2) (sorted by P(M2)); N multiple of 4")
a = ap.parse_args()
names, S, meta, tg, dist = load_lib(a.lib); M = dict(zip(names, range(len(names))))
mech = dict(zip(*[np.load(a.mech)[k] for k in ("names", "h")])) if a.mech else None
tobs = np.arange(TW[0], TW[1] + 1e-9, DT); i0 = int(round(TW[0] - tg[0])); nt = tobs.size
tab = np.load(a.table); j = np.argmin(np.abs(tab["dist"] - 124)); dtt = tab["dt"][:, j]; ok = np.isfinite(dtt); z = np.interp(tobs, dtt[ok][::-1], tab["depth"][ok][::-1])
J = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(a.run, "*.json"))}
key = {"lon": lambda k: (J[k]["lon"], J[k]["lat"]), "lat": lambda k: (-J[k]["lat"], J[k]["lon"]), "lnB21": lambda k: -J[k]["lnB21"]}[a.sort]
order = sorted(J, key=key)
def PM2(k):
    lz = np.array([J[k]["classes"][c]["logZ"] for c in ("M0", "M1", "M2")]); P = np.exp(lz - lz.max()); return P[2] / P.sum()
if a.subset:   # N caps evenly spaced in P(M2) rank, lowest to highest: a one-page sample of the 525 fits
    byP = sorted(J, key=PM2); order = [byP[int(round(i))] for i in np.linspace(0, len(byP) - 1, a.subset)]
per = a.subset if a.subset else 10; ncol = 5 if per % 5 == 0 else 6 if per % 6 == 0 else 4; nrow = -(-per // ncol)
def label(b):
    """'MM0.2-0.4_1600-1700_zc523_w40' -> 'f 0.2/0.4, T 1600/1700 K, 40 km';  'MM0.2_1600' -> 'f 0.2, T 1600 K'"""
    m = b.replace("MM", ""); parts = m.split("_"); f, T = parts[0], parts[1]; w = " (40 km)" if m.endswith("w40") else ""
    return f"f {f.replace('-', '/')}, T {T.replace('-', '/')} K{w}"
with PdfPages(a.out) as pdf:
    for p in range(0, len(order), per):
        fig, axs = plt.subplots(nrow, ncol, figsize=(11, 3.4 * nrow + 0.6), sharey=True); fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.08, hspace=0.55, wspace=0.12)
        for ax, name in zip(axs.flat, order[p:p + per]):
            R = J[name]; W, nrec = bin_weights(os.path.join(a.inputs_dir, name + ".txt"), dist)
            t, pp, x, sg = np.loadtxt(os.path.join(a.obs_dir, name + ".txt"), unpack=True); d = np.interp(tobs, t, x); s = np.interp(tobs, t, sg)
            ax.fill_betweenx(z, d - 2 * s, d + 2 * s, color="0.85"); ax.plot(d, z, "k", lw=1.3)
            lz0 = np.array([R["classes"][k]["logZ"] for k in ("M0", "M1", "M2")]); pref0 = ["M0", "M1", "M2"][int(np.argmax(lz0))]
            for cls, c in sorted((("M0", "C0"), ("M1", "C2"), ("M2", "C3")), key=lambda t: t[0] == pref0):   # favoured class drawn last (on top)
                b = R["classes"][cls]; g = W @ S[M[b["best"]]]
                if mech is not None: g = np.convolve(g, mech[name], mode="same")
                g = apply_d660(g[None, :], tg, R["d660"])[0]; tau = int(b["best_tau"]); ax.plot(b["best_A"] * g[i0 + tau:i0 + tau + nt], z, c, lw=1.1 if cls == pref0 else 0.8, zorder=4 if cls == pref0 else 3)
            lz = np.array([R["classes"][k]["logZ"] for k in ("M0", "M1", "M2")]); P = np.exp(lz - lz.max()); P /= P.sum()
            pref = ["M0", "M1", "M2"][int(np.argmax(P))]
            la, lo = name.split("_"); la = la.replace("n", "−"); ttl = f"{la}°, {lo}°  n = {nrec}\nP(M0,M1,M2) = {P[0]:.2f}, {P[1]:.2f}, {P[2]:.2f}"
            if True:   # five text lines above the axes, favoured class in bold
                lines = [(f"{c}: {label(R['classes'][c]['best'])}", c == pref) for c in ("M2", "M1", "M0")] + [(t, False) for t in ttl.split("\n")[::-1]]
                for j, (t, bold) in enumerate(lines):
                    ax.text(0.5, 1.015 + 0.048 * j, t, transform=ax.transAxes, ha="center", va="bottom", fontsize=7, fontweight="bold" if bold else "normal")
            else:
                ax.set_title(ttl + f"; {pref}", fontsize=7, color={"M0": "C0", "M1": "C2", "M2": "C3"}[pref])
            for zz in (410, 520, 660): ax.axhline(zz, color="0.8", ls=":", lw=0.6)
            ax.axvline(0, color="0.7", lw=0.5); ax.tick_params(labelsize=7)
        for ax in axs.flat[len(order[p:p + per]):]: ax.axis("off")
        axs[0, 0].set_ylim(700, 350)
        for ax in axs[:, 0]: ax.set_ylabel("depth (km)", fontsize=8)
        if True:
            for ax in axs[-1, :]: ax.set_xlabel("SdS/SS", fontsize=7)
            from matplotlib.lines import Line2D
            fig.legend([Line2D([], [], color="k", lw=1.3), Line2D([], [], color="C0"), Line2D([], [], color="C2"), Line2D([], [], color="C3")],
                       ["observed stack (±2σ)", "best M0", "best M1", "best M2"], loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
        if not a.subset: fig.text(0.99, 0.01, f"page {p // per + 1} of {(len(order) + per - 1) // per}", ha="right", va="bottom", fontsize=7, color="0.4")
        pdf.savefig(fig)
        if a.subset: fig.savefig(a.out.replace(".pdf", ".png"), dpi=300)
        plt.close(fig)
print("saved", a.out)
