#!/usr/bin/env python3
"""
Figure 3 — how a cap is classified, and what the classes look like in the data.

  rows 1–2: one cap requiring compositional layering (M2) and one fitted by a uniform MTZ (M0):
     (left) data vs the best synthetic of each class   (middle) reduced chi^2 of every model in the library, by class
     (right) ln evidence of each class relative to the best, with the Bayes factors
  row 3: gallery of six caps (two per assigned class) with the best synthetic of each class

Run in the mtz520 env:  python fig3.py   (from mtzsyn/ or mtzsyn/code/)  ->  figs/fig3.png / .pdf
"""
import os, sys, glob, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "prem_sds_S.npz")) else os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, ROOT)
from invert_bins2 import load_lib, bin_weights, TW, DT, TAUS

CFG = dict(obs="crosssections_2024_SNR/SS/10deg_caps", inputs="crosssections_2024_SNR/SSinputs/10deg_caps", run=["inv_v7_SS10"],
           lib=["lib_shift_0.npz", "lib_shift_1.npz", "lib_shift_2.npz", "lib_shift_3.npz", "lib_shift50_0.npz"],
           mech=["mech_filters_SS10_v7.npz", "mech_filters_SS10.npz", "mech_filters.npz"], table_S="prem_sds_S.npz",
           caps=None,            # [M2 cap, M0 cap] for rows 1–2; None = strongest M2 and clearest M0 in the run
           gallery=None,         # six cap names for row 3; None = two per class, chosen automatically
           out="figs/fig3")
C = {"M0": "#4c78a8", "M1": "#3c9a5a", "M2": "#c8322b"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
                     "axes.linewidth": 0.6, "font.family": "sans-serif"})


def first(paths):
    for p in paths:
        if os.path.exists(p) and (os.path.isfile(p) or glob.glob(f"{p}/*.json")): return p
    raise FileNotFoundError(paths)


def depth_fn(table, dist):
    tab = np.load(table); j = np.argmin(np.abs(tab["dist"] - dist)); dt = tab["dt"][:, j]; ok = np.isfinite(dt)
    return lambda t: np.interp(t, dt[ok][::-1], tab["depth"][ok][::-1])


def load_run(run):
    R = {}
    for f in sorted(glob.glob(f"{run}/*.json")):
        J = json.load(open(f)); c = J["classes"]; lz = np.array([c[k]["logZ"] for k in ("M0", "M1", "M2")]); p = np.exp(lz - lz.max()); p /= p.sum()
        R[J["bin"]] = dict(P=p, cls=2 if J["lnB21"] > 3 else (1 if J["lnB10"] > 3 else 0), J=J)
    return R


class Fitter:
    """Recomputes the per-model misfit of a cap exactly as invert_bins2 does, including the correlated-noise covariance if the run used one."""
    def __init__(self, J0):
        self.names, self.S, self.meta, self.tg, self.dist = load_lib(CFG["lib"]); self.cls = np.array([m["cls"] for m in self.meta])
        z = np.load(first(CFG["mech"])); self.mech = dict(zip(z["names"], z["h"]))
        self.tobs = np.arange(TW[0], TW[1] + 1e-9, DT); self.i0 = int(round(TW[0] - self.tg[0])); self.nt = self.tobs.size; self.Rinv = None
        if J0.get("corr_noise") and os.path.exists(J0["corr_noise"]):
            from scipy.linalg import toeplitz
            R = np.load(J0["corr_noise"])["R"]; lag = np.arange(self.nt); r = np.where(lag < len(R), R[np.minimum(lag, len(R) - 1)], 0.0); self.Rinv = np.linalg.inv(toeplitz(r))

    def fit(self, name, J):
        """Return data, sigma (after the run's noise inflation), and for every model its best (ll, tau, A) under the run's likelihood."""
        W, _ = bin_weights(f"{CFG['inputs']}/{name}.txt", self.dist); G = np.einsum("d,mdt->mt", W, self.S); G = np.array([np.convolve(g, self.mech[name], mode="same") for g in G])
        t, p, x, sg = np.loadtxt(f"{CFG['obs']}/{name}.txt", unpack=True); d = np.interp(self.tobs, t, x); s = np.interp(self.tobs, t, sg) * J.get("noise_scale", 1.0) ** 0.5; dn = d / s
        bl = np.full(len(self.names), -np.inf); bt = np.zeros(len(self.names), int); bA = np.ones(len(self.names))
        for tau in TAUS:
            g = G[:, self.i0 + tau:self.i0 + tau + self.nt] / s[None, :]
            if self.Rinv is not None: gR = g @ self.Rinv; Sgd = gR @ dn; Sgg = (gR * g).sum(1); Sdd = dn @ self.Rinv @ dn
            else: Sgd = g @ dn; Sgg = (g * g).sum(1); Sdd = dn @ dn
            A = np.clip(Sgd / Sgg, 0.3, 3); ll = -0.5 * (Sdd - 2 * A * Sgd + A * A * Sgg); u = ll > bl; bl[u] = ll[u]; bt[u] = tau; bA[u] = A[u]
        return d, s, G, bl, bt, bA

    def best_syn(self, G, bl, bt, bA, cls):
        k = np.where(self.cls == cls)[0]; ib = k[np.argmax(bl[k])]; tau = bt[ib]; return bA[ib] * G[ib, self.i0 + tau:self.i0 + tau + self.nt], -2 * bl[ib] / self.nt


def pick(R):
    good = {k: v for k, v in R.items() if v["J"]["chi2_red_best"] < 2}
    m2 = max((k for k in good if good[k]["cls"] == 2), key=lambda k: good[k]["J"]["lnB21"]); m0 = min((k for k in good if good[k]["cls"] == 0), key=lambda k: max(good[k]["J"]["lnB10"], good[k]["J"]["lnB21"]))   # clearest M0: both layered classes disfavoured
    gal = []
    for c in (0, 1, 2):
        ks = sorted((k for k in good if good[k]["cls"] == c and k not in (m2, m0) and good[k]["P"][c] > 0.6), key=lambda k: -good[k]["J"]["n_records"]); gal += ks[:2]   # well-sampled caps whose assigned class also holds most of the probability
    return [m2, m0], gal


def label(name):
    la, lo = name.replace("n", "-").split("_"); la, lo = float(la), ((float(lo) + 180) % 360) - 180
    return f"{abs(la):.0f}°{'N' if la >= 0 else 'S'} {abs(lo):.0f}°{'E' if lo >= 0 else 'W'}"


def main():
    dep = depth_fn(CFG["table_S"], 124.0); R = load_run(first(CFG["run"])); F = Fitter(next(iter(R.values()))["J"]); z = dep(F.tobs)
    caps, gal = pick(R); caps = CFG["caps"] or caps; gal = CFG["gallery"] or gal
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.38, wspace=0.7, left=0.085, right=0.985, top=0.94, bottom=0.37)      # rows 1–2
    gsg = gridspec.GridSpec(1, 6, figure=fig, wspace=0.55, left=0.08, right=0.985, top=0.235, bottom=0.03)                 # gallery row
    letters = "abcdef"
    for row, name in enumerate(caps):
        J = R[name]["J"]; d, s, G, bl, bt, bA = F.fit(name, J); chi2 = -2 * bl / F.nt
        ax = fig.add_subplot(gs[row, 0:2]); ax.fill_betweenx(z, d - 2 * s, d + 2 * s, color="0.88", lw=0); ax.plot(d, z, "k", lw=1.5, label="data (±2σ)")
        for c in ("M0", "M1", "M2"):
            g, x2 = F.best_syn(G, bl, bt, bA, c); ax.plot(g, z, color=C[c], lw=1.1, label=f"best {c} model")
        ax.set_ylim(700, 350); ax.axvline(0, color="0.6", lw=0.5); ax.set_ylabel("depth (km)"); ax.set_xlabel("S$d$S / SS"); ax.grid(alpha=0.15, lw=0.4)
        for zz in (410, 520, 660): ax.axhline(zz, color="0.55", ls=(0, (2, 2)), lw=0.6)
        if row == 0: fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handlelength=1.4, columnspacing=1.6)
        ax.set_title(f"{letters[3 * row]}   {label(name)}: {'M2 required' if R[name]['cls'] == 2 else 'M0 sufficient'}", loc="left", fontweight="bold")
        ax = fig.add_subplot(gs[row, 2:4]); rng = np.random.default_rng(1)
        for xo, c in enumerate(("M0", "M1", "M2")):
            k = F.cls == c; ax.scatter(xo + rng.uniform(-0.32, 0.32, k.sum()), chi2[k], s=3, color=C[c], alpha=0.45, lw=0); ax.plot([xo - 0.38, xo + 0.38], [chi2[k].min()] * 2, color="k", lw=1.6); ax.text(xo, chi2[k].min() - 0.12, f"{chi2[k].min():.2f}", ha="center", va="top", fontsize=7)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels([f"M0\n({np.sum(F.cls == 'M0')})", f"M1\n({np.sum(F.cls == 'M1')})", f"M2\n({np.sum(F.cls == 'M2')})"]); ax.set_ylim(0, 3); ax.set_ylabel("misfit rᵀC⁻¹r / N"); ax.grid(alpha=0.15, lw=0.4, axis="y")
        ax.set_title(f"{letters[3 * row + 1]}   misfit of all {len(F.names)} models", loc="left", fontweight="bold")
        ax = fig.add_subplot(gs[row, 4:6]); lz = {c: J["classes"][c]["logZ"] for c in ("M0", "M1", "M2")}; top = max(lz.values())
        for xo, c in enumerate(("M0", "M1", "M2")): ax.bar(xo, lz[c] - top, color=C[c], width=0.6)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["M0", "M1", "M2"]); ax.set_ylabel("ln Z − ln Z$_{best}$", labelpad=1); ax.grid(alpha=0.15, lw=0.4, axis="y"); ax.axhline(0, color="0.3", lw=0.6)
        ax.set_title(f"{letters[3 * row + 2]}   evidence of each class", loc="left", fontweight="bold")
        ax.text(0.97, 0.04, f"ln B(M1/M0) = {J['lnB10']:.1f}\nln B(M2/M1) = {J['lnB21']:.1f}\nP(M0, M1, M2) = {R[name]['P'][0]:.2f}, {R[name]['P'][1]:.2f}, {R[name]['P'][2]:.2f}", transform=ax.transAxes, fontsize=7, va="bottom", ha="right", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    for k, name in enumerate(gal):
        J = R[name]["J"]; d, s, G, bl, bt, bA = F.fit(name, J); ax = fig.add_subplot(gsg[0, k])
        ax.fill_betweenx(z, d - 2 * s, d + 2 * s, color="0.88", lw=0); ax.plot(d, z, "k", lw=1.3)
        for c in ("M0", "M1", "M2"):
            g, _ = F.best_syn(G, bl, bt, bA, c); ax.plot(g, z, color=C[c], lw=0.9)
        ax.set_ylim(700, 350); ax.axvline(0, color="0.6", lw=0.5); ax.set_xticks([]); ax.grid(alpha=0.15, lw=0.4)
        for zz in (410, 520, 660): ax.axhline(zz, color="0.55", ls=(0, (2, 2)), lw=0.6)
        if k: ax.tick_params(labelleft=False)
        else: ax.set_ylabel("depth (km)")
        cls = ["M0", "M1", "M2"][R[name]["cls"]]; ax.set_title(f"{label(name)}\n{cls}, P = {R[name]['P'][R[name]['cls']]:.2f}", fontsize=7.5, color=C[cls], fontweight="bold")
    top = max(a.get_position().y1 for a in fig.axes[-6:])
    fig.text(0.08, top + 0.05, "g   six caps, two per class: data and the best model of each class", fontsize=9, fontweight="bold")
    os.makedirs(os.path.dirname(CFG["out"]), exist_ok=True); fig.savefig(CFG["out"] + ".png", dpi=300); fig.savefig(CFG["out"] + ".pdf"); print("saved", CFG["out"])


if __name__ == "__main__":
    main()
