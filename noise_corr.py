#!/usr/bin/env python3
"""
Estimate the sample-to-sample correlation of the noise in the precursor stacks, for the correlated-noise likelihood.

    python noise_corr.py OBS_DIR INPUTS_DIR RUN_DIR out.npz --lib ... --mech ...

For every cap the residual r = d - A g_best (best model of the reference run RUN_DIR, whitened by the bootstrap sigma) is
autocorrelated; the normalised autocorrelation is averaged over caps -> R(lag), lag 0..LMAX s.  Also computed, as a check
that does not depend on any model: the autocorrelation of the whitened data in the window before the 660 precursor
(-360..-300 s), where the stacks contain no coherent arrival at 124 deg.
Prints N_eff = N / sum_k R(k) (sum over all lags, both signs): the number of independent samples in the fit window.
"""
import argparse, glob, json, os, numpy as np
from invert_bins2 import load_lib, bin_weights, TW, DT, TAUS

ap = argparse.ArgumentParser(); ap.add_argument("obs"); ap.add_argument("inputs"); ap.add_argument("run"); ap.add_argument("out")
ap.add_argument("--lib", nargs="+", required=True); ap.add_argument("--mech", required=True); ap.add_argument("--lmax", type=int, default=60)
a = ap.parse_args()
names, S, meta, tg, dist = load_lib(a.lib); M = dict(zip(names, range(len(names)))); z = np.load(a.mech); mech = dict(zip(z["names"], z["h"]))
tobs = np.arange(TW[0], TW[1] + 1e-9, DT); i0 = int(round(TW[0] - tg[0])); nt = tobs.size


def acf(x, lmax):
    x = x - x.mean(); c = np.correlate(x, x, "full")[len(x) - 1:len(x) + lmax]; return c / c[0]


Rres, Rpre, chi = [], [], []
for f in sorted(glob.glob(f"{a.run}/*.json")):
    J = json.load(open(f)); name = J["bin"]; best = max(J["classes"].values(), key=lambda r: r["best_ll"])
    W, _ = bin_weights(f"{a.inputs}/{name}.txt", dist); g = np.convolve(W @ S[M[best["best"]]], mech[name], mode="same")
    tau = int(best["best_tau"]); g = best["best_A"] * g[i0 + tau:i0 + tau + nt]
    t, p, x, sg = np.loadtxt(f"{a.obs}/{name}.txt", unpack=True); d = np.interp(tobs, t, x); s = np.interp(tobs, t, sg) * J.get("noise_scale", 1.0) ** 0.5
    Rres.append(acf((d - g) / s, a.lmax)); chi.append(np.mean(((d - g) / s) ** 2))
    tp = np.arange(-360.0, -300.0 + 1e-9, DT); Rpre.append(acf(np.interp(tp, t, x) / np.interp(tp, t, sg), a.lmax))
Rres = np.mean(Rres, 0); Rpre = np.mean(Rpre, 0)
# taper the tail so the Toeplitz matrix stays positive definite, then check
R = Rres * np.exp(-(np.arange(a.lmax + 1) / a.lmax) ** 4)
from scipy.linalg import toeplitz
lag = np.arange(nt); Rm = toeplitz(np.where(lag <= a.lmax, R[np.minimum(lag, a.lmax)], 0.0)); ev = np.linalg.eigvalsh(Rm).min()
if ev <= 0:
    R = R * np.exp(-(np.arange(a.lmax + 1) / (0.5 * a.lmax)) ** 2); Rm = toeplitz(np.where(lag <= a.lmax, R[np.minimum(lag, a.lmax)], 0.0)); ev = np.linalg.eigvalsh(Rm).min(); print("tail re-tapered;", end=" ")
neff = nt / Rm.sum(1).mean()
print(f"{len(chi)} caps; residual ACF at lags 1,2,5,10,20 s: {np.round(Rres[[1, 2, 5, 10, 20]], 2)}; pre-window ACF: {np.round(Rpre[[1, 2, 5, 10, 20]], 2)}")
print(f"first zero crossing: residual {np.argmax(Rres < 0)} s, pre-window {np.argmax(Rpre < 0)} s; min eigenvalue {ev:.3f}; N_eff = {neff:.1f} of {nt}")
np.savez(a.out, R=R, R_residual=Rres, R_prewindow=Rpre, lag=np.arange(a.lmax + 1), n_eff=neff, n=nt)
