#!/usr/bin/env python3
"""
Waveform inversion of SS-precursor stacks, version 2: per-bin synthetics built with the bin's
real source-receiver geometry.

For bin b and model m the synthetic is
    g_bm(t) = h_b * ( sum_D W_b(D) S_m(D, t) )
  S_m(D,t)  model m's single-source trace at distance D, pre-shifted onto the reference slowness
            curve (pack_models.py)                      -> exact distance effect, per model
  W_b(D)    SNR-weighted distance histogram of the bin's records (from Lauren's input files)
  h_b       per-bin FIR correcting for source mechanisms/depths (mech_filter.py; optional)
Then as before: Gaussian likelihood on -300..-110 s with free amplitude A in [0.3, 3] and shift
tau in [-12, 12] s, evidence per class = mean likelihood over the class grid, Bayes factors.

Usage:
  python invert_bins2.py OBS_DIR INPUTS_DIR OUT_DIR --lib lib_shift_0.npz lib_shift_1.npz ... \
        [--mech mech_filters.npz] [--d660 3.5] [--dT-prior 100 150] [--shape-only] [--noise-factor 1.5]
        [--only NAME] [--synthetic MODELNAME --amp 0.6 --noise 0.004 | --noise-obs --seed 0]
Writes OUT_DIR/<bin>.json and OUT_DIR/summary.csv.
"""
import argparse, glob, json, os, re
import numpy as np
from scipy.ndimage import uniform_filter1d

TW = (-300.0, -110.0); DT = 1.0
TAUS = np.arange(-12, 13, 1)
T2MIN = 1400
FMAX = 0.6


def parse(name):
    m = re.match(r"MM(\d\.\d)_(\d{4})$", name)
    if m:
        f, T = float(m.group(1)), int(m.group(2)); return dict(cls="M0", f1=f, f2=f, T1=T, T2=T, w=0)
    m = re.match(r"MM(\d\.\d)-(\d\.\d)_(\d{4})-(\d{4})_zc\d+_w(\d+)$", name)
    if m:
        f1, f2, T1, T2, w = float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        return dict(cls="M1" if f1 == f2 else "M2", f1=f1, f2=f2, T1=T1, T2=T2, w=w)
    return None


def load_lib(files):
    names, S, meta = [], [], []
    for f in files:
        z = np.load(f)
        for nm, s in zip(z["names"], z["S"]):
            p = parse(str(nm))
            if p is None or p["f1"] > FMAX or p["f2"] > FMAX or not (1300 <= p["T1"] <= 2000) or not (T2MIN <= p["T2"] <= 2000):
                continue
            names.append(str(nm)); S.append(s.astype(np.float32)); meta.append(p)
        tg, dist = z["t"], z["dist"]
    return names, np.array(S), meta, tg, dist


def bin_weights(path, dist):
    D, Wt = [], []
    for line in open(path):
        p = line.split()
        if len(p) < 15:
            continue
        D.append(float(p[7])); Wt.append(max(float(p[14]), 0.0))
    idx = np.clip(np.round(D).astype(int) - int(dist[0]), 0, len(dist) - 1)
    W = np.bincount(idx, weights=Wt, minlength=len(dist))
    return W / W.sum(), len(D)


def apply_d660(G, tg, d660, t_split=-195.0, ramp=20.0):
    """Shift the part of every trace before t_split (the 660 region) by d660 s (scalar, or one value per model)."""
    d660 = np.broadcast_to(np.asarray(d660, float), (len(G),))
    if not np.any(d660):
        return G
    wgt = 0.5 * (1 - np.tanh((tg - t_split) / (ramp / 4)))
    Gs = np.array([np.interp(tg - d, tg, g) for g, d in zip(G, d660)])
    return wgt * Gs + (1 - wgt) * G


def perturb_pulse(g, tg, shift, broaden, t_lo, t_hi, ramp=20.0):
    """Shift (s) and/or broaden (Gaussian sigma, s) the part of trace g between t_lo and t_hi, with smooth edges.
    Used to build synthetic data whose 410 or 660 differs from the library (recovery tests against discontinuity mismatch)."""
    if shift == 0.0 and broaden == 0.0:
        return g
    gp = np.interp(tg - shift, tg, g) if shift else g.copy()
    if broaden:
        k = np.arange(-int(4 * broaden), int(4 * broaden) + 1) * DT; k = np.exp(-0.5 * (k / broaden) ** 2); k /= k.sum(); gp = np.convolve(gp, k, mode="same")
    wgt = 0.5 * (1 + np.tanh((tg - t_lo) / (ramp / 4))) * 0.5 * (1 - np.tanh((tg - t_hi) / (ramp / 4)))
    return wgt * gp + (1 - wgt) * g


def narrow_pulse(g, tg, frac, t_lo, t_hi, ramp=20.0):
    """Sharpen the pulse between t_lo and t_hi by compressing the time axis about its peak by a fraction frac (0.07 = 7 % narrower)."""
    if frac == 0.0:
        return g
    w = (tg >= t_lo) & (tg <= t_hi); tp = tg[w][np.argmax(g[w])]
    gp = np.interp(tp + (tg - tp) / (1.0 - frac), tg, g)
    wgt = 0.5 * (1 + np.tanh((tg - t_lo) / (ramp / 4))) * 0.5 * (1 - np.tanh((tg - t_hi) / (ramp / 4)))
    return wgt * gp + (1 - wgt) * g


def scale_pulse(g, tg, fac, t_lo, t_hi, ramp=20.0):
    """Scale the pulse between t_lo and t_hi by fac (amplitude-only mismatch), with smooth edges."""
    if fac == 1.0:
        return g
    wgt = 0.5 * (1 + np.tanh((tg - t_lo) / (ramp / 4))) * 0.5 * (1 - np.tanh((tg - t_hi) / (ramp / 4)))
    return g * (1 + (fac - 1) * wgt)


def envnorm(x, win=60):
    env = np.sqrt(uniform_filter1d(x * x, int(win / DT), mode="nearest", axis=-1)) + 1e-9
    return x / env, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obs_dir"); ap.add_argument("inputs_dir"); ap.add_argument("out")
    ap.add_argument("--lib", nargs="+", required=True); ap.add_argument("--mech", default=None)
    ap.add_argument("--d660", type=float, default=0.0); ap.add_argument("--dT-prior", type=float, nargs=2, default=None)
    ap.add_argument("--fmax", type=float, default=0.6, help="highest basalt fraction admitted in any layer (default 0.6; the layered models exist only up to 0.6)")
    ap.add_argument("--syn-caps", default=None, metavar="FILE", help="recovery test: apply the --syn-* perturbations only to the caps listed in FILE (one name per line); other caps get the unperturbed model")
    ap.add_argument("--t2min", type=int, default=1400, help="lowest potential temperature admitted for uniform models and for the lower layer (default 1400 K)")
    ap.add_argument("--clapeyron", type=float, default=0.0, metavar="DZDT",
                    help="move the 660 of every model with the temperature of its lower layer: depth change DZDT km/K (e.g. -0.075 for -2.8 MPa/K) relative to --t660-ref, converted to time at 0.35 s/km")
    ap.add_argument("--t660-ref", type=float, default=1600.0, help="temperature at which the library 660 is left unshifted by --clapeyron")
    ap.add_argument("--s-per-km", type=float, default=0.35, help="SdS-SS time change per km of 660 depth (s/km, positive; deeper = earlier)")
    ap.add_argument("--shape-only", action="store_true"); ap.add_argument("--noise-factor", type=float, default=1.5)
    ap.add_argument("--only", default=None)
    ap.add_argument("--synthetic", default=None, help="model name to use as fake data (recovery test), built with each bin's own geometry")
    ap.add_argument("--amp", type=float, default=0.6); ap.add_argument("--noise", type=float, default=0.004); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--syn-660", type=float, nargs=2, default=(0.0, 0.0), metavar=("SHIFT", "BROADEN"), help="recovery test: shift (s) and Gaussian broadening (sigma, s) applied to the 660 pulse (-300..-210 s) of the synthetic data only")
    ap.add_argument("--syn-410", type=float, nargs=2, default=(0.0, 0.0), metavar=("SHIFT", "BROADEN"), help="recovery test: the same for the 410 pulse (-190..-110 s) of the synthetic data only")
    ap.add_argument("--syn-410-amp", type=float, default=1.0, metavar="FAC", help="recovery test: scale the 410 pulse of the synthetic data by FAC (amplitude-only mismatch)")
    ap.add_argument("--syn-660-amp", type=float, default=1.0, metavar="FAC", help="recovery test: the same for the 660 pulse")
    ap.add_argument("--syn-410-narrow", type=float, default=0.0, metavar="FRAC", help="recovery test: narrow the 410 pulse of the synthetic data by this fraction of its width (e.g. 0.07)")
    ap.add_argument("--syn-660-narrow", type=float, default=0.0, metavar="FRAC", help="recovery test: the same for the 660 pulse")
    ap.add_argument("--noise-obs", action="store_true", help="recovery test with each cap's own bootstrap sigma(t) as the noise level instead of --noise")
    ap.add_argument("--no-geometry", action="store_true", help="use the 124-degree single trace for every bin (old behaviour)")
    ap.add_argument("--window", type=float, nargs=2, default=None, metavar=("TMIN", "TMAX"), help="fit window (s rel. main phase); default -300 -110 (SS); PP at 105: -150 -60")
    ap.add_argument("--lowpass", type=float, default=None, help="apply a zero-phase low-pass (corner period in s) to data and synthetics, e.g. 15 for PP to match the SS band")
    ap.add_argument("--tau-max", type=int, default=12, help="max |time shift| in s (12 for SS; ~4 for PP, where 12 s = 75 km of 410 depth)")
    ap.add_argument("--amp-range", type=float, nargs=2, default=(0.3, 3.0), help="bounds of the free amplitude factor")
    ap.add_argument("--marginal", action="store_true", help="marginalise the amplitude factor (flat prior on --amp-range, analytic) and the time shift (flat prior on ±tau-max) instead of maximising them")
    ap.add_argument("--corr-noise", default=None, help="npz with the noise autocorrelation R(lag) (from noise_corr.py); likelihood uses the full covariance diag(s) R diag(s)")
    ap.add_argument("--exclude", nargs="+", default=None, help="drop library models whose name contains any of these strings (e.g. w40 to keep only sharp interfaces)")
    ap.add_argument("--noise-fit", action="store_true",
                    help="per-bin noise inflation: if the best model of all classes has reduced chi2 > 1, scale sigma by sqrt(chi2) so that it equals 1 (never deflate)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    global TW, TAUS, T2MIN, FMAX
    if a.window: TW = tuple(a.window)
    TAUS = np.arange(-a.tau_max, a.tau_max + 1, 1)

    T2MIN = a.t2min; FMAX = a.fmax
    names, S, meta, tg, dist = load_lib(a.lib)
    if a.exclude:
        keep = [i for i, n in enumerate(names) if not any(x in n for x in a.exclude)]; names = [names[i] for i in keep]; S = S[keep]; meta = [meta[i] for i in keep]
        print(f"excluded models containing {a.exclude}: {len(names)} remain", flush=True)
    cls = np.array([m["cls"] for m in meta]); T1 = np.array([m["T1"] for m in meta]); T2 = np.array([m["T2"] for m in meta])
    f1 = np.array([m["f1"] for m in meta]); f2 = np.array([m["f2"] for m in meta])
    print(f"{len(names)} models: M0 {np.sum(cls=='M0')}, M1 {np.sum(cls=='M1')}, M2 {np.sum(cls=='M2')}", flush=True)
    mech = None
    if a.mech:
        z = np.load(a.mech); mech = dict(zip(z["names"], z["h"])); L = int(z["L"])
    tobs = np.arange(TW[0], TW[1] + 1e-9, DT)
    i0 = int(round(TW[0] - tg[0])); nt = tobs.size
    rng = np.random.default_rng(a.seed)
    lp_dT = 0.0 if a.dT_prior is None else -0.5 * ((T2 - T1 - a.dT_prior[0]) / a.dT_prior[1]) ** 2
    isyn = names.index(a.synthetic) if a.synthetic else None
    syn_caps = set(l.strip() for l in open(a.syn_caps) if l.strip()) if a.syn_caps else None
    d660 = a.d660 - a.s_per_km * a.clapeyron * (T2 - a.t660_ref)      # per-model 660 shift: colder lower layer -> deeper 660 -> earlier arrival (negative)
    if a.clapeyron:
        print(f"Clapeyron 660: {a.clapeyron} km/K about {a.t660_ref:.0f} K -> shifts from {d660.min():+.1f} to {d660.max():+.1f} s", flush=True)

    Rinv = None
    if a.corr_noise:
        from scipy.linalg import toeplitz
        R = np.load(a.corr_noise)["R"]; lag = np.arange(nt); r = np.where(lag < len(R), R[np.minimum(lag, len(R) - 1)], 0.0)
        Rm = toeplitz(r); Rinv = np.linalg.inv(Rm); Lchol = np.linalg.cholesky(Rm); print(f"correlated noise: R from {a.corr_noise}, N_eff = {nt / Rm.sum(1).mean():.1f} of {nt} samples", flush=True)
    rows = []
    files = sorted(glob.glob(os.path.join(a.obs_dir, "*.txt")))
    if a.only:
        files = [f for f in files if os.path.basename(f).startswith(a.only)]
    for f in files:
        name = os.path.basename(f)[:-4]
        inp = os.path.join(a.inputs_dir, name + ".txt")
        if not os.path.exists(inp):
            print(name, "no input file"); continue
        W, nrec = bin_weights(inp, dist)
        if a.no_geometry:
            W = np.zeros_like(W); W[int(np.argmin(np.abs(dist - 124)))] = 1.0
        G = np.einsum("d,mdt->mt", W, S)                       # exact per-model stack for this bin
        if mech is not None and name in mech:
            h = mech[name]
            G = np.array([np.convolve(g, h, mode="same") for g in G])   # verified: equals mech_filter X@h
        G = apply_d660(G, tg, d660)
        if a.lowpass:
            from scipy.signal import butter, sosfiltfilt
            sos_lp = butter(4, 1.0 / a.lowpass, btype="low", fs=1.0 / DT, output="sos"); G = sosfiltfilt(sos_lp, G, axis=-1)
        # data
        t, p, x, sg = np.loadtxt(f, unpack=True)
        if a.lowpass:
            sos_d = butter(4, 1.0 / a.lowpass, btype="low", fs=1.0 / (t[1] - t[0]), output="sos")
            x = sosfiltfilt(sos_d, x)   # data are at their own sampling; bootstrap sigma left as is, the per-cap noise fit re-scales it
        if isyn is not None:
            eps = rng.normal(0, 1.0, nt)
            if Rinv is not None: eps = Lchol @ eps          # correlated synthetic noise with the same R as the likelihood
            s = np.interp(tobs, t, sg) if a.noise_obs else np.full(nt, a.noise)   # per-cap bootstrap sigma or a fixed level
            gsyn = G[isyn]
            if syn_caps is None or name in syn_caps:   # discontinuity mismatch, data only (optionally in a subset of caps)
                gsyn = perturb_pulse(gsyn, tg, *a.syn_660, -300.0, -210.0); gsyn = perturb_pulse(gsyn, tg, *a.syn_410, -190.0, -110.0)
                gsyn = narrow_pulse(gsyn, tg, a.syn_410_narrow, -190.0, -140.0); gsyn = narrow_pulse(gsyn, tg, a.syn_660_narrow, -265.0, -210.0)
                gsyn = scale_pulse(gsyn, tg, a.syn_410_amp, -190.0, -140.0); gsyn = scale_pulse(gsyn, tg, a.syn_660_amp, -265.0, -210.0)
            d = a.amp * gsyn[i0:i0 + nt] + s * eps
        else:
            d = np.interp(tobs, t, x); s = np.interp(tobs, t, sg) * a.noise_factor
        if a.shape_only:
            d, env = envnorm(d); s = s / env
            Gn = envnorm(G)[0]
        else:
            Gn = G
        def evaluate(s):
            """profile and (optionally) marginal log-likelihood of every model for noise level s(t); returns best_ll (marginal if requested), best_tau, best_A, profile max"""
            dn = d / s                                                         # whitened data
            best_ll = np.full(len(names), -np.inf); best_tau = np.zeros(len(names)); best_A = np.ones(len(names)); ll_tau = []
            Amin, Amax = a.amp_range
            for tau in TAUS:
                g = Gn[:, i0 + tau:i0 + tau + nt] / s[None, :]                   # whitened synthetic
                if Rinv is not None:
                    gR = g @ Rinv; Sgd = gR @ dn; Sgg = (gR * g).sum(1); Sdd = dn @ Rinv @ dn
                else:
                    Sgd = g @ dn; Sgg = (g * g).sum(1); Sdd = dn @ dn
                Ahat = Sgd / Sgg; A = np.clip(Ahat, Amin, Amax)
                ll = -0.5 * (Sdd - 2 * A * Sgd + A * A * Sgg)                    # = -0.5 * r^T C^-1 r at the (clipped) optimum
                upd = ll > best_ll; best_ll[upd] = ll[upd]; best_tau[upd] = tau; best_A[upd] = A[upd]
                if a.marginal:   # ∫ exp(-0.5 chi2(A)) dA / (Amax - Amin), flat prior on A
                    from scipy.special import erf
                    llA = -0.5 * (Sdd - Sgd ** 2 / Sgg) + 0.5 * np.log(2 * np.pi / Sgg) - np.log(Amax - Amin)
                    z = np.sqrt(Sgg / 2); frac = 0.5 * (erf((Amax - Ahat) * z) - erf((Amin - Ahat) * z))
                    ll_tau.append(llA + np.log(np.maximum(frac, 1e-300)))
            prof_max = best_ll.max(); prof_ll = best_ll.copy()
            if a.marginal:   # flat prior on tau: average over the shifts
                from scipy.special import logsumexp
                best_ll = logsumexp(np.array(ll_tau), axis=0) - np.log(len(TAUS))
            return best_ll, best_tau, best_A, prof_max, prof_ll
        best_ll, best_tau, best_A, prof_max, prof_ll = evaluate(s)
        nscale = 1.0; chi2_nom = -2 * prof_max / nt   # r^T C^-1 r / N of the best model (profile A, tau) at the bootstrap noise
        if a.noise_fit:   # per-cap noise inflation from the best profile fit (r^T C^-1 r / N of the best model of any class); then everything is recomputed with the inflated sigma
            if chi2_nom > 1.0:
                nscale = chi2_nom; s = s * np.sqrt(nscale); best_ll, best_tau, best_A, _, prof_ll = evaluate(s)
        ll = best_ll + lp_dT
        llmax = ll.max()
        res = {}
        for c in ("M0", "M1", "M2"):
            k = cls == c
            if not k.any():
                continue
            wts = np.exp(ll[k] - llmax); Z = wts.mean(); pp = wts / wts.sum(); ib = np.argmax(ll[k]); idx = np.where(k)[0]
            res[c] = dict(n=int(k.sum()), logZ=float(np.log(Z) + llmax), best=names[idx[ib]], best_ll=float(ll[idx[ib]]),
                          best_tau=float(best_tau[idx[ib]]), best_A=float(best_A[idx[ib]]), chi2_best=float(-2 * prof_ll[idx[ib]] / nt),
                          mean_dT=float((pp * (T2[k] - T1[k])).sum()), mean_df=float((pp * (f2[k] - f1[k])).sum()),
                          mean_T1=float((pp * T1[k]).sum()), mean_f1=float((pp * f1[k]).sum()),
                          post_T1={float(v): float(pp[T1[k] == v].sum()) for v in np.unique(T1[k])},
                          post_T2={float(v): float(pp[T2[k] == v].sum()) for v in np.unique(T2[k])},
                          post_f2={float(v): float(pp[f2[k] == v].sum()) for v in np.unique(f2[k])})
        lnB10 = res["M1"]["logZ"] - res["M0"]["logZ"] if "M1" in res else float("nan")
        lnB21 = res["M2"]["logZ"] - res["M1"]["logZ"] if ("M2" in res and "M1" in res) else float("nan")
        chi2 = chi2_nom
        ib = int(np.argmax(ll))
        lat, lon = re.findall(r"(n?\d+\.?\d*)_(\d+\.?\d*)", name)[0]
        lat = -float(lat[1:]) if lat.startswith("n") else float(lat); lon = float(lon)
        out = dict(bin=name, lat=lat, lon=lon, n_records=nrec, n_data=nt, chi2_red_best=float(chi2), lnB10=float(lnB10), lnB21=float(lnB21),
                   d660=a.d660, clapeyron=a.clapeyron, t2min=a.t2min, fmax=a.fmax, syn_caps=a.syn_caps, dT_prior=a.dT_prior, shape_only=a.shape_only, mech=bool(mech), noise_scale=float(nscale), marginal=bool(a.marginal), corr_noise=a.corr_noise, classes=res)
        json.dump(out, open(os.path.join(a.out, name + ".json"), "w"), indent=1)
        rows.append((lat, lon, chi2, lnB10, lnB21, res.get("M1", {}).get("mean_dT", float("nan")), res.get("M2", {}).get("mean_df", float("nan")),
                     f1[ib], f2[ib], T1[ib], T2[ib], best_A[ib], best_tau[ib], nrec))
        print(f"{name}: best {names[ib]} A={best_A[ib]:.2f} tau={best_tau[ib]:+.0f} chi2={chi2:.2f} lnB10={lnB10:+.1f} lnB21={lnB21:+.1f}", flush=True)
    np.savetxt(os.path.join(a.out, "summary.csv"), np.array(rows), delimiter=",", fmt="%.3f",
               header="lat,lon,chi2,lnB10,lnB21,dT_M1,df_M2,f1,f2,T1,T2,A,tau,nrec", comments="")


if __name__ == "__main__":
    main()
