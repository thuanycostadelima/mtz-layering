#!/usr/bin/env python3
"""
Assemble the synthetic (T, f) catalogue from the uniform-model Yspec runs, working
in time relative to SS (not PREM depth), and compare with the observed picks.

Per model (runs/MM<f>_<T>/post/xsec_<ref>.txt):
  * 410 and 660 = the two largest positive peaks in [-300, -110] s with the 660
    earlier than the 410, and t410 - t660 between 40 and 130 s (S410S-S660S for
    any plausible MTZ thickness at this distance).
  * 520 = strongest extremum (either sign) strictly between the 660 and 410
    peaks, excluding +-22 s next to either (their side lobes).
  * amplitudes relative to SS (traces are normalised to SS = 1).
Depth conversion for the figures uses the same PREM table at the same reference
distance as the observed picks, so synthetic and observed are treated identically.

Outputs: catalogue.csv, catalogue.png, offset.png
Usage:
  python build_catalogue.py RUNS_DIR OUT_DIR --ref 124 --table prem_sds_S.npz --picks picks_SS_10.csv
"""
import argparse, glob, os, re
import numpy as np
from scipy.signal import find_peaks


def t2z(t, tab, D):
    j = np.argmin(np.abs(tab["dist"] - D))
    dt = tab["dt"][:, j]
    return np.interp(t, dt[::-1], tab["depth"][::-1], left=np.nan, right=np.nan)


def measure(t, a):
    m = (t > -300) & (t < -110)
    tt, aa = t[m], a[m]
    ip, pr = find_peaks(aa, prominence=0.003)
    if len(ip) < 2:
        return [np.nan] * 8
    cand = sorted(ip, key=lambda i: -aa[i])[:5]          # five strongest positives
    best = None
    for i in cand:
        for j in cand:
            if tt[j] < tt[i] and 40 <= tt[i] - tt[j] <= 130:
                s = aa[i] + aa[j]
                if best is None or s > best[0]:
                    best = (s, i, j)
    if best is None:
        return [np.nan] * 8
    _, i410, i660 = best
    t410, a410, t660, a660 = tt[i410], aa[i410], tt[i660], aa[i660]
    mid = (tt > t660 + 22) & (tt < t410 - 22)   # exclude the 410/660 side lobes (~half of a 40 s period)
    if mid.sum() < 3:
        return [t410, a410, t660, a660, np.nan, np.nan, np.nan, np.nan]
    k = np.argmax(np.abs(aa[mid]))
    t520, a520 = tt[mid][k], aa[mid][k]
    # second extremum of opposite sign in the same interval, if any (for splitting diagnostics)
    opp = np.argmin(np.sign(a520) * aa[mid])
    t2, a2 = tt[mid][opp], aa[mid][opp]
    return [t410, a410, t660, a660, t520, a520, t2, a2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir"); ap.add_argument("out_dir")
    ap.add_argument("--ref", type=float, default=124.0)
    ap.add_argument("--table", default="prem_sds_S.npz")
    ap.add_argument("--picks", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tab = np.load(a.table)

    rows = []
    for d in sorted(glob.glob(os.path.join(a.runs_dir, "MM*"))):
        m = re.match(r"MM(\d\.\d)_(\d{4})$", os.path.basename(d))
        fn = os.path.join(d, "post", f"xsec_{a.ref:05.1f}.txt")
        if not m or not os.path.exists(fn):
            continue
        t, _, amp = np.loadtxt(fn, unpack=True)
        rows.append((float(m.group(1)), int(m.group(2)), *measure(t, amp)))
    R = np.array(rows)
    hdr = "f,T,t410,a410,t660,a660,t520,a520,t520b,a520b"
    np.savetxt(os.path.join(a.out_dir, "catalogue.csv"), R, delimiter=",", header=hdr, comments="", fmt="%.4f")
    print(f"{len(R)} models; {np.isnan(R[:, 2]).sum()} without a 410/660 pair, {np.isnan(R[:, 6]).sum()} without a 520")

    f, T = R[:, 0], R[:, 1]
    z410, z660, z520 = t2z(R[:, 2], tab, a.ref), t2z(R[:, 4], tab, a.ref), t2z(R[:, 6], tab, a.ref)
    frac = (R[:, 6] - R[:, 4]) / (R[:, 2] - R[:, 4])      # 520 position as fraction of 660->410 interval

    obs = None
    if a.picks:
        p = np.genfromtxt(a.picks, delimiter=",", names=True)
        has = (p["n_pos"] + p["n_neg"]) > 0
        obs = dict(z410=np.nanpercentile(p["z410"], [25, 50, 75]), z660=np.nanpercentile(p["z660"], [25, 50, 75]),
                   z520=np.nanpercentile(p["z1"][has], [25, 50, 75]),
                   ratio=np.nanpercentile(p["a1"][has] / p["a410"][has], [25, 50, 75]))
        j = np.argmin(np.abs(tab["dist"] - a.ref)); dtc = tab["dt"][:, j]
        z2t = lambda z: np.interp(z, tab["depth"], dtc)
        obs["frac"] = (z2t(obs["z520"]) - z2t(obs["z660"][1])) / (z2t(obs["z410"][1]) - z2t(obs["z660"][1]))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fs = np.unique(f); cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(2, 3, figsize=(15, 8)); ax = ax.ravel()
    panels = ((z410, "410 depth (km)"), (z660, "660 depth (km)"), (z660 - z410, "MTZ thickness (km)"),
              (z520, "520 depth (km), strongest mid-MTZ extremum"), (R[:, 7] / R[:, 3], "a520 / a410 (signed)"), (frac, "520 position, fraction of 660-410 interval"))
    for k, (y, lab) in enumerate(panels):
        for ff in fs:
            s = f == ff; o = np.argsort(T[s])
            ax[k].plot(T[s][o], y[s][o], "-o", color=cmap(ff / 1.0), ms=3, lw=1, label=f"f={ff:.1f}" if k == 0 else None)
        ax[k].set_xlabel("Tpot (K)"); ax[k].set_title(lab, fontsize=10); ax[k].grid(alpha=0.3)
    if obs:
        for k, key in ((0, "z410"), (1, "z660"), (3, "z520"), (4, "ratio"), (5, "frac")):
            lo, me, hi = obs[key]; ax[k].axhspan(lo, hi, color="r", alpha=0.12); ax[k].axhline(me, color="r", lw=1)
        lo, me, hi = obs["z660"] - obs["z410"][::-1]; ax[2].axhspan(lo, hi, color="r", alpha=0.12); ax[2].axhline(obs["z660"][1] - obs["z410"][1], color="r", lw=1)
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Synthetic catalogue at {a.ref:.0f} deg (SS, 15-50 s), PREM depth at the same distance as the data; red = observed 10 deg picks (median, IQR)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "catalogue.png"), dpi=130)

    if obs:
        # models whose 410 and 660 both fall within the observed IQR: where is their 520?
        ok = (z410 > obs["z410"][0]) & (z410 < obs["z410"][2]) & (z660 > obs["z660"][0]) & (z660 < obs["z660"][2])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        sc = ax[0].scatter(T[ok], f[ok], c=z520[ok] - obs["z520"][1], cmap="coolwarm", vmin=-60, vmax=60, s=60, edgecolor="k")
        ax[0].set_xlabel("Tpot (K)"); ax[0].set_ylabel("basalt fraction"); ax[0].set_title("models matching observed 410 and 660 (IQR)\ncolour = synthetic z520 - observed median", fontsize=9)
        plt.colorbar(sc, ax=ax[0], label="km")
        ax[1].hist((z520[ok] - obs["z520"][1])[np.isfinite(z520[ok])], bins=np.arange(-80, 81, 10), color="0.5", edgecolor="k")
        ax[1].axvline(0, color="r"); ax[1].set_xlabel("synthetic z520 - observed median (km)"); ax[1].set_title(f"{ok.sum()} matching models", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "offset.png"), dpi=130)
        dz = (z520[ok] - obs["z520"][1]); dz = dz[np.isfinite(dz)]
        if len(dz) == 0:
            print("no models match the observed 410 and 660 IQR"); return
        print(f"models matching observed 410 & 660: {ok.sum()}; their z520 - observed: median {np.median(dz):+.1f} km, IQR {np.percentile(dz,25):+.0f} to {np.percentile(dz,75):+.0f}")
        print("matching (T, f):", sorted(zip(T[ok].astype(int), f[ok])))


if __name__ == "__main__":
    main()
