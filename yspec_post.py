#!/usr/bin/env python3
"""
Post-process a Yspec run (SS-precursor geometry from make_yspec_in.py) into
precursor cross-sections comparable with the observed ones.

For each receiver (distance Delta = 90 - lat):
  1. read yspec.out.NNNN  (time, r, theta, phi); take the phi column (SH for M_rphi)
  2. band-pass 15-50 s (SS) or 8-50 s (PP), zero phase
  3. find the SS peak near the PREM SS time, align so SS is at t = 0, normalise to +1
  4. keep the window [-400, +100] s at 0.1 s sampling, like the observed files
  5. convert time -> reflector depth with the PREM table (prem_sds_S.npz) at that Delta

Outputs, in OUT_DIR:
  xsec_<Delta>.txt : t(s)  depth(km)  amp     (one per receiver)
  summary.txt      : Delta, SS time, 410/520/660 peak times and depths, amplitudes
  overview.png     : record section + the trace at the reference distance

Usage:
  python yspec_post.py RUN_DIR OUT_DIR --table prem_sds_S.npz [--phase SS] [--ref 145]
"""
import argparse, glob, os, re
import numpy as np
from scipy.signal import butter, sosfiltfilt
from obspy.taup import TauPyModel


def read_run(run_dir):
    inp = glob.glob(os.path.join(run_dir, "*.in"))[0]
    txt = open(inp).read().splitlines()
    i = [k for k, l in enumerate(txt) if l.startswith("# receiver latitudes")][0]
    recs = [l.split() for l in txt[i + 1:] if l.strip()]
    lats = np.array([float(r[0]) for r in recs])
    dists = 90.0 - lats
    files = sorted(glob.glob(os.path.join(run_dir, "yspec.out.*")))
    assert len(files) == len(dists), f"{len(files)} outputs vs {len(dists)} receivers"
    return dists, files


def bandpass(x, dt, tmin, tmax):
    sos = butter(4, [1.0 / tmax, 1.0 / tmin], btype="band", fs=1.0 / dt, output="sos")
    return sosfiltfilt(sos, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--table", default="prem_sds_S.npz")
    ap.add_argument("--phase", default="SS", choices=["SS", "PP"])
    ap.add_argument("--ref", type=float, default=145.0)
    ap.add_argument("--src-depth", type=float, default=10.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    tmin, tmax = (15.0, 50.0) if a.phase == "SS" else (8.0, 50.0)
    col = 3 if a.phase == "SS" else 1          # phi (SH) for SS, r (vertical) for PP
    tab = np.load(a.table)
    model = TauPyModel("prem")
    dists, files = read_run(a.run_dir)

    tnew = np.arange(-400.0, 100.0 + 1e-9, 0.1)
    section = []
    rows = []
    for D, f in zip(dists, files):
        d = np.loadtxt(f)
        t, x = d[:, 0], d[:, col]
        dt = t[1] - t[0]
        x = bandpass(x, dt, tmin, tmax)
        arr = [p for p in model.get_travel_times(a.src_depth, D, [a.phase]) if p.name == a.phase]
        tss = min(arr, key=lambda p: p.time).time
        w = (t > tss - 60) & (t < tss + 60)
        i0 = np.argmax(np.abs(x[w]))
        tpk = t[w][i0]
        sgn = np.sign(x[w][i0])
        x = sgn * x / abs(x[w][i0])
        xs = np.interp(tnew, t - tpk, x)
        # depth axis for this distance from the PREM table (dt is monotone in depth)
        j = np.argmin(np.abs(tab["dist"] - D))
        dtcol = tab["dt"][:, j]
        ok = np.isfinite(dtcol)
        depth = np.interp(tnew, dtcol[ok][::-1], tab["depth"][ok][::-1], left=np.nan, right=np.nan)
        np.savetxt(os.path.join(a.out_dir, f"xsec_{D:05.1f}.txt"),
                   np.column_stack([tnew, depth, xs]), fmt="%9.2f %9.2f %12.5e")
        section.append(xs)
        # peaks in the usual windows (max |amp|, sign reported)
        def peak(z0, z1):
            m = (depth > z0) & (depth < z1)
            if not m.any():
                return (np.nan,) * 3
            k = np.argmax(np.abs(xs[m]))
            return tnew[m][k], depth[m][k], xs[m][k]
        p410, p520, p660 = peak(360, 460), peak(470, 600), peak(610, 730)
        rows.append((D, tss, tpk, *p410, *p520, *p660))

    rows = np.array(rows)
    hdr = ("dist tSS_prem tSS_pick t410 z410 a410 t520 z520 a520 t660 z660 a660")
    np.savetxt(os.path.join(a.out_dir, "summary.txt"), rows, fmt="%9.3f", header=hdr)

    # overview figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sec = np.array(section)
    fig, ax = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw=dict(width_ratios=[2, 1]))
    gain = 20.0
    for D, xs in zip(dists, sec):
        ax[0].plot(tnew, D + gain * np.clip(xs, -0.05, 0.05), "k", lw=0.5)
    for lab, c in (("t410", "b"), ("t520", "g"), ("t660", "r")):
        k = hdr.split().index(lab)
        ax[0].plot(rows[:, k], rows[:, 0], ".", color=c, ms=3, label=lab[1:])
    ax[0].set_xlim(-320, 40); ax[0].set_xlabel("time rel. SS (s)"); ax[0].set_ylabel("distance (deg)")
    ax[0].legend(loc="upper left"); ax[0].set_title(f"{os.path.basename(a.run_dir.rstrip('/'))}  {a.phase}, {tmin:.0f}-{tmax:.0f} s")
    jr = np.argmin(np.abs(dists - a.ref))
    z = np.loadtxt(os.path.join(a.out_dir, f"xsec_{dists[jr]:05.1f}.txt"))[:, 1]
    ax[1].plot(sec[jr], z, "k"); ax[1].axvline(0, color="0.7", lw=0.5)
    for zz in (410, 520, 660):
        ax[1].axhline(zz, color="0.8", lw=0.5)
    ax[1].set_ylim(800, 300); ax[1].set_xlim(-0.08, 0.08)
    ax[1].set_xlabel("amp / SS"); ax[1].set_ylabel("depth (km)"); ax[1].set_title(f"{dists[jr]:.0f} deg")
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "overview.png"), dpi=130)
    print(np.array2string(rows[np.argmin(np.abs(dists - a.ref))], precision=2))


if __name__ == "__main__":
    main()
