#!/usr/bin/env python3
"""
Sensitivity of 520 / 560 / splitting detection rates to picking choices.

For one set of cross-sections (e.g. the 10 deg SS caps), re-pick every bin under a
grid of settings and report the fraction of bins with: a mid-MTZ arrival, a second
arrival (splitting), a negative arrival. Settings varied:
  - search window: 'wide' 470-620 km (positive search),
                   'lobe-safe' 480-590 km (excludes the 410 and 660 side lobes)
  - prominence threshold (fraction of max |amp| in the window): 0.02 ... 0.4
  - minimum peak width (km): 5, 10, 20
  - significance: |amp| > snr * sigma, snr in (1.0, 1.96, 3.0)
  - polarity: positive only vs both

Usage:
  python pick_sensitivity.py crosssections_2024/SS/10deg_caps sens_10.csv sens_10.png --table prem_sds_S.npz
"""
import argparse, glob, itertools, os
import numpy as np
from scipy.signal import find_peaks
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pick_xsec import load, depth_axis


def count_arrivals(z, a, s, z0, z1, prom, width_km, snr, both):
    m = (z > z0) & (z < z1)
    zz, aa, ss = z[m], a[m], s[m]
    dz = np.nanmedian(np.abs(np.diff(zz)))
    out = []
    for sign in ((+1, -1) if both else (+1,)):
        idx, _ = find_peaks(sign * aa, prominence=prom * np.nanmax(np.abs(aa)),
                            width=max(1, int(width_km / dz)), distance=max(1, int(10 / dz)))
        for i in idx:
            if abs(aa[i]) > snr * ss[i]:
                out.append((zz[i], sign))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xsec_dir"); ap.add_argument("out_csv"); ap.add_argument("out_png")
    ap.add_argument("--table", default="prem_sds_S.npz"); ap.add_argument("--ref", type=float, default=124.0)
    a = ap.parse_args()
    tab = np.load(a.table)
    traces = []
    for f in sorted(glob.glob(os.path.join(a.xsec_dir, "*.txt"))):
        t, p, amp, sig = load(f)
        traces.append((depth_axis(t, a.ref, tab), amp, sig))
    n = len(traces)

    windows = {"wide 470-620": (470, 620), "lobe-safe 480-590": (480, 590)}
    proms = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    widths = [5, 10, 20]
    snrs = [1.0, 1.96, 3.0]
    rows = []
    for (wname, (z0, z1)), prom, w, snr, both in itertools.product(windows.items(), proms, widths, snrs, (False, True)):
        n_any = n_split = n_neg = 0
        for z, amp, sig in traces:
            arr = count_arrivals(z, amp, sig, z0, z1, prom, w, snr, both)
            n_any += len(arr) > 0
            n_split += len(arr) >= 2
            n_neg += any(s < 0 for _, s in arr)
        rows.append((wname, prom, w, snr, int(both), n_any / n, n_split / n, n_neg / n))
    with open(a.out_csv, "w") as fh:
        fh.write("window,prominence,width_km,snr,both_polarities,frac_any,frac_split,frac_negative\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for wname in windows:
        ls = "-" if wname.startswith("lobe") else "--"
        for both, mk in ((0, "o"), (1, "s")):
            for snr, c in zip(snrs, ("0.6", "C0", "C3")):
                sel = [r for r in rows if r[0] == wname and r[4] == both and r[3] == snr and r[2] == 10]
                x = [r[1] for r in sel]
                lab = f"{wname} | {'both polarities' if both else 'positive only'} | snr {snr}"
                for k, col in ((0, 5), (1, 6), (2, 7)):
                    ax[k].plot(x, [r[col] for r in sel], ls, color=c, marker=mk, ms=3, lw=1, label=lab if k == 0 else None)
    for k, ttl in enumerate(("fraction with a mid-MTZ arrival", "fraction with >= 2 arrivals (splitting)", "fraction with a negative arrival")):
        ax[k].set_title(ttl, fontsize=10); ax[k].set_xlabel("prominence (fraction of window max)"); ax[k].set_ylim(0, 1); ax[k].grid(alpha=0.3)
    fig.legend(*ax[0].get_legend_handles_labels(), fontsize=7, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"{os.path.basename(a.xsec_dir.rstrip('/'))}: {n} bins, min width 10 km; dashed = wide window, solid = lobe-safe; circles = positive only, squares = both polarities", fontsize=10)
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(a.out_png, dpi=130)
    print(f"{n} bins, {len(rows)} settings -> {a.out_csv}, {a.out_png}")


if __name__ == "__main__":
    main()
