#!/usr/bin/env python3
"""
Repick observed SS (or PP) precursor cross-sections for 410, 520/560 and 660,
with a picking-parameter sweep and explicit search for negative-polarity arrivals.

Input files: observed cross-sections, one per cap:  lat_lon.txt with columns
    t (s rel. main phase)   slowness (s/deg)   amp (rel. main)   bootstrap sigma
Depth conversion: PREM table (prem_sds_table.py), at the reference distance
inferred per bin from the file's own slowness column (or --ref to force one).

Usage:
  python pick_xsec.py XSEC_DIR OUT.csv --table prem_sds_S.npz [--phase SS] [--ref 145]
         [--radius 10] [--snr 1.96]

Output CSV, one row per bin:
  lat lon radius dist   z410 a410 sig410   z660 a660 sig660
  n_pos n_neg            number of significant mid-MTZ positive peaks / negative troughs
  z1 a1 pol1 stab1 ...   up to 3 mid-MTZ arrivals (470-620 km), strongest first:
                         depth, amp, polarity (+1/-1), stability = fraction of the
                         parameter sweep in which that arrival was found within 10 km
Mid-MTZ arrivals are only reported if |amp| > snr*sigma at the peak sample.
"""
import argparse, glob, os, re, itertools
import numpy as np
from scipy.signal import find_peaks


def load(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]


def infer_distance(t, p, tab, tmin, tmax):
    """Distance whose PREM (dt, dp) curve best matches the file's slowness column
    in the precursor window."""
    m = (t > tmin) & (t < tmax)
    best, bestD = np.inf, np.nan
    for j, D in enumerate(tab["dist"]):
        dt, dp = tab["dt"][:, j], tab["dp"][:, j]
        ok = np.isfinite(dt)
        pp = np.interp(t[m], dt[ok][::-1], dp[ok][::-1], left=np.nan, right=np.nan)
        r = np.nanmean((pp - p[m]) ** 2)
        if r < best:
            best, bestD = r, D
    return bestD, np.sqrt(best)


def depth_axis(t, D, tab):
    j = np.argmin(np.abs(tab["dist"] - D))
    dt = tab["dt"][:, j]
    ok = np.isfinite(dt)
    return np.interp(t, dt[ok][::-1], tab["depth"][ok][::-1], left=np.nan, right=np.nan)


def window_max(z, a, s, z0, z1, sign=+1):
    m = (z > z0) & (z < z1)
    if not m.any():
        return np.nan, np.nan, np.nan
    k = np.argmax(sign * a[m])
    return z[m][k], a[m][k], a[m][k] / s[m][k] if s[m][k] > 0 else np.nan


def mid_mtz(z, a, s, z0, z1, snr, sweep):
    """Return list of (depth, amp, polarity, stability) for significant arrivals."""
    m = (z > z0) & (z < z1)
    zz, aa, ss = z[m], a[m], s[m]
    if zz.size < 5 or not np.isfinite(aa).any():
        return []                                     # stack does not cover the window (short record or bad depth axis)
    dz = np.nanmedian(np.abs(np.diff(zz)))
    found = []  # (depth, amp, pol) per sweep combo
    for prom, width_km, dist_km in sweep:
        for sign in (+1, -1):
            idx, props = find_peaks(sign * aa, prominence=prom * np.nanmax(np.abs(aa)),
                                    width=max(1, int(width_km / dz)),
                                    distance=max(1, int(dist_km / dz)))
            for i in idx:
                if abs(aa[i]) > snr * ss[i]:
                    found.append((zz[i], aa[i], sign))
    if not found:
        return []
    found = np.array(found)
    # cluster by depth (10 km) and polarity; stability = fraction of sweep combos hit
    out = []
    used = np.zeros(len(found), bool)
    order = np.argsort(-np.abs(found[:, 1]))
    ncombo = len(sweep)
    for i in order:
        if used[i]:
            continue
        same = (np.abs(found[:, 0] - found[i, 0]) < 10) & (found[:, 2] == found[i, 2]) & ~used
        used |= same
        out.append((float(np.median(found[same, 0])), float(found[i, 1]), int(found[i, 2]),
                    min(1.0, same.sum() / ncombo)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xsec_dir")
    ap.add_argument("out")
    ap.add_argument("--table", default="prem_sds_S.npz")
    ap.add_argument("--phase", default="SS")
    ap.add_argument("--ref", type=float, default=124.0, help="reference distance (deg); 124 matches the slowness column of the 2024 files")
    ap.add_argument("--radius", type=float, default=np.nan)
    ap.add_argument("--snr", type=float, default=1.96)
    a = ap.parse_args()
    tab = np.load(a.table)

    # picking-parameter sweep: prominence (fraction of max |amp| in window),
    # min width (km), min separation (km)
    sweep = list(itertools.product((0.05, 0.1, 0.2), (5, 10, 20), (10, 20, 30)))

    rows = []
    files = sorted(glob.glob(os.path.join(a.xsec_dir, "*.txt")))
    for f in files:
        la, lo = re.findall(r"(n?-?\d+\.?\d*)_(-?\d+\.?\d*)", os.path.basename(f))[0]
        lat = -float(la[1:]) if la.startswith("n") else float(la); lon = float(lo)   # 'n' prefix = southern hemisphere
        t, p, amp, sig = load(f)
        if a.ref is None:
            D, _ = infer_distance(t, p, tab, -300, -100)
        else:
            D = a.ref
        z = depth_axis(t, D, tab)
        z410 = window_max(z, amp, sig, 350, 480)
        z660 = window_max(z, amp, sig, 600, 730)
        zlo, zhi = (480, 590) if a.phase == 'SS' else (490, 570)   # PP wavelet is shorter: lobes sit closer
        mids = mid_mtz(z, amp, sig, zlo, zhi, a.snr, sweep)
        # side-lobe diagnostics: trailing trough of the 410 (440-490) and leading trough of
        # the 660 (580-630), as fraction of the parent peak
        lobe410 = window_max(z, amp, sig, 440, 490, -1)[1] / z410[1] if z410[1] else np.nan
        lobe660 = window_max(z, amp, sig, 580, 630, -1)[1] / z660[1] if z660[1] else np.nan
        npos = sum(1 for m in mids if m[2] > 0)
        nneg = sum(1 for m in mids if m[2] < 0)
        row = [lat, lon, a.radius, D, *z410, *z660, lobe410, lobe660, npos, nneg]
        for k in range(3):
            row += list(mids[k]) if k < len(mids) else [np.nan] * 4
        rows.append(row)

    hdr = ("lat,lon,radius,dist,z410,a410,sig410,z660,a660,sig660,lobe410,lobe660,n_pos,n_neg,"
           "z1,a1,pol1,stab1,z2,a2,pol2,stab2,z3,a3,pol3,stab3")
    np.savetxt(a.out, np.array(rows, float), delimiter=",", header=hdr, comments="", fmt="%.4f")
    rows = np.array(rows, float)
    print(f"{len(rows)} bins; median inferred distance {np.nanmedian(rows[:, 3]):.1f} deg; "
          f"bins with >=1 mid-MTZ arrival {(rows[:, 12] + rows[:, 13] > 0).mean() * 100:.0f}%; "
          f"with a negative one {(rows[:, 13] > 0).mean() * 100:.0f}%; "
          f"with >=2 (split) {((rows[:, 12] + rows[:, 13]) >= 2).mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
