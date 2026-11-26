#!/usr/bin/env python3
"""
Pack the elementary-source Yspec library (run_calib_library.py output) into compact .npz files.

For each calib_lib/<tensor>_z<depth>/ folder: reads yspec.out.NNNN (t, r, theta, phi displacement),
keeps the time window [--tmin, --tmax] (default 1400..3200 s, which covers SS and PP and their
precursors for 90-180 deg), stores float32 arrays.

Output: calib_pack/<tensor>_z<depth>.npz with
    t      [nt]          time (s)
    dist   [nrec]        epicentral distance (deg), 90..180 step 0.5
    r      [nrec, nt]    vertical component
    phi    [nrec, nt]    transverse component (SH for the lon=0 geometry)
    theta  [nrec, nt]    radial-horizontal component
Resumable; ~5 MB per folder, ~1 GB in total.

Usage: python pack_calib.py calib_lib calib_pack [--tmin 1400 --tmax 3200]
"""
import argparse, glob, os, re, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib"); ap.add_argument("out")
    ap.add_argument("--tmin", type=float, default=1400.0); ap.add_argument("--tmax", type=float, default=3200.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    folders = sorted(d for d in glob.glob(os.path.join(a.lib, "*_z*")) if os.path.isdir(d))
    for d in folders:
        name = os.path.basename(d)
        outf = os.path.join(a.out, name + ".npz")
        if os.path.exists(outf):
            continue
        files = sorted(glob.glob(os.path.join(d, "yspec.out.*")), key=lambda f: int(f.rsplit(".", 1)[1]))
        if not files:
            print(name, "no output, skipped"); continue
        # receiver distances from yspec.in
        lines = open(os.path.join(d, "yspec.in")).read().splitlines()
        k = [i for i, l in enumerate(lines) if l.startswith("# receiver latitudes")][0]
        recs = [l.split() for l in lines[k + 1:] if l.strip()]
        lat = np.array([float(r[0]) for r in recs])
        dist = 90.0 - lat
        if len(dist) != len(files):
            print(name, f"WARNING {len(files)} files but {len(dist)} receivers in yspec.in"); dist = dist[:len(files)]
        R = TH = PH = t = None
        for i, f in enumerate(files):
            x = np.loadtxt(f)
            if t is None:
                m = (x[:, 0] >= a.tmin) & (x[:, 0] <= a.tmax)
                t = x[m, 0].astype(np.float32)
                R = np.zeros((len(files), m.sum()), np.float32); TH = R.copy(); PH = R.copy()
            R[i] = x[m, 1]; TH[i] = x[m, 2]; PH[i] = x[m, 3]
        np.savez_compressed(outf, t=t, dist=dist.astype(np.float32), r=R, theta=TH, phi=PH)
        print(name, "packed", R.shape, flush=True)


if __name__ == "__main__":
    main()
