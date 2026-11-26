#!/usr/bin/env python3
"""
Build a lookup table of SdS-SS (or PdP-PP) differential time and slowness in PREM
for underside reflections at arbitrary depth d and epicentral distance Delta.

TauP only reflects at model discontinuities, so for each depth we build a PREM
copy with a vanishingly small velocity jump at that depth and ask for S^dS.
Result cached to prem_sds_table.npz with arrays depth (km), dist (deg),
dt (s, SdS-SS, negative), dp (s/deg, p_SdS - p_SS).

Usage:  python prem_sds_table.py [--phase S|P] [--src-depth 10]
"""
import argparse, os, re, tempfile
import numpy as np
import obspy
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model


def prem_lines():
    src = os.path.join(os.path.dirname(obspy.__file__), "taup", "data", "prem.nd")
    return open(src).read().splitlines()


def model_with_reflector(depth, workdir):
    """Write prem_dXXX.nd with a 1e-4 relative jump at `depth` and build it."""
    lines = prem_lines()
    out = []
    inserted = False
    prev = None
    for ln in lines:
        if not inserted and ln.strip() and ln.strip()[0].isdigit():
            z = float(ln.split()[0])
            if abs(z - depth) < 1e-6:
                p = ln.split()
                vp, vs = float(p[1]), float(p[2])
                out.append(ln)
                out.append(f"{depth:8.2f} {vp*1.0001:10.5f} {vs*1.0001:9.5f} {p[3]:>9s} {p[4]:>9s} {p[5]:>8s}")
                inserted = True
                prev = (z, vp, vs, float(p[3]), float(p[4]), float(p[5]))
                continue
            if prev is not None and prev[0] < depth < z:
                # interpolate at depth, write two rows with a tiny jump
                z0, vp0, vs0, r0, qk, qm = prev
                z1, vp1, vs1, r1 = z, *[float(x) for x in ln.split()[1:4]]
                f = (depth - z0) / (z1 - z0)
                vp, vs, rho = vp0 + f * (vp1 - vp0), vs0 + f * (vs1 - vs0), r0 + f * (r1 - r0)
                out.append(f"{depth:8.2f} {vp:10.5f} {vs:9.5f} {rho:9.5f} {qk:9.1f} {qm:8.1f}")
                out.append(f"{depth:8.2f} {vp*1.0001:10.5f} {vs*1.0001:9.5f} {rho:9.5f} {qk:9.1f} {qm:8.1f}")
                inserted = True
            p = ln.split()
            prev = (z, float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5]))
        out.append(ln)
    name = f"prem_d{int(round(depth)):04d}"
    nd = os.path.join(workdir, name + ".nd")
    open(nd, "w").write("\n".join(out) + "\n")
    build_taup_model(nd, output_folder=workdir, verbose=False)
    return os.path.join(workdir, name + ".npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="S", choices=["S", "P"])
    ap.add_argument("--src-depth", type=float, default=10.0)
    ap.add_argument("--depths", nargs=3, type=float, default=[300, 900, 5])
    ap.add_argument("--dists", nargs=3, type=float, default=[100, 170, 1])
    ap.add_argument("--out", default="prem_sds_table.npz")
    a = ap.parse_args()

    depths = np.arange(a.depths[0], a.depths[1] + 1e-9, a.depths[2])
    dists = np.arange(a.dists[0], a.dists[1] + 1e-9, a.dists[2])
    main_ph = a.phase * 2
    dt = np.full((len(depths), len(dists)), np.nan)
    dp = np.full_like(dt, np.nan)

    ref = TauPyModel("prem")
    tmain = {}
    for j, D in enumerate(dists):
        arr = [x for x in ref.get_travel_times(a.src_depth, D, [main_ph]) if x.name == main_ph]
        arr = min(arr, key=lambda x: x.time)
        tmain[j] = (arr.time, arr.ray_param_sec_degree)

    with tempfile.TemporaryDirectory() as wd:
        for i, d in enumerate(depths):
            mp = model_with_reflector(d, wd)
            m = TauPyModel(mp)
            ph = f"{a.phase}^{int(round(d))}{a.phase}"
            for j, D in enumerate(dists):
                arr = [x for x in m.get_travel_times(a.src_depth, D, [ph]) if x.name == ph]
                if not arr:
                    continue
                x = min(arr, key=lambda x: x.time)
                dt[i, j] = x.time - tmain[j][0]
                dp[i, j] = x.ray_param_sec_degree - tmain[j][1]
            print(f"d={d:5.0f} km  dt@145={dt[i, np.argmin(abs(dists-145))]:7.1f} s", flush=True)
    np.savez(a.out, depth=depths, dist=dists, dt=dt, dp=dp, phase=a.phase, src_depth=a.src_depth)
    print("saved", a.out)


if __name__ == "__main__":
    main()
