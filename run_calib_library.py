#!/usr/bin/env python3
"""
Elementary-source Yspec library for one reference model, for per-bin stacking calibration.

Runs Yspec for 6 elementary moment tensors (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp = 1e26 N m, others 0)
at source depths 2.5..75 km (step --dz), receivers 90..180 deg every 0.5 deg on the lon=0
meridian, all three components kept. Any CMT's seismogram at any distance/depth is then a
linear combination of these (azimuth rotation handled at stacking time).

Usage (from mtzsyn/):
  python run_calib_library.py uniform_yspec/MM0.2/MM0.2_1600.model calib_lib --yspec-bin Yspec/bin --np 4
Resumable: a run whose yspec.out.* files exist is skipped. ~40 s per run; 6 x 30 = 180 runs ~ 2 h at --np 4.
Output: calib_lib/<tensor>_z<depth>/yspec.out.NNNN  (raw Yspec, kept)
"""
import argparse, glob, os, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
TENSORS = {"Mrr": (1, 0, 0, 0, 0, 0), "Mtt": (0, 0, 0, 1, 0, 0), "Mpp": (0, 0, 0, 0, 0, 1),
           "Mrt": (0, 1, 0, 0, 0, 0), "Mrp": (0, 0, 1, 0, 0, 0), "Mtp": (0, 0, 0, 0, 1, 0)}

TEMPLATE = open(os.path.join(HERE, "make_yspec_in.py")).read().split('TEMPLATE = """')[1].split('"""')[0]


def write_in(path, model, depth, mt, dmin, dmax, dstep, dt, tlen, fmax, lmax):
    import numpy as np
    dists = np.arange(dmin, dmax + 1e-9, dstep)
    recs = "\n".join(f"    {90.0 - d:.3f} 0.0" for d in dists)
    m0 = 1e26
    keys = ("mrr", "mrt", "mrp", "mtt", "mtp", "mpp")
    vals = dict(zip(keys, [m0 * x for x in mt]))
    txt = TEMPLATE.format(prefix="yspec.out", model=model, lmax=lmax, fmax=fmax, tlen=tlen, dt=dt,
                          f21=fmax - 5.0, depth=depth, nrec=len(dists), receivers=recs, **vals)
    open(path, "w").write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("out_dir")
    ap.add_argument("--yspec-bin", default="Yspec/bin"); ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--dz", type=float, default=2.5); ap.add_argument("--zmax", type=float, default=75.0)
    ap.add_argument("--dmin", type=float, default=90.0); ap.add_argument("--dmax", type=float, default=180.0)
    ap.add_argument("--dstep", type=float, default=0.5)
    ap.add_argument("--dt", type=float, default=0.5); ap.add_argument("--tlen", type=float, default=70.0)
    ap.add_argument("--fmax", type=float, default=100.0); ap.add_argument("--lmax", type=int, default=3000)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    import numpy as np
    depths = np.arange(a.dz, a.zmax + 1e-9, a.dz)
    os.makedirs(a.out_dir, exist_ok=True)
    log = open(os.path.join(a.out_dir, "calib.log"), "a")
    mname = os.path.basename(a.model)
    n_ok = n_skip = n_fail = 0
    for z in depths:
        for tname, mt in TENSORS.items():
            wd = os.path.join(a.out_dir, f"{tname}_z{z:05.1f}")
            if glob.glob(os.path.join(wd, "yspec.out.*")):
                n_skip += 1; continue
            os.makedirs(wd, exist_ok=True)
            for exe in ("yspec_pre", "yspec_cal", "yspec_pro", "yspec_mpi"):
                dst = os.path.join(wd, exe)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(os.path.join(a.yspec_bin, exe)), dst)
            shutil.copy(a.model, os.path.join(wd, mname))
            write_in(os.path.join(wd, "yspec.in"), mname, z, mt, a.dmin, a.dmax, a.dstep, a.dt, a.tlen, a.fmax, a.lmax)
            t0 = time.time()
            try:
                with open(os.path.join(wd, "yspec_output"), "w") as fh:
                    r = subprocess.run(["mpirun", "-n", str(a.np), "./yspec_mpi", "yspec.in"], cwd=wd,
                                       stdout=fh, stderr=subprocess.STDOUT, timeout=a.timeout)
                ok = r.returncode == 0 and len(glob.glob(os.path.join(wd, "yspec.out.*"))) > 0
            except subprocess.TimeoutExpired:
                subprocess.run(["pkill", "-f", "yspec_cal"], capture_output=True); ok = False
            n_ok += ok; n_fail += (not ok)
            print(f"{time.strftime('%H:%M:%S')} {tname} z={z:.1f} {'ok' if ok else 'FAILED'} {time.time()-t0:.0f}s", file=log, flush=True)
    print(f"done: {n_ok} ok, {n_skip} skipped, {n_fail} failed")


if __name__ == "__main__":
    main()
