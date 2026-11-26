#!/usr/bin/env python3
"""
Write a yspec.in for SS-precursor (or PP-precursor) synthetics in a given model.

Source at the north pole, receivers along the lon=0 meridian so that epicentral
distance = 90 - latitude. For SS we want SH on the transverse component, which
for this geometry is the phi (east) component; a pure M_{r,phi} source radiates
SH maximally along the meridian. For PP use an explosion (M_rr = M_tt = M_pp).

Usage
-----
  python make_yspec_in.py MODEL.model yspec.in --phase SS [--dmin 100 --dmax 170 --dstep 1]
         [--depth 10] [--tlen 70] [--dt 0.5] [--fmax 100] [--prefix yspec.out]

Defaults follow the paper's SI.3 (PREM-style run, dip-slip for SS, explosion for PP,
1 degree receiver step) and the yspec.in template of the Yspec installation.
"""
import argparse
import numpy as np

TEMPLATE = """# This file contains the parameters needed to run the
# program yspec

# prefix for output files
  {prefix}

# Earth Model
{model}

# attenuation switch: 1 = on, 0 = off
1

# gravitation: 0 = none, 1 = cowling, 2 = self
0

# output: 0 = displacement, 1 = velocity, 2 = acceleration
0

# potential and tilt corrections: 0 = no, 1 = yes
0

# lmin
  1

# lmax
  {lmax}

# fmin (mHz) (this should always be greater than 0)
  0.2

# fmax (mHz)
  {fmax}

#  length of  time series (min)
  {tlen}

# time step (sec)
  {dt}

# f11 filter (mHz)
  0.5

# f12 filter (mHz)
  1

# f21 filter (mHz)
  {f21}

# f22 filter (mHz)
  {fmax}

# source depth (km)
  {depth}

# source latitude (deg)
  90

# source longitude (deg)
  0

# M_{{r,r}} (Nm)
  {mrr}

# M_{{r,theta}} (Nm)
  {mrt}

# M_{{r,phi}} (Nm)
  {mrp}

# M_{{theta,theta}} (Nm)
  {mtt}

# M_{{theta,phi}} (Nm)
  {mtp}

# M_{{phi,phi}} (Nm)
  {mpp}

# receiver depth (km)
  0

# number of receivers
  {nrec}

# receiver latitudes and longitudes

{receivers}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--phase", choices=["SS", "PP"], default="SS")
    ap.add_argument("--dmin", type=float, default=100.0)
    ap.add_argument("--dmax", type=float, default=170.0)
    ap.add_argument("--dstep", type=float, default=1.0)
    ap.add_argument("--depth", type=float, default=10.0)
    ap.add_argument("--tlen", type=float, default=70.0, help="minutes")
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--fmax", type=float, default=100.0, help="mHz (100 mHz = 10 s)")
    ap.add_argument("--lmax", type=int, default=3000)
    ap.add_argument("--prefix", default="yspec.out")
    a = ap.parse_args()

    m0 = 1e26
    if a.phase == "SS":
        mt = dict(mrr=0, mrt=0, mrp=m0, mtt=0, mtp=0, mpp=0)   # dip-slip, SH along meridian
    else:
        mt = dict(mrr=m0, mrt=0, mrp=0, mtt=m0, mtp=0, mpp=m0)  # explosion

    dists = np.arange(a.dmin, a.dmax + 1e-9, a.dstep)
    recs = "\n".join(f"    {90.0 - d:.3f} 0.0" for d in dists)
    text = TEMPLATE.format(prefix=a.prefix, model=a.model, lmax=a.lmax, fmax=a.fmax,
                           tlen=a.tlen, dt=a.dt, f21=a.fmax - 5.0, depth=a.depth,
                           nrec=len(dists), receivers=recs, **mt)
    with open(a.out, "w") as fh:
        fh.write(text)
    print(f"wrote {a.out}: {a.phase}, {len(dists)} receivers {a.dmin}-{a.dmax} deg, "
          f"source {a.depth} km, dt {a.dt} s, {a.tlen} min, fmax {a.fmax} mHz")


if __name__ == "__main__":
    main()
