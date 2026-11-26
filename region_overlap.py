#!/usr/bin/env python3
"""
Which regions of compositional layering survive when the 660 is excluded from the fit?  For the reference run and a
variant (e.g. the -195 to -110 s window), list by region the number of reference M2 caps, how many are still M2 in the
variant, how many keep P(M2) > 0.5, and the mean P(M2) of the region's caps in both runs.
    python region_overlap.py REF_DIR VARIANT_DIR
"""
import sys, glob, json, numpy as np

REGIONS = [("Central and South America, W South Atlantic", -60, 30, -100, -20), ("south-eastern Pacific", -60, 0, -150, -100),
           ("north-western Pacific", 10, 65, 120, 180), ("eastern Europe", 35, 70, 10, 60), ("south-west Pacific", -50, -10, 150, 190),
           ("Southern Ocean south of Africa", -65, -40, 0, 60)]


def load(d):
    J = {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        j = json.load(open(f)); c = j["classes"]
        J[j["bin"]] = (j["lat"], j["lon"], 2 if j["lnB21"] > 3 else (1 if j["lnB10"] > 3 else 0), np.exp(c["M2"]["logZ"]) / sum(np.exp(c[k]["logZ"]) for k in ("M0", "M1", "M2")))
    return J


A = load(sys.argv[1]); B = load(sys.argv[2]); keys = [k for k in A if k in B]
print(f"{'region':45s} {'caps':>5s} {'M2 ref':>7s} {'still M2':>9s} {'P>0.5':>6s} {'mean P ref':>11s} {'variant':>8s} {'r':>5s}")
for name, la0, la1, lo0, lo1 in REGIONS + [("whole globe", -90, 90, -180, 180)]:
    k = [b for b in keys if la0 <= A[b][0] <= la1 and any(lo0 <= A[b][1] + s <= lo1 for s in (-360, 0, 360))]
    m2 = [b for b in k if A[b][2] == 2]
    pa = np.array([A[b][3] for b in k]); pb = np.array([B[b][3] for b in k])
    print(f"{name:45s} {len(k):5d} {len(m2):7d} {sum(B[b][2] == 2 for b in m2):9d} {sum(B[b][3] > 0.5 for b in m2):6d} {pa.mean():11.2f} {pb.mean():8.2f} {np.corrcoef(pa, pb)[0, 1] if len(k) > 2 else float('nan'):5.2f}")
