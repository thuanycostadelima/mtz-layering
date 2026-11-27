#!/usr/bin/env python3
"""
Write the names of the caps whose uniform-class temperature is in the coldest fraction (default half), for a recovery test in
which the discontinuity perturbation is applied only where the lower transition zone is cold (--syn-caps).  Also reports the
fraction of caps whose fitted time shift sits at the bound (|tau| = 12 s).
    python cold_caps.py RUN_DIR cold_caps.txt [--frac 0.5]
"""
import sys, glob, json, numpy as np
run, out = sys.argv[1], sys.argv[2]; frac = float(sys.argv[sys.argv.index("--frac") + 1]) if "--frac" in sys.argv else 0.5
J = [json.load(open(f)) for f in sorted(glob.glob(f"{run}/*.json"))]
T = np.array([j["classes"]["M0"]["mean_T1"] for j in J]); cut = np.percentile(T, 100 * frac)
cold = [j["bin"] for j, t in zip(J, T) if t < cut]
open(out, "w").write("\n".join(cold) + "\n")
cls = np.array([2 if j["lnB21"] > 3 else (1 if j["lnB10"] > 3 else 0) for j in J])
print(f"{len(cold)} of {len(J)} caps with T_M0 < {cut:.0f} K written to {out}; they hold {np.sum(cls[T < cut] == 2)} of the {np.sum(cls == 2)} M2 caps")
tau = np.array([max((j["classes"][c]["best_tau"] for c in ("M0", "M1", "M2")), key=abs) for j in J])
tb = np.array([j["classes"][{0: "M0", 1: "M1", 2: "M2"}[c]]["best_tau"] for j, c in zip(J, cls)])
print(f"time shift of the best model of the assigned class: median {np.median(tb):+.0f} s, at the bound (|tau| = 12 s) in {100 * np.mean(np.abs(tb) >= 12):.0f} % of caps "
      f"(-12 s in {100 * np.mean(tb <= -12):.0f} %, +12 s in {100 * np.mean(tb >= 12):.0f} %); any class at the bound in {100 * np.mean(np.abs(tau) >= 12):.0f} %")
