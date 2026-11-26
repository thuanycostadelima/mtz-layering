#!/usr/bin/env python3
"""
Does the most flexible class win simply because the noise shrinks?  For a run whose JSON files carry chi2_best per class
(invert_bins2.py from round 4), report r^T C^-1 r / N of the best model of each class, by quartile of record count,
for all caps and for the M2 caps, together with the fraction of caps whose noise was inflated and the fraction excluding
layering (P(M2) < 0.05).  If the 1-D library were systematically wrong for well-sampled caps, chi2/N of every class
would rise above 1 with record count and inflation would become the rule there.
    python chi2_by_quartile.py RUN_DIR
"""
import sys, glob, json, numpy as np

rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}/*.json")):
    j = json.load(open(f)); c = j["classes"]
    if "chi2_best" not in c["M0"]: sys.exit("no chi2_best in the JSON files: rerun the inversion with the round-4 invert_bins2.py")
    cls = 2 if j["lnB21"] > 3 else (1 if j["lnB10"] > 3 else 0)
    P2 = np.exp(c["M2"]["logZ"]) / sum(np.exp(c[k]["logZ"]) for k in ("M0", "M1", "M2"))
    ns = j["noise_scale"]   # chi2_best is at the inflated sigma; multiply back to the bootstrap sigma so that all caps are comparable
    rows.append((j["n_records"], cls, c["M0"]["chi2_best"] * ns, c["M1"]["chi2_best"] * ns, c["M2"]["chi2_best"] * ns, ns, P2))
R = np.array(rows, float); q = np.percentile(R[:, 0], [0, 25, 50, 75, 100]); iq = np.clip(np.digitize(R[:, 0], q) - 1, 0, 3)
print(f"{len(R)} caps. chi2/N of the best model of each class at the bootstrap sigma, before any inflation (median over caps)")
print(f"{'quartile':10s} {'records':>11s} {'n':>4s} | {'all caps: M0':>12s} {'M1':>6s} {'M2':>6s} {'inflated':>9s} {'P2<.05':>7s} | {'M2 caps: n':>10s} {'M0':>6s} {'M1':>6s} {'M2':>6s} {'inflated':>9s} {'M2 frac':>8s}")
for k in range(4):
    m = iq == k; m2 = m & (R[:, 1] == 2); r = R[m]; r2 = R[m2]
    print(f"Q{k + 1:<9d} {q[k]:5.0f}-{q[k + 1]:5.0f} {m.sum():4d} | {np.median(r[:, 2]):12.2f} {np.median(r[:, 3]):6.2f} {np.median(r[:, 4]):6.2f} {100 * np.mean(r[:, 5] > 1):8.0f} % {100 * np.mean(r[:, 6] < 0.05):6.0f} % | "
          + (f"{m2.sum():10d} {np.median(r2[:, 2]):6.2f} {np.median(r2[:, 3]):6.2f} {np.median(r2[:, 4]):6.2f} {100 * np.mean(r2[:, 5] > 1):8.0f} %" if m2.any() else f"{0:10d}{'':30s}") + f" {100 * m2.sum() / m.sum():7.0f} %")
m2 = R[:, 1] == 2
print(f"\nall M2 caps ({m2.sum()}): chi2/N of best M1 median {np.median(R[m2, 3]):.2f} (5-95 %: {np.percentile(R[m2, 3], 5):.2f}-{np.percentile(R[m2, 3], 95):.2f}), best M2 {np.median(R[m2, 4]):.2f}; "
      f"best M1 > 1 in {100 * np.mean(R[m2, 3] > 1):.0f} % of M2 caps, best M2 > 1 in {100 * np.mean(R[m2, 4] > 1):.0f} %")
print(f"all caps: best M1 chi2/N median {np.median(R[:, 3]):.2f}; correlation of best-M1 chi2/N with log(records): {np.corrcoef(np.log(R[:, 0]), R[:, 3])[0, 1]:.2f}; of best-M2 chi2/N: {np.corrcoef(np.log(R[:, 0]), R[:, 4])[0, 1]:.2f}")
