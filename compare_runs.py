#!/usr/bin/env python3
"""Compare inversion variants: class fractions at ln B thresholds 3 (B=20) and 2.3 (B=10), P(M2) statistics, agreement with the reference.
   python compare_runs.py REF_DIR "LABEL=DIR" ["LABEL=DIR" ...] [--csv sensitivity.csv]
With --csv the table for Supplementary Fig. S7 is written (first row: reference at B>20; second row: reference at B>10)."""
import sys, glob, json, csv, numpy as np


def load(d):
    R = {}
    for f in sorted(glob.glob(f"{d}/*.json")):
        J = json.load(open(f)); c = J["classes"]; lz = np.array([c[k]["logZ"] for k in ("M0", "M1", "M2")]); p = np.exp(lz - lz.max()); p /= p.sum()
        R[J["bin"]] = dict(P=p, lnB10=J["lnB10"], lnB21=J["lnB21"], df=c["M2"]["mean_df"], dT=c["M1"]["mean_dT"], chi2=J["chi2_red_best"])
    return R


def classes(R, keys, thr):
    return np.array([2 if R[k]["lnB21"] > thr else (1 if R[k]["lnB10"] > thr else 0) for k in keys])


args = sys.argv[1:]; out_csv = None
if "--csv" in args: i = args.index("--csv"); out_csv = args[i + 1]; args = args[:i] + args[i + 2:]
ref = load(args[0]); keys = list(ref); Pref = np.array([ref[k]["P"][2] for k in keys]); cref = classes(ref, keys, 3.0)
print(f"{'variant':40s} {'B>20: M0/M1/M2':>18s} {'B>10: M0/M1/M2':>18s} {'P(M2)>.9':>9s} {'same cls':>9s} {'r(P_M2)':>8s} {'f2>f1':>6s} {'med|lnB21|':>10s}")
rows = []
for arg in [f"reference={args[0]}"] + args[1:]:
    lab, d = arg.split("=", 1); R = load(d); miss = [k for k in keys if k not in R]
    if miss: print(f"  ({lab}: {len(miss)} caps missing, skipped in the comparison)"); kk = [k for k in keys if k in R]
    else: kk = keys
    P = np.array([R[k]["P"][2] for k in kk]); Pr = np.array([ref[k]["P"][2] for k in kk]); cr = classes(ref, kk, 3.0); c20 = classes(R, kk, 3.0); c10 = classes(R, kk, 2.3)
    m2 = c20 == 2; f2 = np.mean(np.array([R[k]["df"] for k in kk])[m2] > 0) * 100 if m2.any() else np.nan
    print(f"{lab:40s} {np.mean(c20==0)*100:5.0f}/{np.mean(c20==1)*100:3.0f}/{np.mean(c20==2)*100:3.0f}     {np.mean(c10==0)*100:5.0f}/{np.mean(c10==1)*100:3.0f}/{np.mean(c10==2)*100:3.0f}     {np.mean(P>0.9)*100:6.0f}%  {np.mean(c20==cr)*100:7.0f}%  {np.corrcoef(P, Pr)[0,1]:7.2f}  {f2:5.0f}%  {np.median(np.abs([R[k]['lnB21'] for k in kk])):9.1f}")
    if lab == "reference":
        rows.append(["reference: correlated noise, marginal A,τ, B > 20", len(kk), *[f"{np.mean(c20==c)*100:.0f}" for c in (0, 1, 2)], f"{f2:.0f}", "1.00", "100"])
        m2b = c10 == 2; f2b = np.mean(np.array([R[k]["df"] for k in kk])[m2b] > 0) * 100 if m2b.any() else np.nan
        rows.append(["reference, B > 10", len(kk), *[f"{np.mean(c10==c)*100:.0f}" for c in (0, 1, 2)], f"{f2b:.0f}", "1.00", f"{np.mean(c10==cr)*100:.0f}"])
    else:
        rows.append([lab, len(kk), *[f"{np.mean(c20==c)*100:.0f}" for c in (0, 1, 2)], f"{f2:.0f}", f"{np.corrcoef(P, Pr)[0,1]:.2f}", f"{np.mean(c20==cr)*100:.0f}"])
if out_csv:
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "caps", "M0 (%)", "M1 (%)", "M2 (%)", "f₂ > f₁ in M2 (%)", "r(P(M2), ref)", "same class as ref (%)"]); w.writerows(rows)
    print("wrote", out_csv)
