#!/usr/bin/env python3
"""
Export per-cap tables from an invert_bins2 run directory.
  python export_fields.py RUN_DIR OUT_PREFIX
writes OUT_PREFIX_fields.csv  (lat, lon, P_M2, P_M1, dT_M1, df_M2, E_df, T_M0, f_M0)
   and OUT_PREFIX_results.csv (class, Bayes factors, posterior means, best model, A, tau, n_records)
"""
import sys, json, glob, numpy as np
run, out = sys.argv[1], sys.argv[2]
F, R = [], []
for f in sorted(glob.glob(f"{run}/*.json")):
    J = json.load(open(f)); c = J["classes"]
    lz = np.array([c[k]["logZ"] for k in ("M0", "M1", "M2")]); p = np.exp(lz - lz.max()); p /= p.sum()
    cls = "M2" if J["lnB21"] > 3 else ("M1" if J["lnB10"] > 3 else "M0"); b = c[cls]
    F.append((J["lat"], J["lon"], p[2], p[1], c["M1"]["mean_dT"], c["M2"]["mean_df"], p[2] * c["M2"]["mean_df"], c["M0"]["mean_T1"], c["M0"]["mean_f1"]))
    R.append([J["bin"], J["lat"], J["lon"], cls, J["lnB10"], J["lnB21"], J["chi2_red_best"], J.get("noise_scale", 1.0) ** 0.5, J["n_records"],
              c["M0"]["mean_T1"], c["M0"]["mean_f1"], c["M1"]["mean_T1"], c["M1"]["mean_dT"], c["M2"]["mean_T1"], c["M2"]["mean_dT"], c["M2"]["mean_f1"], c["M2"]["mean_df"], b["best"], b["best_A"], b["best_tau"]])
np.savetxt(out + "_fields.csv", np.array(F), delimiter=",", header="lat,lon,P_M2,P_M1,dT_M1,df_M2,E_df,T_M0,f_M0", comments="", fmt="%.4f")
with open(out + "_results.csv", "w") as fh:
    fh.write("bin,lat,lon,class,lnB10,lnB21,chi2_best,noise_inflation,n_records,T_M0,f_M0,T1_M1,dT_M1,T1_M2,dT_M2,f1_M2,df_M2,best_model,A,tau\n")
    for r in R: fh.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r) + "\n")
print(len(F), "caps ->", out + "_fields.csv,", out + "_results.csv")
