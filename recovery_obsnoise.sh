#!/bin/bash
# Recovery tests with each cap's own bootstrap sigma (instead of a fixed 0.004): tests whether detection follows record count.
set -e
CODE=$(cd "$(dirname "$0")" && pwd)        # directory of the scripts; data and outputs live in the current directory
DATA=${DATA:-crosssections_2024_SNR}; X=$DATA/SS/10deg_caps; I=$DATA/SSinputs/10deg_caps
LIB="lib_shift_0.npz lib_shift_1.npz lib_shift_2.npz lib_shift_3.npz lib_shift50_0.npz"
for m in MM0.2_1600 MM0.2-0.2_1700-1500_zc523_w0 MM0.2-0.4_1600-1700_zc523_w0 MM0.2-0.6_1700-1700_zc523_w0; do
  python $CODE/invert_bins2.py $X $I recobs_v7_$m --lib $LIB --mech mech_filters_SS10_v7.npz --noise-factor 1.0 --noise-fit \
         --corr-noise noise_R_SS10_v7.npz --marginal --synthetic $m --amp 0.6 --noise-obs > recobs_v7_$m.log
  echo "$(date '+%H:%M') $m done"
done
python - <<'PY'
import numpy as np, csv
ref = {}
for r in csv.DictReader(open("v7_SS10_results.csv")): ref[(round(float(r["lat"]), 2), round(float(r["lon"]), 2))] = int(r["n_records"])
for m in ["MM0.2_1600", "MM0.2-0.2_1700-1500_zc523_w0", "MM0.2-0.4_1600-1700_zc523_w0", "MM0.2-0.6_1700-1700_zc523_w0"]:
    for tag in ("rec_v7_", "recobs_v7_"):
        rows = list(csv.DictReader(open(f"{tag}{m}/summary.csv")))
        b10 = np.array([float(r["lnB10"]) for r in rows]); b21 = np.array([float(r["lnB21"]) for r in rows])
        n = np.array([ref.get((round(float(r["lat"]), 2), round(float(r["lon"]), 2)), -1) for r in rows]); ok = n > 0
        c = np.where(b21 > 3, 2, np.where(b10 > 3, 1, 0)); q = np.searchsorted(np.percentile(n[ok], [25, 50, 75]), n[ok], side="right"); cq = c[ok]
        print(f"{tag}{m:32s} M0/M1/M2 {np.mean(c==0)*100:3.0f}/{np.mean(c==1)*100:3.0f}/{np.mean(c==2)*100:3.0f} %   M2 by record quartile:",
              " ".join(f"{np.mean(cq[q==k]==2)*100:3.0f}" for k in range(4)), f"%   B>10: {np.mean(b21>2.3)*100:.0f} %")
PY
