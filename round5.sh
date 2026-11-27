#!/bin/bash
# Fifth round: 410/660 widths, amplitudes and scatter by class and temperature; a temperature-tracking 660 perturbation
# (broadened and weakened 660 in the coldest half of the caps only); the basalt-fraction prior (f <= 0.4; uniform models
# up to f = 1.0); time shifts at the bound.
set -e
CODE=$(cd "$(dirname "$0")" && pwd)
X=crosssections_2024_SNR/SS/10deg_caps; I=crosssections_2024_SNR/SSinputs/10deg_caps
LIB="lib_shift_0.npz lib_shift_1.npz lib_shift_2.npz lib_shift_3.npz lib_shift50_0.npz"
COMMON="--noise-factor 1.0 --noise-fit"; V7="--corr-noise noise_R_SS10_v7.npz --marginal"
REC="--synthetic MM0.2_1600 --amp 0.6 --noise-obs"
run() { out=$1; shift; python $CODE/invert_bins2.py $X $I $out --lib $LIB --mech mech_filters_SS10_v7.npz $COMMON $V7 "$@" > $out.log; echo "$(date '+%H:%M') $out done: $(grep -m1 'models:' $out.log)"; }
echo "--- pulse widths, amplitudes and scatter by class and temperature"
python $CODE/pulse_by_class.py $X $I inv_v7_SS10_r4 --lib $LIB --mech mech_filters_SS10_v7.npz
echo "--- cold caps and time shifts at the bound"
python $CODE/cold_caps.py inv_v7_SS10 cold_caps.txt
run recmm_cold660     $REC --syn-caps cold_caps.txt --syn-660 0 3 --syn-660-amp 0.8     # 660 broadened 3 s and weakened 20 % in the coldest half only
run recmm_cold660_6s  $REC --syn-caps cold_caps.txt --syn-660 0 6 --syn-660-amp 0.8     # the same with 6 s
run inv_v7_f04        --exclude 0.6                                                       # basalt fraction <= 0.4 in every class
run inv_v7_fmax10     --fmax 1.0                                                          # uniform models up to f = 1.0 (layered models exist only to 0.6)
python $CODE/compare_runs.py inv_v7_SS10 "recovery, uniform model, 660 broadened 3 s and x 0.8 in the coldest half of caps=recmm_cold660" \
  "recovery, uniform model, 660 broadened 6 s and x 0.8 in the coldest half of caps=recmm_cold660_6s" \
  "basalt fraction <= 0.4 in every class=inv_v7_f04" "uniform models up to f = 1.0=inv_v7_fmax10"
echo "--- where do the false detections of the cold-cap test fall?"
python $CODE/region_overlap.py inv_v7_SS10 recmm_cold660
python $CODE/chi2_by_quartile.py recmm_cold660
