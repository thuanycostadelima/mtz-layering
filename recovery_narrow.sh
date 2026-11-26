#!/bin/bash
# Reverse mismatch test: the 410 (or 660) of the synthetic data NARROWER than the library by 7 % (the observed width difference, reversed).
set -e
CODE=$(cd "$(dirname "$0")" && pwd)
X=crosssections_2024_SNR/SS/10deg_caps; I=crosssections_2024_SNR/SSinputs/10deg_caps
LIB="lib_shift_0.npz lib_shift_1.npz lib_shift_2.npz lib_shift_3.npz lib_shift50_0.npz"
C="--mech mech_filters_SS10_v7.npz --noise-factor 1.0 --noise-fit --corr-noise noise_R_SS10_v7.npz --marginal --synthetic MM0.2_1600 --amp 0.6 --noise-obs"
python $CODE/invert_bins2.py $X $I recmm_410n07 --lib $LIB $C --syn-410-narrow 0.07 > recmm_410n07.log; echo "$(date '+%H:%M') 410 narrowed 7 % done"
python $CODE/invert_bins2.py $X $I recmm_410n20 --lib $LIB $C --syn-410-narrow 0.20 > recmm_410n20.log; echo "$(date '+%H:%M') 410 narrowed 20 % done"
python $CODE/invert_bins2.py $X $I recmm_660n07 --lib $LIB $C --syn-660-narrow 0.07 > recmm_660n07.log; echo "$(date '+%H:%M') 660 narrowed 7 % done"
python $CODE/compare_runs.py inv_v7_SS10 "uniform, 410 narrowed 7 %=recmm_410n07" "uniform, 410 narrowed 20 %=recmm_410n20" "uniform, 660 narrowed 7 %=recmm_660n07"
