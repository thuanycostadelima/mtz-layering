#!/bin/bash
# Recovery tests against discontinuity mismatch: synthetic data from a UNIFORM model whose 660 or 410 is shifted / broadened
# relative to the library (data only), inverted against the unmodified library. Reports how often such data are classified M2.
set -e
CODE=$(cd "$(dirname "$0")" && pwd)
X=crosssections_2024_SNR/SS/10deg_caps; I=crosssections_2024_SNR/SSinputs/10deg_caps
LIB="lib_shift_0.npz lib_shift_1.npz lib_shift_2.npz lib_shift_3.npz lib_shift50_0.npz"
C="--mech mech_filters_SS10_v7.npz --noise-factor 1.0 --noise-fit --corr-noise noise_R_SS10_v7.npz --marginal --synthetic MM0.2_1600 --amp 0.6 --noise-obs"
run() { out=$1; shift; python $CODE/invert_bins2.py $X $I $out --lib $LIB $C "$@" > $out.log; echo "$(date '+%H:%M') $out done"; }
run recmm_660s3   --syn-660 3.5 0      # 660 3.5 s later (~ +10 km)
run recmm_660b6   --syn-660 0   6      # 660 broadened, sigma 6 s
run recmm_660s3b6 --syn-660 3.5 6      # both
run recmm_410s3   --syn-410 3.5 0
run recmm_410b6   --syn-410 0   6
run recmm_410s3b6 --syn-410 3.5 6
run recmm_660b3   --syn-660 0   3
run recmm_410b3   --syn-410 0   3
run recmm_both15  --syn-660 0 1.5 --syn-410 0 1.5   # the mismatch actually observed (pulse_widths.py)
python $CODE/compare_runs.py inv_v7_SS10 \
  "uniform, 660 shifted +3.5 s=recmm_660s3" "uniform, 660 broadened 6 s=recmm_660b6" "uniform, 660 shifted and broadened=recmm_660s3b6" \
  "uniform, 410 shifted +3.5 s=recmm_410s3" "uniform, 410 broadened 6 s=recmm_410b6" "uniform, 410 shifted and broadened=recmm_410s3b6" "uniform, 660 broadened 3 s=recmm_660b3" "uniform, 410 broadened 3 s=recmm_410b3" "uniform, both broadened 1.5 s=recmm_both15"
