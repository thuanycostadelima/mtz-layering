#!/bin/bash
# Fourth round of tests: a cold lower layer in the thermal class (T2 floor lowered to 1300 K, 660 moved with the lower-layer
# temperature), combined 410/660 amplitude mismatch, chi2/N of every class by record count, amplitude-ratio residuals,
# block-bootstrap correlations and the regions that survive without the 660.
set -e
CODE=$(cd "$(dirname "$0")" && pwd)
X=crosssections_2024_SNR/SS/10deg_caps; I=crosssections_2024_SNR/SSinputs/10deg_caps
LIB="lib_shift_0.npz lib_shift_1.npz lib_shift_2.npz lib_shift_3.npz lib_shift50_0.npz"
COMMON="--noise-factor 1.0 --noise-fit"; V7="--corr-noise noise_R_SS10_v7.npz --marginal"
REC="--synthetic MM0.2_1600 --amp 0.6 --noise-obs"
run() { out=$1; shift; python $CODE/invert_bins2.py $X $I $out --lib $LIB --mech mech_filters_SS10_v7.npz $COMMON $V7 "$@" > $out.log; echo "$(date '+%H:%M') $out done: $(grep -m1 'models:' $out.log)"; }
run inv_v7_SS10_r4                                  # reference rerun (adds chi2/N of the best model of each class to the JSON files)
run inv_v7_t2min1300   --t2min 1300                 # lower layer (and uniform models) allowed down to 1300 K
run inv_v7_clap        --clapeyron -0.075           # 660 moves with the lower-layer temperature, -0.075 km/K (-2.8 MPa/K) about 1600 K
run inv_v7_clap_t1300  --clapeyron -0.075 --t2min 1300
run recmm_410up660dn   $REC --syn-410-amp 1.2 --syn-660-amp 0.8    # combined amplitude mismatch favouring f2 > f1
run recmm_410dn660up   $REC --syn-410-amp 0.8 --syn-660-amp 1.2    # and the opposite
echo "--- is M2 standing in for a cold lower layer?"
python $CODE/cold_m1.py inv_v7_SS10 inv_v7_t2min1300 inv_v7_clap inv_v7_clap_t1300
echo "--- chi2/N by record-count quartile: data, then the combined-mismatch synthetics"
python $CODE/chi2_by_quartile.py inv_v7_SS10_r4
python $CODE/chi2_by_quartile.py recmm_410up660dn
python $CODE/chi2_by_quartile.py recmm_410dn660up
echo "--- 660/410 amplitude residuals of the best models"
python $CODE/amp_ratio.py $X $I inv_v7_SS10_r4 --lib $LIB --mech mech_filters_SS10_v7.npz
echo "--- correlations with block bootstrap (pygmt env: needs netCDF4)"
python $CODE/block_bootstrap.py v7_SS10_fields.csv TX2011_percent.nc SSall.txt
echo "--- regions that survive without the 660"
python $CODE/region_overlap.py inv_v7_SS10 inv_v7_no660
