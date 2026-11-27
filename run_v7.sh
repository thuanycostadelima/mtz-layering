#!/bin/bash
# Full analysis chain (version 7 = the data set of the paper). Run from mtzsyn/ in the mtz520 env:
#
#   (data: crosssections_2024_SNR/SS (stacks) and /SSinputs (record lists), from the Zenodo deposit)
#   bash run_v7.sh check            # verify the new data layout and count caps/records per radius (seconds)
#   bash run_v7.sh stacks 10        # real-CMT + fixed-source stacks and mechanism filter for one radius (10° ≈ 30–60 min; 5° longest)
#   bash run_v7.sh stacksB          # the model-B stacks at 10° (for the filter-independence test, Fig. S6b)
#   bash run_v7.sh noise            # noise autocorrelation at 10° (needs stacks 10 first)
#   bash run_v7.sh invert 10        # v7 inversion for one radius (needs stacks <r> and noise)
#   bash run_v7.sh recovery         # the four recovery tests at 10°
#   bash run_v7.sh variants         # the S7 sensitivity variants at 10°  (+ writes sensitivity_v7.csv)
#   bash run_v7.sh pp               # PP calibration stacks at 10° (Fig. 2f)      bash run_v7.sh picks   # pick files for S3/S9
#   bash run_v7.sh export           # fields/results CSVs for every radius that has been inverted
#   bash run_v7.sh all              # everything, in order (several hours; resumable — rerun the same command after an interruption)
#
# Outputs carry the suffix v7 / SS<r>_v7; finished steps are skipped on a rerun.
set -e
CODE=$(cd "$(dirname "$0")" && pwd)        # directory of the scripts; data and outputs live in the current directory
DATA=${DATA:-crosssections_2024_SNR}       # $DATA/SS/<r>deg_caps (stacks) and $DATA/SSinputs/<r>deg_caps (record lists)
RADII="5 7.5 10 12.5 15 17.5 20"
LIB="lib_shift_0.npz lib_shift_1.npz lib_shift_2.npz lib_shift_3.npz lib_shift50_0.npz"
COMMON="--noise-factor 1.0 --noise-fit"
V7="--corr-noise noise_R_SS10_v7.npz --marginal"
xs() { echo $DATA/SS/${1}deg_caps; }
inp() { echo $DATA/SSinputs/${1}deg_caps; }
stamp() { echo "$(date '+%d %b %H:%M')  $*"; }

check() {
  for r in $RADII; do
    X=$(xs $r); I=$(inp $r)
    [ -d "$X" ] || { echo "MISSING $X"; continue; }; [ -d "$I" ] || { echo "MISSING $I"; continue; }
    nx=$(ls $X/*.txt 2>/dev/null | wc -l | tr -d ' '); ni=$(ls $I/*.txt 2>/dev/null | wc -l | tr -d ' ')
    f=$(ls $X/*.txt | head -1); cols=$(head -1 $f | awk '{print NF}'); fi=$(ls $I/*.txt | head -1); icols=$(head -1 $fi | awk '{print NF}')
    nrec=$(cat $I/*.txt | wc -l | tr -d ' ')
    echo "radius $r: $nx stacks ($cols columns), $ni record lists ($icols columns), $nrec cap-records"
  done
  python - <<'EOF'
import glob, os, sys
D = os.environ.get("DATA", "crosssections_2024_SNR"); X = sorted(glob.glob(f"{D}/SS/10deg_caps/*.txt")); I = sorted(glob.glob(f"{D}/SSinputs/10deg_caps/*.txt"))
nx = {os.path.basename(f) for f in X}; ni = {os.path.basename(f) for f in I}
print(f"10°: {len(nx & ni)} caps with both files, {len(nx - ni)} stacks without record list, {len(ni - nx)} record lists without stack")
import numpy as np
t, p, x, s = np.loadtxt(X[0], unpack=True); print(f"stack columns look like: t {t[0]:.0f}..{t[-1]:.0f} s (dt {t[1]-t[0]:.2f}), slowness {p.min():.3f}..{p.max():.3f}, amp max {abs(x).max():.3f}, sigma median {np.median(s):.4f}")
ids = set()
for f in I: ids |= {l.split()[0] for l in open(f) if l.strip() and not l.startswith('#')}
print(f"unique seismogram IDs at 10°: {len(ids)}  (108 599 SS seismograms in total)")
EOF
}

stacks() {
  r=$1; X=$(xs $r); I=$(inp $r); stamp "radius $r: $(ls $X/*.txt | wc -l | tr -d ' ') caps"
  python $CODE/stack_bins.py $I $X calib_pack gcmt_all.ndk calib_stacks_v7/SS${r} --phase SS --ref 124 --nproc 1 > /dev/null; stamp "real-CMT stacks done"
  python $CODE/stack_bins.py $I $X calib_pack gcmt_all.ndk calib_stacks_v7/SS${r}_fixed --phase SS --ref 124 --fixed-source > /dev/null; stamp "fixed-source stacks done"
  python $CODE/mech_filter.py calib_stacks_v7/SS${r} calib_stacks_v7/SS${r}_fixed mech_filters_SS${r}_v7.npz
}

stacksB() {
  X=$(xs 10); I=$(inp 10)
  PACKB=calib_pack_B; ls $PACKB/*_z*.npz > /dev/null 2>&1 || PACKB=calib_pack_B_wide     # library B was repacked with the wide window for PP; it covers SS too
  ls $PACKB/*_z*.npz > /dev/null 2>&1 || { echo "no packed library B (calib_pack_B or calib_pack_B_wide)"; exit 1; }; stamp "model-B stacks from $PACKB"
  python $CODE/stack_bins.py $I $X $PACKB gcmt_all.ndk calib_stacks_v7/SS10_B --phase SS --ref 124 --nproc 1 > /dev/null
  python $CODE/stack_bins.py $I $X $PACKB gcmt_all.ndk calib_stacks_v7/SS10_B_fixed --phase SS --ref 124 --fixed-source > /dev/null
  python $CODE/mech_filter.py calib_stacks_v7/SS10_B calib_stacks_v7/SS10_B_fixed mech_filters_SS10_B_v7.npz; stamp "model-B stacks and filter done"
}

noise() {
  # a first pass with independent samples gives the best-fit residuals from which R is measured
  python $CODE/invert_bins2.py $(xs 10) $(inp 10) inv_v7_pre_SS10 --lib $LIB --mech mech_filters_SS10_v7.npz $COMMON > inv_v7_pre_SS10.log
  python $CODE/noise_corr.py $(xs 10) $(inp 10) inv_v7_pre_SS10 noise_R_SS10_v7.npz --lib $LIB --mech mech_filters_SS10_v7.npz; stamp "noise correlation done (N_eff above)"
}

invert() {
  r=$1; python $CODE/invert_bins2.py $(xs $r) $(inp $r) inv_v7_SS${r} --lib $LIB --mech mech_filters_SS${r}_v7.npz $COMMON $V7 > inv_v7_SS${r}.log
  stamp "inversion SS$r done: $(ls inv_v7_SS${r}/*.json | wc -l | tr -d ' ') caps"
}

recovery() {
  for m in MM0.2_1600 MM0.2-0.2_1700-1500_zc523_w0 MM0.2-0.4_1600-1700_zc523_w0 MM0.2-0.6_1700-1700_zc523_w0; do
    python $CODE/invert_bins2.py $(xs 10) $(inp 10) rec_v7_$m --lib $LIB --mech mech_filters_SS10_v7.npz $COMMON $V7 --synthetic $m --amp 0.6 --noise 0.004 > rec_v7_$m.log; stamp "recovery $m done"
  done
}

variants() {
  run() { out=$1; shift; python $CODE/invert_bins2.py $(xs 10) $(inp 10) $out --lib $LIB "$@" > $out.log; stamp "$out done"; }
  run inv_v7_profile  --mech mech_filters_SS10_v7.npz   $COMMON --corr-noise noise_R_SS10_v7.npz
  run inv_v7_nf15     --mech mech_filters_SS10_v7.npz   --noise-factor 1.5 --noise-fit
  run inv_v7_nf20     --mech mech_filters_SS10_v7.npz   --noise-factor 2.0 --noise-fit
  run inv_v7_prior    --mech mech_filters_SS10_v7.npz   $COMMON $V7 --dT-prior 0 150
  run inv_v7_filterB  --mech mech_filters_SS10_B_v7.npz $COMMON $V7
  run inv_v7_nomech                                     $COMMON $V7
  run inv_v7_nogeo    --mech mech_filters_SS10_v7.npz   $COMMON --no-geometry
  run inv_v7_sharp    --mech mech_filters_SS10_v7.npz   $COMMON $V7 --exclude w40
  run inv_v7_grad     --mech mech_filters_SS10_v7.npz   $COMMON $V7 --exclude _w0
  run inv_v7_no660    --mech mech_filters_SS10_v7.npz   $COMMON $V7 --window -195 -110      # which part of the waveform carries the result
  run inv_v7_no410    --mech mech_filters_SS10_v7.npz   $COMMON $V7 --window -300 -185
  run inv_v7_win520   --mech mech_filters_SS10_v7.npz   $COMMON $V7 --window -225 -185
  bash $CODE/recovery_mismatch.sh                                                           # recovery against a mismatched 410 or 660
  bash $CODE/recovery_narrow.sh                                                             # the reverse: 410/660 narrower than the library
  bash $CODE/round3.sh                                                                      # 660 timing, amplitude mismatch, matched grids, R(k), auto-picked records
  bash $CODE/round4.sh                                                                      # cold lower layer, Clapeyron 660, combined amplitude mismatch, chi2 by record count, block bootstrap
  bash $CODE/round5.sh                                                                      # 410/660 widths by class and temperature, 660 degraded in the cold caps only, basalt-fraction prior
  python $CODE/compare_runs.py inv_v7_SS10 \
    "optimised (profile) A,τ instead of marginal=inv_v7_profile" \
    "independent-sample likelihood, bootstrap σ=inv_v7_pre_SS10" \
    "independent samples, σ × 1.5=inv_v7_nf15" \
    "independent samples, σ × 2=inv_v7_nf20" \
    "prior N(0, 150 K) on T₂ − T₁=inv_v7_prior" \
    "mechanism filter from reference model B=inv_v7_filterB" \
    "no mechanism filter=inv_v7_nomech" \
    "single 124° trace, no cap geometry (independent-sample likelihood)=inv_v7_nogeo" \
    "sharp interfaces only=inv_v7_sharp" \
    "gradational interfaces only=inv_v7_grad" \
    "fit window without the 660 (−195 to −110 s)=inv_v7_no660" \
    "fit window without the 410 (−300 to −185 s)=inv_v7_no410" \
    "fit window 520 region only (−225 to −185 s)=inv_v7_win520" \
    "recovery, uniform model, 660 broadened 3 s=recmm_660b3" \
    "recovery, uniform model, 410 broadened 3 s=recmm_410b3" \
    "recovery, uniform model, both broadened 1.5 s (observed)=recmm_both15" \
    "recovery, uniform model, 660 broadened 6 s=recmm_660b6" \
    "recovery, uniform model, 410 broadened 6 s=recmm_410b6" \
    "recovery, uniform model, 410 narrowed 7 %=recmm_410n07" \
    "recovery, uniform model, 410 narrowed 20 %=recmm_410n20" \
    "recovery, uniform model, 660 narrowed 7 %=recmm_660n07" \
    "matched grids across classes=inv_v7_matched" \
    "library 660 shifted +1 s=inv_v7_d660p1" \
    "library 660 shifted -1 s=inv_v7_d660m1" \
    "library 660 shifted +3.5 s=inv_v7_d660p" \
    "library 660 shifted -3.5 s=inv_v7_d660m" \
    "recovery, uniform model, 410 amplitude x 0.8=recmm_410a08" \
    "recovery, uniform model, 410 amplitude x 1.25=recmm_410a125" \
    "recovery, uniform model, 660 amplitude x 0.8=recmm_660a08" \
    "recovery, uniform model, 660 amplitude x 1.25=recmm_660a125" \
    "lower layer and uniform models down to 1300 K=inv_v7_t2min1300" \
    "660 moved with lower-layer temperature (-0.075 km/K)=inv_v7_clap" \
    "both of the above=inv_v7_clap_t1300" \
    "recovery, uniform model, 410 x 1.2 and 660 x 0.8=recmm_410up660dn" \
    "recovery, uniform model, 410 x 0.8 and 660 x 1.2=recmm_410dn660up" \
    "recovery, uniform model, 660 broadened 3 s and x 0.8 in the coldest half of caps=recmm_cold660" \
    "recovery, uniform model, 660 broadened 6 s and x 0.8 in the coldest half of caps=recmm_cold660_6s" \
    --csv sensitivity_v7.csv
  python $CODE/misfit_by_window.py $(xs 10) $(inp 10) inv_v7_SS10 --lib $LIB --mech mech_filters_SS10_v7.npz    # where the M2 improvement sits
  python $CODE/pulse_widths.py     $(xs 10) $(inp 10) inv_v7_SS10 --lib $LIB --mech mech_filters_SS10_v7.npz    # observed vs synthetic pulse widths
}

pp() {
  # PP at 10°: calibration stacks from the wide-window library B (PP at 105°), for Fig. 2f and Fig. 1; no inversion
  X=$DATA/PP/10deg_caps; I=$DATA/PPinputs/10deg_caps
  python $CODE/stack_bins.py $I $X calib_pack_B_wide gcmt_all.ndk calib_stacks_v7/PP10 --phase PP --ref 105 --nproc 1 > /dev/null
  python $CODE/stack_bins.py $I $X calib_pack_B_wide gcmt_all.ndk calib_stacks_v7/PP10_fixed --phase PP --ref 105 --fixed-source > /dev/null
  python $CODE/mech_filter.py calib_stacks_v7/PP10 calib_stacks_v7/PP10_fixed mech_filters_PP10_v7.npz --window -200 -30 -150 -60; stamp "PP stacks and filter done"
}

picks() {
  # arrival picks on the new stacks (Fig. S9 and the 410/660 medians of Fig. S3); the pick-sensitivity figure S4 reruns itself from paperfigs.py
  for r in 5 7.5 10 15; do
    python $CODE/pick_xsec.py $DATA/SS/${r}deg_caps picks_SS_${r}_v7.csv --phase SS --ref 124 --radius $r
    python $CODE/pick_xsec.py $DATA/PP/${r}deg_caps picks_PP105_${r}_v7.csv --phase PP --ref 105 --radius $r --table prem_sds_P.npz
  done; stamp "picks done"
}

reinvert() {
  # rerun every inversion step (not the stacks) with the current invert_bins2.py; ~30 min in total
  for r in $RADII; do [ -f mech_filters_SS${r}_v7.npz ] && invert $r; done; recovery; variants; export_all; stamp "REINVERT DONE"
}

export_all() {
  for r in $RADII; do [ -d inv_v7_SS$r ] && python $CODE/export_fields.py inv_v7_SS$r v7_SS$r && echo "exported v7_SS$r"; done
}

case $1 in
  check) check ;;
  stacks) stacks $2 ;;
  stacksB) stacksB ;;
  noise) noise ;;
  invert) invert $2 ;;
  recovery) recovery ;;
  variants) variants ;;
  pp) pp ;;
  picks) picks ;;
  reinvert) reinvert ;;
  export) export_all ;;
  all) check; for r in $RADII; do stacks $r; done; stacksB; noise; for r in $RADII; do invert $r; done; recovery; variants; pp; picks; export_all; stamp "ALL DONE" ;;
  *) echo "usage: bash run_v7.sh check|stacks <r>|stacksB|noise|invert <r>|recovery|variants|pp|picks|export|reinvert|all"; exit 1 ;;
esac
