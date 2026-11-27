# mtz-layering — compositional layering of the mantle transition zone from SS-precursor waveforms

Code for *A basalt-enriched transition zone below 520 km from SS precursors* (Costa de Lima, Waszek, Tkalčić, Tauzin & Ballmer). It builds a library of synthetic SS-precursor stacks from thermodynamic velocity profiles, reproduces the stacking geometry and source mechanisms of every observed cap, and compares three classes of transition-zone structure (uniform, thermally layered, thermochemically layered) by Bayesian evidence.

The stacked cross-sections, record lists, model library and per-cap results are deposited at Zenodo (DOI in the paper); this repository holds the code only.

## Requirements

`environment.yml` lists the packages.

Python ≥ 3.10 with numpy, scipy, pandas, matplotlib, obspy, cartopy (0.22), pygmt (0.10; GMT 6.4) and gplately (for the paleo-subduction lines). Two conda environments were used: `mtz520` (everything except the maps) and `pygmt` (maps). Yspec (Al-Attar & Woodhouse, 2008) is required only to recompute the synthetic seismograms; the processed library (`lib_shift_*.npz`) is deposited with the data.

## Layout

Clone the repository as `code/` inside the working directory that holds the data (called `mtzsyn/` below) and run every command from `mtzsyn/`; the shell scripts find the other scripts next to themselves. The result tables of the paper are in `tables/`.

```
mtzsyn/
  code/                                                       this repository
  tables/ (in the repository)                                 v7_SS<r>_results.csv, v7_SS<r>_fields.csv, sensitivity_v7.csv, picks_*_v7.csv, figS10_subd*.csv
  gallery/ (in the repository)                                fits_SS<r>.pdf: data and best-fitting model of each class for every cap at every radius (plot_all_fits.py)
  crosssections_2024_SNR/SS/<r>deg_caps/<lat>_<lon>.txt      observed stacks: t, slowness, amplitude, bootstrap σ
  crosssections_2024_SNR/SSinputs/<r>deg_caps/<lat>_<lon>.txt  records of each cap: id, event, station, distance, bounce point, SNR weight
  crosssections_2024_SNR/PP/, PPinputs/                      the same for PP
  models/MM<f>/<T>.tvel                                       Perple_X profiles (Tauzin et al. 2022), f = 0.0 … 1.0, T = 1000 … 2400 K
  models50/                                                   50 K profiles from interp_profiles.py
  runs_uniform/, runs_layered/, runs_50K/                     Yspec output, one directory per model
  lib_shift_0..3.npz, lib_shift50_0.npz                       processed library: SdS/SS traces on the slowness curve, 100–170°
  calib_pack/, calib_pack_B_wide/                             elementary-source calibration libraries (6 moment tensors × 30 depths)
  calib_stacks_v7/SS<r>, SS<r>_fixed, PP10, PP10_fixed        per-cap calibration stacks (real mechanisms / fixed source)
  mech_filters_SS<r>_v7.npz                                   per-cap source-mechanism filters
  noise_R_SS10_v7.npz                                         residual autocorrelation (noise covariance)
  inv_v7_SS<r>/<cap>.json                                     inversion results, one file per cap
  rec_v7_*/, inv_v7_<variant>/                                recovery tests and sensitivity variants
  v7_SS<r>_results.csv, v7_SS<r>_fields.csv                   exported tables (export_fields.py)
  figs/                                                       figures
```

## Pipeline

All commands run from `mtzsyn/`. Steps 1–3 are needed only to rebuild the model library; with the deposited `lib_shift_*.npz` start at step 4.

### 1. Model library (`models/` → Yspec input)

```
python code/interp_profiles.py models models50                       # 50 K profiles by phase-aware interpolation (Supplementary Fig. 5)
python code/splice_layers.py models runs_layered --zc 522.97 --w 0 40  # two-layer profiles joined at 523 km, sharp and 40-km gradational
python code/tvel2yspec.py <profile>.tvel <model>.yspec                # Yspec model file (PREM attenuation, 1 km layers)
python code/make_yspec_in.py <model>  && python code/run_batch.py <list>   # Yspec runs: Mrφ source at 10 km, receivers 100–170° every 1°
python code/yspec_post.py runs_<set>                                  # transverse component, 15–50 s, aligned and normalised on SS
python code/pack_models.py runs_<set> lib_shift_<k>.npz               # traces shifted onto the SdS–SS slowness curve
python code/build_catalogue.py runs_uniform catalogue/catalogue.csv   # apparent 410/520/660 depths of the single-trace synthetics (Supplementary Fig. 3)
python code/prem_sds_table.py                                         # TauP SdS–SS times for the depth axis (prem_sds_S.npz, prem_sds_P.npz)
```

### 2. Source-mechanism calibration library

```
python code/run_calib_library.py MM0.2_1600 calib_raw   # 6 unit moment tensors × depths 2.5–75 km, receivers 90–180° every 0.5°
python code/pack_calib.py calib_raw calib_pack           # packed per depth
```

Library B (model f = 0.2 / 0.6, 1700 K) is built the same way into `calib_pack_B_wide` for the filter-independence test.

### 3. Cap-specific stacks, filters and noise model

```
bash code/run_v7.sh check          # verify the data layout, count caps and records
bash code/run_v7.sh stacks 10      # real-mechanism and fixed-source stacks of every 10° cap; per-cap FIR filter (stack_bins.py, mech_filter.py)
bash code/run_v7.sh stacksB        # library-B stacks at 10° (Supplementary Fig. 6b)
bash code/run_v7.sh noise          # first-pass inversion and residual autocorrelation (noise_corr.py)
```

`stacks <r>` is run for every radius (5, 7.5, 10, 12.5, 15, 17.5, 20).

### 4. Inversion

```
bash code/run_v7.sh invert 10      # invert_bins2.py: evidence of M0, M1, M2 for every cap, correlated noise, marginal A and τ
bash code/run_v7.sh recovery       # recovery tests at a fixed noise level (Supplementary Fig. 6a)
bash code/recovery_obsnoise.sh     # recovery tests at each cap's own noise level, with the comparison by record count
bash code/run_v7.sh variants       # sensitivity variants, fit-window tests, mismatch recovery tests and sensitivity_v7.csv (Supplementary Table 1);
                                   #   also misfit_by_window.py (where the M2 improvement sits), pulse_widths.py (observed vs synthetic pulse widths),
                                   #   round3.sh, round4.sh and round5.sh (660 timing, cold lower layer, amplitude mismatch, matched grids, chi2 by record count, block bootstrap, pulse widths by class, 660 degraded in cold caps only)
bash code/run_v7.sh pp             # PP calibration stacks and filter (Fig. 2f)
bash code/run_v7.sh picks          # arrival picks (Supplementary Figs. 4, 9)
bash code/run_v7.sh export         # v7_SS<r>_results.csv and v7_SS<r>_fields.csv
```

`bash code/run_v7.sh all` runs steps 3–4 for every radius in order (several hours; resumable).

The inversion script `invert_bins2.py` takes, for one radius, the observed stacks, the record lists, the library files (`--lib`), the mechanism filters (`--mech`), the noise covariance (`--corr-noise`) and writes one JSON per cap with the log-evidence of each class, the Bayes factors, posterior means and the best model. Options used in the paper: `--noise-fit` (per-cap σ inflation), `--marginal` (marginalised A and τ), `--synthetic <model> --amp 0.6 --noise 0.004` (recovery tests), `--dT-prior`, `--no-geometry`, `--exclude`, `--window` (sensitivity variants), `--noise-obs`, `--syn-660`, `--syn-410`, `--syn-410-narrow`, `--syn-660-narrow`, `--syn-410-amp`, `--syn-660-amp` (recovery tests at the observed noise level and with a mismatched discontinuity), `--t2min` (lower-layer temperature floor), `--clapeyron` (660 depth moving with the lower-layer temperature), `--fmax` (basalt-fraction range) and `--syn-caps` (perturbation restricted to a list of caps).

### 5. Figures

```
python code/fig1.py ... code/fig6.py          # main figures (fig1, fig4, fig5 need cartopy); fig1.py --suppl for Supplementary Fig. 1; fig4.py --m1 for Supplementary Fig. 12
python code/figS.py all                       # Supplementary Figs. 2-4
python code/suppfigs.py all                   # Supplementary Figs. 5-9 and the full fit gallery (S11)
python code/plot_all_fits.py inv_v7_SS10 crosssections_2024_SNR/SS/10deg_caps crosssections_2024_SNR/SSinputs/10deg_caps figs/figS11.pdf \
       --lib lib_shift_0.npz lib_shift_1.npz lib_shift_2.npz lib_shift_3.npz lib_shift50_0.npz --mech mech_filters_SS10_v7.npz --subset 10
python code/subduction_history.py v7_SS10_fields.csv figs/figS10_subd.csv figs/figS10_subd   # Supplementary Fig. 10 (pygmt env)
python code/make_trenches.py                                                                 # present-day and 20–200 Ma trench lines (gplately)
```

`plot_all_fits.py` without `--subset` writes the fits of every cap (10 per page); run it for each `inv_v7_SS<r>` for the deposited gallery. `maps.py` draws a four-panel diagnostic map of any results CSV. In the code a "bin" is what the paper calls a cap.

## Output format

`inv_v7_SS<r>/<lat>_<lon>.json`: `classes.{M0,M1,M2}.logZ` (log-evidence), `lnB10`, `lnB21`, `chi2_red_best`, per-class posterior means (`mean_T1`, `mean_T2`, `mean_f1`, `mean_f2`, `mean_dT`, `mean_df`), best model and its amplitude factor and shift.

`v7_SS<r>_results.csv`: one row per cap — class (M2 if ln B₂₁ > 3, else M1 if ln B₁₀ > 3, else M0), Bayes factors, χ², noise inflation, number of records, posterior means per class, best model, A, τ.

## Citation

Costa de Lima, T., Waszek, L., Tkalčić, H., Tauzin, B. & Ballmer, M. D. A basalt-enriched transition zone below 520 km from SS precursors. (in review).
