# GEN experiment plot index

This is the presentation-oriented index for the experiments leading to the
final N=420 result. Newly generated PDFs use
`pfn_libtest_plot_overlay.make_plot`, the same plotting function as
`plots/training_overlays/gen_n420_recipe/gen_n420_scaled_lr1e-4.pdf`.

Historical experiments that did not record validation loss retain their
existing training-loss and validation-AUC plots. A job that crashed before
writing an epoch is listed explicitly as having no curve.

## 1. Initial scale studies

| Experiment | Plot(s) | Result / role |
|---|---|---|
| N=42 scaled + null | [loss](plots/gen_n42_scaled_loss.pdf), [AUC](plots/gen_n42_scaled_auc.pdf) | overlap AUC 1.000; established strong reuse signal |
| N=42 raw + null | [loss](plots/gen_n42_raw_loss.pdf), [AUC](plots/gen_n42_raw_auc.pdf) | test AUC 0.996; showed raw pooling also worked |
| N=126 scaled + null | [loss](plots/gen_n126_scaled_loss.pdf), [AUC](plots/gen_n126_scaled_auc.pdf) | overlap AUC 0.947; signal persisted at larger N |
| N=714 scaled pilot | [production-style train-only plot](plots/training_overlays/gen_historical/n714_scaled_pilot.pdf) | AUC 0.433; too few train/validation events |
| N=714 original null | [production-style train-only plot](plots/training_overlays/gen_historical/n714_original_null.pdf) | invalid null construction; AUC 0.638 |
| N=714 corrected null | [production-style train-only plot](plots/training_overlays/gen_historical/n714_corrected_null.pdf) | corrected shared-pool null AUC 0.500 |
| N=210 scaled + null | [loss](plots/gen_n210_scaled_loss.pdf), [AUC](plots/gen_n210_scaled_auc.pdf) | overlap AUC 0.964 |
| N=210 raw + null | [loss](plots/gen_n210_raw_loss.pdf), [AUC](plots/gen_n210_raw_auc.pdf) | original test AUC 0.993 |
| First N=420 scaled + null | [loss](plots/gen_n420_scaled_old_loss.pdf), [AUC](plots/gen_n420_scaled_old_auc.pdf) | scaled model collapsed to AUC 0.501 |
| First N=420 raw + null | [loss](plots/gen_n420_raw_loss.pdf), [AUC](plots/gen_n420_raw_auc.pdf) | overlap AUC 0.943 proved information remained |
| 80-epoch N=420 scaled confirmation | [main](plots/training_overlays/gen_n420_diagnostics/gen_n420_scaled_collapse_repeat.pdf), [null](plots/training_overlays/gen_n420_diagnostics/gen_n420_scaled_collapse_null.pdf) | reproduced scaled collapse at AUC 0.500 |

The oldest histories contain training loss and validation AUC but not
validation loss. Their existing plots are therefore more complete and more
honest than trying to mimic a train/validation-loss overlay.

## 2. GPU capacity and integer-overflow diagnosis

The standalone `pfn_capacity_benchmark.py` sweep measured step time and peak
GPU memory but did not save `history.csv`; its original Slurm stdout is not in
this laptop checkout. It therefore has no honest production loss curve.

Its experimentally established conclusions were:

- approximately 3.8 kB GPU memory per padded particle slot for the original
  wide PFN;
- N=420 batch 4 fit in GPU memory;
- N=420 batch 8 aborted even on an 80 GB GPU;
- the failure boundary was
  `batch * padded_particles * widest_Phi >= 2**31`, not GPU OOM.

The training runs that exercised the resulting boundaries are:

| Experiment | Production-style plot | Result / role |
|---|---|---|
| Full-width raw, batch 4 | [plot](plots/training_overlays/gen_n420_diagnostics/scan_fullwidth_raw_batch4.pdf) | only two epochs saved; unsafe multiplicity-tail headroom |
| Full-width scaled, batch 4 | [plot](plots/training_overlays/gen_n420_diagnostics/scan_fullwidth_scaled_batch4.pdf) | only two epochs saved |
| Full-width null, batch 4 | [plot](plots/training_overlays/gen_n420_diagnostics/scan_fullwidth_null_batch4.pdf) | only three epochs saved |
| Half-width raw, batch 8 | [plot](plots/training_overlays/gen_n420_diagnostics/scan_halfwidth_raw_batch8.pdf) | brief screen reached val AUC 0.657 |

## 3. Explicit-JIT overflow workaround

| Experiment | Plot | Result / role |
|---|---|---|
| JIT raw-small, batch 8 | [only completed epoch](plots/training_overlays/gen_n420_diagnostics/jit_raw_batch8_only_completed_epoch.pdf) | one epoch, val AUC 0.500 |
| JIT raw-large, scaled-large, null-large | no curve: none wrote an epoch | three of four ranks failed to complete an epoch in about 29 minutes; JIT was at least four times slower |

## 4. Pure Phi-width rerun

| Experiment | Production-style plot | Result / role |
|---|---|---|
| Raw, Phi=(100,100,128), batch 4 | [plot](plots/training_overlays/gen_n420_diagnostics/halfphi_raw_batch4.pdf) | four-epoch val AUC 0.714 |
| Scaled, Phi=(100,100,128), batch 4 | [plot](plots/training_overlays/gen_n420_diagnostics/halfphi_scaled_batch4.pdf) | four-epoch val AUC 0.879 |
| Null, Phi=(100,100,128), batch 4 | [plot](plots/training_overlays/gen_n420_diagnostics/halfphi_null_batch4.pdf) | val AUC 0.500 |
| Raw, Phi=(50,50,64), batch 8 | [plot](plots/training_overlays/gen_n420_diagnostics/quarterphi_raw_batch8.pdf) | val AUC 0.503; too little learning capacity |

This selected Phi=(100,100,128), F=(200,200,200), batch 4.

## 5. Local-versus-EnergyFlow and seed checks

### Local raw PFN

- [seed 1](plots/training_overlays/gen_energyflow_check/n42_local_raw_seed1.pdf)
- [seed 2](plots/training_overlays/gen_energyflow_check/n42_local_raw_seed2.pdf)
- [seed 3](plots/training_overlays/gen_energyflow_check/n42_local_raw_seed3.pdf)

### Official EnergyFlow raw PFN

- [seed 1](plots/training_overlays/gen_energyflow_check/n42_energyflow_raw_seed1.pdf)
- [seed 2](plots/training_overlays/gen_energyflow_check/n42_energyflow_raw_seed2.pdf)
- [seed 3](plots/training_overlays/gen_energyflow_check/n42_energyflow_raw_seed3.pdf)

### Official EnergyFlow scaled PFN

- [seed 1](plots/training_overlays/gen_energyflow_check/n42_energyflow_scaled_seed1.pdf)
- [seed 2](plots/training_overlays/gen_energyflow_check/n42_energyflow_scaled_seed2.pdf)
- [seed 3](plots/training_overlays/gen_energyflow_check/n42_energyflow_scaled_seed3.pdf)

The official scaled runs reached best validation AUC 0.997--0.999 across all
three seeds. This established package provenance and better seed stability.

## 6. N=420 fixed-LR seed screen

Seed 1 reached natural completion and is in the existing production set:

- [raw seed 1](plots/training_overlays/gen_n420_development/gen_n420_raw_fixed_lr.pdf)
- [scaled seed 1](plots/training_overlays/gen_n420_development/gen_n420_scaled_fixed_lr.pdf)

The wall-time-limited additional initializations are:

- [raw seed 2](plots/training_overlays/gen_n420_seed_check/raw_fixed_lr_seed2.pdf)
- [raw seed 3](plots/training_overlays/gen_n420_seed_check/raw_fixed_lr_seed3.pdf)
- [scaled seed 2](plots/training_overlays/gen_n420_seed_check/scaled_fixed_lr_seed2.pdf)
- [scaled seed 3](plots/training_overlays/gen_n420_seed_check/scaled_fixed_lr_seed3.pdf)

They reinforced that fixed LR \(10^{-3}\) was initialization-sensitive.

## 7. Warmup and clipping experiments

### N=42 short factorial diagnostic

- [neither](plots/training_overlays/gen_optimizer_diagnostics/n42_raw_none.pdf)
- [warmup only](plots/training_overlays/gen_optimizer_diagnostics/n42_raw_warmup.pdf)
- [clipnorm 1 only](plots/training_overlays/gen_optimizer_diagnostics/n42_raw_clip.pdf)
- [warmup + clipnorm 1](plots/training_overlays/gen_optimizer_diagnostics/n42_raw_both.pdf)

### N=420 short diagnostic

- [raw clipnorm 1](plots/training_overlays/gen_n420_diagnostics/raw_clipnorm1.pdf)
- [raw clipnorm 1000](plots/training_overlays/gen_n420_diagnostics/raw_clipnorm1000.pdf)
- [raw warmup, no clipping](plots/training_overlays/gen_n420_diagnostics/raw_warmup_no_clip.pdf)
- [scaled warmup, no clipping](plots/training_overlays/gen_n420_diagnostics/scaled_warmup_no_clip.pdf)
- [expanded raw clipnorm 1](plots/training_overlays/gen_n420_diagnostics/expanded_raw_clipnorm1.pdf)
- [raw warmup + clipnorm 1000](plots/training_overlays/gen_optimizer_diagnostics/n420_raw_warmup_clipnorm1000.pdf)

The raw clipnorm-1 models converged to chance. Clipnorm 1000 was
inconclusive; no clipping was selected.

The `raw + warmup + clipnorm 1`, `N=420 both`, expanded seed-1, and expanded
seed-3/no-clip attempts crashed or hung before writing an epoch, so they have
no curves.

## 8. Longer pooling and warmup comparison

- [raw, fixed LR](plots/training_overlays/gen_n420_development/gen_n420_raw_fixed_lr.pdf)
- [raw + warmup](plots/training_overlays/gen_n420_development/gen_n420_raw_warmup.pdf)
- [scaled, fixed LR](plots/training_overlays/gen_n420_development/gen_n420_scaled_fixed_lr.pdf)
- [scaled + warmup](plots/training_overlays/gen_n420_development/gen_n420_scaled_warmup.pdf)

Warmup improved the best test AUC from 0.563 to 0.841 for raw and from 0.885
to 0.950 for scaled, but the fixed \(10^{-3}\) post-warmup LR still caused
late collapses.

## 9. Final recipe smoke checks

- [paper features](plots/training_overlays/gen_n420_recipe_smokes/paper_features.pdf)
- [expanded features](plots/training_overlays/gen_n420_recipe_smokes/expanded_features.pdf)

These two-epoch, tiny-statistics runs tested execution only. They are not
evidence that expanded features were more accurate.

The separate timed first-step diagnostic returned at batch sizes 1, 2, and
4 after model and optimizer JIT were explicitly disabled. It printed timings
to Slurm stdout but did not create a training history, so it has no loss
curve. Batch 4 was selected for balanced two-per-class updates.

## 10. Final N=420 recipe grid

- [scaled, LR 1e-4](plots/training_overlays/gen_n420_recipe/gen_n420_scaled_lr1e-4.pdf)
- [scaled, LR 3e-4](plots/training_overlays/gen_n420_recipe/gen_n420_scaled_lr3e-4.pdf)
- [raw, LR 1e-4](plots/training_overlays/gen_n420_recipe/gen_n420_raw_lr1e-4.pdf)
- [raw, LR 3e-4, incomplete](plots/training_overlays/gen_n420_recipe/gen_n420_raw_lr3e-4_incomplete.pdf)

The selected scaled \(10^{-4}\) run gave test AUC 0.98447.

## 11. Final controls

- [exact-configuration repeat](plots/training_overlays/gen_n420_recipe/gen_n420_scaled_lr1e-4_repeat.pdf)
- [matched null](plots/training_overlays/gen_n420_recipe/gen_n420_scaled_lr1e-4_null.pdf)

The repeat gave AUC 0.98519; the null gave 0.48076.
