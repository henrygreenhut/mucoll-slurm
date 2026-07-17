# BIB mother-reuse study

This repository tests whether repeated use of a finite beam-induced-background
(BIB) library creates learnable or reconstruction-level artifacts. It contains
three related pipelines and no stored datasets or analysis results.

Run the workflows on Perlmutter from `~/mucoll/mucoll-slurm`. The
`mucoll-benchmarks` checkout must be its sibling; `config.sh` is the only file
containing site and software paths.

## Research design

The basic comparison holds the nominal decay statistics of a pseudo-crossing
fixed while changing its number of unique sources:

- **U (unique):** combine independent `norm1` sources.
- **R (reused):** combine fewer sources whose particles were coherently
  rotated and cloned.
- **Null:** construct both labels independently from the same `norm1` pool.

Source cycles are separated between training, validation, and testing. A
source used for training therefore never appears in evaluation. Sources may
reappear across pseudo-events within one split; this represents repeated
construction from a finite held-out library, but correlates event scores.
Final GEN uncertainty is consequently obtained with paired source-cycle
resampling, not an event-level binomial error.

The fixed analysis choices live as named constants near the top of each
trainer. The important ones are:

- GEN source split: 50% train, 25% validation, 25% test.
- historical clone factor: 42.
- GEN PFN inputs: `log10(pT)`, `theta`, `cos(phi)`, `sin(phi)`, and five
  particle-ID indicators.
- GEN PFN: local Keras implementation with per-particle MLP
  `(200,200,256)`, masked sum, and event MLP `(200,200,200)`; this preserves
  the implementation used for the existing GEN results.
- RECO PFN: `energyflow.archs.PFN` with `Phi_sizes=(64,64,64)` and
  `F_sizes=(64,64,64)`, matching the original RECO training run.
- every classifier has a matched null test.
- cycle 6291 is excluded when building the mother bank and RECO pools because
  its SIM file is invalid.

## File map

| Purpose | Files |
| --- | --- |
| Shared GEN representation and PFN | `libtest_common.py` |
| Existing norm1 versus norm42 GEN study | `gen_libtest_make_store.py`, `pfn_libtest_train.py`, `pfn_libtest_evaluate.py` |
| On-the-fly variable reuse | `gen_mother_make_store.py`, `variable_reuse_common.py`, `pfn_variable_reuse_train.py` |
| N=420 reconstruction study | `reco_libtest_prepare_pools.py`, `submit_reco_libtest_packed.py`, `run_reco_libtest_task.sh`, `make_reco_libtest_stores.py`, `train_reco_libtest_pfn.py` |
| Simulation chain | `chains/run_chain_pgun.sh` |
| Batch entry points | `submit_*.slurm` |
| Result plotting | `pfn_libtest_compare.py` |
| Software invariants | `test_libtest_training.py`, `test_variable_reuse_common.py` |

Generated logs, plots, stores, and results are ignored. HDF5 stores and EDM4hep
outputs belong under `$PSCRATCH/mucoll/libtest`; compact model results are
written to `pfn_results`, `variable_k_results`, or `reco_pfn_results`.

## 1. Existing GEN libraries: unique versus 42x reuse

Build compact stores once:

```bash
sbatch submit_libtest_convert.slurm
```

The following is the N=420 production comparison. `auto` scales the summed
latent vector by the median particle multiplicity; use `none` for the raw-sum
comparison. A new label is required whenever scientific settings change.

```bash
sbatch --export=ALL,LABEL=gen_n420_scaled_v2,TRAIN_ARGS="--n-files 420 --units-per-epoch 500 --val-units 300 --overlap-test-units 300 --batch-size 1 --patience 20 --min-epochs 80 --latent-scale auto" submit_libtest_train.slurm

sbatch --export=ALL,LABEL=gen_n420_null_v2,TRAIN_ARGS="--n-files 420 --units-per-epoch 500 --val-units 300 --overlap-test-units 300 --batch-size 1 --patience 20 --min-epochs 80 --latent-scale auto --null-test" submit_libtest_train.slurm
```

Each shared-QOS job is one resumable 25-minute training window. If `state.json`
does not say `done: true`, submit the identical command again. The model,
optimizer, epoch, and best-validation state are restored from the checkpoint.

After the main model finishes, estimate its held-out AUC with paired
source-cycle resampling:

```bash
sbatch --export=ALL,LABEL=gen_n420_scaled_eval_v2,TRAIN_ARGS="--source-label gen_n420_scaled_v2 --point-units 500 --bootstrap-reps 200 --bootstrap-units 100 --batch-size 1" submit_libtest_evaluate.slurm
```

That evaluation is also resumable. Its primary products are
`point_summary.json` and `paired_cycle_bootstrap.csv`.

## 2. Variable reuse generated on the fly

Build one compact bank from the split-by-mother, unrotated GEN library:

```bash
sbatch submit_variable_reuse_convert.slurm
```

No rotated library is materialized. For reuse factor `k`, a pseudo-event with
`M` mother-equivalents samples `M/k` distinct mothers, draws `k` independent
angles for each, and concatenates their particles. A rotation by angle
`alpha` applies the same two-dimensional rotation to `(px,py)` and `(vx,vy)`;
`pz`, energy, time, `vz`, and PDG ID are unchanged.

Example: compare no reuse with 10x reuse at the N=420 event scale:

```bash
sbatch --export=ALL,LABEL=variable_k1_k10,TRAIN_ARGS="--reuse-k 1 10 --mother-equivalents 29400 --units-per-epoch 20 --val-units 10 --test-units 30 --rotation-policy all-random --min-epochs 40" submit_variable_reuse_train.slurm

sbatch --export=ALL,LABEL=variable_k1_k10_null,TRAIN_ARGS="--reuse-k 1 10 --mother-equivalents 29400 --units-per-epoch 20 --val-units 10 --test-units 30 --rotation-policy all-random --min-epochs 40 --null-test" submit_variable_reuse_train.slurm
```

`all-random` rotates every copy, including `k=1`, so the label cannot be read
from the mere presence of a rotation. `baseline-unrotated` is retained only to
reproduce the historical unrotated `k=1` construction. The multiclass default
is `k = 1,2,3,6,10,14,21,42`.

## 3. N=420 reconstructed-PFO study

This fixed study overlays, per beam polarity and reconstructed event:

- U: 420 norm1 SIM files;
- R: 10 norm42 SIM files, representing `10 x 42` source copies;
- null: U versus an independently digitized `null_b` sample from norm1.

All classes use the same invisible 100 GeV PDG-14 particle gun. The signal is
there only to drive the simulation chain; reconstructed PFOs come from BIB.
The dataset contains 2,000 train, 400 validation, and two 400-event test
cohorts per class.

Prepare immutable source pools:

```bash
source config.sh
python3 reco_libtest_prepare_pools.py \
  --norm1-sim "$DATA_GROUP_DIR/bib-v3p0-fmt2-norm1/SIM" \
  --norm42-sim "$DATA_GROUP_DIR/bib-v3p0-fmt2-norm42-RandomRot/SIM" \
  --outdir "$PSCRATCH/mucoll/libtest/bib_pools_v2" \
  --exclude-cycle 6291 --force
```

Produce any missing GEN->SIM->DIGI->RECO chunks. The submitter packs up to 64
independent four-core chains per CPU node into one allocation and skips valid
existing outputs:

```bash
python3 submit_reco_libtest_packed.py
```

Rerun that command after a timeout until it reports that nothing remains.
Then build the twelve PFO stores and fill a four-GPU node with two model seeds
for the main comparison and two for the null:

```bash
sbatch submit_reco_libtest_stores.slurm
sbatch submit_reco_libtest_train.slurm
```

Each PFO contributes `log(pT)`, `eta`, `sin(phi)`, `cos(phi)`, `log(E)`,
charge, and charged/photon/neutral indicators. The RECO PFN uses a raw sum: at
roughly O(10) PFOs per event, the large GEN-level sum-saturation issue is not
present. The trainer imports the standard `energyflow.archs.PFN`; do not replace
it with the local GEN builder under an existing result label. `summary.json`
records the EnergyFlow and TensorFlow versions and reports test A, test B, and
combined AUC. Test A and B have disjoint source pools, although events within
either cohort may share sources.

The `_v2` pool, RECO, store, and result names intentionally prevent corrected
shared-pool null data from being mixed with earlier alternating-cycle null
outputs.

## Checks

The retained tests cover only failure modes that would change a result:
64-bit streaming normalization at very large particle counts, rejection of
corrupt cached normalization, the minimum-epoch stopping floor, coherent
momentum/vertex rotations, fixed event statistics, and source-disjoint splits.

```bash
python3 -m unittest -v
```

Do not commit generated results. Before committing code, inspect `git diff` and
run the tests above.
