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
- Current GEN PFN inputs: `log10(pT)`, `theta`, `cos(phi)`, `sin(phi)`,
  `log10(E)`, compressed production time, vertex `z`, vertex radius, and
  five particle-ID indicators.
- Current N=420 GEN PFN: official EnergyFlow PFN/weighted-EFN implementation,
  with per-particle MLP `(100,100,128)`, event MLP `(200,200,200)`, and
  balanced batches of four (two U and two R events).
- RECO PFN: `energyflow.archs.PFN` with `Phi_sizes=(64,64,64)` and
  `F_sizes=(64,64,64)`, matching the original RECO training run.
- every classifier has a matched null test.
- cycle 6291 is excluded when building the mother bank and RECO pools because
  its SIM file is invalid.

## File map

| Purpose | Files |
| --- | --- |
| Shared GEN representation and PFN model | `libtest_common.py` |
| Shared GEN fitting/checkpoint/validation engine | `pfn_training_engine.py` |
| Existing norm1 versus norm42 GEN study | `gen_libtest_make_store.py`, `pfn_libtest_train.py`, `pfn_libtest_evaluate.py` |
| On-the-fly variable reuse | `gen_mother_make_store.py`, `variable_reuse_common.py`, `pfn_variable_reuse_train.py` |
| N=420 reconstruction study | `reco_libtest_prepare_pools.py`, `submit_reco_libtest_packed.py`, `run_reco_libtest_task.sh`, `make_reco_libtest_stores.py`, `train_reco_libtest_pfn.py` |
| Simulation chain | `chains/run_chain_pgun.sh` |
| Batch entry points | `submit_*.slurm` |
| Result plotting | `pfn_libtest_compare.py`, `plot_gen_trials.py` |
| Software invariants | `test_libtest_training.py`, `test_variable_reuse_common.py`, `test_reco_libtest.py` |

Generated logs, plots, stores, and results are ignored. HDF5 stores and EDM4hep
outputs belong under `$PSCRATCH/mucoll/libtest`; compact model results are
written to `pfn_results`, `variable_k_results`, or `reco_pfn_results`.

## 1. Existing GEN libraries: unique versus 42x reuse

Build compact stores once:

```bash
sbatch submit_libtest_convert.slurm
```

The current OSCAR recipe uses fixed shuffled source splits, fixed validation
events, batch size 4, one epoch of warmup, cosine learning-rate decay, and
separate data/model seeds. `scaled` scales the summed latent vector by the
median particle multiplicity; `raw` is the raw-sum comparison. A new label is
required whenever scientific settings change.

```bash
sbatch submit_oscar_n420_recipe.slurm scaled 1e-4 1
sbatch submit_oscar_n420_recipe.slurm scaled 1e-4 1 null
sbatch submit_oscar_n420_recipe.slurm scaled 3e-4 1
sbatch submit_oscar_n420_recipe.slurm raw 1e-4 1
sbatch submit_oscar_n420_recipe.slurm raw 3e-4 1
```

Each command is resumable under its immutable label. The full model, Adam
state, learning-rate position, epoch, and validation-selection state are
restored strictly. Production training reports the AUC on overlapping
held-out events. Paired source-cycle uncertainty is run separately only when
needed.

To repeat a completed configuration from scratch with the same data and model
seeds, give it a new immutable result label with `RUN_TAG`:

```bash
sbatch --export=ALL,RUN_TAG=repro1 \
  submit_oscar_n420_recipe.slurm scaled 1e-4 1 main
```

## 2. Variable reuse generated on the fly

Build one compact bank from the split-by-mother, unrotated GEN library:

```bash
sbatch submit_variable_reuse_convert.slurm
```

On OSCAR, the equivalent bank is built directly from the resident FLUKA
format-2 files:

```bash
sbatch submit_oscar_mother_store.slurm
```

Format 2 records the beam-muon decay position `(x_mu, y_mu, z_mu)`. Exact
equality of that triple defines the particles belonging to one mother decay;
the original GEN task list preserves the cycle-number mapping.

No rotated library is materialized. For reuse factor `k`, a pseudo-event with
`M` mother-equivalents samples `M/k` distinct mothers, draws `k` independent
angles for each, and concatenates their particles. A rotation by angle
`alpha` applies the same two-dimensional rotation to `(px,py)` and `(vx,vy)`;
`pz`, energy, time, `vz`, and PDG ID are unchanged.

The current binary studies compare adjacent reuse regimes at the N=420 event
scale. Every class contains 29,400 mother-equivalents: k=1 samples 29,400
distinct mothers, k=5 samples 5,880, k=10 samples 2,940, and k=42 samples 700.
The selected recipe is the official EnergyFlow scaled-sum network, expanded
GEN features, balanced batches of four, peak learning rate `1e-4`, one-epoch
warmup, and cosine decay. This entry point supplies only mother-level event
construction; model fitting, validation-loss selection, checkpointing,
early stopping, and history are executed by the same `pfn_training_engine.py`
used by the existing norm1-versus-norm42 trainer.

```bash
sbatch submit_oscar_variable_reuse.slurm 1v5 main
sbatch submit_oscar_variable_reuse.slurm 1v5 null
sbatch submit_oscar_variable_reuse.slurm 1v7 main
sbatch submit_oscar_variable_reuse.slurm 1v7 null
sbatch submit_oscar_variable_reuse.slurm 1v21 main
sbatch submit_oscar_variable_reuse.slurm 1v21 null
sbatch submit_oscar_variable_reuse.slurm 5v10 main
sbatch submit_oscar_variable_reuse.slurm 5v10 null
sbatch submit_oscar_variable_reuse.slurm 1v10 main
sbatch submit_oscar_variable_reuse.slurm 1v10 null
sbatch submit_oscar_variable_reuse.slurm 10v42 main
sbatch submit_oscar_variable_reuse.slurm 10v42 null
```

Every new training run saves `best.weights.h5` using the declared selection
metric and independently saves `best_auc.weights.h5` at the maximum validation
AUC.

Main variable-reuse runs use four deterministic event-building workers,
prefetch one batch, and run at least 80 epochs. Their default labels end in
`_min80`. Null runs retain validation-loss early stopping.

Each null permutes labels over the same sampled units as its corresponding
main comparison, removing the association between reuse factor and target
while preserving the full input construction. Evaluations use overlapping
pseudo-events from the held-out source-cycle pool and report a point AUC;
cycle-level uncertainty is run separately if needed.

### Unified Perlmutter GEN reruns

The controlled Perlmutter study runs three comparisons through one
fixed-library training path: k=1 versus synthetic k=5, k=1 versus synthetic
k=42, and k=1 versus the production norm42 library. The synthetic stores are
built from the split-mother GEN bank at
`$PSCRATCH/mucoll/libtest/stores/gen_split_mothers_MUPLUS.h5`. That bank was
built from the Perlmutter library
`bib-v3p0-fmt2-norm1-split-mother-norot/GEN/MUPLUS`, retains 6,665 source
cycles after excluding cycle 6291, and contains 467,045 mother events. The
production comparison uses the already-converted real
`gen_norm42_MUPLUS.h5` library.

Every pseudo-event has N=420 file-equivalents. The k=1 class draws 420 native,
unrotated source cycles. Class k draws `420/k` stored source cycles whose
mothers each have k fixed azimuthal copies. All daughters of one mother-copy
share one angle. The first five rotations of every source in the synthetic
k=42 store are identical to its k=5 rotations. Whole source cycles are split
50/25/25 before pseudo-events are sampled. Because visible mothers per cycle
fluctuate, 29,400 mother-equivalents is the approximate average event size,
not an exact per-event constraint.

All three comparisons use the same EnergyFlow scaled-sum PFN, expanded GEN
features, batch size four, 500 events per class per epoch, 300 fixed
validation and test events per class, peak learning rate `1e-4`, one warmup
epoch, an 80-epoch cosine decay, validation-loss selection, and model seed 1.
Main runs have a 120-epoch cap. K=5 cannot early-stop before epoch 80; the two
k=42 comparisons use ordinary validation-loss early stopping with no epoch
floor. Each null is k-versus-k. These fresh labels intentionally do not resume
historical runs whose k=1 class was rotated, whose event construction was on
the fly, or whose optimizer schedule differed.

Production GPU jobs have a 20-hour Slurm limit and checkpoint cleanly after
19 hours. Nulls checkpoint after seven hours and should be submitted with an
eight-hour Slurm override. With three one-GPU main jobs, one null, and the
four-GPU debug smoke, the hard billing ceiling is 17.5 GPU node-hours.

After pulling the current branch on Perlmutter, build the synthetic k=5 and
k=42 stores. The existing production k=42 store is left unchanged:

```bash
sbatch submit_perlmutter_gen_k_stores.slurm
```

After that job succeeds, run one four-GPU debug smoke:

```bash
sbatch submit_perlmutter_gen_k_smoke.slurm
```

Then submit the three physical comparisons and their matched nulls as one-GPU
shared jobs:

```bash
sbatch submit_perlmutter_gen_k.slurm 5 main
sbatch --time=08:00:00 submit_perlmutter_gen_k.slurm 5 null
sbatch submit_perlmutter_gen_k.slurm 42-synthetic main
sbatch --time=08:00:00 submit_perlmutter_gen_k.slurm 42-synthetic null
sbatch submit_perlmutter_gen_k.slurm 42-production main
sbatch --time=08:00:00 submit_perlmutter_gen_k.slurm 42-production null
```

If a 48-hour job reaches its wall-clock guard before completing, submit the
same command again. It resumes from the saved epoch and optimizer state.

### Fixed rotated GEN libraries

The fixed-library study matches the original norm1-versus-norm42 input
structure. Rotation happens once while building each HDF5 library, never in
the training loop. One stored entry remains one source cycle. At N=420 the
unique class samples 420 native, unrotated mother-bank cycles, while class k samples
`420/k` entries whose mothers already have k fixed azimuthal copies.

Only the reused classes are materialized. Build the k=5 and k=7 stores from
the native mother bank:

```bash
sbatch submit_oscar_fixed_reuse_store.slurm 5
sbatch submit_oscar_fixed_reuse_store.slurm 7
```

The random rotations are deterministic. The first five angles in the k=7
store equal the k=5 angles for the same mother. Train through the original
`pfn_libtest_train.py` path with the successful scaled-sum N=420 recipe and
an 80-epoch cosine decay:

```bash
sbatch submit_oscar_fixed_reuse_train.slurm 5 main smoke
sbatch submit_oscar_fixed_reuse_train.slurm 7 main smoke
sbatch submit_oscar_fixed_reuse_train.slurm 5 main
sbatch submit_oscar_fixed_reuse_train.slurm 7 main
sbatch submit_oscar_fixed_reuse_train.slurm 5 null
sbatch submit_oscar_fixed_reuse_train.slurm 7 null
```

Each fixed-library null is reuse-versus-reuse at its named k: both labels use
the same fixed rotated store and independently sample `420/k` cycles. The
historical unique-versus-unique null remains the default in
`pfn_libtest_train.py` for older studies.

### Variable reuse after detector simulation

The matched RECO study uses the same 29,400 mother-equivalents per polarity
as the GEN study. Source cycles are split 50/25/25 before their mothers are
shuffled into immutable chunks of 140 distinct mothers. Every K library uses
the same chunk partition and the first K angles from one deterministic
21-angle sequence per mother. The exact overlay is:

| K | distinct mothers/event | rotated mothers/chunk | SIM files/event/polarity |
|---:|---:|---:|---:|
| 1 | 29,400 | 140 | 210 |
| 5 | 5,880 | 700 | 42 |
| 7 | 4,200 | 980 | 30 |
| 10 | 2,940 | 1,400 | 21 |
| 21 | 1,400 | 2,940 | 10 |

Build the missing MUMINUS mother bank, immutable chunk manifest, and one
validated GEN→SIM file at every K before committing to full production:

```bash
minus=$(sbatch --parsable submit_oscar_mother_store.slurm MUMINUS)
prepare=$(sbatch --parsable --dependency=afterok:"$minus" \
  submit_reco_variable_k_prepare.slurm)
smoke=$(sbatch --parsable --dependency=afterok:"$prepare" \
  submit_reco_variable_k_smoke.slurm)
printf 'MUMINUS=%s prepare=%s smoke=%s\n' "$minus" "$prepare" "$smoke"
```

After checking the smoke timing and file sizes, write the complete GEN
libraries. These ten 64-way arrays together remain below OSCAR's submitted-job
limit:

```bash
for polarity in MUPLUS MUMINUS; do
  for k in 1 5 7 10 21; do
    sbatch submit_reco_variable_k_gen.slurm "$k" "$polarity"
  done
done
```

Wait for GEN production to finish before submitting the ten SIM arrays; queuing
both stages at once would exceed the account's submitted-job limit:

```bash
for polarity in MUPLUS MUMINUS; do
  for k in 1 5 7 10 21; do
    sbatch submit_reco_variable_k_ddsim.slurm "$k" "$polarity"
  done
done
```

The RECO submitter refuses to run until every expected SIM chunk exists.
It creates 2,000 train, 2,000 validation, and 2,000 test neutrino-gun events
for each K, using the established packed reconstruction chain:

```bash
python3 submit_reco_variable_k.py
```

This production is resumable. Rerun the same command after a timeout; existing
RECO outputs are skipped. Once it reports that everything already exists,
build the five stores:

```bash
sbatch submit_reco_variable_k_stores.slurm
```

Train each physical comparison and its label-permutation null:

```bash
for comparison in 1v5 1v7 1v10 1v21; do
  sbatch submit_reco_variable_k_train.slurm "$comparison" main
  sbatch submit_reco_variable_k_train.slurm "$comparison" null
done
```

K=1 is reconstructed once and shared by all four comparisons. A null does not
require new detector production: labels are deterministically permuted within
each split over the same two physical K samples.

The original 1-vs-5 run early-stopped after 20 epochs, before the successful
1-vs-10 comparison first developed a clear validation signal at epoch 24.
The following controlled rerun keeps the same source split, sampled-unit
seeds, architecture, optimizer, model seed, validation-loss checkpoint rule,
and main/null construction. It raises the epoch cap to 120 and prevents early
stopping before epoch 80. Fresh labels prevent the old epoch-4 checkpoint from
being resumed:

```bash
sbatch --export=ALL,RUN_TAG=min80,EPOCHS=120,MIN_EPOCHS=80 \
  submit_oscar_variable_reuse.slurm 1v5 main
sbatch --export=ALL,RUN_TAG=min80,EPOCHS=120,MIN_EPOCHS=80 \
  submit_oscar_variable_reuse.slurm 1v5 null
```

## 3. N=420 reconstructed-PFO study

This fixed study overlays, per beam polarity and reconstructed event:

- U: 420 norm1 SIM files;
- R: 10 norm42 SIM files, representing `10 x 42` source copies;
- null: U versus an independently digitized `null_b` sample from norm1.

All classes use the same invisible 100 GeV PDG-14 particle gun. The signal is
there only to drive the simulation chain; reconstructed PFOs come from BIB.
The dataset contains 2,000 train, 400 validation, and 800 test events per
class. Source cycles are deterministically shuffled with seed 12345 before the
60/15/25 train/validation/test split.

The MAIA Pandora configuration must pass the reconstructed tracks into PFA:

```python
TrackCollections = ["SiTracks"]
RelTrackCollections = ["MergedTrackerHitsRelations"]
```

If these are empty, Pandora still produces neutral cluster-based PFOs, but
charged PFOs and PFO-to-track links are absent. Such output is not equivalent
to the intended RECO configuration and must not be used for the final PFN
comparison.

### Calorimeter-coning correction

MAIA normally keeps calorimeter hits only inside a 34-degree cone around each
generator particle. BIB MCParticles were not merged into the particle-gun
event, so the old RECO samples selected calorimeter hits around the neutrino
gun rather than around the BIB. The corrected study keeps the existing
energy/time hit selection but passes `--disableCaloConing` during DIGI.

It reuses one signal SIM chunk for all three classes at each split and chunk:

- train and validation use U signal SIM from `reco_n420_pfn_trackfix_val25`;
- test uses U signal SIM from `reco_n420_confirmation`;
- U overlays 420 norm1 files per polarity;
- R overlays 10 norm42 files per polarity;
- null_b repeats U with a digitization-seed offset of 1,000,000.

The frozen `bib_pools_val25` source split is used throughout. The production
contains 2,000 train, 2,000 validation, and 2,000 test events per class. It
runs only DIGI and RECO; GEN and SIM are not repeated.

After applying `patches/maia_disable_calo_coning.patch` to the OSCAR
MAIAConfig checkout, first run the two-chunk benchmark:

```bash
python3 submit_reco_calo_unconed.py --benchmark
```

Inspect both array tasks, then submit the full idempotent production:

```bash
python3 submit_reco_calo_unconed.py
```

The full manifest has 360 logical 50-event chunks packed across 64 OSCAR CPU
array shards. A rerun skips a chunk only when both its RECO file and completion
marker exist. Each completed chunk records the signal SIM, source pools,
overlay count, random seed, configuration hashes, commits, and stage timings.
Build the nine HDF5 stores only after all array tasks succeed:

```bash
sbatch submit_reco_calo_unconed_stores.slurm
```

Prepare immutable source pools:

```bash
source config.sh
python3 reco_libtest_prepare_pools.py \
  --norm1-sim "$DATA_GROUP_DIR/bib-v3p0-fmt2-norm1/SIM" \
  --norm42-sim "$DATA_GROUP_DIR/bib-v3p0-fmt2-norm42-RandomRot/SIM" \
  --outdir "$PSCRATCH/mucoll/libtest/bib_pools_simple" \
  --exclude-cycle 6291 --force
```

Produce any missing GEN->SIM->DIGI->RECO chunks. On OSCAR the submitter uses a
fixed 64-way CPU array, with each task processing its assigned reconstruction
chains sequentially. It skips valid existing outputs:

```bash
python3 submit_reco_libtest_packed.py
```

Rerun that command after a timeout until it reports that nothing remains.
Then build the nine PFO stores and train one main comparison followed by one
null on one L40S GPU, both with the fixed, predeclared seed 12345. Create the
two pinned OSCAR environments once before submitting either stage:

```bash
./setup_oscar_reco_envs.sh
sbatch submit_reco_libtest_stores.slurm
sbatch submit_reco_libtest_train.slurm
```

Each PFO contributes seven unscaled features: `ln(pT/GeV)`, `eta`, `sin(phi)`,
`cos(phi)`, `ln(E/GeV)`, charge in units of `e`, and an explicit
`abs(charge)>0.1e` indicator. The available Pandora photon identification has
not been updated for the muon-collider image and is deliberately excluded.
The charged indicator is information-theoretically redundant with charge but
is retained as a controlled optimization ablation; a separate neutral flag
would be exactly complementary and is omitted. No clipping or learned feature
normalization is applied. Missing required PFO branches and non-positive
energies are fatal rather than being silently replaced by zeros. The RECO PFN
uses a raw sum: at
roughly O(10) PFOs per event, the large GEN-level sum-saturation issue is not
present. The trainer imports the standard `energyflow.archs.PFN`; do not replace
it with the local GEN builder under an existing result label. `summary.json`
records the fixed seed, EnergyFlow and TensorFlow versions, one held-out test
AUC, the resolved store directory, a SHA-256 hash and collection statistics
for every input HDF5 store, the checkpoint hash, and the exact tracked-code
state. `run_context.json` is written before fitting so an interrupted run
still identifies its inputs. Track-fixed jobs refuse dirty tracked code,
stores without PFO-to-track links, ambiguous labels, and nonempty result
directories. Test events may share sources and are therefore correlated.
All RECO production, store, training, and plotting defaults point to the
track-fixed `reco_n420_pfn_trackfix` dataset and
`reco_n420_pfn_stores_trackfix` stores.

Important provenance correction: the legacy
`reco_pfn_results/reco_n420_directlog_*` models were trained by commit
`d043b02` from `reco_n420_pfn_stores_simple`, before the Slurm store path was
changed. They are not track-fixed results and must not be compared to
track-fixed confirmation events. The original nine-feature track-fixed result
is retained under `reco_n420_trackfix_directlog_stabilized_dropout_*`; the
six-feature ablation uses
`reco_n420_trackfix_directlog_minimal6_stabilized_dropout_*`; and the
seven-feature charged-flag ablation uses
`reco_n420_trackfix_directlog_charged7_stabilized_dropout_*`.

The canonical PFO-phi plot has a deliberately separate, one-file workflow:

```text
RECO ROOT files -> PandoraPFO momentum.x/y -> atan2(py, px) -> PDF
```

It reads every U and R event across the train, validation, and test
directories directly; it does not read or create an HDF5 store:

```bash
$PSCRATCH/mucoll/envs/reco-store/bin/python plot_reco_phi.py
```

The default output is
`plots/reco_n420_trackfix_direct_root_phi.pdf`. This is a descriptive
whole-dataset plot, not a held-out performance measurement.

The extended stores also retain selected `SiTracks` (mapped to their
`AllTracks` IP states), `PandoraClusters`, and the number of PFO-to-track
links. Plot the matched N=420 and N=840 `val25` samples, combining the train,
validation, and test partitions only for these descriptive distributions:

```bash
sbatch submit_reco_distributions.slurm all
```

Pass `420` or `840` instead of `all` to plot one size. This writes separate
U-versus-R and matched-null directories under
`plots/reco_n420_trackfix_val25_whole_distributions/` and
`plots/reco_n840_trackfix_val25_whole_distributions/`. Each contains
event-level PFO, track, and cluster multiplicities; all seven PFO inputs
consumed by the PFN; all stored selected-track and Pandora-cluster features;
additional event-level sums; a numerical summary; and descriptive
single-observable AUCs. Positive track momentum and cluster-energy object
distributions use plain natural logarithms. Event-level sums retain a
zero-safe display because the samples contain zero-object events; those
aggregate quantities are not PFN inputs. Existing distribution directories
are not overwritten.

After the unchanged baseline, two fixed optimizer studies reuse the same
stores, source split, features, architecture, batch size, and seed. Both use
Adam with a one-epoch linear warmup to `1e-4`, a 30-epoch cosine decay to
`1e-6`, no clipping, and explicit `jit_compile=False`. The second adds only
EnergyFlow's standard `F_dropouts=0.1`. Each job trains its matched null after
the main classifier:

```bash
train_job=$(sbatch --parsable \
  submit_reco_libtest_recipe.slurm stabilized_dropout)
sbatch --dependency=afterok:"$train_job" \
  submit_reco_libtest_confirmation_evaluate.slurm
```

The first job creates explicitly named track-fixed main and null models. The
dependent job loads their fingerprinted checkpoints without fitting and
evaluates them on the separate 5,000-event/class track-fixed confirmation
stores. Confirmation summaries fingerprint both the training and
confirmation datasets and refuse a checkpoint whose stored hash, dataset
tag, or PFO-track-link requirement does not match.

### Validation-size studies

Two charged-seven-feature studies separate two possible limitations of the
original 400-event/class validation set:

- `val2000` keeps the original 60/15/25 source-cycle split and the existing
  train and test RECO events, but reconstructs 2,000 validation events/class
  from the same validation-cycle pool.
- `val25` uses a 50/25/25 source-cycle split and reconstructs 2,000 train and
  2,000 validation events/class. The deterministic shuffle and unchanged 25%
  test fraction preserve the original 1,664 test cycles exactly, so the
  existing 800-event/class test cohort is reused.

With 6,654 usable paired cycles, `val25` contains 3,326 train, 1,664
validation, and 1,664 test cycles. Both studies use the same seven inputs,
EnergyFlow PFN, stabilized-dropout recipe, seed, 2,000 training events/class,
and frozen test and confirmation cohorts. Thus `val2000` tests event-count
noise alone; comparing `val25` with it additionally tests validation source
diversity (while reducing training source diversity).

After creating the `val25` pool, submit only the RECO partitions that are new:

```bash
python3 reco_libtest_prepare_pools.py \
  --norm1-sim /oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib_norm1_reconstructed/SIM \
  --norm42-sim /oscar/data/mleblan6/mucoll/bib/SIM \
  --outdir "$PSCRATCH/mucoll/libtest/bib_pools_val25" \
  --exclude-cycle 6291 \
  --val-fraction 0.25 \
  --test-fraction 0.25 \
  --force

python3 submit_reco_libtest_packed.py \
  --pools "$PSCRATCH/mucoll/libtest/bib_pools_simple" \
  --outdir "$PSCRATCH/mucoll/libtest/reco_n420_pfn_trackfix_val2000" \
  --splits val \
  --val-events 2000

python3 submit_reco_libtest_packed.py \
  --pools "$PSCRATCH/mucoll/libtest/bib_pools_val25" \
  --outdir "$PSCRATCH/mucoll/libtest/reco_n420_pfn_trackfix_val25" \
  --splits train val \
  --train-events 2000 \
  --val-events 2000
```

After each packed job succeeds, use the matching study name for store
construction, training, and frozen-model confirmation:

```bash
store_job=$(sbatch --parsable submit_reco_libtest_stores.slurm val2000)
train_job=$(sbatch --parsable --dependency=afterok:"$store_job" \
  submit_reco_libtest_recipe.slurm stabilized_dropout val2000)
sbatch --dependency=afterok:"$train_job" \
  submit_reco_libtest_confirmation_evaluate.slurm val2000

store_job=$(sbatch --parsable submit_reco_libtest_stores.slurm val25)
train_job=$(sbatch --parsable --dependency=afterok:"$store_job" \
  submit_reco_libtest_recipe.slurm stabilized_dropout val25)
sbatch --dependency=afterok:"$train_job" \
  submit_reco_libtest_confirmation_evaluate.slurm val25
```

The store builder requires the exact declared event count and records the
source-pool manifest path and SHA-256 fingerprint in every HDF5 store.

### N=840 scale point

The same pipeline accepts `--n-files 840`. This corresponds to 840 distinct
norm1 files for U/null and 20 norm42 files for R (`20 x 42 = 840`), or about
1.26% of a full bunch crossing. N=840 uses the 50/25/25 source pool, 2,000
train, 2,000 validation, and 800 test events/class, the same seven PFO inputs,
and the same stabilized-dropout EnergyFlow PFN. Its raw RECO, stores,
checkpoints, null, and plots all have independent `n840` names. Files from an
N=420 store are rejected when an N=840 model is requested.

Produce all three N=840 partitions:

```bash
python3 submit_reco_libtest_packed.py \
  --n-files 840 \
  --pools "$PSCRATCH/mucoll/libtest/bib_pools_val25" \
  --outdir "$PSCRATCH/mucoll/libtest/reco_n840_pfn_trackfix_val25" \
  --splits train val test \
  --train-events 2000 \
  --val-events 2000 \
  --test-events 800
```

The N=840 default is 16 hours and 16 GB per array task. After recording the
printed packed job ID, queue its store, main-plus-null training, and evaluation
on the 800-event/class held-out test cohort:

```bash
n840_store=$(sbatch --parsable --mem=64G \
  --dependency=afterok:"$n840_cpu" \
  submit_reco_libtest_stores.slurm val25 840)
n840_train=$(sbatch --parsable --dependency=afterok:"$n840_store" \
  submit_reco_libtest_recipe.slurm stabilized_dropout val25 840)
```

An N=840 model cannot use the N=420 confirmation cohort. A separate 5,000
event/class confirmation production is supported when that additional CPU
cost is wanted:

```bash
python3 submit_reco_libtest_confirmation.py \
  --n-files 840 \
  --study val25 \
  --pools "$PSCRATCH/mucoll/libtest/bib_pools_val25" \
  --outdir "$PSCRATCH/mucoll/libtest/reco_n840_confirmation"
```

Its store and evaluator must depend on confirmation production; evaluation
must additionally depend on N=840 training:

```bash
n840_confirm_store=$(sbatch --parsable --mem=64G \
  --dependency=afterok:"$n840_confirm_cpu" \
  submit_reco_libtest_confirmation_stores.slurm 840)
n840_eval=$(sbatch --parsable \
  --dependency=afterok:"$n840_train":"$n840_confirm_store" \
  submit_reco_libtest_confirmation_evaluate.slurm val25 840)
```

### N=1260 scale point

N=1260 corresponds to 1,260 distinct norm1 files for U/null and 30 norm42
files for R (`30 x 42 = 1260`), or about 1.89% of a full bunch crossing. It
uses the same 3,326/1,664/1,664 train/validation/test source-cycle pool as
N=840, so source partitions, event counts, features, and training remain
fixed while only the BIB overlay size changes.

Queue the entire N=1260 pipeline behind the N=840 training job. The packed
submitter defaults to 24 hours and 40 GB per array task above N=840:

```bash
output=$(python3 submit_reco_libtest_packed.py \
  --n-files 1260 \
  --pools "$PSCRATCH/mucoll/libtest/bib_pools_val25" \
  --outdir "$PSCRATCH/mucoll/libtest/reco_n1260_pfn_trackfix_val25" \
  --splits train val test \
  --train-events 2000 \
  --val-events 2000 \
  --test-events 800 \
  --dependency "afterok:${n840_train}")
echo "$output"
n1260_cpu=$(echo "$output" | awk '/submitted packed job/{print $4}')

n1260_store=$(sbatch --parsable --mem=96G \
  --dependency=afterok:"$n1260_cpu" \
  submit_reco_libtest_stores.slurm val25 1260)
n1260_train=$(sbatch --parsable --dependency=afterok:"$n1260_store" \
  submit_reco_libtest_recipe.slurm stabilized_dropout val25 1260)

printf 'N=1260: CPU=%s store=%s train=%s\n' \
  "$n1260_cpu" "$n1260_store" "$n1260_train"
```

This pipeline performs the standard 800-event/class held-out test for the
main U-versus-R classifier and its matched null. It does not queue the optional
5,000-event/class confirmation production.

Extend the same held-out test pool to 2,000 events per class without
retraining or overwriting the original result:

```bash
output=$(python3 submit_reco_libtest_packed.py \
  --n-files 1260 \
  --pools "$PSCRATCH/mucoll/libtest/bib_pools_val25" \
  --outdir "$PSCRATCH/mucoll/libtest/reco_n1260_pfn_trackfix_val25" \
  --splits test \
  --test-events 2000)
echo "$output"
test2000_cpu=$(echo "$output" | awk '/submitted packed job/{print $4}')

test2000_store=$(sbatch --parsable --mem=96G \
  --dependency=afterok:"$test2000_cpu" \
  submit_reco_libtest_stores.slurm test2000 1260)

test2000_eval=$(sbatch --parsable \
  --dependency=afterok:"$test2000_store" \
  submit_reco_libtest_confirmation_evaluate.slurm test2000 1260)

printf 'N=1260 test2000: CPU=%s store=%s eval=%s\n' \
  "$test2000_cpu" "$test2000_store" "$test2000_eval"
```

The submitter skips the existing 800 events and produces only the 1,200
missing events per class. The store and evaluation use separate `test2000`
paths and the frozen original checkpoint. This refines performance for the
same held-out source-cycle pool; it does not constitute an independent
source-cycle fold.

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
