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
| Shared GEN representation and PFN | `libtest_common.py` |
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
sbatch submit_oscar_n420_recipe.slurm scaled 3e-4 1
sbatch submit_oscar_n420_recipe.slurm raw 1e-4 1
sbatch submit_oscar_n420_recipe.slurm raw 3e-4 1
```

Each command is resumable under its immutable label. The full model, Adam
state, learning-rate position, epoch, and validation-selection state are
restored strictly. Production training reports the AUC on overlapping
held-out events. Paired source-cycle uncertainty is run separately only when
needed.

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

The current binary study compares 10x with 42x reuse at the N=420 event
scale. Both classes contain 29,400 mother-equivalents: k=10 samples 2,940
distinct mothers and k=42 samples 700. The selected recipe is the official
EnergyFlow scaled-sum network, expanded GEN features, balanced batches of
four, peak learning rate `1e-4`, one-epoch warmup, and cosine decay.

```bash
sbatch submit_oscar_variable_k10_k42.slurm main
sbatch submit_oscar_variable_k10_k42.slurm null
```

The null permutes labels over the same sampled k=10 and k=42 units, removing
the association between reuse factor and target while preserving the full
input construction. Both evaluations use overlapping pseudo-events from the
held-out source-cycle pool and report a point AUC; cycle-level uncertainty is
run separately if needed.

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

Prepare immutable source pools:

```bash
source config.sh
python3 reco_libtest_prepare_pools.py \
  --norm1-sim "$DATA_GROUP_DIR/bib-v3p0-fmt2-norm1/SIM" \
  --norm42-sim "$DATA_GROUP_DIR/bib-v3p0-fmt2-norm42-RandomRot/SIM" \
  --outdir "$PSCRATCH/mucoll/libtest/bib_pools_simple" \
  --exclude-cycle 6291 --force
```

Produce any missing GEN->SIM->DIGI->RECO chunks. The submitter packs up to 64
independent four-core chains per CPU node into one allocation and skips valid
existing outputs:

```bash
python3 submit_reco_libtest_packed.py
```

Rerun that command after a timeout until it reports that nothing remains.
Then build the nine PFO stores and train one main comparison followed by one
null on a single shared-QOS GPU, both with the fixed, predeclared seed 12345:

```bash
sbatch submit_reco_libtest_stores.slurm
sbatch submit_reco_libtest_train.slurm
```

Each PFO contributes `log(pT)`, `eta`, `sin(phi)`, `cos(phi)`, `log(E)`,
charge, and charged/photon/neutral indicators derived from
`PandoraPFOs.PDG`. Missing required PFO branches are fatal rather than being
silently replaced by zeros. The RECO PFN uses a raw sum: at
roughly O(10) PFOs per event, the large GEN-level sum-saturation issue is not
present. The trainer imports the standard `energyflow.archs.PFN`; do not replace
it with the local GEN builder under an existing result label. `summary.json`
records the fixed seed, EnergyFlow and TensorFlow versions, and one held-out
test AUC. Test events may share sources and are therefore correlated.

The `_simple` pool, RECO, store, and result names prevent this data from being
mixed with earlier, incompatible outputs.

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
