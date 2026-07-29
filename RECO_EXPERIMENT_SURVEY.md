# RECO-level BIB-reuse experiment survey

This is a provenance-first account of the RECO experiments that led to the
current N=420 result and the N=840 scale point.  It deliberately separates:

- the early Perlmutter pilot, whose complete machine-readable context is not
  present locally;
- OSCAR models trained on the legacy `reco_n420_pfn_stores_simple` stores;
- models trained on the regenerated, track-fixed stores;
- later feature, validation-sample, and overlay-size studies.

That distinction is scientifically important.  In particular, the
`reco_n420_directlog_*` labels do **not** denote track-fixed data.

## Common physics construction

The classification target is:

- **U (class 0):** one reconstructed pseudo-event made by overlaying particles
  from 420 distinct unrotated `norm1` BIB source files;
- **R (class 1):** the same nominal BIB amount made from ten
  `norm42-RandomRot` files, each containing the 42 rotated copies of its source
  mothers (`10 x 42 = 420`);
- **null:** two independently constructed U samples, used to verify that the
  pipeline cannot distinguish arbitrary labels when the physical construction
  is the same.

N=420 is approximately `420 / (6666 x 10) = 0.00630`, or **0.63% of one
full bunch crossing** under the working normalization.  The event-processing
chain uses a neutrino particle gun plus the BIB overlay, detector simulation,
digitization, and MAIA reconstruction.  The neutrino supplies the event
framework without adding a visible hard-scatter object.

At training time the PFN consumes reconstructed `PandoraPFOs`.  Tracks and
clusters are retained in the later HDF5 stores and are useful for validation
plots, but they are not separate PFN inputs in any result in this survey.

Unless a section says otherwise, the modern classifier is the standard
`energyflow.archs.PFN`:

- per-PFO network `Phi = (64, 64, 64)`;
- permutation-invariant **raw sum** over the per-PFO latent vectors;
- event-level network `F = (64, 64, 64)`;
- binary cross-entropy;
- batch size 32;
- fixed training seed 12345;
- zero padding, masked by the PFN;
- test AUC evaluated once on the selected best-validation-loss checkpoint.

The test events are source-disjoint from training and validation, but distinct
test events may reuse members of the fixed held-out source pool.  Consequently
the quoted AUCs are point estimates; treating all event scores as independent
would understate uncertainty.

## Result map

| Stage | Dataset actually used | Main test AUC | Null test AUC | Main plot |
|---|---|---:|---:|---|
| Perlmutter pilot, seed 1 | legacy pilot; full context unavailable | 0.587 | 0.501 | [plot](plots/reco_n420_energyflow_training_history_seed1.pdf) |
| Perlmutter pilot, seed 2 | same pilot construction, new initialization | 0.521 | 0.496 | [plot](plots/reco_n420_energyflow_training_history_seed2.pdf) |
| Original preprocessing | legacy simple stores | 0.519 | 0.496 | [plot](plots/training_overlays/reco_n420/reco_n420_baseline.pdf) |
| Stabilized optimizer | legacy simple stores | 0.529 | 0.487 | [plot](plots/training_overlays/reco_n420/reco_n420_stabilized.pdf) |
| Stabilized + dropout | legacy simple stores | 0.534 | 0.491 | [plot](plots/training_overlays/reco_n420/reco_n420_stabilized_dropout.pdf) |
| Direct-log baseline | legacy simple stores | 0.737 | 0.506 | [plot](plots/training_overlays/reco_n420_directlog/reco_n420_directlog_baseline.pdf) |
| Direct-log stabilized | legacy simple stores | 0.696 | 0.497 | [plot](plots/training_overlays/reco_n420_directlog/reco_n420_directlog_stabilized.pdf) |
| Direct-log stabilized + dropout | legacy simple stores | 0.664 | 0.514 | [plot](plots/training_overlays/reco_n420_directlog/reco_n420_directlog_stabilized_dropout.pdf) |
| Track-fixed, nine features | track-fixed stores | 0.653 | 0.501 | [plot](plots/training_overlays/reco_n420_trackfix_directlog/reco_n420_trackfix_directlog_stabilized_dropout.pdf) |
| Track-fixed, six features | same track-fixed stores | 0.624 | 0.497 | [plot](plots/training_overlays/reco_n420_trackfix_directlog_minimal6/reco_n420_trackfix_directlog_minimal6_stabilized_dropout.pdf) |
| Track-fixed, seven features | same track-fixed stores | 0.649 | 0.487 | [plot](plots/training_overlays/reco_n420_trackfix_directlog_charged7/reco_n420_trackfix_directlog_charged7_stabilized_dropout.pdf) |
| N=420, seven features, 25% validation pool | track-fixed `val25` stores | **0.659** | 0.508 | [plot](plots/training_overlays/reco_n420_trackfix_validation/reco_n420_trackfix_charged7_val25.pdf) |
| N=840, otherwise matched to preceding row | track-fixed N=840 `val25` stores | **0.680** | 0.507 | [plot](plots/training_overlays/reco_n840_trackfix/reco_n840_trackfix_charged7_val25.pdf) |

The associated null plots are linked in the detailed sections below.

## 1. Reconstruction-occupancy pilot: choose a viable BIB size

Before training, N=42, N=210, and N=420 reconstructed samples were used to
check whether enough PFOs survived reconstruction:

| Overlay size | U events with zero PFOs | R events with zero PFOs |
|---:|---:|---:|
| N=42 | 38/50 | 39/50 |
| N=210 | 4/10 | 1/10 |
| N=420 | 0/10 | 0/10 |

Architecture: no classifier; this was a reconstruction feasibility study.

Result: N=42 was mostly empty, N=210 was marginal, and every event in the
small N=420 pilot contained at least one PFO.

Takeaway: use N=420 for the first full RECO PFN.  This choice was driven by
reconstruction occupancy, not by classifier performance.

Useful surviving descriptive plot:
[whole-sample track-fixed PFO multiplicity](plots/reco_n420_trackfix_whole_distributions/main/pfo_multiplicity.pdf).
The original three-size slide plot is not currently present as a file in this
checkout.

## 2. Initial Perlmutter EnergyFlow PFN pilot

Plots:

- [seed 1](plots/reco_n420_energyflow_training_history_seed1.pdf)
- [seed 2](plots/reco_n420_energyflow_training_history_seed2.pdf)

Architecture: standard EnergyFlow PFN on reconstructed PFOs, U versus R and a
matched null.  These plots predate the provenance-complete OSCAR result
format, so they should not be used to quote an exact feature transform, source
store fingerprint, or track-link status.

Results:

- seed 1: main AUC **0.587**, null AUC **0.501**;
- seed 2: main AUC **0.521**, null AUC **0.496**;
- training loss decreased while validation loss worsened in the main runs.

Takeaway: there might be a weak RECO-level reuse signal, but the seed spread
and diverging validation curve made the pilot insufficient as a result.  It
motivated a reproducible OSCAR pipeline with explicit nulls and recorded
configuration.

## 3. Reproducible OSCAR baseline with the original feature transform

Plots:

- [main](plots/training_overlays/reco_n420/reco_n420_baseline.pdf)
- [null](plots/training_overlays/reco_n420/reco_n420_baseline_null.pdf)

Dataset: legacy `reco_n420_pfn_stores_simple` stores.  This is not the later
track-fixed dataset.

Features:

1. `ln(1 + pT/GeV) / 6`
2. clipped `eta / 5`
3. `sin(phi)`
4. `cos(phi)`
5. `ln(1 + E/GeV) / 6`
6. clipped `(charge/e) / 3`
7. charged indicator
8. photon indicator based on the Pandora PDG assignment
9. other-neutral indicator

Architecture and training: EnergyFlow PFN `(64,64,64)`/sum/`(64,64,64)`,
batch 32, EnergyFlow's original optimizer compile, 150-epoch cap, patience 20.
The data split contained 2,000 train, 400 validation, and 800 test events per
class.

Result: main AUC **0.519** after 33 epochs; null AUC **0.496** after 45
epochs.

Takeaway: the first reproducible setup was consistent with no useful
discrimination.  The null behaved correctly, so attention moved to training
stability and preprocessing rather than a label leak.

## 4. Optimizer stabilization and dropout scan

Plots:

- stabilized:
  [main](plots/training_overlays/reco_n420/reco_n420_stabilized.pdf),
  [null](plots/training_overlays/reco_n420/reco_n420_stabilized_null.pdf)
- stabilized plus dropout:
  [main](plots/training_overlays/reco_n420/reco_n420_stabilized_dropout.pdf),
  [null](plots/training_overlays/reco_n420/reco_n420_stabilized_dropout_null.pdf)

Dataset and features: unchanged legacy simple stores and original nine
transformed features.

The stabilized recipe changed only optimization:

- Adam;
- one-epoch linear warmup to `1e-4`;
- cosine decay over 30 epochs to `1e-6`;
- no gradient clipping;
- `jit_compile=False`;
- best checkpoint selected by validation loss.

The dropout variant additionally applied `F_dropout=0.1` in the event-level
network.

Results:

- stabilized: main **0.529**, null **0.487**;
- stabilized + dropout: main **0.534**, null **0.491**.

Takeaway: neither optimizer stabilization nor modest event-network dropout
rescued the original input representation.  The next controlled change was
the feature transform.

## 5. Direct-log feature experiment on the legacy stores

Plots:

- fixed-LR baseline:
  [main](plots/training_overlays/reco_n420_directlog/reco_n420_directlog_baseline.pdf),
  [null](plots/training_overlays/reco_n420_directlog/reco_n420_directlog_baseline_null.pdf)
- stabilized:
  [main](plots/training_overlays/reco_n420_directlog/reco_n420_directlog_stabilized.pdf),
  [null](plots/training_overlays/reco_n420_directlog/reco_n420_directlog_stabilized_null.pdf)
- stabilized + dropout:
  [main](plots/training_overlays/reco_n420_directlog/reco_n420_directlog_stabilized_dropout.pdf),
  [null](plots/training_overlays/reco_n420_directlog/reco_n420_directlog_stabilized_dropout_null.pdf)

Dataset: still the legacy `reco_n420_pfn_stores_simple` stores.

Features: the same nine semantic quantities, but with the hand clipping and
`/6`, `/5`, and `/3` scale factors removed:

`ln(pT/GeV), eta, sin(phi), cos(phi), ln(E/GeV), charge/e, is_charged,
is_photon, is_neutral`.

No learned normalization or clipping was applied.  The PFN architecture,
batch size, split sizes, and seed remained unchanged.

Results:

- fixed LR `1e-3`: main **0.737**, null **0.506**;
- stabilized: main **0.696**, null **0.497**;
- stabilized + dropout: main **0.664**, null **0.514**.

Takeaway: preprocessing had a much larger effect than the optimizer scan.
However, these unexpectedly strong numbers could not be promoted to the
physics result because the models were trained on the old simple stores.  The
next step was to regenerate the chain with the Pandora track inputs fixed and
then repeat the controlled model.

## 6. Pandora track-link fix and regenerated RECO dataset

The relevant MAIA Pandora configuration was changed from empty track inputs
to the intended tracking collections:

`TrackCollections = ["SiTracks"]` and
`RelTrackCollections = ["MergedTrackerHitsRelations"]`.

A ten-event check then found:

- 118 PFOs;
- 61 charged PFOs;
- 61 PFO-to-track links.

Architecture: no classifier in this check.

Takeaway: the regenerated dataset demonstrably contained PFO-track
associations.  From this point forward, production result directories record
the HDF5 hashes, collection statistics, dataset tag, checkpoint hash, code
commit, and whether track links were required.

Descriptive plots:

- [PFO multiplicity](plots/reco_n420_trackfix_whole_distributions/main/pfo_multiplicity.pdf)
- [track multiplicity](plots/reco_n420_trackfix_whole_distributions/main/track_multiplicity.pdf)
- [cluster multiplicity](plots/reco_n420_trackfix_whole_distributions/main/cluster_multiplicity.pdf)

These distributions combine train, validation, and test only for descriptive
purposes; the classifier split remains source-separated.

## 7. Track-fixed nine-feature PFN

Plots:

- [main](plots/training_overlays/reco_n420_trackfix_directlog/reco_n420_trackfix_directlog_stabilized_dropout.pdf)
- [null](plots/training_overlays/reco_n420_trackfix_directlog/reco_n420_trackfix_directlog_stabilized_dropout_null.pdf)

Dataset: fingerprinted track-fixed stores with 2,000/400/800 events per class
for train/validation/test.

Architecture: direct-log nine-feature PFN, stabilized + dropout recipe,
batch 32, seed 12345.

Results:

- original held-out test: main **0.653**, null **0.501**;
- separate frozen-checkpoint confirmation on 5,000 newly reconstructed events
  per class: main **0.648**, null **0.504**.

Confirmation plot:
[original versus confirmation](plots/reco_n420_trackfix_directlog/reco_n420_trackfix_auc_confirmation.pdf).

Takeaway: unlike the earlier incorrectly matched confirmation attempt, this
was a like-for-like track-fixed evaluation.  The main AUC reproduced on a much
larger cohort while the null remained at chance.  This established a real,
modest RECO-level signal and shifted the question to feature justification.

## 8. PFO category-feature ablation

### Six continuous/charge features

Plots:

- [main](plots/training_overlays/reco_n420_trackfix_directlog_minimal6/reco_n420_trackfix_directlog_minimal6_stabilized_dropout.pdf)
- [null](plots/training_overlays/reco_n420_trackfix_directlog_minimal6/reco_n420_trackfix_directlog_minimal6_stabilized_dropout_null.pdf)

Features:
`ln(pT/GeV), eta, sin(phi), cos(phi), ln(E/GeV), charge/e`.

Everything else, including the HDF5 stores, split, seed, PFN, and optimizer,
was fixed.

Results:

- original test: main **0.624**, null **0.497**;
- 5,000/class confirmation: main **0.626**, null **0.498**.

Takeaway: removing all three category flags produced a small but reproducible
loss in discrimination.  This also removed the untrustworthy `is_photon`
definition from the input.

### Add back only `is_charged`

Plots:

- [main](plots/training_overlays/reco_n420_trackfix_directlog_charged7/reco_n420_trackfix_directlog_charged7_stabilized_dropout.pdf)
- [null](plots/training_overlays/reco_n420_trackfix_directlog_charged7/reco_n420_trackfix_directlog_charged7_stabilized_dropout_null.pdf)

Features: the six above plus `abs(charge) > 0.1e`.

Results:

- original test: main **0.649**, null **0.487**;
- 5,000/class confirmation: main **0.640**, null **0.509**.

Takeaway: the explicit charged flag recovered most of the nine-feature
performance without relying on the Pandora photon label.  Although it is
mathematically derivable from charge, it is a simple nonlinear threshold that
the ablation showed was useful to expose directly.  This seven-feature set
became the controlled architecture for the validation-size and N-scaling
studies.

## 9. Larger validation sample and larger validation source pool

Plot:

- [main](plots/training_overlays/reco_n420_trackfix_validation/reco_n420_trackfix_charged7_val25.pdf)
- [null](plots/training_overlays/reco_n420_trackfix_validation/reco_n420_trackfix_charged7_val25_null.pdf)
- [slightly zoomed main](plots/training_overlays/reco_n420_trackfix_validation/reco_n420_trackfix_charged7_val25_zoom.pdf)

Dataset:

- 6,654 usable paired cycles;
- 3,326 train, 1,664 validation, and 1,664 test cycles;
- 2,000 train, 2,000 validation, and 800 test events per class;
- the held-out test-cycle pool remained the same 1,664-cycle pool used by the
  preceding N=420 track-fixed studies.

Architecture: unchanged seven-feature EnergyFlow PFN and stabilized-dropout
recipe.

Results:

- main: 118 epochs, best epoch 102, best validation loss **0.650**, test AUC
  **0.659**;
- null: 45 epochs, best validation loss **0.697**, test AUC **0.508**.

Takeaway: the larger and more source-diverse validation sample produced a
validation curve that clearly went below `ln(2)` for the physical comparison
and selected a late checkpoint.  Its test AUC was consistent with, but not
dramatically higher than, the earlier seven-feature result.  Because both the
number of validation events and the validation-cycle fraction changed, this
run alone does not isolate which change helped.

A planned `val2000` control would have increased only the validation event
count while retaining the original source split.  No completed
`reco_n420_trackfix_val2000_*` result is present in the local result tree, so
no result is claimed for it here.

## 10. N=840 overlay-size extension

Plots:

- [main](plots/training_overlays/reco_n840_trackfix/reco_n840_trackfix_charged7_val25.pdf)
- [null](plots/training_overlays/reco_n840_trackfix/reco_n840_trackfix_charged7_val25_null.pdf)

Construction:

- U uses 840 distinct unrotated source files;
- R uses 20 rotated files (`20 x 42 = 840`);
- approximately **1.26% of a full bunch crossing**;
- same 50/25/25 source split and 2,000/2,000/800 event counts;
- same seven features, PFN, optimizer, dropout, batch size, and seed.

Results:

- main: 47 epochs, best epoch 27, best validation loss **0.674**, test AUC
  **0.680**;
- null: reached the 150-epoch cap, best validation loss **0.710**, test AUC
  **0.507**.

Takeaway: increasing the BIB amount from N=420 to N=840 increased the point
estimate from 0.659 to 0.680 while the null stayed near chance.  This is
suggestive that reuse becomes more visible at higher occupancy, but it is only
one training seed and there is not yet an N=840 5,000/class confirmation
cohort or source-level uncertainty estimate.

## 11. N=1260 status

The pipeline was configured for N=1260:

- U: 1,260 distinct unrotated files;
- R: 30 rotated files;
- approximately **1.89% of a full crossing**;
- otherwise the same N=840 `val25` source split, event counts, seven-feature
  PFN, and training recipe.

No `reco_n1260_*` result directory or plot is present locally.  It should be
reported as planned/queued, not as a completed measurement.

## What the sequence established

1. N=420 was the first reduced crossing size with reliable nonzero PFO
   occupancy in the pilot.
2. The early PFN result was seed-sensitive and overfit.
3. Optimizer stabilization and dropout alone did not rescue the original
   hand-scaled feature representation.
4. Direct logarithms exposed a stronger signal, but the first direct-log
   models were trained on legacy stores and are development results only.
5. After the Pandora track-link fix, a modest AUC near 0.65 reproduced on an
   independent 5,000-event/class cohort while the null stayed at chance.
6. Removing the unsupported photon flag was safe; retaining a charged
   indicator recovered most of the lost performance.
7. A larger validation pool made the validation behavior more interpretable
   without materially changing the N=420 test AUC.
8. The controlled N=840 point increased the AUC to about 0.68, motivating the
   unfinished N=1260 extension.

## Presentation-safe results

The cleanest N=420 statement is:

> A seven-feature, track-fixed EnergyFlow PFN distinguishes unique-mother and
> 42x-reuse N=420 RECO pseudo-events with test AUC 0.659; its matched null gives
> 0.508.  An earlier model with the same track-fixed reconstruction but nine
> features was independently confirmed at AUC 0.648 on 5,000 events per class,
> with null AUC 0.504.

The controlled scale comparison is:

> Keeping the source split, reconstructed event counts, seven PFO inputs,
> EnergyFlow PFN, optimizer, dropout, batch size, and seed fixed, the test AUC
> changes from 0.659 at N=420 to 0.680 at N=840; the corresponding nulls are
> 0.508 and 0.507.

Neither statement is yet a precision uncertainty measurement because
held-out pseudo-events overlap in their finite source pools and only one
model-initialization seed was used for the modern controlled runs.
