# GEN experiment survey: path to the final N=420 PFN

This is a chronological audit of the distinct GEN-level experiments that
changed the N=420 analysis. Resubmitted wall-clock windows are treated as one
experiment. The later synthetic variable-\(k\) studies are not included:
they used the final recipe rather than leading to it.

## What stayed common

- **Task:** classify a pseudo-event made from unique, unrotated BIB source
  cycles (class U) against one with 42-fold within-event mother reuse (class
  R).
- **Meaning of N:** N is the number of `norm1` source-file equivalents per
  pseudo-event. Thus N=42 compares 42 unrotated cycle files with one 42-fold
  rotated file; N=420 compares 420 unrotated files with ten 42-fold files.
  N=420 is approximately \(420/(6666\times10)=0.00630\), or **0.63% of one
  bunch crossing**.
- **Source protection:** source cycle IDs are assigned to train, validation,
  and test before pseudo-events are built. A cycle never crosses those
  boundaries. Files are sampled without replacement inside a pseudo-event.
  Held-out events may reuse held-out cycles across events; the final quoted
  AUC is therefore the marginal AUC for random events from the held-out
  finite cycle pool.
- **PFN form:** per-particle features pass through a shared \(\Phi\) MLP,
  masked latent vectors are summed, and an event-level \(F\) MLP produces a
  two-class softmax. Zero padding only rectangularizes a variable-length
  batch and is masked before pooling.
- **Inputs before the final recipe:** the nine "paper" features
  \(\log_{10}p_T,\theta,\cos\phi,\sin\phi\), and indicators for photon,
  neutron, electron, muon, and other.
- **Objective:** categorical cross-entropy. AUC is a ranking diagnostic; loss
  also tests probability calibration.

## Results at a glance

| Stage | Main result | What it changed |
|---|---:|---|
| N=42 feasibility | scaled overlap AUC 1.000; raw original test AUC 0.996 | established a strong learnable reuse signature and a clean null |
| N=126 scaled | overlap AUC 0.947 | showed the signature persisted for larger pseudo-events |
| N=210 scaled/raw | scaled overlap AUC 0.964; raw original test AUC 0.993 | motivated retaining both pooling choices |
| first N=420 | scaled overlap AUC 0.501; raw overlap AUC 0.943 | diagnosed optimization collapse, not absence of physical information |
| GPU-capacity benchmark | wide-PFN batch 4 fit; batch 8 aborted even on 80 GB | separated GPU memory from a \(2^{31}\) kernel-index limit |
| full-width/JIT/half-width scans | explicit JIT was unusably slow; half-width batch 4 ran | selected batch 4, \(\Phi=(100,100,128)\), and JIT off |
| learning-capacity scan | best short-run val AUC 0.879 for scaled half-width model | rejected the still-smaller \(\Phi=(50,50,64)\) model |
| EnergyFlow check | scaled val AUC 0.997–0.999 across three seeds | justified official EnergyFlow provenance and exposed raw-sum seed sensitivity |
| warmup/pooling | scaled+warmup test AUC 0.950; raw+warmup 0.841 | warmup helped, but fixed \(10^{-3}\) LR remained unstable |
| clipping check | clipnorm 1 converged to AUC 0.502 at N=420 | removed gradient clipping from the production recipe |
| final grid | scaled \(10^{-4}\): test AUC 0.984 | selected scaled pooling, lower LR, decay, expanded inputs |
| final controls | repeat AUC 0.985; null AUC 0.481 | demonstrated numerical reproducibility and no null separation |

## 1. Initial N=42 feasibility experiment

[Scaled training loss](plots/gen_n42_scaled_loss.pdf)  
[Scaled validation AUC](plots/gen_n42_scaled_auc.pdf)  
[Raw training loss](plots/gen_n42_raw_loss.pdf)  
[Raw validation AUC](plots/gen_n42_raw_auc.pdf)

### Training architecture

- Local Keras PFN, \(\Phi=(200,200,256)\), \(F=(200,200,200)\), ReLU
  hidden layers, two-output softmax; about 226k trainable parameters for the
  nine-feature input.
- Two pooling variants:
  - **raw:** \(\sum_i\Phi(x_i)\);
  - **scaled:** \(c\sum_i\Phi(x_i)\), where the single class-blind constant
    \(c\) is the inverse median training-event particle multiplicity.
    This controls latent magnitude but does not divide each event by its own
    multiplicity.
- Adam with fixed learning rate \(10^{-3}\), no warmup, clipping, dropout, or
  weight decay.
- Batch size 8. Scaled training used 2,000 events/class/epoch; raw used
  1,000/class/epoch. Validation used 300/class.
- The null compares two independently constructed samples from the same
  norm1 distribution.

### Result

- Scaled run: validation AUC 1.000; original mutually-disjoint test AUC
  0.9991; later 1,000/class overlapping held-out evaluation AUC **1.0000**.
- Raw run: validation AUC 0.9998; original test AUC **0.9963**.
- Scaled null: validation AUC 0.5217; overlapping test AUC **0.5156**.

### Takeaway

The classifier could very clearly identify the 42-fold construction at a
small, computationally cheap scale, while the null stayed compatible with
chance. This justified increasing the pseudo-event size. It did **not** yet
show whether performance would remain stable when the latent sum contained an
order of magnitude more particles.

## 2. Scaling the pseudo-event to N=126

[Training loss](plots/gen_n126_scaled_loss.pdf)  
[Validation AUC](plots/gen_n126_scaled_auc.pdf)

### Training architecture

- Same nine inputs and wide local PFN as N=42.
- Scaled pooling, fixed Adam \(10^{-3}\), no stabilization.
- Batch size reduced to 2 because events were larger.
- 1,000 training events/class/epoch and 300 fixed validation events/class.
- Source split was 60/15/25 for train/validation/test.

### Result

- Validation AUC 0.9827.
- Original disjoint test AUC 0.9540.
- Later 1,000/class overlapping held-out point estimate: **AUC 0.9471**.
- Repeated-cycle null: **AUC 0.5000**.

### Takeaway

The reuse signature survived at three times the original event size. The
lower AUC than N=42 also warned against treating AUC as a simple monotonic
function of N: batch size, source split, event statistics, and optimization
had changed simultaneously.

## 2a. The discarded N=714 pilot

[Scaled pilot](plots/training_overlays/gen_historical/n714_scaled_pilot.pdf)  
[Original invalid null](plots/training_overlays/gen_historical/n714_original_null.pdf)  
[Corrected shared-pool null](plots/training_overlays/gen_historical/n714_corrected_null.pdf)

### Training architecture

- Same wide nine-input scaled PFN with fixed Adam \(10^{-3}\).
- Batch 1, only 100 training events/class/epoch and 50 validation/class.
- The main test used 100 overlapping events/class.

### Result

- Main validation AUC 0.569; test AUC 0.433.
- The first null was invalid: its construction partitioned an already small
  source pool differently between the two labels, producing validation AUC
  1.0 and test AUC 0.638.
- The corrected shared-pool null returned exactly AUC 0.500.

### Takeaway

N=714 was too source- and validation-statistics-limited for the original
setup, and the first null exposed a construction bug rather than a physics
effect. This prompted the better-controlled N=210 point and the later
overlapping-test evaluation policy. These histories predate validation-loss
recording, so their production-style plots honestly show only the available
training loss, selected epoch, and test AUC.

## 3. N=210 and the raw-versus-scaled question

[Scaled training loss](plots/gen_n210_scaled_loss.pdf)  
[Scaled validation AUC](plots/gen_n210_scaled_auc.pdf)  
[Raw training loss](plots/gen_n210_raw_loss.pdf)  
[Raw validation AUC](plots/gen_n210_raw_auc.pdf)

### Training architecture

- Same wide nine-input local PFN.
- Batch 2; 1,000 training events/class/epoch; 300 validation events/class.
- 50/25/25 cycle split.
- Separate scaled- and raw-sum trainings; fixed Adam \(10^{-3}\).
- Nulls used the same within/across-event reuse rules as the main samples.

### Result

- Scaled: validation AUC 0.9919; disjoint test AUC 0.9665; later
  1,000/class overlapping test AUC **0.9638**.
- Raw: validation AUC 0.9836; original disjoint test AUC **0.9927**.
  No matching post-hoc overlapping raw evaluation was saved.
- Scaled null AUC 0.5000; raw null AUC 0.5010.

### Takeaway

Both pooling definitions retained a strong signal, and both nulls were clean.
Raw pooling appeared competitive or better, suggesting that the overall
latent magnitude might contain useful information. This is why scaled pooling
was not adopted exclusively, even though it was numerically safer.

The exercise also established the preferred evaluation target: many random
held-out pseudo-events with cross-event overlap, while preserving strict
train/validation/test source separation. Mutual test-event disjointness was
too data-starved at large N.

## 4. The first N=420 result: scaled collapse versus raw signal

[Original scaled training loss](plots/gen_n420_scaled_old_loss.pdf)  
[Original scaled validation AUC](plots/gen_n420_scaled_old_auc.pdf)  
[Original raw training loss](plots/gen_n420_raw_loss.pdf)  
[Original raw validation AUC](plots/gen_n420_raw_auc.pdf)  
[Production-style scaled-collapse confirmation](plots/training_overlays/gen_n420_diagnostics/gen_n420_scaled_collapse_repeat.pdf)

### Training architecture

- Same wide local PFN and nine paper features.
- N=420 means 420 norm1 files versus 10 norm42 files per event.
- Batch size 1, 500 training events/class/epoch, 300 validation/class.
- Fixed Adam \(10^{-3}\), no warmup or decay.
- Separate scaled, raw, and matched-null runs.

### Result

- Scaled: validation AUC 0.5017; overlapping held-out AUC **0.5010**.
- A separate 80-epoch scaled rerun also ended at **AUC 0.5000**.
- Raw: validation AUC 0.9773; overlapping held-out AUC **0.9433**.
- Scaled null AUC 0.4992; raw null AUC 0.5000.

### Takeaway

This was the pivotal diagnosis. N=420 plainly contained class information,
because the raw PFN found it, but the original scaled training failed to
learn. The flat scaled result could not be interpreted as "scaling removed
the signal"; it was entangled with batch size 1, a high fixed learning rate,
the wide network, and a much larger padded tensor. The next work therefore
targeted capacity and optimizer stability rather than abandoning scaled
pooling.

## 5. GPU memory, batch size, \(\Phi\) width, and integer overflow

[Full-width raw, batch 4](plots/training_overlays/gen_n420_diagnostics/scan_fullwidth_raw_batch4.pdf)  
[Full-width scaled, batch 4](plots/training_overlays/gen_n420_diagnostics/scan_fullwidth_scaled_batch4.pdf)  
[Half-width raw, batch 8](plots/training_overlays/gen_n420_diagnostics/scan_halfwidth_raw_batch8.pdf)  
[Half-width scaled, batch 4](plots/training_overlays/gen_n420_diagnostics/halfphi_scaled_batch4.pdf)  
[Quarter-width raw, batch 8](plots/training_overlays/gen_n420_diagnostics/quarterphi_raw_batch8.pdf)

This was not one experiment. It was a sequence of hardware and software
diagnostics that determined which architecture could be trained safely.

### 5a. Synthetic and real-data capacity benchmark

#### Benchmark architecture

- The original nine-input local PFN:
  \(\Phi=(200,200,256), F=(200,200,200)\).
- Representative padded particle counts:
  - N=42: about 125,000 particles/event;
  - N=126: about 375,000;
  - N=210: about 625,000;
  - N=420: about 1,255,800.
- Batch-size sweep 1, 2, 4, and 8.
- Fixed-shape synthetic batches measured the pure dense-tensor cost.
- Fresh real events checked the effect of variable particle counts and
  retracing.
- The sweep was run on A100 GPUs, including an 80 GB node, and recorded
  step time and peak GPU memory. It was a capacity benchmark, not a
  classification training.

#### Memory result

Measured peak memory for the wide PFN was approximately **3.8 kB per padded
particle slot** across the N sweep. The resulting rough N=420 costs were:

| Batch | Approximate wide-PFN memory at 1.256M particles/event |
|---:|---:|
| 1 | 4.4 GB |
| 2 | 8.9 GB |
| 4 | 17.8 GB |
| 8 | 35.5 GB |

The original pre-measurement estimate had assumed 8.2 kB/slot and was about
2.1 times too conservative. Consequently, some early N=210 batch-8 and N=420
batch-4 combinations were reported as **skipped by the safety estimator**;
they had not actually run out of memory.

#### Integer-overflow result

The independent hard constraint was

\[
 B\,N_{\rm padded}\,W_{\Phi,\max} < 2^{31},
\]

where \(B\) is batch size, \(N_{\rm padded}\) is the largest particle count
in the batch, and \(W_{\Phi,\max}\) is the widest per-particle layer. Above
that boundary, the TensorFlow/CUDA kernel computed a negative 32-bit work
element count and hard-aborted with
`Check failed: work_element_count >= 0`.

For the original width 256:

| Batch | Maximum particles/event below the int32 boundary |
|---:|---:|
| 1 | 8.39M |
| 2 | 4.19M |
| 4 | 2.10M |
| 8 | 1.05M |

Thus typical N=420 at batch 4 fit, but batch 8 was mathematically unsafe:
\(8\times1.256{\rm M}\times256 > 2^{31}\). It aborted on an 80 GB GPU while
more than 60 GB remained free. That is the decisive evidence that this
failure was **not an out-of-memory condition**.

Batch 4 with width 256 was also uncomfortably close to the boundary. A
variable N=420 draw reached approximately 2.07M padded particles, just below
the 2.097M limit. Because padding uses the largest event in the batch, one
high-multiplicity event controls the activation shape for all four events.

### 5b. Original full-width architecture scan

#### Training architecture

Four N=420 debug ranks were packed onto the four GPUs of one node:

| Run | Pooling | \(\Phi\) | \(F\) | Batch |
|---|---|---|---|---:|
| `scan_raw_large` | raw | (200,200,256) | (200,200,200) | 4 |
| `scan_raw_small` | raw | (100,100,128) | (100,100,100) | 8 |
| `scan_scaled_large` | scaled | (200,200,256) | (200,200,200) | 4 |
| `scan_null_large` | raw null | (200,200,256) | (200,200,200) | 4 |

All used N=420, 500 training events/class/epoch, 300 validation/class, and
the fixed \(10^{-3}\) optimizer.

#### Result

The three full-width batch-4 ranks wrote only two or three epochs before the
screen failed/stopped; their best validation AUCs were 0.500, 0.519, and
0.500. The narrower batch-8 run reached validation AUC 0.657 in its brief
screen. These are incomplete diagnostics, not scientific AUC results.

#### Takeaway

Average N=420 memory was not the only problem. Variable padding could push a
full-width batch toward the integer limit unpredictably. A safer production
model needed a smaller widest \(\Phi\) layer.

### 5c. Explicit XLA-JIT workaround attempt

The same four configurations were rerun with model JIT compilation enabled,
testing whether XLA-generated kernels would avoid the legacy 32-bit launch
path.

- Three of four ranks failed to complete even one epoch in roughly
  29 minutes.
- The only saved history, the small raw model, contained one epoch at
  validation AUC 0.500.
- Runtime was at least four times worse because the padded particle dimension
  changed between batches and triggered expensive compilation/recompilation.

JIT therefore was not a practical overflow workaround.

### 5d. Pure \(\Phi\)-width rerun

The original scan was repeated with each \(\Phi\) network halved while
leaving each run's batch size and event-level \(F\) network unchanged:

- large models:
  \(\Phi=(100,100,128), F=(200,200,200)\), batch 4;
- small model:
  \(\Phi=(50,50,64), F=(100,100,100)\), batch 8.

Halving the widest large-model layer from 256 to 128 doubled the batch-4
int32-safe ceiling from 2.10M to 4.19M particles/event.

All four configurations then completed four debug epochs:

- tiny raw model: best validation AUC 0.503;
- half-width raw: 0.714;
- half-width scaled: 0.879;
- half-width null: 0.500.

The tiny network had greater numerical headroom but did not learn adequately.
The half-width \(\Phi=(100,100,128), F=(200,200,200)\) model was both safe and
capable, so it became the production architecture.

### 5e. Real first-step batch diagnostic

After optimizer JIT was explicitly disabled, the diagnostic separately timed:

1. loading both HDF5 stores;
2. constructing one unique event;
3. constructing one reused event;
4. padding the batch;
5. entering `train_on_batch`;
6. returning from `train_on_batch`.

It executed one or two real optimizer steps at batch sizes 1, 2, and 4. All
three returned. This ruled out an unconditional batch-4 first-step failure
for the half-width architecture.

Batch 2 would have reduced memory and integer-index pressure further, but
each balanced batch would contain only one event/class and produce noisier
updates. Batch 4 still had ample headroom with width 128 and allowed exactly
two events/class, so batch 4 was selected.

### 5f. Host-memory/store side investigation

An expanded `_v2` HDF5 bank that also stored charge increased store RAM by
roughly 15% (about 4.5 GB per loaded store in the observed setup). The
charge-including N=420 job thrashed at its host-memory ceiling even though the
earlier charge-free stores fit. Charge was excluded from the final GEN input:
the existing absolute-PDG indicators already carried most of that information,
whereas retaining it required a materially larger in-memory bank.

### Combined takeaway

The production choice was not simply “a smaller neural network trains
faster.” It was:

- batch 8 + width 256: impossible at N=420 because of the int32 limit;
- batch 4 + width 256: usually fits memory but has inadequate tail headroom;
- explicit JIT: too slow and later associated with compilation hangs;
- batch 4 + width 128: safe against observed multiplicity tails and still
  learns;
- batch 2 + width 128: safer but unnecessarily noisy;
- batch 4 + \(\Phi=(100,100,128)\): the chosen compromise.

## 6. Official EnergyFlow implementation and initialization-seed check

[Official raw PFN, seed 1](plots/training_overlays/gen_energyflow_check/n42_energyflow_raw_seed1.pdf)  
[Official scaled PFN, seed 1](plots/training_overlays/gen_energyflow_check/n42_energyflow_scaled_seed1.pdf)  
The corresponding seed-2 and seed-3 production-style plots are in the same
directory.

### Training architecture

- N=42 diagnostic, nine features, wide
  \(\Phi=(200,200,256), F=(200,200,200)\), batch 8.
- Three model initialization seeds for each pooling choice.
- **Raw:** `energyflow.archs.PFN`.
- **Scaled:** `energyflow.archs.EFN` used as the package-provided weighted
  aggregation graph, with \(z_i=c\) for a real particle and zero for padding.
  This computes \(F(c\sum_i\Phi(x_i))\); it is a scaled PFN built on EFN,
  not a physical energy-weighted EFN.
- Fixed Adam \(10^{-3}\); no warmup or regularization.

### Result

- Official raw PFN best validation AUCs: **0.867, 0.839, 0.882**.
- Official scaled model best validation AUCs:
  **0.9968, 0.9971, 0.9988**.
- These wall-time-limited checks did not reach final held-out evaluation.

### Takeaway

This was a software-provenance and robustness check, not a replacement
physics measurement. It showed that the official EnergyFlow graph could be
used directly and that scaled pooling was dramatically more stable across
these three initializations. The raw result's seed dependence reinforced the
need to control model seed and optimizer dynamics. The final run therefore
used EnergyFlow rather than presenting a local reimplementation as the
package PFN.

## 7. Warmup and pooling at N=420

[Raw, fixed learning rate](plots/training_overlays/gen_n420_development/gen_n420_raw_fixed_lr.pdf)  
[Raw with warmup](plots/training_overlays/gen_n420_development/gen_n420_raw_warmup.pdf)  
[Scaled, fixed learning rate](plots/training_overlays/gen_n420_development/gen_n420_scaled_fixed_lr.pdf)  
[Scaled with warmup](plots/training_overlays/gen_n420_development/gen_n420_scaled_warmup.pdf)

### Training architecture

- N=420; nine paper features.
- Official EnergyFlow-backed models.
- \(\Phi=(100,100,128), F=(200,200,200)\); batch 4.
- 500 training events/class/epoch; 300 validation/class.
- Raw and scaled pooling, each with either:
  - fixed Adam \(10^{-3}\), or
  - a one-epoch linear warmup to the same fixed \(10^{-3}\).
- No clipping, dropout, L2, or post-warmup decay.

### Result

- Raw, fixed LR: best val AUC 0.583; test AUC **0.563**.
- Raw + warmup: best val AUC 0.853; point test AUC **0.841**.
- Scaled, fixed LR: best val AUC 0.861; test AUC **0.885**.
- Scaled + warmup: best val AUC 0.952; point test AUC **0.950**.

The curves are unstable: good epochs are followed by sharp collapses and
partial recovery. Best checkpoints preserve the peaks, but this is not a
desirable production training dynamic.

### Takeaway

Warmup clearly helped both pooling choices, and scaled pooling was the better
starting point. Warmup alone protected only the beginning of training:
holding \(10^{-3}\) indefinitely still permitted destructive late updates.
This led directly to the final lower peak learning rates and cosine decay.

## 8. Gradient clipping diagnostic

[Raw with clipnorm 1](plots/training_overlays/gen_n420_diagnostics/raw_clipnorm1.pdf)  
[Raw with clipnorm 1000](plots/training_overlays/gen_n420_diagnostics/raw_clipnorm1000.pdf)  
[Raw with warmup and no clipping](plots/training_overlays/gen_n420_diagnostics/raw_warmup_no_clip.pdf)  
[Scaled with warmup and no clipping](plots/training_overlays/gen_n420_diagnostics/scaled_warmup_no_clip.pdf)

### Training architecture

- Same half-width N=420 setup as the previous section.
- Short comparisons among raw+clipnorm 1, raw+warmup/no clipping, and
  scaled+warmup/no clipping.
- Fixed peak LR \(10^{-3}\).

### Result

- Raw clipnorm 1 collapsed to validation AUC **0.5017** and loss near
  \(\ln 2\).
- Raw warmup/no clipping reached validation AUC **0.703** within five epochs.
- Scaled warmup/no clipping reached validation AUC **0.785** within five
  epochs.
- An expanded-feature raw run with clipnorm 1 later also died near chance;
  clipnorm 1000 was too short/inconclusive to justify retaining clipping.

### Takeaway

The raw-sum network begins with extremely large summed activations and
gradients. A global norm cap of 1 was many orders of magnitude too restrictive
and removed the corrective updates needed to escape chance. Production
therefore used **no gradient clipping**.

Separately, several XLA/JIT runs hung after compilation with zero GPU use.
The first-step diagnostic proved that batch sizes 1, 2, and 4 returned when
the model and optimizer were explicitly compiled with `jit_compile=False`.
JIT was consequently disabled in the final recipe. This is a software
stability conclusion, not a classifier-performance result.

## 9. Stabilized expanded-feature N=420 recipe grid

[Scaled, peak LR \(10^{-4}\)](plots/training_overlays/gen_n420_recipe/gen_n420_scaled_lr1e-4.pdf)  
[Scaled, peak LR \(3\times10^{-4}\)](plots/training_overlays/gen_n420_recipe/gen_n420_scaled_lr3e-4.pdf)  
[Raw, peak LR \(10^{-4}\)](plots/training_overlays/gen_n420_recipe/gen_n420_raw_lr1e-4.pdf)  
[Raw, peak LR \(3\times10^{-4}\), incomplete](plots/training_overlays/gen_n420_recipe/gen_n420_raw_lr3e-4_incomplete.pdf)

### Training architecture

Changes relative to the warmup experiment:

- **13 expanded truth features:** the nine paper inputs plus
  \(\log_{10}E,\operatorname{asinh}(t),\operatorname{asinh}(v_z)\), and
  \(\operatorname{asinh}(\sqrt{v_x^2+v_y^2})\).
- Official EnergyFlow architecture; \(\Phi=(100,100,128)\),
  \(F=(200,200,200)\); 131,030 trainable parameters.
- Batch 4 with enforced two-U/two-R balance.
- 500 events/class/epoch, or 250 optimizer steps.
- One-epoch linear warmup, then cosine decay over 30 epochs to \(10^{-6}\).
- Peak-LR comparisons \(10^{-4}\) and \(3\times10^{-4}\).
- Scaled-versus-raw pooling comparison.
- No clipping, dropout, L2, or JIT.
- 50/25/25 cycle split with data seed 1701; model seed 1.
- Fixed 300/class validation set. Checkpoint/early stopping selected by
  validation **loss**, requiring an improvement larger than one estimated
  standard error; patience 15, maximum 80 epochs.
- Primary test: 300 overlapping held-out events/class, point estimate only.

### Result

- Scaled \(10^{-4}\): best epoch 17, best val loss 0.1425,
  val AUC 0.9886, test AUC **0.9845**.
- Scaled \(3\times10^{-4}\): best epoch 17, best val loss 0.1333,
  val AUC 0.9875, test AUC **0.9826**.
- Raw \(10^{-4}\): best epoch 9, best val loss 0.9402,
  val AUC 0.9417, test AUC **0.9433**.
- Raw \(3\times10^{-4}\) did not complete and is not quoted as a result.

### Takeaway

Both lower-LR scaled runs were strong, but \(10^{-4}\) was chosen as the
more conservative peak rate and gave the highest test AUC. Raw pooling still
ranked events well, confirming that scaling did not create the effect, but
its very large latent magnitudes produced poor cross-entropy calibration and
less stable curves. The expanded features intentionally maximize GEN-level
sensitivity to exact mother reuse; whether these invariants survive detector
reconstruction is a separate RECO question.

## 10. Final null and exact-configuration repeat

[Final presentation plot](plots/training_overlays/gen_n420_recipe/gen_n420_scaled_lr1e-4.pdf)  
[Exact-configuration repeat](plots/training_overlays/gen_n420_recipe/gen_n420_scaled_lr1e-4_repeat.pdf)  
[Matched null](plots/training_overlays/gen_n420_recipe/gen_n420_scaled_lr1e-4_null.pdf)

### Training architecture

Exactly the selected scaled \(10^{-4}\) recipe above.

- **Repeat:** same scientific configuration, same data seed 1701, and same
  model seed 1. It is a reproducibility rerun, not an independent data split
  or independent initialization.
- **Null:** the same training and evaluation machinery, but both labels are
  constructed from the same source distribution.

### Result

- Main: test AUC **0.98447**.
- Exact-configuration repeat: test AUC **0.98519**.
- Null: test AUC **0.48076**; validation AUC 0.5164 and validation loss
  0.6930.

### Takeaway

The main result reproduced to about \(7\times10^{-4}\) in AUC under an exact
configuration rerun, while the null remained compatible with chance. This is
the final defensible N=420 statement:

> At approximately 0.63% of a bunch crossing, a GEN-level scaled PFN can
> strongly distinguish unique-source pseudo-events from events with 42-fold
> within-event mother reuse (held-out overlapping-event AUC about 0.984),
> and the matched null does not separate.

The quoted test AUC is a point estimate on 300 events/class. It has no
calibrated uncertainty until a source-cycle-level resampling or repeated-fold
analysis is run.

## Interpretation boundaries

- This study measures the **within-event cloning/rotation artifact** with
  source-separated train/validation/test pools. It is not the same as the
  original pileup-style question in which train and test events draw from the
  same finite underlying library.
- The early N sweep is not a controlled measurement of AUC versus N because
  batch size, source fractions, validation construction, and optimizer
  behavior changed. It is an engineering/scientific development sequence.
- AUC above 0.5 with loss near or above \(\ln 2\) means useful ranking with
  poor calibration; it does not by itself imply a trustworthy probabilistic
  classifier.
- The final expanded truth inputs make the GEN test deliberately sensitive.
  They include exact rotation invariants such as production time and vertex
  coordinates. RECO degradation is therefore acceptable and scientifically
  informative rather than a contradiction.

## Reproducing the production-style plots

```bash
python3 pfn_libtest_plot_overlay.py RESULT_DIRECTORY \
  --title "PLOT TITLE" \
  --out plots/training_overlays/OUTPUT.pdf
```

Every new plot linked above uses this same function as the final N=420
presentation plot. Historical runs that predate validation-loss recording
retain their already-existing training-loss and validation-AUC plots.
