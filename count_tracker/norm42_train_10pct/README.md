# Norm42 10%-BX retraining study

This directory isolates the 10%-BX retraining and validation study. It selects
one immutable development corpus containing 167 Norm42 files per polarity,
audits sensor-condition coverage, constructs provenance-preserving training
arrays, trains the six collection models, and validates them against held-out
rows. Reconstruction remains a separate downstream stage.

The source selection matches the principal SIM-versus-COUNT studies:

- MUPLUS and MUMINUS are drawn independently;
- selection is without replacement within each polarity;
- the known-invalid MUMINUS cycle 6291 is excluded;
- matching cycle IDs across polarities are allowed but not required;
- a fixed seed and exact source manifest make the cohort reproducible.

The complete Norm42 arrays use row-aligned source IDs with MUMINUS encoded as
`cycle` and MUPLUS as `6666 + cycle`.  The audit streams these arrays with
memory mapping and does not alter the shared data.

## Prepare the source cohort on OSCAR

Run from the checked-out `mucoll-slurm` repository:

```bash
export REPO="$(git rev-parse --show-toplevel)"
export CT_DIR="$REPO/count_tracker"
export STUDY_ROOT="/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_train_10pct"
export SIM_ROOT="/oscar/data/mleblan6/mucoll/bib/SIM"

python "$CT_DIR/norm42_train_10pct/prepare_sources.py" \
  --sim-root "$SIM_ROOT" \
  --seed 42 \
  --cohort norm42_train10pct \
  --output "$STUDY_ROOT/source_selection"
```

Preparation refuses to replace an existing output directory.  This is
intentional: changing the selected cohort requires a new explicitly named
study root.

## Queue the coverage audit

```bash
export MANIFEST="$STUDY_ROOT/source_selection/sources.json"
export DATA_ROOT="/oscar/data/mleblan6/mucoll/speng44/bib_gen_mother_muon/data/norm42/input"
export SOURCE_ID_ROOT="/oscar/data/mleblan6/mucoll/speng44/bib_gen_mother_muon/data/norm42_source_ids"
export AUDIT_OUTPUT="$STUDY_ROOT/coverage_audit"

mkdir -p "$STUDY_ROOT/logs"

AUDIT_JOB=$(sbatch --parsable \
  --export=ALL,CT_DIR="$CT_DIR",MANIFEST="$MANIFEST",DATA_ROOT="$DATA_ROOT",SOURCE_ID_ROOT="$SOURCE_ID_ROOT",AUDIT_OUTPUT="$AUDIT_OUTPUT" \
  -o "$STUDY_ROOT/logs/audit_%j.out" \
  -e "$STUDY_ROOT/logs/audit_%j.err" \
  "$CT_DIR/norm42_train_10pct/submit_audit.slurm")

echo "audit=$AUDIT_JOB"
```

Review:

```bash
python -m json.tool "$AUDIT_OUTPUT/coverage_audit.json" | less
column -s, -t < "$AUDIT_OUTPUT/missing_sensor_conditions.csv" | less -S
```

If `passed_complete_sensor_coverage` is false, stop before building training
arrays.  Missing conditions require an explicit decision; they are not
silently backfilled or dropped by this workflow.

## Build the reference-format datasets

The approved split matches the baseline model's final preparation: rows are
split 80/20 with `random_state=42` and stratification by the packed
`(side, layer, module, sensor)` condition.  Conditions represented by exactly
one selected row cannot be stratified.  Those rows are explicitly placed in
training; every such exception is written to the collection's
`split_manifest.json`.  Validation remains a row-level validation sample, not
a source-disjoint sample.  Reconstruction evaluation will use held-out source
files separately.

First archive the exact Paper1 code commit recorded by the baseline model into
the study directory.  This avoids importing the reorganized current checkout:

```bash
export PAPER1_SOURCE="/oscar/data/mleblan6/mucoll/speng44/bib_gen_mother_muon/paper1interface"
export PAPER1_10PCT="$STUDY_ROOT/paper1_90bd576c"

bash "$CT_DIR/norm42_train_10pct/archive_paper1_commit.sh" \
  "$PAPER1_SOURCE" "$PAPER1_10PCT"
```

Then queue dataset construction:

```bash
export AUDIT_REPORT="$AUDIT_OUTPUT/coverage_audit.json"
export DATASET_OUTPUT="$STUDY_ROOT/datasets"

DATASET_JOB=$(sbatch --parsable \
  --export=ALL,CT_DIR="$CT_DIR",MANIFEST="$MANIFEST",AUDIT_REPORT="$AUDIT_REPORT",DATA_ROOT="$DATA_ROOT",SOURCE_ID_ROOT="$SOURCE_ID_ROOT",PAPER1_10PCT="$PAPER1_10PCT",DATASET_OUTPUT="$DATASET_OUTPUT" \
  -o "$STUDY_ROOT/logs/datasets_%j.out" \
  -e "$STUDY_ROOT/logs/datasets_%j.err" \
  "$CT_DIR/norm42_train_10pct/submit_build_datasets.slurm")

echo "datasets=$DATASET_JOB"
```

The builder verifies the successful coverage audit and its source-manifest
identity, transforms the selected raw rows using the exact recorded Paper1
geometry code, writes the official TabDDPM NumPy layout, and retains full-row
indices and source IDs under each collection's `provenance/` directory.  It
uses the established `genbib` environment because that environment contains
the scikit-learn implementation used for the reference split.  It records the
Python, NumPy, and scikit-learn versions and does not train a model.

After completion, inspect:

```bash
python -m json.tool "$DATASET_OUTPUT/dataset_manifest.json"
python -m json.tool "$DATASET_OUTPUT/OTBC/split_manifest.json" | less
```

## Train the six collection models

Training uses the exact archived Paper1 trainer and the baseline hyperparameters:
200,000 steps, batch size 4096, learning rate `5e-4` with the trainer's linear
decay, 1,000 diffusion timesteps, five 2048-wide hidden layers, timestep
dimension 1024, quantile normalization, cosine diffusion scheduler, and seed
42.  The array requests one GPU per collection and caps concurrency at two.

```bash
export DATASET_ROOT="$STUDY_ROOT/datasets"
export MODEL_OUTPUT="$STUDY_ROOT/models"

mkdir -p "$STUDY_ROOT/logs"

TRAIN_JOB=$(sbatch --parsable \
  --export=ALL,CT_DIR="$CT_DIR",DATASET_ROOT="$DATASET_ROOT",PAPER1_10PCT="$PAPER1_10PCT",MODEL_OUTPUT="$MODEL_OUTPUT" \
  -o "$STUDY_ROOT/logs/train_%A_%a.out" \
  -e "$STUDY_ROOT/logs/train_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/submit_training.slurm")

echo "training=$TRAIN_JOB"
```

The exact trainer saves final and EMA weights but does not use validation for
early stopping or checkpoint selection.  Each task writes to a `.partial`
directory and publishes the final model directory only after reaching step
200,000 and validating all required outputs.

Check status with:

```bash
squeue -j "$TRAIN_JOB"
python "$CT_DIR/norm42_train_10pct/training_status.py" \
  --model-root "$MODEL_OUTPUT"
```

## Validate the trained models on held-out rows

The reconstruction sampler explicitly loads `model.pt`; it does not load
`model_ema.pt`.  Validation therefore uses `model.pt` as well.  Each array task
selects 20,000 rows uniformly without replacement from one collection's
held-out validation rows and requests generated hits with exactly the same
sensor-condition multiplicities.  It then applies the production sampler's
inverse geometry, endcap z snapping, material-map rejection, and retry path.

The output records rejected/unfilled conditions, one-dimensional physical
feature summaries, and a fixed hit-level classifier two-sample diagnostic.
The classifier is a validation diagnostic, not the reconstruction-level PFN.
Its inputs and hyperparameters are fully recorded in each JSON report.

```bash
export VALIDATION_OUTPUT="$STUDY_ROOT/heldout_validation_20k"

VALIDATION_JOB=$(sbatch --parsable \
  --export=ALL,CT_DIR="$CT_DIR",MODEL_ROOT="$MODEL_OUTPUT",PAPER1_ROOT="$PAPER1_10PCT",VALIDATION_OUTPUT="$VALIDATION_OUTPUT",HITS=20000,SEED=42,MAX_ROUNDS=100 \
  -o "$STUDY_ROOT/logs/validation_%A_%a.out" \
  -e "$STUDY_ROOT/logs/validation_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/submit_validation.slurm")

echo "validation=$VALIDATION_JOB"
```

The array runs at most two collections concurrently. It refuses to replace an
existing per-collection output, so a partial or failed result must be inspected
before any targeted retry.

After all six tasks complete:

```bash
python "$CT_DIR/norm42_train_10pct/summarize_validation.py" \
  --validation-root "$VALIDATION_OUTPUT" \
  --output-json "$VALIDATION_OUTPUT/summary.json" \
  --output-csv "$VALIDATION_OUTPUT/summary.csv"
```

## Source-disjoint three-arm track checkpoint

The first reconstruction test is one 167-file-per-polarity event, corresponding
to `167/1666 = 10.024%` of the established full-BX reference. It compares the
same requested sensor occupancy in three arms:

- empirical `SIM`;
- `COUNT_FULL`, from Shiyu's existing full-corpus Norm42 model;
- `COUNT_10PCT`, from the new 10%-corpus model.

MUPLUS and MUMINUS are drawn independently, without replacement within each
polarity. Because MUMINUS is the flipped MUPLUS source, a cycle is excluded
from both evaluation pools if either polarity appeared in the 10%-model
training corpus. Thus the event is physically source-disjoint from the new
model's training sources. The two COUNT arms use the same conditions, sampler,
Paper1 commit, rejection policy, input construction, v3 digitization, and
five-theta-region CKF reconstruction.

Prepare the immutable checkpoint manifest interactively in `count-hdf`:

```bash
export TRACK_ROOT="$STUDY_ROOT/track_checkpoint_10pct"
export TRAINING_SOURCES="$STUDY_ROOT/source_selection/sources.json"

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate count-hdf

python "$CT_DIR/norm42_train_10pct/prepare_track_checkpoint.py" \
  --training-sources "$TRAINING_SOURCES" \
  --seed 12345 --cohort train10pct_checkpoint \
  --output "$TRACK_ROOT"
```

Then define the established model and reconstruction inputs:

```bash
export MODEL_FULL="/oscar/data/mleblan6/mucoll/speng44/bib_gen_mother_muon/new_diffusion_norm42"
export MODEL_10PCT="$STUDY_ROOT/models"
export PAPER1_ROOT="$STUDY_ROOT/paper1_90bd576c"
export GENBIB_DIR="/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/GenBIB-ML"
export GEOMAP="/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_input_rebuild/maia_v2_11_sensor_geometry.npz"
export IMAGE_V2_11="/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif"
export IMAGE="/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif"
export BENCHMARK_DIR="/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks"
mkdir -p "$TRACK_ROOT/logs"
```

Queue the dependency chain:

```bash
CONDITIONS=$(sbatch --parsable --array=0-0 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$TRACK_ROOT" \
  -o "$TRACK_ROOT/logs/conditions_%A_%a.out" \
  -e "$TRACK_ROOT/logs/conditions_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_conditions.slurm")

SAMPLE=$(sbatch --parsable --dependency="afterok:$CONDITIONS" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$TRACK_ROOT",MODEL_FULL="$MODEL_FULL",MODEL_10PCT="$MODEL_10PCT",PAPER1_ROOT="$PAPER1_ROOT",SEED_BASE=12345,MAX_ROUNDS=100 \
  -o "$TRACK_ROOT/logs/sample_%A_%a.out" \
  -e "$TRACK_ROOT/logs/sample_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/submit_track_sampling.slurm")

INPUT=$(sbatch --parsable --dependency="afterok:$SAMPLE" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$TRACK_ROOT",GENBIB_DIR="$GENBIB_DIR",GEOMAP="$GEOMAP",IMAGE_V2_11="$IMAGE_V2_11" \
  -o "$TRACK_ROOT/logs/input_%j.out" \
  -e "$TRACK_ROOT/logs/input_%j.err" \
  "$CT_DIR/norm42_train_10pct/submit_track_inputs.slurm")

DIGI=$(sbatch --parsable --dependency="afterok:$INPUT" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$TRACK_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR" \
  -o "$TRACK_ROOT/logs/digi_%A_%a.out" \
  -e "$TRACK_ROOT/logs/digi_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/submit_track_digi.slurm")

SPLIT=$(sbatch --parsable --dependency="afterok:$DIGI" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$TRACK_ROOT",IMAGE="$IMAGE" \
  -o "$TRACK_ROOT/logs/split_%A_%a.out" \
  -e "$TRACK_ROOT/logs/split_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/submit_track_split.slurm")

RECO=$(sbatch --parsable --dependency="afterok:$SPLIT" \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$TRACK_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR" \
  -o "$TRACK_ROOT/logs/reco_%A_%a.out" \
  -e "$TRACK_ROOT/logs/reco_%A_%a.err" \
  "$CT_DIR/norm42_train_10pct/submit_track_reco.slurm")

printf 'conditions=%s sampling=%s input=%s digi=%s split=%s reco=%s\n' \
  "$CONDITIONS" "$SAMPLE" "$INPUT" "$DIGI" "$SPLIT" "$RECO"
```

After reconstruction completes:

```bash
conda activate count-hdf
python "$CT_DIR/norm42_train_10pct/track_checkpoint_report.py" \
  --scan-root "$TRACK_ROOT" \
  --output "$TRACK_ROOT/track_checkpoint_report.json"
```

## Nine-event regional follow-up cohort

The completed checkpoint supplies one event.  This follow-up prepares nine
new events under a different cohort label, giving ten events in total without
duplicating the checkpoint's deterministic draw.  Every event independently
draws 167 MUPLUS and 167 MUMINUS files without replacement within a polarity.
Sources may be reused between events, matching the established event-building
procedure.  All events remain disjoint from the union of physical cycles used
to train the 10%-corpus model.  `source_overlap.json` records every pairwise
overlap rather than treating the events as source-disjoint from one another.

Input construction is parallelized by event and arm.  Each task still invokes
the same chunked CellID assignment and v2.11 EDM4hep writer; this changes only
scheduling.  SIM extraction also opens each source ROOT file once for all six
tracker collections instead of reopening it once per collection.  Neither
optimization changes hit selection, ordering, CellID decisions, or output
values.

Prepare the follow-up manifest in `count-hdf`:

```bash
export TRACK_COHORT_ROOT="$STUDY_ROOT/track_cohort_10pct_additional9"

python "$CT_DIR/norm42_train_10pct/prepare_track_checkpoint.py" \
  --training-sources "$TRAINING_SOURCES" \
  --seed 12345 --cohort train10pct_regions \
  --events 9 \
  --output "$TRACK_COHORT_ROOT"

python -m json.tool "$TRACK_COHORT_ROOT/source_overlap.json" | less
```

With the model, image, geometry, and benchmark variables from the checkpoint
section still exported, queue the complete dependency chain:

```bash
export SCAN_ROOT="$TRACK_COHORT_ROOT"
bash "$CT_DIR/norm42_train_10pct/queue_track_cohort.sh"
```

The default limits run at most six input arms concurrently and ten theta-region
reconstruction tasks concurrently.  Override only the scheduling limits, not
the physics, with `INPUT_CONCURRENCY`, `DIGI_CONCURRENCY`,
`SPLIT_CONCURRENCY`, or `RECO_CONCURRENCY` before invoking the queue script.

After completion, summarize the nine new events:

```bash
conda activate count-hdf
python "$CT_DIR/norm42_train_10pct/track_checkpoint_report.py" \
  --scan-root "$TRACK_COHORT_ROOT" \
  --output "$TRACK_COHORT_ROOT/track_cohort_report.json"
```

The report contains per-event counts, aggregate counts, and the mean, sample
standard deviation, range, and individual COUNT/SIM ratios in every theta
region.  Combine the original checkpoint as the tenth event without rerunning
it by repeating `--scan-root`:

```bash
python "$CT_DIR/norm42_train_10pct/track_checkpoint_report.py" \
  --scan-root "$STUDY_ROOT/track_checkpoint_10pct" \
  --scan-root "$TRACK_COHORT_ROOT" \
  --output "$TRACK_COHORT_ROOT/combined_10_event_report.json"
```

## Local tests

```bash
cd "$CT_DIR/norm42_train_10pct"
python -m unittest -v test_workflow.py
```
