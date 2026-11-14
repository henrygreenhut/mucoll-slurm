# Norm42 10%-BX retraining study

This directory isolates the first, read-only preparation stage of the study.
It selects one immutable development corpus containing 167 Norm42 files per
polarity and audits whether that corpus covers the sensor conditions in the
complete flattened Norm42 arrays.  It does not create training arrays, train a
model, submit reconstruction, or choose a policy for unsupported sensors.

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

## Local tests

```bash
cd "$CT_DIR/norm42_train_10pct"
python -m unittest -v test_workflow.py
```
