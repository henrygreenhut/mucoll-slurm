# Norm42 v3 five-region occupancy scan

This scan measures the reconstructed `COUNT/SIM` SiTrack ratio as the amount
of BIB in one reconstruction event changes.  It preserves the established
full-occupancy configuration:

- Norm42 SIM bank and retrained Norm42 COUNT model;
- independent MUPLUS and MUMINUS source selections without replacement;
- raw `SimTrackerHit.time < 1e7 ns` model-domain selection;
- matched per-sensor SIM and COUNT requested occupancies;
- BIB-only input construction used by the successful full-occupancy run;
- v3 digitization followed by native v3 partitioning into `[0,30)`, `[30,70)`,
  `[70,110)`, `[110,150)`, and `[150,180]` degrees;
- independent v3 CKF reconstruction in each region and summed `SiTracks`.

Each occupancy contains five paired events.  Within a replicate, lower
occupancies are strict prefixes of the same independently shuffled MUPLUS and
MUMINUS source lists used at higher occupancies.  This reduces source-sampling
noise in the occupancy comparison.  Different replicates use independent
deterministic streams, and source files may recur between replicates.
The source inventory excludes known-invalid cycle 6291 and restricts both
polarities to their shared set of valid cycle IDs, matching the historical
pool construction.

The log-spaced scan uses 10, 36, 129, 464, and 1,666 actual Norm42 files per
polarity.  Relative to the established 1,666-file full-BX reference, these are
0.60%, 2.16%, 7.74%, 27.85%, and 100%.  Since every Norm42 file represents 42
Norm1 equivalents, the corresponding per-polarity normalizations are 420,
1,512, 5,418, 19,488, and 69,972 Norm1-file equivalents.

## Prepare manifests

On OSCAR, pull the reviewed commit and set paths:

```bash
cd ~/mydir/mucoll/mucoll-slurm
git pull --ff-only

export REPO="$(git rev-parse --show-toplevel)"
export CT_DIR="$REPO/count_tracker"
export SCAN_ROOT="/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_v3_occupancy_scan"
export SIM_ROOT="/oscar/data/mleblan6/mucoll/bib/SIM"

export MODEL_ROOT="/oscar/data/mleblan6/mucoll/speng44/bib_gen_mother_muon/new_diffusion_norm42"
export PAPER1_ROOT="/oscar/data/mleblan6/mucoll/speng44/bib_gen_model/ddpm_outputs/tabddpm/local_phi/paper1-inference"
export GENBIB_DIR="/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/GenBIB-ML"
export GEOMAP="/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_input_rebuild/maia_v2_11_sensor_geometry.npz"
export IMAGE_V2_11="/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif"
export IMAGE="/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif"
export BENCHMARK_DIR="/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks"

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate count-hdf
python "$CT_DIR/v3_occupancy_scan/prepare_scan.py" \
  --sim-root "$SIM_ROOT" \
  --file-counts 10 36 129 464 1666 \
  --events 5 --seed 12345 --cohort occupancy_scan \
  --output "$SCAN_ROOT"

mkdir -p "$SCAN_ROOT/logs"
```

Preparation refuses to replace an existing scan.  Inspect `scan.json` and the
five `sources.json` files before submitting compute jobs.

## Submit the dependency chain

The array limits keep at most two GPU samplers, two large input builders, four
digitizers/splitters, and ten CKF regions active simultaneously.  Every stage
is resumable and refuses to treat a partial output as complete.

```bash
COND=$(sbatch --parsable --array=0-4%2 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT" \
  -o "$SCAN_ROOT/logs/conditions_%A_%a.out" \
  -e "$SCAN_ROOT/logs/conditions_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_conditions.slurm")

SAMPLE=$(sbatch --parsable --dependency="afterok:$COND" --array=0-4%2 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",MODEL_ROOT="$MODEL_ROOT",PAPER1_ROOT="$PAPER1_ROOT",SEED_BASE=12345,MAX_ROUNDS=100,UNFILLED_POLICY=drop \
  -o "$SCAN_ROOT/logs/sample_%A_%a.out" \
  -e "$SCAN_ROOT/logs/sample_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_sampling.slurm")

INPUT=$(sbatch --parsable --dependency="afterok:$SAMPLE" --array=0-24%2 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",GENBIB_DIR="$GENBIB_DIR",GEOMAP="$GEOMAP",IMAGE_V2_11="$IMAGE_V2_11",EVENTS_PER_POINT=5 \
  -o "$SCAN_ROOT/logs/input_%A_%a.out" \
  -e "$SCAN_ROOT/logs/input_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_inputs.slurm")

DIGI=$(sbatch --parsable --dependency="afterok:$INPUT" --array=0-49%4 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR",EVENTS_PER_POINT=5 \
  -o "$SCAN_ROOT/logs/digi_%A_%a.out" \
  -e "$SCAN_ROOT/logs/digi_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_digi.slurm")

SPLIT=$(sbatch --parsable --dependency="afterok:$DIGI" --array=0-249%4 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",IMAGE="$IMAGE",EVENTS_PER_POINT=5 \
  -o "$SCAN_ROOT/logs/split_%A_%a.out" \
  -e "$SCAN_ROOT/logs/split_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_split.slurm")

RECO=$(sbatch --parsable --dependency="afterok:$SPLIT" --array=0-249%10 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR",EVENTS_PER_POINT=5 \
  -o "$SCAN_ROOT/logs/reco_%A_%a.out" \
  -e "$SCAN_ROOT/logs/reco_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_reco.slurm")

printf 'conditions=%s sampling=%s inputs=%s digitization=%s split=%s reco=%s\n' \
  "$COND" "$SAMPLE" "$INPUT" "$DIGI" "$SPLIT" "$RECO"
```

This is a large production chain: 25 paired events, 50 digitizations, 250
theta-split files, and 250 regional CKF runs.  Check available scratch quota
before submission.

## Report

After the final array completes:

```bash
module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate count-hdf
python "$CT_DIR/v3_occupancy_scan/report_scan.py" \
  --scan-root "$SCAN_ROOT" \
  --output "$SCAN_ROOT/occupancy_report.json"
```

The report retains paired per-event counts and prints, for every occupancy,
the total SIM and COUNT tracks, `COUNT/SIM` ratio, and digitization-survival
fractions.

## Unsplit-CKF control

The five theta regions exactly partition the digitized hits, but independent
CKF invocations do not have the same combinatorics as one CKF invocation over
their union.  The unsplit control reuses the exact digitized SIM and COUNT
files above and changes only whether the ordinary v3 tracker-hit mergers feed
one CKF or five regional CKFs.  Its outputs are isolated under
`events/<event>/<arm>/unsplit_reco`.

Start with the 10- and 36-file points (array indices 0--19):

```bash
UNSPLIT_LOW=$(sbatch --parsable --array=0-19%4 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR",EVENTS_PER_POINT=5 \
  -o "$SCAN_ROOT/logs/reco_unsplit_low_%A_%a.out" \
  -e "$SCAN_ROOT/logs/reco_unsplit_low_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_reco_unsplit.slurm")
```

If these finish cleanly, run the 129-file point:

```bash
UNSPLIT_129=$(sbatch --parsable --array=20-29%4 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR",EVENTS_PER_POINT=5 \
  -o "$SCAN_ROOT/logs/reco_unsplit_129_%A_%a.out" \
  -e "$SCAN_ROOT/logs/reco_unsplit_129_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_reco_unsplit.slurm")
```

For the 464-file point, first reconstruct only its first paired event.  Tasks
30 and 31 are respectively its SIM and COUNT arms:

```bash
UNSPLIT_464_SMOKE=$(sbatch --parsable --array=30-31%2 \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR",EVENTS_PER_POINT=5 \
  -o "$SCAN_ROOT/logs/reco_unsplit_464_smoke_%A_%a.out" \
  -e "$SCAN_ROOT/logs/reco_unsplit_464_smoke_%A_%a.err" \
  "$CT_DIR/v3_occupancy_scan/submit_reco_unsplit.slurm")
```

After validating the smoke runtime and memory, tasks 32--39 complete the
remaining four paired events.  The full-BX point is intentionally excluded
from this control unless the lower-occupancy runtime supports attempting it.

Compare completed points with the five-region results:

```bash
module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate count-hdf
python "$CT_DIR/v3_occupancy_scan/report_unsplit.py" \
  --scan-root "$SCAN_ROOT" --point-indices 0 1 \
  --output "$SCAN_ROOT/unsplit_comparison_10_36.json"
```

## Full-BX angular-patch PFN

The preliminary full-BX classifier uses the completed `n1666` five-region
reconstruction.  It does not rerun reconstruction.  Every regional `SiTrack`
is assigned exactly once to one of 72 fixed five-degree sectors using its
fitted AtIP momentum phi.  The stored angular input is delta-phi from the
sector centre.  Empty sectors are retained.  The first three parent event
pairs are training, the fourth is validation, and the fifth is test; every
patch from a parent remains in the same split.

Build the stores on a compute node in `count-hdf`:

```bash
export REPO="$(git rev-parse --show-toplevel)"
export CT_DIR="$REPO/count_tracker"
export SCAN_ROOT="/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_v3_occupancy_scan_03"
export PATCH_ROOT="$SCAN_ROOT/n1666/angular_patches_phi72"

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate count-hdf

python "$CT_DIR/v3_occupancy_scan/build_angular_patches.py" \
  --scan-root "$SCAN_ROOT" \
  --point n1666 \
  --phi-sectors 72 \
  --split-counts 3 1 1 \
  --output "$PATCH_ROOT"
```

This writes 1,080/360/360 patches per arm in train/validation/test.  The
manifests record the parent event, theta region, global phi-sector boundary,
sector centre, local-phi definition, ROOT source, and track count for every
patch.  The builder verifies that partitioning preserves the complete summed
five-region track count in each arm.

Submit the physical SIM-versus-COUNT PFN in the established TensorFlow,
EnergyFlow, and `tf_keras` environment:

```bash
export PFN_OUT="$SCAN_ROOT/n1666/angular_patch_pfn"
mkdir -p "$PFN_OUT/logs"

TRAIN_JOB=$(sbatch --parsable \
  --export=ALL,CT_DIR="$CT_DIR",PATCH_ROOT="$PATCH_ROOT",PFN_OUT="$PFN_OUT" \
  -o "$PFN_OUT/logs/train_%j.out" \
  -e "$PFN_OUT/logs/train_%j.err" \
  "$CT_DIR/v3_occupancy_scan/submit_angular_pfn.slurm")
echo "training=$TRAIN_JOB"
```

## Reduced-chi-squared PFN variant

To isolate the value of fitted-track quality, rebuild the same angular patches
with the untransformed `AllTracks.chi2 / AllTracks.ndf` value appended to every
track.  This preserves the five parent events, 3/1/1 parent split, 72 phi
sectors, features, PFN architecture, optimizer, and training seed used above.
The baseline stores and results are not modified.

```bash
export PATCH_ROOT_CHI2="$SCAN_ROOT/n1666/angular_patches_phi72_chi2ndf"
export PFN_OUT="$SCAN_ROOT/n1666/angular_patch_pfn"
mkdir -p "$PATCH_ROOT_CHI2/logs" "$PFN_OUT/logs"

PATCH_JOB=$(sbatch --parsable \
  --export=ALL,CT_DIR="$CT_DIR",SCAN_ROOT="$SCAN_ROOT",PATCH_ROOT="$PATCH_ROOT_CHI2",POINT=n1666,PHI_SECTORS=72,INCLUDE_CHI2_NDF=1 \
  -o "$PATCH_ROOT_CHI2/logs/build_%j.out" \
  -e "$PATCH_ROOT_CHI2/logs/build_%j.err" \
  "$CT_DIR/v3_occupancy_scan/submit_angular_patches.slurm")

TRAIN_CHI2=$(sbatch --parsable \
  --dependency="afterok:$PATCH_JOB" \
  --export=ALL,CT_DIR="$CT_DIR",PATCH_ROOT="$PATCH_ROOT_CHI2",PFN_OUT="$PFN_OUT",CONSTRUCTION=norm42_fullbx_theta_phi72_chi2ndf,LABEL=fullbx_theta_phi72_chi2ndf_ceiling1000,EPOCHS=1000,PATIENCE=100 \
  -o "$PFN_OUT/logs/train_chi2_%j.out" \
  -e "$PFN_OUT/logs/train_chi2_%j.err" \
  "$CT_DIR/v3_occupancy_scan/submit_angular_pfn.slurm")

printf 'patches=%s training=%s\n' "$PATCH_JOB" "$TRAIN_CHI2"
```

The chi-squared input remains zero only in padding.  Valid tracks must have a
finite, nonnegative reduced chi-squared.  The new construction name and result
label prevent accidental mixing with the six-feature baseline.

The trainer recognizes the patch-to-parent mapping.  It never splits patches
from one parent across datasets, and its bootstrap grouping uses parent event
pairs rather than treating patches as independent events.  Because this fixed
split has only one test parent, no bootstrap interval is produced.  Independent
patch-label permutation is rejected; a null must be built from paired physical
SIM/SIM or COUNT/COUNT samples with the same patch procedure.
