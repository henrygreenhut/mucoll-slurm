# COUNT tracker v2.11 diagnostic

This directory is a separate compatibility path for repeating one prepared
SIM/COUNT tracker event with the Muon Collider v2.11 image.  It does not alter
the v3 workflow in `../run_count_tracker_event.sh` or any GenBIB-ML file.

The v2.11 podio runtime cannot read the newer ROOT files directly.  The usable
one-event diagnostic therefore extracts the numerical BIB hit fields with
uproot, writes BIB-only inputs with v2.11, and runs the old digitization and
tracking.  No neutrino signal is needed because it contributes no tracker
hits.

## What remains the same

- The same six tracker SimTrackerHit collections enter digitization.
- SIM and COUNT use the same prepared BIB event, with no signal record.
- Vertex resolutions are 5 micrometres in `u` and `v`, with 30 ps timing.
- Inner/outer resolutions are 7 micrometres in `u`, 90 micrometres in `v`,
  with 60 ps timing.
- Digitization corrects arrival times for propagation and applies
  `[-3 sigma_t, 5 sigma_t]`: `[-0.09, 0.15] ns` in the vertex and
  `[-0.18, 0.30] ns` in the inner/outer trackers.
- `IsStrip=False` and `ForceHitsOntoSurface=True`.
- CKF writes `AllTracks`; duplicate removal writes `SiTracks`.

## Changes required by v2.11

1. The image is
   `/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif`.
2. v2.11 exposes digitization and CKF through `MarlinProcessorWrapper`, so
   explicit EDM4hep-to-LCIO and LCIO-to-EDM4hep converter tools surround the
   same tracker stages.  The steering configures the application at module
   scope because `k4run` loads it as a configuration file.  Its first Marlin
   processor is `AIDAProcessor`, as required by this release; that processor
   also hosts the EDM4hep-to-LCIO input converter.
3. CKF uses the v2.11 `ACTSSeededCKFTrackingProc` interface and the image's
   `MAIA_v0.root`, `MAIA_v0.json`, and `MAIA_v0_material.json`.
4. MAIA seeding layers come from `mucoll-benchmarks` commit `1b702ec` (the
   historical MAIA tracking configuration immediately preceding the v2.11
   image).  Common v3/v2 settings remain explicit: chi2 cutoff 10, minimum
   seed pT 500 MeV, maximum seed radius 150 mm, impact maximum 3 mm,
   collision region 6 mm, and scattering sigma 50.  The old interface uses
   one candidate per surface (`CKF_NumMeasurementsCutOff=1`), as in that
   historical configuration; the current v3 file uses 2.
5. Branch-stopper, outlier-cutoff, and native merged-hit options in the v3 CKF
   API do not exist in the v2.11 Marlin processor.  The v2 processor receives
   the six hit collections directly.
6. SIM and COUNT CellIDs use a map built from the v2.11 MAIA geometry.  The
   image lacks SciPy, so map construction runs in the image and the unchanged
   GenBIB assignment algorithm runs with the existing `genbib` Python.

The v0.9 benchmark tag remains the reference for the older Marlin workflow
and timing configuration, but it cannot be executed unchanged: its hard-coded
release-2.8 `MuColl_v1` ACTS paths are replaced by the native v2.11 MAIA files.
The later MAIA-specific seeding list is taken from commit `1b702ec` rather
than guessed.

## Unsupported exact-file diagnostic

`run_existing_pair_v2_11.sh` records the attempted byte-for-byte test, but the
old podio runtime sees zero events in these newer files.  Do not use or queue
this path for physics results.  `run_reco_v2_11.sh` now rejects the missing
output rather than reporting a false success.

```bash
export REPO="$(git rev-parse --show-toplevel)"
export CT_DIR="$REPO/count_tracker"
export IMAGE_V2_11=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif

export EVENT_ID=norm42_cached_closure100_test_000000
export V3_EVENT=/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_cached_closure100/events/$EVENT_ID
export V2_EVENT=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_exact_input/$EVENT_ID

bash "$CT_DIR/v2_11/run_existing_pair_v2_11.sh" \
  --input-event "$V3_EVENT" \
  --output "$V2_EVENT"
```

To queue the same command instead of holding an interactive allocation:

```bash
mkdir -p "$(dirname "$V2_EVENT")/logs"
sbatch \
  --export=ALL,CT_DIR="$CT_DIR",INPUT_EVENT="$V3_EVENT",OUTPUT="$V2_EVENT",IMAGE_V2_11="$IMAGE_V2_11" \
  -o "$(dirname "$V2_EVENT")/logs/v2_11_%j.out" \
  -e "$(dirname "$V2_EVENT")/logs/v2_11_%j.err" \
  "$CT_DIR/v2_11/submit_existing_pair_v2_11.slurm"
```

This command is retained only as documentation of the failed compatibility
check.

## Full v2.11 input rebuild

Run from the `mucoll-slurm` repository on an allocated OSCAR CPU node after
activating `genbib`, which provides NumPy and SciPy for CellID assignment:

```bash
module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate genbib

export REPO="$(git rev-parse --show-toplevel)"
export CT_DIR="$REPO/count_tracker"
export GENBIB_DIR=/path/to/GenBIB-ML
export IMAGE_V2_11=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif
export ASSIGN_PYTHON="$(command -v python)"
conda deactivate
conda activate count-hdf
export EXTRACT_PYTHON="$(command -v python)"

bash "$CT_DIR/v2_11/run_count_tracker_event_v2_11.sh" \
  --conditions /path/to/conditions \
  --construction norm42 \
  --split test \
  --event-id norm42_cached_closure100_test_000000 \
  --count-samples /path/to/count_samples \
  --geomap /oscar/scratch/$USER/mucoll/count_tracker_v2_11/maia_v2_11_sensor_geometry.npz \
  --output /oscar/scratch/$USER/mucoll/count_tracker_v2_11/event_000000
```

If the geometry map does not exist, the event wrapper creates it once.  The
newer podio files cannot be read by v2.11, so the wrapper reads only numerical
BIB hit branches through uproot, saves them as NumPy arrays, and writes fresh
BIB-only records with the v2.11 runtime.  Each record contains one neutral
`EventHeader` (`run=0`, `event=0`, `timestamp=0`) required by the v2.11
EDM4hep-to-LCIO converter.  No neutrino signal is included; it has no tracker
hits and is irrelevant to this diagnostic.  Both SIM and COUNT hits are
assigned CellIDs from their XYZ positions with the same v2.11 map and the same
assignment algorithm.  Any assignment losses are recorded for each arm.  Keep
this output directory separate from both the v3 event and the failed
exact-input v2 diagnostic.

Queue the BIB-only rebuild with a dedicated v2.11 geometry map:

```bash
export CONDITIONS=/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_cached_closure100/conditions
export COUNT_SAMPLES=/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_cached_closure100/count_samples
export GEOMAP=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_input_rebuild/maia_v2_11_sensor_geometry.npz
export REBUILD_EVENT=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_input_rebuild/$EVENT_ID
mkdir -p "$(dirname "$REBUILD_EVENT")/logs"

sbatch \
  --export=ALL,CT_DIR="$CT_DIR",GENBIB_DIR="$GENBIB_DIR",CONDITIONS="$CONDITIONS",CONSTRUCTION=norm42,SPLIT=test,EVENT_ID="$EVENT_ID",COUNT_SAMPLES="$COUNT_SAMPLES",GEOMAP="$GEOMAP",OUTPUT="$REBUILD_EVENT",IMAGE_V2_11="$IMAGE_V2_11" \
  -o "$(dirname "$REBUILD_EVENT")/logs/v2_11_rebuild_%j.out" \
  -e "$(dirname "$REBUILD_EVENT")/logs/v2_11_rebuild_%j.err" \
  "$CT_DIR/v2_11/submit_input_rebuild_v2_11.slurm"
```

This is not a new Geant4 simulation: the hit values come from the existing SIM
and COUNT samples.  It is a symmetric v2.11 geometry assignment followed by a
BIB-only digitization/reconstruction comparison under the old stack.

Compare entering tracker hits, digitized tracker hits and survival by
collection, `AllTracks`, and deduplicated `SiTracks` for the same BIB event.
Record the SIM and COUNT losses from v2.11 CellID assignment alongside those
reconstruction results.

## Full-density five-region reconstruction

For the full-density diagnostic, do not run all digitized hits through one CKF
process.  Follow the Paper 1 procedure: digitize each arm once, then run CKF
independently in the five documented polar-angle regions `[0,30)`, `[30,70)`,
`[70,110)`, `[110,150)`, and `[150,180)` degrees.  The splitter acts on the
digitized hit position and carries the matching simulated hits and relations
into each regional process.

The implementation follows `osg/steerBIBtracking.py` from
`rmastand/fast_tracker_BIBgen_mucoll` at commit
`984b45ed3e646dc6ba8f6795cc187c1005c08495`.  The installed v2.11 image's
Marlin processor interface was checked directly before adding the steering.

This path changes only the CKF input collections.  It retains the established
v2.11 diagnostic's geometry, timing, seeding, CKF, and duplicate-removal
settings so density and regional processing are not mixed with another
configuration change.  The unsplit path remains the default when no theta
bounds are supplied.

Set the completed full-density paths on OSCAR:

```bash
export REPO="$(git rev-parse --show-toplevel)"
export CT_DIR="$REPO/count_tracker"
export GENBIB_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/GenBIB-ML
export IMAGE_V2_11=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif

export BASE=/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_fullbx1666
export CONDITIONS="$BASE/conditions"
export COUNT_SAMPLES="$BASE/count_samples"
export EVENT_ID=norm42_fullbx1666_test_000000
export EVENT_ROOT="$BASE/v2_11_theta/$EVENT_ID"
export GEOMAP=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_input_rebuild/maia_v2_11_sensor_geometry.npz
mkdir -p "$BASE/v2_11_theta/logs"
```

Before the production submission, exercise the complete regional path using
an existing prepared v2.11 pair.  This digitizes only 128 hits per collection
and then checks all five regional outputs:

```bash
export SMOKE_SOURCE=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_cohort/events/norm42_cached_closure100_test_000000
export THETA_SMOKE=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_theta_smoke_01
bash "$CT_DIR/v2_11/run_theta_smoke_v2_11.sh" \
  --input-event "$SMOKE_SOURCE" \
  --sample SIM \
  --hits-per-collection 128 \
  --output "$THETA_SMOKE"
```

Use a new smoke output directory for another attempt.  Proceed to the
full-density jobs only after the smoke report says `exact count partition`.

First prepare the two v2.11-native BIB inputs.  The command overrides the
smaller cohort defaults because this event contains about 38 million hits:

```bash
PREP_JOB=$(sbatch --parsable \
  --mem=64G --time=24:00:00 \
  --export=ALL,CT_DIR="$CT_DIR",GENBIB_DIR="$GENBIB_DIR",CONDITIONS="$CONDITIONS",CONSTRUCTION=norm42,SPLIT=test,EVENT_ID="$EVENT_ID",COUNT_SAMPLES="$COUNT_SAMPLES",GEOMAP="$GEOMAP",OUTPUT="$EVENT_ROOT",IMAGE_V2_11="$IMAGE_V2_11",STOP_AFTER=input \
  -o "$BASE/v2_11_theta/logs/prepare_%j.out" \
  -e "$BASE/v2_11_theta/logs/prepare_%j.err" \
  "$CT_DIR/v2_11/submit_input_rebuild_v2_11.slurm")
echo "prepare=$PREP_JOB"
```

Digitize SIM and COUNT in parallel after input preparation succeeds:

```bash
DIGI_JOB=$(sbatch --parsable \
  --dependency="afterok:$PREP_JOB" --array=0-1 \
  --export=ALL,CT_DIR="$CT_DIR",EVENT_ROOT="$EVENT_ROOT",IMAGE_V2_11="$IMAGE_V2_11" \
  -o "$BASE/v2_11_theta/logs/digi_%A_%a.out" \
  -e "$BASE/v2_11_theta/logs/digi_%A_%a.err" \
  "$CT_DIR/v2_11/submit_digi_pair_v2_11.slurm")
echo "digitization=$DIGI_JOB"
```

After both digitization tasks succeed, run five SIM and five COUNT regional
jobs.  The array mapping is fixed: tasks 0--4 are SIM in increasing theta and
tasks 5--9 are COUNT in increasing theta.

```bash
THETA_JOB=$(sbatch --parsable \
  --dependency="afterok:$DIGI_JOB" --array=0-9 \
  --export=ALL,CT_DIR="$CT_DIR",EVENT_ROOT="$EVENT_ROOT",IMAGE_V2_11="$IMAGE_V2_11" \
  -o "$BASE/v2_11_theta/logs/theta_%A_%a.out" \
  -e "$BASE/v2_11_theta/logs/theta_%A_%a.err" \
  "$CT_DIR/v2_11/submit_theta_reco_v2_11.slurm")
echo "theta_reco=$THETA_JOB"
```

Each regional output is write-once and a completed output is skipped on
resubmission.  Failed regions can therefore be retried with only their array
indices.  The input and digitization stages are not repeated.

After all ten regional jobs complete, run the partition and result check from
an allocated node.  It verifies that the five regional hit counts sum exactly
to every unsplit digitized collection and reports digitization survival and
the summed `SiTracks` count:

```bash
for SAMPLE in SIM COUNT; do
  apptainer exec \
    --bind "$CT_DIR:$CT_DIR:ro,$EVENT_ROOT:$EVENT_ROOT" \
    "$IMAGE_V2_11" bash -lc '
      source /opt/setup_mucoll.sh
      python3 "$1/v2_11/validate_theta_partition_v2_11.py" \
        --digi-file "$2/$3/digi/digi_output.edm4hep.root" \
        --regional-root "$2/$3/theta_reco" \
        --output "$2/$3/theta_report.json"
    ' _ "$CT_DIR" "$EVENT_ROOT" "$SAMPLE"
done
```

## Read-only preflight

From an allocated OSCAR CPU node, check the image, detector assets, installed
Marlin processors, and both `k4run` configurations without creating output or
processing an event:

```bash
export INPUT_FILE=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_input_rebuild/norm42_cached_closure100_test_000000/SIM/input/input.edm4hep.root
bash "$CT_DIR/v2_11/preflight_v2_11.sh"
```

This catches missing processors or assets, a missing `EventHeader`, a steering
file that is not loaded, and incorrect algorithm ordering.  It cannot catch
failures that occur only while processing the event; the one-event batch
checkpoint remains necessary for those.

## Interactive integration smoke

Use the smoke harness while developing so changes can be tested before they
are committed.  From the laptop, copy the complete `count_tracker` directory
to an isolated OSCAR scratch directory:

```bash
export DEV_ROOT=/oscar/scratch/hgreenhu/mucoll/count_tracker_v2_11_dev
ssh oscar "mkdir -p '$DEV_ROOT/count_tracker'"
rsync -a mucoll-slurm/count_tracker/ oscar:"$DEV_ROOT/count_tracker/"
```

On OSCAR, request one interactive CPU node and prepare fresh v2-native inputs
without starting digitization:

```bash
srun --partition=batch --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash

module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"

export DEV_ROOT=/oscar/scratch/$USER/mucoll/count_tracker_v2_11_dev
export CT_DIR="$DEV_ROOT/count_tracker"
export GENBIB_DIR=/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/GenBIB-ML
export IMAGE_V2_11=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif
export EVENT_ID=norm42_cached_closure100_test_000000
export CONDITIONS=/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_cached_closure100/conditions
export COUNT_SAMPLES=/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_cached_closure100/count_samples
export GEOMAP=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_input_rebuild/maia_v2_11_sensor_geometry.npz
export PREPARED=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_smoke_prepared/$EVENT_ID
export SMOKE=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_smoke_run/$EVENT_ID
export SMOKE_LOG=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_smoke_run/logs/$EVENT_ID.log
mkdir -p "$(dirname "$SMOKE_LOG")"

conda activate genbib
export ASSIGN_PYTHON="$(command -v python)"
conda deactivate
conda activate count-hdf
export EXTRACT_PYTHON="$(command -v python)"

bash "$CT_DIR/v2_11/run_count_tracker_event_v2_11.sh" \
  --conditions "$CONDITIONS" \
  --construction norm42 \
  --split test \
  --event-id "$EVENT_ID" \
  --count-samples "$COUNT_SAMPLES" \
  --geomap "$GEOMAP" \
  --output "$PREPARED" \
  --stop-after input
```

Run the actual digitization and CKF stages on 128 deterministic hits from each
tracker collection.  Start with SIM; use `--sample both` after the SIM smoke
passes:

```bash
bash "$CT_DIR/v2_11/run_smoke_v2_11.sh" \
  --input-event "$PREPARED" \
  --output "$SMOKE" \
  --sample SIM \
  --hits-per-collection 128 \
  2>&1 | tee "$SMOKE_LOG"
```

The harness validates the input, digitized output, and reconstruction output
in a new process after each stage.  The reduced input writer copies numerical
values into fresh EDM4hep objects rather than retaining objects owned by its
source podio reader.  `run_reco_v2_11.sh --stage digi` and `--stage reco` can also
be called separately to resume at the failed stage.  Each output directory is
write-once; use a new directory for another attempt so partial files cannot be
mistaken for successful output.

The installed v2.11 stack can terminate successfully and finalize a readable
podio file, then abort in allocator cleanup.  The reconstruction wrapper
handles only that narrow case: Bash status 134 (`SIGABRT`; SLURM reports the
same signal as `6:0`) is accepted only after a fresh podio reader verifies the
`EventHeader` and all required collections and types for that stage.  Any
other nonzero status, or any output that fails validation, remains fatal.
