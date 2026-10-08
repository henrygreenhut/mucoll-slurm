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
   same tracker stages.
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
BIB-only records with the v2.11 runtime.  No neutrino signal is included; it
has no tracker hits and is irrelevant to this diagnostic.  Both SIM and COUNT
hits are assigned CellIDs from their XYZ positions with the same v2.11 map and
the same assignment algorithm.  Any assignment losses are recorded for each
arm.  Keep this output directory separate from both the v3 event and the
failed exact-input v2 diagnostic.

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
