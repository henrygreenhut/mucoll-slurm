# COUNT tracker v2.11 diagnostic

This directory is a separate compatibility path for repeating one prepared
SIM/COUNT tracker event with the Muon Collider v2.11 image.  It does not alter
the v3 workflow in `../run_count_tracker_event.sh` or any GenBIB-ML file.

The intended first use is a one-event software-version diagnostic using the
**exact existing pre-digitization SIM and COUNT ROOT files**.  This holds all
upstream choices fixed and changes only the image and tracker implementation.
A second wrapper can rebuild the event with a v2.11 CellID map if the narrow
test shows that a fully geometry-coherent v2 study is useful.

## What remains the same

- The same six tracker SimTrackerHit collections enter digitization.
- SIM and COUNT use the same prepared event and signal record.
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
6. COUNT CellIDs use a map built from the v2.11 MAIA geometry.  The v2.11
   image lacks SciPy, so map construction runs in the image and the unchanged
   GenBIB assignment algorithm runs with the existing `count-hdf` Python.

The v0.9 benchmark tag remains the reference for the older Marlin workflow
and timing configuration, but it cannot be executed unchanged: its hard-coded
release-2.8 `MuColl_v1` ACTS paths are replaced by the native v2.11 MAIA files.
The later MAIA-specific seeding list is taken from commit `1b702ec` rather
than guessed.

## Recommended one-event command: change only reconstruction

Point `V3_EVENT` at one completed event from the cached norm42 cohort.  It must
contain both `SIM/input/input.edm4hep.root` and
`COUNT/input/input.edm4hep.root`.

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

This command does not build a new signal, reassign CellIDs, or rewrite either
input record.  If v2.11 cannot read the newer EDM4hep file schema, it will fail
before reconstruction; do not silently convert the file because that would add
another changed axis.

## Full v2.11 input rebuild

Run from the `mucoll-slurm` repository on an allocated OSCAR CPU node after
activating `count-hdf`:

```bash
module load miniforge3/25.3.0-3-a6hh
eval "$(conda shell.bash hook)"
conda activate count-hdf

export REPO="$(git rev-parse --show-toplevel)"
export CT_DIR="$REPO/count_tracker"
export GENBIB_DIR=/path/to/GenBIB-ML
export IMAGE_V2_11=/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24_v2.11-amd64.sif
export ASSIGN_PYTHON="$(command -v python)"

bash "$CT_DIR/v2_11/run_count_tracker_event_v2_11.sh" \
  --conditions /path/to/conditions \
  --construction norm42 \
  --split test \
  --event-id norm42_cached_closure100_test_000000 \
  --signal /path/to/neutrino_sim.edm4hep.root \
  --signal-entry 0 \
  --count-samples /path/to/count_samples \
  --geomap /oscar/scratch/$USER/mucoll/count_tracker_v2_11/maia_v2_11_sensor_geometry.npz \
  --output /oscar/scratch/$USER/mucoll/count_tracker_v2_11/event_000000
```

If the geometry map does not exist, the event wrapper creates it once.  This
full rebuild changes COUNT CellID assignment to the v2.11 geometry and writes
both input records with the old EDM4hep runtime.  Existing SIM hits retain
their stored CellIDs, so their compatibility with the v2.11 MAIA map must be
audited before treating this as a production comparison.  Keep this output
directory separate from both the v3 event and the exact-input v2 diagnostic.

It can be queued alongside the exact-input job with a distinct output and a
dedicated v2.11 geometry map:

```bash
export CONDITIONS=/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_cached_closure100/conditions
export COUNT_SAMPLES=/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_cached_closure100/count_samples
export SIGNAL=/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_cached_closure100/signal/signal.root
export GEOMAP=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_input_rebuild/maia_v2_11_sensor_geometry.npz
export REBUILD_EVENT=/oscar/scratch/$USER/mucoll/count_tracker_cmp/v2_11_input_rebuild/$EVENT_ID
mkdir -p "$(dirname "$REBUILD_EVENT")/logs"

sbatch \
  --export=ALL,CT_DIR="$CT_DIR",GENBIB_DIR="$GENBIB_DIR",CONDITIONS="$CONDITIONS",CONSTRUCTION=norm42,SPLIT=test,EVENT_ID="$EVENT_ID",SIGNAL="$SIGNAL",SIGNAL_ENTRY=80,COUNT_SAMPLES="$COUNT_SAMPLES",GEOMAP="$GEOMAP",OUTPUT="$REBUILD_EVENT",IMAGE_V2_11="$IMAGE_V2_11" \
  -o "$(dirname "$REBUILD_EVENT")/logs/v2_11_rebuild_%j.out" \
  -e "$(dirname "$REBUILD_EVENT")/logs/v2_11_rebuild_%j.err" \
  "$CT_DIR/v2_11/submit_input_rebuild_v2_11.slurm"
```

Use the exact signal file and entry recorded for the original event.  Although
this is test event zero, its recorded signal entry is 80 because the signal
file also contains the preceding train and validation records.  This job is
intentionally not called a fully re-simulated v2 event: the SIM hits were
produced previously and their stored CellIDs are preserved.

For either test, compare entering tracker hits, digitized tracker hits and
survival by collection, `AllTracks`, and deduplicated `SiTracks` for the same
event.  The exact-input test is the clean answer to whether the software
version changes the multiplicity ratio.
