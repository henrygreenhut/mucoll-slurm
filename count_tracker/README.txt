Reconstructed-track two-sample classifiers
==========================================

Direct norm1 mother-muon track comparison
-----------------------------------------
This comparison uses the original unrotated norm1 split-mother SIM files. It
draws from all 6666 complete source cycles and uses neither an analysis split
nor the diffusion model's saved train/validation split. It is a 20-event track
comparison diagnostic and is explicitly not a classifier-ready cohort.

Each source ROOT file is one FLUKA cycle and its event entries are individual
mother-muon groups. One comparison event draws 420 cycle IDs with replacement
for MUPLUS and, independently, 420 cycle IDs with replacement for MUMINUS. All
mother entries and all stored tracker hits in each draw are aggregated. A
cycle may therefore recur within an event or between events.

The empirical SIM arm and COUNT condition arm are derived from exactly the same
stored hits. No preprocessing timing, energy, or spatial selection is applied.
The normal digitizer later applies the same detector timing selection to both
arms. The manifest explicitly records that neither the model split nor an
analysis split was used and that this cohort is not a model-validation holdout.

Prepare all 20 event definitions on an OSCAR compute node. Event zero will be
the end-to-end checkpoint and remains the same event when the cohort is scaled:

    srun --partition=batch --cpus-per-task=4 --mem=32G --time=04:00:00 --pty bash
    module load miniforge3/25.3.0-3-a6hh
    eval "$(conda shell.bash hook)"
    conda activate count-hdf

    REPO="$(git rev-parse --show-toplevel)"
    CT_DIR="$REPO/count_tracker"
    BASE="/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm1_mother_direct20"
    SIM_ROOT="/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/bib-v3p0-fmt2-norm1-split-mother-norot/SIM"
    DATA_ROOT="/oscar/data/mleblan6/mucoll/speng44/bib_gen_mother_moun/data"

    python "$CT_DIR/count_tracker_mother_direct.py" \
      --sim-root "$SIM_ROOT" \
      --metadata "$DATA_ROOT/primary_muon_npy/metadata" \
      --cohort trackcmp --seed 12345 \
      --events 20 \
      --output "$BASE/conditions"

The event IDs are ``norm1_mother_direct_trackcmp_000000`` through
``...000019``. The output manifest records the complete source pool, every
ordered draw and source file, independent polarity streams, source metadata,
per-collection counts, and hashes of the prepared SIM and condition arrays.

Use the same model runtime, signal, geometry, and reconstruction checkout as
the successful earlier comparison. First sample and reconstruct only event
zero:

    CONDITIONS="$BASE/conditions"
    COUNT_SAMPLES="$BASE/count_samples"
    EVENTS_OUT="$BASE/events"
    MODEL_ROOT="/oscar/data/mleblan6/mucoll/speng44/bib_gen_mother_moun/new_diffusion"
    SIGNAL="/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_mother_muon20/signal/signal.root"
    GEOMAP="/oscar/scratch/$USER/mucoll/count_tracker_smoke/geomap/maia_cellid_sensor_geometry.npz"
    GENBIB_DIR="/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/GenBIB-ML"
    BENCHMARK_DIR="/oscar/data/mleblan6/mucoll/hgreenhu/mucoll/mucoll-benchmarks"
    IMAGE="/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif"
    PAPER1_ROOT="/oscar/data/mleblan6/mucoll/speng44/bib_gen_model/ddpm_outputs/tabddpm/local_phi/paper1-inference"
    EVENT0="norm1_mother_direct_trackcmp_000000"
    mkdir -p "$BASE/logs"

    SAMPLE0=$(sbatch --parsable \
      --export=ALL,CONDITIONS="$CONDITIONS",SPLIT=test,MODEL_ROOT="$MODEL_ROOT",OUTPUT="$COUNT_SAMPLES",CT_DIR="$CT_DIR",PAPER1_ROOT="$PAPER1_ROOT",EVENT_ID="$EVENT0",MAX_ROUNDS=100,UNFILLED_POLICY=drop \
      -o "$BASE/logs/sample0_%j.out" -e "$BASE/logs/sample0_%j.err" \
      "$CT_DIR/submit_count_tracker_sample.slurm")

    RECO0=$(sbatch --parsable --dependency="afterok:$SAMPLE0" --array=0-0 \
      --export=ALL,CONDITIONS="$CONDITIONS",CONSTRUCTION=norm1_mother_direct,SPLIT=test,SIGNAL="$SIGNAL",COUNT_SAMPLES="$COUNT_SAMPLES",GEOMAP="$GEOMAP",EVENTS_OUT="$EVENTS_OUT",CT_DIR="$CT_DIR",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR",GENBIB_DIR="$GENBIB_DIR",NEV=1 \
      -o "$BASE/logs/reco0_%A_%a.out" -e "$BASE/logs/reco0_%A_%a.err" \
      "$CT_DIR/submit_count_tracker_reco.slurm")

    echo "sample checkpoint: $SAMPLE0  reco checkpoint: $RECO0"

After both jobs finish, make the one-event survival and track-count report:

    apptainer exec --bind /oscar:/oscar "$IMAGE" \
      python "$CT_DIR/count_tracker_cohort_report.py" \
      --conditions "$CONDITIONS" --events-root "$EVENTS_OUT" --split test \
      --event-id "$EVENT0" --output "$BASE/reports/checkpoint.json"

After inspecting the sampler manifest, input manifests, and checkpoint report,
fill the remaining cohort. Both job scripts are resumable and skip event zero:

    SAMPLE_ALL=$(sbatch --parsable --array=0-19%10 \
      --export=ALL,CONDITIONS="$CONDITIONS",SPLIT=test,MODEL_ROOT="$MODEL_ROOT",OUTPUT="$COUNT_SAMPLES",CT_DIR="$CT_DIR",PAPER1_ROOT="$PAPER1_ROOT",MAX_ROUNDS=100,UNFILLED_POLICY=drop \
      -o "$BASE/logs/sample_%A_%a.out" -e "$BASE/logs/sample_%A_%a.err" \
      "$CT_DIR/submit_count_tracker_sample.slurm")

    RECO_ALL=$(sbatch --parsable --dependency="afterok:$SAMPLE_ALL" --array=0-19%10 \
      --export=ALL,CONDITIONS="$CONDITIONS",CONSTRUCTION=norm1_mother_direct,SPLIT=test,SIGNAL="$SIGNAL",COUNT_SAMPLES="$COUNT_SAMPLES",GEOMAP="$GEOMAP",EVENTS_OUT="$EVENTS_OUT",CT_DIR="$CT_DIR",IMAGE="$IMAGE",BENCHMARK_DIR="$BENCHMARK_DIR",GENBIB_DIR="$GENBIB_DIR" \
      -o "$BASE/logs/reco_%A_%a.out" -e "$BASE/logs/reco_%A_%a.err" \
      "$CT_DIR/submit_count_tracker_reco.slurm")

    echo "full sampling: $SAMPLE_ALL  full reconstruction: $RECO_ALL"

After reconstruction, aggregate digitization survival and track multiplicity,
extract the agreed fitted-track observables, and make separate plots:

    STORES="$BASE/track_stores"
    mkdir -p "$BASE/reports" "$STORES"

    apptainer exec --bind /oscar:/oscar "$IMAGE" \
      python "$CT_DIR/count_tracker_cohort_report.py" \
      --conditions "$CONDITIONS" --events-root "$EVENTS_OUT" \
      --split test --output "$BASE/reports/cohort.json"

    for SAMPLE in SIM COUNT; do
      apptainer exec --bind /oscar:/oscar "$IMAGE" \
        python "$CT_DIR/count_tracker_features.py" \
        --conditions "$CONDITIONS" --events-root "$EVENTS_OUT" \
        --sample "$SAMPLE" --split test \
        --output "$STORES/norm1_mother_direct_${SAMPLE}_test"
    done

    apptainer exec --bind /oscar:/oscar "$IMAGE" \
      python "$CT_DIR/plot_count_tracker_track_features.py" \
      --store-dir "$STORES" --construction norm1_mother_direct --split test \
      --output-dir "$BASE/reports/track_plots" \
      --title "Direct norm1 mother-muon: SIM vs COUNT"

The cohort report contains per-collection and total counts entering and
surviving digitization, survival fractions, paired SiTrack multiplicities, and
the total COUNT/SIM track ratio. The plotter writes separate pT, eta, phi, d0,
z0, and event-level multiplicity plots. Shape histograms are normalized per
arm; multiplicity is not normalized.

Direct SIM-vs-COUNT cohorts use every stored SIM tracker hit when constructing
per-sensor conditions. No timing selection is applied before digitization;
the shared digitization configuration determines timing acceptance for both
arms. `count_tracker_conditions.py` therefore defaults to `all-stored`.
`flight-corrected` remains available only to reproduce the earlier selected
control cohort and must be requested explicitly.

The retrained norm42 model is a separate, training-domain-matched comparison.
Its recorded data preparation retained raw SimTrackerHits satisfying
``time < 1.0e7 ns`` before forming the model train/validation split. Request
this exact, uncorrected-time selection with
``--hit-selection raw-time-lt-1e7-ns``. The condition builder applies it while
counting per-sensor occupancy, and the input writer applies it again while
copying the paired SIM hits. It is not the flight-corrected detector timing
window; normal digitization still runs identically on both arms afterward.

Mother-muon diffusion checkpoint
--------------------------------
For the direct norm42 comparison, use the six conditional local-phi models in

  /oscar/data/mleblan6/mucoll/speng44/bib_gen_mother_moun/new_diffusion

Each collection has one complete checkpoint. The sampler discovers the model
without assuming architecture tokens in its directory name and records the
resolved directory in its output manifest. These checkpoints use five
2048-unit layers, DIM_T=1024, 1000 diffusion steps, and 200000 training steps.
Their run configurations identify primary-muon training data and Paper 1
commit 90bd576c619416dbd7889eb1900ec80ac687ab92.

The corresponding retrained norm42 checkpoints are in

  /oscar/data/mleblan6/mucoll/speng44/bib_gen_mother_muon/new_diffusion_norm42/paper1_training_20260921_012450/models

They use the same generated features and sensor conditions, and record the
same Paper 1 commit. For a controlled comparison, reuse the existing norm42
source manifest but prepare a new conditions directory with
``raw-time-lt-1e7-ns``; write all COUNT samples and reconstruction products to
a new output root.

The first controlled check reuses `norm42_cmp_000000` conditions, signal entry
0, SIM result, geometry map, digitization, and reconstruction from the earlier
direct comparison. Write the new samples and COUNT reconstruction to new
directories; do not replace the earlier products. This isolates the diffusion
checkpoint as the changed input. Because the recorded Paper 1 checkout is not
available at its run-config path, pass the existing inference checkout
explicitly and treat successful one-event sampling/reconstruction as the
compatibility checkpoint before scaling up.

The sampler remains strict by default (`--unfilled-policy error`, up to 10000
rounds). For checkpoints with unsupported sensor conditions, use an explicit
finite policy such as `--max-rounds 100 --unfilled-policy drop`. Accepted hits
are published, while every unresolved five-column condition is saved beside
the collection as `tabddpm_<SHORT>_unfilled_conditions.npy`; requested,
generated, and unfilled counts are recorded in the sampler manifest. The input
writer separately records the resulting occupancy deficit after CellID
assignment. This policy must remain fixed across compared COUNT cohorts.

After that checkpoint, compare the unchanged prior SIM arm with the new COUNT
arm using `count_tracker_checkpoint_report.py`. It reports the six collection
counts before and after digitization, their survival fractions, and the number
of duplicate-removed SiTracks. Its inputs are arm directories (the directories
that contain `reco/digi_output.edm4hep.root`).

First cohort: norm42_reservoir. Its SIM arm draws empirical hits from the
verified inside_bounds=True COUNT training arrays, and its COUNT arm is drawn
from the model for the same per-sensor condition counts. Those counts come from
the broad-time-selected norm42 template events; only the hit values come from
the COUNT training input. This is an exact conditional training-input-domain
comparison for hit values, not a reconstruction of the training event. It tests
conditional hit and reconstruction
closure, not physical event-level hit correlations. The all-stored direct
norm42 comparison is a separate later domain-transfer study.
All empirical SIM rows ultimately come from one physical training event, and
the COUNT model was trained on that same source domain. A single reservoir
preparation with split-isolated row pools can prevent identical empirical rows
from appearing across classifier splits; separately prepared cohorts do not
guarantee this. Neither approach tests generalization to unseen physical BIB.

The baseline compares duplicate-removed SiTracks from two samples reconstructed
with the same tracker configuration. Each track uses the EDM4hep AtIP state.
The stored physical observables are pt [GeV], eta, phi [rad], d0 [mm],
and z0 [mm]. The store also keeps signed omega [1/mm] so the momentum
conversion can be checked. For MAIA's 5 T field,
pt [GeV] = 0.0015 / abs(omega [1/mm]).

The PFN uses log(pt / GeV), eta, sin(phi), cos(phi), d0, and z0.
Omega and charge are not classifier inputs. Valid tracks are identified by
the stored n_tracks array; padding is zeroed before the EnergyFlow PFN sums
per-track representations. There are no physics cuts, clipping, or fitted
normalization. Extraction stops if a selected track has no valid AtIP state
instead of silently reducing event multiplicity.

To build one store after reconstruction, run from the mucoll-slurm root in
an environment with uproot and numpy:

    python count_tracker/count_tracker_features.py \
      --conditions /path/to/conditions \
      --events-root /path/to/reconstructed/events \
      --sample SIM --split train \
      --output /path/to/stores/norm42_reservoir_SIM_train

Repeat for SIM and COUNT in train, val, and test, using the same
construction and split definitions. The output prefix must match
<construction>_<sample>_<split>. Version-1 stores made with the incorrect
momentum conversion are rejected and must be rebuilt from reco ROOT files.

The existing ten-event reservoir pilot is a test split only. After syncing this
code to OSCAR, build its corrected stores on a compute node in the v3 detector
container (which provides uproot):

    REPO="$(git rev-parse --show-toplevel)"
    BASE="/oscar/scratch/$USER/mucoll/count_tracker_cmp/norm42_reservoir"
    IMAGE="/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif"
    STORES="$BASE/track_stores_v2"
    mkdir -p "$STORES"
    for SAMPLE in SIM COUNT; do
      apptainer exec --bind /oscar:/oscar "$IMAGE" \
        python "$REPO/count_tracker/count_tracker_features.py" \
        --conditions "$BASE/conditions" --events-root "$BASE/events" \
        --sample "$SAMPLE" --split test \
        --output "$STORES/norm42_reservoir_${SAMPLE}_test"
    done
    apptainer exec --bind /oscar:/oscar "$IMAGE" \
      python "$REPO/count_tracker/count_tracker_multiplicity.py" \
      --store-dir "$STORES" --construction norm42_reservoir \
      --output "$BASE/multiplicity_test_v2.json"

When train, val, and test reservoir cohorts from one split-isolated preparation
have been reconstructed
and all six stores exist, run in the PFN environment (EnergyFlow, TensorFlow,
and tf_keras). Give the event and track fits separate result labels:

    python count_tracker/count_tracker_train.py \
      --store-dir /path/to/stores \
      --construction norm42_reservoir --unit event \
      --sample-a SIM --sample-b COUNT \
      --label reservoir_event \
      --outdir /path/to/pfn_results

    python count_tracker/count_tracker_train.py \
      --store-dir /path/to/stores \
      --construction norm42_reservoir --unit track \
      --sample-a SIM --sample-b COUNT \
      --label reservoir_single_track \
      --outdir /path/to/pfn_results

The event PFN labels one complete track set; the one-track PFN labels one
fitted track and weights nonempty events equally within each class. Both use
the same observables and source splits. Their AUCs have different statistical
units and should not be compared as if they measure the same probability.
The default `--recipe fixed` uses Adam at a constant learning rate of 1e-3
with no dropout. `--recipe fixed_dropout` keeps that rate and sets F dropout
to 0.1. The earlier `stabilized` recipes retain their 1e-4 warmup/cosine
schedule only to reproduce the preliminary fits.
The trainer writes the held-out test
AUC, a paired-event bootstrap interval conditional on the reused source pool,
training history, model weights, feature definitions, and store hashes.
--permute-labels is a training-code
sanity control; it is not the physical SIM-SIM or COUNT-COUNT null. Those nulls
require independently produced track stores from the same respective procedure.
The current ten-event pilot cannot train a held-out PFN yet.
