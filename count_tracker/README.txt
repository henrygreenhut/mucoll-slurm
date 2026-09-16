Reconstructed-track two-sample classifiers
==========================================

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

The first controlled check reuses `norm42_cmp_000000` conditions, signal entry
0, SIM result, geometry map, digitization, and reconstruction from the earlier
direct comparison. Write the new samples and COUNT reconstruction to new
directories; do not replace the earlier products. This isolates the diffusion
checkpoint as the changed input. Because the recorded Paper 1 checkout is not
available at its run-config path, pass the existing inference checkout
explicitly and treat successful one-event sampling/reconstruction as the
compatibility checkpoint before scaling up.

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
