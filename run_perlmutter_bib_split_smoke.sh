#!/bin/bash

set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
BENCH=${MUCOLL_BENCHMARKS_PATH:-$REPO/../mucoll-benchmarks}
PROJECT=/global/cfs/cdirs/m5197/mleblanc/MuonCollider/data/bib-v3p0-fmt2-split-muon-v1
MANIFEST=$PROJECT/manifests/partition.json
POLARITY=$1
WORK=$PSCRATCH/mucoll/bib_split_muon_smoke/$SLURM_JOB_ID/$POLARITY

set +u
source /opt/setup_mucoll.sh
set -u

mkdir -p "$WORK"

cycle=$(python3 -c "
import json
m=json.load(open('$MANIFEST'))
r=[x for x in m['polarities']['$POLARITY'] if len(x['muon_entries']) >= 2]
r.sort(key=lambda x: x['bulk_particles'])
print(r[0]['cycle'])
")

python3 -u "$REPO/bib_split_muon_gen.py" write-gen \
    --manifest "$MANIFEST" \
    --output-root "$WORK" \
    --polarity "$POLARITY" \
    --cycle "$cycle"

bulk_gen=$WORK/bulk-norm42/GEN/$POLARITY/bib_gen_${cycle}.edm4hep.root
muon_gen=$WORK/decays-containing-muon-norm1-norot/GEN/$POLARITY/bib_gen_${cycle}.edm4hep.root
bulk_sim=$WORK/bulk-norm42/SIM/$POLARITY/bib_sim_${cycle}.edm4hep.root
muon_sim=$WORK/decays-containing-muon-norm1-norot/SIM/$POLARITY/bib_sim_${cycle}.edm4hep.root
mkdir -p "$(dirname "$bulk_sim")" "$(dirname "$muon_sim")"

muon_events=$(python3 -c "import uproot; print(uproot.open('$muon_gen')['events'].num_entries)")

set +u
set +o pipefail
source "$BENCH/setup_config.sh" "$BENCH" MAIA_v0
set -o pipefail
set -u

ddsim \
    --steeringFile "$BENCH/simulation/steer_baseline.py" \
    --numberOfEvents 1 \
    --inputFiles "$bulk_gen" \
    --outputFile "$bulk_sim"

ddsim \
    --steeringFile "$BENCH/simulation/steer_baseline.py" \
    --numberOfEvents "$muon_events" \
    --inputFiles "$muon_gen" \
    --outputFile "$muon_sim"

python3 -c "
import uproot
for path, expected in [('$bulk_sim', 1), ('$muon_sim', $muon_events)]:
    f=uproot.open(path)
    assert 'podio_metadata' in f
    assert f['events'].num_entries == expected
    assert 'MCParticles' in f['events']
    print(path, 'events=', expected)
"

echo "smoke passed polarity=$POLARITY cycle=$cycle muon_entries=$muon_events work=$WORK"
