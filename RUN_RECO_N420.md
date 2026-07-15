# N=420 reconstructed-BIB PFN

This is one fixed study:

- `U`: 420 unrotated norm1 SIM files per beam polarity and event.
- `R`: 10 norm42 SIM files per beam polarity and event (10 x 42 mothers).
- Null: `U` versus an independently seeded `null_b`, both drawn from the same
  norm1 pool.
- Invisible 100 GeV PDG-14 gun, with matched generator seeds across classes.
- Source-cycle-separated `train`, `val`, `test_a`, and `test_b` pools.
- PFO PFN with sum aggregation. Test A and B use disjoint source-cycle pools.

Run every command below on Perlmutter from `~/mucoll/mucoll-slurm`.

## 1. Rebuild source pools without cycle 6291

```bash
module load python
source config.sh

python3 reco_libtest_prepare_pools.py \
  --norm1-sim "$DATA_GROUP_DIR/bib-v3p0-fmt2-norm1/SIM" \
  --norm42-sim "$DATA_GROUP_DIR/bib-v3p0-fmt2-norm42-RandomRot/SIM" \
  --outdir "$PSCRATCH/mucoll/libtest/bib_pools" \
  --exclude-cycle 6291 \
  --force
```

Confirm that the manifest records the exclusion:

```bash
python3 -c 'import json; print(json.load(open("'"$PSCRATCH"'/mucoll/libtest/bib_pools/manifest.json"))["excluded_cycles"])'
```

Expected output: `[6291]`.

## 2. Produce the complete RECO data set

The fixed sample has 2,000 training, 400 validation, and two independent
400-event test cohorts per class. Fifty events are processed per CPU job.
Each chain is submitted to the CPU `shared` QOS as one 4-CPU, 16-GB task, so
Slurm can place many chains on a node and charge only the occupied fraction.

```bash
python3 submit_reco_libtest.py --split train  --events-per-class 2000
python3 submit_reco_libtest.py --split val    --events-per-class 400
python3 submit_reco_libtest.py --split test_a --events-per-class 400
python3 submit_reco_libtest.py --split test_b --events-per-class 400
```

All three required samples (`U R null_b`) are submitted by default. The U
sample is reused as the first null class; producing a separate `null_a` with
the same seed would create an identical, redundant data set. Repeating a
command skips completed outputs and resubmits missing jobs.

Count completed files with:

```bash
for sample in U R null_b; do
  for split in train val test_a test_b; do
    directory="$PSCRATCH/mucoll/libtest/reco_n420_pfn/reco_libtest_n420_${sample}/${split}"
    printf '%-12s %-7s ' "$sample" "$split"
    find "$directory" -name 'reco_output_*.edm4hep.root' -type f 2>/dev/null | wc -l
  done
done
```

Expected files per sample: train 40, validation 8, test A 8, test B 8.

## 3. Convert RECO files to PFO stores

```bash
sbatch submit_reco_libtest_stores.slurm
```

The conversion job writes twelve files below
`$PSCRATCH/mucoll/libtest/reco_n420_pfn_stores/`.

## 4. Train the classifier and its null

```bash
sbatch submit_reco_libtest_train.slurm
```

The debug job fills one four-GPU node with two independent U-vs-R seeds and
two shared-pool-null seeds. Results are written to:

```text
reco_pfn_results/reco_n420_U_vs_R_seed1/
reco_pfn_results/reco_n420_U_vs_R_seed2/
reco_pfn_results/reco_n420_null_seed1/
reco_pfn_results/reco_n420_null_seed2/
```

Each `summary.json` reports AUC separately on test A and test B and on their
combination. Test A and test B have disjoint source cycles; events within a
cohort may share source files and are therefore correlated.

## PFN input

Each reconstructed PFO contributes nine values:

```text
log(pT), eta, sin(phi), cos(phi), log(E), charge,
charged flag, photon flag, neutral flag
```

The network applies a small per-PFO MLP `(64,64,64)`, sums the learned PFO
representations, and applies an event MLP `(64,64,64)`. The unnormalised sum is
intentional: multiplicity and total activity are possible consequences of
mother reuse. With roughly O(10) PFOs per N=420 event, the GEN-level
large-sum saturation problem is absent.
