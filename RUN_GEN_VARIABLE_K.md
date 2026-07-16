# Variable mother-reuse PFN at GEN level

This study constructs fixed-size pseudo-events directly from the
split-by-mother, unrotated GEN library. Rotations are generated coherently and
deterministically in the loader; no 42x materialized rotation library is
required.

The default event contains 29,400 mother-equivalents, approximately the mean
content of 420 of the old cycle files. The eight classes are reuse factors
`k = 1, 2, 3, 6, 10, 14, 21, 42`. Thus every class has the same nominal decay
statistics while the number of unique source mothers varies from 29,400 to 700.

Run from `~/mucoll/mucoll-slurm` on Perlmutter.

## 1. Build the compact mother bank

```bash
sbatch submit_variable_reuse_convert.slurm
```

Expected output:

```text
$PSCRATCH/mucoll/libtest/stores/gen_split_mothers_MUPLUS.h5
```

Cycle 6291 is excluded. The converter preserves particle offsets for every
mother event and cycle-level provenance for leakage-free splits.

## 2. Small software/throughput smoke

This deliberately uses only 420 mother-equivalents and is not a physics result:

```bash
sbatch --export=ALL,LABEL=variable_k_smoke,TRAIN_ARGS="--mother-equivalents 420 --reuse-k 1 2 3 6 10 14 21 42 --units-per-epoch 1 --val-units 1 --test-units 1 --epochs 2 --patience 2 --rotation-policy all-random" submit_variable_reuse_train.slurm
```

Check `variable_k_train_variable_k_pfn_JOBID.{out,err}` and require a completed
`variable_k_results/variable_k_smoke/summary.json` before production.

## 3. Four-GPU production bundle

```bash
sbatch submit_variable_reuse_bundle.slurm
```

The four A100s run:

1. scaled PFN sum, every class randomly rotated;
2. raw PFN sum, every class randomly rotated;
3. scaled sum with the `k=1` baseline left unrotated;
4. shuffled-label multiclass null.

The all-random comparison isolates repeated-source structure from the trivial
fact that the historical `k=1` library was unrotated. The baseline-unrotated
model reproduces that historical construction difference.

Each label checkpoints after a debug window. Resubmit the same bundle only
after its previous job has left `squeue`; completed labels return quickly and
unfinished labels resume from `last.weights.h5`.

Final point metrics are in `variable_k_results/LABEL/summary.json`. Test
pseudo-events reuse held-out mothers across events, so these initial metrics do
not yet carry a source-cycle-bootstrap uncertainty.

## Validate synthetic k=42 against the production library

Before interpreting the variable-k sweep, check that the on-the-fly rotation
method reproduces the existing materialized `norm42-RandomRot` construction:

```bash
sbatch submit_synthetic42_validation.slurm
```

This fills one debug node with three model seeds of synthetic42 versus original
norm42 and one synthetic42-versus-synthetic42 null. Each class in a pair uses
the exact same source cycles. Test pairs use mutually disjoint held-out cycles,
and the validation features include both momentum and vertex azimuth.
`summary.json` reports a paired-cycle bootstrap uncertainty. Expected AUC
for both the main comparison and null is approximately 0.5. A reproducibly
non-null main AUC means the two rotation/conversion implementations differ and
must be investigated before using the variable-k sweep as a production proxy.
