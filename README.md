# mucoll-slurm

Run the Muon Collider simulation chain (**GEN → SIM → DIGI → RECO**) on Oscar,
both interactively and as SLURM batch jobs, using the **mucoll-spack v3.0**
container image. Particle-gun studies can be run **with or without BIB**

---

## 1. First-time setup

Check out both repositories side-by-side (they must be siblings). The
benchmarks repo is the **official MuonColliderSoft** one and pulls its
detector configs in as submodules, so clone it with `--recurse-submodules`:

```bash
cd ~/work                     # or wherever you keep code
git clone https://github.com/leblanc-lab/mucoll-slurm.git
git clone --recurse-submodules https://github.com/MuonColliderSoft/mucoll-benchmarks.git
```

Your tree should look like:

```
<your work dir>/
├── mucoll-slurm/        <- this repo
└── mucoll-benchmarks/   <- MuonColliderSoft main, with configs/MAIAConfig etc.
```

The v3.0 image has already been pulled and is cached in the shared group data directory:

```
/oscar/data/mleblan6/mucoll/mucoll-sim-ubuntu24:v3.0.sif
```

If you ever need to re-pull it e.g. on another cluster or with an updated image, you can do so with:

```bash
export APPTAINER_TMPDIR=/oscar/scratch/$USER/apptainer_tmp
export APPTAINER_CACHEDIR=/oscar/scratch/$USER/apptainer_cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR
apptainer pull mucoll-sim-ubuntu24:v3.0.sif \
    docker://ghcr.io/muoncollidersoft/mucoll-sim-ubuntu24:v3.0
```

Finally, open [`config.sh`](config.sh) and skim it. Most values auto-detect; the
first one you may want to change is `OUTPUT_BASE_DIR` (defaults to your own
`$USER` folder inside the shared group area). The image path, benchmarks path,
geometry, and BIB sample locations are already set.

`config.sh` is read by the Python submitters via
`slurm_common.load_config()` and the shell chains `source` it directly, so
there is exactly one place to edit paths. Per-job physics settings (particle, pT,
θ, event/job counts, BIB on/off) and SLURM resources (`TIME`, `MEM`, `CPUS`)
live at the top of each `submit_*.py`.

---

## 2. Run the chain interactively (for debugging)

```bash
source scripts/interact.sh     # grab a worker node (don't run on the login nodes!)
source scripts/shell.sh        # enter the v3.0 container
source scripts/setup.sh        # set up the spack environment
```

Then you can run the same chain a job uses, by hand:

```bash
bash chains/run_chain_pgun.sh --job-id 0 --nevents 1 --outdir /tmp/test \
     --pdg 13 --pt 100 --theta-min 10 --theta-max 170        # add --bib for BIB
```

Or step through the GEN/SIM/DIGI/RECO stages individually — see the per-stage
READMEs in `mucoll-benchmarks/`.

## 3. Submit particle-gun jobs to slurm batch system

Edit the settings at the top of [`submit_pgun.py`](submit_pgun.py) — particle, pT,
theta range, number of jobs/events, and the **`BIB`** switch — then:

```bash
python submit_pgun.py          # run on a login node: it only calls sbatch;
                               # the heavy work runs in the container on the nodes
```

* **Without BIB:** leave `BIB = False`.
* **With BIB:** set `BIB = True`. `run_chain_pgun.sh` then appends to the
  digitization step:

  ```
  --doOverlayFull \
  --OverlayFullPathToMuPlus  $BIB_MUPLUS \
  --OverlayFullPathToMuMinus $BIB_MUMINUS \
  --OverlayFullNumberBackground $BIB_NUMBER
  ```

  The BIB sample directories (`MUPLUS/`, `MUMINUS/` of `.edm4hep.root` files)
  and the overlay count are set in `config.sh`. `BIB_NUMBER` is the number of
  BIB files overlaid per signal event per polarity (default 812; tune per study).

Output layout:

```
$OUTPUT_BASE_DIR/<study>/
├── logs/job_N.{out,err}
└── job_N/{gen,sim,digi,reco}_output_N.edm4hep.root
```

`<study>` auto-names from the particle/pT/θ and BIB state, e.g.
`pgun_pdg13_pt100_theta10-170_nobib`.

### Quick test

Before launching a big batch, sanity-check with one short job: set
`NUM_JOBS = 1`, `NEVENTS_PER_JOB = 1` in `submit_pgun.py` and submit. Confirm the
job finishes and produces a `reco_output_0.edm4hep.root`.

### Parameter scan

To scan over several particles / momenta / angles at once, edit
`PDG_LIST` / `PT_LIST` / `THETA_LIST` (and `BIB`) in
[`submit_pgun_scan.py`](submit_pgun_scan.py), then:

```bash
python submit_pgun_scan.py
```

This creates `$OUTPUT_BASE_DIR/scan[_bib]/pdg{P}_pt{T}_theta{lo}-{hi}/job_N/`.


---

## 4. Whizard signal production (advanced)

[`submit_whizard.py`](submit_whizard.py) drives the Whizard WWZ/ZZZ hadronic
chains (steering `.sin` files live in [`whizard/`](whizard/)):

1. **(Once)** build the integration grids: `python make_gridpack.py` — writes
   `.vg` grids under `$DATA_GROUP_DIR/gridpacks/`.
2. In `submit_whizard.py`, pick the process(es) in `PROCESSES`, optionally set
   `GRIDPACK_DIR` to the grids from step 1 (leave `""` to integrate in-job),
   then `python submit_whizard.py`.

> **Known limitation (signal only):** the v3.0 *sim* image has no Whizard, while
> the digi/reco configs now target v3.0. So the single-container WWZ/ZZZ chains
> can't yet run generation (needs a Whizard image, `WHIZARD_IMAGE` in
> `config.sh`) and v3.0 digi/reco in one job — that needs a gen→reco image split.
> The chains are wired to the new layout and guard loudly on a missing Whizard.
> **Particle-gun studies (the intern workflow) are fully working on v3.0.**

---
