#!/usr/bin/env python3

import os
import sys

import slurm_common as sc


NUM_JOBS = 20
NEVENTS_PER_JOB = 5

PT = 100
THETA_MIN = 10
THETA_MAX = 170

TIME = "02:00:00"
MEM = "16G"
CPUS = 2

STUDY_PREFIX = f"pfn_mu_pi_pt{PT}"

SAMPLES = [
    ("mu", 13, False),
    ("pi", 211, False),
    ("mu", 13, True),
    ("pi", 211, True),
]


def study_name(particle, bib):
    tag = "bib" if bib else "nobib"
    return f"{STUDY_PREFIX}_{particle}_{tag}"


def main():
    if len(sys.argv) > 1:
        raise SystemExit("has constants- no arguments")

    cfg = sc.load_config()
    sc.validate_paths(cfg)

    chain = os.path.join(cfg["WORK_DIR"], "mucoll-slurm/chains/run_chain_pgun.sh")
    os.chmod(chain, 0o755)

    total_jobs = len(SAMPLES) * NUM_JOBS
    print(f"Submitting {total_jobs} jobs")
    print(f"Events per sample: {NUM_JOBS * NEVENTS_PER_JOB}")
    print(f"pT={PT} GeV, theta=[{THETA_MIN}, {THETA_MAX}]")

    submitted = 0
    for particle, pdg, bib in SAMPLES:
        study = study_name(particle, bib)
        out_dir = os.path.join(cfg["OUTPUT_BASE_DIR"], study)
        log_dir = os.path.join(out_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        print(f"- {study}")

        for job_id in range(NUM_JOBS):
            chain_args = [
                "--job-id", job_id,
                "--nevents", NEVENTS_PER_JOB,
                "--outdir", out_dir,
                "--pdg", pdg,
                "--pt", PT,
                "--theta-min", THETA_MIN,
                "--theta-max", THETA_MAX,
            ]
            if bib:
                chain_args.append("--bib")

            body = (
                'echo "Host: $(hostname)"\n'
                f'echo "{study} job {job_id}"\n\n'
                + sc.apptainer_cmd(cfg, chain, chain_args)
            )
            script = sc.make_slurm_script(
                job_name=f"{particle}_{'bib' if bib else 'nobib'}_{job_id}",
                out_log=os.path.join(log_dir, f"job_{job_id}.out"),
                err_log=os.path.join(log_dir, f"job_{job_id}.err"),
                sbatch_directives=[
                    f"--time={TIME}",
                    f"--mem={MEM}",
                    "--nodes=1",
                    "--ntasks=1",
                    f"--cpus-per-task={CPUS}",
                ],
                body=body,
            )

            print(f"  job {job_id}:", end="")
            if sc.submit(script, f"_submit_pfn_{study}_{job_id}.sh"):
                submitted += 1

    print(f"\nSubmitted {submitted}/{total_jobs} jobs.")


if __name__ == "__main__":
    main()
