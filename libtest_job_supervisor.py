#!/usr/bin/env python3
"""Laptop-side supervisor for resumable Perlmutter libtest jobs.

By default one Slurm window is kept alive per unfinished label.  Plans may set
``bundle_size`` and ``bundle_script`` to pack several independent labels into
one full-node GPU allocation.  When a window leaves squeue, the supervisor
waits one poll for final output, then resumes incomplete labels.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def log(message):
    stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print("[{}] {}".format(stamp, message), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep one resumable Slurm window alive per libtest run")
    parser.add_argument("config", help="JSON supervisor plan")
    parser.add_argument("--once", action="store_true",
                        help="perform one poll/submission pass and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="show submissions without calling sbatch")
    parser.add_argument("--poll-seconds", type=int,
                        help="override the plan polling interval")
    parser.add_argument("--state-file", help="override local restart state")
    return parser.parse_args()


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def validate_plan(plan):
    required = ["remote_host", "remote_dir", "runs"]
    missing = [key for key in required if key not in plan]
    if missing:
        raise SystemExit("missing plan keys: {}".format(", ".join(missing)))
    if not plan["runs"]:
        raise SystemExit("the plan contains no runs")
    labels = [run.get("label", "") for run in plan["runs"]]
    bad = [label for label in labels if not SAFE_LABEL.fullmatch(label)]
    if bad:
        raise SystemExit("unsafe or empty labels: {}".format(", ".join(bad)))
    if len(set(labels)) != len(labels):
        raise SystemExit("run labels must be unique")
    bundle_size = int(plan.get("bundle_size", 1))
    if bundle_size < 1:
        raise SystemExit("bundle_size must be positive")
    if bundle_size > 1:
        bundle_min_size = int(plan.get("bundle_min_size", 3))
        if not 2 <= bundle_min_size <= bundle_size:
            raise SystemExit(
                "bundle_min_size must be between 2 and bundle_size")
        for key in ("bundle_script", "bundle_config"):
            if not plan.get(key):
                raise SystemExit("{} is required with bundle_size".format(key))
    for run in plan["runs"]:
        if int(run.get("max_windows", 10)) < 1:
            raise SystemExit("max_windows must be positive for {}".format(
                run["label"]))


def default_state_path(config_path):
    return (Path.home() / ".local" / "state" / "mucoll-libtest" /
            "{}.state.json".format(Path(config_path).stem))


def load_state(path):
    if path.is_file():
        state = load_json(path)
    else:
        state = {"runs": {}}
    state.setdefault("runs", {})
    return state


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def shell_path(path):
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/{}'.format(shlex.quote(path[2:]))
    return shlex.quote(path)


def ssh_command(host, remote_command):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30",
         "-o", "ServerAliveInterval=20", host, remote_command],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=90)


def remote_snapshot(plan):
    """Return ({label: complete}, all of the user's live Slurm job IDs).

    Count every QOS here.  Site-specific QOS display/filter behavior can differ
    from the value passed to sbatch, and missing a live job is more dangerous
    than conservatively waiting for an unrelated job to leave the queue.
    """
    result_dir = plan.get("result_dir", "pfn_results")
    commands = ["cd {}".format(shell_path(plan["remote_dir"]))]
    for run in plan["runs"]:
        label = run["label"]
        summary = "{}/{}/auc_summary.json".format(result_dir, label)
        commands.append(
            "if [ -s {p} ]; then echo RESULT {l} complete; "
            "else echo RESULT {l} incomplete; fi".format(
                p=shlex.quote(summary), l=shlex.quote(label)))
    commands.append("squeue --me -h -o 'JOB %A'")
    result = ssh_command(plan["remote_host"], " && ".join(commands))
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError("remote status check failed: {}".format(detail))

    complete = {}
    debug_jobs = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == "RESULT":
            complete[fields[1]] = fields[2] == "complete"
        elif len(fields) == 2 and fields[0] == "JOB":
            debug_jobs.add(fields[1])
    return complete, debug_jobs


def submission_command(plan, run):
    label = run["label"]
    train_args = run["train_args"]
    script = run.get("script", "submit_libtest_train.slurm")
    job_name = run.get("job_name", "lt_{}".format(label))[:128]
    return (
        "cd {directory} && "
        "LABEL={label} TRAIN_ARGS={args} "
        "sbatch --parsable --job-name={job_name} "
        "--export=ALL,LABEL,TRAIN_ARGS {script}"
    ).format(
        directory=shell_path(plan["remote_dir"]),
        label=shlex.quote(label), args=shlex.quote(train_args),
        job_name=shlex.quote(job_name), script=shlex.quote(script))


def submit_window(plan, run, dry_run=False):
    command = submission_command(plan, run)
    if dry_run:
        log("DRY RUN {}: {}".format(run["label"], command))
        return "dry-run-{}".format(run["label"])
    result = ssh_command(plan["remote_host"], command)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError("submission failed for {}: {}".format(
            run["label"], detail))
    lines = [line.strip().split(";", 1)[0]
             for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].isdigit():
        raise RuntimeError("unexpected sbatch output for {}: {!r}".format(
            run["label"], result.stdout))
    return lines[0]


def submission_command_bundle(plan, runs):
    labels = ",".join(run["label"] for run in runs)
    job_name = "lt_bundle_{}".format("_".join(
        run["label"].replace("A0_", "")[:12] for run in runs))[:128]
    return (
        "cd {directory} && "
        "BUNDLE_CONFIG={config} BUNDLE_LABELS={labels} "
        "sbatch --parsable --job-name={job_name} "
        "--export=ALL,BUNDLE_CONFIG,BUNDLE_LABELS {script}"
    ).format(
        directory=shell_path(plan["remote_dir"]),
        config=shlex.quote(plan["bundle_config"]),
        labels=shlex.quote(labels),
        job_name=shlex.quote(job_name),
        script=shlex.quote(plan["bundle_script"]),
    )


def submit_bundle(plan, runs, dry_run=False):
    command = submission_command_bundle(plan, runs)
    labels = ", ".join(run["label"] for run in runs)
    if dry_run:
        log("DRY RUN bundle [{}]: {}".format(labels, command))
        return "dry-run-bundle"
    result = ssh_command(plan["remote_host"], command)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError("bundle submission failed for [{}]: {}".format(
            labels, detail))
    lines = [line.strip().split(";", 1)[0]
             for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].isdigit():
        raise RuntimeError("unexpected sbatch output for bundle [{}]: {!r}"
                           .format(labels, result.stdout))
    return lines[0]


def run_pass(plan, state, state_path, dry_run=False):
    complete, debug_jobs = remote_snapshot(plan)
    max_submitted = int(plan.get("max_submitted", 4))
    available = max(0, max_submitted - len(debug_jobs))
    disappeared = set()

    for run in plan["runs"]:
        label = run["label"]
        info = state["runs"].setdefault(
            label, {"job_ids": [], "active_job_id": None})
        active = info.get("active_job_id")
        if active and active not in debug_jobs:
            # Do not resubmit in this pass.  On the next poll the result file
            # check will observe output written as the old job was exiting.
            log("{} window {} left squeue; checking output next poll".format(
                label, active))
            info["active_job_id"] = None
            disappeared.add(label)

    save_state(state_path, state)

    status = []
    for run in plan["runs"]:
        label = run["label"]
        info = state["runs"][label]
        if complete.get(label, False):
            status.append("{}=complete".format(label))
        elif info.get("active_job_id") in debug_jobs:
            status.append("{}=job {}".format(label, info["active_job_id"]))
        else:
            status.append("{}=needs window".format(label))
    log(" | ".join(status))

    if all(complete.get(run["label"], False) for run in plan["runs"]):
        return "complete"

    # Let the next remote snapshot observe files written while a job was
    # exiting.  In particular, do not fill its slot with a lower-priority run
    # before learning whether the just-finished label needs continuation.
    if disappeared:
        return "working"

    bundle_size = int(plan.get("bundle_size", 1))
    if bundle_size > 1:
        eligible = []
        for run in plan["runs"]:
            label = run["label"]
            info = state["runs"][label]
            if complete.get(label, False):
                continue
            if info.get("active_job_id") in debug_jobs:
                continue
            if len(info["job_ids"]) >= int(run.get("max_windows", 10)):
                continue
            eligible.append(run)

        bundle_min_size = int(plan.get("bundle_min_size", 3))
        while available > 0 and eligible:
            # A debug allocation is billed as a whole four-GPU node.  Three or
            # four ranks use that node reasonably; a one/two-run tail is less
            # expensive as independent fractionally billed shared-QOS jobs.
            if len(eligible) >= bundle_min_size:
                group = eligible[:bundle_size]
                eligible = eligible[bundle_size:]
                job_id = submit_bundle(plan, group, dry_run)
                log("submitted GPU bundle job {} for {}".format(
                    job_id, ", ".join(run["label"] for run in group)))
                if dry_run:
                    break
                submitted_at = dt.datetime.now(dt.timezone.utc).isoformat()
                for run in group:
                    info = state["runs"][run["label"]]
                    info["job_ids"].append(job_id)
                    info["active_job_id"] = job_id
                    info["last_submitted_at"] = submitted_at
                save_state(state_path, state)
                available -= 1
                continue

            run = eligible.pop(0)
            job_id = submit_window(plan, run, dry_run)
            log("submitted shared-QOS tail job {} for {}".format(
                job_id, run["label"]))
            if dry_run:
                break
            info = state["runs"][run["label"]]
            info["job_ids"].append(job_id)
            info["active_job_id"] = job_id
            info["last_submitted_at"] = dt.datetime.now(
                dt.timezone.utc).isoformat()
            save_state(state_path, state)
            available -= 1
    else:
        for run in plan["runs"]:
            if available <= 0:
                break
            label = run["label"]
            info = state["runs"][label]
            if complete.get(label, False):
                continue
            if info.get("active_job_id") in debug_jobs:
                continue
            max_windows = int(run.get("max_windows", 10))
            if len(info["job_ids"]) >= max_windows:
                continue
            job_id = submit_window(plan, run, dry_run)
            log("submitted {} window {} as job {}".format(
                label, len(info["job_ids"]) + 1, job_id))
            if not dry_run:
                info["job_ids"].append(job_id)
                info["active_job_id"] = job_id
                info["last_submitted_at"] = dt.datetime.now(
                    dt.timezone.utc).isoformat()
                save_state(state_path, state)
            available -= 1

    exhausted = [
        run["label"] for run in plan["runs"]
        if not complete.get(run["label"], False)
        and not state["runs"][run["label"]].get("active_job_id")
        and len(state["runs"][run["label"]]["job_ids"])
        >= int(run.get("max_windows", 10))
    ]
    if exhausted:
        log("maximum windows reached without completion: {}".format(
            ", ".join(exhausted)))
        return "exhausted"
    return "working"


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    plan = load_json(config_path)
    validate_plan(plan)
    poll_seconds = args.poll_seconds or int(plan.get("poll_seconds", 300))
    if poll_seconds < 10 and not args.once:
        raise SystemExit("poll interval must be at least 10 seconds")
    state_path = (Path(args.state_file).expanduser() if args.state_file
                  else default_state_path(config_path))
    state_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_handle = open(lock_path, "w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another supervisor is already using {}".format(
            state_path))

    state = load_state(state_path)
    log("supervising {} runs via {}; state {}".format(
        len(plan["runs"]), plan["remote_host"], state_path))
    while True:
        try:
            outcome = run_pass(plan, state, state_path, args.dry_run)
            if outcome == "complete":
                log("all configured runs completed")
                return 0
            if outcome == "exhausted":
                return 2
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            log("temporary supervisor error: {}".format(exc))
        if args.once or args.dry_run:
            return 1
        time.sleep(poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
