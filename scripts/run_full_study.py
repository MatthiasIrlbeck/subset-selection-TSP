#!/usr/bin/env python3
"""Run the full study with fail-closed manifests and explicit methodology.

Unlike the historical driver, this command never silently changes solver
methodology, never accepts a merely parseable file as resumable, and never
returns success for an incomplete publication campaign unless --allow-partial
is explicitly supplied. Every cell is probed for exact resolved and method
fingerprints before execution.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from campaign_manifest import (
    ManifestError,
    load_or_create_manifest,
    make_entry,
    require_complete,
    update_entry,
    validate_result,
)

HERE = Path(__file__).resolve().parent


def log(message: str, stream) -> None:
    line = f"[{_datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


def detect_lkh(explicit: str | None) -> str | None:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(explicit)
        return str(Path(found).resolve()) if found else None
    for name in ("LKH", "LKH.exe", "lkh"):
        found = shutil.which(name)
        if found:
            return str(Path(found).resolve())
    return None


def run_entry(
    entry: dict,
    manifest_path: Path,
    manifest: dict,
    timeout: int | None,
    log_stream,
) -> bool:
    output = Path(entry["output"])
    if output.exists():
        validated = validate_result(output, entry)
        update_entry(
            manifest_path, manifest, entry["cell_id"], status="complete",
            result_sha256=validated["sha256"],
        )
        log(f"resume verified: {entry['cell_id']}", log_stream)
        return True
    try:
        completed = subprocess.run(
            [entry["executable"], *entry["argv"]],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        update_entry(manifest_path, manifest, entry["cell_id"], status="timeout")
        log(f"TIMEOUT: {entry['cell_id']}", log_stream)
        return False
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().replace("\n", " ")
        update_entry(manifest_path, manifest, entry["cell_id"], status="failed")
        log(
            f"FAIL: {entry['cell_id']} exit={completed.returncode} :: {detail[:400]}",
            log_stream,
        )
        return False
    try:
        validated = validate_result(output, entry)
    except ManifestError as exc:
        update_entry(manifest_path, manifest, entry["cell_id"], status="invalid")
        log(f"INVALID: {entry['cell_id']} :: {exc}", log_stream)
        return False
    update_entry(
        manifest_path, manifest, entry["cell_id"], status="complete",
        result_sha256=validated["sha256"],
    )
    log(f"complete: {entry['cell_id']}", log_stream)
    return True


def execute_plan(
    manifest_path: Path,
    campaign: dict,
    entries: list[dict],
    *,
    timeout: int | None,
    allow_partial: bool,
    dry_run: bool,
    log_stream,
) -> dict:
    manifest = load_or_create_manifest(manifest_path, campaign, entries)
    log(
        f"manifest {manifest_path}: {len(entries)} cells, "
        f"fingerprint={manifest['manifest_fingerprint'][:16]}",
        log_stream,
    )
    if dry_run:
        return manifest
    failures = 0
    for entry in manifest["entries"]:
        try:
            if not run_entry(entry, manifest_path, manifest, timeout, log_stream):
                failures += 1
        except ManifestError as exc:
            failures += 1
            log(f"INVALID RESUME: {entry['cell_id']} :: {exc}", log_stream)
    require_complete(manifest, allow_partial=allow_partial)
    if failures and not allow_partial:
        raise ManifestError(f"{failures} campaign cells failed")
    return manifest


def metadata_args(args) -> list[str]:
    return [
        "--campaign-id", args.campaign_id,
        "--campaign-shard", str(args.campaign_shard),
        "--replicate-offset", str(args.replicate_offset),
        "--point-seed", str(args.point_seed),
        "--search-seed", str(args.search_seed),
        "--solver-policy-id", args.solver_policy_id,
        "--fidelity-level", args.fidelity_level,
    ]


def make_sweep_entries(args, threads: int) -> list[dict]:
    directory = args.out_dir / "sweep"
    directory.mkdir(parents=True, exist_ok=True)
    p_values = "0.005,0.02,0.05,0.1,0.2,0.5,1.0"
    configurations = [
        ("sweep_tor_2000", 2000, p_values, True, args.instances, True),
        ("sweep_sq_2000", 2000, p_values, False, args.instances, True),
        (
            "sweep_tor_100k", 100000, "0.002,0.01,0.05", True,
            max(4, args.instances // 3), False,
        ),
    ]
    entries = []
    for cell_id, n, p_text, periodic, instances, held_karp in configurations:
        output = directory / f"{cell_id}.json"
        p_values_list = [float(value) for value in p_text.split(",")]
        argv = [
            "--N", str(n), "--instances", str(instances), "--threads", str(threads),
            "--p-values", p_text, "--control-variate", "--include-instance-rows",
            "--search-policy", args.search_policy,
            "--sa-iters-per-n", str(args.sa_iters_per_n),
            *metadata_args(args), "--output-durability", "full",
            "--output", str(output),
        ]
        if periodic:
            argv.append("--periodic")
        if held_karp:
            argv.append("--held-karp")
        entries.append(make_entry(
            cell_id=cell_id, output=output, executable=args.exe, argv=argv,
            expected_n=n, expected_instances=instances,
            expected_p_values=p_values_list,
        ))
    return entries


def make_campaign_entries(args, threads: int, lkh: str | None) -> list[dict]:
    directory = args.out_dir / "campaign"
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in args.ps:
        for k in args.ks:
            n = max(k, round(k / p))
            if n > args.max_n:
                continue
            cell_id = f"p{p:g}_k{k}"
            output = directory / f"{cell_id}.json"
            argv = [
                "--N", str(n), "--instances", str(args.instances),
                "--threads", str(threads), "--p-values", f"{p:g}",
                "--periodic", "--control-variate", "--held-karp",
                "--include-instance-rows", "--search-policy", args.search_policy,
                "--sa-iters-per-n", str(args.sa_iters_per_n),
                *metadata_args(args), "--output-durability", "full",
                "--output", str(output),
            ]
            required_oracle = None
            if lkh:
                argv += [
                    "--oracle", "lkh", "--oracle-format", "matrix",
                    "--lkh-path", lkh, "--oracle-tsp-top", "1",
                    "--oracle-subset-top", "1", "--oracle-max-k", "3000",
                    "--oracle-lkh-runs", str(args.lkh_runs),
                    "--restarts", str(args.builtin_restarts),
                ]
                required_oracle = "lkh"
            else:
                argv += ["--restarts", str(args.builtin_restarts)]
            entries.append(make_entry(
                cell_id=cell_id, output=output, executable=args.exe, argv=argv,
                expected_n=n, expected_instances=args.instances,
                expected_p_values=[p], require_oracle=required_oracle,
            ))
    if not entries:
        raise ManifestError("campaign plan is empty after applying --max-n")
    return entries


def load_complete_files(manifest_path: Path, allow_partial: bool) -> list[str]:
    if not manifest_path.is_file():
        raise ManifestError(f"missing campaign manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_complete(manifest, allow_partial=allow_partial)
    files = []
    for entry in manifest["entries"]:
        if entry.get("status") != "complete":
            continue
        validate_result(Path(entry["output"]), entry)
        files.append(entry["output"])
    if len(files) < 2:
        raise ManifestError("fewer than two verified campaign cells are available")
    return files


def run_analysis(args, log_stream) -> None:
    files = load_complete_files(
        args.out_dir / "campaign" / "campaign-manifest.json", args.allow_partial
    )
    report = args.out_dir / "analysis_report.txt"
    commands = [
        [sys.executable, str(HERE / "extrapolate_fpN.py"), *files],
        [
            sys.executable, str(HERE / "analyze_campaign.py"),
            "--pmax", str(args.pmax), "--boot", str(args.boot),
            "--solver-policy-id", args.solver_policy_id,
            "--fidelity-level", args.fidelity_level,
            "--control-variate-mode", "required",
            "--plot", str(args.out_dir / "fit.png"), *files,
        ],
    ]
    with report.open("w", encoding="utf-8") as output:
        for command in commands:
            completed = subprocess.run(command, capture_output=True, text=True)
            output.write("$ " + " ".join(command) + "\n")
            output.write(completed.stdout + "\n" + completed.stderr + "\n")
            if completed.returncode != 0:
                raise ManifestError(
                    f"analysis command failed: {' '.join(command[:2])}"
                )
            log(completed.stdout.rstrip(), log_stream)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--exe", required=True)
    parser.add_argument("--lkh-path")
    parser.add_argument("--allow-oracle-fallback", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("study"))
    parser.add_argument("--stage", choices=("all", "sweep", "campaign", "analysis"), default="all")
    parser.add_argument("--instances", type=int, default=24)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--ps", default="0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--ks", default="250,500,1000,2000")
    parser.add_argument("--campaign-id", default="full-study")
    parser.add_argument("--campaign-shard", type=int, default=0)
    parser.add_argument("--replicate-offset", type=int, default=0)
    parser.add_argument("--point-seed", type=int, default=2024)
    parser.add_argument("--search-seed", type=int, default=2024)
    parser.add_argument("--solver-policy-id", default="publication")
    parser.add_argument("--fidelity-level", default="strong")
    parser.add_argument(
        "--search-policy", required=True,
        choices=("legacy-balanced", "heldout-balanced", "heldout-quality"),
    )
    parser.add_argument("--allow-nonpublication-policy", action="store_true")
    parser.add_argument("--lkh-runs", type=int, default=10)
    parser.add_argument("--builtin-restarts", type=int, default=8)
    parser.add_argument("--sa-iters-per-n", type=int, required=True)
    parser.add_argument("--max-n", type=int, default=300000)
    parser.add_argument("--pmax", type=float, default=0.2)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--batch-timeout", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    executable = Path(args.exe)
    if not executable.is_file() and shutil.which(args.exe) is None:
        parser.error(f"executable not found: {args.exe}")
    args.exe = str(executable.resolve()) if executable.is_file() else str(Path(shutil.which(args.exe)).resolve())
    if args.instances < 1 or args.sa_iters_per_n < 1:
        parser.error("--instances and --sa-iters-per-n must be positive")
    if args.campaign_shard < 0 or args.replicate_offset < 0:
        parser.error("campaign shard and replicate offset must be nonnegative")
    if args.search_policy != "heldout-quality" and not args.allow_nonpublication_policy:
        parser.error(
            "publication studies require --search-policy heldout-quality; "
            "use --allow-nonpublication-policy for exploratory work"
        )
    try:
        args.ps = [float(value) for value in args.ps.split(",")]
        args.ks = [int(value) for value in args.ks.split(",")]
    except ValueError as exc:
        parser.error(str(exc))
    if not args.ps or any(not 0.0 < value <= 1.0 for value in args.ps):
        parser.error("--ps values must lie in (0,1]")
    if not args.ks or any(value < 3 for value in args.ks):
        parser.error("--ks values must be >= 3")

    lkh = detect_lkh(args.lkh_path)
    if args.stage in {"all", "campaign"} and not lkh and not args.allow_oracle_fallback:
        parser.error(
            "LKH is unavailable; publication methodology will not silently fall back. "
            "Install/provide LKH or pass --allow-oracle-fallback explicitly."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    timeout = args.batch_timeout or None

    try:
        with (args.out_dir / "run.log").open("a", encoding="utf-8") as log_stream:
            log("################ full study run ################", log_stream)
            log(
                f"exe={args.exe} threads={threads} policy={args.search_policy} "
                f"sa-iters-per-n={args.sa_iters_per_n} LKH={lkh or 'explicit fallback'}",
                log_stream,
            )
            if args.stage in {"all", "sweep"}:
                entries = make_sweep_entries(args, threads)
                execute_plan(
                    args.out_dir / "sweep" / "sweep-manifest.json",
                    {"kind": "robustness-sweep", "campaign_id": args.campaign_id,
                     "search_policy": args.search_policy,
                     "sa_iters_per_n": args.sa_iters_per_n},
                    entries, timeout=timeout, allow_partial=args.allow_partial,
                    dry_run=args.dry_run, log_stream=log_stream,
                )
            if args.stage in {"all", "campaign"}:
                entries = make_campaign_entries(args, threads, lkh)
                execute_plan(
                    args.out_dir / "campaign" / "campaign-manifest.json",
                    {"kind": "publication-campaign", "campaign_id": args.campaign_id,
                     "search_policy": args.search_policy,
                     "sa_iters_per_n": args.sa_iters_per_n,
                     "oracle": "lkh" if lkh else "explicit-built-in-fallback"},
                    entries, timeout=timeout, allow_partial=args.allow_partial,
                    dry_run=args.dry_run, log_stream=log_stream,
                )
            if args.stage in {"all", "analysis"} and not args.dry_run:
                run_analysis(args, log_stream)
            log("################ done ################", log_stream)
    except ManifestError as exc:
        print(f"study failed closed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
