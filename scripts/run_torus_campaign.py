#!/usr/bin/env python3
"""Run a fail-closed publication torus campaign with an exact manifest.

Every planned cell is fingerprinted from the executable's resolved options
before execution. Resume accepts an existing result only when its problem,
method, exact configuration, completion counts, oracle identity, digest-bound
timing receipt, and manifest entry all match. Missing/failed cells make the
command fail unless --allow-partial is explicitly supplied.
"""
from __future__ import annotations

import argparse
import itertools
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


def parse_csv_floats(text: str) -> list[float]:
    values = [float(value) for value in text.split(",") if value.strip()]
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        raise argparse.ArgumentTypeError("p values must lie in (0,1]")
    return values


def parse_csv_ints(text: str) -> list[int]:
    values = [int(value) for value in text.split(",") if value.strip()]
    if not values or any(value < 3 for value in values):
        raise argparse.ArgumentTypeError("k values must be integers >= 3")
    return values


def run_entry(entry: dict, manifest_path: Path, manifest: dict) -> bool:
    output = Path(entry["output"])
    if output.exists():
        validated = validate_result(output, entry)
        update_entry(
            manifest_path, manifest, entry["cell_id"], status="complete",
            result_sha256=validated["sha256"],
        )
        print(f"  resume verified: {entry['cell_id']}")
        return True

    command = [entry["executable"], *entry["argv"]]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().replace("\n", " ")
        update_entry(manifest_path, manifest, entry["cell_id"], status="failed")
        print(f"  FAILED: {entry['cell_id']}: {detail[:400]}", file=sys.stderr)
        return False
    try:
        validated = validate_result(output, entry)
    except ManifestError as exc:
        update_entry(manifest_path, manifest, entry["cell_id"], status="invalid")
        print(f"  INVALID: {entry['cell_id']}: {exc}", file=sys.stderr)
        return False
    update_entry(
        manifest_path, manifest, entry["cell_id"], status="complete",
        result_sha256=validated["sha256"],
    )
    print(f"  complete: {entry['cell_id']}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--exe", required=True)
    parser.add_argument("--lkh-path", required=True)
    parser.add_argument("--ps", type=parse_csv_floats,
                        default=parse_csv_floats("0.02,0.05,0.1,0.2,0.4"))
    parser.add_argument("--ks", type=parse_csv_ints,
                        default=parse_csv_ints("250,500,1000,2000"))
    parser.add_argument("--instances", type=int, default=24)
    parser.add_argument("--lkh-runs", type=int, default=10)
    parser.add_argument(
        "--sa-iters-per-n", type=int, required=True,
        help="validated positive SA iterations per candidate point; flat budgets are refused",
    )
    parser.add_argument(
        "--search-policy", required=True,
        choices=("legacy-balanced", "heldout-balanced", "heldout-quality"),
        help="explicit search-controller policy; publication runs should use heldout-quality",
    )
    parser.add_argument("--allow-nonpublication-policy", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("torus_campaign"))
    parser.add_argument("--campaign-id", default="torus-campaign")
    parser.add_argument("--campaign-shard", type=int, default=0)
    parser.add_argument("--replicate-offset", type=int, default=0)
    parser.add_argument("--point-seed", type=int, default=2024)
    parser.add_argument("--search-seed", type=int, default=2024)
    parser.add_argument("--solver-policy-id", default="publication")
    parser.add_argument("--fidelity-level", default="strong")
    parser.add_argument("--f0", type=float)
    parser.add_argument("--min-n", type=int, default=40)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.instances < 1 or args.lkh_runs < 1 or args.sa_iters_per_n < 1:
        parser.error("instances, LKH runs, and --sa-iters-per-n must be positive")
    if args.campaign_shard < 0 or args.replicate_offset < 0:
        parser.error("campaign shard and replicate offset must be nonnegative")
    if args.search_policy != "heldout-quality" and not args.allow_nonpublication_policy:
        parser.error(
            "publication campaigns require --search-policy heldout-quality; "
            "use --allow-nonpublication-policy for an explicitly exploratory run"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_args = [
        "--campaign-id", args.campaign_id,
        "--campaign-shard", str(args.campaign_shard),
        "--replicate-offset", str(args.replicate_offset),
        "--point-seed", str(args.point_seed),
        "--search-seed", str(args.search_seed),
        "--solver-policy-id", args.solver_policy_id,
        "--fidelity-level", args.fidelity_level,
    ]

    plan: list[tuple[float, int, int]] = []
    for p, k in itertools.product(args.ps, args.ks):
        n = max(k, round(k / p))
        if n >= args.min_n:
            plan.append((p, k, n))
    if not plan:
        parser.error("campaign plan is empty")

    entries = []
    for p, k, n in plan:
        output = args.out_dir / f"p{p:g}_k{k}.json"
        argv = [
            "--N", str(n), "--instances", str(args.instances),
            "--p-values", f"{p:g}", "--periodic", "--control-variate",
            "--held-karp", "--include-instance-rows",
            "--search-policy", args.search_policy,
            "--sa-iters-per-n", str(args.sa_iters_per_n),
            *metadata_args,
            "--oracle", "lkh", "--oracle-format", "matrix",
            "--lkh-path", args.lkh_path,
            "--oracle-tsp-top", "1", "--oracle-subset-top", "1",
            "--oracle-max-k", "3000", "--oracle-lkh-runs", str(args.lkh_runs),
            "--output-durability", "full", "--output", str(output),
        ]
        entries.append(make_entry(
            cell_id=f"p{p:g}_k{k}", output=output, executable=args.exe,
            argv=argv, expected_n=n, expected_instances=args.instances,
            expected_p_values=[p], require_oracle="lkh",
        ))

    campaign = {
        "kind": "torus-publication",
        "campaign_id": args.campaign_id,
        "campaign_shard": args.campaign_shard,
        "solver_policy_id": args.solver_policy_id,
        "fidelity_level": args.fidelity_level,
        "search_policy": args.search_policy,
        "sa_iters_per_n": args.sa_iters_per_n,
        "point_seed": args.point_seed,
        "search_seed": args.search_seed,
    }
    manifest_path = args.out_dir / "campaign-manifest.json"
    try:
        manifest = load_or_create_manifest(manifest_path, campaign, entries)
    except ManifestError as exc:
        parser.error(str(exc))

    print(f"campaign plan: {len(entries)} exact cells")
    for entry in entries:
        expected = entry["expected"]
        print(
            f"  {entry['cell_id']}: N={expected['N']} "
            f"config={expected['configuration_fingerprint'][:12]} "
            f"method={expected['method_fingerprint'][:12]}"
        )
    if args.dry_run:
        return 0

    failures = 0
    for entry in manifest["entries"]:
        try:
            if not run_entry(entry, manifest_path, manifest):
                failures += 1
        except ManifestError as exc:
            failures += 1
            print(f"  INVALID RESUME: {entry['cell_id']}: {exc}", file=sys.stderr)

    try:
        require_complete(manifest, allow_partial=args.allow_partial)
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if failures and not args.allow_partial:
        return 2

    files = [entry["output"] for entry in manifest["entries"]
             if entry.get("status") == "complete"]
    here = Path(__file__).resolve().parent
    extrapolate = here / "extrapolate_fpN.py"
    command = [sys.executable, str(extrapolate), *files]
    if args.f0 is not None:
        command += ["--alpha", "--f0", str(args.f0)]
    completed = subprocess.run(command)
    if completed.returncode != 0:
        return completed.returncode
    figure = args.out_dir / "extrapolation.png"
    subprocess.run(
        [sys.executable, str(extrapolate), "--plot", str(figure), *files],
        check=True,
    )
    print(
        "analysis command:\n  "
        f"{here / 'analyze_campaign.py'} --solver-policy-id "
        f"{args.solver_policy_id} --fidelity-level {args.fidelity_level} "
        + " ".join(files)
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ManifestError as exc:
        print(f"campaign manifest error: {exc}", file=sys.stderr)
        raise SystemExit(2)
