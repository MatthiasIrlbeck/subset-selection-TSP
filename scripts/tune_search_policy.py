#!/usr/bin/env python3
"""Paired, matched-compute tuning for subset-search policies.

The harness executes every policy on identical point and search streams, records
wall and accumulated worker time, and compares per-instance values by stable
point-stream identity. It is intentionally a tuner, not an automatic default
changer: use held-out evaluation instances before promoting a candidate policy.

Policy file format::

    {
      "schema_version": 1,
      "reference_policy": "balanced",
      "policies": [
        {"id": "balanced", "args": []},
        {"id": "broader", "args": ["--restarts", "8",
                                    "--strong-polish-finalists", "3"]}
      ]
    }

Example::

    python3 scripts/tune_search_policy.py \
      --exe build/aldous_tsp \
      --policies config/search_policy_candidates.json \
      --N 2000 --p-values 0.02,0.4,0.8 --instances 20 \
      --sa-iters 5000 --repetitions 3 --out-dir policy-tuning
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Cell = tuple[float, int]
ObservationKey = tuple[int, str, float, int]


@dataclass(frozen=True)
class Policy:
    policy_id: str
    args: tuple[str, ...]
    description: str = ""


@dataclass
class RunRecord:
    policy_id: str
    repetition: int
    command: list[str]
    output: Path
    wall_seconds: float
    worker_seconds: float
    observations: dict[ObservationKey, float]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _ci(values: list[float]) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def _safe_id(value: str) -> str:
    output = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value)
    output = output.strip("-.")
    if not output:
        raise ValueError(f"policy id has no filename-safe characters: {value!r}")
    return output


def load_policies(path: Path) -> tuple[list[Policy], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("policy file schema_version must be 1")
    raw_policies = payload.get("policies")
    if not isinstance(raw_policies, list) or len(raw_policies) < 2:
        raise ValueError("policy file must contain at least two policies")

    policies: list[Policy] = []
    seen: set[str] = set()
    for raw in raw_policies:
        if not isinstance(raw, dict):
            raise ValueError("every policy entry must be an object")
        policy_id = raw.get("id")
        args = raw.get("args", [])
        description = raw.get("description", "")
        if not isinstance(policy_id, str) or not policy_id:
            raise ValueError("every policy requires a nonempty string id")
        if policy_id in seen:
            raise ValueError(f"duplicate policy id: {policy_id}")
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError(f"policy {policy_id}: args must be an array of strings")
        if not isinstance(description, str):
            raise ValueError(f"policy {policy_id}: description must be a string")
        _safe_id(policy_id)
        seen.add(policy_id)
        policies.append(Policy(policy_id, tuple(args), description))

    reference = payload.get("reference_policy", policies[0].policy_id)
    if reference not in seen:
        raise ValueError(f"reference_policy is not defined: {reference}")
    return policies, reference


def extract_observations(document: dict[str, Any], repetition: int) -> dict[ObservationKey, float]:
    rows = document.get("instance_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("solver output has no instance_rows; --include-instance-rows is required")
    observations: dict[ObservationKey, float] = {}
    for row in rows:
        if not row.get("ok", False):
            raise ValueError(f"failed instance row in solver output: {row.get('index')}")
        point_stream = row.get("point_stream_id")
        if not isinstance(point_stream, str) or not point_stream:
            raise ValueError("instance row is missing point_stream_id")
        p_results = row.get("p_results")
        if not isinstance(p_results, list):
            raise ValueError("instance row is missing p_results")
        for result in p_results:
            p = float(result["p"])
            k = int(result["k"])
            value = float(result["value"])
            if not math.isfinite(value):
                raise ValueError("nonfinite per-instance result")
            key = (repetition, point_stream, p, k)
            if key in observations:
                raise ValueError(f"duplicate observation key: {key}")
            observations[key] = value
    return observations


def _worker_seconds(document: dict[str, Any]) -> float:
    stats = document.get("search_stats", {})
    subset = float(stats.get("subset_seconds", 0.0))
    tsp = float(stats.get("tsp_seconds", 0.0))
    if not math.isfinite(subset) or not math.isfinite(tsp) or subset < 0.0 or tsp < 0.0:
        raise ValueError("invalid worker-time telemetry")
    return subset + tsp


def paired_summary(
    reference: dict[ObservationKey, float],
    candidate: dict[ObservationKey, float],
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise ValueError(
            "paired policy outputs do not share the same identified observations; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    by_block: dict[tuple[int, str], list[float]] = {}
    by_cell: dict[Cell, list[float]] = {}
    wins = ties = losses = 0
    for key in sorted(reference):
        repetition, point_stream, p, k = key
        delta = candidate[key] - reference[key]
        by_block.setdefault((repetition, point_stream), []).append(delta)
        by_cell.setdefault((p, k), []).append(delta)
        if delta < -1e-12:
            wins += 1
        elif delta > 1e-12:
            losses += 1
        else:
            ties += 1

    block_means = [_mean(values) for values in by_block.values()]
    mean_delta = _mean(block_means)
    rng = random.Random(seed)
    bootstrap_values: list[float] = []
    if bootstrap > 0:
        for _ in range(bootstrap):
            draw = [block_means[rng.randrange(len(block_means))] for _ in block_means]
            bootstrap_values.append(_mean(draw))

    return {
        "mean_delta": mean_delta,
        "ci": _ci(bootstrap_values),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "blocks": len(block_means),
        "observations": len(reference),
        "cells": [
            {
                "p": p,
                "k": k,
                "mean_delta": _mean(values),
                "wins": sum(value < -1e-12 for value in values),
                "ties": sum(abs(value) <= 1e-12 for value in values),
                "losses": sum(value > 1e-12 for value in values),
            }
            for (p, k), values in sorted(by_cell.items())
        ],
    }


def _policy_order(policies: list[Policy], repetition: int) -> list[Policy]:
    return policies if repetition % 2 == 0 else list(reversed(policies))


def _run_policy(
    exe: Path,
    policy: Policy,
    repetition: int,
    args: argparse.Namespace,
    out_dir: Path,
) -> RunRecord:
    safe = _safe_id(policy.policy_id)
    output = out_dir / f"{repetition:03d}-{safe}.json"
    point_seed = args.point_seed + repetition
    search_seed = args.search_seed + repetition
    replicate_offset = args.replicate_offset + repetition * args.instances
    command = [
        str(exe),
        "--N", str(args.N),
        "--instances", str(args.instances),
        "--threads", str(args.threads),
        "--restart-threads", str(args.restart_threads),
        "--p-values", args.p_values,
        "--sa-iters", str(args.sa_iters),
        "--point-seed", str(point_seed),
        "--search-seed", str(search_seed),
        "--replicate-offset", str(replicate_offset),
        "--campaign-id", args.campaign_id,
        "--campaign-shard", str(repetition),
        "--solver-policy-id", policy.policy_id,
        "--fidelity-level", "policy-tuning",
        "--include-instance-rows",
        "--output", str(output),
        "--force",
    ]
    if args.periodic:
        command.append("--periodic")
    command.extend(args.solver_arg)
    command.extend(policy.args)

    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"policy {policy.policy_id} repetition {repetition} exited "
            f"with status {completed.returncode}"
        )
    document = json.loads(output.read_text(encoding="utf-8"))
    return RunRecord(
        policy_id=policy.policy_id,
        repetition=repetition,
        command=command,
        output=output,
        wall_seconds=float(document["wall_seconds"]),
        worker_seconds=_worker_seconds(document),
        observations=extract_observations(document, repetition),
    )


def summarize(
    policies: list[Policy],
    reference_id: str,
    records: list[RunRecord],
    bootstrap: int,
    seed: int,
    budget_ratio: float,
    budget_tolerance: float,
) -> dict[str, Any]:
    grouped: dict[str, list[RunRecord]] = {policy.policy_id: [] for policy in policies}
    for record in records:
        grouped[record.policy_id].append(record)
    for policy_id, runs in grouped.items():
        if not runs:
            raise ValueError(f"policy has no completed runs: {policy_id}")

    merged: dict[str, dict[ObservationKey, float]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for policy in policies:
        runs = sorted(grouped[policy.policy_id], key=lambda run: run.repetition)
        observations: dict[ObservationKey, float] = {}
        for run in runs:
            overlap = observations.keys() & run.observations.keys()
            if overlap:
                raise ValueError(f"duplicate observations across repetitions: {next(iter(overlap))}")
            observations.update(run.observations)
        merged[policy.policy_id] = observations
        summaries[policy.policy_id] = {
            "description": policy.description,
            "args": list(policy.args),
            "runs": len(runs),
            "median_wall_seconds": statistics.median(run.wall_seconds for run in runs),
            "total_worker_seconds": sum(run.worker_seconds for run in runs),
            "mean_worker_seconds": _mean([run.worker_seconds for run in runs]),
            "observations": len(observations),
        }

    reference_worker = float(summaries[reference_id]["total_worker_seconds"])
    reference_observations = merged[reference_id]
    comparisons: dict[str, dict[str, Any]] = {}
    for index, policy in enumerate(policies):
        policy_id = policy.policy_id
        worker_ratio = (
            float(summaries[policy_id]["total_worker_seconds"]) / reference_worker
            if reference_worker > 0.0
            else float("inf")
        )
        comparison = paired_summary(
            reference_observations,
            merged[policy_id],
            bootstrap,
            seed ^ (0x9E3779B9 * (index + 1)),
        )
        comparison["worker_ratio"] = worker_ratio
        comparison["within_budget"] = abs(worker_ratio - budget_ratio) <= budget_tolerance
        comparisons[policy_id] = comparison

    eligible = [
        policy.policy_id
        for policy in policies
        if comparisons[policy.policy_id]["within_budget"]
    ]
    if not eligible:
        eligible = [reference_id]
    recommended = min(
        eligible,
        key=lambda policy_id: (
            comparisons[policy_id]["mean_delta"],
            comparisons[policy_id]["worker_ratio"],
            policy_id,
        ),
    )

    # Pareto frontier in (worker seconds, paired quality delta), both minimized.
    frontier: list[str] = []
    for candidate in policies:
        cid = candidate.policy_id
        cwork = comparisons[cid]["worker_ratio"]
        cquality = comparisons[cid]["mean_delta"]
        dominated = False
        for other in policies:
            oid = other.policy_id
            if oid == cid:
                continue
            owork = comparisons[oid]["worker_ratio"]
            oquality = comparisons[oid]["mean_delta"]
            if (owork <= cwork and oquality <= cquality) and (
                owork < cwork or oquality < cquality
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(cid)

    return {
        "schema_version": 1,
        "reference_policy": reference_id,
        "budget": {
            "target_worker_ratio": budget_ratio,
            "tolerance": budget_tolerance,
        },
        "policies": summaries,
        "comparisons": comparisons,
        "recommended_within_budget": recommended,
        "pareto_frontier": frontier,
    }


def run_self_test() -> int:
    reference: dict[ObservationKey, float] = {}
    better: dict[ObservationKey, float] = {}
    for repetition in range(2):
        for replicate in range(8):
            point = f"{repetition:02x}{replicate:014x}"
            for p, k in ((0.1, 100), (0.4, 400)):
                key = (repetition, point, p, k)
                reference[key] = 0.75 + 0.01 * p + 0.0001 * replicate
                better[key] = reference[key] - 0.002
    paired = paired_summary(reference, better, 500, 19)
    paired_ok = (
        math.isclose(paired["mean_delta"], -0.002, abs_tol=1e-12)
        and paired["wins"] == len(reference)
        and paired["losses"] == 0
        and paired["ci"][1] < 0.0
    )

    policies = [Policy("reference", ()), Policy("better", ("--restarts", "5"))]
    records = [
        RunRecord("reference", 0, [], Path("r0"), 1.0, 10.0, {
            key: value for key, value in reference.items() if key[0] == 0
        }),
        RunRecord("better", 0, [], Path("b0"), 1.1, 10.5, {
            key: value for key, value in better.items() if key[0] == 0
        }),
        RunRecord("reference", 1, [], Path("r1"), 1.0, 10.0, {
            key: value for key, value in reference.items() if key[0] == 1
        }),
        RunRecord("better", 1, [], Path("b1"), 1.1, 10.5, {
            key: value for key, value in better.items() if key[0] == 1
        }),
    ]
    report = summarize(policies, "reference", records, 200, 7, 1.0, 0.10)
    summary_ok = (
        report["recommended_within_budget"] == "better"
        and set(report["pareto_frontier"]) == {"reference", "better"}
        and report["comparisons"]["better"]["within_budget"] is True
    )

    checks = {"paired block bootstrap": paired_ok, "budgeted recommendation": summary_ok}
    for name, passed in checks.items():
        print(f"self-test: {name:<28} {'PASS' if passed else 'FAIL'}")
    return 0 if all(checks.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build/aldous_tsp")
    parser.add_argument("--policies", default="config/search_policy_candidates.json")
    parser.add_argument("--out-dir", default="policy-tuning")
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--p-values", default="0.02,0.4,0.8")
    parser.add_argument("--instances", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--restart-threads", type=int, default=1)
    parser.add_argument("--sa-iters", type=int, default=5000)
    parser.add_argument("--point-seed", type=int, default=12001)
    parser.add_argument("--search-seed", type=int, default=13001)
    parser.add_argument("--replicate-offset", type=int, default=0)
    parser.add_argument("--campaign-id", default="policy-tuning")
    parser.add_argument("--periodic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--solver-arg", action="append", default=[], metavar="ARG")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=92341)
    parser.add_argument("--budget-ratio", type=float, default=1.0)
    parser.add_argument("--budget-tolerance", type=float, default=0.15)
    parser.add_argument("--max-quality-regression", type=float, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    if args.N < 3 or args.instances < 1 or args.repetitions < 1:
        parser.error("N >= 3, instances >= 1, and repetitions >= 1 are required")
    if args.threads < 1 or args.restart_threads < 1 or args.sa_iters < 0:
        parser.error("thread counts must be positive and sa-iters must be nonnegative")
    if args.bootstrap < 0 or args.budget_ratio <= 0.0 or args.budget_tolerance < 0.0:
        parser.error("invalid bootstrap or budget arguments")

    exe = Path(args.exe).resolve()
    if not exe.is_file():
        parser.error(f"executable does not exist: {exe}")
    policy_path = Path(args.policies)
    policies, reference_id = load_policies(policy_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[RunRecord] = []
    for repetition in range(args.repetitions):
        for policy in _policy_order(policies, repetition):
            records.append(_run_policy(exe, policy, repetition, args, out_dir))

    report = summarize(
        policies,
        reference_id,
        records,
        args.bootstrap,
        args.bootstrap_seed,
        args.budget_ratio,
        args.budget_tolerance,
    )
    report["run_config"] = {
        "N": args.N,
        "p_values": args.p_values,
        "instances": args.instances,
        "repetitions": args.repetitions,
        "periodic": args.periodic,
        "sa_iters": args.sa_iters,
        "point_seed": args.point_seed,
        "search_seed": args.search_seed,
    }
    report["runs"] = [
        {
            "policy_id": record.policy_id,
            "repetition": record.repetition,
            "output": str(record.output),
            "wall_seconds": record.wall_seconds,
            "worker_seconds": record.worker_seconds,
            "command": record.command,
        }
        for record in records
    ]
    report_path = out_dir / "policy_tuning_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\npolicy                   worker/ref    paired delta       95% CI       W/T/L")
    for policy in policies:
        comparison = report["comparisons"][policy.policy_id]
        interval = comparison["ci"]
        print(
            f"{policy.policy_id:<24} {comparison['worker_ratio']:>10.3f} "
            f"{comparison['mean_delta']:>15.7f} "
            f"[{interval[0]:+.7f},{interval[1]:+.7f}] "
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']}"
        )
    print(f"\nrecommended within budget: {report['recommended_within_budget']}")
    print(f"Pareto frontier: {', '.join(report['pareto_frontier'])}")
    print(f"wrote {report_path}")

    if args.max_quality_regression is not None:
        recommended = report["recommended_within_budget"]
        delta = report["comparisons"][recommended]["mean_delta"]
        if delta > args.max_quality_regression:
            print(
                f"recommended policy quality regression {delta:.7g} exceeds "
                f"{args.max_quality_regression:.7g}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
