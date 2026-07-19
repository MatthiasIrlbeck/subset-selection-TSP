#!/usr/bin/env python3
"""Paired hot-path performance and quality canaries.

Each baseline/candidate pair uses the same seed and complete CLI configuration.
Execution order alternates by repetition to reduce thermal and scheduler bias.
The production suite is intentionally large enough to expose torus-distance,
pair-exchange, and subset-swap regressions; CI runs only the smoke suite.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Scenario:
    name: str
    n: int
    instances: int
    p_values: str
    seed: int
    sa_iters: int
    restarts: int = 1
    extra: tuple[str, ...] = ()
    focus_phase: str | None = None


COMMON = (
    "--threads", "1",
    "--restart-threads", "1",
    "--tsp-restarts", "1",
    "--tsp-ils", "0",
    "--oracle", "none",
)

SMOKE = (
    Scenario(
        "smoke-torus",
        160,
        1,
        "0.4",
        73001,
        160,
        extra=("--periodic", "--disable-pair-exchange", "--disable-subset-swap",
               "--disable-ruin-recreate", "--disable-path-relink"),
        focus_phase="sa_seconds",
    ),
    Scenario(
        "smoke-neighborhoods",
        180,
        1,
        "0.4",
        73002,
        120,
        extra=("--pair-exchange-passes", "1", "--subset-swap-passes", "1",
               "--ruin-recreate-rounds", "0", "--path-relink-top", "0"),
        focus_phase="pair_exchange_seconds",
    ),
)

PRODUCTION = (
    Scenario(
        "torus-distance-n5000-k2000",
        5000,
        1,
        "0.4",
        73101,
        2000,
        extra=("--periodic", "--disable-pair-exchange", "--disable-subset-swap",
               "--disable-ruin-recreate", "--disable-path-relink"),
        focus_phase="sa_seconds",
    ),
    Scenario(
        "pair-exchange-n5000-k2000",
        5000,
        1,
        "0.4",
        73102,
        1000,
        extra=("--pair-exchange-passes", "1", "--disable-subset-swap",
               "--disable-ruin-recreate", "--disable-path-relink"),
        focus_phase="pair_exchange_seconds",
    ),
    Scenario(
        "subset-swap-n5000-k2000",
        5000,
        1,
        "0.4",
        73103,
        1000,
        extra=("--subset-swap-passes", "1", "--disable-pair-exchange",
               "--disable-ruin-recreate", "--disable-path-relink"),
        focus_phase="subset_swap_seconds",
    ),
    Scenario(
        "combined-n5000-k2000",
        5000,
        2,
        "0.4",
        73104,
        20000,
        restarts=2,
        extra=("--pair-exchange-passes", "1", "--subset-swap-passes", "1",
               "--ruin-recreate-rounds", "1", "--path-relink-top", "0"),
    ),
)


def executable(path: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise argparse.ArgumentTypeError(f"executable not found: {result}")
    return result


def summary_means(doc: dict) -> dict[str, float]:
    rows = doc.get("summary_rows", [])
    if rows:
        return {f"{float(row['p']):g}": float(row["mean"]) for row in rows}
    return {f"{float(key):g}": float(row["mean"]) for key, row in doc.get("summary", {}).items()}


def phase_value(doc: dict, key: str | None) -> float | None:
    if key is None:
        return None
    phases = doc.get("search_stats", {}).get("phase_timing", {})
    value = phases.get(key)
    return None if value is None else float(value)


def scenario_command(exe: Path, output: Path, scenario: Scenario, extra: Iterable[str]) -> list[str]:
    return [
        str(exe),
        "--N", str(scenario.n),
        "--instances", str(scenario.instances),
        "--p-values", scenario.p_values,
        "--seed", str(scenario.seed),
        "--sa-iters", str(scenario.sa_iters),
        "--restarts", str(scenario.restarts),
        *COMMON,
        *scenario.extra,
        *extra,
        "--output", str(output),
        "--force",
    ]


def run_once(label: str, exe: Path, scenario: Scenario, repetition: int,
             out_dir: Path, extra: Iterable[str]) -> dict:
    output = out_dir / f"{scenario.name}-{label}-r{repetition}.json"
    cmd = scenario_command(exe, output, scenario, extra)
    print("+", " ".join(cmd), flush=True)
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    process_wall = time.perf_counter() - started
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"{label} failed for {scenario.name} with exit {proc.returncode}")
    doc = json.loads(output.read_text())
    return {
        "label": label,
        "repetition": repetition,
        "process_wall_seconds": process_wall,
        "reported_wall_seconds": float(doc.get("wall_seconds", process_wall)),
        "means": summary_means(doc),
        "focus_phase_seconds": phase_value(doc, scenario.focus_phase),
        "output": output.name,
    }


def select_scenarios(suite: str) -> tuple[Scenario, ...]:
    if suite == "smoke":
        return SMOKE
    if suite == "production":
        return PRODUCTION
    return (*SMOKE, *PRODUCTION)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def aggregate(scenario: Scenario, runs: list[dict]) -> dict:
    by_label = {label: [run for run in runs if run["label"] == label]
                for label in ("baseline", "candidate")}
    result: dict[str, object] = {
        "scenario": scenario.name,
        "N": scenario.n,
        "instances": scenario.instances,
        "p_values": scenario.p_values,
        "seed": scenario.seed,
        "sa_iters": scenario.sa_iters,
        "restarts": scenario.restarts,
        "focus_phase": scenario.focus_phase,
        "runs": runs,
    }
    for label, samples in by_label.items():
        result[f"{label}_wall_median"] = median([r["reported_wall_seconds"] for r in samples])
        phase_samples = [r["focus_phase_seconds"] for r in samples
                         if r["focus_phase_seconds"] is not None]
        result[f"{label}_focus_phase_median"] = median(phase_samples) if phase_samples else None
        p_keys = sorted(samples[0]["means"])
        result[f"{label}_means"] = {
            key: median([float(r["means"][key]) for r in samples]) for key in p_keys
        }
    baseline_wall = float(result["baseline_wall_median"])
    candidate_wall = float(result["candidate_wall_median"])
    result["wall_speedup"] = baseline_wall / candidate_wall if candidate_wall > 0.0 else None
    bp = result["baseline_focus_phase_median"]
    cp = result["candidate_focus_phase_median"]
    result["focus_phase_speedup"] = (float(bp) / float(cp)) if bp is not None and cp not in (None, 0.0) else None
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paired production hot-path canaries.")
    parser.add_argument("--baseline-exe", required=True, type=executable)
    parser.add_argument("--candidate-exe", required=True, type=executable)
    parser.add_argument("--out-dir", default="hotpath-canary")
    parser.add_argument("--suite", choices=("smoke", "production", "all"), default="production")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--max-wall-ratio", type=float, default=1.05,
                        help="maximum candidate/baseline median wall ratio")
    parser.add_argument("--max-quality-regression", type=float, default=1e-8,
                        help="maximum absolute increase in any paired mean L/k")
    parser.add_argument("--min-speedup", type=float, default=0.0,
                        help="optional minimum median wall speedup for every scenario")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be >= 1")
    if args.max_wall_ratio <= 0.0:
        parser.error("--max-wall-ratio must be > 0")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregates: list[dict] = []
    failures: list[str] = []

    for scenario in select_scenarios(args.suite):
        runs: list[dict] = []
        for repetition in range(args.repetitions):
            order = (("baseline", args.baseline_exe), ("candidate", args.candidate_exe))
            if repetition & 1:
                order = tuple(reversed(order))
            for label, exe in order:
                runs.append(run_once(label, exe, scenario, repetition, out_dir, args.extra))
        row = aggregate(scenario, runs)
        aggregates.append(row)
        speedup = float(row["wall_speedup"])
        print(f"{scenario.name}: wall speedup {speedup:.3f}x", flush=True)

        baseline_means = row["baseline_means"]
        candidate_means = row["candidate_means"]
        assert isinstance(baseline_means, dict) and isinstance(candidate_means, dict)
        for key, baseline_mean in baseline_means.items():
            candidate_mean = float(candidate_means[key])
            regression = candidate_mean - float(baseline_mean)
            print(f"  p={key}: baseline={float(baseline_mean):.9f} "
                  f"candidate={candidate_mean:.9f} delta={regression:+.3g}")
            if args.check and regression > args.max_quality_regression:
                failures.append(
                    f"{scenario.name} p={key}: quality regression {regression:.6g} exceeds "
                    f"{args.max_quality_regression:.6g}")
        wall_ratio = 1.0 / speedup if speedup > 0.0 else float("inf")
        if args.check and wall_ratio > args.max_wall_ratio:
            failures.append(
                f"{scenario.name}: candidate/baseline wall ratio {wall_ratio:.3f} exceeds "
                f"{args.max_wall_ratio:.3f}")
        if args.check and args.min_speedup > 0.0 and speedup < args.min_speedup:
            failures.append(
                f"{scenario.name}: speedup {speedup:.3f} below {args.min_speedup:.3f}")

    manifest = {
        "suite": args.suite,
        "repetitions": args.repetitions,
        "baseline_exe": str(args.baseline_exe),
        "candidate_exe": str(args.candidate_exe),
        "max_wall_ratio": args.max_wall_ratio,
        "max_quality_regression": args.max_quality_regression,
        "min_speedup": args.min_speedup,
        "scenarios": aggregates,
        "failures": failures,
    }
    manifest_json = out_dir / "hotpath_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    csv_rows = []
    for row in aggregates:
        csv_rows.append({
            "scenario": row["scenario"],
            "N": row["N"],
            "instances": row["instances"],
            "p_values": row["p_values"],
            "baseline_wall_median": row["baseline_wall_median"],
            "candidate_wall_median": row["candidate_wall_median"],
            "wall_speedup": row["wall_speedup"],
            "focus_phase": row["focus_phase"],
            "baseline_focus_phase_median": row["baseline_focus_phase_median"],
            "candidate_focus_phase_median": row["candidate_focus_phase_median"],
            "focus_phase_speedup": row["focus_phase_speedup"],
        })
    manifest_csv = out_dir / "hotpath_manifest.csv"
    with manifest_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {manifest_json}")
    print(f"Wrote {manifest_csv}")

    if failures:
        for failure in failures:
            print(f"hot-path canary failure: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
