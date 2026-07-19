#!/usr/bin/env python3
"""Run repeatable Aldous TSP benchmark and ablation scenarios.

The script shells out to the CLI so benchmark runs exercise the same entry point
used in experiments. It writes one JSON file per scenario plus a CSV manifest
with runtime, quality, move-counter, oracle, and ablation metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Scenario:
    name: str
    n: int
    instances: int
    p_values: str
    mode: str = "balanced"
    sa_iters: int = 5000
    restarts: int = 2
    tsp_restarts: int = 2
    tsp_ils: int = 80
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    suite: str = "standard"


SMOKE_SCENARIOS = [
    Scenario("tiny-balanced", 60, 2, "0.05,0.10,0.25,0.50,1.0", sa_iters=250, tsp_ils=8, suite="smoke"),
    Scenario("small-hybrid", 120, 2, "0.02,0.05,0.10,0.25,0.50,0.80,1.0", mode="hybrid", sa_iters=900, tsp_ils=24, suite="smoke"),
]

LARGER_SCENARIOS = [
    Scenario("medium-balanced", 180, 2, "0.02,0.05,0.10,0.20,0.40,0.60,0.80,1.0", sa_iters=1400, restarts=2, tsp_restarts=2, tsp_ils=32, suite="larger"),
    Scenario("medium-hybrid", 180, 2, "0.02,0.03,0.05,0.10,0.20,0.50,0.80,1.0", mode="hybrid", sa_iters=1600, restarts=2, tsp_restarts=2, tsp_ils=36, suite="larger"),
]

ABLATION_SCENARIOS = [
    Scenario("ablation-baseline", 140, 2, "0.05,0.10,0.25,0.50,1.0", mode="hybrid", sa_iters=1000, restarts=2, tsp_restarts=2, tsp_ils=28, suite="ablation"),
    Scenario("ablation-no-two-opt", 140, 2, "0.05,0.10,0.25,0.50,1.0", mode="hybrid", sa_iters=1000, restarts=2, tsp_restarts=2, tsp_ils=28, extra_args=("--disable-two-opt",), suite="ablation"),
    Scenario("ablation-no-or-opt", 140, 2, "0.05,0.10,0.25,0.50,1.0", mode="hybrid", sa_iters=1000, restarts=2, tsp_restarts=2, tsp_ils=28, extra_args=("--disable-or-opt",), suite="ablation"),
    Scenario("ablation-no-lns", 140, 2, "0.05,0.10,0.25,0.50,1.0", mode="hybrid", sa_iters=1000, restarts=2, tsp_restarts=2, tsp_ils=28, extra_args=("--disable-pair-exchange", "--disable-ruin-recreate"), suite="ablation"),
    Scenario("ablation-no-relink", 140, 2, "0.05,0.10,0.25,0.50,1.0", mode="hybrid", sa_iters=1000, restarts=2, tsp_restarts=2, tsp_ils=28, extra_args=("--disable-path-relink",), suite="ablation"),
]


def choose_scenarios(suite: str) -> list[Scenario]:
    if suite == "smoke":
        return SMOKE_SCENARIOS
    if suite == "larger":
        return LARGER_SCENARIOS
    if suite == "ablation":
        return ABLATION_SCENARIOS
    if suite == "all":
        return [*SMOKE_SCENARIOS, *LARGER_SCENARIOS, *ABLATION_SCENARIOS]
    return [*SMOKE_SCENARIOS, *LARGER_SCENARIOS]


def scenario_command(exe: Path, out_path: Path, scenario: Scenario, threads: int, extra: Iterable[str]) -> list[str]:
    return [
        str(exe),
        "--mode", scenario.mode,
        "--N", str(scenario.n),
        "--instances", str(scenario.instances),
        "--threads", str(threads),
        "--p-values", scenario.p_values,
        "--restarts", str(scenario.restarts),
        "--sa-iters", str(scenario.sa_iters),
        "--tsp-restarts", str(scenario.tsp_restarts),
        "--tsp-ils", str(scenario.tsp_ils),
        "--output", str(out_path),
        "--force",
        *scenario.extra_args,
        *extra,
    ]


def best_summary(doc: dict) -> tuple[str | None, float | None]:
    rows = doc.get("summary_rows")
    if rows:
        best = min(rows, key=lambda row: float(row["mean"]))
        return str(best["p"]), float(best["mean"])
    best_p = None
    best_mean = None
    for p_key, row in doc.get("summary", {}).items():
        mean = float(row["mean"])
        if best_mean is None or mean < best_mean:
            best_mean = mean
            best_p = p_key
    return best_p, best_mean


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeatable benchmark and ablation scenarios for aldous_tsp.")
    parser.add_argument("--exe", default="build/aldous_tsp", help="Path to aldous_tsp executable")
    parser.add_argument("--out-dir", default="benchmark-runs", help="Output directory")
    parser.add_argument("--threads", type=int, default=1, help="Worker threads passed to the CLI")
    parser.add_argument("--suite", choices=["smoke", "standard", "larger", "ablation", "all"], default="standard", help="Scenario suite to run")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra arguments appended to every CLI run")
    args = parser.parse_args()

    exe = Path(args.exe)
    exe_check = exe if exe.is_absolute() else Path.cwd() / exe
    if not exe_check.exists():
        print(f"Executable not found: {exe}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    rows: list[dict[str, object]] = []
    for scenario in choose_scenarios(args.suite):
        json_path = out_dir / f"{scenario.name}.json"
        cmd = scenario_command(exe, json_path, scenario, args.threads, args.extra)
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        doc = json.loads(json_path.read_text())
        stats = doc.get("search_stats", {})
        best_p, best_mean = best_summary(doc)
        rows.append(
            {
                "scenario": scenario.name,
                "suite": scenario.suite,
                "N": scenario.n,
                "instances": scenario.instances,
                "mode": scenario.mode,
                "extra_args": " ".join(scenario.extra_args),
                "wall_seconds": doc.get("wall_seconds", 0.0),
                "best_p": best_p,
                "best_mean": best_mean,
                "two_opt_improvements": stats.get("two_opt_improvements", 0),
                "or_opt_improvements": stats.get("or_opt_improvements", 0),
                "subset_swap_improvements": stats.get("subset_swap_improvements", 0),
                "highp_exchange_improvements": stats.get("highp_exchange_improvements", 0),
                "knn_build_seconds": stats.get("knn_build_seconds", 0.0),
                "knn_requested_grid_instances": stats.get("knn_requested_grid_instances", 0),
                "knn_effective_grid_instances": stats.get("knn_effective_grid_instances", 0),
                "knn_effective_bruteforce_instances": stats.get("knn_effective_bruteforce_instances", 0),
                "knn_bruteforce_fallback_instances": stats.get("knn_bruteforce_fallback_instances", 0),
                "knn_grid_cell_capped_instances": stats.get("knn_grid_cell_capped_instances", 0),
                "grid_cell_effective_min": stats.get("grid_cell_effective_min", 0.0),
                "grid_cell_effective_max": stats.get("grid_cell_effective_max", 0.0),
                "tsp_seconds": stats.get("tsp_seconds", 0.0),
                "subset_seconds": stats.get("subset_seconds", 0.0),
                "pair_exchange_improvements": stats.get("pair_exchange_improvements", 0),
                "ruin_recreate_improvements": stats.get("ruin_recreate_improvements", 0),
                "path_relink_feasible": stats.get("path_relink_feasible", 0),
                "path_relink_best_improvements": stats.get("path_relink_best_improvements", 0),
                "oracle_calls": stats.get("oracle_calls", 0),
                "oracle_improved": stats.get("oracle_improved", 0),
                "oracle_gain": stats.get("oracle_gain", 0.0),
                "json": json_path.name,
            }
        )

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest_path}")

    manifest_json_path = out_dir / "manifest.json"
    manifest_doc = {
        "suite": args.suite,
        "threads": args.threads,
        "exe": str(exe),
        "extra_args": list(args.extra),
        "scenarios": rows,
    }
    manifest_json_path.write_text(json.dumps(manifest_doc, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_json_path}")

    ablation_rows = [row for row in rows if row.get("suite") == "ablation"]
    if ablation_rows:
        ablation_manifest_path = out_dir / "ablation_manifest.csv"
        with ablation_manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ablation_rows)
        print(f"Wrote {ablation_manifest_path}")
        ablation_manifest_json_path = out_dir / "ablation_manifest.json"
        ablation_manifest_json_path.write_text(json.dumps({"scenarios": ablation_rows}, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {ablation_manifest_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
