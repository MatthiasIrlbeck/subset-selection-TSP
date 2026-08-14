#!/usr/bin/env python3
"""Benchmark/parity harness for internal backends and optional baseline executables.

The default mode compares this repository's grid and brute-force KNN backends on
small deterministic scenarios. If ``--baseline-exe`` is provided, the harness can
also compare against either a current-compatible executable or the upstream
original prototype CLI.

The upstream original prototype does not understand newer flags such as
``--p-values``, ``--knn-backend``, ``--final-exhaustive-k`` or neighborhood
ablation controls. Use ``--baseline-kind original`` (or leave ``auto`` if the
baseline's ``--help`` output does not advertise current flags) to run an
original-compatible command line and compare against a current run that also uses
original-compatible defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Scenario:
    name: str
    n: int
    instances: int
    p_values: str | None
    mode: str = "balanced"
    seed: int = 2024
    sa_iters: int = 80
    restarts: int = 1
    tsp_restarts: int = 1
    tsp_ils: int = 8


SCENARIOS = [
    Scenario("tiny-grid-parity", 28, 2, "0.25,0.5,1.0", seed=101, sa_iters=12, tsp_ils=2),
    Scenario("small-hybrid-parity", 36, 2, "0.1,0.3,0.8,1.0", mode="hybrid", seed=202, sa_iters=18, tsp_ils=2),
]

ORIGINAL_COMPAT_SMOKE_SCENARIOS = [
    Scenario("original-compat-tiny", 12, 1, "0.5,1.0", seed=303, sa_iters=0, tsp_ils=0),
]


def executable_prefix(executable: Path) -> list[str]:
    """Return a cross-platform command prefix for native tools or Python scripts."""
    if executable.suffix.lower() == ".py":
        return [sys.executable, str(executable)]
    return [str(executable)]


def run_checked(cmd: Sequence[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def current_cmd(
    exe: Path,
    out: Path,
    scenario: Scenario,
    backend: str | None,
    extra: Iterable[str],
    *,
    original_compatible: bool = False,
) -> list[str]:
    cmd = [
        *executable_prefix(exe),
        "--mode", scenario.mode,
        "--N", str(scenario.n),
        "--instances", str(scenario.instances),
        "--threads", "1",
        "--seed", str(scenario.seed),
        "--restarts", str(scenario.restarts),
        "--sa-iters", str(scenario.sa_iters),
        "--tsp-restarts", str(scenario.tsp_restarts),
        "--tsp-ils", str(scenario.tsp_ils),
        "--output", str(out),
        "--force",
    ]
    if scenario.p_values is not None and not original_compatible:
        cmd.extend(["--p-values", scenario.p_values])
    if backend is not None:
        cmd.extend(["--knn-backend", backend])
    if not original_compatible:
        cmd.extend([
            "--final-exhaustive-k", "128",
            "--pair-exchange-passes", "0",
            "--ruin-recreate-rounds", "0",
            "--path-relink-top", "0",
        ])
        cmd.extend(extra)
    return cmd


def baseline_original_cmd(exe: Path, out: Path, scenario: Scenario, extra: Iterable[str]) -> list[str]:
    # The upstream original prototype supports this minimal option set. It does
    # not support custom p-values or the newer neighborhood/backend flags, so the
    # paired current run also omits those options when comparing to this command.
    cmd = [
        *executable_prefix(exe),
        "--mode", scenario.mode,
        "--N", str(scenario.n),
        "--instances", str(scenario.instances),
        "--threads", "1",
        "--seed", str(scenario.seed),
        "--restarts", str(scenario.restarts),
        "--sa-iters", str(scenario.sa_iters),
        "--tsp-restarts", str(scenario.tsp_restarts),
        "--tsp-ils", str(scenario.tsp_ils),
        "--output", str(out),
        "--force",
    ]
    cmd.extend(extra)
    return cmd


def run_cmd(cmd: Sequence[str], out: Path) -> dict:
    run_checked(cmd)
    return json.loads(out.read_text())


def summary_means(doc: dict) -> dict[str, float]:
    if doc.get("summary_rows"):
        return {format(float(row["p"]), ".17g"): float(row["mean"]) for row in doc["summary_rows"]}
    out: dict[str, float] = {}
    for key, row in doc.get("summary", {}).items():
        try:
            p_key = format(float(key), ".17g")
        except ValueError:
            p_key = str(key)
        out[p_key] = float(row["mean"])
    return out


def max_abs_mean_delta(a: dict[str, float], b: dict[str, float]) -> tuple[float, int]:
    keys = sorted(set(a) & set(b))
    if not keys:
        return math.nan, 0
    return max(abs(a[k] - b[k]) for k in keys), len(keys)


def detect_baseline_kind(baseline: Path) -> str:
    try:
        proc = subprocess.run(
            [*executable_prefix(baseline), "--help"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        )
    except Exception:
        return "original"
    text = proc.stdout or ""
    if "--p-values" in text and "--knn-backend" in text:
        return "current"
    return "original"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backend parity and optional original-prototype benchmark scenarios.")
    parser.add_argument("--exe", default="build/aldous_tsp", help="Path to current aldous_tsp executable")
    parser.add_argument("--baseline-exe", default=None, help="Optional path to original/baseline executable")
    parser.add_argument("--baseline-kind", choices=["auto", "current", "original"], default="auto", help="Baseline CLI compatibility mode")
    parser.add_argument("--out-dir", default="parity-runs", help="Output directory")
    parser.add_argument("--suite", choices=["default", "original-compat-smoke"], default="default", help="Scenario suite to run")
    parser.add_argument("--max-grid-bruteforce-delta", type=float, default=1e-9, help="Allowed mean delta for grid vs brute-force small parity runs")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra arguments appended to current-compatible runs only")
    parser.add_argument("--baseline-extra", nargs=argparse.REMAINDER, default=[], help="Extra arguments appended to baseline runs only")
    args = parser.parse_args()

    exe = Path(args.exe)
    exe_check = exe if exe.is_absolute() else Path.cwd() / exe
    if not exe_check.exists():
        print(f"current executable not found: {exe}", file=sys.stderr)
        return 2
    baseline = Path(args.baseline_exe) if args.baseline_exe else None
    if baseline is not None:
        baseline_check = baseline if baseline.is_absolute() else Path.cwd() / baseline
        if not baseline_check.exists():
            print(f"baseline executable not found: {baseline}", file=sys.stderr)
            return 2

    baseline_kind = "none"
    if baseline is not None:
        baseline_kind = detect_baseline_kind(baseline) if args.baseline_kind == "auto" else args.baseline_kind
        print(f"Baseline compatibility mode: {baseline_kind}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failed = False

    scenarios = ORIGINAL_COMPAT_SMOKE_SCENARIOS if args.suite == "original-compat-smoke" else SCENARIOS

    for scenario in scenarios:
        grid_doc = run_cmd(current_cmd(exe, out_dir / f"{scenario.name}-grid.json", scenario, "grid", args.extra), out_dir / f"{scenario.name}-grid.json")
        brute_doc = run_cmd(current_cmd(exe, out_dir / f"{scenario.name}-bruteforce.json", scenario, "bruteforce", args.extra), out_dir / f"{scenario.name}-bruteforce.json")
        grid_means = summary_means(grid_doc)
        brute_means = summary_means(brute_doc)
        delta, matched = max_abs_mean_delta(grid_means, brute_means)
        if not math.isfinite(delta) or delta > args.max_grid_bruteforce_delta:
            failed = True
        rows.append({
            "scenario": scenario.name,
            "comparison": "grid-vs-bruteforce",
            "baseline_kind": "none",
            "matched_p_values": matched,
            "max_abs_mean_delta": delta,
            "current_wall_seconds": grid_doc.get("wall_seconds", 0.0),
            "bruteforce_wall_seconds": brute_doc.get("wall_seconds", 0.0),
            "baseline_wall_seconds": "",
            "baseline_max_abs_mean_delta": "",
        })
        if baseline is not None:
            if baseline_kind == "original":
                current_out = out_dir / f"{scenario.name}-current-original-compatible.json"
                base_out = out_dir / f"{scenario.name}-baseline-original.json"
                current_doc = run_cmd(current_cmd(exe, current_out, scenario, None, [], original_compatible=True), current_out)
                base_doc = run_cmd(baseline_original_cmd(baseline, base_out, scenario, args.baseline_extra), base_out)
            else:
                current_doc = grid_doc
                base_out = out_dir / f"{scenario.name}-baseline-current-compatible.json"
                base_doc = run_cmd(current_cmd(baseline, base_out, scenario, None, [*args.extra, *args.baseline_extra]), base_out)
            base_delta, base_matched = max_abs_mean_delta(summary_means(current_doc), summary_means(base_doc))
            rows.append({
                "scenario": scenario.name,
                "comparison": "current-vs-baseline",
                "baseline_kind": baseline_kind,
                "matched_p_values": base_matched,
                "max_abs_mean_delta": "",
                "current_wall_seconds": current_doc.get("wall_seconds", 0.0),
                "bruteforce_wall_seconds": "",
                "baseline_wall_seconds": base_doc.get("wall_seconds", 0.0),
                "baseline_max_abs_mean_delta": base_delta,
            })

    manifest = out_dir / "parity_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "parity_manifest.json").write_text(json.dumps({"rows": rows, "baseline_kind": baseline_kind}, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
