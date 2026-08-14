#!/usr/bin/env python3
"""Render docs/known_good_benchmarks.md from validation artifact manifests.

The script is intentionally lightweight: it reads the bundled CSV/JSON manifests
under validation_runs/current and produces a human-readable checklist plus compact
observed-result tables. Historical evidence under validation_archive is intentionally
excluded. It is safe to rerun after a fresh validation pass.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt_float(value: object, digits: int = 4) -> str:
    try:
        f = float(str(value))
    except Exception:
        return str(value or "")
    if abs(f) >= 1000 or (0 < abs(f) < 1e-3):
        return f"{f:.3e}"
    return f"{f:.{digits}f}".rstrip("0").rstrip(".")


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    out = []
    out.append("| " + " | ".join(md_cell(h) for h in headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(md_cell(cell) for cell in row) + " |")
    return "\n".join(out)


def render(validation_dir: Path) -> str:
    summary_path = validation_dir / "validation_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    backend = read_csv(validation_dir / "backend-parity" / "parity_manifest.csv")
    benchmark_root = validation_dir / "benchmark-smoke"
    if not benchmark_root.exists():
        benchmark_root = validation_dir / "benchmark-all"
    benchmark = read_csv(benchmark_root / "manifest.csv")
    ablation = read_csv(benchmark_root / "ablation_manifest.csv")
    policy = read_csv(validation_dir / "exhaustive-policy" / "manifest.csv")
    original_compat = read_csv(validation_dir / "original-compat" / "parity_manifest.csv")
    oracle_manifest_path = validation_dir / "real-oracle-smoke" / "real_oracle_manifest.json"
    oracle_manifest = json.loads(oracle_manifest_path.read_text()) if oracle_manifest_path.exists() else {}

    lines: list[str] = []
    lines.append("# Known-good benchmark and validation commands")
    lines.append("")
    lines.append("This page combines reusable validation commands with compact tables rendered from the bundled current-release manifests in `validation_runs/current/`. Historical artifacts under `validation_archive/` are not included.")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/render_validation_report.py --validation-dir validation_runs/current --output docs/known_good_benchmarks.md")
    lines.append("```")
    lines.append("")
    lines.append("The bundled artifacts are smoke/validation runs, not large Monte Carlo evidence. Use them to check release health, deterministic parity, and output-schema stability before running larger experiments.")
    lines.append("")

    if summary:
        bsum = summary.get("benchmark_summary", {})
        lines.append("## Bundled validation summary")
        lines.append("")
        lines.append(md_table(
            ["Check", "Observed result"],
            [
                [
                    "Release/Python CTest inventory",
                    (f"{summary.get('ctest_inventory')} tests; {summary.get('ctest_status', 'see CI')}"
                     if summary.get("ctest_inventory") is not None else summary.get("ctest_status", "see CI")),
                ],
                ["Backend parity", f"max_abs_mean_delta = {summary.get('backend_parity', {}).get('max_abs_mean_delta', 'n/a')}"],
                ["Benchmark suite", f"{bsum.get('scenarios', 'n/a')} scenarios, total wall {fmt_float(bsum.get('total_wall_seconds', bsum.get('total_wall', '')))}s"],
                ["Real oracle smoke", oracle_manifest.get("status", "unknown") + (f" ({oracle_manifest.get('reason')})" if oracle_manifest.get("reason") else "")],
                ["Original-compatible routing", "fake original CLI routing passed; not a solver-quality parity result"],
            ],
        ))
        lines.append("")

    if backend:
        lines.append("## Backend parity manifest")
        lines.append("")
        lines.append(md_table(
            ["Scenario", "Comparison", "Matched p", "Max |Δmean|", "Grid wall (s)", "Brute wall (s)"],
            ([r.get("scenario", ""), r.get("comparison", ""), r.get("matched_p_values", ""), fmt_float(r.get("max_abs_mean_delta", "")), fmt_float(r.get("current_wall_seconds", "")), fmt_float(r.get("bruteforce_wall_seconds", ""))] for r in backend),
        ))
        lines.append("")

    if benchmark:
        lines.append("## Benchmark suite manifest excerpt")
        lines.append("")
        lines.append(md_table(
            ["Scenario", "Suite", "N", "Instances", "Mode", "Wall (s)", "Best p", "Best mean", "2-opt imp", "High-p imp"],
            ([r.get("scenario", ""), r.get("suite", ""), r.get("N", ""), r.get("instances", ""), r.get("mode", ""), fmt_float(r.get("wall_seconds", "")), r.get("best_p", ""), fmt_float(r.get("best_mean", "")), r.get("two_opt_improvements", ""), r.get("highp_exchange_improvements", "")] for r in benchmark),
        ))
        lines.append("")

    if ablation:
        lines.append("## Ablation manifest excerpt")
        lines.append("")
        lines.append(md_table(
            ["Scenario", "Disabled features", "Wall (s)", "Best mean", "2-opt imp", "Or-opt imp", "Pair imp", "LNS imp", "Relink feasible"],
            ([r.get("scenario", ""), r.get("extra_args", "") or "baseline", fmt_float(r.get("wall_seconds", "")), fmt_float(r.get("best_mean", "")), r.get("two_opt_improvements", ""), r.get("or_opt_improvements", ""), r.get("pair_exchange_improvements", ""), r.get("ruin_recreate_improvements", ""), r.get("path_relink_feasible", "")] for r in ablation),
        ))
        lines.append("")

    if policy:
        lines.append("## Exhaustive 2-opt policy comparison")
        lines.append("")
        lines.append(md_table(
            ["Policy", "Wall (s)", "Best mean", "2-opt scans", "2-opt imp", "Or-opt scans", "TSP s", "Subset s"],
            ([r.get("policy", ""), fmt_float(r.get("wall_seconds", "")), fmt_float(r.get("best_mean", "")), r.get("two_opt_scans", ""), r.get("two_opt_improvements", ""), r.get("or_opt_scans", ""), fmt_float(r.get("tsp_seconds", "")), fmt_float(r.get("subset_seconds", ""))] for r in policy),
        ))
        lines.append("")
        try:
            by_policy = {r["policy"]: r for r in policy}
            final_scans = int(float(by_policy["final-only"]["two_opt_scans"]))
            all_scans = int(float(by_policy["all-polish"]["two_opt_scans"]))
            lines.append(f"`final-only` reduced two-opt scans by `{all_scans - final_scans}` relative to `all-polish` in the bundled validation run.")
            lines.append("")
        except (KeyError, TypeError, ValueError):
            # The explanatory delta sentence is optional when legacy rows are incomplete.
            pass

    if original_compat:
        lines.append("## Original-compatible parity routing")
        lines.append("")
        lines.append("This table validates command routing only. It uses `tests/fake_original_cli.py` to ensure current-only flags do not leak into an original-style baseline command. It is **not** a solver-quality comparison with the upstream prototype.")
        lines.append("")
        lines.append(md_table(
            ["Scenario", "Comparison", "Baseline kind", "Matched p", "Current wall (s)", "Baseline wall (s)", "Baseline max |Δmean|"],
            ([r.get("scenario", ""), r.get("comparison", ""), r.get("baseline_kind", ""), r.get("matched_p_values", ""), fmt_float(r.get("current_wall_seconds", "")), fmt_float(r.get("baseline_wall_seconds", "")), fmt_float(r.get("baseline_max_abs_mean_delta", ""))] for r in original_compat),
        ))
        lines.append("")

    lines.append("## Local validation checklist")
    lines.append("")
    lines.append("```bash")
    lines.append("cmake -S . -B build-release -G Ninja \\")
    lines.append("  -DCMAKE_BUILD_TYPE=Release \\")
    lines.append("  -DBUILD_TESTING=ON \\")
    lines.append("  -DALDOUS_TSP_ENABLE_PYTHON_TESTS=ON")
    lines.append("cmake --build build-release --parallel")
    lines.append("ctest --test-dir build-release --output-on-failure")
    lines.append("```")
    lines.append("")
    lines.append("Recommended debug/sanitizer pass:")
    lines.append("")
    lines.append("```bash")
    lines.append("cmake -S . -B build-asan -G Ninja \\")
    lines.append("  -DCMAKE_BUILD_TYPE=Debug \\")
    lines.append("  -DBUILD_TESTING=ON \\")
    lines.append("  -DALDOUS_TSP_ENABLE_SANITIZERS=ON \\")
    lines.append("  -DALDOUS_TSP_ENABLE_PYTHON_TESTS=ON")
    lines.append("cmake --build build-asan --parallel")
    lines.append("ctest --test-dir build-asan --output-on-failure")
    lines.append("```")
    lines.append("")
    lines.append("Package-consumer smoke:")
    lines.append("")
    lines.append("```bash")
    lines.append("cmake --install build-release --prefix install-root")
    lines.append("cmake -S tests/consumer -B build-consumer -G Ninja \\")
    lines.append("  -DCMAKE_PREFIX_PATH=$PWD/install-root")
    lines.append("cmake --build build-consumer --parallel")
    lines.append("```")
    lines.append("")
    lines.append("## Reusable benchmark commands")
    lines.append("")
    lines.append("Backend parity:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/benchmark_parity.py \\")
    lines.append("  --exe build-release/aldous_tsp \\")
    lines.append("  --out-dir parity-current")
    lines.append("```")
    lines.append("")
    lines.append("Full bundled benchmark suite:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/benchmark.py \\")
    lines.append("  --exe build-release/aldous_tsp \\")
    lines.append("  --suite all \\")
    lines.append("  --threads 1 \\")
    lines.append("  --out-dir benchmark-all")
    lines.append("```")
    lines.append("")
    lines.append("Original-vs-current parity requires a compiled original prototype executable:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/benchmark_parity.py \\")
    lines.append("  --exe build-release/aldous_tsp \\")
    lines.append("  --baseline-exe /path/to/original/aldous_tsp \\")
    lines.append("  --baseline-kind original \\")
    lines.append("  --out-dir parity-original")
    lines.append("```")
    lines.append("")
    lines.append("Real oracle smoke requires LKH and/or Concorde installed locally:")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/oracle_real_smoke.py \\")
    lines.append("  --exe build-release/aldous_tsp \\")
    lines.append("  --out-dir real-oracle-smoke")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render known-good benchmark documentation from validation manifests.")
    parser.add_argument("--validation-dir", default="validation_runs/current", help="Directory containing current-release validation manifests")
    parser.add_argument("--output", default="docs/known_good_benchmarks.md", help="Markdown output path")
    args = parser.parse_args()
    validation_dir = Path(args.validation_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(validation_dir), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
