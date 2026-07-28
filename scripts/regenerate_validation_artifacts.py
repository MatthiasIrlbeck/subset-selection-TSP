#!/usr/bin/env python3
"""Regenerate the compact, current-release validation evidence bundle.

Historical evidence belongs under ``validation_archive/``.  This script replaces
``validation_runs/current`` atomically only after every requested smoke command
has completed successfully.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]


def project_version() -> str:
    text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"project\(aldous_tsp VERSION ([^\s)]+)", text)
    if not match:
        raise RuntimeError("could not determine project version")
    return match.group(1)


def current_schema_version() -> int:
    schema = json.loads((ROOT / "schema" / "results.schema.json").read_text(encoding="utf-8"))
    value = schema["properties"]["schema_version"].get("const")
    if not isinstance(value, int):
        raise RuntimeError("current result schema has no integer schema_version const")
    return value


def run_logged(command: Sequence[str], log_path: Path) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"validation command exited with {completed.returncode}: {' '.join(command)}; "
            f"see {log_path}"
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def ctest_inventory(build_dir: Path | None) -> int | None:
    if build_dir is None:
        return None
    completed = subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "-N"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    match = re.search(r"Total Tests:\s*(\d+)", completed.stdout)
    return int(match.group(1)) if match else None



def normalize_staging_paths(stage: Path, output: Path) -> None:
    replacements = [
        (str(stage.resolve()), str(output.relative_to(ROOT))),
        (str(stage.relative_to(ROOT)), str(output.relative_to(ROOT))),
    ]
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        normalized = text
        for source, destination in replacements:
            normalized = normalized.replace(source, destination)
        if normalized != text:
            path.write_text(normalized, encoding="utf-8")

def compact_summary(stage: Path, build_dir: Path | None) -> dict[str, Any]:
    parity_rows = read_csv(stage / "backend-parity" / "parity_manifest.csv")
    benchmark_rows = read_csv(stage / "benchmark-smoke" / "manifest.csv")
    policy_rows = read_csv(stage / "exhaustive-policy" / "manifest.csv")
    oracle_path = stage / "real-oracle-smoke" / "real_oracle_manifest.json"
    oracle = json.loads(oracle_path.read_text(encoding="utf-8")) if oracle_path.exists() else {}

    parity_deltas = [
        float(row["max_abs_mean_delta"])
        for row in parity_rows
        if row.get("max_abs_mean_delta") not in (None, "")
    ]
    policy_scans = {
        row.get("policy", ""): int(float(row.get("two_opt_scans", "0") or 0))
        for row in policy_rows
    }
    scan_reduction = None
    if "final-only" in policy_scans and "all-polish" in policy_scans:
        scan_reduction = policy_scans["all-polish"] - policy_scans["final-only"]

    total_wall = 0.0
    for row in benchmark_rows:
        for key in ("wall_seconds", "elapsed_seconds"):
            if row.get(key) not in (None, ""):
                total_wall += float(row[key])
                break

    inventory = ctest_inventory(build_dir)
    return {
        "project_version": project_version(),
        "schema_version": current_schema_version(),
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_tree": git_value("rev-parse", "HEAD^{tree}"),
        "ctest_inventory": inventory,
        "ctest_status": "run separately; see release validation and CI",
        "backend_parity": {
            "rows": len(parity_rows),
            "max_abs_mean_delta": max(parity_deltas, default=None),
        },
        "benchmark_summary": {
            "scenarios": len(benchmark_rows),
            "total_wall_seconds": total_wall,
        },
        "exhaustive_policy": {
            "policies": sorted(policy_scans),
            "scan_reduction": scan_reduction,
        },
        "original_compatible_routing": "passed with tests/fake_original_cli.py",
        "real_oracle_smoke": oracle,
        "scope": (
            "Compact release-health evidence only; not publication-scale Monte Carlo evidence "
            "and not a substitute for real LKH/Concorde runs."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build-release/aldous_tsp")
    parser.add_argument("--output-dir", default="validation_runs/current")
    parser.add_argument("--build-dir", default="build-release")
    args = parser.parse_args()

    exe = Path(args.exe)
    exe_check = exe if exe.is_absolute() else ROOT / exe
    if not exe_check.is_file():
        print(f"executable not found: {exe}", file=sys.stderr)
        return 2
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    build_dir = Path(args.build_dir) if args.build_dir else None
    if build_dir is not None and not build_dir.is_absolute():
        build_dir = ROOT / build_dir

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aldous-current-validation-", dir=parent) as raw:
        stage = Path(raw) / "current"
        stage.mkdir(parents=True)
        (stage / "ARTIFACT_VERSION").write_text(project_version() + "\n", encoding="utf-8")
        (stage / "ARTIFACT_SCHEMA").write_text(str(current_schema_version()) + "\n", encoding="utf-8")
        (stage / "README.md").write_text(
            "# Current release validation evidence\n\n"
            "These compact deterministic/smoke artifacts were generated by "
            f"solver {project_version()} with result schema {current_schema_version()}. "
            "They validate release health, backend parity, and output contracts; they are not "
            "publication-scale Monte Carlo evidence. Historical artifacts live under "
            "`validation_archive/`.\n",
            encoding="utf-8",
        )

        smoke_dir = stage / "smoke"
        smoke_dir.mkdir()
        run_logged(
            [
                str(exe), "--N", "64", "--instances", "2", "--threads", "1",
                "--restart-threads", "1", "--p-values", "0.1,0.4,1.0",
                "--sa-iters", "80", "--restarts", "2", "--tsp-restarts", "2",
                "--tsp-ils", "4", "--path-relink-top", "0", "--oracle", "none",
                "--output", str((smoke_dir / "quick-smoke.json").relative_to(ROOT)), "--force",
            ],
            smoke_dir / "quick-smoke.log",
        )
        run_logged(
            [sys.executable, "scripts/benchmark_parity.py", "--exe", str(exe),
             "--out-dir", str((stage / "backend-parity").relative_to(ROOT))],
            stage / "backend-parity.log",
        )
        run_logged(
            [sys.executable, "scripts/benchmark.py", "--exe", str(exe), "--suite", "smoke",
             "--threads", "1", "--out-dir", str((stage / "benchmark-smoke").relative_to(ROOT))],
            stage / "benchmark-smoke.log",
        )
        run_logged(
            [
                sys.executable, "scripts/benchmark_exhaustive_policy.py", "--exe", str(exe),
                "--out-dir", str((stage / "exhaustive-policy").relative_to(ROOT)),
                "--N", "72", "--instances", "1", "--threads", "1",
                "--p-values", "0.25,1.0", "--sa-iters", "20", "--restarts", "1",
                "--tsp-restarts", "1", "--tsp-ils", "4", "--final-exhaustive-k", "96",
                "--check",
            ],
            stage / "exhaustive-policy.log",
        )
        profile_dir = stage / "profile"
        profile_dir.mkdir()
        run_logged(
            [
                sys.executable, "scripts/profile_run.py", "--exe", str(exe),
                "--output", str((profile_dir / "profile.json").relative_to(ROOT)),
                "--N", "120", "--instances", "1", "--p-values", "0.2,1.0",
                "--threads", "1", "--extra", "--sa-iters", "50", "--restarts", "1",
                "--tsp-restarts", "1", "--tsp-ils", "2", "--path-relink-top", "0",
            ],
            profile_dir / "profile.log",
        )
        run_logged(
            [
                sys.executable, "scripts/benchmark_parity.py", "--exe", str(exe),
                "--baseline-exe", "tests/fake_original_cli.py", "--baseline-kind", "original",
                "--suite", "original-compat-smoke",
                "--out-dir", str((stage / "original-compat").relative_to(ROOT)),
            ],
            stage / "original-compat.log",
        )
        run_logged(
            [
                sys.executable, "scripts/oracle_real_smoke.py", "--exe", str(exe),
                "--schema", "schema/results.schema.json",
                "--out-dir", str((stage / "real-oracle-smoke").relative_to(ROOT)),
            ],
            stage / "real-oracle-smoke.log",
        )

        normalize_staging_paths(stage, output)
        summary = compact_summary(stage, build_dir)
        (stage / "validation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)

    print(f"regenerated current validation evidence in {output.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
