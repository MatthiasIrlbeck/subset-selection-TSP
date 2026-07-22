#!/usr/bin/env python3
"""Run a provenance-checked integration matrix against real LKH/Concorde.

The test remains optional for ordinary developer builds.  With ``--require`` or
one of the solver-specific requirement flags it becomes a release gate.  Every
case validates native JSON against the repository schema and records the exact
binary SHA-256 used by both the harness and the C++ oracle context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA = ROOT / "schema" / "results.schema.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:512]
    return "unknown"


def probe_version(path: Path) -> str:
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return first_line(completed.stdout)


def binary_identity(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve(strict=True)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "version_probe": probe_version(path),
    }


def validate_schema(document: dict[str, Any], schema_path: Path) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError as exc:  # pragma: no cover - CI installs jsonschema
        raise RuntimeError("jsonschema is required for real-oracle validation") from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(document),
        key=lambda error: list(error.path),
    )
    if errors:
        first = errors[0]
        where = "/".join(str(part) for part in first.path) or "<root>"
        raise RuntimeError(f"schema validation failed at {where}: {first.message}")


def copied_solver_path(identity: dict[str, Any], solver: str, out_dir: Path) -> Path:
    source = Path(identity["path"])
    suffix = source.suffix if os.name == "nt" else ""
    destination_dir = out_dir / "solver binaries with spaces"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"real {solver} binary${suffix}"
    shutil.copy2(source, destination)
    if os.name != "nt":
        destination.chmod(destination.stat().st_mode | 0o111)
    if sha256_file(destination) != identity["sha256"]:
        raise RuntimeError(f"copied {solver} binary changed content")
    return destination


def run_case(
    exe: Path,
    solver: str,
    solver_path: Path,
    identity: dict[str, Any],
    fmt: str,
    periodic: bool,
    out_dir: Path,
    schema_path: Path,
    case_index: int,
) -> dict[str, Any]:
    geometry = "periodic" if periodic else "open"
    case_id = f"{solver}-{geometry}-{fmt}"
    output = out_dir / f"{case_id}.json"
    log_path = out_dir / f"{case_id}.log"
    cmd = [
        str(exe),
        "--N", "40",
        "--instances", "1",
        "--threads", "1",
        "--restart-threads", "1",
        "--p-values", "0.5,1.0",
        "--sa-iters", "0",
        "--restarts", "1",
        "--continuation-restarts", "0",
        "--strong-polish-finalists", "1",
        "--tsp-restarts", "2",
        "--tsp-candidate-starts", "2",
        "--tsp-ils", "0",
        "--path-relink-top", "0",
        "--oracle", solver,
        f"--{solver}-path", str(solver_path),
        "--oracle-format", fmt,
        "--oracle-min-k", "3",
        "--oracle-max-k", "64",
        "--oracle-tsp-top", "1",
        "--oracle-subset-top", "1",
        "--oracle-time-limit", "20",
        "--include-instance-rows",
        "--seed", str(910000 + case_index),
        "--output", str(output),
        "--force",
    ]
    if periodic:
        cmd.append("--periodic")
    print("+", " ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{case_id} exited with {completed.returncode}; see {log_path}"
        )
    document = json.loads(output.read_text(encoding="utf-8"))
    validate_schema(document, schema_path)

    config = document["config"]
    expected_path = str(solver_path.resolve())
    if config.get("oracle_resolved") != solver:
        raise RuntimeError(f"{case_id}: resolved {config.get('oracle_resolved')!r}")
    if Path(config.get("oracle_exec_path", "")).resolve() != solver_path.resolve():
        raise RuntimeError(f"{case_id}: executable path provenance mismatch")
    if config.get("oracle_exec_sha256") != identity["sha256"]:
        raise RuntimeError(f"{case_id}: executable SHA-256 provenance mismatch")

    records = document.get("oracle_call_records", [])
    if not records:
        raise RuntimeError(f"{case_id}: no oracle_call_records")
    observed_types = {record.get("type") for record in records}
    if not {"subset", "tsp"}.issubset(observed_types):
        raise RuntimeError(
            f"{case_id}: expected subset and tsp records, got {sorted(observed_types)}"
        )
    for record in records:
        if record.get("status") not in {"improved", "solved_no_improvement"}:
            raise RuntimeError(
                f"{case_id}: oracle call failed: {record.get('error', '')}"
            )
        if record.get("solver") != solver or record.get("format") != fmt:
            raise RuntimeError(f"{case_id}: per-call solver/format mismatch")
        if Path(record.get("exec_path", "")).resolve() != solver_path.resolve():
            raise RuntimeError(f"{case_id}: per-call executable path mismatch")
        if record.get("exec_sha256") != identity["sha256"]:
            raise RuntimeError(f"{case_id}: per-call executable SHA-256 mismatch")
        if record.get("solver_version") != config.get("oracle_version"):
            raise RuntimeError(f"{case_id}: per-call version provenance mismatch")
        before = float(record["before_length"])
        after = float(record["after_length"])
        gain = float(record["gain"])
        if not (before > 0.0 and after > 0.0 and gain >= 0.0):
            raise RuntimeError(f"{case_id}: invalid length/gain telemetry")
        if record["status"] == "improved" and not after < before:
            raise RuntimeError(f"{case_id}: improved status without a shorter tour")

    stats = document["search_stats"]
    if stats.get("oracle_calls") != len(records):
        raise RuntimeError(f"{case_id}: aggregate call count does not reconcile")
    if stats.get("oracle_failed") != 0:
        raise RuntimeError(f"{case_id}: aggregate oracle failures are nonzero")
    return {
        "id": case_id,
        "solver": solver,
        "geometry": geometry,
        "format": fmt,
        "command": cmd,
        "output": str(output),
        "log": str(log_path),
        "oracle_version": config.get("oracle_version", "unknown"),
        "oracle_calls": len(records),
        "oracle_improved": stats.get("oracle_improved", 0),
        "statuses": [record["status"] for record in records],
        "expected_exec_path": expected_path,
        "exec_sha256": identity["sha256"],
    }


def write_manifest(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "real_oracle_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build/aldous_tsp")
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--lkh-path", default=None)
    parser.add_argument("--concorde-path", default=None)
    parser.add_argument("--require", action="store_true", help="require at least one real solver")
    parser.add_argument("--require-lkh", action="store_true")
    parser.add_argument("--require-concorde", action="store_true")
    parser.add_argument(
        "--no-copied-path",
        action="store_true",
        help="use the discovered path directly instead of a copy containing spaces and '$'",
    )
    args = parser.parse_args()

    exe = Path(args.exe).expanduser().resolve()
    if not exe.is_file():
        print(f"executable not found: {exe}", file=sys.stderr)
        return 2
    schema_path = args.schema.expanduser().resolve()
    if not schema_path.is_file():
        print(f"schema not found: {schema_path}", file=sys.stderr)
        return 2

    tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
    if args.out_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="aldous_real_oracle_")
        out_dir = Path(tmp_ctx.name)
    else:
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    discovered = {
        "lkh": args.lkh_path or shutil.which("LKH"),
        "concorde": args.concorde_path or shutil.which("concorde"),
    }
    missing_required = []
    if args.require_lkh and not discovered["lkh"]:
        missing_required.append("lkh")
    if args.require_concorde and not discovered["concorde"]:
        missing_required.append("concorde")
    if missing_required:
        payload = {
            "status": "failed",
            "reason": f"required solver(s) unavailable: {', '.join(missing_required)}",
            "solvers": {},
            "cases": [],
        }
        write_manifest(out_dir, payload)
        return 1

    candidates = {name: path for name, path in discovered.items() if path}
    if not candidates:
        payload = {
            "status": "skipped",
            "reason": "No real LKH or Concorde executable found",
            "solvers": {},
            "cases": [],
        }
        write_manifest(out_dir, payload)
        print("No real LKH or Concorde executable found; real-oracle smoke skipped.")
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        return 1 if args.require else 0

    identities: dict[str, dict[str, Any]] = {}
    execution_paths: dict[str, Path] = {}
    errors: list[str] = []
    for solver, path_text in candidates.items():
        try:
            identity = binary_identity(path_text)
            identities[solver] = identity
            execution_paths[solver] = (
                Path(identity["path"])
                if args.no_copied_path
                else copied_solver_path(identity, solver, out_dir)
            )
        except (OSError, RuntimeError) as exc:
            errors.append(f"{solver}: {exc}")

    cases: list[dict[str, Any]] = []
    case_index = 0
    for solver in sorted(execution_paths):
        identity = identities[solver]
        # EUC_2D is meaningful only for the open geometry. An explicit matrix
        # is required to preserve toroidal minimum-image distances.
        matrix = [("matrix", False), ("matrix", True), ("euc2d", False)]
        for fmt, periodic in matrix:
            try:
                cases.append(
                    run_case(
                        exe,
                        solver,
                        execution_paths[solver],
                        identity,
                        fmt,
                        periodic,
                        out_dir,
                        schema_path,
                        case_index,
                    )
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                errors.append(f"{solver}/{fmt}/{'periodic' if periodic else 'open'}: {exc}")
            case_index += 1

    if args.require and not cases:
        errors.append("no real-oracle case completed")
    payload = {
        "status": "failed" if errors else "passed",
        "schema": str(schema_path),
        "solvers": identities,
        "execution_paths": {key: str(value) for key, value in execution_paths.items()},
        "cases": cases,
        "errors": errors,
    }
    write_manifest(out_dir, payload)
    if tmp_ctx is not None:
        tmp_ctx.cleanup()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
