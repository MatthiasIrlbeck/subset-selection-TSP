#!/usr/bin/env python3
"""Migrate Aldous subset-selection TSP result JSON from schema 15 to 16.

Schema 16 makes experiment timing explicit and records the work and sampling
variation of the independently estimated control reference. Historical schema
15 files did not time the control-reference or aggregation phases, so those
unknown durations are represented as zero and recorded as unrecoverable rather
than being presented as measured values.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCHEMA_PATH = ROOT / "schema" / "results-v15.schema.json"
CURRENT_SCHEMA_PATH = ROOT / "schema" / "results.schema.json"
TOOL_NAME = "scripts/migrate_schema15_to16.py"


class MigrationError(RuntimeError):
    """Raised when a result document cannot be migrated safely."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MigrationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MigrationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError("result document must be a JSON object")
    return value


def _prior_steps(metadata: object) -> list[dict[str, Any]]:
    if metadata is None:
        return []
    if not isinstance(metadata, dict) or not isinstance(metadata.get("steps"), list):
        raise MigrationError("schema-15 migration_metadata has an unexpected shape")
    return copy.deepcopy(metadata["steps"])


def migrate_document(
    document: dict[str, Any], source_bytes: bytes | None = None
) -> dict[str, Any]:
    """Return a schema-16 copy of *document*; schema-16 input is idempotent."""
    version = document.get("schema_version")
    if version == 16:
        return copy.deepcopy(document)
    if version != 15:
        raise MigrationError(f"expected schema_version 15, got {version!r}")
    if not isinstance(document.get("config"), dict):
        raise MigrationError("schema-15 document has no config object")

    result = copy.deepcopy(document)
    steps = _prior_steps(result.pop("migration_metadata", None))
    inferred: list[str] = []
    unrecoverable: list[str] = []
    notes: list[str] = []

    config = result["config"]
    if "cv_max_point_ops" not in config:
        config["cv_max_point_ops"] = 100_000_000
        inferred.append("/config/cv_max_point_ops")
        notes.append(
            "/config/cv_max_point_ops: assigned the schema-16 default; schema 15 "
            "did not impose an exact point-operation cap"
        )

    old_wall = float(result.get("wall_seconds", 0.0))
    result["timing"] = {
        "solver_wall_seconds": max(0.0, old_wall),
        "control_reference_seconds": 0.0,
        "aggregation_seconds": 0.0,
        "experiment_wall_seconds": max(0.0, old_wall),
    }
    inferred.extend([
        "/timing/solver_wall_seconds",
        "/timing/control_reference_seconds",
        "/timing/aggregation_seconds",
        "/timing/experiment_wall_seconds",
    ])
    unrecoverable.extend([
        "/timing/control_reference_seconds",
        "/timing/aggregation_seconds",
        "/timing/experiment_wall_seconds",
    ])
    notes.append(
        "/timing: schema 15 wall_seconds stopped before control-reference and "
        "aggregation work; those durations cannot be reconstructed"
    )

    samples = int(result.get("full_bound_expectation_samples", 0) or 0)
    stderr = float(result.get("full_bound_expectation_stderr", 0.0) or 0.0)
    if samples > 0 and "full_bound_expectation_stddev" not in result:
        result["full_bound_expectation_stddev"] = max(0.0, stderr) * math.sqrt(samples)
        inferred.append("/full_bound_expectation_stddev")
    if samples > 0 and "full_bound_expectation_point_operations" not in result:
        result["full_bound_expectation_point_operations"] = int(result.get("N", 0)) * samples
        inferred.append("/full_bound_expectation_point_operations")

    for index, row in enumerate(result.get("summary_rows", [])):
        if not isinstance(row, dict) or "cv_stderr" not in row:
            continue
        if "cv_sampling_stderr" not in row:
            row["cv_sampling_stderr"] = row["cv_stderr"]
            inferred.append(f"/summary_rows/{index}/cv_sampling_stderr")
        if "cv_reference_stderr" not in row:
            row["cv_reference_stderr"] = 0.0
            inferred.append(f"/summary_rows/{index}/cv_reference_stderr")
            unrecoverable.append(f"/summary_rows/{index}/cv_reference_stderr")
    if any("cv_stderr" in row for row in result.get("summary_rows", []) if isinstance(row, dict)):
        notes.append(
            "schema-15 cv_stderr excluded control-reference Monte-Carlo uncertainty; "
            "migrated cv_reference_stderr is an explicit unknown placeholder"
        )

    if source_bytes is None:
        source_bytes = json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    steps.append({
        "source_schema_version": 15,
        "target_schema_version": 16,
        "tool": TOOL_NAME,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "inferred_fields": sorted(set(inferred)),
        "unrecoverable_fields": sorted(set(unrecoverable)),
        "notes": sorted(set(notes)),
    })
    result["schema_version"] = 16
    result["migration_metadata"] = {"steps": steps}
    return result


def _validate(document: dict[str, Any], schema_path: Path) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError as exc:
        raise MigrationError("jsonschema is required for migration validation") from exc
    schema = _load_json(schema_path)
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(document),
        key=lambda error: list(error.path),
    )
    if errors:
        first = errors[0]
        where = "/".join(str(part) for part in first.path) or "<root>"
        raise MigrationError(
            f"document fails schema validation at {where}: {first.message}"
        )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="schema-15 result JSON")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path, help="write migrated JSON here")
    output.add_argument("--in-place", action="store_true", help="atomically replace input")
    parser.add_argument("--stdout", action="store_true", help="write JSON to stdout")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    if args.stdout and (args.output is not None or args.in_place):
        parser.error("--stdout cannot be combined with --output or --in-place")
    if not args.stdout and args.output is None and not args.in_place:
        parser.error("choose --output, --in-place, or --stdout")

    try:
        source_bytes = args.input.read_bytes()
        source = json.loads(source_bytes)
        if not isinstance(source, dict):
            raise MigrationError("result document must be a JSON object")
        if not args.no_validate:
            _validate(
                source,
                CURRENT_SCHEMA_PATH if source.get("schema_version") == 16
                else LEGACY_SCHEMA_PATH,
            )
        migrated = migrate_document(source, source_bytes)
        if not args.no_validate:
            _validate(migrated, CURRENT_SCHEMA_PATH)
        text = json.dumps(migrated, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.stdout:
            print(text, end="")
        else:
            target = args.input if args.in_place else args.output
            assert target is not None
            _atomic_write(target, text)
    except (OSError, json.JSONDecodeError, MigrationError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
