#!/usr/bin/env python3
"""Migrate Aldous subset-selection TSP result JSON from schema 13 to 14.

Schema 14 requires a complete configuration record.  Schema-13 documents can
therefore be upgraded only by deriving a few fields from existing values and by
filling later options with compatibility defaults.  Every synthesized field is
listed in ``migration_metadata``; fields whose historical value cannot be
proven are listed separately under ``unrecoverable_fields``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "config" / "options-v14.json"
CURRENT_SCHEMA_PATH = ROOT / "schema" / "results-v14.schema.json"
LEGACY_SCHEMA_PATH = ROOT / "schema" / "results-v13.schema.json"
TOOL_NAME = "scripts/migrate_schema13_to14.py"


class MigrationError(RuntimeError):
    """Raised when a document cannot be migrated without ambiguity or loss."""


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


def _metadata() -> dict[str, Any]:
    value = _load_json(METADATA_PATH)
    if value.get("schema_version") != 14:
        raise MigrationError("option metadata does not describe schema 14")
    return value


def _pointer(key: str) -> str:
    return "/config/" + key.replace("~", "~0").replace("/", "~1")


def _compatibility_value(
    option: dict[str, Any],
    document: dict[str, Any],
    periodic_override: bool | None,
) -> tuple[Any, bool, str]:
    """Return (value, exact_derivation, explanation) for one missing config key."""
    option_id = option["id"]
    config = document["config"]
    if option_id == "periodic":
        if periodic_override is not None:
            return periodic_override, True, "supplied explicitly to the migration tool"
        return False, False, "schema 13 did not reliably record periodic geometry"
    if option_id in {"point_seed", "search_seed"}:
        if "seed" not in config:
            raise MigrationError(f"cannot derive {option['json_key']} without config.seed")
        return config["seed"], True, "derived from schema-13 config.seed"
    if option_id == "p_values":
        values = document.get("p_values")
        if not isinstance(values, list) or not values:
            raise MigrationError("cannot derive config.p_values from top-level p_values")
        return copy.deepcopy(values), True, "copied from top-level p_values"
    if option_id == "mode":
        mode = document.get("mode")
        if not isinstance(mode, str) or not mode:
            raise MigrationError("cannot derive config.mode from top-level mode")
        return mode, True, "copied from top-level mode"
    if option_id == "output_durability":
        # Schema-13 output predates the explicit durability contract.  "none"
        # most accurately describes a document for which fsync guarantees were
        # neither requested nor recorded.
        return "none", False, "schema 13 did not record an output durability contract"
    if "default" in option:
        return copy.deepcopy(option["default"]), False, "filled with the schema-14 compatibility default"
    raise MigrationError(f"no migration rule for required config field {option['json_key']}")


def migrate_document(
    document: dict[str, Any],
    source_bytes: bytes | None = None,
    periodic_override: bool | None = None,
) -> dict[str, Any]:
    """Return a schema-14 copy of *document*.

    Calling the function on a document already migrated by this tool is
    idempotent and returns an equivalent deep copy.
    """
    version = document.get("schema_version")
    if version == 14:
        migration = document.get("migration_metadata")
        if migration is None or migration.get("tool") == TOOL_NAME:
            return copy.deepcopy(document)
        raise MigrationError("schema-14 document has incompatible migration metadata")
    if version != 13:
        raise MigrationError(f"expected schema_version 13, got {version!r}")
    if not isinstance(document.get("config"), dict):
        raise MigrationError("schema-13 document has no config object")
    recorded_periodic = document["config"].get("periodic")
    if periodic_override is not None and recorded_periodic is not None:
        if bool(recorded_periodic) != periodic_override:
            raise MigrationError(
                "periodic override conflicts with the value recorded in config.periodic"
            )

    result = copy.deepcopy(document)
    result["schema_version"] = 14
    inferred: list[str] = []
    unrecoverable: list[str] = []
    notes: list[str] = []

    metadata = _metadata()
    for option in metadata["options"]:
        key = option.get("json_key")
        if not key or not option.get("required", True):
            continue
        if key in result["config"]:
            continue
        value, exact, explanation = _compatibility_value(
            option, result, periodic_override
        )
        result["config"][key] = value
        pointer = _pointer(key)
        inferred.append(pointer)
        if not exact:
            unrecoverable.append(pointer)
        notes.append(f"{pointer}: {explanation}")

    if source_bytes is None:
        source_bytes = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if "memory_plan" not in result:
        effective_threads = int(result.get("threads", 1))
        result["memory_plan"] = {
            "budget_bytes": 0,
            "fixed_overhead_bytes": 0,
            "estimated_instance_bytes": 0,
            "estimated_peak_bytes": 0,
            "requested_threads": max(1, effective_threads),
            "resolved_threads": max(1, effective_threads),
            "effective_threads": max(1, effective_threads),
            "limited_by_budget": False,
            "reverse_knn_enabled": True,
        }
        for field in result["memory_plan"]:
            pointer = f"/memory_plan/{field}"
            inferred.append(pointer)
            unrecoverable.append(pointer)
        notes.append("/memory_plan: schema 13 did not record memory estimates or budget limiting")

    result["migration_metadata"] = {
        "source_schema_version": 13,
        "target_schema_version": 14,
        "tool": TOOL_NAME,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "inferred_fields": sorted(inferred),
        "unrecoverable_fields": sorted(unrecoverable),
        "notes": sorted(notes),
    }
    return result


def _validate(document: dict[str, Any], schema_path: Path) -> None:
    try:
        import jsonschema  # type: ignore
    except ImportError as exc:
        raise MigrationError("jsonschema is required for migration validation") from exc
    schema = _load_json(schema_path)
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(document), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        where = "/".join(str(part) for part in first.path) or "<root>"
        raise MigrationError(f"migrated document fails schema validation at {where}: {first.message}")


def _atomic_write(path: Path, text: str) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        if os.name != "nt":
            try:
                dir_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            except OSError:
                dir_fd = -1
            if dir_fd >= 0:
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
    except Exception:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="schema-13 result JSON")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path, help="write migrated JSON to this path")
    output.add_argument("--in-place", action="store_true", help="atomically replace the input file")
    parser.add_argument("--stdout", action="store_true", help="write migrated JSON to stdout")
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument(
        "--periodic",
        dest="periodic_override",
        action="store_true",
        help="record known periodic geometry when schema 13 omitted it",
    )
    geometry.add_argument(
        "--non-periodic",
        dest="periodic_override",
        action="store_false",
        help="record known open-square geometry when schema 13 omitted it",
    )
    parser.set_defaults(periodic_override=None)
    parser.add_argument("--no-validate", action="store_true", help="skip schema-13 and schema-14 validation")
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
        if not args.no_validate and source.get("schema_version") == 13:
            _validate(source, LEGACY_SCHEMA_PATH)
        migrated = migrate_document(
            source, source_bytes, periodic_override=args.periodic_override
        )
        if not args.no_validate:
            _validate(migrated, CURRENT_SCHEMA_PATH)
        text = json.dumps(migrated, indent=2, ensure_ascii=False) + "\n"
        if args.stdout:
            print(text, end="")
        else:
            destination = args.input if args.in_place else args.output
            assert destination is not None
            _atomic_write(destination, text)
    except (OSError, json.JSONDecodeError, MigrationError) as exc:
        parser.exit(1, f"migration failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
