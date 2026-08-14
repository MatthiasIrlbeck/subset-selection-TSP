#!/usr/bin/env python3
"""Migrate Aldous subset-selection TSP result JSON from schema 14 to 15.

Schema 15 records the explicit search-controller preset. Native schema-14
results predate that field, so migration assigns ``legacy-balanced``: this is
the only controller that reproduces the schema-14 automatic behavior. Existing
schema-13-to-14 migration provenance is preserved as the first history step.
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
LEGACY_SCHEMA_PATH = ROOT / "schema" / "results-v14.schema.json"
CURRENT_SCHEMA_PATH = ROOT / "schema" / "results-v15.schema.json"
TOOL_NAME = "scripts/migrate_schema14_to15.py"


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


def _legacy_step(metadata: dict[str, Any]) -> dict[str, Any]:
    required = {
        "source_schema_version",
        "target_schema_version",
        "tool",
        "source_sha256",
        "inferred_fields",
        "unrecoverable_fields",
        "notes",
    }
    if set(metadata) != required:
        raise MigrationError("schema-14 migration_metadata has an unexpected shape")
    return copy.deepcopy(metadata)


def migrate_document(
    document: dict[str, Any],
    source_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Return a schema-15 copy of *document*; schema-15 input is idempotent."""
    version = document.get("schema_version")
    if version == 15:
        return copy.deepcopy(document)
    if version != 14:
        raise MigrationError(f"expected schema_version 14, got {version!r}")
    if not isinstance(document.get("config"), dict):
        raise MigrationError("schema-14 document has no config object")

    result = copy.deepcopy(document)
    prior = result.pop("migration_metadata", None)
    steps: list[dict[str, Any]] = []
    if prior is not None:
        if not isinstance(prior, dict):
            raise MigrationError("schema-14 migration_metadata must be an object")
        steps.append(_legacy_step(prior))

    inferred: list[str] = []
    notes: list[str] = []
    if "search_policy_preset" not in result["config"]:
        result["config"]["search_policy_preset"] = "legacy-balanced"
        inferred.append("/config/search_policy_preset")
        notes.append(
            "/config/search_policy_preset: schema 14 predates presets; "
            "legacy-balanced exactly preserves its automatic controller"
        )

    if source_bytes is None:
        source_bytes = json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    steps.append(
        {
            "source_schema_version": 14,
            "target_schema_version": 15,
            "tool": TOOL_NAME,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "inferred_fields": sorted(inferred),
            "unrecoverable_fields": [],
            "notes": sorted(notes),
        }
    )
    result["schema_version"] = 15
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
            try:
                directory_fd = os.open(
                    path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
            except OSError:
                directory_fd = -1
            if directory_fd >= 0:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            # Cleanup is idempotent when the temporary file was never created.
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="schema-14 result JSON")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path, help="write migrated JSON here")
    output.add_argument(
        "--in-place", action="store_true", help="atomically replace the input"
    )
    parser.add_argument("--stdout", action="store_true", help="write JSON to stdout")
    parser.add_argument(
        "--no-validate", action="store_true", help="skip input/output validation"
    )
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
                CURRENT_SCHEMA_PATH
                if source.get("schema_version") == 15
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
