#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    import jsonschema
except Exception as exc:
    print(f"jsonschema is required for validation_artifacts_schema.py: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

from validation_paths import archived_validation_dirs, current_validation_dir


def project_version(root: Path) -> str:
    text = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(r"project\(aldous_tsp VERSION ([^\s)]+)", text)
    if not match:
        raise RuntimeError("could not determine project version from CMakeLists.txt")
    return match.group(1)


def schema_version(schema: dict) -> int:
    value = schema["properties"]["schema_version"].get("const")
    if not isinstance(value, int):
        raise RuntimeError("schema_version const missing from schema")
    return value


def is_native_result(document: dict) -> bool:
    return all(
        key in document
        for key in ("schema_version", "run_metadata", "build_metadata", "summary_rows", "search_stats")
    )


def load_schemas(root: Path) -> dict[int, dict]:
    schemas: dict[int, dict] = {}
    for path in [root / "schema" / "results.schema.json", *sorted((root / "schema").glob("results-v*.schema.json"))]:
        document = json.loads(path.read_text(encoding="utf-8"))
        schemas[schema_version(document)] = document
    return schemas


def validate_tree(
    *,
    root: Path,
    validation_root: Path,
    validators: dict[int, jsonschema.Draft202012Validator],
    expected_project: str | None,
    expected_schema: int | None,
    label: str,
) -> tuple[int, list[str]]:
    errors: list[str] = []
    count = 0
    for path in sorted(validation_root.rglob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{path.relative_to(root)}: invalid JSON: {exc}")
            continue
        if not is_native_result(document):
            continue
        count += 1
        version = document.get("schema_version")
        validator = validators.get(version)
        if validator is None:
            errors.append(
                f"{path.relative_to(root)}: unsupported schema_version {version}; "
                f"available versions are {sorted(validators)}"
            )
            continue
        if expected_schema is not None and version != expected_schema:
            errors.append(
                f"{path.relative_to(root)}: schema_version {version!r} != current {expected_schema!r}"
            )
        for error in sorted(validator.iter_errors(document), key=lambda value: list(value.path)):
            location = "/".join(str(part) for part in error.path) or "<root>"
            errors.append(f"{path.relative_to(root)} at {location}: {error.message}")
        if expected_project is not None:
            actual = document.get("run_metadata", {}).get("project_version")
            if actual != expected_project:
                errors.append(
                    f"{path.relative_to(root)}: project_version {actual!r} != {expected_project!r}"
                )

    if count == 0:
        errors.append(f"{label}: no native result JSON documents were found")

    stale_patterns = ["/mnt/data/", "validate_work/", "audit_fix_", "0.8.5-cleanup"]
    for path in sorted(validation_root.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in stale_patterns:
            if needle in text:
                errors.append(
                    f"{path.relative_to(root)} contains environment-specific path fragment {needle!r}"
                )
                break
    return count, errors


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validation_artifacts_schema.py <repo_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    current = current_validation_dir(root)
    if not current.is_dir():
        print(f"current validation directory not found: {current}", file=sys.stderr)
        return 1

    schemas = load_schemas(root)
    validators = {
        version: jsonschema.Draft202012Validator(document)
        for version, document in schemas.items()
    }
    current_project = project_version(root)
    current_schema = max(schemas)

    artifact_version_path = current / "ARTIFACT_VERSION"
    artifact_schema_path = current / "ARTIFACT_SCHEMA"
    errors: list[str] = []
    if not artifact_version_path.is_file():
        errors.append("validation_runs/current/ARTIFACT_VERSION is missing")
    elif artifact_version_path.read_text(encoding="utf-8").strip() != current_project:
        errors.append("validation_runs/current/ARTIFACT_VERSION does not match the project version")
    if not artifact_schema_path.is_file():
        errors.append("validation_runs/current/ARTIFACT_SCHEMA is missing")
    elif artifact_schema_path.read_text(encoding="utf-8").strip() != str(current_schema):
        errors.append("validation_runs/current/ARTIFACT_SCHEMA does not match the current schema")

    current_count, current_errors = validate_tree(
        root=root,
        validation_root=current,
        validators=validators,
        expected_project=current_project,
        expected_schema=current_schema,
        label="current validation",
    )
    errors.extend(current_errors)

    archived_count = 0
    for archived in archived_validation_dirs(root):
        count, archive_errors = validate_tree(
            root=root,
            validation_root=archived,
            validators=validators,
            expected_project=None,
            expected_schema=None,
            label=str(archived.relative_to(root)),
        )
        archived_count += count
        errors.extend(archive_errors)

    if errors:
        for line in errors[:200]:
            print(line, file=sys.stderr)
        if len(errors) > 200:
            print(f"... {len(errors) - 200} more errors", file=sys.stderr)
        return 1

    print(
        f"validated {current_count} current {current_project}/schema-{current_schema} results "
        f"and {archived_count} explicitly archived historical results"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
