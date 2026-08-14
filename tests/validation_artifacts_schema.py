#!/usr/bin/env python3
from __future__ import annotations

import hashlib
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


def validate_current_receipts(root: Path, current: Path) -> list[str]:
    errors: list[str] = []
    native_results: list[Path] = []
    for path in sorted(current.rglob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if is_native_result(document):
            native_results.append(path)

    for result_path in native_results:
        receipt_path = Path(str(result_path) + ".receipt")
        if not receipt_path.is_file():
            errors.append(f"{result_path.relative_to(root)} has no adjacent timing receipt")
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{receipt_path.relative_to(root)}: invalid JSON: {exc}")
            continue
        payload = result_path.read_bytes()
        expected_path = result_path.relative_to(root).as_posix()
        if receipt.get("result_path") != expected_path:
            errors.append(
                f"{receipt_path.relative_to(root)}: result_path {receipt.get('result_path')!r} "
                f"!= {expected_path!r}"
            )
        expected_digest = hashlib.sha256(payload).hexdigest()
        if receipt.get("result_sha256") != expected_digest:
            errors.append(f"{receipt_path.relative_to(root)}: result_sha256 does not match result")
        if receipt.get("result_bytes") != len(payload):
            errors.append(f"{receipt_path.relative_to(root)}: result_bytes does not match result")

    return errors


def validate_current_provenance(root: Path, current: Path) -> list[str]:
    errors: list[str] = []
    records: set[tuple[object, ...]] = set()
    for path in sorted(current.rglob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not is_native_result(document):
            continue
        metadata = document.get("run_metadata", {})
        records.add(
            (
                metadata.get("project_version"),
                document.get("schema_version"),
                metadata.get("git_commit"),
                metadata.get("git_tree"),
                metadata.get("source_dirty"),
                metadata.get("revision_source"),
            )
        )

    if len(records) != 1:
        errors.append(
            "validation_runs/current must contain exactly one native source provenance; "
            f"found {sorted(records, key=repr)!r}"
        )
        return errors

    project, schema, commit, tree, dirty, revision_source = next(iter(records))
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        errors.append(f"validation_runs/current has invalid source commit {commit!r}")
    if not isinstance(tree, str) or re.fullmatch(r"[0-9a-f]{40}", tree) is None:
        errors.append(f"validation_runs/current has invalid source tree {tree!r}")
    if dirty is not False:
        errors.append("validation_runs/current must be generated from a clean source tree")
    if revision_source not in {"git", "source-archive"}:
        errors.append(
            f"validation_runs/current has invalid revision source {revision_source!r}"
        )

    summary_path = current / "validation_summary.json"
    if not summary_path.is_file():
        errors.append("validation_runs/current/validation_summary.json is missing")
    else:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"{summary_path.relative_to(root)}: invalid JSON: {exc}")
        else:
            expected = {
                "project_version": project,
                "schema_version": schema,
                "source_commit": commit,
                "source_tree": tree,
            }
            for key, value in expected.items():
                if summary.get(key) != value:
                    errors.append(
                        f"{summary_path.relative_to(root)}: {key} {summary.get(key)!r} "
                        f"!= native evidence {value!r}"
                    )
            if summary.get("source_dirty", False) is not False:
                errors.append(
                    f"{summary_path.relative_to(root)}: source_dirty must be false"
                )
            if summary.get("revision_source", revision_source) != revision_source:
                errors.append(
                    f"{summary_path.relative_to(root)}: revision_source does not match "
                    "native evidence"
                )

    readme_path = current / "README.md"
    if not readme_path.is_file():
        errors.append("validation_runs/current/README.md is missing")
    else:
        readme = readme_path.read_text(encoding="utf-8")
        for label, value in (("solver commit", commit), ("source tree", tree)):
            if isinstance(value, str) and f"{label}: `{value}`" not in readme:
                errors.append(
                    f"{readme_path.relative_to(root)} does not state its exact {label}"
                )
        if "not publication-scale Monte Carlo evidence" not in readme:
            errors.append(
                f"{readme_path.relative_to(root)} does not state the evidence scope"
            )

    return errors


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
    errors.extend(validate_current_receipts(root, current))
    errors.extend(validate_current_provenance(root, current))

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
