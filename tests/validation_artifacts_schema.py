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


def project_version(root: Path) -> str:
    text = (root / "CMakeLists.txt").read_text()
    m = re.search(r"project\(aldous_tsp VERSION ([^\s)]+)", text)
    if not m:
        raise RuntimeError("could not determine project version from CMakeLists.txt")
    return m.group(1)


def schema_version(schema: dict) -> int:
    value = schema["properties"]["schema_version"].get("const")
    if not isinstance(value, int):
        raise RuntimeError("schema_version const missing from schema")
    return value


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validation_artifacts_schema.py <repo_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    validation_dir = root / "validation_runs"
    if not validation_dir.exists():
        print(f"validation directory not found: {validation_dir}", file=sys.stderr)
        return 1

    current_schema = json.loads((root / "schema" / "results.schema.json").read_text())
    current_version = schema_version(current_schema)
    schemas = {current_version: current_schema}
    legacy_path = root / "schema" / "results-v13.schema.json"
    if legacy_path.exists():
        legacy_schema = json.loads(legacy_path.read_text())
        schemas[schema_version(legacy_schema)] = legacy_schema
    validators = {
        version: jsonschema.Draft202012Validator(document)
        for version, document in schemas.items()
    }
    # Artifacts declare their vintage in ARTIFACT_VERSION, and every artifact
    # must match it (catches accidentally mixed vintages, which is the guard's
    # real purpose). The pin is deliberately NOT required to equal the current
    # project version: the real-oracle artifacts need an external LKH binary to
    # regenerate, so tying them to the current version would block every version
    # bump -- which is exactly how the project version froze at 0.8.7 while
    # releases were being named 0.9.x, leaving misleading provenance in every
    # newly produced result JSON. Drift from the current version is reported as
    # a visible warning; regenerating the artifacts updates the pin.
    current_project = project_version(root)
    pin_path = root / "validation_runs" / "ARTIFACT_VERSION"
    expected_project = pin_path.read_text().strip() if pin_path.exists() else current_project
    result_files: list[Path] = []
    errors: list[str] = []

    for path in sorted(validation_dir.rglob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            errors.append(f"{path.relative_to(root)}: invalid JSON: {exc}")
            continue
        if "schema_version" not in doc:
            continue
        # Only current executable result JSONs are validated against the current
        # result schema. The validation directory also contains tool manifests
        # and fake-original outputs used to test original-compatible command
        # routing; those are intentionally not current-schema result files.
        if not all(key in doc for key in ("run_metadata", "build_metadata", "summary_rows", "search_stats")):
            continue
        result_files.append(path)
        document_version = doc.get("schema_version")
        validator = validators.get(document_version)
        if validator is None:
            errors.append(
                f"{path.relative_to(root)}: unsupported schema_version {document_version}; "
                f"available versions are {sorted(validators)}"
            )
            continue
        for error in sorted(validator.iter_errors(doc), key=lambda e: list(e.path)):
            location = "/".join(str(part) for part in error.path) or "<root>"
            errors.append(f"{path.relative_to(root)} at {location}: {error.message}")
        project = doc.get("run_metadata", {}).get("project_version")
        if project != expected_project:
            errors.append(f"{path.relative_to(root)}: project_version {project!r} != {expected_project!r}")

    if not result_files:
        errors.append("no validation result JSON files with schema_version were found")

    if expected_project != current_project:
        print(f"NOTE: validation artifacts are from solver {expected_project}; current is "
              f"{current_project}. They remain schema-valid; regenerate them (and update "
              f"validation_runs/ARTIFACT_VERSION) when convenient.")

    # Bundled source packages should not carry paths from the build sandbox that
    # generated their validation artifacts. Paths make the evidence harder to
    # compare across release hosts and have previously hidden stale artifacts.
    stale_patterns = ["/mnt/data/", "validate_work/", "audit_fix_", "0.8.5-cleanup"]
    for path in sorted(validation_dir.rglob("*")):
        if path.is_file():
            text = path.read_text(errors="ignore")
            for needle in stale_patterns:
                if needle in text:
                    errors.append(f"{path.relative_to(root)} contains stale/environment-specific path fragment {needle!r}")
                    break

    if errors:
        for line in errors[:200]:
            print(line, file=sys.stderr)
        if len(errors) > 200:
            print(f"... {len(errors) - 200} more errors", file=sys.stderr)
        return 1
    versions = sorted({json.loads(path.read_text()).get("schema_version") for path in result_files})
    print(
        f"validated {len(result_files)} bundled validation result JSON files "
        f"against archived/current schemas {versions} / project {expected_project}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
