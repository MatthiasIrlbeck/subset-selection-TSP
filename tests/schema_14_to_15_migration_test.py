#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import jsonschema


def load_migrator(root: Path):
    path = root / "scripts" / "migrate_schema14_to15.py"
    spec = importlib.util.spec_from_file_location("schema14_to15_migrator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schema14_documents(root: Path):
    for path in sorted((root / "validation_runs").rglob("*.json")):
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if document.get("schema_version") == 14 and all(
            key in document
            for key in ("run_metadata", "build_metadata", "summary_rows", "search_stats")
        ):
            yield path, document


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: schema_14_to_15_migration_test.py <repo_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    migrator = load_migrator(root)
    current_schema = json.loads((root / "schema" / "results.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(current_schema)

    count = 0
    first_path: Path | None = None
    for path, document in schema14_documents(root):
        first_path = first_path or path
        migrated = migrator.migrate_document(document, path.read_bytes())
        errors = list(validator.iter_errors(migrated))
        assert not errors, f"{path}: {errors[0].message if errors else ''}"
        assert migrated["schema_version"] == 15
        assert migrated["config"]["search_policy_preset"] == "legacy-balanced"
        steps = migrated["migration_metadata"]["steps"]
        assert steps[-1]["source_schema_version"] == 14
        assert steps[-1]["target_schema_version"] == 15
        assert "/config/search_policy_preset" in steps[-1]["inferred_fields"]
        if "migration_metadata" in document:
            assert len(steps) == 2
            assert steps[0]["source_schema_version"] == 13
            assert steps[0]["target_schema_version"] == 14
        assert migrator.migrate_document(migrated) == migrated
        count += 1

    # Validation artifacts may all predate native schema 14. In that case,
    # produce a real schema-14 document by chaining the existing migrator.
    if first_path is None:
        old_test = root / "tests" / "schema_migration_test.py"
        assert old_test.exists()
        for path in sorted((root / "validation_runs").rglob("*.json")):
            document = json.loads(path.read_text())
            if document.get("schema_version") == 13 and "run_metadata" in document:
                old_spec = importlib.util.spec_from_file_location(
                    "schema13_to14_migrator", root / "scripts" / "migrate_schema13_to14.py"
                )
                assert old_spec and old_spec.loader
                old = importlib.util.module_from_spec(old_spec)
                old_spec.loader.exec_module(old)
                document14 = old.migrate_document(document, path.read_bytes())
                with tempfile.TemporaryDirectory(prefix="aldous-schema15-source-") as tmp:
                    generated = Path(tmp) / "source-v14.json"
                    generated.write_text(json.dumps(document14))
                    first_path = generated
                    migrated = migrator.migrate_document(document14, generated.read_bytes())
                    errors = list(validator.iter_errors(migrated))
                    assert not errors, errors[0].message if errors else ""
                    count = 1
                break

    assert count > 0 and first_path is not None

    # Exercise atomic output and idempotent in-place behavior with a source
    # document that survives the temporary fallback above.
    source_document = None
    for _, document in schema14_documents(root):
        source_document = document
        break
    if source_document is None:
        # Chain one bundled v13 artifact again for CLI testing.
        for path in sorted((root / "validation_runs").rglob("*.json")):
            document = json.loads(path.read_text())
            if document.get("schema_version") == 13 and "run_metadata" in document:
                old_spec = importlib.util.spec_from_file_location(
                    "schema13_to14_migrator_cli", root / "scripts" / "migrate_schema13_to14.py"
                )
                assert old_spec and old_spec.loader
                old = importlib.util.module_from_spec(old_spec)
                old_spec.loader.exec_module(old)
                source_document = old.migrate_document(document, path.read_bytes())
                break
    assert source_document is not None

    with tempfile.TemporaryDirectory(prefix="aldous-schema15-migration-") as tmp:
        tmpdir = Path(tmp)
        source = tmpdir / "source.json"
        source.write_text(json.dumps(source_document))
        output = tmpdir / "output.json"
        subprocess.run(
            [sys.executable, str(root / "scripts" / "migrate_schema14_to15.py"),
             str(source), "--output", str(output)],
            check=True,
        )
        before = output.read_bytes()
        subprocess.run(
            [sys.executable, str(root / "scripts" / "migrate_schema14_to15.py"),
             str(output), "--in-place"],
            check=True,
        )
        assert output.read_bytes() == before
        assert not list(tmpdir.glob(".*.tmp"))

    print(f"schema 14-to-15 migration passed: {count} document(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
