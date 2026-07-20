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
    path = root / "scripts" / "migrate_schema13_to14.py"
    spec = importlib.util.spec_from_file_location("schema_migrator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_documents(root: Path):
    for path in sorted((root / "validation_runs").rglob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if doc.get("schema_version") == 13 and all(
            key in doc for key in ("run_metadata", "build_metadata", "summary_rows", "search_stats")
        ):
            yield path, doc


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: schema_migration_test.py <repo_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    migrator = load_migrator(root)
    schema = json.loads((root / "schema" / "results.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)

    migrated_count = 0
    first_path: Path | None = None
    for path, document in result_documents(root):
        first_path = first_path or path
        source_bytes = path.read_bytes()
        migrated = migrator.migrate_document(document, source_bytes)
        errors = list(validator.iter_errors(migrated))
        assert not errors, f"{path}: {errors[0].message if errors else ''}"
        meta = migrated["migration_metadata"]
        assert meta["source_schema_version"] == 13
        assert meta["target_schema_version"] == 14
        assert meta["inferred_fields"] == sorted(set(meta["inferred_fields"]))
        assert meta["unrecoverable_fields"] == sorted(set(meta["unrecoverable_fields"]))
        assert "/config/p_values" in meta["inferred_fields"]
        assert migrated["config"]["p_values"] == migrated["p_values"]
        assert migrated["config"]["point_seed"] == document["config"]["seed"]
        assert migrated["config"]["search_seed"] == document["config"]["seed"]
        assert migrator.migrate_document(migrated) == migrated
        migrated_count += 1

    assert migrated_count > 0 and first_path is not None

    with tempfile.TemporaryDirectory(prefix="aldous-schema-migration-") as tmp:
        tmpdir = Path(tmp)
        source = tmpdir / "source.json"
        source.write_bytes(first_path.read_bytes())
        output = tmpdir / "output.json"
        subprocess.run(
            [sys.executable, str(root / "scripts" / "migrate_schema13_to14.py"), str(source), "--output", str(output)],
            check=True,
        )
        before = output.read_bytes()
        subprocess.run(
            [sys.executable, str(root / "scripts" / "migrate_schema13_to14.py"), str(output), "--in-place"],
            check=True,
        )
        assert output.read_bytes() == before, "in-place migration must be idempotent"
        assert source.read_bytes() == first_path.read_bytes(), "--output must not mutate source"
        assert not list(tmpdir.glob(".*.tmp")), "migration left temporary files"

        missing_geometry = json.loads(first_path.read_text())
        missing_geometry["config"].pop("periodic", None)
        geometry_source = tmpdir / "geometry-source.json"
        geometry_output = tmpdir / "geometry-output.json"
        geometry_source.write_text(json.dumps(missing_geometry))
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "migrate_schema13_to14.py"),
                str(geometry_source),
                "--output",
                str(geometry_output),
                "--periodic",
            ],
            check=True,
        )
        geometry_doc = json.loads(geometry_output.read_text())
        assert geometry_doc["config"]["periodic"] is True
        assert "/config/periodic" in geometry_doc["migration_metadata"]["inferred_fields"]
        assert "/config/periodic" not in geometry_doc["migration_metadata"]["unrecoverable_fields"]

    print(f"schema migration passed: {migrated_count} schema-13 artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
