#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import jsonschema

from validation_paths import iter_validation_json


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: schema_15_to_16_migration_test.py <repo_root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    migrator = load_module(root / "scripts" / "migrate_schema15_to16.py", "m15to16")
    m14to15 = load_module(root / "scripts" / "migrate_schema14_to15.py", "m14to15")
    m13to14 = load_module(root / "scripts" / "migrate_schema13_to14.py", "m13to14")
    schema = json.loads((root / "schema" / "results.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)

    source15 = None
    for path in iter_validation_json(root):
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if doc.get("schema_version") == 15:
            source15 = doc
            break
        if doc.get("schema_version") == 14:
            source15 = m14to15.migrate_document(doc, path.read_bytes())
            break
        if doc.get("schema_version") == 13 and "run_metadata" in doc:
            doc14 = m13to14.migrate_document(doc, path.read_bytes())
            source15 = m14to15.migrate_document(doc14)
            break
    assert source15 is not None
    migrated = migrator.migrate_document(source15)
    errors = list(validator.iter_errors(migrated))
    assert not errors, errors[0].message if errors else ""
    assert migrated["schema_version"] == 16
    assert migrated["config"]["cv_max_point_ops"] == 100_000_000
    if "campaign_metadata" in source15:
        campaign = migrated["campaign_metadata"]
        assert len(campaign["configuration_fingerprint"]) == 64
        assert len(campaign["method_fingerprint"]) == 64
        assert campaign["configuration_fingerprint"] != campaign["method_fingerprint"]
    assert migrated["timing"]["solver_wall_seconds"] == source15["wall_seconds"]
    step = migrated["migration_metadata"]["steps"][-1]
    assert step["source_schema_version"] == 15
    assert step["target_schema_version"] == 16
    assert "/timing/control_reference_seconds" in step["unrecoverable_fields"]
    if "campaign_metadata" in source15:
        assert "/campaign_metadata/configuration_fingerprint" in step["unrecoverable_fields"]
        assert "/campaign_metadata/method_fingerprint" in step["unrecoverable_fields"]
    assert migrator.migrate_document(migrated) == migrated

    identified15 = json.loads(json.dumps(source15))
    identified15["campaign_metadata"] = {
        "campaign_id": "legacy-identified", "campaign_shard": 0,
        "replicate_offset": 0, "point_seed": 1, "search_seed": 2,
        "solver_policy_id": "legacy", "fidelity_level": "strong",
    }
    identified16 = migrator.migrate_document(identified15)
    campaign = identified16["campaign_metadata"]
    assert len(campaign["configuration_fingerprint"]) == 64
    assert len(campaign["method_fingerprint"]) == 64
    assert campaign["configuration_fingerprint"] != campaign["method_fingerprint"]
    identified_errors = list(validator.iter_errors(identified16))
    assert not identified_errors, identified_errors[0].message if identified_errors else ""
    identified_step = identified16["migration_metadata"]["steps"][-1]
    assert "/campaign_metadata/configuration_fingerprint" in identified_step["unrecoverable_fields"]
    assert "/campaign_metadata/method_fingerprint" in identified_step["unrecoverable_fields"]

    with tempfile.TemporaryDirectory(prefix="aldous-schema16-") as temporary:
        directory = Path(temporary)
        source = directory / "source.json"
        output = directory / "output.json"
        source.write_text(json.dumps(source15), encoding="utf-8")
        subprocess.run([
            sys.executable, str(root / "scripts" / "migrate_schema15_to16.py"),
            str(source), "--output", str(output)
        ], check=True)
        before = output.read_bytes()
        subprocess.run([
            sys.executable, str(root / "scripts" / "migrate_schema15_to16.py"),
            str(output), "--in-place"
        ], check=True)
        assert output.read_bytes() == before
        assert not list(directory.glob(".*.tmp"))

    print("schema 15-to-16 migration passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
