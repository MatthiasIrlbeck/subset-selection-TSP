#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    import jsonschema
except Exception as exc:
    print(f"jsonschema is required for schema_validation.py: {exc}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: schema_validation.py <aldous_tsp_exe> <repo_root> <out_dir>", file=sys.stderr)
        return 2
    exe = Path(sys.argv[1]).resolve()
    root = Path(sys.argv[2]).resolve()
    out_dir = Path(sys.argv[3]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "schema-smoke.json"
    cmd = [
        str(exe), "--mode", "hybrid", "--N", "32", "--instances", "1", "--threads", "1",
        "--p-values", "0.125,0.5,1.0", "--seed", "123", "--restarts", "1", "--sa-iters", "12",
        "--tsp-restarts", "2", "--tsp-ils", "3", "--tsp-patience", "2", "--knn-backend", "grid",
        "--grid-cell", "1e-9", "--verify-knn", "16", "--final-exhaustive-k", "64",
        "--subset-swap-passes", "1", "--pair-exchange-passes", "1", "--ruin-recreate-rounds", "1",
        "--path-relink-top", "1", "--second-sweep", "--include-instance-rows",
        "--oracle", "none", "--output", str(output), "--force",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    doc = json.loads(output.read_text())
    schema = json.loads((root / "schema" / "results.schema.json").read_text())
    sys.path.insert(0, str(root / "scripts"))
    from restart_metadata import (
        DEFAULT_KINDS,
        DEFAULT_SWEEPS,
        KIND_NAMES,
        SWEEP_NAMES,
        load_restart_kinds,
    )

    shared_kinds = load_restart_kinds(root)
    restart_properties = schema["$defs"]["instance_p_value_row"]["properties"]
    schema_kind_property = restart_properties["restart_kinds"]
    schema_sweep_property = restart_properties["restart_sweeps"]
    assert restart_properties["restart_values"]["items"].get("minimum") == 0
    assert "exclusiveMinimum" not in restart_properties["restart_values"]["items"]
    schema_kinds = schema_kind_property["items"]["enum"]
    assert shared_kinds == KIND_NAMES
    assert tuple(shared_kinds) == DEFAULT_KINDS
    assert schema_kinds == list(shared_kinds), (schema_kinds, shared_kinds)
    assert schema_kind_property["x-enumNames"] == list(shared_kinds.values())
    assert schema_sweep_property["items"]["enum"] == list(DEFAULT_SWEEPS)
    assert schema_sweep_property["x-enumNames"] == list(SWEEP_NAMES.values())
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(doc), key=lambda e: list(e.path))
    if errors:
        for error in errors:
            path = "/".join(str(part) for part in error.path) or "<root>"
            print(f"schema validation error at {path}: {error.message}", file=sys.stderr)
        return 1
    assert doc["schema_version"] == 13, doc["schema_version"]
    assert doc["mode"] == "hybrid", doc["mode"]
    assert doc["distance_backend"] == "coords_exact_grid_knn", doc["distance_backend"]
    assert doc["config"]["oracle_mode"] == "none", doc["config"]
    assert doc["config"]["grid_cell"] > 0.0, doc["config"]
    assert doc["config"]["exhaustive_two_opt_policy"] == "final-only", doc["config"]
    assert doc["build_metadata"]["optimization_profile"] in {"release-optimized", "low-memory"}, doc["build_metadata"]
    assert doc["build_metadata"]["effective_optimization_level"] in {"O0", "O1", "O2", "O3", "Os", "Oz", "compiler-default", "config-dependent", "unknown"}, doc["build_metadata"]
    assert isinstance(doc["build_metadata"]["low_memory_build"], bool), doc["build_metadata"]
    assert len(doc["summary_rows"]) == len(doc["p_values"]), doc["summary_rows"]
    assert len(doc["instance_rows"]) == 1, doc["instance_rows"]
    assert len(doc["instance_rows"][0]["p_results"]) == len(doc["p_values"]), doc["instance_rows"]
    p_results = doc["instance_rows"][0]["p_results"]
    for index, p_row in enumerate(p_results):
        arrays = [
            p_row["restart_values"],
            p_row["restart_kinds"],
            p_row["restart_sweeps"],
            p_row["restart_centroids_x"],
            p_row["restart_centroids_y"],
            p_row["restart_radii"],
        ]
        assert all(len(values) == p_row["executed_restarts"] for values in arrays), p_row
        assert 0 <= p_row["best_restart"] < p_row["executed_restarts"], p_row
        if p_row["k"] == doc["N"]:
            assert p_row["restart_kinds"] == [8, 9], p_row
            assert p_row["restart_sweeps"] == [0, 0], p_row
        elif index > 0:
            assert p_row["restart_sweeps"] == [0, 1], p_row
    assert all("p" in row and "key" in row for row in doc["summary_rows"]), doc["summary_rows"]
    assert "knn_build_seconds" in doc["search_stats"], doc["search_stats"]
    assert doc["search_stats"]["knn_requested_grid_instances"] == 1, doc["search_stats"]
    assert doc["search_stats"]["knn_effective_grid_instances"] + doc["search_stats"]["knn_effective_bruteforce_instances"] == 1, doc["search_stats"]
    assert "target_compile_options" in doc["build_metadata"], doc["build_metadata"]
    assert "effective_compile_options" in doc["build_metadata"], doc["build_metadata"]
    assert "highp_exchange_scans" in doc["search_stats"], doc["search_stats"]
    assert "region_restarts" in doc["search_stats"], doc["search_stats"]
    assert "dense_restarts" in doc["search_stats"], doc["search_stats"]
    phases = doc["search_stats"]["phase_timing"]
    expected_phase_fields = {
        "seed_construction_seconds", "tsp_construction_seconds",
        "initial_polish_seconds", "sa_seconds",
        "sa_checkpoint_polish_seconds", "post_sa_polish_seconds",
        "subset_swap_seconds", "highp_exchange_seconds",
        "pair_exchange_seconds", "ruin_recreate_seconds",
        "path_relink_seconds", "tsp_ils_seconds",
        "final_polish_seconds", "oracle_seconds",
        "sa_proposal_samples", "sa_insertion_samples",
        "sa_proposal_sample_seconds", "sa_insertion_sample_seconds",
    }
    assert set(phases) == expected_phase_fields, phases
    assert all(value >= 0 for value in phases.values()), phases
    assert phases["sa_proposal_samples"] > 0, phases
    assert phases["sa_insertion_samples"] > 0, phases
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
