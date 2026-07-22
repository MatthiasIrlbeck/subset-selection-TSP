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
    raise SystemExit(2) from exc


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
        PROMOTION_STAGE_NAMES,
        SWEEP_NAMES,
        load_restart_kinds,
    )

    shared_kinds = load_restart_kinds(root)
    restart_properties = schema["$defs"]["instance_p_value_row"]["properties"]
    schema_kind_property = restart_properties["restart_kinds"]
    schema_sweep_property = restart_properties["restart_sweeps"]
    schema_promotion_property = restart_properties["restart_promotion_stages"]
    assert restart_properties["restart_values"]["items"].get("minimum") == 0
    assert "exclusiveMinimum" not in restart_properties["restart_values"]["items"]
    schema_kinds = schema_kind_property["items"]["enum"]
    assert shared_kinds == KIND_NAMES
    assert tuple(shared_kinds) == DEFAULT_KINDS
    assert schema_kinds == list(shared_kinds), (schema_kinds, shared_kinds)
    assert schema_kind_property["x-enumNames"] == list(shared_kinds.values())
    assert schema_sweep_property["items"]["enum"] == list(DEFAULT_SWEEPS)
    assert schema_sweep_property["x-enumNames"] == list(SWEEP_NAMES.values())
    assert schema_promotion_property["items"]["enum"] == list(PROMOTION_STAGE_NAMES)
    assert schema_promotion_property["x-enumNames"] == list(PROMOTION_STAGE_NAMES.values())
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(doc), key=lambda e: list(e.path))
    if errors:
        for error in errors:
            path = "/".join(str(part) for part in error.path) or "<root>"
            print(f"schema validation error at {path}: {error.message}", file=sys.stderr)
        return 1
    assert doc["schema_version"] == 15, doc["schema_version"]
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
            p_row["restart_roles"],
            p_row["restart_variants"],
            p_row["restart_promotion_stages"],
            p_row["restart_sa_iterations"],
            p_row["restart_sa_t0"],
            p_row["restart_sa_t1"],
            p_row["restart_sa_temperature_samples"],
            p_row["restart_sa_temperature_calibrated"],
            p_row["restart_strong_polished"],
            p_row["restart_centroids_x"],
            p_row["restart_centroids_y"],
            p_row["restart_radii"],
        ]
        assert all(len(values) == p_row["executed_restarts"] for values in arrays), p_row
        assert 0 <= p_row["best_restart"] < p_row["executed_restarts"], p_row
        if p_row["k"] == doc["N"]:
            assert p_row["restart_kinds"] == [9, 9], p_row
            assert p_row["restart_sweeps"] == [0, 0], p_row
            assert p_row["restart_roles"] == [4, 4], p_row
            assert p_row["restart_promotion_stages"] == [2, 2], p_row
            assert p_row["restart_strong_polished"] == [True, True], p_row
        elif index > 0:
            # Continuation is supplemental: all primary records precede the
            # single secondary-sweep continuation record in this smoke run.
            assert p_row["restart_sweeps"][-1] == 1, p_row
            assert all(code == 0 for code in p_row["restart_sweeps"][:-1]), p_row
            assert p_row["restart_roles"][-1] == 1, p_row
    assert all("p" in row and "key" in row for row in doc["summary_rows"]), doc["summary_rows"]
    stats = doc["search_stats"]
    assert sum(stats["sa_decile_moves"]) == stats["sa_moves"], stats
    assert sum(stats["sa_decile_accepted"]) == stats["sa_accepted"], stats
    assert stats["sa_temperature_schedules"] >= stats["sa_temperature_calibrations"], stats
    assert stats["sa_temperature_schedules"] >= stats["sa_temperature_fallbacks"], stats
    assert "knn_build_seconds" in doc["search_stats"], doc["search_stats"]
    assert doc["search_stats"]["knn_requested_grid_instances"] == 1, doc["search_stats"]
    assert doc["search_stats"]["knn_effective_grid_instances"] + doc["search_stats"]["knn_effective_bruteforce_instances"] == 1, doc["search_stats"]
    assert "target_compile_options" in doc["build_metadata"], doc["build_metadata"]
    assert "effective_compile_options" in doc["build_metadata"], doc["build_metadata"]
    assert "highp_exchange_scans" in doc["search_stats"], doc["search_stats"]
    assert "region_restarts" in doc["search_stats"], doc["search_stats"]
    assert "dense_restarts" in doc["search_stats"], doc["search_stats"]
    assert doc["config"]["search_policy_preset"] == "legacy-balanced", doc["config"]
    assert doc["config"]["pair_exchange_max_k"] == 5000, doc["config"]
    assert doc["config"]["exact_subset_max_n"] == 0, doc["config"]
    assert doc["config"]["racing_candidates"] == 0, doc["config"]
    assert doc["config"]["racing_survivors"] == 2, doc["config"]
    assert doc["config"]["racing_pilot_iters"] == 2000, doc["config"]
    assert doc["config"]["racing_min_jaccard"] == 0.05, doc["config"]
    assert doc["config"]["tsp_candidate_starts"] == 12, doc["config"]
    assert doc["config"]["tsp_farthest_starts"] == 0, doc["config"]
    assert doc["config"]["tsp_min_edge_jaccard"] == 0.02, doc["config"]
    assert doc["config"]["staged_search"] is True, doc["config"]
    assert doc["config"]["strong_polish_finalists"] == 3, doc["config"]
    assert doc["config"]["strong_polish_min_jaccard"] == 0.02, doc["config"]
    assert doc["config"]["path_relink_diverse_reserve"] == 1, doc["config"]
    assert doc["config"]["path_relink_max_pairs"] == 3, doc["config"]
    assert doc["campaign_metadata"]["campaign_id"] == "default", doc["campaign_metadata"]
    assert doc["campaign_metadata"]["point_seed"] == 123, doc["campaign_metadata"]
    assert doc["campaign_metadata"]["search_seed"] == 123, doc["campaign_metadata"]
    assert len(doc["instance_rows"][0]["point_stream_id"]) == 16, doc["instance_rows"][0]
    assert len(doc["instance_rows"][0]["search_stream_id"]) == 16, doc["instance_rows"][0]
    assert "pair_exchange_skipped_large_k" in doc["search_stats"], doc["search_stats"]
    assert doc["search_stats"]["exact_subset_calls"] == 0, doc["search_stats"]
    assert doc["search_stats"]["exact_subset_solved"] == 0, doc["search_stats"]
    assert "exact_subset_states" in doc["search_stats"], doc["search_stats"]
    assert "exact_subset_transitions" in doc["search_stats"], doc["search_stats"]
    assert "exact_subset_peak_memory_bytes" in doc["search_stats"], doc["search_stats"]
    assert doc["search_stats"]["racing_pilot_restarts"] == 0, doc["search_stats"]
    assert doc["search_stats"]["racing_promoted_restarts"] == 0, doc["search_stats"]
    assert doc["search_stats"]["tsp_candidate_starts"] == 12, doc["search_stats"]
    assert doc["search_stats"]["tsp_promoted_restarts"] == 2, doc["search_stats"]
    assert "strong_polish_candidates" in doc["search_stats"], doc["search_stats"]
    assert "strong_polish_finalists" in doc["search_stats"], doc["search_stats"]
    assert "path_relink_pairs_considered" in doc["search_stats"], doc["search_stats"]
    assert "path_relink_candidate_scans" in doc["search_stats"], doc["search_stats"]
    phases = doc["search_stats"]["phase_timing"]
    expected_phase_fields = {
        "seed_construction_seconds", "tsp_construction_seconds",
        "initial_polish_seconds", "sa_seconds",
        "sa_checkpoint_polish_seconds", "post_sa_polish_seconds",
        "subset_swap_seconds", "highp_exchange_seconds",
        "pair_exchange_seconds", "ruin_recreate_seconds",
        "ejection_chain_seconds", "exact_subset_seconds",
        "path_relink_seconds", "tsp_ils_seconds",
        "final_polish_seconds", "oracle_seconds",
        "sa_proposal_samples", "sa_insertion_samples",
        "sa_proposal_sample_seconds", "sa_insertion_sample_seconds",
    }
    assert set(phases) == expected_phase_fields, phases
    assert all(value >= 0 for value in phases.values()), phases
    assert phases["sa_proposal_samples"] > 0, phases
    assert phases["sa_insertion_samples"] > 0, phases
    assert all(row["exact_optimal_instances"] == 0 for row in doc["summary_rows"]), doc["summary_rows"]
    assert all(not p_row["exact_optimal"] for p_row in p_results), p_results
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
