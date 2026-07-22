#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_case(exe: Path, args: list[str], out_path: Path) -> dict:
    cmd = [str(exe), *args, "--output", str(out_path), "--force"]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(out_path.read_text())


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: cli_regressions.py <aldous_tsp_exe> <repo_root>", file=sys.stderr)
        return 2
    exe = Path(sys.argv[1]).resolve()
    root = Path(sys.argv[2]).resolve()
    fake_lkh = root / "tests" / "fake_lkh.sh"

    help_run = subprocess.run(
        [str(exe), "--help"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert help_run.stderr == "", help_run.stderr
    assert len(help_run.stdout) > 8000, len(help_run.stdout)
    for marker in (
        "--dense-exact-insertion",
        "--oracle-inline-feedback",
        "--disable-path-relink",
        "--pair-exchange-max-k",
        "--racing-candidates",
        "--exact-subset-max-n",
        "--tsp-candidate-starts",
        "--staged-search",
        "--path-relink-max-pairs",
        "--campaign-id",
        "Boolean flags accept plain presence as true",
    ):
        assert marker in help_run.stdout, marker
    assert "+58% wall" in help_run.stdout, help_run.stdout
    assert "quality conditional on that subset" in help_run.stdout, help_run.stdout
    assert "~99% of optimal" not in help_run.stdout, help_run.stdout
    assert "live spatial index (default: false)" in help_run.stdout, help_run.stdout

    with tempfile.TemporaryDirectory(prefix="aldous_cli_regressions_") as td:
        tmp = Path(td)

        disable_doc = run_case(
            exe,
            [
                "--N", "40",
                "--instances", "1",
                "--threads", "1",
                "--p-values", "0.3,1.0",
                "--sa-iters", "1100",
                "--restarts", "2",
                "--tsp-restarts", "2",
                "--tsp-ils", "4",
                "--subset-swap-passes", "2",
                "--pair-exchange-passes", "1",
                "--ruin-recreate-rounds", "1",
                "--path-relink-top", "2",
                "--disable-two-opt",
            ],
            tmp / "disable-two-opt.json",
        )
        stats = disable_doc["search_stats"]
        assert stats["two_opt_scans"] == 0, stats
        assert stats["two_opt_improvements"] == 0, stats

        grid_doc = run_case(
            exe,
            [
                "--N", "48",
                "--instances", "1",
                "--threads", "1",
                "--p-values", "1.0",
                "--sa-iters", "0",
                "--restarts", "1",
                "--tsp-restarts", "1",
                "--tsp-ils", "0",
                "--grid-cell", "1e-12",
                "--verify-knn", "48",
            ],
            tmp / "tiny-grid-cell.json",
        )
        assert grid_doc["done"] == 1, grid_doc
        assert grid_doc["config"]["verify_knn_checks"] == 48, grid_doc["config"]

        assert grid_doc["config"]["grid_cell"] > 0.0, grid_doc["config"]
        assert grid_doc["search_stats"]["knn_requested_grid_instances"] == 1, grid_doc["search_stats"]
        assert grid_doc["search_stats"]["knn_grid_cell_capped_instances"] >= 1, grid_doc["search_stats"]
        assert "e-" in (tmp / "tiny-grid-cell.json").read_text(), "tiny grid-cell should serialize in scientific precision"

        rows_doc = run_case(
            exe,
            [
                "--N", "18",
                "--instances", "2",
                "--threads", "1",
                "--p-values", "0.5,1.0",
                "--sa-iters", "0",
                "--restarts", "1",
                "--tsp-restarts", "1",
                "--tsp-ils", "0",
                "--include-instance-rows",
            ],
            tmp / "instance-rows.json",
        )
        assert rows_doc["config"]["include_instance_rows"] is True, rows_doc["config"]
        assert len(rows_doc["instance_rows"]) == 2, rows_doc["instance_rows"]
        assert all(len(row["p_results"]) == 2 for row in rows_doc["instance_rows"]), rows_doc["instance_rows"]

        exact_doc = run_case(
            exe,
            [
                "--N", "10",
                "--instances", "1",
                "--threads", "1",
                "--p-values", "0.5,1.0",
                "--exact-subset-max-n", "10",
                "--second-sweep",
                "--include-instance-rows",
            ],
            tmp / "exact-subset.json",
        )
        assert exact_doc["config"]["exact_subset_max_n"] == 10, exact_doc["config"]
        assert exact_doc["search_stats"]["exact_subset_calls"] == 2, exact_doc["search_stats"]
        assert exact_doc["search_stats"]["exact_subset_solved"] == 2, exact_doc["search_stats"]
        assert exact_doc["search_stats"]["exact_subset_states"] > 0, exact_doc["search_stats"]
        assert exact_doc["search_stats"]["exact_subset_transitions"] > 0, exact_doc["search_stats"]
        assert exact_doc["search_stats"]["exact_subset_peak_memory_bytes"] > 0, exact_doc["search_stats"]
        assert exact_doc["search_stats"]["phase_timing"]["exact_subset_seconds"] >= 0.0, exact_doc["search_stats"]
        exact_p_rows = exact_doc["instance_rows"][0]["p_results"]
        assert all(row["exact_optimal"] for row in exact_p_rows), exact_p_rows
        assert all(row["executed_restarts"] == 0 for row in exact_p_rows), exact_p_rows
        assert all(row["best_restart"] == -1 for row in exact_p_rows), exact_p_rows
        restart_arrays = {
            "restart_values", "restart_kinds", "restart_sweeps", "restart_roles",
            "restart_variants", "restart_promotion_stages", "restart_sa_iterations",
            "restart_sa_t0", "restart_sa_t1", "restart_sa_temperature_samples",
            "restart_sa_temperature_calibrated", "restart_strong_polished", "restart_centroids_x", "restart_centroids_y", "restart_radii",
        }
        assert all(restart_arrays.isdisjoint(row) for row in exact_p_rows), exact_p_rows
        assert all(row["exact_optimal_instances"] == 1 for row in exact_doc["summary_rows"]), exact_doc["summary_rows"]

        racing_doc = run_case(
            exe,
            [
                "--N", "64",
                "--instances", "1",
                "--threads", "1",
                "--p-values", "0.4",
                "--sa-iters", "20",
                "--restarts", "2",
                "--continuation-restarts", "0",
                "--racing-candidates", "4",
                "--racing-survivors", "2",
                "--racing-pilot-iters", "5",
                "--racing-min-jaccard", "0.1",
                "--restart-threads", "2",
                "--disable-subset-swap",
                "--disable-pair-exchange",
                "--disable-ruin-recreate",
                "--disable-path-relink",
                "--include-instance-rows",
            ],
            tmp / "restart-racing.json",
        )
        racing_row = racing_doc["instance_rows"][0]["p_results"][0]
        assert racing_doc["config"]["racing_candidates"] == 4, racing_doc["config"]
        assert racing_doc["config"]["racing_survivors"] == 2, racing_doc["config"]
        assert racing_doc["config"]["racing_pilot_iters"] == 5, racing_doc["config"]
        assert racing_doc["config"]["racing_min_jaccard"] == 0.1, racing_doc["config"]
        assert racing_doc["search_stats"]["racing_pilot_restarts"] == 4, racing_doc["search_stats"]
        assert racing_doc["search_stats"]["racing_promoted_restarts"] == 2, racing_doc["search_stats"]
        assert racing_row["restart_roles"] == [0, 0, 4, 4, 4, 4], racing_row
        assert racing_row["restart_promotion_stages"].count(1) == 2, racing_row
        assert racing_row["restart_promotion_stages"].count(2) == 2, racing_row
        assert sorted(racing_row["restart_sa_iterations"][-4:]) == [5, 5, 25, 25], racing_row
        assert len(racing_row["restart_sa_t0"]) == racing_row["executed_restarts"], racing_row
        assert len(racing_row["restart_sa_t1"]) == racing_row["executed_restarts"], racing_row
        assert all(
            (iterations == 0 and t0 == 0.0 and t1 == 0.0)
            or (iterations > 0 and t0 > t1 > 0.0)
            for iterations, t0, t1 in zip(
                racing_row["restart_sa_iterations"],
                racing_row["restart_sa_t0"],
                racing_row["restart_sa_t1"],
            )
        ), racing_row
        assert racing_doc["search_stats"]["sa_temperature_schedules"] > 0, racing_doc["search_stats"]
        assert sum(racing_doc["search_stats"]["sa_decile_moves"]) == racing_doc["search_stats"]["sa_moves"], racing_doc["search_stats"]

        identity_doc = run_case(
            exe,
            [
                "--N", "16",
                "--instances", "2",
                "--threads", "1",
                "--p-values", "0.5,1.0",
                "--sa-iters", "0",
                "--restarts", "2",
                "--strong-polish-finalists", "1",
                "--tsp-restarts", "2",
                "--tsp-candidate-starts", "4",
                "--tsp-farthest-starts", "0",
                "--tsp-ils", "0",
                "--path-relink-top", "3",
                "--path-relink-diverse-reserve", "1",
                "--path-relink-max-pairs", "1",
                "--path-relink-max-removed", "8",
                "--path-relink-max-removed-sum", "8",
                "--path-relink-max-candidate-scans", "1000",
                "--campaign-id", "cli-regression",
                "--campaign-shard", "4",
                "--replicate-offset", "20",
                "--point-seed", "111",
                "--search-seed", "222",
                "--solver-policy-id", "staged-v1",
                "--fidelity-level", "cheap",
                "--include-instance-rows",
            ],
            tmp / "campaign-identities.json",
        )
        assert identity_doc["campaign_metadata"] == {
            "campaign_id": "cli-regression",
            "campaign_shard": 4,
            "replicate_offset": 20,
            "point_seed": 111,
            "search_seed": 222,
            "solver_policy_id": "staged-v1",
            "fidelity_level": "cheap",
        }, identity_doc["campaign_metadata"]
        assert [row["replicate_id"] for row in identity_doc["instance_rows"]] == [20, 21]
        assert all(len(row["point_stream_id"]) == 16 for row in identity_doc["instance_rows"])
        assert all(len(row["search_stream_id"]) == 16 for row in identity_doc["instance_rows"])
        assert identity_doc["config"]["tsp_candidate_starts"] == 4
        assert identity_doc["config"]["staged_search"] is True
        assert identity_doc["config"]["strong_polish_finalists"] == 1
        assert identity_doc["config"]["path_relink_max_pairs"] == 1
        assert identity_doc["search_stats"]["tsp_candidate_starts"] == 8
        assert identity_doc["search_stats"]["tsp_promoted_restarts"] == 4

        tiny_p_doc = run_case(
            exe,
            [
                "--N", "12",
                "--instances", "1",
                "--threads", "1",
                "--p-values", "0.000000001,0.000000002,1.0",
                "--sa-iters", "0",
                "--restarts", "1",
                "--tsp-restarts", "1",
                "--tsp-ils", "0",
                "--disable-two-opt",
                "--disable-or-opt",
            ],
            tmp / "tiny-p-values.json",
        )
        assert len(tiny_p_doc["p_values"]) == 3, tiny_p_doc["p_values"]
        assert len(tiny_p_doc["summary"]) == 3, tiny_p_doc["summary"]
        assert len(tiny_p_doc["summary_rows"]) == 3, tiny_p_doc["summary_rows"]
        assert tiny_p_doc["p_values"][0] < tiny_p_doc["p_values"][1] < tiny_p_doc["p_values"][2], tiny_p_doc["p_values"]
        expected_auto_threads = max(1, min(os.cpu_count() or 1, 2))
        threads_default_doc = run_case(
            exe,
            [
                "--N", "18",
                "--instances", "2",
                "--p-values", "1.0",
                "--sa-iters", "0",
                "--restarts", "1",
                "--tsp-restarts", "1",
                "--tsp-ils", "0",
                "--disable-two-opt",
                "--disable-or-opt",
            ],
            tmp / "threads-default-auto.json",
        )
        assert threads_default_doc["threads"] == expected_auto_threads, threads_default_doc["threads"]
        assert threads_default_doc["config"]["threads"] == expected_auto_threads, threads_default_doc["config"]

        threads_zero_doc = run_case(
            exe,
            [
                "--N", "18",
                "--instances", "2",
                "--threads", "0",
                "--p-values", "1.0",
                "--sa-iters", "0",
                "--restarts", "1",
                "--tsp-restarts", "1",
                "--tsp-ils", "0",
                "--disable-two-opt",
                "--disable-or-opt",
            ],
            tmp / "threads-zero-auto.json",
        )
        assert threads_zero_doc["threads"] == expected_auto_threads, threads_zero_doc["threads"]
        assert threads_zero_doc["config"]["threads"] == expected_auto_threads, threads_zero_doc["config"]

        bad_threads = subprocess.run(
            [str(exe), "--threads", "-1", "--dry-run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert bad_threads.returncode != 0, (bad_threads.stdout, bad_threads.stderr)
        assert "--threads must be >= 0" in bad_threads.stderr, bad_threads.stderr

        bad_tsp_candidates = subprocess.run(
            [str(exe), "--tsp-candidate-starts", "0", "--dry-run"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        assert bad_tsp_candidates.returncode != 0
        assert "--tsp-candidate-starts must be >= 1" in bad_tsp_candidates.stderr

        bad_relink_reserve = subprocess.run(
            [str(exe), "--path-relink-top", "2", "--path-relink-diverse-reserve", "3", "--dry-run"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        assert bad_relink_reserve.returncode != 0
        assert "--path-relink-diverse-reserve must not exceed" in bad_relink_reserve.stderr

        bad_campaign_shard = subprocess.run(
            [str(exe), "--campaign-shard", "-1", "--dry-run"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        assert bad_campaign_shard.returncode != 0
        assert "--campaign-shard must be >= 0" in bad_campaign_shard.stderr

        bad_pair_gate = subprocess.run(
            [str(exe), "--pair-exchange-max-k", "-1", "--dry-run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert bad_pair_gate.returncode != 0, (bad_pair_gate.stdout, bad_pair_gate.stderr)
        assert "--pair-exchange-max-k must be >= 0" in bad_pair_gate.stderr, bad_pair_gate.stderr

        bad_exact_limit = subprocess.run(
            [str(exe), "--exact-subset-max-n", "19", "--dry-run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert bad_exact_limit.returncode != 0, (bad_exact_limit.stdout, bad_exact_limit.stderr)
        assert "--exact-subset-max-n must be in [0,18]" in bad_exact_limit.stderr, bad_exact_limit.stderr

        bad_racing_quota = subprocess.run(
            [str(exe), "--racing-candidates", "2", "--racing-survivors", "3", "--dry-run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert bad_racing_quota.returncode != 0, (bad_racing_quota.stdout, bad_racing_quota.stderr)
        assert "--racing-survivors must not exceed" in bad_racing_quota.stderr, bad_racing_quota.stderr

        bad_racing_anytime = subprocess.run(
            [str(exe), "--racing-candidates", "2", "--racing-survivors", "1",
             "--time-budget-per-p", "0.1", "--dry-run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert bad_racing_anytime.returncode != 0, (bad_racing_anytime.stdout, bad_racing_anytime.stderr)
        assert "incompatible with --time-budget-per-p" in bad_racing_anytime.stderr, bad_racing_anytime.stderr




        bool_false_doc = run_case(
            exe,
            [
                "--N", "24",
                "--instances", "1",
                "--threads", "1",
                "--p-values", "1.0",
                "--sa-iters", "0",
                "--restarts", "1",
                "--tsp-restarts", "1",
                "--tsp-ils", "0",
                "--disable-two-opt=false",
                "--disable-or-opt=off",
                "--include-instance-rows=yes",
            ],
            tmp / "boolean-false-and-yes.json",
        )
        assert bool_false_doc["config"]["disable_two_opt"] is False, bool_false_doc["config"]
        assert bool_false_doc["config"]["disable_or_opt"] is False, bool_false_doc["config"]
        assert bool_false_doc["config"]["include_instance_rows"] is True, bool_false_doc["config"]
        assert len(bool_false_doc["instance_rows"]) == 1, bool_false_doc["instance_rows"]

        bad_bool = subprocess.run(
            [str(exe), "--disable-two-opt=maybe", "--dry-run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert bad_bool.returncode != 0, (bad_bool.stdout, bad_bool.stderr)
        assert "Invalid boolean value" in bad_bool.stderr, bad_bool.stderr

        quick_before = subprocess.run(
            [str(exe), "--quick", "--N", "30", "--instances", "1", "--dry-run"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        quick_after = subprocess.run(
            [str(exe), "--N", "30", "--instances", "1", "--quick", "--dry-run"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert "N=30" in quick_before.stdout, quick_before.stdout
        assert "N=30" in quick_after.stdout, quick_after.stdout
        assert "instances=1" in quick_before.stdout, quick_before.stdout
        assert "instances=1" in quick_after.stdout, quick_after.stdout



        existing = tmp / "existing.json"
        existing.write_text("{}")
        no_force = subprocess.run(
            [
                str(exe),
                "--N", "12",
                "--instances", "1",
                "--threads", "1",
                "--p-values", "1.0",
                "--sa-iters", "0",
                "--restarts", "1",
                "--tsp-restarts", "1",
                "--tsp-ils", "0",
                "--output", str(existing),
                "--force=false",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert no_force.returncode != 0, (no_force.stdout, no_force.stderr)
        assert "Refusing to overwrite" in no_force.stderr, no_force.stderr

        quick_false = subprocess.run(
            [str(exe), "--quick=false", "--N", "30", "--instances", "1", "--p-values", "1.0", "--dry-run"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert "N=30" in quick_false.stdout, quick_false.stdout
        assert "p_values=1" in quick_false.stdout, quick_false.stdout

        highp_doc = run_case(
            exe,
            [
                "--mode", "hybrid",
                "--N", "72",
                "--instances", "1",
                "--threads", "1",
                "--p-values", "0.75,1.0",
                "--sa-iters", "0",
                "--restarts", "1",
                "--tsp-restarts", "2",
                "--tsp-ils", "0",
                "--disable-two-opt",
                "--disable-or-opt",
                "--disable-subset-swap",
                "--disable-pair-exchange",
                "--disable-ruin-recreate",
                "--disable-path-relink",
                "--disable-smallp-seeds",
            ],
            tmp / "highp-accounting.json",
        )
        assert highp_doc["search_stats"]["subset_swap_scans"] == 0, highp_doc["search_stats"]
        assert highp_doc["search_stats"]["highp_exchange_scans"] > 0, highp_doc["search_stats"]

        top0_doc = run_case(
            exe,
            [
                "--N", "30",
                "--instances", "1",
                "--threads", "1",
                "--p-values", "1.0",
                "--sa-iters", "0",
                "--restarts", "1",
                "--tsp-restarts", "4",
                "--tsp-ils", "0",
                "--disable-two-opt",
                "--disable-or-opt",
                "--oracle", "lkh",
                "--lkh-path", str(fake_lkh),
                "--oracle-min-k", "17",
                "--oracle-max-k", "50",
                "--oracle-tsp-top", "0",
            ],
            tmp / "oracle-top0.json",
        )
        assert top0_doc["search_stats"]["oracle_tsp_calls"] == 0, top0_doc["search_stats"]
        assert top0_doc["oracle_call_records"] == [], top0_doc["oracle_call_records"]

        top2_doc = run_case(
            exe,
            [
                "--N", "30",
                "--instances", "1",
                "--threads", "1",
                "--p-values", "1.0",
                "--sa-iters", "0",
                "--restarts", "1",
                "--tsp-restarts", "4",
                "--tsp-ils", "0",
                "--disable-two-opt",
                "--disable-or-opt",
                "--oracle", "lkh",
                "--lkh-path", str(fake_lkh),
                "--oracle-min-k", "17",
                "--oracle-max-k", "50",
                "--oracle-tsp-top", "2",
            ],
            tmp / "oracle-top2.json",
        )
        calls = top2_doc["search_stats"]["oracle_tsp_calls"]
        expected_oracle_hash = hashlib.sha256(fake_lkh.read_bytes()).hexdigest()
        assert top2_doc["config"]["oracle_exec_sha256"] == expected_oracle_hash, top2_doc["config"]
        assert calls == 2, top2_doc["search_stats"]
        assert len(top2_doc["oracle_call_records"]) == 2, top2_doc["oracle_call_records"]
        assert {r["type"] for r in top2_doc["oracle_call_records"]} == {"tsp"}, top2_doc["oracle_call_records"]
        assert all(r["exec_sha256"] == expected_oracle_hash for r in top2_doc["oracle_call_records"]), top2_doc["oracle_call_records"]
        assert all(r["solver_version"] for r in top2_doc["oracle_call_records"]), top2_doc["oracle_call_records"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
