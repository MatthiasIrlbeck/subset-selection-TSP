#!/usr/bin/env python3
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
        "Boolean flags accept plain presence as true",
    ):
        assert marker in help_run.stdout, marker
    assert "+58% wall" in help_run.stdout, help_run.stdout
    assert "~99% of optimal" in help_run.stdout, help_run.stdout
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
        assert calls == 2, top2_doc["search_stats"]
        assert len(top2_doc["oracle_call_records"]) == 2, top2_doc["oracle_call_records"]
        assert {r["type"] for r in top2_doc["oracle_call_records"]} == {"tsp"}, top2_doc["oracle_call_records"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
