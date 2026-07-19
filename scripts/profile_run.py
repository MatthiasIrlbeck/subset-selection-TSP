#!/usr/bin/env python3
"""Run one CLI scenario and summarize phase-level timing/counter telemetry."""
from __future__ import annotations
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

def fmt_seconds(value: float) -> str:
    return f"{value:.4f}s"

def main() -> int:
    parser = argparse.ArgumentParser(description="Run aldous_tsp and summarize profiling counters.")
    parser.add_argument("--exe", default="build/aldous_tsp")
    parser.add_argument("--output", default=None)
    parser.add_argument("--N", type=int, default=240)
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--p-values", default="0.05,0.10,0.25,0.50,1.0")
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    exe = Path(args.exe)
    exe_check = exe if exe.is_absolute() else Path.cwd() / exe
    if not exe_check.exists():
        print(f"executable not found: {exe}", file=sys.stderr); return 2
    tmp_ctx = None
    if args.output is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="aldous_profile_")
        output = Path(tmp_ctx.name) / "profile.json"
    else:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(exe), "--mode", args.mode, "--N", str(args.N), "--instances", str(args.instances), "--threads", str(args.threads), "--p-values", args.p_values, "--output", str(output), "--force", *args.extra]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    doc = json.loads(output.read_text())
    stats = doc.get("search_stats", {})
    print("\nTiming summary\n--------------")
    print(f"wall_seconds      {fmt_seconds(float(doc.get('wall_seconds', 0.0)))}")
    print(f"knn_build_seconds {fmt_seconds(float(stats.get('knn_build_seconds', 0.0)))}")
    print(f"tsp_seconds       {fmt_seconds(float(stats.get('tsp_seconds', 0.0)))}")
    print(f"subset_seconds    {fmt_seconds(float(stats.get('subset_seconds', 0.0)))}")
    oracle_seconds = sum(float(r.get("seconds", 0.0)) for r in doc.get("oracle_call_records", []))
    print(f"oracle_seconds    {fmt_seconds(oracle_seconds)}")
    print("\nEffective KNN\n-------------")
    for key in ["knn_requested_grid_instances", "knn_requested_bruteforce_instances", "knn_effective_grid_instances", "knn_effective_bruteforce_instances", "knn_bruteforce_fallback_instances", "knn_grid_cell_capped_instances", "knn_grid_cells_max", "grid_cell_effective_min", "grid_cell_effective_max"]:
        print(f"{key:36s} {stats.get(key, 0)}")
    print("\nMove counters\n-------------")
    for key in ["two_opt_scans","two_opt_improvements","or_opt_scans","or_opt_improvements","sa_moves","sa_accepted","subset_swap_scans","subset_swap_improvements","highp_exchange_scans","highp_exchange_improvements","pair_exchange_scans","pair_exchange_improvements","ruin_recreate_attempts","ruin_recreate_improvements","path_relink_attempts","path_relink_feasible","path_relink_best_improvements","oracle_calls","oracle_improved"]:
        print(f"{key:32s} {stats.get(key, 0)}")
    print(f"\nResult JSON: {output}")
    if tmp_ctx is not None: tmp_ctx.cleanup()
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
