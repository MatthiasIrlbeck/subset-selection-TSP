#!/usr/bin/env python3
"""Compare exhaustive 2-opt policies on one deterministic scenario.

The default solver policy is ``final-only``: exhaustive 2-opt is reserved for
final reporting polish while regular search polishing uses candidate 2-opt. This
script compares that default against the legacy ``all-polish`` behavior and
writes a CSV manifest with runtime, quality, and two-opt scan deltas.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


def best_mean(doc: dict) -> float:
    rows = doc.get("summary_rows") or []
    if not rows:
        vals = [float(row["mean"]) for row in doc.get("summary", {}).values()]
        return min(vals) if vals else math.nan
    return min(float(row["mean"]) for row in rows)


def run_policy(exe: Path, out_dir: Path, policy: str, args: argparse.Namespace) -> dict:
    output = out_dir / f"{policy}.json"
    cmd = [
        str(exe),
        "--mode", args.mode,
        "--N", str(args.N),
        "--instances", str(args.instances),
        "--threads", str(args.threads),
        "--p-values", args.p_values,
        "--seed", str(args.seed),
        "--restarts", str(args.restarts),
        "--sa-iters", str(args.sa_iters),
        "--tsp-restarts", str(args.tsp_restarts),
        "--tsp-ils", str(args.tsp_ils),
        "--tsp-patience", str(args.tsp_patience),
        "--final-exhaustive-k", str(args.final_exhaustive_k),
        "--exhaustive-two-opt-policy", policy,
        "--output", str(output),
        "--force",
        *args.extra,
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    doc = json.loads(output.read_text())
    stats = doc.get("search_stats", {})
    return {
        "policy": policy,
        "output": str(output),
        "wall_seconds": float(doc.get("wall_seconds", 0.0)),
        "best_mean": best_mean(doc),
        "two_opt_scans": int(stats.get("two_opt_scans", 0)),
        "two_opt_improvements": int(stats.get("two_opt_improvements", 0)),
        "or_opt_scans": int(stats.get("or_opt_scans", 0)),
        "knn_build_seconds": float(stats.get("knn_build_seconds", 0.0)),
        "tsp_seconds": float(stats.get("tsp_seconds", 0.0)),
        "subset_seconds": float(stats.get("subset_seconds", 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare final-only vs all-polish exhaustive 2-opt policies.")
    parser.add_argument("--exe", default="build/aldous_tsp")
    parser.add_argument("--out-dir", default="exhaustive-policy-benchmark")
    parser.add_argument("--N", type=int, default=120)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--p-values", default="0.10,0.25,0.50,1.0")
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--sa-iters", type=int, default=400)
    parser.add_argument("--tsp-restarts", type=int, default=2)
    parser.add_argument("--tsp-ils", type=int, default=20)
    parser.add_argument("--tsp-patience", type=int, default=8)
    parser.add_argument("--final-exhaustive-k", type=int, default=160)
    parser.add_argument("--check", action="store_true", help="Fail if final-only scans exceed all-polish scans.")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    exe = Path(args.exe)
    exe_check = exe if exe.is_absolute() else Path.cwd() / exe
    if not exe_check.exists():
        print(f"executable not found: {exe}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [run_policy(exe, out_dir, "final-only", args), run_policy(exe, out_dir, "all-polish", args)]
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_policy = {row["policy"]: row for row in rows}
    final = by_policy["final-only"]
    legacy = by_policy["all-polish"]
    scan_delta = legacy["two_opt_scans"] - final["two_opt_scans"]
    quality_delta = final["best_mean"] - legacy["best_mean"]
    print("\nPolicy comparison")
    print("-----------------")
    print(f"final-only scans : {final['two_opt_scans']}")
    print(f"all-polish scans : {legacy['two_opt_scans']}")
    print(f"scan reduction   : {scan_delta}")
    print(f"quality delta    : {quality_delta:.12g} (final-only minus all-polish best mean)")
    print(f"manifest         : {manifest}")

    if args.check and final["two_opt_scans"] > legacy["two_opt_scans"]:
        print("final-only unexpectedly used more two-opt scans than all-polish", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
