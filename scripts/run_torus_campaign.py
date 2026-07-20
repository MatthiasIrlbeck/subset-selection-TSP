#!/usr/bin/env python3
"""Run the subset-TSP campaign: torus k-ladders with LKH, then extrapolate + fit alpha.

For each p and each target tour size k it runs one instance batch at N = k/p on
the flat torus, using LKH as the oracle to improve tour ordering on each chosen
subset, plus the full-set control variate. Conditional two-NN/Held-Karp bounds
diagnose the tour through the chosen subset; they do not certify that the subset
itself is globally optimal. The script then extrapolates every p to N -> inf via
extrapolate_fpN.py and, given f(0+), fits the small-p exponent.

Only the EXPLICIT matrix format is torus-correct and it is O(k^2), so keep k
targets <= ~3000; the torus's O(1/N) convergence makes moderate k sufficient.

Example (real LKH required):
  scripts/run_torus_campaign.py --exe ./build/aldous_tsp --lkh-path /usr/bin/LKH \\
      --ps 0.02,0.05,0.1,0.2,0.4 --ks 250,500,1000,2000 --instances 24
Then, once f(0+) is estimated (e.g. from the smallest-p extrapolation):
  ... --f0 0.62
"""
import argparse
import itertools
import json
import os
import subprocess
import sys


def run_batch(exe, lkh, p, N, instances, runs, out, metadata):
    cmd = [
        exe, "--N", str(N), "--instances", str(instances), "--p-values", f"{p:g}",
        "--periodic", "--control-variate", "--held-karp",
        "--include-instance-rows",
        "--campaign-id", metadata["campaign_id"],
        "--campaign-shard", str(metadata["campaign_shard"]),
        "--replicate-offset", str(metadata["replicate_offset"]),
        "--point-seed", str(metadata["point_seed"]),
        "--search-seed", str(metadata["search_seed"]),
        "--solver-policy-id", metadata["solver_policy_id"],
        "--fidelity-level", metadata["fidelity_level"],
        "--oracle", "lkh", "--oracle-format", "matrix", "--lkh-path", lkh,
        "--oracle-tsp-top", "1", "--oracle-subset-top", "1",
        "--oracle-max-k", "3000", "--oracle-lkh-runs", str(runs),
        "--output", out, "--force",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--lkh-path", required=True)
    ap.add_argument("--ps", default="0.02,0.05,0.1,0.2,0.4",
                    help="comma-separated p values")
    ap.add_argument("--ks", default="250,500,1000,2000",
                    help="comma-separated target tour sizes k (keep <= ~3000)")
    ap.add_argument("--instances", type=int, default=24)
    ap.add_argument("--lkh-runs", type=int, default=10)
    ap.add_argument("--out-dir", default="torus_campaign")
    ap.add_argument("--campaign-id", default="torus-campaign")
    ap.add_argument("--campaign-shard", type=int, default=0)
    ap.add_argument("--replicate-offset", type=int, default=0)
    ap.add_argument("--point-seed", type=int, default=2024)
    ap.add_argument("--search-seed", type=int, default=2024)
    ap.add_argument("--solver-policy-id", default="publication")
    ap.add_argument("--fidelity-level", default="strong")
    ap.add_argument("--f0", type=float, default=None,
                    help="f(0+) estimate; enables the alpha fit")
    ap.add_argument("--min-n", type=int, default=40,
                    help="skip (p,k) whose N=k/p is below this")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the (p,k,N) plan without running")
    args = ap.parse_args()
    if args.campaign_shard < 0:
        ap.error("--campaign-shard must be nonnegative")
    if args.replicate_offset < 0:
        ap.error("--replicate-offset must be nonnegative")

    metadata = {
        "campaign_id": args.campaign_id,
        "campaign_shard": args.campaign_shard,
        "replicate_offset": args.replicate_offset,
        "point_seed": args.point_seed,
        "search_seed": args.search_seed,
        "solver_policy_id": args.solver_policy_id,
        "fidelity_level": args.fidelity_level,
    }

    ps = [float(x) for x in args.ps.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    plan = []
    for p, k in itertools.product(ps, ks):
        N = max(k, round(k / p))
        if N < args.min_n:
            continue
        plan.append((p, k, N))

    print(f"campaign plan: {len(plan)} (p,k) points")
    print(f"{'p':>7} {'k':>6} {'N=k/p':>8}")
    for p, k, N in plan:
        print(f"{p:>7g} {k:>6} {N:>8}")
    if args.dry_run:
        return 0

    files = []
    for p, k, N in plan:
        out = os.path.join(args.out_dir, f"p{p:g}_k{k}.json")
        print(f"  running p={p:g} k={k} N={N} ...", flush=True)
        run_batch(
            args.exe, args.lkh_path, p, N, args.instances, args.lkh_runs,
            out, metadata,
        )
        files.append(out)

    here = os.path.dirname(os.path.abspath(__file__))
    extr = os.path.join(here, "extrapolate_fpN.py")
    extr_cmd = [sys.executable, extr] + files
    if args.f0 is not None:
        extr_cmd += ["--alpha", "--f0", str(args.f0)]
    print("\n=== per-p extrapolation to N -> inf, and alpha fit ===")
    res = subprocess.run(extr_cmd, capture_output=True, text=True)
    print(res.stdout, end="")
    if res.stderr:
        print(res.stderr, file=sys.stderr)

    # Also drop a figure.
    fig = os.path.join(args.out_dir, "extrapolation.png")
    subprocess.run([sys.executable, extr, "--plot", fig] + files,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"\nfigure: {fig}")
    print(f"For f(0+) and alpha with bootstrap confidence intervals, run:\n"
          f"  scripts/analyze_campaign.py --solver-policy-id {args.solver_policy_id} "
          f"--fidelity-level {args.fidelity_level} {args.out_dir}/*.json")
    if args.f0 is None:
        print("Tip: estimate f(0+) from the smallest-p intercept (or a p->0 fit of "
              "f(p)) and rerun with --f0 to fit alpha.")
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())
