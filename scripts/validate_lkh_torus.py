#!/usr/bin/env python3
"""Validate the LKH oracle on the flat torus: does p=1 converge to beta?

The built-in solver leaves a residual suboptimality at large k that biases the
finite-size extrapolation. LKH removes it. This harness runs a p=1 torus
k-ladder with LKH as the oracle, checks each point against the BHH constant
beta ~ 0.7124, and extrapolates (via extrapolate_fpN.py) to confirm the
intercept lands on beta. It must be run on a machine with a real LKH binary
(the in-repo reference_lkh.py is only an exactness stand-in for small k).

Only the EXPLICIT matrix format is torus-correct, so --oracle-format matrix is
forced; because the matrix is O(k^2), keep ladder k <= ~3000 (the torus's
O(1/N) convergence makes moderate k sufficient).

Example:
  scripts/validate_lkh_torus.py --exe ./build/aldous_tsp --lkh-path /usr/local/bin/LKH
"""
import argparse
import json
import os
import subprocess
import sys

BETA = 0.7124  # Beardwood-Halton-Hammersley constant, d=2 (Percus-Martin/JMR)


def run_point(exe, lkh, N, instances, runs, out):
    cmd = [
        exe, "--N", str(N), "--instances", str(instances), "--p-values", "1.0",
        "--periodic", "--control-variate",
        "--oracle", "lkh", "--oracle-format", "matrix",
        "--lkh-path", lkh, "--oracle-tsp-top", "1",
        "--oracle-max-k", "3000", "--oracle-lkh-runs", str(runs),
        "--output", out, "--force",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open(out, encoding="utf-8") as result_file:
        return json.load(result_file)["summary_rows"][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True, help="path to the aldous_tsp binary")
    ap.add_argument("--lkh-path", required=True, help="path to the real LKH binary")
    ap.add_argument("--ns", default="200,500,1000,2000",
                    help="comma-separated N ladder (k=N at p=1; keep <= ~3000)")
    ap.add_argument("--instances", type=int, default=16)
    ap.add_argument("--lkh-runs", type=int, default=10)
    ap.add_argument("--out-dir", default="lkh_torus_validation")
    ap.add_argument("--tol", type=float, default=0.004,
                    help="allowed |extrapolated f(1) - beta|")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ns = [int(x) for x in args.ns.split(",")]
    files = []
    print(f"{'N':>6} {'L/N (LKH)':>11} {'stderr':>8} {'|L/N - beta|':>13}")
    for N in ns:
        out = os.path.join(args.out_dir, f"p1_N{N}.json")
        row = run_point(args.exe, args.lkh_path, N, args.instances, args.lkh_runs, out)
        files.append(out)
        dev = abs(row["mean"] - BETA)
        print(f"{N:>6} {row['mean']:>11.4f} {row['stderr']:>8.4f} {dev:>13.4f}")

    # Extrapolate with the sibling tool.
    here = os.path.dirname(os.path.abspath(__file__))
    extr = os.path.join(here, "extrapolate_fpN.py")
    print("\n--- finite-size extrapolation (1/N form) ---")
    res = subprocess.run([sys.executable, extr] + files, capture_output=True, text=True)
    print(res.stdout, end="")
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        return 2

    # Parse the extrapolated intercept from the tool output (the p=1 row).
    fp = None
    for line in res.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0].strip().startswith("1 "):
            try:
                fp = float(parts[1].split()[0])
            except (ValueError, IndexError):
                # Keep scanning: unrelated table rows are intentionally ignored.
                continue
    if fp is None:
        print("\ncould not parse extrapolated f(1); inspect the table above", file=sys.stderr)
        return 2
    dev = abs(fp - BETA)
    verdict = "PASS" if dev <= args.tol else "FAIL"
    print(f"\nextrapolated f(1) = {fp:.4f}   beta = {BETA}   |diff| = {dev:.4f}   [{verdict}]")
    if verdict == "FAIL":
        print("  If this fails: raise --lkh-runs, add ladder points, or check that "
              "LKH is being invoked (rerun one point with --oracle ... and inspect "
              "oracle_solved/oracle_improved in the JSON search_stats).")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
