#!/usr/bin/env python3
"""Quality canary: catch gross search-quality regressions in CI.

Runs a fixed-seed workload and asserts that each per-p mean L(k)/k estimate
stays within a tolerance of recorded reference values. The solver is bitwise
deterministic for a fixed seed on a given binary, but floating-point library
and code-generation differences across compilers can legitimately shift search
trajectories, so the tolerance is intentionally coarse (6%): this canary is
meant to catch broken neighborhoods and budget plumbing, not sub-percent
quality drift. Lower means are better and never fail.

Reference values were recorded with the workload below (GCC/Linux, Release):
    aldous_tsp --N 250 --instances 8 --p-values 0.05,0.2,0.5 --seed 4242 \
        --sa-iters 30000

To re-baseline after an intentional search change, run the workload and
replace REFERENCE_MEANS with the new per-p means.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REFERENCE_MEANS = {
    "0.05": 0.555024,
    "0.2": 0.70861,
    "0.5": 0.671242,
}
TOLERANCE = 1.06

WORKLOAD = [
    "--N", "250",
    "--instances", "8",
    "--threads", "0",
    "--p-values", "0.05,0.2,0.5",
    "--seed", "4242",
    "--sa-iters", "30000",
    "--force",
]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: quality_canary.py <aldous_tsp_exe> <out_dir>", file=sys.stderr)
        return 2
    exe = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "quality_canary.json"

    cmd = [str(exe), *WORKLOAD, "--output", str(out_json)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"canary run failed: {proc.stderr}", file=sys.stderr)
        return 1

    doc = json.loads(out_json.read_text())
    failures: list[str] = []
    seen: set[str] = set()
    for row in doc["summary_rows"]:
        key = f"{row['p']:g}"
        if key not in REFERENCE_MEANS:
            continue
        seen.add(key)
        reference = REFERENCE_MEANS[key]
        mean = float(row["mean"])
        limit = reference * TOLERANCE
        status = "ok" if mean <= limit else "FAIL"
        print(f"p={key}: mean={mean:.6f} reference={reference:.6f} limit={limit:.6f} [{status}]")
        if mean > limit:
            failures.append(f"p={key}: mean {mean:.6f} exceeds {limit:.6f} (reference {reference:.6f} * {TOLERANCE})")
    missing = set(REFERENCE_MEANS) - seen
    if missing:
        failures.append(f"missing p rows in canary output: {sorted(missing)}")

    if failures:
        for line in failures:
            print(f"quality canary failure: {line}", file=sys.stderr)
        return 1
    print("quality canary passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
