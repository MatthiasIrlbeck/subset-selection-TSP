#!/usr/bin/env python3
"""Tiny stand-in for the upstream original CLI used by parity-harness tests.

It accepts only the older option surface and intentionally fails if the parity
harness passes current-only flags such as --p-values or --knn-backend.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

DEFAULT_P = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]
FLAGS_WITH_VALUES = {
    "--mode", "--N", "--instances", "--restarts", "--sa-iters", "--seed", "--threads",
    "--knn", "--grid-cell", "--verify-knn", "--tsp-restarts", "--tsp-ils", "--tsp-patience",
    "--output", "--oracle", "--oracle-format", "--lkh-path", "--concorde-path",
    "--oracle-time-limit", "--oracle-scale", "--oracle-tsp-top", "--oracle-subset-top",
    "--oracle-min-k", "--oracle-max-k", "--oracle-lkh-runs", "--oracle-lkh-trials",
}
BOOLEAN_FLAGS = {
    "--help", "--quick", "--self-test", "--force", "--verbose-p", "--oracle-no-tsp",
    "--oracle-no-subset", "--oracle-inline-feedback", "--oracle-verbose",
}
CURRENT_ONLY_PREFIXES = (
    "--p-values", "--p-range", "--p-file", "--knn-backend", "--final-exhaustive-k",
    "--pair-exchange-passes", "--ruin-recreate-rounds", "--path-relink-top",
    "--disable-", "--subset-swap-passes", "--highp-delete-passes",
)


def fail(msg: str) -> int:
    print(msg, file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    if "--help" in argv:
        print("Usage: fake_original_cli [older options]")
        print("  --N --instances --restarts --sa-iters --threads --output --force")
        return 0
    out = Path("results.json")
    n = 500
    instances = 15
    i = 1
    while i < len(argv):
        arg = argv[i]
        if any(arg == p or arg.startswith(p + "=") for p in CURRENT_ONLY_PREFIXES):
            return fail(f"fake original received unsupported current-only flag: {arg}")
        if "=" in arg:
            flag, value = arg.split("=", 1)
            if flag not in FLAGS_WITH_VALUES:
                return fail(f"unknown flag: {flag}")
        else:
            flag = arg
            value = None
        if flag in FLAGS_WITH_VALUES:
            if value is None:
                i += 1
                if i >= len(argv):
                    return fail(f"missing value for {flag}")
                value = argv[i]
            if flag == "--output":
                out = Path(value)
            elif flag == "--N":
                n = int(value)
            elif flag == "--instances":
                instances = int(value)
        elif flag in BOOLEAN_FLAGS:
            pass
        else:
            return fail(f"unknown flag: {flag}")
        i += 1

    summary = {}
    for p in DEFAULT_P:
        k = max(3, round(p * n))
        mean = 0.5 + 0.1 * p
        key = f"{p:.4f}".rstrip("0").rstrip(".") if p != 1.0 else "1.0"
        summary[key] = {"k": k, "mean": mean, "std": 0.0, "stderr": 0.0, "n": instances, "values": [mean] * instances}
    doc = {
        "schema_version": 4,
        "N": n,
        "p_values": DEFAULT_P,
        "done": instances,
        "target": instances,
        "wall_seconds": 0.001,
        "summary": summary,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
