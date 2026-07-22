#!/usr/bin/env python3
"""Generate an in-toto Statement carrying SLSA provenance v1 metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--builder-id", default="local:aldous-tsp-release")
    return parser.parse_args()


def timestamp() -> str:
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    now = datetime.fromtimestamp(int(raw), timezone.utc) if raw else datetime.now(timezone.utc)
    return now.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    subjects = [
        {"name": path.name, "digest": {"sha256": sha256_file(path)}}
        for path in sorted(args.artifact, key=lambda value: value.name)
    ]
    created = timestamp()
    statement = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": subjects,
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": (
                    "https://github.com/MatthiasIrlbeck/subset-selection-TSP/"
                    "release/v1"
                ),
                "externalParameters": {
                    "version": args.version,
                    "gitCommit": args.commit,
                    "gitTree": args.tree,
                    "sourceDateEpoch": os.environ.get("SOURCE_DATE_EPOCH", ""),
                },
                "internalParameters": {
                    "githubWorkflow": os.environ.get("GITHUB_WORKFLOW", ""),
                    "githubRunId": os.environ.get("GITHUB_RUN_ID", ""),
                    "githubRunAttempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
                },
                "resolvedDependencies": [
                    {
                        "uri": (
                            "git+https://github.com/MatthiasIrlbeck/"
                            f"subset-selection-TSP@{args.commit}"
                        ),
                        "digest": {"gitCommit": args.commit, "gitTree": args.tree},
                    }
                ],
            },
            "runDetails": {
                "builder": {"id": args.builder_id},
                "metadata": {
                    "invocationId": os.environ.get("GITHUB_RUN_ID", "local"),
                    "startedOn": created,
                    "finishedOn": created,
                },
                "byproducts": [],
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(statement, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
