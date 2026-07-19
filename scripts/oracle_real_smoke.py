#!/usr/bin/env python3
"""Optional real LKH/Concorde smoke checks.

The script exits successfully when no real solver is present unless ``--require``
is passed. When an output directory is supplied it always writes a small manifest
so CI logs make the skip/run decision explicit.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_solver(exe: Path, solver: str, solver_path: str, fmt: str, out_dir: Path) -> dict:
    output = out_dir / f"real-{solver}-{fmt}.json"
    cmd = [
        str(exe),
        "--N", "32",
        "--instances", "1",
        "--threads", "1",
        "--p-values", "1.0",
        "--sa-iters", "0",
        "--restarts", "1",
        "--tsp-restarts", "2",
        "--tsp-ils", "0",
        "--oracle", solver,
        f"--{solver}-path", solver_path,
        "--oracle-format", fmt,
        "--oracle-min-k", "17",
        "--oracle-max-k", "64",
        "--oracle-tsp-top", "1",
        "--oracle-time-limit", "10",
        "--output", str(output),
        "--force",
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return json.loads(output.read_text())


def write_manifest(out_dir: Path, payload: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "real_oracle_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run optional real LKH/Concorde smoke checks.")
    parser.add_argument("--exe", default="build/aldous_tsp")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--lkh-path", default=None)
    parser.add_argument("--concorde-path", default=None)
    parser.add_argument("--require", action="store_true")
    args = parser.parse_args()

    exe = Path(args.exe)
    exe_check = exe if exe.is_absolute() else Path.cwd() / exe
    if not exe_check.exists():
        print(f"executable not found: {exe}", file=sys.stderr)
        return 2

    tmp_ctx = None
    if args.out_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="aldous_real_oracle_")
        out_dir = Path(tmp_ctx.name)
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[str, str]] = []
    lkh = args.lkh_path or shutil.which("LKH")
    con = args.concorde_path or shutil.which("concorde")
    if lkh:
        candidates.append(("lkh", lkh))
    if con:
        candidates.append(("concorde", con))

    if not candidates:
        payload = {
            "status": "skipped",
            "reason": "No real LKH or Concorde executable found",
            "solvers": [],
            "oracle_call_records": [],
        }
        write_manifest(out_dir, payload)
        print("No real LKH or Concorde executable found; real-oracle smoke skipped.")
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        return 1 if args.require else 0

    records: list[dict] = []
    errors: list[str] = []
    for solver, path in candidates:
        for fmt in ["matrix", "euc2d"]:
            try:
                doc = run_solver(exe, solver, path, fmt, out_dir)
            except subprocess.CalledProcessError as exc:
                errors.append(f"{solver}/{fmt} exited with {exc.returncode}")
                continue
            stats = doc.get("search_stats", {})
            solver_records = doc.get("oracle_call_records", [])
            records.extend(solver_records)
            if stats.get("oracle_calls", 0) < 1:
                errors.append(f"{solver}/{fmt} produced no oracle calls")
            if not solver_records:
                errors.append(f"{solver}/{fmt} produced no oracle_call_records")
            for record in solver_records:
                if record.get("status") not in {"improved", "solved_no_improvement", "not_applicable"}:
                    errors.append(f"{solver}/{fmt} unexpected record status: {record.get('status')}")

    payload = {
        "status": "failed" if errors else "passed",
        "solvers": candidates,
        "errors": errors,
        "oracle_call_records": records,
    }
    write_manifest(out_dir, payload)
    if tmp_ctx is not None:
        tmp_ctx.cleanup()
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
