#!/usr/bin/env python3
"""Run the full study end to end: robustness sweep, Aldous campaign, analysis.

Designed to be started once and left alone. It:
  * auto-detects LKH (explicit --lkh-path, else PATH) and uses it when present,
    falling back to the built-in solver with extra restarts otherwise -- so a
    missing or half-built LKH never blocks the run;
  * is resumable -- batches whose output JSON already exists and parses are
    skipped, so re-running after any interruption continues where it left off;
  * is error-isolated -- a failing batch is logged and the run continues;
  * logs progress with timestamps to <out-dir>/run.log and the console.

Stages (select with --stage; default all):
  sweep    - robustness grid over p on both boundary conditions, plus a large-N
             throughput point. Confirms the code behaves for all reasonable
             parameter choices.
  campaign - small-p torus k-ladders (N = k/p) with the control variate and
             Held-Karp bound, LKH-polished when available.
  analysis - extrapolate f(p) and fit f(0+), alpha with bootstrap CIs.

Example (Windows):
  python scripts\\run_full_study.py --exe .\\build\\Release\\aldous_tsp.exe --threads 12
  python scripts\\run_full_study.py --exe .\\build\\Release\\aldous_tsp.exe --lkh-path C:\\tools\\LKH.exe --threads 12
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg, logf):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    logf.write(line + "\n")
    logf.flush()


def detect_lkh(explicit):
    if explicit:
        if os.path.isfile(explicit) or shutil.which(explicit):
            return explicit
        return None
    for name in ("LKH", "LKH.exe", "lkh"):
        found = shutil.which(name)
        if found:
            return found
    return None


def valid_json(path):
    if not os.path.exists(path):
        return False
    try:
        d = json.load(open(path))
        return bool(d.get("summary_rows"))
    except Exception:
        return False


def run_batch(exe, out, base_args, oracle_args, restarts_if_builtin, timeout, logf, tag):
    if valid_json(out):
        log(f"  skip (exists): {tag}", logf)
        return True
    cmd = [exe] + base_args
    if oracle_args:
        cmd += oracle_args
    else:
        cmd += ["--restarts", str(restarts_if_builtin)]
    cmd += ["--output", out, "--include-instance-rows", "--force"]
    t0 = datetime.datetime.now()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT: {tag} (> {timeout}s) -- skipping", logf)
        return False
    dt = (datetime.datetime.now() - t0).total_seconds()
    if r.returncode != 0 or not valid_json(out):
        detail = (r.stderr or r.stdout or "").strip().replace("\n", " ")[:200]
        log(f"  FAIL: {tag} (exit {r.returncode}, {dt:.0f}s) :: {detail}", logf)
        return False
    log(f"  ok: {tag} ({dt:.0f}s)", logf)
    return True


def stage_sweep(exe, out_dir, instances, threads, sa_iters_per_n, timeout, logf):
    log("=== STAGE: robustness sweep (both boundary conditions) ===", logf)
    d = os.path.join(out_dir, "sweep")
    os.makedirs(d, exist_ok=True)
    pl = "0.005,0.02,0.05,0.1,0.2,0.5,1.0"
    # The sweep gets the same validated budget as the campaign: its batches face
    # the same subset-selection problem, so a flat budget under-converges them
    # the same way, just at these fixed N.
    common = ["--instances", str(instances), "--threads", str(threads),
              "--control-variate", "--held-karp",
              "--sa-iters-per-n", str(sa_iters_per_n)]
    configs = [
        ("sweep_tor_2000", ["--N", "2000", "--p-values", pl, "--periodic"] + common),
        ("sweep_sq_2000", ["--N", "2000", "--p-values", pl] + common),
        # Large-N throughput/memory check: HK off (k=N too big), fewer instances.
        ("sweep_tor_100k", ["--N", "100000", "--instances", str(max(4, instances // 3)),
                            "--threads", str(threads), "--p-values", "0.002,0.01,0.05",
                            "--periodic", "--control-variate",
                            "--sa-iters-per-n", str(sa_iters_per_n)]),
    ]
    ok = 0
    for tag, args in configs:
        if run_batch(exe, os.path.join(d, tag + ".json"), args, None, 8, timeout, logf, tag):
            ok += 1
    log(f"sweep done: {ok}/{len(configs)} batches ok", logf)


def stage_campaign(exe, out_dir, lkh, ps, ks, instances, threads, lkh_runs,
                   builtin_restarts, sa_iters_per_n, max_n, timeout, logf):
    mode = f"LKH ({lkh})" if lkh else "built-in solver (no LKH found)"
    log(f"=== STAGE: Aldous campaign -- solver mode: {mode} ===", logf)
    d = os.path.join(out_dir, "campaign")
    os.makedirs(d, exist_ok=True)
    oracle = None
    if lkh:
        oracle = ["--oracle", "lkh", "--oracle-format", "matrix", "--lkh-path", lkh,
                  "--oracle-tsp-top", "1", "--oracle-subset-top", "1",
                  "--oracle-max-k", "3000", "--oracle-lkh-runs", str(lkh_runs),
                  "--restarts", str(builtin_restarts)]
    plan = []
    for p in ps:
        for k in ks:
            N = max(k, round(k / p))
            if N > max_n:
                log(f"  skip (N={N} > max-n={max_n}): p={p:g} k={k}", logf)
                continue
            plan.append((p, k, N))
    if sa_iters_per_n <= 0:
        worst = max(N for _, _, N in plan) if plan else 0
        log("  *** WARNING: --sa-iters-per-n is 0, so the subset-search budget is FLAT in N.", logf)
        log(f"  *** The largest batch has N={worst} candidates. A flat budget cannot even propose", logf)
        log("  *** each candidate once there, so the extra candidates that a small p buys are never", logf)
        log("  *** examined. That biases f(p) upward, and MORE at small p -- which fabricates the", logf)
        log("  *** very p-trend this campaign exists to measure. This is exactly how the July 2026", logf)
        log("  *** run produced f(0+)=0.709 / alpha=3.01, both artefacts.", logf)
        log("  *** Run scripts/convergence_study.py first and pass a validated --sa-iters-per-n.", logf)
    log(f"campaign plan: {len(plan)} batches", logf)
    ok = 0
    for p, k, N in plan:
        tag = f"p{p:g}_k{k}"
        base = ["--N", str(N), "--instances", str(instances), "--threads", str(threads),
                "--p-values", f"{p:g}", "--periodic", "--control-variate", "--held-karp",
                "--sa-iters-per-n", str(sa_iters_per_n)]
        if run_batch(exe, os.path.join(d, tag + ".json"), base, oracle,
                     builtin_restarts, timeout, logf, tag):
            ok += 1
    log(f"campaign done: {ok}/{len(plan)} batches ok", logf)


def stage_analysis(out_dir, pmax, boot, logf):
    log("=== STAGE: analysis ===", logf)
    d = os.path.join(out_dir, "campaign")
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")] if os.path.isdir(d) else []
    files = [f for f in files if valid_json(f)]
    if len(files) < 2:
        log("  not enough campaign JSONs for analysis; skipping", logf)
        return
    report = os.path.join(out_dir, "analysis_report.txt")
    with open(report, "w") as rep:
        for script, extra in [("extrapolate_fpN.py", []),
                              ("analyze_campaign.py",
                               ["--pmax", str(pmax), "--boot", str(boot),
                                "--plot", os.path.join(out_dir, "fit.png")])]:
            cmd = [sys.executable, os.path.join(HERE, script)] + extra + files
            r = subprocess.run(cmd, capture_output=True, text=True)
            rep.write(f"$ {script} {' '.join(extra)}\n{r.stdout}\n{r.stderr}\n\n")
            log(f"  {script}:\n{r.stdout.rstrip()}", logf)
    log(f"analysis written to {report}", logf)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True, help="path to aldous_tsp(.exe)")
    ap.add_argument("--lkh-path", default=None, help="path to LKH; auto-detected on PATH if omitted")
    ap.add_argument("--out-dir", default="study")
    ap.add_argument("--stage", choices=["all", "sweep", "campaign", "analysis"], default="all")
    ap.add_argument("--instances", type=int, default=24)
    ap.add_argument("--threads", type=int, default=0, help="0 = auto")
    ap.add_argument("--ps", default="0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--ks", default="250,500,1000,2000")
    ap.add_argument("--lkh-runs", type=int, default=10)
    ap.add_argument("--builtin-restarts", type=int, default=8,
                    help="subset restarts when LKH is not used")
    ap.add_argument("--sa-iters-per-n", type=int, default=0,
                    help="SA iterations per CANDIDATE point (pool size N=k/p). The default 0 "
                         "keeps the historical FLAT budget, which leaves the subset search "
                         "badly under-converged at small p and fabricates a p-trend in f(p). "
                         "Validate a value with scripts/convergence_study.py before trusting a campaign.")
    ap.add_argument("--max-n", type=int, default=300000, help="skip campaign (p,k) with N above this")
    ap.add_argument("--pmax", type=float, default=0.2, help="analysis small-p cutoff")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--batch-timeout", type=int, default=0, help="per-batch seconds (0 = none)")
    args = ap.parse_args()

    if not (os.path.isfile(args.exe) or shutil.which(args.exe)):
        sys.exit(f"executable not found: {args.exe}")
    os.makedirs(args.out_dir, exist_ok=True)
    timeout = args.batch_timeout or None
    threads = args.threads if args.threads > 0 else os.cpu_count() or 1
    ps = [float(x) for x in args.ps.split(",")]
    ks = [int(x) for x in args.ks.split(",")]

    with open(os.path.join(args.out_dir, "run.log"), "a") as logf:
        lkh = detect_lkh(args.lkh_path)
        log("################ full study run ################", logf)
        log(f"exe={args.exe}  threads={threads}  instances={args.instances}", logf)
        log(f"LKH: {'FOUND -> ' + lkh if lkh else 'not found -> built-in fallback (--restarts %d)' % args.builtin_restarts}", logf)
        log(f"campaign ps={ps} ks={ks} max-n={args.max_n}", logf)

        if args.stage in ("all", "sweep"):
            stage_sweep(args.exe, args.out_dir, args.instances, threads,
                        args.sa_iters_per_n, timeout, logf)
        if args.stage in ("all", "campaign"):
            stage_campaign(args.exe, args.out_dir, lkh, ps, ks, args.instances, threads,
                           args.lkh_runs, args.builtin_restarts, args.sa_iters_per_n,
                           args.max_n, timeout, logf)
        if args.stage in ("all", "analysis"):
            stage_analysis(args.out_dir, args.pmax, args.boot, logf)
        log("################ done ################", logf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
