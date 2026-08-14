#!/usr/bin/env python3
"""Convergence harness for the SUBSET-SELECTION search.

Why this exists
---------------
f(p) is a MINIMUM over subsets. Every number the solver reports is therefore an
UPPER bound, and an under-converged search biases it upward. That bias is not
uniform: the candidate pool is N = k/p, so the search has more to do at small p,
and the upward bias is largest exactly where the small-p physics lives. An
under-converged run does not merely add noise to f(p) -- it manufactures a fake
p-trend, and the small-p exponent alpha is fitted straight out of that artefact.

This has already happened once in this project. A campaign ran with a flat
sa_iters = 60000 regardless of N. At (p=0.01, k=2000, N=200000) that is 0.3 SA
proposals per candidate point: the search could not look at each candidate even
once, so the 20x larger pool that p=0.01 buys was never examined. The resulting
fit produced f(0+) = 0.709 with alpha = 3.01 -- and both were artefacts.

So: never run a campaign without first establishing that the search is converged
at the campaign's budget. That is what this script is for.

Two stages
----------
budget
    Fix (p, k). Sweep the SA budget upward and watch L/k. If L/k is still
    falling when the budget stops rising, the search is NOT converged and any
    f(p) from that budget is an artefact. Reports the ladder plus a verdict.

monotone
    A hard mathematical invariant, and the cheapest real test of search quality.
    At FIXED k, a larger candidate pool can only help: choosing k points out of
    N' > N points admits every subset the smaller pool did, so

        the optimum L/k is NON-INCREASING in N.

    Any configuration whose measured L/k *rises* with N has a search that is
    failing to use the candidates it is given. This is a necessary condition,
    not a sufficient one -- passing it does not prove convergence -- but failing
    it is proof of a broken measurement, and it is what first exposed the bug
    above.

Notes on interpretation
-----------------------
* Run the budget sweep WITHOUT --lkh-path. The external oracle is a post-hoc
  polish of the final subset; it does not steer the subset search, so it only
  slows the sweep down. Read the trend, then confirm the chosen budget with LKH.
* Watch the HK bound column, not just L/k. The Held-Karp bound is computed on
  the subset the solver CHOSE, so it measures the quality of the SUBSET
  independently of how well the tour through it was solved. If the HK bound is
  still falling, the subsets are still improving and the search is not done.
* The HK bound is NOT a lower bound on f(p). It lower-bounds the tour length of
  the subset we found; a better subset drives the tour AND its HK bound down
  together. It brackets tour-solving error only, never subset-selection error.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys


def run_case(exe, N, p, instances, threads, restarts, sa_iters, sa_iters_per_n,
             lkh, out_path, timeout, extra_args=None):
    """Run one solver invocation and return (mean, stderr, hk_bound, seconds)."""
    cmd = [exe, "--N", str(N), "--instances", str(instances), "--threads", str(threads),
           "--p-values", f"{p:g}", "--periodic", "--held-karp",
           "--restarts", str(restarts), "--sa-iters", str(sa_iters),
           "--sa-iters-per-n", str(sa_iters_per_n),
           "--include-instance-rows",
           "--output", out_path, "--force"]
    if extra_args:
        cmd += list(extra_args)
    if lkh:
        cmd += ["--oracle", "lkh", "--lkh-path", lkh, "--oracle-format", "matrix",
                "--oracle-subset-top", "1", "--oracle-tsp-top", "1"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-500:] + "\n")
        return None
    with open(out_path) as f:
        doc = json.load(f)
    row = doc["summary_rows"][0]
    conditional_hk = row.get(
        "conditional_held_karp_bound_mean", row.get("held_karp_bound_mean", 0.0))
    return (row["mean"], row["stderr"], conditional_hk,
            doc.get("wall_seconds", 0.0))


def plateau_verdict(points, tol, instances=None):
    """Decide whether a budget ladder has converged.

    `points` is [(budget, mean, stderr), ...] in increasing budget order.
    Returns (status, message) with status in {converged, not_converged,
    inconclusive, insufficient}.

    The three outcomes that are NOT convergence are deliberately distinguished,
    because they call for opposite responses: a search that is still improving
    needs more BUDGET, while a ladder whose steps are buried in noise needs more
    INSTANCES. Reporting the second as "not converged" would send you off buying
    compute that cannot help.
    """
    if len(points) < 3:
        return ("insufficient", "need at least 3 budget points to judge a plateau")
    (_, m_prev, s_prev) = points[-2]
    (_, m_last, s_last) = points[-1]
    delta = m_last - m_prev
    sigma = math.hypot(s_prev, s_last)
    total = points[-1][1] - points[0][1]

    if sigma > 0.0 and delta < -2.0 * sigma:
        return ("not_converged",
                f"still improving by {delta:+.4f} on the last step ({abs(delta) / sigma:.1f} sigma); "
                f"total gain across the ladder {total:+.4f} -- raise the budget")
    if sigma >= tol:
        need = ""
        if instances:
            need = f"; about {math.ceil(instances * (sigma / tol) ** 2)} instances would resolve it"
        return ("inconclusive",
                f"cannot tell: the step stderr {sigma:.4f} is itself above the tolerance {tol:.4f}{need}")
    if abs(delta) >= tol:
        return ("not_converged",
                f"last step moved {delta:+.4f}, above tolerance {tol:.4f} -- raise the budget")
    return ("converged",
            f"last step {delta:+.4f} (within {tol:.4f}); total gain across the ladder {total:+.4f}")


def monotonicity_violations(rows, z=2.0):
    """Find violations of `optimum is non-increasing in N` at fixed k.

    `rows` is [(N, mean, stderr), ...] sorted by increasing N. Returns the list
    of (N_prev, N_curr, delta, sigmas) where the measured value ROSE with N by at
    least `z` sigma. A rise is mathematically impossible for the true optimum, so
    a significant one is direct proof the search is not using the larger pool.
    """
    out = []
    for i in range(1, len(rows)):
        n_prev, m_prev, s_prev = rows[i - 1]
        n_curr, m_curr, s_curr = rows[i]
        delta = m_curr - m_prev
        if delta <= 0.0:
            continue
        sigma = math.hypot(s_prev, s_curr)
        sigmas = delta / sigma if sigma > 0.0 else float("inf")
        if sigmas >= z or sigma == 0.0:
            out.append((n_prev, n_curr, delta, sigmas))
    return out


def no_gain_warning(rows, tol):
    """Detect the OTHER failure mode: the pool grows a lot and nothing improves.

    A rise is a proof of failure, but the worst real case was not a rise. In the
    July campaign at k=2000 the pool went from N=10,000 to N=200,000 -- twenty
    times more candidates -- and L/k moved by 0.0005. Nothing rose, so a
    rise-only test would have passed it, yet the search was plainly not using the
    candidates: at k=250, where the search WAS converged, the same pool growth
    was worth about 0.03.

    A flat curve is not a mathematical contradiction, so this is a WARNING rather
    than a proof. But over a large N range it is the signature of a search that
    has stopped looking. Returns None, or (N_min, N_max, total_delta).
    """
    if len(rows) < 2:
        return None
    n_min, m_min, _ = rows[0]
    n_max, m_max, _ = rows[-1]
    if n_max < 4 * n_min:
        return None                      # pool did not grow enough to expect much
    total = m_max - m_min                # should be comfortably NEGATIVE
    if total > -tol:
        return (n_min, n_max, total)
    return None


def stage_budget(args):
    N = max(args.k, round(args.k / args.p))
    print(f"=== BUDGET LADDER: p={args.p:g}, k={args.k}, N={N}, "
          f"{args.instances} instances, restarts={args.restarts} ===")
    print("Sweeping SA iterations per candidate point. If L/k is still falling at")
    print("the top of the ladder, the search is not converged and f(p) is an artefact.")
    print()
    print(f"{'iters/cand':>11} {'sa_iters/restart':>17} {'L/k':>9} {'stderr':>8} {'HK bound':>9} {'delta':>9} {'sec':>7}")
    points = []
    prev = None
    for ratio in args.ratios:
        out = os.path.join(args.out_dir, f"budget_p{args.p:g}_k{args.k}_r{ratio}.json")
        res = run_case(args.exe, N, args.p, args.instances, args.threads, args.restarts,
                       args.sa_iters, ratio, args.lkh_path, out, args.timeout, extra_args=args.solver_arg)
        if res is None:
            print(f"{ratio:>11} {'':>17}   (run failed or timed out)")
            continue
        mean, stderr, hk, secs = res
        eff = args.sa_iters + ratio * N
        delta = "" if prev is None else f"{mean - prev:+.4f}"
        print(f"{ratio:>11} {eff:>17,} {mean:>9.4f} {stderr:>8.4f} {hk:>9.4f} {delta:>9} {secs:>7.0f}")
        points.append((eff, mean, stderr))
        prev = mean
    print()
    status, msg = plateau_verdict(points, args.tol, args.instances)
    print(f"  VERDICT [{status}]: {msg}")
    if status == "not_converged":
        print("  -> do NOT run a campaign at this budget. The f(p) it produces is biased upward,")
        print("     and biased MORE at small p (where N is larger), which fabricates a p-trend.")
    elif status == "inconclusive":
        print("  -> the budget may well be fine; you simply cannot see it at this instance count.")
        print("     Add instances (not budget) and re-run.")
    return 0 if status == "converged" else 1


def stage_monotone(args):
    print(f"=== MONOTONICITY ACCEPTANCE TEST: k={args.k}, {args.instances} instances ===")
    print("At fixed k, a larger pool N can only help -- the optimum is NON-INCREASING in N.")
    print("A measured RISE is mathematically impossible, so it proves the search is not")
    print("using the extra candidates. (Necessary condition, not sufficient.)")
    print()
    print(f"{'p':>7} {'N':>9} {'iters/cand':>11} {'L/k':>9} {'stderr':>8} {'HK bound':>9}")
    rows = []
    for p in sorted(args.ps, reverse=True):   # descending p == ascending N
        N = max(args.k, round(args.k / p))
        out = os.path.join(args.out_dir, f"mono_k{args.k}_p{p:g}.json")
        res = run_case(args.exe, N, p, args.instances, args.threads, args.restarts,
                       args.sa_iters, args.sa_iters_per_n, args.lkh_path, out, args.timeout, extra_args=args.solver_arg)
        if res is None:
            print(f"{p:>7g} {N:>9}   (run failed or timed out)")
            continue
        mean, stderr, hk, _ = res
        eff = args.sa_iters + args.sa_iters_per_n * N
        print(f"{p:>7g} {N:>9} {eff / N:>11.1f} {mean:>9.4f} {stderr:>8.4f} {hk:>9.4f}")
        rows.append((N, mean, stderr))
    print()
    viol = monotonicity_violations(rows, args.z)
    warn = no_gain_warning(rows, args.tol)
    rc = 0
    if viol:
        rc = 1
        print(f"  FAIL -- {len(viol)} monotonicity violation(s). L/k rose with N, which the true")
        print("         optimum cannot do. The search is not using the larger pool:")
        for n_prev, n_curr, delta, sigmas in viol:
            print(f"    N {n_prev} -> {n_curr}: L/k ROSE by {delta:+.4f} ({sigmas:.1f} sigma)")
    if warn is not None:
        rc = 1
        n_min, n_max, total = warn
        print(f"  FAIL -- no gain from a {n_max / n_min:.0f}x larger candidate pool "
              f"(N {n_min} -> {n_max}): L/k moved {total:+.4f}.")
        print("         Not a mathematical contradiction, but the signature of a search that has")
        print("         stopped looking: where the search IS converged, a pool this much bigger")
        print("         is worth far more than this.")
    if rc == 0:
        print("  PASS -- L/k is non-increasing in N, and a larger pool measurably helps.")
        print("  (Necessary, not sufficient: still confirm with the budget ladder.)")
    else:
        print("  -> raise --sa-iters-per-n until this passes, then re-check the budget ladder.")
    return rc



def allocation_verdict(results, tol, z=2.0):
    """Judge allocation-invariance at fixed total budget.

    `results` is [(restarts, values_list), ...] with per-instance values on
    IDENTICAL instances (same seed), ordered by increasing restart count. All
    comparisons are PAIRED against the first configuration, which cancels
    instance-to-instance variance -- typically a 3-5x tighter test than
    comparing the means.

    Rationale: a budget plateau alone can be an attractor floor of the search
    dynamics rather than the optimum. The July 2026 study demonstrated this:
    4 restarts x ratio 120 was flat ("converged"), yet 8 x 60 -- the SAME total
    budget, redistributed -- dug 0.002 below it and was still descending. A
    plateau deserves trust only if it is ALSO invariant under reallocating the
    budget across restarts.
    """
    if len(results) < 2:
        return ("insufficient", "need at least 2 allocations to compare")
    base_r, base_vals = results[0]
    rows = []
    for r, vals in results[1:]:
        n = min(len(vals), len(base_vals))
        if n < 2:
            return ("insufficient", "need at least 2 paired instances")
        diffs = [vals[i] - base_vals[i] for i in range(n)]
        m = sum(diffs) / n
        var = sum((d - m) ** 2 for d in diffs) / (n - 1)
        se = (var / n) ** 0.5
        rows.append((r, m, se))
    worst = max(rows, key=lambda t: abs(t[1]))
    sig_down = [t for t in rows if t[1] < 0 and se_sig(t, z)]
    sig_up = [t for t in rows if t[1] > 0 and se_sig(t, z)]
    if sig_down:
        r, m, se = min(sig_down, key=lambda t: t[1])
        return ("restart_limited",
                f"{r} restarts beats {base_r} by {m:+.4f} +/- {se:.4f} at the SAME budget -- "
                f"the search gains from diversification; a budget plateau at {base_r} restarts "
                f"is an attractor floor, not the optimum")
    if sig_up:
        r, m, se = max(sig_up, key=lambda t: t[1])
        return ("depth_limited",
                f"{r} restarts is WORSE than {base_r} by {m:+.4f} +/- {se:.4f} at the same budget -- "
                f"anneals are too short at high restart counts; the sweet spot is at or below {base_r}")
    if abs(worst[1]) < tol:
        return ("invariant",
                f"all allocations agree within {tol:.4f} (worst paired diff {worst[1]:+.4f} "
                f"+/- {worst[2]:.4f}) -- the level is robust to how the budget is spent")
    return ("inconclusive",
            f"worst paired diff {worst[1]:+.4f} +/- {worst[2]:.4f} is neither significant nor "
            f"within tolerance {tol:.4f}; add instances")


def se_sig(t, z):
    return t[2] > 0 and abs(t[1]) / t[2] >= z


def stage_allocation(args):
    N = max(args.k, round(args.k / args.p))
    B = args.budget
    print(f"=== ALLOCATION SCAN: p={args.p:g}, k={args.k}, N={N}, total budget B={B} iters/cand, "
          f"{args.instances} instances ===")
    print("Same total SA work, split across different restart counts, on IDENTICAL instances.")
    print("A trustworthy plateau must be invariant under this split; the July 2026 study's")
    print("4x120 'plateau' failed exactly this test against 8x60.")
    print()
    print(f"{'restarts':>9} {'ratio':>7} {'L/k':>9} {'stderr':>8} {'HK bound':>9} {'sec':>7}")
    results = []
    for r in args.splits:
        if B % r != 0:
            print(f"{r:>9}   (skipped: budget {B} not divisible by {r})")
            continue
        ratio = B // r
        out = os.path.join(args.out_dir, f"alloc_B{B}_r{r}x{ratio}.json")
        res = run_case(args.exe, N, args.p, args.instances, args.threads, r,
                       args.sa_iters, ratio, args.lkh_path, out, args.timeout, extra_args=args.solver_arg)
        if res is None:
            print(f"{r:>9} {ratio:>7}   (run failed or timed out)")
            continue
        mean, stderr, hk, secs = res
        with open(out) as f:
            vals = json.load(f)["summary_rows"][0]["values"]
        results.append((r, vals))
        print(f"{r:>9} {ratio:>7} {mean:>9.4f} {stderr:>8.4f} {hk:>9.4f} {secs:>7.0f}")
    print()
    if len(results) >= 2:
        print("  paired diffs vs the first allocation (negative = better):")
        base_r, base_vals = results[0]
        for r, vals in results[1:]:
            n = min(len(vals), len(base_vals))
            diffs = [vals[i] - base_vals[i] for i in range(n)]
            m = sum(diffs) / n
            var = sum((d - m) ** 2 for d in diffs) / max(1, n - 1)
            se = (var / n) ** 0.5
            print(f"    {r:>3} restarts vs {base_r}: {m:+.5f} +/- {se:.5f}")
        print()
    status, msg = allocation_verdict(results, args.tol, args.z)
    print(f"  VERDICT [{status}]: {msg}")
    return 0 if status == "invariant" else 1


def self_test():
    # plateau_verdict
    assert plateau_verdict([(1, 0.70, 0.001), (2, 0.69, 0.001)], 0.002)[0] == "insufficient"
    still = [(1, 0.700, 0.001), (2, 0.680, 0.001), (3, 0.660, 0.001)]
    status, msg = plateau_verdict(still, 0.002)
    assert status == "not_converged" and "still improving" in msg, msg
    flat = [(1, 0.700, 0.001), (2, 0.681, 0.001), (3, 0.6805, 0.001)]
    assert plateau_verdict(flat, 0.002)[0] == "converged"
    # A ladder whose steps are buried in noise must be reported as INCONCLUSIVE,
    # not as a convergence failure: the fix is instances, not budget. Calling
    # this "not converged" would send the user off buying compute that cannot help.
    noisy = [(1, 0.700, 0.010), (2, 0.699, 0.010), (3, 0.6985, 0.010)]
    status, msg = plateau_verdict(noisy, 0.002, instances=4)
    assert status == "inconclusive", (status, msg)
    assert "instances would resolve it" in msg, msg
    # and it must size the instance count correctly: sigma/tol = 7.07, so the
    # instance count must grow by ~50x (variance scales as 1/n).
    need = int(re.search(r"about (\d+) instances", msg).group(1))
    assert 195 <= need <= 205, msg

    # monotonicity_violations
    good = [(1000, 0.70, 0.001), (5000, 0.68, 0.001), (25000, 0.67, 0.001)]
    assert monotonicity_violations(good) == [], "monotone decreasing data must pass"
    bad = [(1000, 0.700, 0.001), (5000, 0.710, 0.001)]
    v = monotonicity_violations(bad)
    assert len(v) == 1 and v[0][2] > 0, v
    noise = [(1000, 0.7000, 0.010), (5000, 0.7005, 0.010)]
    assert monotonicity_violations(noise) == [], "a rise inside noise must not be flagged"

    # --- the harness must catch the failure that actually happened ---
    # These are the real numbers from the July 2026 campaign, which reported
    # f(0+) = 0.709 and alpha = 3.01 -- both artefacts of an unconverged search.
    #
    # k=1000: L/k rose from N=20,000 to N=50,000. The true optimum cannot rise.
    k1000 = [(5000, 0.6927, 0.0019), (10000, 0.6941, 0.0025), (20000, 0.6872, 0.0019),
             (50000, 0.6918, 0.0019), (100000, 0.6916, 0.0020)]
    assert monotonicity_violations(k1000, z=1.5), "must flag the k=1000 rise (1.7 sigma)"
    assert not monotonicity_violations(k1000, z=3.0), "a 1.7 sigma rise is not a 3-sigma event"

    # k=2000: nothing ROSE, so a rise-only test passes it -- yet a 20x bigger pool
    # bought 0.0005. This is the failure mode the no-gain check exists to catch.
    k2000 = [(10000, 0.7035, 0.0017), (20000, 0.7053, 0.0017), (40000, 0.7028, 0.0013),
             (100000, 0.7032, 0.0013), (200000, 0.7030, 0.0013)]
    assert not monotonicity_violations(k2000, z=2.0), "no rise is significant at 2 sigma here"
    warn = no_gain_warning(k2000, tol=0.002)
    assert warn is not None, "must flag that a 20x larger pool bought nothing"
    assert abs(warn[2]) < 0.002, warn

    # k=250, where the search WAS converged: a bigger pool genuinely helped, so it
    # must pass both checks and not be flagged as a false positive.
    k250 = [(1250, 0.6808, 0.0035), (2500, 0.6633, 0.0041), (5000, 0.6500, 0.0031),
            (12500, 0.6558, 0.0031), (25000, 0.6504, 0.0037)]
    assert not monotonicity_violations(k250, z=2.0), "k=250 has no significant rise"
    assert no_gain_warning(k250, tol=0.002) is None, "k=250 gained from the larger pool"

    # allocation_verdict: paired, on identical instances
    import random as _rnd
    _rnd.seed(4)
    base = [0.63 + _rnd.gauss(0, 0.004) for _ in range(12)]
    same = [(4, base), (8, [v + _rnd.gauss(0, 0.0005) for v in base]),
            (16, [v + _rnd.gauss(0, 0.0005) for v in base])]
    st, _ = allocation_verdict(same, tol=0.002)
    assert st == "invariant", st
    deeper = [(4, base), (8, [v - 0.004 + _rnd.gauss(0, 0.0005) for v in base])]
    st, msg = allocation_verdict(deeper, tol=0.002)
    assert st == "restart_limited", (st, msg)
    shallow = [(4, base), (32, [v + 0.006 + _rnd.gauss(0, 0.0005) for v in base])]
    st, msg = allocation_verdict(shallow, tol=0.002)
    assert st == "depth_limited", (st, msg)
    assert allocation_verdict([(4, base)], 0.002)[0] == "insufficient"

    print("convergence_study self-test: OK")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--exe", help="path to aldous_tsp(.exe)")
    ap.add_argument("--stage", choices=["budget", "monotone", "allocation", "all"], default="all")
    ap.add_argument("--p", type=float, default=0.01, help="p for the budget ladder")
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--ps", type=float, nargs="+", default=[0.2, 0.1, 0.05, 0.02, 0.01],
                    help="p values for the monotonicity test (k is held fixed)")
    ap.add_argument("--ratios", type=int, nargs="+", default=[1, 3, 10, 30, 60],
                    help="SA iterations per candidate point to sweep")
    ap.add_argument("--budget", type=int, default=960,
                    help="allocation stage: total SA iterations per candidate (restarts x ratio)")
    ap.add_argument("--splits", type=int, nargs="+", default=[4, 8, 16, 32],
                    help="allocation stage: restart counts to split the budget across")
    ap.add_argument("--solver-arg", action="append", default=[],
                    help="extra flag passed to the solver verbatim (repeatable), "
                         "e.g. --solver-arg=--kick-restarts --solver-arg=28")
    ap.add_argument("--instances", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--sa-iters", type=int, default=0, help="flat SA term (the per-N term does the work)")
    ap.add_argument("--sa-iters-per-n", type=int, default=20, help="budget for the monotonicity stage")
    ap.add_argument("--lkh-path", default=None, help="optional; the oracle is post-hoc and does not steer the search")
    ap.add_argument("--out-dir", default="convergence")
    ap.add_argument("--tol", type=float, default=0.002,
                    help="plateau tolerance; keep it well below the effect you mean to measure")
    ap.add_argument("--z", type=float, default=2.0, help="sigma threshold for flagging a monotonicity violation")
    ap.add_argument("--timeout", type=int, default=0)
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.exe:
        ap.error("--exe is required (or use --self-test)")
    args.timeout = args.timeout if args.timeout > 0 else None
    os.makedirs(args.out_dir, exist_ok=True)

    rc = 0
    if args.stage == "allocation":
        return stage_allocation(args)
    if args.stage in ("budget", "all"):
        rc |= stage_budget(args)
        print()
    if args.stage in ("monotone", "all"):
        rc |= stage_monotone(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
