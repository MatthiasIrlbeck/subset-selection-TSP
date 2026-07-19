#!/usr/bin/env python3
"""Endpoint estimation for the search's basin-value distribution.

WHY THIS EXISTS
---------------
The paired allocation scans at k=2000 established that, once the anneal is deep
enough to converge (~60 iterations/candidate), the solver is a pure independent
multistart: doubling the restart count buys a CONSTANT ~0.0025, forever. That is
best-of-m behaviour, and best-of-m has no plateau -- so no budget ladder will
ever certify a value. Reading off min(restarts) and calling it converged is
exactly the mistake the whole convergence-criteria saga has been about.

The quantity we actually want is the LEFT ENDPOINT theta of the distribution the
restarts are drawing from. That is an extreme-value problem, not a compute
problem: fit the lower tail, extrapolate m -> infinity, and put an honest
confidence interval on it.

WHAT IT DOES
------------
Per instance, per p:

  1. Strata. Restart values are a MIXTURE across seed kinds (a uniformly random
     subset at small p never contracts and lands at L/k ~ 1.5; a dense seed lands
     at ~0.65) and, when --second-sweep is enabled, across warm-start directions.
     A tail fit on the pooled sample is a fit to a contaminated distribution, so
     draws are filterable by --kinds, --sweeps, and --roles. The default role
     is independent diagnostic only; continuation and production-raced draws are
     excluded from endpoint fitting. Excluded mass is reported, not hidden.

  2. Domain of attraction. The moment estimator (Dekkers-Einmahl-de Haan, applied
     to the reflected sample) gives the extreme-value index gamma. A finite left
     endpoint requires gamma < 0. If the estimate is not comfortably negative,
     the endpoint is not identified and the script says so instead of printing a
     number.

  3. Endpoint. Two estimators, deliberately:
       - moment/Weissman-type:  theta = x_(1) - a(m/k) * ... (closed form below)
       - 3-parameter Weibull MLE on the m lowest draws (profile likelihood in
         theta).
     They fail differently. Agreement is evidence; disagreement is a warning.

  4. Instability, displayed. Endpoint estimators converge slowly and are famously
     sensitive to the number of tail draws used. The script sweeps that choice
     and prints the whole path. A stable plateau across the sweep is the only
     thing that licenses quoting a number.

USAGE
    python scripts/restart_evt.py results.json [more.json ...] --p 0.01
    python scripts/restart_evt.py "alloc_*/**/*.json" --p 0.01 --bootstrap 400

Requires the run to have been made with --include-instance-rows (0.9.5+), which
records restart_values and restart_kinds. Current outputs also record
restart_sweeps and restart_roles; older outputs are interpreted as primary-only
with an unknown mixed role.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict

from restart_metadata import (DEFAULT_KINDS, DEFAULT_ROLES, DEFAULT_SWEEPS,
                              KIND_NAMES, ROLE_NAMES, SWEEP_NAMES)

import numpy as np
from scipy.optimize import minimize_scalar


def load_runs(patterns):
    files = []
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        files.extend(hits if hits else ([pat] if pat.endswith(".json") else []))
    docs = []
    for f in sorted(set(files)):
        try:
            with open(f) as fh:
                docs.append((f, json.load(fh)))
        except (OSError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"skipping {f}: {exc}\n")
    return docs


def collect(docs, p, kinds, sweeps, roles, max_value):
    """-> {instance_index: np.array of retained draws}, plus census."""
    per_instance = defaultdict(list)
    kind_census = defaultdict(int)
    sweep_census = defaultdict(int)
    role_census = defaultdict(int)
    dropped_by_value = 0
    for fname, doc in docs:
        rows = doc.get("instance_rows")
        if not rows:
            sys.stderr.write(
                f"{fname}: no instance_rows -- rerun with --include-instance-rows\n")
            continue
        for row in rows:
            for pv in row.get("p_results", []):
                if abs(pv.get("p", -1) - p) > 1e-12:
                    continue
                vals = pv.get("restart_values")
                kds = pv.get("restart_kinds")
                sws = pv.get("restart_sweeps")
                rls = pv.get("restart_roles")
                if vals is None or vals == []:
                    continue
                if not isinstance(vals, list):
                    sys.stderr.write(
                        f"{fname}: instance {row.get('index')} p={p} has a non-array "
                        "restart_values field; skipping\n")
                    continue

                expected = len(vals)
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in vals
                ):
                    sys.stderr.write(
                        f"{fname}: instance {row.get('index')} p={p} has "
                        "non-finite or non-numeric restart values; skipping\n")
                    continue

                malformed = False
                for field_name in (
                    "restart_kinds", "restart_sweeps", "restart_roles",
                    "restart_variants", "restart_promotion_stages",
                    "restart_sa_iterations", "restart_centroids_x",
                    "restart_centroids_y", "restart_radii",
                ):
                    column = pv.get(field_name)
                    if column is not None and (
                        not isinstance(column, list) or len(column) != expected
                    ):
                        actual = len(column) if isinstance(column, list) else "non-array"
                        sys.stderr.write(
                            f"{fname}: instance {row.get('index')} p={p} has "
                            f"{expected} restart_values but {actual} {field_name}; skipping\n")
                        malformed = True
                        break
                if malformed:
                    continue

                executed = pv.get("executed_restarts")
                if executed is not None and (
                    not isinstance(executed, int) or executed != expected
                ):
                    sys.stderr.write(
                        f"{fname}: instance {row.get('index')} p={p} reports "
                        f"executed_restarts={executed!r} but serializes {expected} "
                        "restart values; skipping\n")
                    continue
                best = pv.get("best_restart")
                if best is not None and (
                    not isinstance(best, int) or best < 0 or best >= expected
                ):
                    sys.stderr.write(
                        f"{fname}: instance {row.get('index')} p={p} has "
                        f"best_restart={best!r} outside [0, {expected}); skipping\n")
                    continue

                if kds is None:
                    kds = [-1] * expected
                if sws is None:
                    sws = [0] * expected
                if rls is None:
                    rls = [-1] * expected
                if any(isinstance(code, bool) or not isinstance(code, int) for code in kds):
                    sys.stderr.write(
                        f"{fname}: instance {row.get('index')} p={p} has "
                        "non-integer restart kinds; skipping\n")
                    continue
                if any(isinstance(code, bool) or not isinstance(code, int) for code in sws):
                    sys.stderr.write(
                        f"{fname}: instance {row.get('index')} p={p} has "
                        "non-integer restart sweeps; skipping\n")
                    continue
                if any(isinstance(code, bool) or not isinstance(code, int) for code in rls):
                    sys.stderr.write(
                        f"{fname}: instance {row.get('index')} p={p} has "
                        "non-integer restart roles; skipping\n")
                    continue
                key = (fname, row.get("index"))
                for v, kd, sw, role in zip(vals, kds, sws, rls):
                    kind_census[kd] += 1
                    sweep_census[sw] += 1
                    role_census[role] += 1
                    if kd not in kinds and kd != -1:
                        continue
                    if sw not in sweeps:
                        continue
                    if role not in roles and role != -1:
                        continue
                    if max_value is not None and v > max_value:
                        dropped_by_value += 1
                        continue
                    per_instance[key].append(v)
    return ({k: np.sort(np.asarray(v)) for k, v in per_instance.items()},
            kind_census, sweep_census, role_census, dropped_by_value)


def moment_index(x_sorted, m):
    """Extreme-value index gamma of the LEFT tail, via the moment estimator.

    Reflect (y = -x) so the left tail becomes a right tail, then apply
    Dekkers-Einmahl-de Haan to the m+1 largest y. gamma < 0 <=> finite endpoint.
    """
    y = np.sort(-x_sorted)[::-1]          # descending
    if m + 1 >= len(y) or y[m] <= 0:
        # shift into the positive half-line; the estimator needs y > 0
        shift = 1.0 - y.min() if y.min() <= 0 else 0.0
        y = y + shift
    if m + 1 >= len(y):
        return float("nan")
    logs = np.log(y[:m]) - np.log(y[m])
    m1 = logs.mean()
    m2 = (logs ** 2).mean()
    if m1 <= 0 or m2 <= 0:
        return float("nan")
    return m1 + 1.0 - 0.5 / (1.0 - m1 * m1 / m2)


def endpoint_moment(x_sorted, m):
    """Closed-form endpoint from the moment estimator (Dekkers-de Haan)."""
    gamma = moment_index(x_sorted, m)
    if not np.isfinite(gamma) or gamma >= 0:
        return float("nan"), gamma
    y = np.sort(-x_sorted)[::-1]
    shift = 1.0 - y.min() if y.min() <= 0 else 0.0
    ys = y + shift
    logs = np.log(ys[:m]) - np.log(ys[m])
    m1 = logs.mean()
    # de Haan & Ferreira (4.3.3): a(n/m) = X_{n-m,n} * M1 * (1 - gamma_minus),
    # and the endpoint is anchored at the (m+1)-th order statistic, NOT the
    # extreme one: x* = X_{n-m,n} - a/gamma. Anchoring at the max instead (and
    # adding a/gamma with gamma < 0) yields an "endpoint" INSIDE the sample --
    # which is how this bug announced itself: theta came out above the best
    # observed draw, an impossibility for a left endpoint.
    gamma_minus = gamma - m1
    a = ys[m] * m1 * (1.0 - gamma_minus)
    y_endpoint = ys[m] - a / gamma  # gamma < 0 => finite right endpoint of y
    theta = -(y_endpoint - shift)
    if not np.isfinite(theta) or theta >= x_sorted[0]:
        return float("nan"), gamma  # not a valid left endpoint; refuse to report
    return float(theta), gamma


def endpoint_weibull_mle(x_sorted, m):
    """3-parameter Weibull MLE on the m lowest draws, profiled over theta.

    Model: (X - theta) ~ Weibull(shape c, scale s) on the lower tail, theta the
    left endpoint. For fixed theta the (c, s) MLE reduces to a 1-D root; we just
    profile the concentrated log-likelihood in theta directly.
    """
    x = x_sorted[:m]
    x_min = float(x[0])

    def nll(theta):
        z = x - theta
        if np.any(z <= 0):
            return 1e18
        lz = np.log(z)
        # concentrated MLE for c given theta (Weibull, no censoring)
        def eq(c):
            zc = z ** c
            return (zc * lz).sum() / zc.sum() - 1.0 / c - lz.mean()
        lo, hi = 1e-3, 50.0
        flo, fhi = eq(lo), eq(hi)
        if not np.isfinite(flo) or not np.isfinite(fhi) or flo * fhi > 0:
            return 1e18
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if eq(lo) * eq(mid) <= 0:
                hi = mid
            else:
                lo = mid
        c = 0.5 * (lo + hi)
        s = ((z ** c).mean()) ** (1.0 / c)
        n = len(z)
        ll = (n * math.log(c) - n * c * math.log(s)
              + (c - 1.0) * lz.sum() - ((z / s) ** c).sum())
        return -ll

    span = max(x_sorted[-1] - x_min, 1e-6)
    res = minimize_scalar(nll, bounds=(x_min - 2.0 * span, x_min - 1e-9),
                          method="bounded", options={"xatol": 1e-8})
    if not res.success or res.fun >= 1e17:
        return float("nan")
    return float(res.x)


def sweep(x, frac_grid):
    """Endpoint estimates across tail-fraction choices. Instability shows here."""
    out = []
    n = len(x)
    for frac in frac_grid:
        m = max(5, int(round(frac * n)))
        if m + 2 >= n:
            continue
        theta_m, gamma = endpoint_moment(x, m)
        theta_w = endpoint_weibull_mle(x, m)
        out.append((m, gamma, theta_m, theta_w))
    return out


def bootstrap_ci(x, m, reps, rng):
    est = []
    for _ in range(reps):
        xb = np.sort(rng.choice(x, size=len(x), replace=True))
        t, _ = endpoint_moment(xb, m)
        if np.isfinite(t):
            est.append(t)
    if len(est) < max(20, reps // 10):
        return float("nan"), float("nan")
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="result JSON files or globs")
    ap.add_argument("--p", type=float, required=True, help="which p to analyse")
    ap.add_argument("--kinds", type=int, nargs="+", default=list(DEFAULT_KINDS),
                    choices=sorted(KIND_NAMES),
                    help="restart kinds to keep (default: every defined kind)")
    ap.add_argument("--sweeps", type=int, nargs="+", default=list(DEFAULT_SWEEPS),
                    choices=sorted(SWEEP_NAMES),
                    help="restart sweeps to keep: 0=primary, 1=secondary "
                         "(default: both; old files are primary-only)")
    ap.add_argument("--roles", type=int, nargs="+", default=list(DEFAULT_ROLES),
                    choices=sorted(ROLE_NAMES),
                    help="restart roles to keep (default: independent diagnostic only; "
                         "legacy rows without roles are retained as unknown)")
    ap.add_argument("--max-value", type=float, default=0.9,
                    help="drop draws above this L/k as non-contracted "
                         "(default: 0.9; set 0 to disable)")
    ap.add_argument("--tail-frac", type=float, default=0.5,
                    help="fraction of the lowest draws used for the headline fit")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="bootstrap reps for the CI (0 to skip)")
    ap.add_argument("--seed", type=int, default=20260714)
    args = ap.parse_args()

    docs = load_runs(args.results)
    if not docs:
        sys.exit("no result files matched")
    max_value = None if args.max_value == 0 else args.max_value
    per_instance, kind_census, sweep_census, role_census, dropped = collect(
        docs, args.p, set(args.kinds), set(args.sweeps), set(args.roles), max_value)
    if not per_instance:
        sys.exit(f"no restart_values found at p={args.p} "
                 f"(was the run made with --include-instance-rows?)")

    total = sum(kind_census.values())
    print(f"=== restart-value census, p={args.p} ===")
    for kd in sorted(kind_census):
        name = KIND_NAMES.get(kd, f"kind{kd}")
        retained = kd in args.kinds or kd == -1
        print(f"  kind {name:24s} {kind_census[kd]:5d} draws  "
              f"({100*kind_census[kd]/total:4.1f}%)"
              + ("" if retained else "   [EXCLUDED by --kinds]"))
    for sw in sorted(sweep_census):
        name = SWEEP_NAMES.get(sw, f"sweep{sw}")
        print(f"  sweep {name:21s} {sweep_census[sw]:5d} draws  "
              f"({100*sweep_census[sw]/total:4.1f}%)"
              + ("" if sw in args.sweeps else "   [EXCLUDED by --sweeps]"))
    for role in sorted(role_census):
        name = ROLE_NAMES.get(role, "legacy-unknown" if role == -1 else f"role{role}")
        retained = role in args.roles or role == -1
        print(f"  role {name:22s} {role_census[role]:5d} draws  "
              f"({100*role_census[role]/total:4.1f}%)"
              + ("" if retained else "   [EXCLUDED by --roles]"))
    if dropped:
        print(f"  dropped as non-contracted (L/k > {max_value}): {dropped} draws")
    print()

    rng = np.random.default_rng(args.seed)
    frac_grid = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    headline, best_of_m = [], []

    for (fname, idx), x in sorted(per_instance.items()):
        if len(x) < 12:
            print(f"instance {idx}: only {len(x)} usable draws -- "
                  f"endpoint estimation needs more restarts; skipping")
            continue
        rows = sweep(x, frac_grid)
        print(f"instance {idx}  ({len(x)} draws, best {x[0]:.4f}, median {np.median(x):.4f})")
        print(f"    {'m':>4s} {'gamma':>8s} {'theta_moment':>13s} {'theta_weibull':>14s}")
        for m, gamma, tm, tw in rows:
            flag = "" if (np.isfinite(gamma) and gamma < -0.02) else "   <- endpoint not identified"
            print(f"    {m:4d} {gamma:8.3f} {tm:13.4f} {tw:14.4f}{flag}")
        m_head = max(5, int(round(args.tail_frac * len(x))))
        # The moment estimator is the headline: on samples this size the
        # 3-parameter Weibull MLE is badly behaved (it chases the lowest draw and
        # swings by 0.2 between adjacent m). Weibull stays as a cross-check --
        # when the two disagree, neither is trustworthy.
        theta, gamma = endpoint_moment(x, m_head)
        theta_w = endpoint_weibull_mle(x, m_head)
        if np.isfinite(theta):
            lo, hi = ((float("nan"), float("nan")) if args.bootstrap == 0
                      else bootstrap_ci(x, m_head, args.bootstrap, rng))
            ci = "" if not np.isfinite(lo) else f"   95% CI [{lo:.4f}, {hi:.4f}]"
            print(f"    headline (moment, m={m_head}): theta = {theta:.4f}{ci}")
            if np.isfinite(theta_w) and abs(theta_w - theta) > 0.02:
                print(f"    WARNING: Weibull MLE says {theta_w:.4f} -- the two estimators "
                      f"disagree by {abs(theta_w - theta):.3f}; do not quote either")
            headline.append(theta)
            best_of_m.append(float(x[0]))
        else:
            print("    headline: endpoint NOT identified at this m")
        if len(x) < 100:
            print(f"    NOTE: {len(x)} draws is far too few for endpoint estimation. "
                  f"Tail fits want O(100) draws per instance;")
            print(f"          at converged depth (~60 iters/candidate) restarts are cheap, "
                  f"so spend budget on RESTARTS, not instances.")
        print()

    if headline:
        th = np.asarray(headline)
        bo = np.asarray(best_of_m)
        se = th.std(ddof=1) / math.sqrt(len(th)) if len(th) > 1 else float("nan")
        print("=== across instances ===")
        print(f"  best-of-m (what the solver reports): {bo.mean():.4f}")
        print(f"  endpoint estimate theta:             {th.mean():.4f} +/- {se:.4f}")
        print(f"  extrapolation gap:                   {bo.mean() - th.mean():+.4f}")
        print()
        print("  Read this as a lower bound on how much search is still on the table,")
        print("  NOT as an estimate of f(p): theta is the endpoint of the SEARCH's basin")
        print("  distribution, which equals the true optimum only if the sampler can reach")
        print("  it at all. Quote it only if the sweep above is flat in m and gamma < 0")
        print("  across the board.")


if __name__ == "__main__":
    main()
