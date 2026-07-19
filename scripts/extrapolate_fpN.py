#!/usr/bin/env python3
"""Finite-size extrapolation of the subset-TSP curve f(p) from a k-ladder.

Aldous's problem asks for f(p) = lim_{N->inf} L(k)/k with k = pN. A single run
at finite N only gives f(p, N), which still carries a finite-size correction.
The *form* of that correction depends on the domain boundary:

  * open square  : f(p, N) = f(p) + a / sqrt(N) + ...   (O(1/sqrt N) surface term)
  * flat torus   : f(p, N) = f(p) + c / N       + ...   (O(1/N), surface term gone)

(Percus & Martin, PRL 76, 1188, 1996; the limit f(p) is the same either way,
Jaillet 1993.) So on the torus we extrapolate f(p, N) linearly in 1/N -- or,
equivalently at fixed p, in 1/k -- and read off the intercept as f(p).

This tool ingests one or more results JSON files forming a ladder (same p,
varying N), groups by p, and for each p:

  * fits f(p, N) = intercept + slope / k    (the torus O(1/N) form)
  * optionally applies the Percus-Martin nearest-neighbor rescaling, dividing
    the raw value by (1 + 1/(8k)) before fitting -- this removes the finite-k
    inflation of the local length scale and flattens the residual correction
  * for contrast also fits the 1/sqrt(k) (boundary) form and reports the
    residual RMS of both, so the data can confirm which form applies
  * if conditional bound output is present, also extrapolates that diagnostic.
    It bounds the tour through the subset selected by the heuristic, not the
    optimum over all size-k subsets, so it is not a bracket on f(p)

An optional second stage fits the small-p exponent alpha in
f(p) - f0 ~ p^alpha once several extrapolated f(p) are available.

Usage:
  extrapolate_fpN.py run1.json run2.json ...            # extrapolate each p
  extrapolate_fpN.py --rescale run*.json                # with NN rescaling
  extrapolate_fpN.py --alpha --f0 0.712 run*.json       # also fit alpha
  extrapolate_fpN.py --self-test                         # validate the fitter
  extrapolate_fpN.py --plot out.png run*.json            # write a figure
"""
import argparse
import json
import math
import sys
from collections import defaultdict


def weighted_linear_fit(x, y, w):
    """Weighted least squares y = a + b*x. Returns (a, b, se_a, se_b, rms).

    Weights w are 1/variance. se_a, se_b are standard errors of the parameters.
    """
    n = len(x)
    if n < 2:
        raise ValueError("need at least two points to fit")
    sw = sum(w)
    swx = sum(wi * xi for wi, xi in zip(w, x))
    swy = sum(wi * yi for wi, yi in zip(w, y))
    swxx = sum(wi * xi * xi for wi, xi in zip(w, x))
    swxy = sum(wi * xi * yi for wi, xi, yi in zip(w, x, y))
    denom = sw * swxx - swx * swx
    if abs(denom) < 1e-300:
        raise ValueError("degenerate design (all x equal?)")
    b = (sw * swxy - swx * swy) / denom
    a = (swy - b * swx) / sw
    # Parameter variances from the weighted normal equations.
    var_a = swxx / denom
    var_b = sw / denom
    # Weighted residual RMS (goodness of the functional form).
    resid = [yi - (a + b * xi) for xi, yi in zip(x, y)]
    if n > 2:
        chi2 = sum(wi * ri * ri for wi, ri in zip(w, resid))
        scale = chi2 / (n - 2)
    else:
        scale = 1.0
    # Inflate parameter SEs by sqrt(reduced chi^2) so they reflect scatter
    # beyond the reported per-point errors (conservative).
    se_a = math.sqrt(var_a * max(scale, 1.0))
    se_b = math.sqrt(var_b * max(scale, 1.0))
    rms = math.sqrt(sum(r * r for r in resid) / n)
    return a, b, se_a, se_b, rms


def collect_ladder(paths):
    """Read JSON files; return {p_key: {'p':p, 'points':[(N,k,mean,stderr,sub,cvm,cvse)]}}."""
    by_p = defaultdict(lambda: {"p": None, "points": []})
    for path in paths:
        with open(path) as fh:
            doc = json.load(fh)
        N = doc["N"]
        for row in doc.get("summary_rows", []):
            key = row["key"]
            by_p[key]["p"] = row["p"]
            # Prefer the tight conditional Held-Karp diagnostic when present,
            # else fall back to conditional two-NN. Legacy schema-13 aliases
            # are accepted for old result files.
            lb = row.get("conditional_held_karp_bound_mean")
            if lb is None:
                lb = row.get("conditional_two_nn_bound_mean")
            if lb is None:
                lb = row.get("held_karp_bound_mean")
            if lb is None:
                lb = row.get("subset_bound_mean")
            cvm = row.get("cv_mean")
            cvse = row.get("cv_stderr")
            by_p[key]["points"].append(
                (N, row["k"], row["mean"], row.get("stderr", 0.0), lb, cvm, cvse)
            )
    for entry in by_p.values():
        entry["points"].sort()
    return by_p


def extrapolate_series(points, rescale, use_cv):
    """Extrapolate one p's ladder. Returns a dict of results, or None if <2 pts."""
    pts = [pt for pt in points if pt[1] > 0]
    if len(pts) < 2:
        return None

    def value_of(pt):
        # Prefer the control-variate-corrected mean when requested and present.
        N, k, mean, stderr, sub, cvm, cvse = pt
        if use_cv and cvm is not None:
            return cvm, (cvse if cvse else stderr)
        return mean, stderr

    ks = [pt[1] for pt in pts]
    vals, ses = zip(*(value_of(pt) for pt in pts))
    vals = list(vals)
    ses = list(ses)

    def rescaled(v, k):
        return v / (1.0 + 1.0 / (8.0 * k)) if rescale else v

    yv = [rescaled(v, k) for v, k in zip(vals, ks)]
    # Weights: 1/se^2, with a floor so a zero-stderr point does not dominate.
    floor = max((s for s in ses if s > 0), default=1e-6) * 0.25
    w = [1.0 / max(s, floor) ** 2 for s in ses]

    x_inv = [1.0 / k for k in ks]
    x_isqrt = [1.0 / math.sqrt(k) for k in ks]
    a1, b1, sea1, _, rms1 = weighted_linear_fit(x_inv, yv, w)      # torus 1/k
    a2, b2, sea2, _, rms2 = weighted_linear_fit(x_isqrt, yv, w)    # boundary 1/sqrt(k)

    out = {
        "p": None,
        "n_points": len(pts),
        "k_min": min(ks),
        "k_max": max(ks),
        "f_inv": a1, "f_inv_se": sea1, "slope_inv": b1, "rms_inv": rms1,
        "f_isqrt": a2, "f_isqrt_se": sea2, "rms_isqrt": rms2,
        "rescaled": rescale,
        "used_cv": use_cv,
    }

    # Extrapolate the fixed-selected-subset tour diagnostic if available for
    # every ladder point. This is deliberately not named as a global bound.
    subs = [pt[4] for pt in pts]
    if all(s is not None for s in subs):
        ysub = [rescaled(s, k) for s, k in zip(subs, ks)]
        # bounds carry no separate stderr here; fit unweighted.
        wa = [1.0] * len(ks)
        lb, _, lbse, _, _ = weighted_linear_fit(x_inv, ysub, wa)
        out["conditional_bound_inv"] = lb
        out["conditional_bound_inv_se"] = lbse
        # Backward-compatible aliases for callers of older script versions.
        out["lb_inv"] = lb
        out["lb_inv_se"] = lbse
    return out


def fit_alpha(fp_by_p, f0):
    """Fit f(p) - f0 ~ C p^alpha by linear regression of log(f-f0) on log(p)."""
    xs, ys = [], []
    for p, f in sorted(fp_by_p.items()):
        d = f - f0
        if p > 0 and d > 0:
            xs.append(math.log(p))
            ys.append(math.log(d))
    if len(xs) < 2:
        return None
    a, b, sea, seb, rms = weighted_linear_fit(xs, ys, [1.0] * len(xs))
    return {"alpha": b, "alpha_se": seb, "logC": a, "rms": rms, "n": len(xs)}


def run_self_test():
    """Verify the fitter recovers a known intercept from synthetic 1/k data."""
    import random
    random.seed(1)
    true_f, true_slope = 0.7124, 1.35
    ks = [50, 100, 200, 400, 800, 1600]
    ok = True
    for trial in range(200):
        pts = []
        for k in ks:
            noise = random.gauss(0.0, 0.002)
            v = true_f + true_slope / k + noise
            pts.append((k, k, v, 0.002, None, None, None))
        res = extrapolate_series(pts, rescale=False, use_cv=False)
        if abs(res["f_inv"] - true_f) > 4 * res["f_inv_se"]:
            ok = False
    # On clean synthetic 1/k data the 1/k form must fit better than 1/sqrt(k).
    clean = [(k, k, true_f + true_slope / k, 0.001, None, None, None) for k in ks]
    r = extrapolate_series(clean, rescale=False, use_cv=False)
    form_ok = r["rms_inv"] < r["rms_isqrt"]
    print(f"self-test: intercept recovery {'PASS' if ok else 'FAIL'}; "
          f"1/k beats 1/sqrt(k) on 1/k data {'PASS' if form_ok else 'FAIL'} "
          f"(rms_inv={r['rms_inv']:.2e} vs rms_isqrt={r['rms_isqrt']:.2e})")
    return 0 if (ok and form_ok) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="results JSON files forming a k-ladder")
    ap.add_argument("--rescale", action="store_true",
                    help="apply the (1 + 1/(8k)) nearest-neighbor rescaling before fitting")
    ap.add_argument("--cv", action="store_true",
                    help="use the control-variate-corrected mean when present")
    ap.add_argument("--alpha", action="store_true",
                    help="also fit f(p) - f0 ~ p^alpha across the extrapolated f(p)")
    ap.add_argument("--f0", type=float, default=None,
                    help="f(0+) to subtract for the alpha fit")
    ap.add_argument("--plot", metavar="PNG", default=None, help="write a figure")
    ap.add_argument("--self-test", action="store_true", help="validate the fitter and exit")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.files:
        ap.error("no input files (or use --self-test)")

    by_p = collect_ladder(args.files)
    print(f"{'p':>7} {'pts':>4} {'k-range':>13} | {'f(p) [1/N]':>12} {'+/-':>8} "
          f"| {'f(p) [1/sqrtN]':>14} | {'rms 1/N':>9} {'rms 1/sqrtN':>11} | {'cond. bnd':>9}")
    fp_by_p = {}
    for key in sorted(by_p, key=lambda k: by_p[k]["p"]):
        entry = by_p[key]
        res = extrapolate_series(entry["points"], args.rescale, args.cv)
        if res is None:
            continue
        p = entry["p"]
        fp_by_p[p] = res["f_inv"]
        lb = (f"{res['conditional_bound_inv']:.4f}"
              if "conditional_bound_inv" in res else "   --")
        krange = f"{res['k_min']}-{res['k_max']}"
        print(f"{p:>7.4g} {res['n_points']:>4} {krange:>13} | "
              f"{res['f_inv']:>12.4f} {res['f_inv_se']:>8.4f} | "
              f"{res['f_isqrt']:>14.4f} | {res['rms_inv']:>9.2e} {res['rms_isqrt']:>11.2e} | {lb:>9}")

    note = []
    if args.rescale:
        note.append("nearest-neighbor rescaled")
    if args.cv:
        note.append("control-variate mean")
    if note:
        print("  [" + ", ".join(note) + "]")
    print("  The correct form on the torus is 1/N; a smaller rms under 1/N than "
          "1/sqrt(N) confirms the boundary term is absent. Cond. bnd is the "
          "extrapolated lower bound on tours through the selected subsets; it "
          "is not a floor on f(p), which also minimizes over subsets.")

    if args.alpha:
        if args.f0 is None:
            print("\n--alpha requires --f0 (an estimate of f(0+))", file=sys.stderr)
            return 2
        fit = fit_alpha(fp_by_p, args.f0)
        if fit is None:
            print("\nnot enough p-values (or f(p) <= f0) to fit alpha", file=sys.stderr)
        else:
            print(f"\nalpha fit (f(p) - {args.f0} ~ C p^alpha over {fit['n']} points): "
                  f"alpha = {fit['alpha']:.3f} +/- {fit['alpha_se']:.3f}, "
                  f"C = {math.exp(fit['logC']):.3f}, log-log rms = {fit['rms']:.3f}")

    if args.plot:
        make_plot(by_p, args.rescale, args.cv, args.plot)
        print(f"\nwrote {args.plot}")
    return 0


def make_plot(by_p, rescale, use_cv, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    keys = sorted(by_p, key=lambda k: by_p[k]["p"])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    cmap = plt.get_cmap("viridis")
    for i, key in enumerate(keys):
        entry = by_p[key]
        res = extrapolate_series(entry["points"], rescale, use_cv)
        if res is None:
            continue
        pts = [pt for pt in entry["points"] if pt[1] > 0]
        ks = np.array([pt[1] for pt in pts], float)
        vals = np.array([(pt[5] if use_cv and pt[5] is not None else pt[2]) for pt in pts])
        if rescale:
            vals = vals / (1.0 + 1.0 / (8.0 * ks))
        color = cmap(i / max(len(keys) - 1, 1))
        ax.scatter(1.0 / ks, vals, color=color, s=45, zorder=5)
        xx = np.linspace(0, (1.0 / ks).max() * 1.05, 50)
        ax.plot(xx, res["f_inv"] + res["slope_inv"] * xx, color=color, lw=1.5,
                label=f"p={entry['p']:.4g}: f={res['f_inv']:.4f}")
        ax.scatter([0], [res["f_inv"]], color=color, marker="*", s=140, zorder=6,
                   edgecolor="k", linewidth=0.5)
    ax.set_xlabel("1 / k")
    ax.set_ylabel("f(p, N)" + ("  (rescaled)" if rescale else ""))
    ax.set_title("Torus finite-size extrapolation: f(p,N) = f(p) + slope/k")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    ax.set_xlim(left=-0.0005)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    sys.exit(main())
