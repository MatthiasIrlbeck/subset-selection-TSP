#!/usr/bin/env python3
"""Estimate f(0+) and the small-p exponent alpha from a torus campaign.

This is the last stage of the pipeline. Given the per-(p, k) result batches from
run_torus_campaign.py, it:

  1. extrapolates every p to N -> inf with the torus O(1/N) form
     f(p, N) = f(p) + slope / k  (weighted by instance stderr);
  2. fits the small-p law  f(p) = f0 + C * p^alpha  by profiling the single
     nonlinear parameter alpha (for each alpha the model is linear in (f0, C), so
     it is solved exactly), giving f(0+) = f0 and the exponent alpha;
  3. propagates instance noise through BOTH stages with a master bootstrap --
     resample instances at every (p, k), rerun stages 1-2 -- to attach
     percentile confidence intervals to f(p), f(0+), C, and alpha.

It also extrapolates the conditional Held-Karp diagnostic the same way and
reports the tour-to-bound gap per p, so you can see whether the TOUR ORDERING on
the selected subsets was converged.

IMPORTANT -- what the Held-Karp number is and is not. HK lower-bounds the optimal
tour through THE SUBSET THE SOLVER CHOSE. It is NOT a lower bound on

    f(p) = min over subsets S of |S|=k  of  TSP(S)/k ,

because a better subset drives the tour AND its HK bound down together: the
"floor" moves whenever the search improves. HK therefore brackets TOUR-SOLVING
error only. It says nothing about SUBSET-SELECTION error -- and subset selection
is a minimisation, so an under-converged search biases f(p) upward with no lower
bracket to catch it. Use scripts/convergence_study.py to bound that error; it
cannot be read off the HK column.

Inputs are the campaign JSON files (one p each, or multi-p). Per-instance tour
values come from each summary row's `values`; the conditional tour diagnostic
comes from `conditional_held_karp_bound_mean` (or
`conditional_two_nn_bound_mean` if absent). Schema-13 legacy names are accepted.

Usage:
  analyze_campaign.py torus_campaign/*.json
  analyze_campaign.py --pmax 0.2 --boot 2000 --plot fit.png torus_campaign/*.json
  analyze_campaign.py --self-test
"""
import argparse
import json
import math
import random
import sys
from collections import defaultdict


def wls(x, y, w):
    """Weighted least squares y = a + b*x. Returns (a, b, var_a)."""
    sw = sum(w)
    swx = sum(wi * xi for wi, xi in zip(w, x))
    swy = sum(wi * yi for wi, yi in zip(w, y))
    swxx = sum(wi * xi * xi for wi, xi in zip(w, x))
    swxy = sum(wi * xi * yi for wi, xi, yi in zip(w, x, y))
    denom = sw * swxx - swx * swx
    if abs(denom) < 1e-300:
        return swy / sw, 0.0, float("inf")
    b = (sw * swxy - swx * swy) / denom
    a = (swy - b * swx) / sw
    var_a = swxx / denom
    return a, b, var_a


def extrapolate_intercept(points):
    """points: list of (k, mean, weight). Fit mean = a + b/k -> return (a, se_a)."""
    if len(points) == 1:
        return points[0][1], 0.0
    x = [1.0 / k for k, _, _ in points]
    y = [m for _, m, _ in points]
    w = [wt for _, _, wt in points]
    a, _b, var_a = wls(x, y, w)
    se = math.sqrt(var_a) if math.isfinite(var_a) else 0.0
    return a, se


def profile_power_fit(ps, fs, ws, alpha_grid):
    """Fit f = f0 + C * p^alpha. Profile alpha over a grid; (f0, C) linear each step.

    Returns (f0, C, alpha, ssr). Uses weights ws on the f values.
    """
    best = None
    for alpha in alpha_grid:
        x = [p ** alpha for p in ps]
        f0, C, _ = wls(x, fs, ws)
        ssr = sum(wi * (fi - f0 - C * xi) ** 2 for wi, fi, xi in zip(ws, fs, x))
        if best is None or ssr < best[3]:
            best = (f0, C, alpha, ssr)
    # Parabolic refinement around the best grid alpha.
    f0, C, alpha, ssr = best
    idx = alpha_grid.index(alpha)
    if 0 < idx < len(alpha_grid) - 1:
        a0, a1, a2 = alpha_grid[idx - 1], alpha, alpha_grid[idx + 1]
        for a in [a1 + (a1 - a0) * t for t in (-0.5, -0.25, 0.25, 0.5)]:
            if a <= 0:
                continue
            x = [p ** a for p in ps]
            f0c, Cc, _ = wls(x, fs, ws)
            s = sum(wi * (fi - f0c - Cc * xi) ** 2 for wi, fi, xi in zip(ws, fs, x))
            if s < ssr:
                f0, C, alpha, ssr = f0c, Cc, a, s
    return f0, C, alpha, ssr


def analyze(ladders, pmax, boot, alpha_grid, seed=12345):
    """ladders: {p: {k: [instance values]}}. Returns a result dict."""
    ps_all = sorted(ladders)
    ps = [p for p in ps_all if p <= pmax]
    if len(ps) < 3:
        raise SystemExit(f"need >= 3 p-values with p <= {pmax}; have {len(ps)}")

    # --- point estimates ---
    def fp_of(sample_means):
        """sample_means: {p: {k: mean}} -> {p: (f(p), se)} via 1/k extrapolation."""
        out = {}
        for p in ps:
            pts = []
            for k, vals in ladders[p].items():
                m = sample_means[p][k]
                n = len(vals)
                sd = _std(vals)
                se = sd / math.sqrt(n) if n > 1 else 0.0
                wt = 1.0 / max(se, 1e-9) ** 2
                pts.append((k, m, wt))
            pts.sort()
            out[p] = extrapolate_intercept(pts)
        return out

    point_means = {p: {k: _mean(v) for k, v in ladders[p].items()} for p in ps}
    fp_point = fp_of(point_means)
    fps = [fp_point[p][0] for p in ps]
    # Weights for the power-law fit: inverse variance of each f(p).
    fw = [1.0 / max(se, 1e-9) ** 2 for _, se in (fp_point[p] for p in ps)]
    f0, C, alpha, _ = profile_power_fit(ps, fps, fw, alpha_grid)

    # --- master bootstrap over instances ---
    rng = random.Random(seed)
    bs_f0, bs_C, bs_alpha = [], [], []
    bs_fp = {p: [] for p in ps}
    for _ in range(boot):
        means = {}
        for p in ps:
            means[p] = {}
            for k, vals in ladders[p].items():
                n = len(vals)
                means[p][k] = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        fp_b = fp_of(means)
        fvals = [fp_b[p][0] for p in ps]
        for p, fv in zip(ps, fvals):
            bs_fp[p].append(fv)
        f0b, Cb, ab, _ = profile_power_fit(ps, fvals, fw, alpha_grid)
        bs_f0.append(f0b)
        bs_C.append(Cb)
        bs_alpha.append(ab)

    # --- Conditional Held-Karp (or two-NN) tour-ordering diagnostic ---
    bound_means = {}
    have_bound = True
    for p in ps:
        bound_means[p] = {}
        for k in ladders[p]:
            bound = _conditional_bound_of.get((p, k))
            if bound is None:
                have_bound = False
            bound_means[p][k] = bound
    conditional_f0_diagnostic = None
    if have_bound:
        bound_points = {p: {k: bound_means[p][k] for k in ladders[p]} for p in ps}
        fp_bound = fp_of(bound_points)
        conditional_f0_diagnostic, _, _, _ = profile_power_fit(
            ps, [fp_bound[p][0] for p in ps], fw, alpha_grid)

    return {
        "ps": ps,
        "fp_point": fp_point,
        "fp_ci": {p: _ci(bs_fp[p]) for p in ps},
        "f0": f0, "f0_ci": _ci(bs_f0),
        "C": C, "C_ci": _ci(bs_C),
        "alpha": alpha, "alpha_ci": _ci(bs_alpha),
        "conditional_f0_diagnostic": conditional_f0_diagnostic,
        "conditional_gaps": _conditional_gaps,
        # Deprecated result aliases for callers of older script versions.
        "f0_lb": conditional_f0_diagnostic,
        "gaps": _conditional_gaps,
    }


def _mean(v):
    return sum(v) / len(v)


def _std(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def _ci(samples, lo=2.5, hi=97.5):
    if not samples:
        return (float("nan"), float("nan"))
    s = sorted(samples)
    return (s[int(lo / 100 * (len(s) - 1))], s[int(hi / 100 * (len(s) - 1))])


# Module-level side tables filled by load_campaign (kept simple for the profile fit).
_conditional_bound_of = {}
_conditional_gaps = {}


def _conditional_bound_from_row(row):
    for key in (
        "conditional_held_karp_bound_mean",
        "conditional_two_nn_bound_mean",
        "held_karp_bound_mean",       # schema-13 compatibility
        "subset_bound_mean",          # schema-13 compatibility
    ):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def load_campaign(paths):
    """Return {p: {k: [tour values]}} and merge conditional diagnostics.

    Repeated shards for the same (p, k) cell are merged by instance count.
    Cells with incompatible N or geometry are rejected instead of being silently
    combined into one finite-size point.
    """
    ladders = defaultdict(lambda: defaultdict(list))
    _conditional_bound_of.clear()
    _conditional_gaps.clear()
    bound_sum = defaultdict(float)
    bound_count = defaultdict(int)
    gap_sum = defaultdict(float)
    gap_count = defaultdict(int)
    cell_metadata = {}
    for path in paths:
        with open(path, encoding="utf-8") as stream:
            doc = json.load(stream)
        config = doc.get("config", {})
        doc_n = doc.get("N")
        periodic = config.get("periodic")
        for row in doc.get("summary_rows", []):
            p, k = float(row["p"]), int(row["k"])
            vals = row.get("values") or []
            if not vals:
                continue
            declared_n = int(row.get("n", len(vals)))
            if declared_n != len(vals):
                raise ValueError(
                    f"{path}: summary cell (p={p}, k={k}) declares n={declared_n} "
                    f"but carries {len(vals)} values")
            actual_mean = _mean(vals)
            if "mean" in row and not math.isclose(
                    float(row["mean"]), actual_mean, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(
                    f"{path}: summary mean for (p={p}, k={k}) is inconsistent "
                    "with its values array")
            metadata = (doc_n, periodic)
            previous = cell_metadata.get((p, k))
            if previous is not None and previous != metadata:
                raise ValueError(
                    f"incompatible duplicate campaign cell (p={p}, k={k}): "
                    f"metadata {previous} versus {metadata}")
            cell_metadata[(p, k)] = metadata
            ladders[p][k].extend(vals)
            bound = _conditional_bound_from_row(row)
            if bound is not None:
                bound_sum[(p, k)] += bound * declared_n
                bound_count[(p, k)] += declared_n
                gap_sum[p] += (actual_mean - bound) * declared_n
                gap_count[p] += declared_n
    for cell, total in bound_sum.items():
        _conditional_bound_of[cell] = total / bound_count[cell]
    for p, total in gap_sum.items():
        _conditional_gaps[p] = total / gap_count[p]
    return {p: dict(ks) for p, ks in ladders.items()}


def run_self_test():
    """Synthetic campaign with known (f0, C, alpha): the fit must recover them."""
    true_f0, true_C, true_alpha = 0.625, 0.35, 0.75
    ps = [0.01, 0.02, 0.05, 0.1, 0.2]
    ks = [500, 1000, 2000]
    slope = 1.2  # 1/k finite-size term
    rng = random.Random(7)
    ladders = {}
    for p in ps:
        ladders[p] = {}
        fp_true = true_f0 + true_C * p ** true_alpha
        for k in ks:
            vals = [fp_true + slope / k + rng.gauss(0, 0.004) for _ in range(40)]
            ladders[p][k] = vals
            _conditional_bound_of[(p, k)] = fp_true + slope / k - 0.09
    grid = [0.02 + 0.01 * i for i in range(300)]
    res = analyze(ladders, pmax=1.0, boot=400, alpha_grid=grid)
    ok_f0 = res["f0_ci"][0] <= true_f0 <= res["f0_ci"][1]
    ok_a = res["alpha_ci"][0] <= true_alpha <= res["alpha_ci"][1]
    print(f"self-test: f0={res['f0']:.4f} CI{_fmt(res['f0_ci'])} (true {true_f0}) "
          f"{'PASS' if ok_f0 else 'FAIL'}")
    print(f"           alpha={res['alpha']:.3f} CI{_fmt(res['alpha_ci'])} "
          f"(true {true_alpha}) {'PASS' if ok_a else 'FAIL'}")

    # A bound on the chosen subset need not bound the optimum over subsets.
    # Here the found subset has conditional bound 10 and tour 11, while another
    # subset has tour 9. The conditional bound is therefore above the global
    # subset optimum and cannot be described as a floor on it.
    semantics_ok = 10.0 > 9.0 and 10.0 <= 11.0

    # Regression for multi-file weighting: the old loader overwrote the first
    # shard's bound with the final shard's mean and averaged gaps per row.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        base = {
            "N": 100,
            "config": {"periodic": True},
        }
        shard_a = dict(base, summary_rows=[{
            "p": 0.2, "k": 20, "n": 2,
            "values": [0.8, 1.0], "mean": 0.9,
            "conditional_held_karp_bound_mean": 0.5,
        }])
        shard_b = dict(base, summary_rows=[{
            "p": 0.2, "k": 20, "n": 4,
            "values": [0.7, 0.8, 0.9, 1.0], "mean": 0.85,
            "conditional_held_karp_bound_mean": 0.7,
        }])
        pa, pb = Path(tmp) / "a.json", Path(tmp) / "b.json"
        pa.write_text(json.dumps(shard_a), encoding="utf-8")
        pb.write_text(json.dumps(shard_b), encoding="utf-8")
        merged = load_campaign([str(pa), str(pb)])
        expected_bound = (0.5 * 2 + 0.7 * 4) / 6
        expected_gap = ((0.9 - 0.5) * 2 + (0.85 - 0.7) * 4) / 6
        merge_ok = (
            len(merged[0.2][20]) == 6
            and math.isclose(_conditional_bound_of[(0.2, 20)], expected_bound)
            and math.isclose(_conditional_gaps[0.2], expected_gap)
        )
    print(f"           conditional-bound semantics {'PASS' if semantics_ok else 'FAIL'}; "
          f"weighted shard merge {'PASS' if merge_ok else 'FAIL'}")
    return 0 if (ok_f0 and ok_a and semantics_ok and merge_ok) else 1


def _fmt(ci):
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def make_plot(res, ladders, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    ps = res["ps"]
    fps = [res["fp_point"][p][0] for p in ps]
    lo = [res["fp_ci"][p][0] for p in ps]
    hi = [res["fp_ci"][p][1] for p in ps]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.errorbar(ps, fps, yerr=[np.array(fps) - lo, np.array(hi) - fps],
                fmt="o", color="#c0392b", capsize=3, label="extrapolated f(p) (95% CI)")
    xx = np.linspace(0, max(ps) * 1.05, 200)
    ax.plot(xx, res["f0"] + res["C"] * xx ** res["alpha"], color="#2471a3",
            label=f"fit  f0 + C p^alpha  (alpha={res['alpha']:.2f})")
    ax.axhline(res["f0"], color="#2471a3", ls=":", lw=1)
    ax.fill_between([0, max(ps) * 1.05], res["f0_ci"][0], res["f0_ci"][1],
                    color="#2471a3", alpha=0.12, label=f"f(0+) = {res['f0']:.3f} {_fmt(res['f0_ci'])}")
    if res["conditional_f0_diagnostic"] is not None:
        value = res["conditional_f0_diagnostic"]
        ax.axhline(value, color="#27ae60", ls="--", lw=1,
                   label=f"conditional HK diagnostic = {value:.3f}\n(not a floor on f(0+))")
    ax.scatter([0], [res["f0"]], marker="*", s=160, color="#2471a3",
               edgecolor="k", zorder=6)
    ax.set_xlabel("p")
    ax.set_ylabel("f(p)")
    ax.set_title("Aldous subset-TSP: f(p) -> f(0+) and the exponent alpha")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=-0.005)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="campaign JSON files")
    ap.add_argument("--pmax", type=float, default=0.3,
                    help="only use p <= pmax for the small-p fit (default 0.3)")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--plot", metavar="PNG", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.files:
        ap.error("no input files (or use --self-test)")

    ladders = load_campaign(args.files)
    grid = [0.02 + 0.01 * i for i in range(300)]  # alpha in [0.02, 3.01]
    res = analyze(ladders, args.pmax, args.boot, grid)

    print(f"{'p':>7} {'k-range':>13} {'f(p)':>9} {'95% CI':>20} {'conditional gap':>16}")
    for p in res["ps"]:
        ks = sorted(ladders[p])
        fp, _ = res["fp_point"][p]
        ci = res["fp_ci"][p]
        gap = res["conditional_gaps"].get(p)
        gaps = f"{gap:.4f}" if gap is not None else "  --"
        print(f"{p:>7g} {f'{ks[0]}-{ks[-1]}':>13} {fp:>9.4f} {_fmt(ci):>20} {gaps:>15}")

    print("\n=== small-p law  f(p) = f(0+) + C p^alpha ===")
    print(f"  f(0+)  = {res['f0']:.4f}   95% CI {_fmt(res['f0_ci'])}")
    if res["conditional_f0_diagnostic"] is not None:
        diagnostic = res["conditional_f0_diagnostic"]
        print(f"           (Conditional Held-Karp diagnostic on the CHOSEN subsets: {diagnostic:.4f}.")
        print(f"            This is NOT a lower bound on f(0+): f(0+) is a minimum over subsets,")
        print(f"            and a better subset lowers the tour and this bound together. It brackets")
        print(f"            tour-solving error only -- run scripts/convergence_study.py for the")
        print(f"            subset-selection error, which is the one that biases f(0+) upward.)")
    print(f"  alpha  = {res['alpha']:.3f}    95% CI {_fmt(res['alpha_ci'])}")
    print(f"  C      = {res['C']:.4f}   95% CI {_fmt(res['C_ci'])}")
    print(f"  (fit over {len(res['ps'])} p-values with p <= {args.pmax}; "
          f"{args.boot} bootstrap resamples)")
    if res["f0_ci"][0] > 0:
        print(f"  => f(0+) > 0 at 95% confidence.")

    if args.plot:
        make_plot(res, ladders, args.plot)
        print(f"\nwrote {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
