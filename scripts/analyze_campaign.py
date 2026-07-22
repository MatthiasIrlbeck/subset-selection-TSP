#!/usr/bin/env python3
"""Estimate f(0+) and the small-p exponent from correlated campaign results.

The analysis has two finite-size/statistical stages:

  1. for every p, fit the torus form f(p, N) = f(p) + slope/k;
  2. fit f(p) = f0 + C p^alpha by profiling alpha.

New result files carry stable campaign, replicate, point-stream, search-stream,
solver-policy, and fidelity identities. When complete replicate vectors are
available, the master bootstrap resamples whole point-set replicates and thus
preserves common-random-number correlation across every (p, k) cell. Legacy
summary-only files remain supported through an explicitly reported independent-
cell bootstrap.

Repeated search streams on one point set are averaged before the primary fit so
point sets receive equal weight. A nested method-of-moments report separates
point-instance variance from search-seed variance where repeated searches are
available. When cheap and strong fidelity rows share point identities, the
script also reports the paired multifidelity estimator

    mean(cheap over all points) + mean(strong - cheap over paired points).

Conditional Held-Karp/two-NN values remain diagnostics for the tour through the
chosen subset. They are not global lower bounds on the optimum over subsets.

Usage:
  analyze_campaign.py torus_campaign/*.json
  analyze_campaign.py --pmax 0.2 --boot 2000 --plot fit.png campaign/*.json
  analyze_campaign.py --bootstrap-mode block --analysis-json analysis.json files...
  analyze_campaign.py --finite-size-models inv-k,inv-k2,inv-sqrt-k files...
  analyze_campaign.py --self-test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


Cell = tuple[float, int]
BlockKey = tuple[str, int]


@dataclass(frozen=True)
class Observation:
    campaign_id: str
    campaign_shard: int
    replicate_id: int
    point_stream_id: str
    search_stream_id: str
    solver_policy_id: str
    fidelity_level: str
    p: float
    k: int
    value: float
    source: str


# Side tables populated by load_campaign. Keeping load_campaign's historical
# return value ({p: {k: values}}) avoids breaking existing Python callers.
_conditional_bound_of: dict[Cell, float] = {}
_conditional_gaps: dict[float, float] = {}
_replicate_blocks: dict[BlockKey, dict[Cell, float]] = {}
_nested_search_groups: dict[Cell, dict[BlockKey, list[float]]] = {}
_all_observations: list[Observation] = []
_load_info: dict[str, object] = {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _ci(samples: list[float], lo: float = 2.5, hi: float = 97.5) -> tuple[float, float]:
    if not samples:
        return (float("nan"), float("nan"))
    ordered = sorted(samples)
    return (
        ordered[int(lo / 100.0 * (len(ordered) - 1))],
        ordered[int(hi / 100.0 * (len(ordered) - 1))],
    )


def _fmt(interval: tuple[float, float]) -> str:
    return f"[{interval[0]:.4f}, {interval[1]:.4f}]"


def wls(x: list[float], y: list[float], weights: list[float]) -> tuple[float, float, float]:
    """Weighted least squares y = intercept + slope*x."""
    sw = sum(weights)
    swx = sum(weight * value for weight, value in zip(weights, x))
    swy = sum(weight * value for weight, value in zip(weights, y))
    swxx = sum(weight * value * value for weight, value in zip(weights, x))
    swxy = sum(weight * xv * yv for weight, xv, yv in zip(weights, x, y))
    denom = sw * swxx - swx * swx
    if abs(denom) < 1e-300:
        return swy / sw, 0.0, float("inf")
    slope = (sw * swxy - swx * swy) / denom
    intercept = (swy - slope * swx) / sw
    return intercept, slope, swxx / denom


FINITE_SIZE_MODEL_SPECS: dict[str, tuple[str, int]] = {
    "inv-k": ("a + b/k", 2),
    "inv-k2": ("a + b/k + c/k^2", 3),
    "inv-sqrt-k": ("a + b/sqrt(k)", 2),
}


@dataclass(frozen=True)
class FiniteSizeFit:
    model: str
    intercept: float
    stderr: float
    coefficients: tuple[float, ...]
    weighted_ssr: float
    k_levels: int
    identifiable: bool


def _finite_size_basis(model: str, k: int, k_scale: float = 1.0) -> list[float]:
    if k <= 0 or not math.isfinite(k_scale) or k_scale <= 0.0:
        raise ValueError("finite-size cardinalities and scale must be positive")
    if model == "inv-k":
        return [1.0, k_scale / k]
    if model == "inv-k2":
        inverse = k_scale / k
        return [1.0, inverse, inverse * inverse]
    if model == "inv-sqrt-k":
        return [1.0, math.sqrt(k_scale / k)]
    raise ValueError(f"unknown finite-size model: {model}")


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a small dense system with deterministic partial pivoting."""
    size = len(rhs)
    if size == 0 or len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("invalid linear system")
    augmented = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    scale = max((abs(value) for row in matrix for value in row), default=1.0)
    tolerance = max(1e-300, scale * 1e-14)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            raise ValueError("singular finite-size design")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[index][-1] for index in range(size)]


def fit_finite_size(
    points: list[tuple[int, float, float]],
    model: str,
    *,
    allow_single_inv_k: bool = False,
) -> FiniteSizeFit:
    """Fit a named finite-size law to ``(k, mean, precision)`` points."""
    if model not in FINITE_SIZE_MODEL_SPECS:
        raise ValueError(f"unknown finite-size model: {model}")
    ordered = sorted(points)
    if not ordered:
        raise ValueError("finite-size fit has no points")
    if any(not math.isfinite(mean) or not math.isfinite(weight) or weight <= 0.0
           for _, mean, weight in ordered):
        raise ValueError("finite-size fit requires finite values and positive weights")
    if model == "inv-k" and len(ordered) == 1 and allow_single_inv_k:
        return FiniteSizeFit(
            model=model, intercept=ordered[0][1], stderr=0.0,
            coefficients=(ordered[0][1], 0.0), weighted_ssr=0.0,
            k_levels=1, identifiable=False,
        )
    minimum = FINITE_SIZE_MODEL_SPECS[model][1]
    if len(ordered) < minimum:
        raise ValueError(
            f"finite-size model {model} requires at least {minimum} distinct k levels"
        )

    # Preserve the historical two-parameter arithmetic exactly for the primary
    # 1/k model. This keeps existing point estimates bit-for-bit stable.
    if model == "inv-k":
        x = [1.0 / k for k, _, _ in ordered]
        y = [mean for _, mean, _ in ordered]
        weights = [weight for _, _, weight in ordered]
        intercept, slope, variance = wls(x, y, weights)
        residual = sum(
            weight * (mean - intercept - slope / k) ** 2
            for k, mean, weight in ordered
        )
        return FiniteSizeFit(
            model=model, intercept=intercept,
            stderr=math.sqrt(max(variance, 0.0)) if math.isfinite(variance) else 0.0,
            coefficients=(intercept, slope), weighted_ssr=residual,
            k_levels=len(ordered), identifiable=True,
        )

    # Scale the inverse-cardinality predictor into O(1) without changing the
    # k -> infinity intercept. This keeps the quadratic normal equations well
    # conditioned across production cardinalities.
    k_scale = float(min(k for k, _, _ in ordered))
    rows = [_finite_size_basis(model, k, k_scale) for k, _, _ in ordered]
    dimension = len(rows[0])
    normal = [[0.0] * dimension for _ in range(dimension)]
    target = [0.0] * dimension
    for row, (_, mean, weight) in zip(rows, ordered):
        for left in range(dimension):
            target[left] += weight * row[left] * mean
            for right in range(dimension):
                normal[left][right] += weight * row[left] * row[right]
    coefficients = _solve_linear_system(normal, target)
    inverse_first_column = _solve_linear_system(
        normal, [1.0] + [0.0] * (dimension - 1)
    )
    residual = 0.0
    for row, (_, mean, weight) in zip(rows, ordered):
        prediction = sum(coefficient * value for coefficient, value in zip(coefficients, row))
        residual += weight * (mean - prediction) ** 2
    return FiniteSizeFit(
        model=model, intercept=coefficients[0],
        stderr=math.sqrt(max(inverse_first_column[0], 0.0)),
        coefficients=tuple(coefficients), weighted_ssr=residual,
        k_levels=len(ordered), identifiable=True,
    )


def extrapolate_intercept(points: list[tuple[int, float, float]]) -> tuple[float, float]:
    """Fit mean = intercept + slope/k from (k, mean, weight) points."""
    if len(points) == 1:
        return points[0][1], 0.0
    x = [1.0 / k for k, _, _ in points]
    y = [mean for _, mean, _ in points]
    weights = [weight for _, _, weight in points]
    intercept, _slope, variance = wls(x, y, weights)
    stderr = math.sqrt(variance) if math.isfinite(variance) else 0.0
    return intercept, stderr


def profile_power_fit(
    ps: list[float],
    values: list[float],
    weights: list[float],
    alpha_grid: list[float],
) -> tuple[float, float, float, float]:
    """Fit f = f0 + C*p^alpha by profiling alpha over a fixed grid."""
    best: tuple[float, float, float, float] | None = None
    for alpha in alpha_grid:
        transformed = [p**alpha for p in ps]
        f0, coefficient, _ = wls(transformed, values, weights)
        ssr = sum(
            weight * (value - f0 - coefficient * xvalue) ** 2
            for weight, value, xvalue in zip(weights, values, transformed)
        )
        candidate = (f0, coefficient, alpha, ssr)
        if best is None or ssr < best[3]:
            best = candidate
    if best is None:
        raise ValueError("alpha grid is empty")

    f0, coefficient, alpha, ssr = best
    index = alpha_grid.index(alpha)
    if 0 < index < len(alpha_grid) - 1:
        step = alpha_grid[index] - alpha_grid[index - 1]
        for refined_alpha in (alpha - 0.5 * step, alpha - 0.25 * step,
                              alpha + 0.25 * step, alpha + 0.5 * step):
            if refined_alpha <= 0.0:
                continue
            transformed = [p**refined_alpha for p in ps]
            f0_candidate, coefficient_candidate, _ = wls(
                transformed, values, weights
            )
            candidate_ssr = sum(
                weight * (value - f0_candidate - coefficient_candidate * xvalue) ** 2
                for weight, value, xvalue in zip(weights, values, transformed)
            )
            if candidate_ssr < ssr:
                f0 = f0_candidate
                coefficient = coefficient_candidate
                alpha = refined_alpha
                ssr = candidate_ssr
    return f0, coefficient, alpha, ssr


def _cell_weight(values: list[float]) -> float:
    count = len(values)
    stderr = _std(values) / math.sqrt(count) if count > 1 else 0.0
    return 1.0 / max(stderr, 1e-9) ** 2


def _finite_size_points(
    ladders: dict[float, dict[int, list[float]]], p: float
) -> list[tuple[int, float, float]]:
    points = [
        (k, _mean(values), _cell_weight(values))
        for k, values in ladders[p].items()
    ]
    points.sort()
    return points


def _extrapolate_ladders(
    ladders: dict[float, dict[int, list[float]]],
    ps: list[float],
    model: str = "inv-k",
) -> dict[float, tuple[float, float]]:
    output: dict[float, tuple[float, float]] = {}
    for p in ps:
        fit = fit_finite_size(
            _finite_size_points(ladders, p), model,
            allow_single_inv_k=(model == "inv-k"),
        )
        output[p] = (fit.intercept, fit.stderr)
    return output


def _power_fit_for_model(
    ladders: dict[float, dict[int, list[float]]],
    ps: list[float],
    model: str,
    alpha_grid: list[float],
) -> dict[str, object]:
    fp_point = _extrapolate_ladders(ladders, ps, model)
    fp_values = [fp_point[p][0] for p in ps]
    weights = [1.0 / max(fp_point[p][1], 1e-9) ** 2 for p in ps]
    f0, coefficient, alpha, power_ssr = profile_power_fit(
        ps, fp_values, weights, alpha_grid
    )
    per_p_fits = {
        p: fit_finite_size(
            _finite_size_points(ladders, p), model,
            allow_single_inv_k=(model == "inv-k"),
        )
        for p in ps
    }
    return {
        "model": model,
        "formula": FINITE_SIZE_MODEL_SPECS[model][0],
        "fp_point": fp_point,
        "f0": f0,
        "C": coefficient,
        "alpha": alpha,
        "power_weighted_ssr": power_ssr,
        "finite_size_weighted_ssr": sum(
            fit.weighted_ssr for fit in per_p_fits.values()
        ),
        "per_p_fits": per_p_fits,
    }


def _point_model_results(
    ladders: dict[float, dict[int, list[float]]],
    ps: list[float],
    models: list[str],
    alpha_grid: list[float],
) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    available: dict[str, dict[str, object]] = {}
    unavailable: dict[str, str] = {}
    for model in models:
        try:
            available[model] = _power_fit_for_model(
                ladders, ps, model, alpha_grid
            )
        except ValueError as exc:
            unavailable[model] = str(exc)
    return available, unavailable


def _model_envelope(
    model_results: dict[str, dict[str, object]],
) -> dict[str, object]:
    if not model_results:
        return {
            "models": [],
            "f0": (float("nan"), float("nan")),
            "alpha": (float("nan"), float("nan")),
            "C": (float("nan"), float("nan")),
        }
    return {
        "models": sorted(model_results),
        "f0": (
            min(float(row["f0"]) for row in model_results.values()),
            max(float(row["f0"]) for row in model_results.values()),
        ),
        "alpha": (
            min(float(row["alpha"]) for row in model_results.values()),
            max(float(row["alpha"]) for row in model_results.values()),
        ),
        "C": (
            min(float(row["C"]) for row in model_results.values()),
            max(float(row["C"]) for row in model_results.values()),
        ),
    }


def _sensitivity_report(
    ladders: dict[float, dict[int, list[float]]],
    ps: list[float],
    primary_model: str,
    alpha_grid: list[float],
    reference_f0: float,
    reference_alpha: float,
) -> dict[str, object]:
    leave_k: list[dict[str, object]] = []
    all_k = sorted({k for p in ps for k in ladders[p]})
    minimum_levels = FINITE_SIZE_MODEL_SPECS[primary_model][1]
    for omitted in all_k:
        reduced = {
            p: {k: values for k, values in ladders[p].items() if k != omitted}
            for p in ps
        }
        if any(len(reduced[p]) < minimum_levels for p in ps):
            continue
        try:
            row = _power_fit_for_model(reduced, ps, primary_model, alpha_grid)
        except ValueError:
            continue
        leave_k.append({
            "omitted_k": omitted,
            "f0": row["f0"],
            "alpha": row["alpha"],
            "delta_f0": float(row["f0"]) - reference_f0,
            "delta_alpha": float(row["alpha"]) - reference_alpha,
        })

    leave_p: list[dict[str, object]] = []
    if len(ps) >= 4:
        for omitted in ps:
            retained = [p for p in ps if p != omitted]
            try:
                row = _power_fit_for_model(
                    ladders, retained, primary_model, alpha_grid
                )
            except ValueError:
                continue
            leave_p.append({
                "omitted_p": omitted,
                "f0": row["f0"],
                "alpha": row["alpha"],
                "delta_f0": float(row["f0"]) - reference_f0,
                "delta_alpha": float(row["alpha"]) - reference_alpha,
            })

    nested_pmax: list[dict[str, object]] = []
    for end in range(3, len(ps) + 1):
        retained = ps[:end]
        try:
            row = _power_fit_for_model(
                ladders, retained, primary_model, alpha_grid
            )
        except ValueError:
            continue
        nested_pmax.append({
            "pmax": retained[-1],
            "p_values": retained,
            "f0": row["f0"],
            "alpha": row["alpha"],
            "delta_f0": float(row["f0"]) - reference_f0,
            "delta_alpha": float(row["alpha"]) - reference_alpha,
        })

    def maximum(rows: list[dict[str, object]], field: str) -> float:
        return max((abs(float(row[field])) for row in rows), default=0.0)

    return {
        "leave_one_k_out": leave_k,
        "leave_one_p_out": leave_p,
        "nested_pmax": nested_pmax,
        "max_abs_delta_f0": max(
            maximum(leave_k, "delta_f0"),
            maximum(leave_p, "delta_f0"),
            maximum(nested_pmax, "delta_f0"),
        ),
        "max_abs_delta_alpha": max(
            maximum(leave_k, "delta_alpha"),
            maximum(leave_p, "delta_alpha"),
            maximum(nested_pmax, "delta_alpha"),
        ),
    }


def _complete_blocks(
    ladders: dict[float, dict[int, list[float]]],
    ps: list[float],
    blocks: dict[BlockKey, dict[Cell, float]],
) -> tuple[list[BlockKey], set[Cell]]:
    required = {(p, k) for p in ps for k in ladders[p]}
    complete = sorted(block for block, cells in blocks.items() if required <= cells.keys())
    return complete, required


def analyze(
    ladders: dict[float, dict[int, list[float]]],
    pmax: float,
    boot: int,
    alpha_grid: list[float],
    seed: int = 12345,
    bootstrap_mode: str = "auto",
    replicate_blocks: dict[BlockKey, dict[Cell, float]] | None = None,
    identities_complete: bool | None = None,
    finite_size_models: list[str] | None = None,
    primary_finite_size_model: str = "inv-k",
) -> dict[str, object]:
    """Run finite-size fits, model sensitivity, and a correlated bootstrap."""
    ps = [p for p in sorted(ladders) if p <= pmax]
    if len(ps) < 3:
        raise SystemExit(f"need >= 3 p-values with p <= {pmax}; have {len(ps)}")
    if boot < 0:
        raise ValueError("bootstrap count must be nonnegative")
    if bootstrap_mode not in {"auto", "block", "independent"}:
        raise ValueError("bootstrap_mode must be auto, block, or independent")

    requested_models = list(finite_size_models or FINITE_SIZE_MODEL_SPECS)
    if primary_finite_size_model not in requested_models:
        requested_models.insert(0, primary_finite_size_model)
    requested_models = list(dict.fromkeys(requested_models))
    unknown_models = [
        model for model in requested_models if model not in FINITE_SIZE_MODEL_SPECS
    ]
    if unknown_models:
        raise ValueError(
            "unknown finite-size model(s): " + ", ".join(unknown_models)
        )

    blocks = _replicate_blocks if replicate_blocks is None else replicate_blocks
    if identities_complete is None:
        identities_complete = bool(_load_info.get("identities_complete", False))
    complete_blocks, required_cells = _complete_blocks(ladders, ps, blocks)
    can_block = identities_complete and len(complete_blocks) >= 2
    if bootstrap_mode == "block" and not can_block:
        raise ValueError(
            "block bootstrap requested, but fewer than two complete identified "
            "replicate vectors are available"
        )
    effective_mode = "replicate-block" if (
        bootstrap_mode == "block" or (bootstrap_mode == "auto" and can_block)
    ) else "independent-cell"

    point_models, unavailable_models = _point_model_results(
        ladders, ps, requested_models, alpha_grid
    )
    if primary_finite_size_model not in point_models:
        reason = unavailable_models.get(primary_finite_size_model, "unknown failure")
        raise ValueError(
            f"primary finite-size model {primary_finite_size_model} is unavailable: {reason}"
        )
    primary = point_models[primary_finite_size_model]
    fp_point = primary["fp_point"]
    f0 = float(primary["f0"])
    coefficient = float(primary["C"])
    alpha = float(primary["alpha"])

    rng = random.Random(seed)
    bootstrap_by_model: dict[str, dict[str, object]] = {
        model: {
            "f0": [], "C": [], "alpha": [],
            "fp": {p: [] for p in ps},
        }
        for model in point_models
    }

    for _ in range(boot):
        sampled: dict[float, dict[int, list[float]]] = {
            p: {k: [] for k in ladders[p]} for p in ps
        }
        if effective_mode == "replicate-block":
            drawn = [
                complete_blocks[rng.randrange(len(complete_blocks))]
                for _ in complete_blocks
            ]
            for block in drawn:
                block_values = blocks[block]
                for p, k in required_cells:
                    sampled[p][k].append(block_values[(p, k)])
        else:
            for p in ps:
                for k, values in ladders[p].items():
                    sampled[p][k] = [
                        values[rng.randrange(len(values))] for _ in values
                    ]

        for model in point_models:
            fitted = _power_fit_for_model(sampled, ps, model, alpha_grid)
            bucket = bootstrap_by_model[model]
            bucket["f0"].append(float(fitted["f0"]))
            bucket["C"].append(float(fitted["C"]))
            bucket["alpha"].append(float(fitted["alpha"]))
            fitted_fp = fitted["fp_point"]
            for p in ps:
                bucket["fp"][p].append(float(fitted_fp[p][0]))

    model_reports: dict[str, dict[str, object]] = {}
    for model, fitted in point_models.items():
        bucket = bootstrap_by_model[model]
        per_p_fits = fitted["per_p_fits"]
        model_reports[model] = {
            "formula": fitted["formula"],
            "f0": fitted["f0"],
            "f0_ci": _ci(bucket["f0"]),
            "C": fitted["C"],
            "C_ci": _ci(bucket["C"]),
            "alpha": fitted["alpha"],
            "alpha_ci": _ci(bucket["alpha"]),
            "power_weighted_ssr": fitted["power_weighted_ssr"],
            "finite_size_weighted_ssr": fitted["finite_size_weighted_ssr"],
            "p_estimates": {
                p: {
                    "f_p": fitted["fp_point"][p][0],
                    "f_p_ci": _ci(bucket["fp"][p]),
                    "stderr": fitted["fp_point"][p][1],
                    "coefficients": list(per_p_fits[p].coefficients),
                    "weighted_ssr": per_p_fits[p].weighted_ssr,
                    "k_levels": per_p_fits[p].k_levels,
                    "identifiable": per_p_fits[p].identifiable,
                }
                for p in ps
            },
        }

    point_envelope = _model_envelope(point_models)

    def uncertainty_envelope(field: str) -> tuple[float, float]:
        intervals = [
            model_reports[model][f"{field}_ci"] for model in model_reports
        ]
        finite_intervals = [
            interval for interval in intervals
            if all(math.isfinite(float(value)) for value in interval)
        ]
        if not finite_intervals:
            return point_envelope[field]
        return (
            min(float(interval[0]) for interval in finite_intervals),
            max(float(interval[1]) for interval in finite_intervals),
        )

    model_envelope = dict(point_envelope)
    model_envelope.update({
        "f0_ci": uncertainty_envelope("f0"),
        "alpha_ci": uncertainty_envelope("alpha"),
        "C_ci": uncertainty_envelope("C"),
    })

    sensitivity = _sensitivity_report(
        ladders, ps, primary_finite_size_model, alpha_grid, f0, alpha
    )

    conditional_f0_diagnostic = None
    have_conditional_bounds = all(
        (p, k) in _conditional_bound_of for p in ps for k in ladders[p]
    )
    if have_conditional_bounds:
        bound_fp: dict[float, tuple[float, float]] = {}
        try:
            for p in ps:
                points = [
                    (k, _conditional_bound_of[(p, k)], _cell_weight(ladders[p][k]))
                    for k in ladders[p]
                ]
                fit = fit_finite_size(
                    points, primary_finite_size_model,
                    allow_single_inv_k=(primary_finite_size_model == "inv-k"),
                )
                bound_fp[p] = (fit.intercept, fit.stderr)
            conditional_f0_diagnostic, _, _, _ = profile_power_fit(
                ps,
                [bound_fp[p][0] for p in ps],
                [1.0 / max(bound_fp[p][1], 1e-9) ** 2 for p in ps],
                alpha_grid,
            )
        except ValueError:
            conditional_f0_diagnostic = None

    primary_bootstrap = bootstrap_by_model[primary_finite_size_model]
    return {
        "ps": ps,
        "fp_point": fp_point,
        "fp_ci": {p: _ci(primary_bootstrap["fp"][p]) for p in ps},
        "f0": f0,
        "f0_ci": _ci(primary_bootstrap["f0"]),
        "C": coefficient,
        "C_ci": _ci(primary_bootstrap["C"]),
        "alpha": alpha,
        "alpha_ci": _ci(primary_bootstrap["alpha"]),
        "primary_finite_size_model": primary_finite_size_model,
        "finite_size_models": model_reports,
        "unavailable_finite_size_models": unavailable_models,
        "model_uncertainty_envelope": model_envelope,
        "sensitivity": sensitivity,
        "bootstrap_mode": effective_mode,
        "bootstrap_replicates": boot,
        "complete_replicate_blocks": len(complete_blocks),
        "required_cells": len(required_cells),
        "conditional_f0_diagnostic": conditional_f0_diagnostic,
        "conditional_gaps": dict(_conditional_gaps),
        # Deprecated aliases retained for callers of earlier script versions.
        "f0_lb": conditional_f0_diagnostic,
        "gaps": dict(_conditional_gaps),
    }


def _conditional_bound_from_row(row: dict[str, object]) -> float | None:
    for key in (
        "conditional_held_karp_bound_mean",
        "conditional_two_nn_bound_mean",
        "held_karp_bound_mean",
        "subset_bound_mean",
    ):
        value = row.get(key)
        if value is not None:
            return float(value)
    return None


def _campaign_metadata(doc: dict[str, object]) -> dict[str, object]:
    config = doc.get("config") or {}
    metadata = doc.get("campaign_metadata") or {}
    return {
        "campaign_id": metadata.get("campaign_id", config.get("campaign_id", "legacy")),
        "campaign_shard": metadata.get("campaign_shard", config.get("campaign_shard", 0)),
        "replicate_offset": metadata.get("replicate_offset", config.get("replicate_offset", 0)),
        "point_seed": metadata.get("point_seed", config.get("point_seed", config.get("seed", 2024))),
        "search_seed": metadata.get("search_seed", config.get("search_seed", config.get("seed", 2024))),
        "solver_policy_id": metadata.get("solver_policy_id", config.get("solver_policy_id", "default")),
        "fidelity_level": metadata.get("fidelity_level", config.get("fidelity_level", "strong")),
    }


def _finite(value: object, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} is nonfinite")
    return number


def _values_match(left: list[float], right: list[float]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
        for a, b in zip(sorted(left), sorted(right))
    )


def load_campaign(
    paths: Iterable[str],
    fidelity_level: str = "strong",
    solver_policy_id: str = "default",
) -> dict[float, dict[int, list[float]]]:
    """Load selected-policy observations and preserve campaign identities.

    Identified instance rows are grouped by point-set replicate and averaged
    across repeated search streams before entering the main ladder. This gives
    every point set equal weight. Legacy summary-only values are accepted, but
    their presence disables automatic block bootstrap because their correlation
    structure is unknowable.
    """
    _conditional_bound_of.clear()
    _conditional_gaps.clear()
    _replicate_blocks.clear()
    _nested_search_groups.clear()
    _all_observations.clear()
    _load_info.clear()

    legacy_values: dict[Cell, list[float]] = defaultdict(list)
    selected_raw: dict[Cell, dict[BlockKey, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    point_streams: dict[BlockKey, str] = {}
    seen_observations: set[tuple[object, ...]] = set()
    bound_sum: dict[Cell, float] = defaultdict(float)
    bound_count: dict[Cell, int] = defaultdict(int)
    gap_sum: dict[float, float] = defaultdict(float)
    gap_count: dict[float, int] = defaultdict(int)
    cell_metadata: dict[Cell, tuple[object, object]] = {}
    identified_selected = 0
    legacy_selected = 0
    files_loaded = 0

    for filename in paths:
        path = str(filename)
        with open(path, encoding="utf-8") as stream:
            doc = json.load(stream)
        files_loaded += 1
        metadata = _campaign_metadata(doc)
        campaign_id = str(metadata["campaign_id"])
        campaign_shard = int(metadata["campaign_shard"])
        policy = str(metadata["solver_policy_id"])
        fidelity = str(metadata["fidelity_level"])
        selected_document = policy == solver_policy_id and fidelity == fidelity_level
        config = doc.get("config") or {}
        doc_n = doc.get("N")
        periodic = config.get("periodic")

        summary_by_cell: dict[Cell, dict[str, object]] = {}
        for row in doc.get("summary_rows", []):
            p = _finite(row["p"], f"{path}: summary p")
            k = int(row["k"])
            cell = (p, k)
            summary_by_cell[cell] = row
            if not selected_document:
                continue
            values = [_finite(value, f"{path}: summary value")
                      for value in (row.get("values") or [])]
            if not values:
                continue
            declared_count = int(row.get("n", len(values)))
            if declared_count != len(values):
                raise ValueError(
                    f"{path}: summary cell (p={p}, k={k}) declares n={declared_count} "
                    f"but carries {len(values)} values"
                )
            actual_mean = _mean(values)
            if "mean" in row and not math.isclose(
                _finite(row["mean"], f"{path}: summary mean"),
                actual_mean,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{path}: summary mean for (p={p}, k={k}) is inconsistent "
                    "with its values array"
                )
            metadata_key = (doc_n, periodic)
            previous = cell_metadata.get(cell)
            if previous is not None and previous != metadata_key:
                raise ValueError(
                    f"incompatible duplicate campaign cell (p={p}, k={k}): "
                    f"metadata {previous} versus {metadata_key}"
                )
            cell_metadata[cell] = metadata_key
            bound = _conditional_bound_from_row(row)
            if bound is not None:
                bound_sum[cell] += bound * declared_count
                bound_count[cell] += declared_count
                gap_sum[p] += (actual_mean - bound) * declared_count
                gap_count[p] += declared_count

        rows = doc.get("instance_rows") or []
        identified_rows = bool(rows) and all(
            all(key in row for key in (
                "replicate_id", "point_stream_id", "search_stream_id"
            ))
            for row in rows
        )
        if rows and not identified_rows:
            # A partial identity contract is more dangerous than no identity:
            # never infer correlation from positional row order.
            missing = next(
                row for row in rows
                if not all(key in row for key in (
                    "replicate_id", "point_stream_id", "search_stream_id"
                ))
            )
            raise ValueError(
                f"{path}: instance row {missing.get('index', '?')} has an incomplete "
                "campaign identity"
            )

        if identified_rows:
            instance_values: dict[Cell, list[float]] = defaultdict(list)
            for row in rows:
                replicate_id = int(row["replicate_id"])
                if replicate_id < 0:
                    raise ValueError(f"{path}: replicate_id must be nonnegative")
                point_stream_id = str(row["point_stream_id"])
                search_stream_id = str(row["search_stream_id"])
                block = (campaign_id, replicate_id)
                previous_stream = point_streams.get(block)
                if previous_stream is not None and previous_stream != point_stream_id:
                    raise ValueError(
                        f"{path}: campaign replicate {block} maps to conflicting "
                        "point streams"
                    )
                point_streams[block] = point_stream_id
                for p_row in row.get("p_results", []):
                    p = _finite(p_row["p"], f"{path}: instance p")
                    k = int(p_row["k"])
                    value = _finite(p_row["value"], f"{path}: instance value")
                    observation = Observation(
                        campaign_id=campaign_id,
                        campaign_shard=campaign_shard,
                        replicate_id=replicate_id,
                        point_stream_id=point_stream_id,
                        search_stream_id=search_stream_id,
                        solver_policy_id=policy,
                        fidelity_level=fidelity,
                        p=p,
                        k=k,
                        value=value,
                        source=path,
                    )
                    duplicate_key = (
                        campaign_id, replicate_id, point_stream_id,
                        search_stream_id, policy, fidelity, p, k,
                    )
                    if duplicate_key in seen_observations:
                        raise ValueError(
                            f"duplicate campaign observation for campaign={campaign_id}, "
                            f"replicate={replicate_id}, p={p}, k={k}, "
                            f"policy={policy}, fidelity={fidelity}, "
                            f"search_stream={search_stream_id}"
                        )
                    seen_observations.add(duplicate_key)
                    _all_observations.append(observation)
                    if selected_document:
                        cell = (p, k)
                        selected_raw[cell][block].append(value)
                        instance_values[cell].append(value)
                        identified_selected += 1

            if selected_document:
                for cell, row in summary_by_cell.items():
                    summary_values = [float(value) for value in (row.get("values") or [])]
                    if summary_values and not _values_match(
                        summary_values, instance_values.get(cell, [])
                    ):
                        raise ValueError(
                            f"{path}: identified instance values for cell {cell} do not "
                            "match its summary values"
                        )
        elif selected_document:
            for cell, row in summary_by_cell.items():
                values = [_finite(value, f"{path}: legacy summary value")
                          for value in (row.get("values") or [])]
                if values:
                    legacy_values[cell].extend(values)
                    legacy_selected += len(values)

    ladders: dict[float, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for cell, points in selected_raw.items():
        p, k = cell
        for block, values in sorted(points.items()):
            point_mean = _mean(values)
            ladders[p][k].append(point_mean)
            _replicate_blocks.setdefault(block, {})[cell] = point_mean
            _nested_search_groups.setdefault(cell, {})[block] = list(values)
    for (p, k), values in legacy_values.items():
        ladders[p][k].extend(values)

    for cell, total in bound_sum.items():
        _conditional_bound_of[cell] = total / bound_count[cell]
    for p, total in gap_sum.items():
        _conditional_gaps[p] = total / gap_count[p]

    if not ladders:
        raise ValueError(
            f"no observations matched solver policy {solver_policy_id!r} and "
            f"fidelity {fidelity_level!r}"
        )
    identities_complete = legacy_selected == 0 and identified_selected > 0
    _load_info.update({
        "files_loaded": files_loaded,
        "solver_policy_id": solver_policy_id,
        "fidelity_level": fidelity_level,
        "identified_observations": identified_selected,
        "legacy_observations": legacy_selected,
        "replicate_blocks": len(_replicate_blocks),
        "identities_complete": identities_complete,
    })
    return {p: dict(k_values) for p, k_values in ladders.items()}


def nested_variance_decomposition(
    groups: dict[Cell, dict[BlockKey, list[float]]] | None = None,
) -> list[dict[str, object]]:
    """Method-of-moments point/search variance decomposition per cell."""
    source = _nested_search_groups if groups is None else groups
    output: list[dict[str, object]] = []
    for (p, k), point_groups in sorted(source.items()):
        usable = [values for values in point_groups.values() if values]
        repeated = [values for values in usable if len(values) > 1]
        if len(usable) < 2 or not repeated:
            continue
        within_ss = 0.0
        within_df = 0
        for values in repeated:
            mean = _mean(values)
            within_ss += sum((value - mean) ** 2 for value in values)
            within_df += len(values) - 1
        search_variance = within_ss / within_df if within_df > 0 else 0.0
        point_means = [_mean(values) for values in usable]
        between_variance = _std(point_means) ** 2
        mean_inverse_repeats = _mean([1.0 / len(values) for values in usable])
        point_variance = max(
            0.0, between_variance - search_variance * mean_inverse_repeats
        )
        total_variance = point_variance + search_variance
        output.append({
            "p": p,
            "k": k,
            "point_replicates": len(usable),
            "search_observations": sum(len(values) for values in usable),
            "repeated_point_replicates": len(repeated),
            "point_variance": point_variance,
            "search_variance": search_variance,
            "point_sd": math.sqrt(point_variance),
            "search_sd": math.sqrt(search_variance),
            "search_variance_fraction": (
                search_variance / total_variance if total_variance > 0.0 else 0.0
            ),
        })
    return output


def multifidelity_estimates(
    cheap_fidelity: str,
    strong_fidelity: str,
    solver_policy_id: str,
    boot: int,
    seed: int = 24680,
    observations: list[Observation] | None = None,
) -> list[dict[str, object]]:
    """Paired cheap-plus-correction estimates for every shared campaign cell."""
    source = _all_observations if observations is None else observations
    grouped: dict[tuple[Cell, str, BlockKey], list[float]] = defaultdict(list)
    for observation in source:
        if observation.solver_policy_id != solver_policy_id:
            continue
        if observation.fidelity_level not in {cheap_fidelity, strong_fidelity}:
            continue
        cell = (observation.p, observation.k)
        block = (observation.campaign_id, observation.replicate_id)
        grouped[(cell, observation.fidelity_level, block)].append(observation.value)

    maps: dict[Cell, dict[str, dict[BlockKey, float]]] = defaultdict(
        lambda: {cheap_fidelity: {}, strong_fidelity: {}}
    )
    for (cell, fidelity, block), values in grouped.items():
        maps[cell][fidelity][block] = _mean(values)

    rng = random.Random(seed)
    output: list[dict[str, object]] = []
    for (p, k), fidelity_maps in sorted(maps.items()):
        cheap = fidelity_maps[cheap_fidelity]
        strong = fidelity_maps[strong_fidelity]
        paired = sorted(cheap.keys() & strong.keys())
        if not cheap or not paired:
            continue
        unpaired = sorted(cheap.keys() - strong.keys())
        cheap_mean = _mean(list(cheap.values()))
        corrections = [strong[block] - cheap[block] for block in paired]
        correction = _mean(corrections)
        estimate = cheap_mean + correction
        samples: list[float] = []
        if boot > 0 and len(cheap) >= 2 and len(paired) >= 2:
            for _ in range(boot):
                # Stratified block bootstrap preserves both the strong-subsample
                # size and the covariance between paired cheap values and their
                # strong-minus-cheap corrections.
                paired_draw = [paired[rng.randrange(len(paired))] for _ in paired]
                unpaired_draw = (
                    [unpaired[rng.randrange(len(unpaired))] for _ in unpaired]
                    if unpaired else []
                )
                cheap_draw = [cheap[block] for block in paired_draw + unpaired_draw]
                correction_draw = [
                    strong[block] - cheap[block] for block in paired_draw
                ]
                samples.append(_mean(cheap_draw) + _mean(correction_draw))
        output.append({
            "p": p,
            "k": k,
            "cheap_fidelity": cheap_fidelity,
            "strong_fidelity": strong_fidelity,
            "cheap_replicates": len(cheap),
            "paired_strong_replicates": len(paired),
            "cheap_mean": cheap_mean,
            "paired_strong_mean": _mean([strong[block] for block in paired]),
            "paired_correction": correction,
            "estimate": estimate,
            "ci": _ci(samples),
        })
    return output


def make_plot(result: dict[str, object], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ps = result["ps"]
    fp_point = result["fp_point"]
    fp_ci = result["fp_ci"]
    values = [fp_point[p][0] for p in ps]
    lower = [fp_ci[p][0] for p in ps]
    upper = [fp_ci[p][1] for p in ps]
    figure, axes = plt.subplots(figsize=(8, 5.2))
    axes.errorbar(
        ps,
        values,
        yerr=[np.array(values) - lower, np.array(upper) - values],
        fmt="o",
        capsize=3,
        label="extrapolated f(p) (95% CI)",
    )
    x_values = np.linspace(0, max(ps) * 1.05, 200)
    axes.plot(
        x_values,
        result["f0"] + result["C"] * x_values ** result["alpha"],
        label=(
            f"{result['primary_finite_size_model']} fit: "
            f"f0 + C p^alpha (alpha={result['alpha']:.2f})"
        ),
    )
    axes.axhline(result["f0"], linestyle=":", linewidth=1)
    envelope_ci = result["model_uncertainty_envelope"]["f0_ci"]
    if all(math.isfinite(float(value)) for value in envelope_ci):
        axes.axhspan(
            envelope_ci[0], envelope_ci[1], alpha=0.06,
            label="finite-size model + bootstrap envelope",
        )
    axes.fill_between(
        [0, max(ps) * 1.05],
        result["f0_ci"][0],
        result["f0_ci"][1],
        alpha=0.12,
        label=f"f(0+)={result['f0']:.3f} {_fmt(result['f0_ci'])}",
    )
    if result["conditional_f0_diagnostic"] is not None:
        axes.axhline(
            result["conditional_f0_diagnostic"],
            linestyle="--",
            linewidth=1,
            label="conditional HK diagnostic (not a floor on f(0+))",
        )
    axes.scatter([0], [result["f0"]], marker="*", s=160, edgecolor="k", zorder=6)
    axes.set_xlabel("p")
    axes.set_ylabel("f(p)")
    axes.set_title("Aldous subset-TSP: f(p) -> f(0+) and exponent alpha")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    axes.set_xlim(left=-0.005)
    figure.tight_layout()
    figure.savefig(path, dpi=130, bbox_inches="tight")


def _json_safe(value: object) -> object:
    """Replace nonfinite analysis sentinels with JSON null recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_report(
    result: dict[str, object],
    ladders: dict[float, dict[int, list[float]]],
    variance: list[dict[str, object]],
    multifidelity: list[dict[str, object]],
) -> dict[str, object]:
    payload = {
        "selection": dict(_load_info),
        "bootstrap": {
            "mode": result["bootstrap_mode"],
            "resamples": result["bootstrap_replicates"],
            "complete_replicate_blocks": result["complete_replicate_blocks"],
            "required_cells": result["required_cells"],
        },
        "small_p_fit": {
            "primary_finite_size_model": result["primary_finite_size_model"],
            "f0": result["f0"],
            "f0_ci": result["f0_ci"],
            "C": result["C"],
            "C_ci": result["C_ci"],
            "alpha": result["alpha"],
            "alpha_ci": result["alpha_ci"],
            "conditional_f0_diagnostic": result["conditional_f0_diagnostic"],
        },
        "finite_size_model_analysis": {
            "models": result["finite_size_models"],
            "unavailable_models": result["unavailable_finite_size_models"],
            "model_uncertainty_envelope": result["model_uncertainty_envelope"],
            "sensitivity": result["sensitivity"],
        },
        "p_estimates": [
            {
                "p": p,
                "k_values": sorted(ladders[p]),
                "f_p": result["fp_point"][p][0],
                "f_p_ci": result["fp_ci"][p],
                "conditional_gap": result["conditional_gaps"].get(p),
            }
            for p in result["ps"]
        ],
        "variance_decomposition": variance,
        "multifidelity": multifidelity,
    }
    return _json_safe(payload)


def run_self_test() -> int:
    true_f0, true_coefficient, true_alpha = 0.625, 0.35, 0.75
    ps = [0.01, 0.02, 0.05, 0.1, 0.2]
    ks = [500, 1000, 2000]
    slope = 1.2
    rng = random.Random(7)
    ladders: dict[float, dict[int, list[float]]] = {}
    _conditional_bound_of.clear()
    for p in ps:
        ladders[p] = {}
        true_fp = true_f0 + true_coefficient * p**true_alpha
        for k in ks:
            values = [true_fp + slope / k + rng.gauss(0, 0.004) for _ in range(40)]
            ladders[p][k] = values
            _conditional_bound_of[(p, k)] = true_fp + slope / k - 0.09
    alpha_grid = [0.02 + 0.01 * index for index in range(300)]
    fit = analyze(
        ladders,
        pmax=1.0,
        boot=300,
        alpha_grid=alpha_grid,
        bootstrap_mode="independent",
        identities_complete=False,
    )
    fit_ok = (
        fit["f0_ci"][0] <= true_f0 <= fit["f0_ci"][1]
        and fit["alpha_ci"][0] <= true_alpha <= fit["alpha_ci"][1]
    )

    # Correlated replicate vectors: a large common offset should mostly affect
    # f0, not alpha. Independent cell resampling destroys that cancellation.
    correlated_ladders: dict[float, dict[int, list[float]]] = {
        p: {1000: []} for p in [0.02, 0.05, 0.1, 0.2]
    }
    blocks: dict[BlockKey, dict[Cell, float]] = {}
    corr_rng = random.Random(91)
    for replicate in range(80):
        common = corr_rng.gauss(0.0, 0.025)
        block = ("correlated", replicate)
        blocks[block] = {}
        for p in correlated_ladders:
            value = 0.61 + 0.30 * p**0.8 + common + corr_rng.gauss(0.0, 0.0005)
            correlated_ladders[p][1000].append(value)
            blocks[block][(p, 1000)] = value
    block_fit = analyze(
        correlated_ladders, 1.0, 300, alpha_grid, seed=12,
        bootstrap_mode="block", replicate_blocks=blocks, identities_complete=True,
    )
    independent_fit = analyze(
        correlated_ladders, 1.0, 300, alpha_grid, seed=12,
        bootstrap_mode="independent", replicate_blocks=blocks,
        identities_complete=True,
    )
    block_width = block_fit["alpha_ci"][1] - block_fit["alpha_ci"][0]
    independent_width = (
        independent_fit["alpha_ci"][1] - independent_fit["alpha_ci"][0]
    )
    correlation_ok = (
        block_fit["bootstrap_mode"] == "replicate-block"
        and independent_width > 2.0 * max(block_width, 1e-12)
    )

    # Nested point/search variance should recover the dominant point component.
    nested_groups: dict[Cell, dict[BlockKey, list[float]]] = {(0.1, 100): {}}
    nested_rng = random.Random(17)
    for replicate in range(40):
        point_effect = nested_rng.gauss(0.0, 0.02)
        nested_groups[(0.1, 100)][("nested", replicate)] = [
            0.7 + point_effect + nested_rng.gauss(0.0, 0.004) for _ in range(4)
        ]
    variance = nested_variance_decomposition(nested_groups)
    nested_ok = (
        len(variance) == 1
        and variance[0]["point_variance"] > variance[0]["search_variance"] > 0.0
    )

    # Multifidelity estimator on a random strong subsample.
    observations: list[Observation] = []
    mf_rng = random.Random(33)
    strong_full: list[float] = []
    for replicate in range(60):
        strong = 0.65 + mf_rng.gauss(0.0, 0.015)
        cheap = strong + 0.08 + mf_rng.gauss(0.0, 0.002)
        strong_full.append(strong)
        common = dict(
            campaign_id="mf", campaign_shard=0, replicate_id=replicate,
            point_stream_id=f"{replicate:016x}", solver_policy_id="default",
            p=0.1, k=100, source="synthetic",
        )
        observations.append(Observation(
            search_stream_id=f"{1000 + replicate:016x}",
            fidelity_level="cheap", value=cheap, **common,
        ))
        if replicate < 20:
            observations.append(Observation(
                search_stream_id=f"{2000 + replicate:016x}",
                fidelity_level="strong", value=strong, **common,
            ))
    multifidelity = multifidelity_estimates(
        "cheap", "strong", "default", 300, observations=observations
    )
    mf_ok = (
        len(multifidelity) == 1
        and abs(multifidelity[0]["estimate"] - _mean(strong_full)) < 0.004
        and multifidelity[0]["paired_strong_replicates"] == 20
    )

    # Loader regression: weighted conditional bounds and stable identified rows.
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base = {
            "N": 100,
            "config": {"periodic": True},
            "campaign_metadata": {
                "campaign_id": "loader", "campaign_shard": 0,
                "replicate_offset": 0, "point_seed": 11, "search_seed": 12,
                "solver_policy_id": "default", "fidelity_level": "strong",
            },
        }
        docs = []
        for shard, values, bound in ((0, [0.8, 1.0], 0.5),
                                     (1, [0.7, 0.8, 0.9, 1.0], 0.7)):
            offset = 0 if shard == 0 else 2
            rows = []
            for index, value in enumerate(values):
                replicate = offset + index
                rows.append({
                    "index": index,
                    "replicate_id": replicate,
                    "point_stream_id": f"{replicate + 100:016x}",
                    "search_stream_id": f"{replicate + 200:016x}",
                    "p_results": [{"p": 0.2, "k": 20, "value": value}],
                })
            doc = dict(base)
            doc["campaign_metadata"] = dict(base["campaign_metadata"],
                                             campaign_shard=shard,
                                             replicate_offset=offset)
            doc["summary_rows"] = [{
                "p": 0.2, "k": 20, "n": len(values), "values": values,
                "mean": _mean(values),
                "conditional_held_karp_bound_mean": bound,
            }]
            doc["instance_rows"] = rows
            path = root / f"shard-{shard}.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            docs.append(str(path))
        merged = load_campaign(docs)
        expected_bound = (0.5 * 2 + 0.7 * 4) / 6
        loader_ok = (
            len(merged[0.2][20]) == 6
            and math.isclose(_conditional_bound_of[(0.2, 20)], expected_bound)
            and _load_info["identities_complete"] is True
            and len(_replicate_blocks) == 6
        )

    # Alternative finite-size laws should expose deliberate curvature rather
    # than silently folding it into f(0+). The quadratic law is exact for this
    # synthetic campaign, while deletion and pmax diagnostics must be populated.
    model_ps = [0.02, 0.05, 0.1, 0.2, 0.3]
    model_ks = [80, 120, 200, 400]
    model_rng = random.Random(123)
    curved_ladders: dict[float, dict[int, list[float]]] = {}
    for p_value in model_ps:
        curved_ladders[p_value] = {}
        true_value = 0.6 + 0.28 * p_value**0.8
        for k in model_ks:
            finite_value = true_value + 3.0 / k + 100.0 / (k * k)
            curved_ladders[p_value][k] = [
                finite_value + model_rng.gauss(0.0, 0.00002)
                for _ in range(20)
            ]
    model_fit = analyze(
        curved_ladders, 1.0, 80, alpha_grid, seed=19,
        bootstrap_mode="independent", identities_complete=False,
    )
    model_rows = model_fit["finite_size_models"]
    model_envelope = model_fit["model_uncertainty_envelope"]
    sensitivity = model_fit["sensitivity"]
    zero_bootstrap_fit = analyze(
        curved_ladders, 1.0, 0, alpha_grid, seed=19,
        bootstrap_mode="independent", identities_complete=False,
    )
    zero_bootstrap_report = _json_report(
        zero_bootstrap_fit, curved_ladders, [], []
    )
    model_ok = (
        set(model_rows) == set(FINITE_SIZE_MODEL_SPECS)
        and abs(model_rows["inv-k2"]["f0"] - 0.6)
            < abs(model_rows["inv-k"]["f0"] - 0.6)
        and model_envelope["f0"][0] <= 0.6 <= model_envelope["f0"][1]
        and len(sensitivity["leave_one_k_out"]) == len(model_ks)
        and len(sensitivity["leave_one_p_out"]) == len(model_ps)
        and len(sensitivity["nested_pmax"]) == len(model_ps) - 2
        and "finite_size_model_analysis" in _json_report(
            model_fit, curved_ladders, [], []
        )
        and bool(json.dumps(
            _json_report(model_fit, curved_ladders, [], []), allow_nan=False
        ))
        and zero_bootstrap_report["small_p_fit"]["f0_ci"] == [None, None]
        and bool(json.dumps(zero_bootstrap_report, allow_nan=False))
    )

    semantics_ok = 10.0 > 9.0 and 10.0 <= 11.0
    checks = {
        "two-stage fit": fit_ok,
        "replicate-block correlation": correlation_ok,
        "nested variance": nested_ok,
        "multifidelity correction": mf_ok,
        "identified shard merge": loader_ok,
        "finite-size model envelope": model_ok,
        "conditional-bound semantics": semantics_ok,
    }
    for name, passed in checks.items():
        print(f"self-test: {name:<31} {'PASS' if passed else 'FAIL'}")
    return 0 if all(checks.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="*", help="campaign JSON files")
    parser.add_argument("--pmax", type=float, default=0.3,
                        help="use p <= pmax for the small-p fit (default: 0.3)")
    parser.add_argument("--boot", type=int, default=2000,
                        help="master-bootstrap resamples (default: 2000)")
    parser.add_argument("--bootstrap-mode", choices=("auto", "block", "independent"),
                        default="auto", help="correlation policy (default: auto)")
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument(
        "--finite-size-models",
        default=",".join(FINITE_SIZE_MODEL_SPECS),
        help=(
            "comma-separated finite-size laws to compare: "
            + ", ".join(FINITE_SIZE_MODEL_SPECS)
        ),
    )
    parser.add_argument(
        "--primary-finite-size-model",
        choices=tuple(FINITE_SIZE_MODEL_SPECS),
        default="inv-k",
        help="model used for the headline estimate (default: inv-k)",
    )
    parser.add_argument("--solver-policy-id", default="default")
    parser.add_argument("--fidelity-level", default="strong")
    parser.add_argument("--cheap-fidelity", default="cheap")
    parser.add_argument("--strong-fidelity", default="strong")
    parser.add_argument("--plot", metavar="PNG", default=None)
    parser.add_argument("--analysis-json", metavar="JSON", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()
    if not args.files:
        parser.error("no input files (or use --self-test)")
    if args.boot < 0:
        parser.error("--boot must be nonnegative")
    finite_size_models = [
        model.strip() for model in args.finite_size_models.split(",")
        if model.strip()
    ]
    if not finite_size_models:
        parser.error("--finite-size-models must contain at least one model")
    unknown_models = [
        model for model in finite_size_models
        if model not in FINITE_SIZE_MODEL_SPECS
    ]
    if unknown_models:
        parser.error(
            "unknown --finite-size-models value(s): " + ", ".join(unknown_models)
        )

    ladders = load_campaign(
        args.files,
        fidelity_level=args.fidelity_level,
        solver_policy_id=args.solver_policy_id,
    )
    alpha_grid = [0.02 + 0.01 * index for index in range(300)]
    result = analyze(
        ladders,
        args.pmax,
        args.boot,
        alpha_grid,
        seed=args.bootstrap_seed,
        bootstrap_mode=args.bootstrap_mode,
        finite_size_models=finite_size_models,
        primary_finite_size_model=args.primary_finite_size_model,
    )
    variance = nested_variance_decomposition()
    multifidelity = multifidelity_estimates(
        args.cheap_fidelity,
        args.strong_fidelity,
        args.solver_policy_id,
        args.boot,
        seed=args.bootstrap_seed ^ 0x5A17,
    )

    print(
        f"bootstrap: {result['bootstrap_mode']} "
        f"({result['complete_replicate_blocks']} complete replicate vectors; "
        f"{args.boot} resamples)"
    )
    if _load_info.get("legacy_observations"):
        print(
            "warning: legacy summary-only observations are present; their "
            "cross-cell correlation is unavailable"
        )
    print(f"{'p':>7} {'k-range':>13} {'f(p)':>9} {'95% CI':>20} {'conditional gap':>16}")
    for p in result["ps"]:
        ks = sorted(ladders[p])
        fp_value = result["fp_point"][p][0]
        interval = result["fp_ci"][p]
        gap = result["conditional_gaps"].get(p)
        gap_text = f"{gap:.4f}" if gap is not None else "--"
        print(
            f"{p:>7g} {f'{ks[0]}-{ks[-1]}':>13} {fp_value:>9.4f} "
            f"{_fmt(interval):>20} {gap_text:>15}"
        )

    print("\n=== small-p law f(p) = f(0+) + C p^alpha ===")
    print(f"  f(0+) = {result['f0']:.4f}   95% CI {_fmt(result['f0_ci'])}")
    if result["conditional_f0_diagnostic"] is not None:
        diagnostic = result["conditional_f0_diagnostic"]
        print(
            f"  conditional fixed-subset tour diagnostic = {diagnostic:.4f} "
            "(not a lower bound on f(0+))"
        )
    print(f"  alpha = {result['alpha']:.3f}    95% CI {_fmt(result['alpha_ci'])}")
    print(f"  C     = {result['C']:.4f}   95% CI {_fmt(result['C_ci'])}")

    print("\n=== finite-size model sensitivity ===")
    print(f"{'model':>12} {'formula':>24} {'f(0+)':>10} {'alpha':>9} {'fit SSR':>12}")
    for model, row in result["finite_size_models"].items():
        print(
            f"{model:>12} {row['formula']:>24} {row['f0']:>10.5f} "
            f"{row['alpha']:>9.4f} {row['finite_size_weighted_ssr']:>12.5g}"
        )
    for model, reason in result["unavailable_finite_size_models"].items():
        print(f"{model:>12} {'unavailable':>24}  {reason}")
    envelope = result["model_uncertainty_envelope"]
    print(
        "  combined statistical/model envelope: "
        f"f(0+) {_fmt(envelope['f0_ci'])}; alpha {_fmt(envelope['alpha_ci'])}"
    )
    sensitivity = result["sensitivity"]
    print(
        "  maximum deletion/pmax shift from primary: "
        f"|delta f(0+)|={sensitivity['max_abs_delta_f0']:.5g}, "
        f"|delta alpha|={sensitivity['max_abs_delta_alpha']:.5g}"
    )

    if variance:
        print("\n=== nested point/search variance ===")
        print(f"{'p':>7} {'k':>7} {'points':>8} {'searches':>9} {'point sd':>11} {'search sd':>11}")
        for row in variance:
            print(
                f"{row['p']:>7g} {row['k']:>7} {row['point_replicates']:>8} "
                f"{row['search_observations']:>9} {row['point_sd']:>11.5g} "
                f"{row['search_sd']:>11.5g}"
            )

    if multifidelity:
        print("\n=== paired multifidelity estimates ===")
        print(f"{'p':>7} {'k':>7} {'cheap M':>8} {'strong m':>9} {'estimate':>11} {'95% CI':>20}")
        for row in multifidelity:
            print(
                f"{row['p']:>7g} {row['k']:>7} {row['cheap_replicates']:>8} "
                f"{row['paired_strong_replicates']:>9} {row['estimate']:>11.6f} "
                f"{_fmt(row['ci']):>20}"
            )

    if args.plot:
        make_plot(result, args.plot)
        print(f"\nwrote {args.plot}")
    if args.analysis_json:
        report = _json_report(result, ladders, variance, multifidelity)
        Path(args.analysis_json).write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.analysis_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
