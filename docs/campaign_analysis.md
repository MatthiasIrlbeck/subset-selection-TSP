# Campaign analysis and model uncertainty

`scripts/analyze_campaign.py` estimates the infinite-size curve in two stages:

1. extrapolate every selected probability across cardinalities;
2. fit `f(p) = f(0+) + C p^alpha` over `p <= --pmax`.

The primary historical finite-size law remains:

```text
inv-k: f(p,k) = f(p) + b/k
```

Two alternatives are evaluated by default:

```text
inv-k2:      f(p,k) = f(p) + b/k + c/k^2
inv-sqrt-k:  f(p,k) = f(p) + b/sqrt(k)
```

Use `--primary-finite-size-model` to select the headline law and
`--finite-size-models` to control the comparison set. For example:

```bash
python3 scripts/analyze_campaign.py \
  --bootstrap-mode block \
  --primary-finite-size-model inv-k \
  --finite-size-models inv-k,inv-k2,inv-sqrt-k \
  --analysis-json analysis.json \
  campaign/*.json
```

## Statistical and model uncertainty

Every available finite-size model is evaluated inside the same master
bootstrap. When stable replicate identities are present, each draw resamples
complete point-set replicate vectors, preserving common-random-number
correlation across all `(p,k)` cells and all fitted models.

The report distinguishes:

- the primary model's point estimate and bootstrap interval;
- each alternative model's estimate and interval;
- a point-estimate envelope across finite-size laws;
- a combined statistical/model envelope spanning all model-specific bootstrap
  intervals.

The combined envelope is a sensitivity diagnostic, not a formal posterior
credible interval and not a proof that the listed models exhaust finite-size
bias.

## Deletion and range sensitivity

The analyzer also recomputes the complete two-stage fit under:

- leave-one-`k`-out deletion, whenever every probability retains enough
  cardinality levels for the primary law;
- leave-one-`p`-out deletion, whenever at least three probabilities remain;
- every nested `pmax` prefix containing at least three probability values.

The JSON report stores every refit and the largest absolute shifts in `f(0+)`
and `alpha`. Large shifts indicate that the headline result depends materially
on one size, one probability, or the chosen small-`p` range.

## Required campaign identities

For correlation-aware inference, every instance row should carry:

```text
campaign_id
replicate_id
point_stream_id
search_stream_id
solver_policy_id
fidelity_level
```

Repeated search streams on one point set are averaged before the primary fit,
so point sets receive equal weight. Where repeats exist, the report separately
estimates point-instance and search-seed variance. Cheap and strong policies
that share point identities also produce a paired multifidelity estimate.

Legacy summary-only documents remain readable, but their cross-cell covariance
cannot be reconstructed. The analyzer therefore labels and uses an
independent-cell bootstrap for those inputs.

## Interpretation

Conditional Held-Karp and two-nearest-neighbor values concern the optimal tour
through the subset found by the heuristic. They are diagnostics for tour
ordering and do not lower-bound the global optimum over all size-`k` subsets.
The exact small-instance subset solver is the appropriate calibration tool for
that distinction.
