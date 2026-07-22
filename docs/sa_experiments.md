# Experimental simulated-annealing controls

The release default remains the historical fixed geometric temperature schedule
and one candidate proposal per iteration.  Those defaults are deliberately
unchanged: both alternatives below are experimental policies that must earn a
place on a matched-worker-second frontier before becoming release defaults.

## Restart-local temperature calibration

Enable calibration with:

```bash
--sa-auto-temperature \
--sa-temperature-samples 256 \
--sa-temperature-quantile 0.5 \
--sa-initial-uphill-acceptance 0.6 \
--sa-final-uphill-acceptance 0.01
```

Before a restart's anneal, the solver copies that restart's RNG and samples
valid positive move deltas from the same proposal and insertion policy.  The
copy is important: calibration never consumes the RNG stream used by the real
search.  For the configured positive-delta quantile `d` and target acceptance
`a`, the endpoint is

```text
T = d / -log(a).
```

Calibration has a bounded attempt count.  If too few positive deltas are found,
or if the derived schedule is nonfinite or nondecreasing, the restart safely
uses the configured fixed `sa_t0` and `sa_t1` endpoints.  Elite kicks retain
their lower-temperature cap.

## Multiple-candidate proposals

`--sa-candidate-trials n` evaluates `n` independently generated candidate
swaps per SA iteration.  With `n=1`, the exact historical proposal path and RNG
consumption are preserved.  With `n>1`, the lowest-delta valid trial is proposed
except with probability
`--sa-multiple-try-random-probability`, when a uniformly random valid trial is
used to retain broader exploration.  The selected move then passes through the
ordinary Metropolis acceptance test.

This is a heuristic multiple-candidate proposal controller, not a claim of a
reversible multiple-try Markov-chain kernel.  Its objective is lower tour length
at fixed compute, not stationary-distribution sampling.

## Telemetry

Every recorded restart reports:

```text
restart_sa_t0
restart_sa_t1
restart_sa_temperature_samples
restart_sa_temperature_calibrated
```

Aggregate search statistics report candidate evaluations, calibrated and
fallback schedule counts, calibration attempts and samples, endpoint summaries,
and move/acceptance counts in ten temperature-schedule deciles.  The decile
arrays reconcile exactly with the aggregate SA move and acceptance counters.

Use `scripts/tune_search_policy.py` to compare these policies on identical point
and search streams.  Tune on one replicate set and evaluate the selected policy
on held-out point streams.  Acceptance targets should be selected separately by
`p`, geometry, scale, seed kind, and controller role when the measured frontier
shows material interactions.
