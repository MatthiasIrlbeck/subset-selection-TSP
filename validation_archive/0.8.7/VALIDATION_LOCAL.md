# Local validation summary

This file records local validation completed on this machine.

## Current repo tests

Status: passed.

- Release/Python CTest: 13/13 passed.
- ASan/UBSan CTest: 13/13 passed.
- Manual ASan/UBSan hybrid smoke: passed.
- Manual ASan smoke JSON schema validation: passed.

## Original-vs-current parity

Status: completed.

Original upstream repository was cloned and built locally.

Main harness:
- Grid backend vs brute-force backend matched exactly: max_abs_mean_delta = 0.0.
- Current-vs-original tiny-grid-parity: baseline_max_abs_mean_delta ≈ 0.05209208.
- Current-vs-original small-hybrid-parity: baseline_max_abs_mean_delta ≈ 0.11283812.

Controlled --knn comparison:
- tiny balanced max_abs_delta ≈ 0.05209208.
- hybrid max_abs_delta ≈ 0.20276506.

Interpretation:
These are small deterministic validation scenarios. They prove the parity harness works against the original executable, but they are not publication-scale solver-quality benchmarks.

Raw parity files are stored under:
- external_validation/original-parity/

Summary parity files are stored under:
- validation_runs/original-parity/

## Real LKH oracle validation

Status: passed.

LKH executable:
- /usr/local/bin/LKH

Smoke results:
- matrix format: improved
- euc2d format: improved
- real_oracle_manifest.json status: passed

Files are stored under:
- validation_runs/real-oracle/

## Benchmark sanity run

Status: completed.

Scenarios:
- balanced_N300_i3
- hybrid_N300_i3

Both result JSON files validated against schema.
Plots and CSVs were generated.

Files are stored under:
- validation_runs/publication-benchmark/

## Remaining optional work

For publication-quality evidence, run larger benchmarks with more instances and larger N.
