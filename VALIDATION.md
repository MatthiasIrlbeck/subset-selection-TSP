# 0.8.7 refreshed validation note

Version 0.8.7 refreshes the bundled validation artifacts with the current executable and schema, adds `validation_artifacts_schema` regression coverage, and rerenders `docs/known_good_benchmarks.md` from the refreshed manifests.

The bundled artifacts under `validation_runs/` are smoke/validation evidence, not publication-scale Monte Carlo evidence. They are intended to prove that the package builds, runs, validates its result schema, and preserves deterministic backend parity on small scenarios.

## Automated build/test validation

Release build with Python regressions enabled:

```bash
cmake -S . -B build-validation -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DALDOUS_TSP_ENABLE_PYTHON_TESTS=ON
cmake --build build-validation --parallel 2
ctest --test-dir build-validation --output-on-failure
```

The source package includes CTest coverage for:

```text
core_unit_tests
cli_self_test
cli_quick_smoke
cli_fake_oracle_smoke
backend_parity_smoke
original_parity_compat_smoke
real_oracle_smoke_optional
cli_regressions
schema_validation
exhaustive_policy_benchmark_smoke
install_tree_regressions
validation_report_render
validation_artifacts_schema
```

`validation_artifacts_schema` validates every bundled current-executable result JSON under `validation_runs/` against `schema/results.schema.json` and checks that the result files use the current project version. Non-result manifests and fake-original compatibility outputs are intentionally skipped.

## Bundled validation artifacts

The bundled validation directory was regenerated with project version `0.8.7` and result schema version `13`.

| Check | Result |
|---|---|
| Backend parity | grid vs brute force matched with `max_abs_mean_delta = 0.0` |
| Benchmark suite | 9 validation-sized scenarios completed |
| Exhaustive 2-opt policy | `final-only` reduced two-opt scans relative to `all-polish` in the smoke scenario |
| Profile smoke | completed and emitted phase/counter telemetry |
| Real oracle smoke | explicitly skipped because no LKH/Concorde executable was available |
| Original-compatible routing | passed with `tests/fake_original_cli.py`; this is not upstream solver parity |
| Bundled result JSON schema validation | 19 current-executable result JSON files validated |

Regenerate the compact report with:

```bash
python3 scripts/render_validation_report.py \
  --validation-dir validation_runs \
  --output docs/known_good_benchmarks.md
```

## Backend parity

Command:

```bash
python3 scripts/benchmark_parity.py \
  --exe build-validation/aldous_tsp \
  --out-dir validation_runs/backend-parity
```

Result: grid and brute-force KNN backends matched exactly on the bundled parity scenarios.

Artifacts:

```text
validation_runs/backend-parity/parity_manifest.csv
validation_runs/backend-parity/parity_manifest.json
validation_runs/backend-parity/*-grid.json
validation_runs/backend-parity/*-bruteforce.json
```

## Validation-sized benchmark suite

Command:

```bash
python3 scripts/benchmark.py \
  --exe build-validation/aldous_tsp \
  --suite all \
  --threads 1 \
  --out-dir validation_runs/benchmark-all
```

The built-in scenarios are intentionally validation-sized so CI and source-package checks complete quickly. For publication-scale Monte Carlo evidence, copy or extend `scripts/benchmark.py` scenarios and use larger `N`, more instances, and higher restart/iteration budgets.

Artifacts:

```text
validation_runs/benchmark-all/manifest.csv
validation_runs/benchmark-all/manifest.json
validation_runs/benchmark-all/ablation_manifest.csv
validation_runs/benchmark-all/ablation_manifest.json
validation_runs/benchmark-all/*.json
validation_runs/benchmark-all.log
```

## Exhaustive 2-opt policy comparison

Command:

```bash
python3 scripts/benchmark_exhaustive_policy.py \
  --exe build-validation/aldous_tsp \
  --out-dir validation_runs/exhaustive-policy \
  --N 72 \
  --instances 1 \
  --p-values 0.25,1.0 \
  --sa-iters 20 \
  --restarts 1 \
  --tsp-restarts 1 \
  --tsp-ils 4 \
  --final-exhaustive-k 96 \
  --check
```

Artifacts:

```text
validation_runs/exhaustive-policy/final-only.json
validation_runs/exhaustive-policy/all-polish.json
validation_runs/exhaustive-policy/manifest.csv
```

## Profile smoke

Command:

```bash
python3 scripts/profile_run.py \
  --exe build-validation/aldous_tsp \
  --output validation_runs/profile/profile.json \
  --N 120 \
  --instances 1 \
  --p-values 0.05,0.25,1.0 \
  --mode hybrid \
  --threads 1 \
  --extra --sa-iters 200 --restarts 1 --tsp-restarts 1 --tsp-ils 8
```

Artifacts:

```text
validation_runs/profile/profile.json
validation_runs/profile/profile.log
```

## Real oracle smoke

Command:

```bash
python3 scripts/oracle_real_smoke.py \
  --exe build-validation/aldous_tsp \
  --out-dir validation_runs/real-oracle-smoke
```

The container used for this package had no LKH or Concorde executable installed, so the smoke script wrote an explicit skip manifest:

```text
validation_runs/real-oracle-smoke/real_oracle_manifest.json
```

To require real solvers locally:

```bash
python3 scripts/oracle_real_smoke.py \
  --exe build-validation/aldous_tsp \
  --out-dir real-oracle-smoke \
  --require
```

## Original-vs-current parity status

The original-compatible command path was exercised with `tests/fake_original_cli.py`:

```bash
python3 scripts/benchmark_parity.py \
  --exe build-validation/aldous_tsp \
  --baseline-exe tests/fake_original_cli.py \
  --baseline-kind original \
  --suite original-compat-smoke \
  --out-dir validation_runs/original-compat
```

This verifies command routing only: current-only flags do not leak into an original-style baseline command. It is **not** a solver-quality comparison with the upstream prototype.

True upstream parity was not run in this container because `github.com` DNS resolution failed and no compiled original executable was available. The fetch attempt is recorded in:

```text
validation_runs/original-source-fetch.log
```

Run true parity locally with:

```bash
python3 scripts/benchmark_parity.py \
  --exe build-validation/aldous_tsp \
  --baseline-exe /path/to/original/aldous_tsp \
  --baseline-kind original \
  --out-dir parity-original
```
