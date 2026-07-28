# Known-good benchmark and validation commands

This page combines reusable validation commands with compact tables rendered from the bundled current-release manifests in `validation_runs/current/`. Historical artifacts under `validation_archive/` are not included.

```bash
python3 scripts/render_validation_report.py --validation-dir validation_runs/current --output docs/known_good_benchmarks.md
```

The bundled artifacts are smoke/validation runs, not large Monte Carlo evidence. Use them to check release health, deterministic parity, and output-schema stability before running larger experiments.

## Bundled validation summary

| Check | Observed result |
| --- | --- |
| Release/Python CTest inventory | 32 tests; run separately; see release validation and CI |
| Backend parity | max_abs_mean_delta = 0.0 |
| Benchmark suite | 2 scenarios, total wall 0.7351s |
| Real oracle smoke | skipped (No real LKH or Concorde executable found) |
| Original-compatible routing | fake original CLI routing passed; not a solver-quality parity result |

## Backend parity manifest

| Scenario | Comparison | Matched p | Max \|Δmean\| | Grid wall (s) | Brute wall (s) |
| --- | --- | --- | --- | --- | --- |
| tiny-grid-parity | grid-vs-bruteforce | 3 | 0 | 0.062 | 0.0587 |
| small-hybrid-parity | grid-vs-bruteforce | 4 | 0 | 0.011 | 0.0114 |

## Benchmark suite manifest excerpt

| Scenario | Suite | N | Instances | Mode | Wall (s) | Best p | Best mean | 2-opt imp | High-p imp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny-balanced | smoke | 60 | 2 | balanced | 0.5223 | 0.05 | 0.2648 | 952 | 3 |
| small-hybrid | smoke | 120 | 2 | hybrid | 0.2127 | 0.02 | 0.256 | 2886 | 8 |

## Exhaustive 2-opt policy comparison

| Policy | Wall (s) | Best mean | 2-opt scans | 2-opt imp | Or-opt scans | TSP s | Subset s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final-only | 0.0132 | 0.6688 | 23840 | 320 | 160148 | 0.0103 | 0.0026 |
| all-polish | 0.0131 | 0.6688 | 472514 | 392 | 332123 | 0.0089 | 0.002 |

`final-only` reduced two-opt scans by `448674` relative to `all-polish` in the bundled validation run.

## Original-compatible parity routing

This table validates command routing only. It uses `tests/fake_original_cli.py` to ensure current-only flags do not leak into an original-style baseline command. It is **not** a solver-quality comparison with the upstream prototype.

| Scenario | Comparison | Baseline kind | Matched p | Current wall (s) | Baseline wall (s) | Baseline max \|Δmean\| |
| --- | --- | --- | --- | --- | --- | --- |
| original-compat-tiny | grid-vs-bruteforce | none | 2 | 0.004 |  |  |
| original-compat-tiny | current-vs-baseline | original | 14 | 0.009 | 0.001 | 0.3464 |

## Local validation checklist

```bash
cmake -S . -B build-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DALDOUS_TSP_ENABLE_PYTHON_TESTS=ON
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

Recommended debug/sanitizer pass:

```bash
cmake -S . -B build-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON \
  -DALDOUS_TSP_ENABLE_SANITIZERS=ON \
  -DALDOUS_TSP_ENABLE_PYTHON_TESTS=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
```

Package-consumer smoke:

```bash
cmake --install build-release --prefix install-root
cmake -S tests/consumer -B build-consumer -G Ninja \
  -DCMAKE_PREFIX_PATH=$PWD/install-root
cmake --build build-consumer --parallel
```

## Reusable benchmark commands

Backend parity:

```bash
python3 scripts/benchmark_parity.py \
  --exe build-release/aldous_tsp \
  --out-dir parity-current
```

Full bundled benchmark suite:

```bash
python3 scripts/benchmark.py \
  --exe build-release/aldous_tsp \
  --suite all \
  --threads 1 \
  --out-dir benchmark-all
```

Original-vs-current parity requires a compiled original prototype executable:

```bash
python3 scripts/benchmark_parity.py \
  --exe build-release/aldous_tsp \
  --baseline-exe /path/to/original/aldous_tsp \
  --baseline-kind original \
  --out-dir parity-original
```

Real oracle smoke requires LKH and/or Concorde installed locally:

```bash
python3 scripts/oracle_real_smoke.py \
  --exe build-release/aldous_tsp \
  --out-dir real-oracle-smoke
```
