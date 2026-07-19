# Known-good benchmark and validation commands

This page combines reusable validation commands with compact tables rendered from the bundled `validation_runs/` manifests. Regenerate it with:

```bash
python3 scripts/render_validation_report.py --validation-dir validation_runs --output docs/known_good_benchmarks.md
```

The bundled artifacts are smoke/validation runs, not large Monte Carlo evidence. Use them to check release health, deterministic parity, and output-schema stability before running larger experiments.

## Bundled validation summary

| Check | Observed result |
| --- | --- |
| Release/Python CTest | passed 11/11 |
| Backend parity | max_abs_mean_delta = 0.0 |
| Benchmark suite | 9 scenarios, total wall 12.7343s |
| Real oracle smoke | skipped (No real LKH or Concorde executable found) |
| Original-compatible routing | fake original CLI routing passed; not a solver-quality parity result |

## Backend parity manifest

| Scenario | Comparison | Matched p | Max \|Δmean\| | Grid wall (s) | Brute wall (s) |
| --- | --- | --- | --- | --- | --- |
| tiny-grid-parity | grid-vs-bruteforce | 3 | 0 | 0.1375 | 0.1406 |
| small-hybrid-parity | grid-vs-bruteforce | 4 | 0 | 0.0081 | 0.0089 |

## Benchmark suite manifest excerpt

| Scenario | Suite | N | Instances | Mode | Wall (s) | Best p | Best mean | 2-opt imp | High-p imp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny-balanced | smoke | 60 | 2 | balanced | 0.7919 | 0.05 | 0.2408 | 394 | 9 |
| small-hybrid | smoke | 120 | 2 | hybrid | 0.7548 | 0.02 | 0.256 | 1551 | 22 |
| medium-balanced | larger | 180 | 2 | balanced | 1.941 | 0.02 | 0.2704 | 2869 | 23 |
| medium-hybrid | larger | 180 | 2 | hybrid | 3.1904 | 0.02 | 0.2704 | 3219 | 23 |
| ablation-baseline | ablation | 140 | 2 | hybrid | 1.423 | 0.05 | 0.4842 | 2186 | 12 |
| ablation-no-two-opt | ablation | 140 | 2 | hybrid | 1.0711 | 0.05 | 0.4696 | 0 | 12 |
| ablation-no-or-opt | ablation | 140 | 2 | hybrid | 0.996 | 0.05 | 0.4842 | 2208 | 12 |
| ablation-no-lns | ablation | 140 | 2 | hybrid | 1.0697 | 0.05 | 0.4842 | 2234 | 12 |
| ablation-no-relink | ablation | 140 | 2 | hybrid | 1.4963 | 0.05 | 0.4842 | 2142 | 12 |

## Ablation manifest excerpt

| Scenario | Disabled features | Wall (s) | Best mean | 2-opt imp | Or-opt imp | Pair imp | LNS imp | Relink feasible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ablation-baseline | baseline | 1.423 | 0.4842 | 2186 | 298 | 20 | 31 | 24 |
| ablation-no-two-opt | --disable-two-opt | 1.0711 | 0.4696 | 0 | 789 | 19 | 38 | 24 |
| ablation-no-or-opt | --disable-or-opt | 0.996 | 0.4842 | 2208 | 0 | 21 | 39 | 24 |
| ablation-no-lns | --disable-pair-exchange --disable-ruin-recreate | 1.0697 | 0.4842 | 2234 | 298 | 0 | 0 | 24 |
| ablation-no-relink | --disable-path-relink | 1.4963 | 0.4842 | 2142 | 278 | 19 | 36 | 0 |

## Exhaustive 2-opt policy comparison

| Policy | Wall (s) | Best mean | 2-opt scans | 2-opt imp | Or-opt scans | TSP s | Subset s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final-only | 0.0628 | 0.8427 | 4679 | 19 | 27038 | 0.0011 | 0.0284 |
| all-polish | 0.0273 | 0.8427 | 101578 | 53 | 70128 | 9.401e-04 | 0.025 |

`final-only` reduced two-opt scans by `96899` relative to `all-polish` in the bundled validation run.

## Original-compatible parity routing

This table validates command routing only. It uses `tests/fake_original_cli.py` to ensure current-only flags do not leak into an original-style baseline command. It is **not** a solver-quality comparison with the upstream prototype.

| Scenario | Comparison | Baseline kind | Matched p | Current wall (s) | Baseline wall (s) | Baseline max \|Δmean\| |
| --- | --- | --- | --- | --- | --- | --- |
| original-compat-tiny | grid-vs-bruteforce | none | 2 | 0.0043 |  |  |
| original-compat-tiny | current-vs-baseline | original | 14 | 0.0144 | 0.001 | 0.3464 |

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
