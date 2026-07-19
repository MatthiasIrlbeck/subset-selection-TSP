# Benchmarking protocol

Benchmarking should compare both solution quality and runtime. Always record the full command line, p-grid, seed, compiler, KNN backend, oracle mode, and JSON schema version.

See `docs/known_good_benchmarks.md` for the compact release-validation checklist and known-good smoke commands.

## Quick benchmark runner

A small wrapper is provided:

```bash
python3 scripts/benchmark.py --exe build/aldous_tsp --out-dir benchmark-runs --threads 4
```

The script runs a small default scenario set and writes move counters, high-p exchange counters, KNN/TSP/subset phase timings, oracle counters, and solution-quality summaries.

```text
benchmark-runs/<scenario>.json
benchmark-runs/manifest.csv
benchmark-runs/manifest.json
```

Extra CLI arguments can be appended after `--extra`, for example:

```bash
python3 scripts/benchmark.py --exe build/aldous_tsp --out-dir benchmark-oracle --threads 4 --extra --oracle auto --oracle-time-limit 10
```


## Phase/counter profiling

Before using heavier profilers, run the lightweight profiling wrapper:

```bash
python3 scripts/profile_run.py --exe build/aldous_tsp --N 240 --instances 2 --mode hybrid
```

It runs one scenario, then prints wall time, KNN build time, TSP time, subset time, oracle time, and the local-search/neighborhood counters. Use this to decide whether to profile KNN construction, 2-opt, subset exchange, LNS, path relinking, or oracle execution.


## Exhaustive 2-opt policy comparison

The default exhaustive 2-opt policy is `final-only`: regular search polishing uses candidate 2-opt, and exhaustive 2-opt is reserved for the final reporting polish when `k <= --final-exhaustive-k`. The legacy behavior can be reproduced with `--exhaustive-two-opt-policy all-polish`. Compare both policies with:

```bash
python3 scripts/benchmark_exhaustive_policy.py \
  --exe build/aldous_tsp \
  --out-dir exhaustive-policy-benchmark \
  --N 240 \
  --instances 2 \
  --p-values 0.05,0.10,0.25,0.50,1.0
```

The script writes `manifest.csv` and reports wall time, best mean, and two-opt scan deltas for `final-only` versus `all-polish`. Use this before changing the exhaustive policy or threshold.

## Optional real-oracle smoke

Fake-oracle tests are part of CTest. To exercise real external solvers installed on a machine, run:

```bash
python3 scripts/oracle_real_smoke.py --exe build/aldous_tsp --out-dir real-oracle-smoke
```

The script detects `LKH` and/or `concorde` unless explicit paths are supplied. It exits successfully with a skip message when no real solver is available; pass `--require` to make missing solvers a failure.

## Manual baseline

```bash
./build/aldous_tsp \
  --mode hybrid \
  --N 1000 \
  --instances 12 \
  --threads 12 \
  --p-values 0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.30,0.50,0.80,1.00 \
  --restarts 3 \
  --sa-iters 60000 \
  --tsp-restarts 5 \
  --tsp-ils 800 \
  --output results-hybrid.json \
  --force
```

## Benchmark suites

The benchmark wrapper supports `--suite smoke`, `--suite standard`, `--suite larger`, `--suite ablation`, and `--suite all`. The ablation suite writes both `ablation_manifest.csv` and `ablation_manifest.json` in addition to the overall manifest.

## Ablation examples

```bash
./build/aldous_tsp --quick --disable-smallp-seeds --output no-smallp.json --force
./build/aldous_tsp --quick --disable-highp-delete --output no-highp.json --force
./build/aldous_tsp --quick --disable-path-relink --output no-relink.json --force
./build/aldous_tsp --quick --disable-ruin-recreate --output no-lns.json --force
```

## Oracle comparison

Run internal and oracle-enabled commands with identical seeds and p-grids:

```bash
./build/aldous_tsp --N 500 --instances 5 --p-values 0.2,0.5,1.0 --seed 2024 --output internal.json --force
./build/aldous_tsp --N 500 --instances 5 --p-values 0.2,0.5,1.0 --seed 2024 --oracle auto --oracle-time-limit 10 --output oracle.json --force
```

Compare `summary`, `search_stats.oracle_*`, and wall time. Treat external oracle results as a separate experimental condition, not as the same solver configuration.

## Original-vs-current parity

Use `scripts/benchmark_parity.py` with `--baseline-exe` to compare this repository against an already built original prototype executable:

```bash
python3 scripts/benchmark_parity.py \
  --exe build/aldous_tsp \
  --baseline-exe /path/to/original/aldous_tsp \
  --baseline-kind original \
  --out-dir parity-original
```

Without `--baseline-exe`, the same script runs deterministic grid-vs-brute-force backend parity scenarios for the current executable. The packaging validation ran this internal parity mode successfully with zero mean-delta between the two exact KNN backends.

The harness has two baseline modes:

- `--baseline-kind current` compares against an executable with this repository's modern CLI.
- `--baseline-kind original` compares against the upstream original prototype CLI. This mode intentionally omits current-only flags such as `--p-values`, `--knn-backend`, `--final-exhaustive-k`, and neighborhood ablation flags. The paired current run also uses original-compatible defaults so the common p-grid can be compared.
- `--baseline-kind auto` runs `baseline --help` and chooses `current` only if modern flags are advertised; otherwise it falls back to `original`.

### Original-prototype parity status

The original-vs-current benchmark is intentionally explicit: it requires a compiled original prototype executable. The packaging container attempted to clone the upstream repository but could not resolve `github.com`, so no real original binary was available there. The package therefore includes:

1. completed current grid-vs-brute-force backend parity,
2. an original-compatible parity harness,
3. a CTest compatibility smoke using `tests/fake_original_cli.py`, which fails if current-only flags are passed to an original-style baseline, and
4. a reproducible command for the real original executable once available.

| comparison | status | result |
| --- | --- | --- |
| current grid vs current brute-force KNN | completed in packaging validation | `max_abs_mean_delta = 0.0` on bundled small deterministic scenarios |
| parity harness vs fake original-style CLI | completed in CTest | verifies original-compatible command construction |
| current grid vs real original prototype | pending external baseline executable | run `scripts/benchmark_parity.py --baseline-exe /path/to/original/aldous_tsp --baseline-kind original` |

When real original parity is run, archive `parity-original/parity_manifest.csv`, `parity-original/parity_manifest.json`, the scenario JSON files, compiler details, and both executable versions.


## Bundled validation benchmark size

The built-in benchmark scenarios are intentionally validation-sized so CI and source-package checks complete quickly. For publication-scale Monte Carlo evidence, copy or extend `scripts/benchmark.py` scenarios and run with larger `N`, more instances, and higher restart/iteration budgets.
