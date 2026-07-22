# Aldous subset-selection TSP solver

This repository estimates the curve in David Aldous's subset-selection traveling-salesperson problem. Given `N` random points in a square of area `N`, the solver estimates the normalized cycle length `L(k) / k` for `k = pN` across a grid of subset fractions `p`.

The project is organized as a reusable C++17 library plus a thin command-line application. It keeps the high-performance heuristic layers from the earlier prototype while presenting a cleaner public API, installable CMake package, and validation workflow:

- CMake core library + CLI split
- conventional public-header/implementation split: `include/aldous_tsp/` for public API, `src/` for library implementation, and `apps/` for executables
- namespaced public API under `aldous_tsp::`
- object-oriented facade classes (`TspSolver`, `SubsetSolver`, and `ExperimentRunner`) over performance-oriented search kernels
- deterministic xoshiro/splitmix RNG streams
- exact grid KNN backend over arbitrary finite coordinate bounds, with brute-force backend and sampled verification hooks
- reverse-KNN wakeups in candidate-set 2-opt with don't-look bits
- batched distance scoring, including an AVX2 kernel behind `ALDOUS_TSP_ENABLE_NATIVE` (off by default) and a scalar fallback; the native build was measured *slower* than the default build on an AVX-512 host with GCC (downclocking/codegen effects), so benchmark on your target before enabling it
- incremental tour mutation operations for 2-opt, node moves, and subset swaps
- collision-safe elite-pool deduplication using canonical keys
- full-TSP candidate screening with scalable nearest-neighbor starts, optional farthest insertion, deterministic diversity-aware promotion, parallel ILS, 3-cut perturbations, 2-opt, and Or-opt-1
- subset simulated annealing with KNN-guided swap candidates
- optional global exact cardinality-`k` subset-and-tour dynamic program for `N <= 18`, disabled by default
- small-p spatial/dense seed pools
- high-p deletion seeds and high-p reference-guided exchange descent
- a staged subset-search funnel, deterministic subset swap descent, two-for-two pair exchange, ruin/recreate LNS, ejection chains, and budgeted elite path relinking
- fixed simulated-annealing temperature schedule, plus opt-in held-out p-aware multiple-candidate search presets
- optional exhaustive final 2-opt threshold for small tours; by default exhaustive 2-opt is final-only rather than used in every polishing pass
- configurable p-grid via `--p-values`, `--p-range`, or `--p-file`
- locale-independent UTF-8-safe JSON output with atomic no-clobber/replace policies, explicit durability status, schema-versioned metadata, stable campaign/replicate identities, and search statistics
- CTest unit tests, CLI smoke tests, Python regression tests, sanitizer-compatible build, and strict JSON schema validation
- plotting utility and CSV summary export
- optional external LKH/Concorde oracle post-processing with timeout, tour validation, fake-oracle tests, and top-N elite polishing
- exact ablation flags for solver-controlled local-search components
- safe forced grid-cell handling for exact grid KNN
- optional original-vs-current parity harness via `scripts/benchmark_parity.py --baseline-exe`, including original-CLI compatibility mode
- build metadata in JSON, including build type, configured flags, CMake/compiler details, CPU model, and hardware threads
- benchmark scenarios for smoke, larger, and ablation suites with manifest CSV output
- split path-relink counters for attempts, feasible candidates, elite insertions, and best-solution improvements
- per-oracle-call diagnostics in result JSON, including status, gain, runtime, and failure reason
- normal Release builds are optimized by default; constrained builders can opt into `-DALDOUS_TSP_LOW_MEMORY_BUILD=ON`
- array-form `summary_rows` result output in addition to the legacy p-keyed `summary` map
- phase-level profiling support through `knn_build_seconds` and `scripts/profile_run.py`
- optional real LKH/Concorde smoke script through `scripts/oracle_real_smoke.py`, with explicit skip/pass/fail manifests

The production solver is heuristic by default. For calibration and regression work,
`--exact-subset-max-n` can instead prove the globally optimal size-`k` subset and
its cycle for instances with `N <= 18`. The ordinary small-tour Held-Karp routine
optimizes only the ordering of a fixed subset and is not a subset-optimality proof.

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Or use presets:

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Sanitizers:

```bash
cmake --preset debug-asan
cmake --build --preset debug-asan
ctest --preset debug-asan
```

## Run

Fast smoke run:

```bash
./build/aldous_tsp --quick --output results.json --force
```

Custom run:

```bash
./build/aldous_tsp \
  --mode hybrid \
  --N 1000 \
  --instances 12 \
  --threads 12 \
  --p-values 0.02,0.03,0.05,0.10,0.20,0.50,0.80,1.00 \
  --restarts 3 \
  --sa-iters 60000 \
  --tsp-restarts 5 \
  --tsp-ils 800 \
  --knn-backend grid \
  --output results.json \
  --force
```

Dry-run resolved configuration:

```bash
./build/aldous_tsp --N 500 --p-range 0.02:1.0:12 --dry-run
```

The published 0.10 controller remains the default. Evidence-backed search presets are opt-in:

```bash
# Matched-compute four-candidate SA through p=0.35
./build/aldous_tsp --search-policy heldout-balanced ...

# Deeper subset policy, periodic extension through p=0.50, and stronger p=1 search
./build/aldous_tsp --search-policy heldout-quality ...
```

See [`docs/heldout_search_policy.md`](docs/heldout_search_policy.md) for activation rules, held-out evidence, and interpretation limits.

`--quick` is a preset applied before explicit user options, so overrides work regardless of order:

```bash
./build/aldous_tsp --quick --N 100 --instances 1 --dry-run
./build/aldous_tsp --N 100 --instances 1 --quick --dry-run   # same resolved N/instances
```

Plot results:

```bash
python3 scripts/plot_results.py results.json -o curve.png --csv summary.csv

# Lightweight phase/counter profiling
python3 scripts/profile_run.py --exe build/aldous_tsp --N 240 --instances 2
```

## CLI reference

Important flags:

- `--quick[=true|false]`
- `--dry-run[=true|false]`
- `--force[=true|false]`
- `--mode balanced|smallp-region|highp-delete|hybrid`
- `--N <int>`
- `--instances <int>`
- `--threads <int>` (`0` means auto; negative values are rejected)
- `--seed <int>`
- `--p-values <csv>`
- `--p-range <start:end:count>`
- `--p-file <file>`
- `--search-policy legacy-balanced|heldout-balanced|heldout-quality`
- `--restarts <int>`
- `--sa-iters <int>`
- `--sa-iters-per-k <int>` (extra SA iterations per subset element; effective budget is `sa_iters + sa_iters_per_k * k`)
- `--time-budget-per-p <seconds>` (anytime mode: runs at least the configured restarts, then keeps launching restarts until the wall-clock budget per `(instance, p)` solve elapses; `0` disables)
- `--restart-threads <int>` (parallel subset restarts within one `(instance, p)` solve; `0` = auto from the leftover thread budget, `1` = sequential; results are invariant to this value outside time-budget mode)
- `--second-sweep[=bool]` (after the descending warm-start sweep over `p`, run an ascending sweep seeded from the grown solution at the next smaller `p` and keep the better result per `p`)
- `--tsp-restarts <int>`
- `--tsp-ils <int>`
- `--tsp-patience <int>`
- `--knn <int>`
- `--knn-backend grid|bruteforce`
- `--grid-cell <float>`
- `--verify-knn <int>`
- `--exact-subset-max-n <int>` (`0` disables; values through the hard cap of `18` globally solve both subset choice and cycle)
- `--final-exhaustive-k <int>`
- `--exhaustive-two-opt-policy never|final-only|all-polish`
- `--subset-swap-passes <int>`
- `--pair-exchange-passes <int>`
- `--pair-exchange-max-k <int>` (`5000` by default; `0` removes the large-`k` safety gate)
- `--ruin-recreate-rounds <int>`
- `--adaptive-ruin-recreate[=true|false]`
- `--ruin-recreate-max-fraction <float>`
- `--ruin-recreate-max-nodes <int>` (`0` removes the absolute cap)
- `--ruin-recreate-pool-cap <int>`
- `--ejection-chain-starts <int>`
- `--ejection-chain-depth <int>`
- `--ejection-chain-candidates <int>`
- `--ejection-chain-remove-cap <int>` (`0` considers every eligible member)
- `--ejection-chain-max-uphill <float>`
- `--elite-diversity-slots <int>`
- `--elite-min-jaccard <float>`
- `--elite-quality-slack <float>`
- `--path-relink-top <int>`
- `--disable-two-opt[=true|false]`
- `--disable-or-opt[=true|false]`
- `--disable-subset-swap[=true|false]`
- `--disable-pair-exchange[=true|false]`
- `--disable-ruin-recreate[=true|false]`
- `--disable-ejection-chain[=true|false]`
- `--disable-path-relink[=true|false]`
- `--disable-smallp-seeds[=true|false]`
- `--disable-highp-delete[=true|false]`
- `--oracle none|auto|lkh|concorde`
- `--oracle-format matrix|euc2d`
- `--lkh-path <path>`
- `--concorde-path <path>`
- `--oracle-time-limit <int>`
- `--oracle-inline-feedback`

External post-processing is available through `--oracle auto`, `--oracle lkh`, or `--oracle concorde`. It is disabled by default with `--oracle none`. Boolean flags accept plain presence as true or explicit `=true`/`=false` values (`yes/no`, `on/off`, and `1/0` are also accepted). See `docs/oracles.md` for setup, TSPLIB format choices, and caveats.

Campaign inference supports replicate-block bootstrap, finite-size model envelopes, deletion/range sensitivity, nested search-seed variance, and paired multifidelity correction. See [`docs/campaign_analysis.md`](docs/campaign_analysis.md).


## Design style

The hardened public solve path uses an immutable `PreparedInstance`. Build one
with `InstanceBuilder`, which validates the numerical metric domain and creates
the exact KNN/grid representation before returning an object that exposes only
const state. The older mutable `Instance` surface remains source-compatible;
passing it to a solver performs a complete checked canonical conversion first.
The hottest local-search operators are still implemented as free functions over
the internal const instance view. That keeps stateless numerical kernels easy to
test and profile without allowing application-owned buffers into search code.

For application code, use the facade classes:

```cpp
aldous_tsp::SolverOptions options;
aldous_tsp::PreparedInstance instance = aldous_tsp::InstanceBuilder()
    .generate(120, rng)
    .build(32, aldous_tsp::KnnBackend::GridExact);
aldous_tsp::TspSolver tsp_solver(options);
auto tsp = tsp_solver.solve(instance, rng);

aldous_tsp::SubsetSolver subset_solver(options);
auto subset = subset_solver.solve_with_warm_start(instance, 40, rng, tsp.tour.nodes);

aldous_tsp::ExperimentRunner runner(run_options);
auto results = runner.run();
```

For direct exact calibration on a small instance, include
`<aldous_tsp/exact_subset.hpp>` and call:

```cpp
auto proof = aldous_tsp::exact_subset_cycle(instance, k);
if (proof.proven_optimal) {
    // proof.cycle globally minimizes the implemented metric over all
    // cardinality-k subsets, and proof.length is its optimal cycle length.
}
```

The direct exact API does not require KNN construction. Instances above the hard
limit return `solved == false` without allocating the exponential DP table. The
heuristic solver and Held–Karp APIs accept `PreparedInstance`; compatibility
overloads taking mutable `Instance` validate and canonicalize before execution.

See `examples/library_usage.cpp` for a complete library-use example.

## Project structure

The implementation is split into focused solver and CLI translation units. The public API remains in `include/aldous_tsp/`, while `src/solver_*.cpp` covers construction, local search, neighborhoods, seeding, subset orchestration, and TSP orchestration; `src/cli_*.cpp` covers parsing, self-tests, and command-line coordination; `apps/` contains executable entry points.


```text
include/aldous_tsp/      public library headers
src/                     core library and CLI implementation
apps/                    command-line executable entry points
tests/                   CTest unit tests
scripts/                 plotting, profiling, benchmarking, and oracle smoke utilities
docs/                    algorithm, reproducibility, development, and validation notes
schema/                  JSON schema for results
examples/                example p-grid files
.github/workflows/       CI
```

See `docs/known_good_benchmarks.md` for validation commands that have been run successfully and for manifest-derived benchmark tables. Regenerate that page with `scripts/render_validation_report.py`. This package includes current-version generated validation manifests and JSON outputs under `validation_runs/`; they are summarized in `docs/known_good_benchmarks.md` and checked by the optional Python CTest suite.

## License

This package is distributed under the MIT License. See `LICENSE`.

## Development workflow

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug

# Include Python-driven regression and schema-validation tests
cmake --preset release-regression
cmake --build --preset release-regression
ctest --preset release-regression
```

Recommended local checks:

```bash
cmake --preset debug-asan
cmake --build --preset debug-asan
ctest --preset debug-asan
python3 scripts/plot_results.py build/quick-smoke.json -o build/quick-smoke.png || true
```

## Current limitations

The internal heuristic stack and optional external oracle path are now integrated in the cleaner structure. Research-grade claims still require benchmark comparisons, larger Monte Carlo sample sizes, and sensitivity/ablation analysis. Per-oracle-call diagnostics are serialized, but research-grade oracle comparisons still need real LKH/Concorde runs on the target machine.


### Reproducibility metadata

Result JSON records build flags, target compile options, effective KNN backend/cell-size telemetry, and optionally per-instance rows via `--include-instance-rows`.

Historical schema-13 result files can first be converted to frozen schema 14 with `scripts/migrate_schema13_to14.py`, then to current schema 15 with `scripts/migrate_schema14_to15.py`. Migration records every inferred or unrecoverable field and preserves the complete step history; it does not invent missing replicate identities.

Large campaigns can set `--memory-budget-mb` to cap active instance workers before allocation. Result JSON records the conservative per-instance/peak estimate and requested, resolved, and effective concurrency under `memory_plan`. Reverse-KNN wakeup adjacency is created lazily and can be disabled with `--reverse-knn=false` when memory is more valuable than wakeup acceleration.
