# Development

## Build presets

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release

cmake --preset release-regression
# Optional constrained-builder profile
cmake --preset release-low-memory
cmake --build --preset release-regression
ctest --preset release-regression

cmake --preset debug-asan
cmake --build --preset debug-asan
ctest --preset debug-asan
```

## CMake options

- `ALDOUS_TSP_BUILD_CLI`
- `ALDOUS_TSP_BUILD_TESTS`
- `ALDOUS_TSP_ENABLE_WARNINGS`
- `ALDOUS_TSP_ENABLE_WERROR`
- `ALDOUS_TSP_ENABLE_NATIVE`
- `ALDOUS_TSP_ENABLE_SANITIZERS`
- `ALDOUS_TSP_ENABLE_CLANG_TIDY`
- `ALDOUS_TSP_ENABLE_PYTHON_TESTS`
- `ALDOUS_TSP_LOW_MEMORY_BUILD` — opt into low-optimization source-file overrides for constrained builders. Keep this `OFF` for normal performance builds.

## Testing strategy

Current tests cover:

- RNG determinism,
- KNN correctness across random and duplicate-point cases,
- tour invariants,
- exact small TSP,
- exhaustive differential checks for the global exact subset oracle in open
  and periodic geometry,
- 2-opt crossing removal,
- collision-safe elite deduplication,
- solver smoke tests,
- JSON atomic writing,
- CLI self-test and quick smoke via CTest,
- Python-driven backend parity, CLI regression, and schema-validation tests when `ALDOUS_TSP_ENABLE_PYTHON_TESTS=ON`.

Next testing improvements should add larger scenario benchmarks and per-oracle-call golden fixtures.


## P3 modular source layout

The core implementation is deliberately modularized:

- `src/solver_common.cpp` contains shared scoring/stat helpers.
- `src/exact_subset.cpp` contains the bounded global subset-and-tour dynamic
  program exposed through `include/aldous_tsp/exact_subset.hpp`.
- `src/solver_construction.cpp` contains tour construction and exact-small routines.
- `src/solver_local_search.cpp` contains 2-opt and Or-opt logic.
- `src/solver_neighborhoods.cpp` contains subset exchange, LNS, and path-relink neighborhoods.
- `src/solver_seeds.cpp` contains small-p and high-p seed generation.
- `src/solver_subset.cpp` and `src/solver_tsp.cpp` coordinate subset and full-TSP solves.
- `src/cli_parse.cpp`, `src/cli_self_test.cpp`, and `src/cli_main.cpp` split CLI parsing, self-tests, and orchestration; reusable experiment execution lives in `aldous_tsp::ExperimentRunner`.

JSON output still has no third-party runtime dependency. `src/json_writer.hpp` centralizes escaping and numeric/array emission so future schema changes do not duplicate formatting logic.


See `docs/known_good_benchmarks.md` for the compact release-validation checklist and known-good smoke commands.

## Architecture and API boundaries

See `docs/architecture.md` for the library/CLI split. Public headers belong under `include/aldous_tsp/`; executable entry points belong under `apps/`; implementation details belong under `src/`. Prefer adding user-facing workflows through `TspSolver`, `SubsetSolver`, or `ExperimentRunner` rather than expanding the CLI internals.
