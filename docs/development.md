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

Dedicated libFuzzer targets and a scheduled ASan/UBSan campaign cover the CLI parser, instance/KNN construction, mutable tours, and bounded solver execution. Larger scenario benchmarks and per-oracle-call golden fixtures remain useful additions.


## Generated option surfaces

`config/options.json` is the authoritative source for option fields, defaults,
CLI spellings and aliases, scalar constraints, generated help, JSON
configuration keys, the strict current result-schema `config` object, and the
generated option reference. After changing it, run:

```bash
python3 scripts/generate_options.py
python3 scripts/generate_options.py --check
```

Do not edit files under `include/aldous_tsp/generated/`,
`src/generated_options_*.cpp`, or `docs/generated/options.md` by hand.
Cross-field and instance-dependent rules remain in `src/validation.cpp`; their
field-local ranges are generated from the same metadata.

## Modular source layout

The core implementation is deliberately split at algorithmic boundaries:

- `src/solver_moves.cpp` and `src/solver_spatial.cpp`: shared move scoring,
  insertion, candidate-table, and spatial-index kernels.
- `src/exact_subset.cpp`: bounded global subset-and-tour dynamic program.
- `src/solver_construction.cpp`: tour construction and exact-small routines.
- `src/solver_local_search.cpp`: 2-opt, Or-opt, and polishing.
- `src/solver_exchange.cpp`: one-for-one and high-p membership exchange.
- `src/solver_pair_exchange.cpp`: two-for-two exchange.
- `src/solver_lns.cpp`: adaptive ruin/recreate.
- `src/solver_ejection_chain.cpp`: variable-depth membership chains.
- `src/solver_path_relink.cpp`: exact relinking steps.
- `src/solver_seeds.cpp`: small-p, high-p, dense, and continuation seeds.
- `src/solver_subset.cpp` and `src/solver_tsp.cpp`: search controllers.
- `src/results.cpp` and `src/results_io.cpp`: JSON construction and durable
  atomic output, respectively.
- `src/cli_parse.cpp`, `src/cli_values.cpp`, `src/cli_summary.cpp`,
  `src/cli_self_test.cpp`, and `src/cli_main.cpp`: thin CLI layers.

The former `tests/test_core.cpp` is registered through `tests/test_main.cpp`
and split into six source-oriented suites. CTest registers each suite
separately, giving failures and reviews a bounded ownership surface.

JSON output still has no third-party runtime dependency. `src/json_writer.hpp`
centralizes escaping and numeric/array emission.


See `docs/known_good_benchmarks.md` for the compact release-validation checklist and known-good smoke commands.

## Architecture and API boundaries

See `docs/architecture.md` for the library/CLI split. Public headers belong under `include/aldous_tsp/`; executable entry points belong under `apps/`; implementation details belong under `src/`. Prefer adding user-facing workflows through `TspSolver`, `SubsetSolver`, or `ExperimentRunner` rather than expanding the CLI internals.
