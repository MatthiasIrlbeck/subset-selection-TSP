# Public C++ API

The installed library is exposed through headers under `include/aldous_tsp/` and CMake target
`aldous_tsp::core`. The supported public path uses immutable prepared instances.

## Building a prepared instance

```cpp
#include <aldous_tsp/instance.hpp>
#include <aldous_tsp/rng.hpp>

aldous_tsp::Rng rng(12345);
auto instance = aldous_tsp::InstanceBuilder()
    .periodic(true)
    .generate(1000, rng)
    .build(40, aldous_tsp::KnnBackend::GridExact);
```

Preparation validates coordinate finiteness and the supported numerical metric domain,
normalizes periodic coordinates, builds exact KNN/grid structures, and checks deterministic
`(distance, node ID)` ordering. `PreparedInstance` exposes const state and is safe to share
between independent solves. The compatibility overload accepting mutable `Instance` performs a
complete checked preparation before solving; callers should migrate to `PreparedInstance`.

## Solving

```cpp
#include <aldous_tsp/solver.hpp>

aldous_tsp::SolverOptions options;
aldous_tsp::TspSolver tsp_solver(options);
auto tsp = tsp_solver.solve(instance, rng);

aldous_tsp::SubsetSolver subset_solver(options);
auto subset = subset_solver.solve(instance, 300, rng);
```

Successful solves guarantee an exact requested cardinality, unique in-range nodes, finite
nonnegative length, internally consistent membership/position indexes, and agreement between
incremental and fully recomputed cycle length. Invalid inputs and violated postconditions throw
standard exceptions; no public solve API returns a silent partial solution.

The production solver is heuristic. A successful heuristic result is not a global-optimality
certificate. Fixed-subset two-NN and Held--Karp diagnostics certify only tour ordering on the
selected subset.

## Exact small-instance calibration

```cpp
#include <aldous_tsp/exact_subset.hpp>

auto proof = aldous_tsp::exact_subset_cycle(instance, k);
if (proof.proven_optimal) {
    // proof.cycle and proof.length are globally optimal for this instance and k.
}
```

The exact cardinality-layered dynamic program is limited to the documented hard cap (`N <= 18`)
and can be disabled in experiment configuration. Use
`estimate_exact_subset_memory(n, k)` before scheduling many concurrent exact solves.

## Experiment runner and results

`ExperimentRunner` owns point generation, stable point/search streams, memory planning,
continuation sweeps, control-reference work, and aggregation. Results contain resolved
configuration, source/build provenance, timing phases, search statistics, campaign identities,
and strict schema-versioned output.

Result publication uses atomic replace/no-replace policies and explicit durability levels. On
POSIX, newly created output files are owner-readable/writable (`0600`) by design; copy or chmod
them explicitly when group sharing is required.

## Thread safety and determinism

- A `PreparedInstance` is immutable and may be read concurrently.
- Solver objects should be treated as independent per call unless their documentation says
  otherwise.
- Fixed-budget results are invariant to restart-thread count; elapsed-time modes are inherently
  schedule dependent.
- User callbacks run on the caller thread and may throw; worker failures are joined and
  rethrown through the public boundary.

## Installed CMake package

```cmake
find_package(aldous_tsp CONFIG REQUIRED)
add_executable(example main.cpp)
target_link_libraries(example PRIVATE aldous_tsp::core)
```

See `examples/library_usage.cpp`, `include/aldous_tsp/config.hpp`, and
`include/aldous_tsp/results.hpp` for the complete types. The generated CLI/default option
reference is in `docs/generated/options.md`.
