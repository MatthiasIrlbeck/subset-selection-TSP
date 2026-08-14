# Architecture

The project is organized as a reusable C++ library with a thin command-line application on top.

```text
apps/aldous_tsp_main.cpp
  └── aldous_tsp::cli_main
       └── aldous_tsp::ExperimentRunner
            ├── InstanceBuilder → immutable PreparedInstance / exact KNN backend
            ├── optional exact cardinality-k subset DP (N <= 18)
            ├── TspSolver
            │    ├── tour construction
            │    ├── candidate 2-opt
            │    └── Or-opt-1
            ├── SubsetSolver
            │    ├── warm starts and dense/small-p seeds
            │    ├── simulated annealing
            │    ├── subset-swap descent
            │    ├── pair exchange and ruin/recreate LNS
            │    ├── membership ejection chains
            │    ├── high-p deletion exchange
            │    └── path relinking
            ├── optional LKH/Concorde oracle polishing
            └── ResultsDocument / strict JSON schema
```

## Public API

Public headers live under `include/aldous_tsp/`. The most important types are:

- `PreparedInstance`: immutable, validated point set, bounds, KNN/grid data, and
  distance helpers accepted by hardened solver/lower-bound APIs.
- `InstanceBuilder`: preferred construction surface for generated or imported
  point sets.
- `Instance`: mutable compatibility/construction type; solver overloads taking
  it perform a checked canonical conversion before search.
- `Tour`: mutable cycle with index/membership/edge-cache invariants.
- `ExactSubsetSolution` and `exact_subset_cycle()`: global small-instance
  cardinality-`k` subset-and-tour proof API.
- `TspSolver`: object-oriented facade for full-TSP solves.
- `SubsetSolver`: object-oriented facade for fixed-size subset solves.
- `ExperimentRunner`: object-oriented facade for Monte Carlo curve estimation.
- `ResultsDocument`: schema-backed result container.

## Internal implementation

Implementation files live under `src/`. The solver is split by responsibility:

- `solver_construction.cpp`: initial tour construction.
- `exact_subset.cpp`: exponential global cardinality-`k` dynamic program for
  supported small instances.
- `solver_local_search.cpp`: 2-opt, Or-opt, and polishing.
- `solver_exchange.cpp` and `solver_pair_exchange.cpp`: exact membership exchanges.
- `solver_lns.cpp`: adaptive ruin/recreate neighborhoods.
- `solver_ejection_chain.cpp`: variable-depth membership chains.
- `solver_path_relink.cpp`: exact relinking steps.
- `solver_moves.cpp` and `solver_spatial.cpp`: shared move and spatial kernels.
- `solver_seeds.cpp`: small-p/high-p/warm-start seed generation.
- `solver_subset.cpp`: subset-solver orchestration.
- `solver_tsp.cpp`: full-TSP orchestration.
- `solver_facade.cpp` and `experiment.cpp`: public object-oriented facade implementations.

The low-level search kernels are intentionally free functions: they are stateless algorithms over domain objects and are easier to profile and optimize in this form. The public facades provide the object-oriented API expected by application code.

## Configuration and CLI boundary

`config/options.json` is the single metadata source for public option fields,
defaults, CLI parsing, generated help, field-local validation, JSON
configuration, schema entries, and the generated reference. Handwritten code
contains only cross-option and instance-dependent validation.

The CLI is deliberately thin. It applies generated parsing, invokes the shared
validation layer, builds an oracle context, then delegates execution to
`ExperimentRunner`. This keeps the experiment engine reusable from C++ code
without depending on command-line parsing.
