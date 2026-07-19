# Architecture

The project is organized as a reusable C++ library with a thin command-line application on top.

```text
apps/aldous_tsp_main.cpp
  └── aldous_tsp::cli_main
       └── aldous_tsp::ExperimentRunner
            ├── Instance / exact KNN backend
            ├── TspSolver
            │    ├── tour construction
            │    ├── candidate 2-opt
            │    └── Or-opt-1
            ├── SubsetSolver
            │    ├── warm starts and dense/small-p seeds
            │    ├── simulated annealing
            │    ├── subset-swap descent
            │    ├── pair exchange and ruin/recreate LNS
            │    ├── high-p deletion exchange
            │    └── path relinking
            ├── optional LKH/Concorde oracle polishing
            └── ResultsDocument / strict JSON schema
```

## Public API

Public headers live under `include/aldous_tsp/`. The most important types are:

- `Instance`: point set, bounds, KNN data, and distance helpers.
- `Tour`: mutable cycle with index/membership/edge-cache invariants.
- `TspSolver`: object-oriented facade for full-TSP solves.
- `SubsetSolver`: object-oriented facade for fixed-size subset solves.
- `ExperimentRunner`: object-oriented facade for Monte Carlo curve estimation.
- `ResultsDocument`: schema-backed result container.

## Internal implementation

Implementation files live under `src/`. The solver is split by responsibility:

- `solver_construction.cpp`: initial tour construction.
- `solver_local_search.cpp`: 2-opt, Or-opt, and polishing.
- `solver_neighborhoods.cpp`: subset exchange, LNS, and relinking neighborhoods.
- `solver_seeds.cpp`: small-p/high-p/warm-start seed generation.
- `solver_subset.cpp`: subset-solver orchestration.
- `solver_tsp.cpp`: full-TSP orchestration.
- `solver_facade.cpp` and `experiment.cpp`: public object-oriented facade implementations.

The low-level search kernels are intentionally free functions: they are stateless algorithms over domain objects and are easier to profile and optimize in this form. The public facades provide the object-oriented API expected by application code.

## CLI boundary

The CLI is deliberately thin. It parses arguments, validates configuration, builds an oracle context, then delegates execution to `ExperimentRunner`. This keeps the experiment engine reusable from C++ code without depending on command-line parsing.
