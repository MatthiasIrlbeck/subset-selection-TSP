# Aldous subset selection TSP

![CI](https://github.com/MatthiasIrlbeck/subset-selection-TSP/actions/workflows/ci.yml/badge.svg)

This project attacks an open traveling salesman problem (TSP) posed by David Aldous. Scatter $N$ points uniformly in a square of area $N$ and let $L(k)$ denote the length of the shortest cycle through exactly $k$ of them. The ratio $f(p) = E[L(pN)] / (pN)$ is believed to converge to a constant as $N$ grows, and at $p = 1$ the Beardwood-Halton-Hammersley (1959) theorem gives $f(1) \approx 0.7124$. Aldous asks for the shape of $f$ over the full interval $(0, 1]$: does it decrease monotonically? What is its limiting form?

The program estimates $f(p)$ by Monte Carlo simulation over a grid of $p$ values, solving the resulting combinatorial problems with heuristic local search. For background see [Aldous's problem page](https://www.stat.berkeley.edu/~aldous/Research/OP/simTSP.html).

## Algorithmic approach

### Overview

For each value of $p$, the program has to solve two linked problems.

First, it has to decide **which** $k = pN$ points should be visited.

Second, it has to decide **in what order** those selected points should be visited.

If $p = 1$, there is no subset choice and the task is an ordinary Euclidean traveling salesman problem on all $N$ points. If $p < 1$, the task is harder: the program must optimize both the subset and the tour through that subset.

The code estimates $f(p)$ by Monte Carlo simulation. It repeatedly generates a random point set, solves the optimization problem for each requested value of $p$, records the normalized tour length $L(k)/k$, and then averages those values over many independent instances.

The solver is mainly heuristic. It is designed to produce good solutions quickly and consistently over many runs, not to certify global optimality. For very small subset sizes, the code does switch to exact search, but the main regime is heuristic local search.

### Full TSP solver for $p = 1$

When $p = 1$, the only question is the visiting order.

The full tour solver uses **multiple restarts**. Each restart begins from a different initial tour. One start is built by **farthest insertion**: start with a far-apart pair of points, then repeatedly insert the point that is currently farthest from the tour in the cheapest place. Other starts are built by **nearest-neighbor construction**, which repeatedly appends a nearby unvisited point.

Each initial tour is then improved by local search.

The two main local moves are:

1. **2-opt**: remove two edges, reconnect the tour in the other possible way, and reverse the affected segment if that shortens the cycle.

2. **Or opt-1**: remove one point from the current tour and reinsert it somewhere else if that reduces the length.

In Euclidean TSP instances, good improving moves usually involve nearby points, so the code does not compare every pair of nodes against every other pair. Instead, it precomputes exact nearest-neighbor lists and restricts most local search to those candidates. This makes the search much faster while still keeping the important moves.

To escape local minima, the full-tour solver uses **iterated local search**. After a tour has been polished, the code cuts it at three positions, rearranges the resulting segments, and runs local search again. That perturbation is large enough to leave the current basin of attraction, but still structured enough that the next local search phase starts from a reasonable tour.

The best restart solutions are stored in a small **elite pool**, meaning a short list of the strongest tours found so far. Those tours are then polished again with slightly broader search settings before the best one is reported.

### Subset solver for $p < 1$

When $p < 1$, the problem is no longer just "find a good tour." It becomes "find a good set of $k$ points and a good tour through them."

That is the core of the project.

Each subset run starts from one or more seed solutions. A seed is simply an initial cycle on $k$ selected points. Seeds can come from several sources:

- good solutions from nearby values of $p$
- simple geometric constructions
- spatially localized candidate sets
- random samples

Using nearby $p$-values matters because the optimal subset usually changes gradually as $p$ changes. A strong solution at one $p$ is often a useful starting point for the next.

Once a seed has been built, the code runs **simulated annealing**. In simple terms, simulated annealing is a search process that usually accepts improving moves, but also occasionally accepts worsening moves early in the run. That makes it less likely to get trapped immediately in a bad local minimum.

The main subset move is a **swap move**:

- remove one point that is currently in the selected subset
- insert one point that is currently outside the subset
- reconnect the cycle in the cheapest available place

This keeps the subset size fixed at $k$ while allowing the selected set itself to evolve.

The subset solver also uses occasional **2-opt** moves inside the current cycle. That is important because even if the selected point set is good, the ordering of those points may still be poor.

At regular intervals, and again after annealing finishes, the solver switches from stochastic search to deterministic cleanup. It runs several improvement passes, including:

- 2-opt on the current cycle
- single node reinsertion
- one for one subset swap descent
- in larger cases, limited two for two subset exchanges

The two for two exchanges matter because some improvements cannot be found by replacing only one selected point at a time.

### Larger neighborhood improvement

Single swaps are useful, but they can still be too local. For that reason the solver also uses a larger repair step often called **ruin and recreate**.

The idea is simple:

1. remove a small group of points from the current cycle
2. build a restricted pool of plausible replacement candidates
3. rebuild that part of the solution
4. polish the repaired tour again

This gives the search a way to reorganize a bad region of the subset without throwing away the whole solution.

The solver also combines good solutions with each other. It keeps an elite pool of the best subsets found so far and applies two forms of recombination.

The first is **path relinking**. Starting from one strong subset, the code gradually changes it toward another strong subset and checks the intermediate solutions along the way.

The second is direct **recombination**. Here the code builds a child solution that inherits structure from two parent subsets, then re-optimizes that child.

These combination steps help the search reuse information from different restarts instead of treating each restart as completely independent.

### Special regimes

The main documented mode is `balanced`. This is the default end to end path for estimating the full $f(p)$ curve.

There are also specialized experimental regimes for the ends of the $p$-range.

For very small $p$, the solver can generate geometrically focused seeds that look for compact local regions before the main search begins. This is useful because when only a small fraction of the points must be visited, good solutions often come from dense local structure.

For large $p$, the solver can start from a full TSP tour and **delete** points that are cheap to remove, then improve the remaining cycle. This is a natural strategy near $p = 1$, because the selected subset is close to the full point set.


## Build

CMake build (recommended):
```bash
cmake -S . -B build
cmake --build build -j
```

Single-config generators default to Release if `CMAKE_BUILD_TYPE` is unset. Pass `-DALDOUS_TSP_ENABLE_NATIVE=ON` for `-march=native` tuning on the local machine.

Direct build (fallback):
```bash
g++ -O3 -std=c++17 -pthread src/main.cpp src/tsp_solver.cpp src/oracle.cpp \
    src/subset_seed.cpp src/subset_pool.cpp src/subset_smallp.cpp \
    src/subset_highp.cpp src/subset_search.cpp src/run_cli.cpp \
    src/run_cli_common.cpp src/run_cli_parse.cpp src/run_cli_report.cpp \
    src/run_cli_self_test.cpp src/run_cli_orchestrate.cpp \
    -Iinclude -o aldous_tsp
```

Run the self-test suite:
```bash
./build/aldous_tsp --self-test
```

## Run

```bash
./build/aldous_tsp --mode balanced --oracle none --quick       # small fast run (N=200, 3 instances)
./build/aldous_tsp --mode balanced --oracle none --N 2000 --instances 12 --threads 12
```

By default the program writes `results.json` in the current working directory. Existing files are preserved unless `--force` is passed. Use `--output <file>` to choose a different destination.

External solver use is optional. `--oracle none` is the default and keeps runs machine-independent. Use `--oracle auto`, `--oracle lkh`, or `--oracle concorde` when you want external post-processing.

## Modes

- `balanced` -- default mode for end-to-end curve runs.
- `smallp-region` -- experimental low-p search path.
- `highp-delete` -- experimental high-p deletion path.
- `hybrid` -- experimental combined path.

For comparisons and for the examples above, `balanced` is the recommended baseline.

## Design decisions

The solver has no external dependencies beyond a C++17 compiler and pthreads. Distances are computed from coordinates on the fly rather than stored in an $O(N^2)$ matrix, keeping memory linear in $N$ and allowing larger $N$ runs. KNN candidate sets are built over a uniform grid with expanding ring search rather than a k-d tree; for uniform random points in two dimensions the grid is simpler, cache friendlier, and competitive in practice. The RNG is xoshiro256. External solver integration (LKH, Concorde) is available as an optional post-processing step but is never required.

## Project structure

- `include/config.hpp` -- tuning constants and numerical tolerances
- `include/problem.hpp`, `include/tour.hpp`, `include/oracle.hpp` -- shared data structures and interfaces split by responsibility
- `include/core.hpp` -- umbrella include for the shared project types
- `include/tsp_solver.hpp` -- full-TSP solver interface
- `include/subset_solver.hpp` -- subset solver and oracle interface
- `include/run_cli.hpp` -- CLI entry declaration
- `src/tsp_solver.cpp` -- full-TSP construction and local search
- `src/oracle.cpp` -- external-solver integration and TSPLIB/process plumbing
- `src/subset_seed.cpp`, `src/subset_pool.cpp`, `src/subset_smallp.cpp`, `src/subset_highp.cpp`, `src/subset_search.cpp` -- subset construction, seed pools, regime-specific search, and local improvement
- `src/run_cli.cpp`, `src/run_cli_parse.cpp`, `src/run_cli_report.cpp`, `src/run_cli_self_test.cpp`, `src/run_cli_orchestrate.cpp`, `src/run_cli_common.cpp` -- CLI entry, argument parsing, reporting/JSON, self-tests, orchestration, and shared CLI helpers
- `src/main.cpp` -- entry point

## Example output

The `examples/` directory contains example results. 

## Plotting

A plotting script is included at `scripts/plot_results.py`. It reads the JSON output and produces a curve of estimated mean edge length against subset fraction. Requires matplotlib.

    python3 scripts/plot_results.py results.json -o curve.png
