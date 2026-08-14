# Aldous subset selection TSP

[![CI](https://github.com/MatthiasIrlbeck/subset-selection-TSP/actions/workflows/ci.yml/badge.svg)](https://github.com/MatthiasIrlbeck/subset-selection-TSP/actions/workflows/ci.yml)
[![Fuzzing](https://github.com/MatthiasIrlbeck/subset-selection-TSP/actions/workflows/fuzz.yml/badge.svg)](https://github.com/MatthiasIrlbeck/subset-selection-TSP/actions/workflows/fuzz.yml)
[![CodeQL](https://github.com/MatthiasIrlbeck/subset-selection-TSP/actions/workflows/codeql.yml/badge.svg)](https://github.com/MatthiasIrlbeck/subset-selection-TSP/actions/workflows/codeql.yml)
[![Latest release](https://img.shields.io/github/v/release/MatthiasIrlbeck/subset-selection-TSP)](https://github.com/MatthiasIrlbeck/subset-selection-TSP/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This project studies an open traveling salesman problem (TSP) posed by David Aldous. Scatter
$N$ points uniformly in a square of area $N$, and let $L_N(k)$ denote the length of the shortest
cycle through exactly $k$ of them. For $k \approx pN$, the quantity

$$
f_N(p) = \frac{\mathbb{E}[L_N(k)]}{k}
$$

is expected to approach a limiting function as $N$ grows. At $p=1$, this is the usual Euclidean
TSP constant from the Beardwood--Halton--Hammersley theorem, numerically about $0.7124$.
Aldous asks for the shape of the full curve on $(0,1]$: does it decrease monotonically, and what
is its limiting form?

For the original problem statement and background, see
**[Aldous's problem page](https://www.stat.berkeley.edu/~aldous/Research/OP/simTSP.html)**.

The program estimates the curve by Monte Carlo simulation over a grid of $p$-values, solving the
resulting combinatorial problems with heuristic local search. Production-scale results are therefore
upper bounds on the unknown optimum. For small instances, an exact dynamic program can globally
optimize both the selected subset and its cycle.

## Algorithmic approach

### Overview

For each value of $p$, the program has to solve two linked problems.

First, it has to decide **which** $k$ points should be visited.

Second, it has to decide **in what order** those selected points should be visited.

If $p=1$, there is no subset choice and the task is an ordinary Euclidean TSP on all $N$ points.
If $p<1$, the program must optimize the subset and the tour through that subset simultaneously.

A simulation repeatedly generates an independent random point set, solves the problem for each
requested $p$, records the normalized tour length $L_N(k)/k$, and averages those values over many
instances. Point-generation seeds and search seeds are tracked separately so that statistical and
heuristic-search variation can be studied independently.

### Full TSP solver for $p=1$

When $p=1$, the only question is the visiting order.

The full-tour solver uses multiple starts. Candidate tours are built by nearest-neighbor and
insertion constructions, screened cheaply, and then the most promising and sufficiently diverse
starts are promoted to stronger local search.

The main local moves are:

1. **2-opt**: remove two edges, reconnect the tour in the other possible way, and reverse the
   affected segment when that shortens the cycle.
2. **Or-opt**: remove one or more consecutive points and reinsert them elsewhere in the tour.

Good Euclidean TSP moves usually involve nearby points, so the solver precomputes exact
nearest-neighbor lists and uses them as candidate sets instead of comparing every pair of nodes.
To escape local minima, the promoted tours undergo iterated local search with structured kicks and
re-optimization. The best tours are retained in a small elite pool and polished again before the
final result is reported.

### Subset solver for $p<1$

When $p<1$, the problem is no longer just "find a good tour." It becomes "find a good set of
$k$ points and a good tour through them."

That is the core of the project.

Each subset run starts from several seed solutions. Seeds can come from:

- good solutions at nearby values of $p$;
- compact geometric or dense-region constructions;
- reductions of a full TSP tour in the high-$p$ regime;
- randomized and diversity-oriented starts;
- elite solutions found by earlier restarts.

Using nearby $p$-values matters because a strong solution at one density is often a useful starting
point for a neighboring density. The solver can sweep through the $p$-grid in both directions and
retain the better continuation.

After a seed has been built, the solver runs simulated annealing. The main subset move removes one
currently selected point, inserts one currently unselected point, and reconnects the cycle while
keeping the subset size fixed. The default controller preserves the validated one-candidate search;
held-out policies can instead evaluate several candidate moves per annealing step in the regimes
where this produced better matched-compute results.

During and after annealing, deterministic cleanup applies:

- 2-opt and Or-opt within the current cycle;
- one-for-one subset exchange descent;
- bounded two-for-two exchanges;
- exact batched insertion evaluation;
- optional exhaustive finishing for sufficiently small tours.

### Larger-neighborhood improvement

Single exchanges are useful but can be too local. The solver therefore also includes several larger
repair and recombination steps.

**Ruin and recreate** removes a small group of selected points, builds a restricted replacement
pool, reconstructs the damaged region, and polishes the resulting tour.

**Ejection chains** follow a bounded sequence of dependent exchanges that can cross barriers which
no single improving swap can cross.

**Path relinking** moves gradually between two strong elite subsets and tests the intermediate
solutions. Diversity-aware elite storage prevents the archive from collapsing to near-duplicates.

These steps let different restarts share useful structure instead of behaving as completely
independent searches.

### Special regimes and search policies

The main documented mode is `balanced`, which is the baseline end-to-end path for estimating the
full curve. Additional modes emphasize the ends of the $p$-range:

- `smallp-region` -- geometrically focused seeds for very small $p$;
- `highp-delete` -- starts from a full tour and removes inexpensive points near $p=1$;
- `hybrid` -- combines the regime-specific mechanisms.

The search controller is selected separately:

- `legacy-balanced` -- default compatibility policy;
- `heldout-balanced` -- matched-compute multi-candidate annealing in its validated range;
- `heldout-quality` -- deeper subset search and a stronger full-TSP controller.

The held-out presets are opt-in because their validated ranges depend on geometry and $p$. See
[`docs/heldout_search_policy.md`](docs/heldout_search_policy.md) for the experiments, activation
rules, and interpretation limits.

## Build

A C++17 compiler and CMake are required. Ninja is recommended but not mandatory.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Or use the supplied presets:

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

The core library and CLI have no mandatory third-party solver dependency. Python is used only for
analysis, plotting, campaign management, and some regression tests.

## Run

A small smoke run:

```bash
./build/aldous_tsp --mode balanced --oracle none --quick --output results.json --force
```

A larger run:

```bash
./build/aldous_tsp \
  --mode balanced \
  --oracle none \
  --N 2000 \
  --instances 12 \
  --threads 12 \
  --output results.json \
  --force
```

A quality-oriented held-out policy:

```bash
./build/aldous_tsp \
  --search-policy heldout-quality \
  --N 2000 \
  --instances 12 \
  --threads 12 \
  --output results-quality.json \
  --force
```

By default, the program writes `results.json` in the current working directory. Existing files are
preserved unless `--force` is supplied. A digest-bound adjacent receipt records the durable output
commit and end-to-end timing.

Use

```bash
./build/aldous_tsp --help
```

for the complete generated option reference, or

```bash
./build/aldous_tsp --dry-run --N 500 --p-range 0.02:1.0:12
```

to inspect the fully resolved configuration without running a simulation.

## Exact small-instance calibration

For supported instances up to the hard cap $N=18$, the solver can globally optimize both the
cardinality-$k$ subset and the cycle through it:

```bash
./build/aldous_tsp --N 18 --exact-subset-max-n 18 ...
```

This exact mode is intended for regression testing and calibration. Large production instances use
heuristic search and do not claim global optimality.

## Optional external TSP solvers

External post-processing is disabled by default with `--oracle none`. LKH and Concorde can be used
when installed explicitly:

```bash
./build/aldous_tsp --oracle lkh --lkh-path /path/to/LKH ...
./build/aldous_tsp --oracle concorde --concorde-path /path/to/concorde ...
```

The integration records executable identity, enforces time and output limits, validates returned
tours, and does not silently fall back to another method in publication campaigns. See
[`docs/oracles.md`](docs/oracles.md).

## Design decisions

Distances are computed from coordinates rather than stored in a full $O(N^2)$ matrix. Exact KNN
candidate lists are built either by brute force or by a uniform-grid search; the grid backend keeps
memory near-linear in $N$ for large random instances.

The public solver boundary uses an immutable `PreparedInstance`. Coordinates, numerical range,
geometry, KNN rows, and derived spatial structures are validated before search begins. Successful
solves are checked again for cardinality, uniqueness, finite length, and consistency.

Open-square and periodic/toroidal geometry share canonical distance routines. The random-number
generator and tie rules are deterministic, while separate point and search streams make paired
experiments reproducible.

Native output uses strict schema 16 with locale-independent UTF-8 JSON, complete resolved options,
source/build identity, restart diagnostics, phase timing, memory planning, and campaign
fingerprints. Atomic no-clobber or replace semantics prevent partial or accidental output loss.

For implementation details, see [`docs/algorithm.md`](docs/algorithm.md). For the reusable C++ API,
see [`docs/public_api.md`](docs/public_api.md) and
[`examples/library_usage.cpp`](examples/library_usage.cpp).

## Project structure

- `include/aldous_tsp/` -- public C++ library headers;
- `src/` -- geometry, instances, solvers, exact methods, result handling, and CLI implementation;
- `apps/` -- command-line executable entry point;
- `tests/` -- C++ and Python regression tests;
- `scripts/` -- plotting, profiling, campaign, migration, and analysis tools;
- `docs/` -- algorithm, reproducibility, validation, API, and release documentation;
- `schema/` -- current and frozen historical JSON schemas;
- `examples/` -- library example and example input grids;
- `validation_runs/current/` -- compact evidence generated for the current release;
- `validation_archive/` -- explicitly historical validation fixtures.

## Plotting

The plotting script reads the result JSON and produces the estimated normalized-length curve:

```bash
python3 scripts/plot_results.py results.json -o curve.png
```

It can also export the plotted summary:

```bash
python3 scripts/plot_results.py results.json -o curve.png --csv summary.csv
```

## Reproducibility and validation

Result files contain the exact source revision, resolved configuration, point/search stream policy,
and method fingerprint. Publication campaign tools use exact manifests, reject stale or mismatched
resume files, and fail on incomplete campaigns unless partial output is explicitly authorized.

Campaign analysis supports replicate-block bootstrap, separate point-set and search-seed variance,
multifidelity correction, cross-fitted control variates, and finite-size model/range sensitivity.
See [`docs/reproducibility.md`](docs/reproducibility.md) and
[`docs/campaign_analysis.md`](docs/campaign_analysis.md).

Current compact validation evidence is summarized in
[`docs/known_good_benchmarks.md`](docs/known_good_benchmarks.md). The hosted release gate covers GCC,
Clang, AppleClang, MSVC, ASan/UBSan, ThreadSanitizer, Ruff, clang-tidy, CodeQL, packaging, and a pinned
historical performance comparison.

## Development provenance

Development of version 2.0.0 used extensive AI-assisted implementation and review under the
maintainer's direction. The public commit identities and validation policy are explained in
[`DEVELOPMENT.md`](DEVELOPMENT.md). They do not imply sponsorship or endorsement by OpenAI.

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE).
