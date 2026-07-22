# Algorithm notes

The solver separates the problem into instance/KNN construction, full-tour optimization, subset optimization, and reporting/reproducibility.

## Instance and KNN layer

`aldous_tsp::Instance` stores Euclidean coordinates only. Distances are computed exactly from coordinates rather than from an `O(N^2)` distance matrix.

Two exact KNN backends are available:

- `grid` / `grid-exact`: uniform-grid expanding-ring KNN search over the actual coordinate bounding box, with lower-bound termination and reverse-KNN metadata. The implementation is exact for generated square instances and for arbitrary finite coordinates supplied through `set_points()`.
- `bruteforce`: exact all-pairs ranking, useful for tests and pathological debugging.

Use `--verify-knn <checks>` to compare sampled KNN rows against brute force. If `<checks> >= N`, verification walks all rows deterministically.

## Exact small-instance subset layer

`--exact-subset-max-n <threshold>` replaces the heuristic solve with a global
cardinality-constrained dynamic program whenever `N <= threshold`. It is off by
default and has a compile-time safety cap of `18`. The public
`exact_subset_cycle()` API exposes the same solver directly and does not require
KNN construction.

For each nonempty membership mask `M`, let `a` be its least-numbered node and
let `D[M,j]` be the shortest path that starts at `a`, visits every node in `M`
once, and ends at `j`. The recurrence is

```text
D[{a},a] = 0
D[M,j] = min_i D[M \ {j},i] + d(i,j)
```

where finite predecessor states retain the same anchor. For every mask of
cardinality `k`, the closing edge `d(j,a)` produces a cycle candidate. Taking
the minimum across all such masks simultaneously optimizes the selected subset
and its tour. The `k = N` case is the full TSP; `k = 0` and `k = 1` have length
zero. Exact ties are resolved deterministically by membership mask, endpoint,
and predecessor identifiers.

The worst-case time is `O(N^2 2^N)` and storage is `O(N 2^N)`. At `N = 18`,
the DP, parent, and mask-cardinality tables occupy about 43 MiB before ordinary
process overhead. Raising the hard cap therefore requires explicit memory and
runtime evidence rather than only changing a constant. Each concurrent instance
worker owns its own table, so exact campaigns should choose `--threads` with the
corresponding memory multiplication in mind.

A solved result proves the global optimum under the instance's implemented
double-precision distance metric. It sets `exact_optimal`, records DP state and
transition counts, and emits no heuristic restart records. Conditional two-NN
or Held-Karp bounds remain separate diagnostics and are not used as proof.

## Full TSP layer

For `p = 1`, the solver uses a screened multi-start iterated local search:

1. build `--tsp-candidate-starts` cheap deterministic candidates with exact KNN-guided nearest-unvisited construction,
2. optionally include one cubic farthest-insertion diagnostic start through `--tsp-farthest-starts 1`,
3. apply a common initial polish and rank candidates by length with an edge-Jaccard diversity preference,
4. promote exactly `--tsp-restarts` candidates (or all candidates when fewer were requested),
5. run promoted ILS searches in safe deterministic parallel waves with independent restart streams,
6. use exact small-tour ordering of the already fixed node set when `k <= 16`,
7. use exhaustive 2-opt up to `--final-exhaustive-k`, otherwise KNN-candidate 2-opt with don't-look bits and reverse-KNN wakeups,
8. apply Or-opt-1, 3-cut segment-shuffle perturbations controlled by `--tsp-ils` and `--tsp-patience`, and elite-pool final polishing.

The implementation intentionally keeps `--final-exhaustive-k` because sparse subset tours can need an exhaustive final local-opt check even when KNN-candidate search is faster. Candidate scoring uses batched distance evaluation; an AVX2 kernel exists behind `ALDOUS_TSP_ENABLE_NATIVE` (default OFF) with a scalar fallback. A `-march=native` build was measured ~38% slower than the default build on an AVX-512 host with GCC, so treat the native kernel as experimental and benchmark on the target machine before enabling it. Accepted 2-opt, node-move, and subset-swap moves use incremental tour mutation and local edge-cache repair.

## Subset layer

For `p < 1`, the solver combines several original prototype heuristics in the cleaner structure.

Seed sources:

- warm starts resized from the previous larger `p`,
- dense spatial/small-p seeds,
- high-p deletion seeds from a larger parent/full tour,
- segment deletion variants near high `p`,
- random samples ordered by nearest-neighbor or farthest insertion.

Search stages:

1. construct a broad, stream-stable population of independent, continuation, and optional raced seeds,
2. give every restart ordinary seed polishing and simulated annealing,
3. select quality-and-Jaccard-diverse finalists separately within each controller role,
4. give only those finalists deterministic subset-swap descent, two-for-two pair exchange, adaptive ruin/recreate LNS, and bounded variable-depth membership ejection chains,
5. rebuild the elite archive from final restart states,
6. rank elite pairs and run path relinking under literal node, pair, symmetric-difference, and candidate-scan budgets,
7. apply final fixed-subset polishing.

With automatic restart allocation, staged search explores 12 restarts for `p <= 0.08` and 5 otherwise, then strongly polishes three finalists by default. Disabling staged search restores the historical automatic 8/3 populations and runs the strong neighborhoods inline for every non-pilot restart.

The two-for-two stage evaluates the same regret-2 repair neighborhood without rebuilding a tour for every candidate pair. For each removal pair it batches candidate-to-cycle distances, computes stable best/second-best insertion profiles, evaluates all unordered add pairs by edge deltas, and materializes only the winning repaired cycle. `pair_exchange_max_k` defaults to `5000` as a resource safety gate while production-scale memory and runtime coverage expands; `0` removes the gate.

The ruin/recreate stage is multi-scale by default. Successive rounds cycle
through worst-marginal, contiguous-segment, spatial-cluster, long-edge, and
uniform-random ruin operators while increasing the ruined fraction from 0.25%
through 5%. Absolute and fractional caps bound production working sets. Repair
uses exact regret-2 candidate and edge tie ordering with incrementally maintained
insertion profiles, so larger ruins do not require rescanning every candidate
against every tour edge after every insertion. The legacy tiny alternating
segment/worst policy remains available with `--adaptive-ruin-recreate=false`.

The annealing temperature schedule is computed from the iteration index directly:

```text
T(it) = T0 * exp(log(T1/T0) * it/(iters-1))
```

This guarantees the first iteration uses `T0` and the final iteration uses `T1`; for a one-iteration run, the solver uses `T0`. The release default uses the fixed `sa_t0` and `sa_t1` endpoints and one candidate proposal per iteration. Opt-in restart-local temperature calibration and multiple-candidate proposals are described in [Experimental simulated-annealing controls](sa_experiments.md); both preserve the historical path when disabled and expose per-restart and temperature-decile telemetry for matched-compute tuning.

### Iteration budgets

The SA budget for a size-`k` solve is `sa_iters + sa_iters_per_k * k` (saturating). The default `sa_iters_per_k = 0` keeps the historical flat budget; a positive value allocates more work where the move space is larger, which is where flat budgets under-converge on multi-`p` curves.

With `time_budget_per_p > 0`, the subset and full-TSP restart loops become anytime: they run at least their configured restarts and then keep launching additional restarts (cycling the seed pool, RNG stream advancing) until the wall-clock budget for the `(instance, p)` solve has elapsed. The final restart always runs to completion. Restart counts become machine-dependent, so bitwise reproducibility across machines is intentionally traded for uniform convergence pressure; the executed restart counts are recorded in the search statistics.

### Restart parallelism

Subset restarts within one `(instance, p)` solve run in waves of `restart_threads` worker threads. Each restart draws from its own RNG stream derived from a single base draw and the restart index, and restart outcomes (solution, statistics, elite candidates) are merged strictly in restart-index order after each wave. Results are therefore a pure function of the configuration and are bitwise invariant to `restart_threads` outside time-budget mode (covered by a unit test and a CLI-level check); in budget mode, more threads simply execute more restarts per unit wall-clock. This restructuring replaced the previous single shared RNG stream across restarts, which changed search trajectories once (measured neutral-to-slightly-better on the quality-canary workload; references re-baselined).

### Second sweep

The per-instance `p` loop normally runs once in descending order, warm-starting each `p` from the shrunk solution at the next larger `p`. With `--second-sweep`, a second ascending pass seeds each `p` from the cheapest-insertion-grown best solution at the next smaller `p` and keeps the better result per row; `p = 1` rows are skipped (the full TSP solver ignores subset warm starts) and the smallest `p` anchors the chain. The two sweeps approach each basin from opposite directions, which is why this is the one search change in this release cycle with a decisively positive quality measurement: on `N=500` four-point curves (4 instances, seeds 21/99) it improved mid-`p` means by 4-9% (18 per-instance wins, 0 losses; never-worse holds by construction) for roughly +30% subset wall-clock.

### Convergence diagnostics

## Seed kinds, insertion policy, and kicks

The restart pool mixes seed *kinds* (warm, small-p, high-p, dense, random), and kind is not cosmetic: it selects both the specialised operators a restart runs and — since 0.9.5 — the insertion kernel it uses.

**Insertion policy.** The windowed kernel (0.9.4) evaluates only insertion slots near the removed tour position or near the incoming node's in-tour neighbours. That suits seeds that begin spatially concentrated. It is fatal for *exploration* seeds (random, high-p), which start spread across the square and must relocate globally to contract: with only local slots on offer they stall at `L/k ≈ 1.0–1.8`. Exploration seeds therefore use the exact `O(k)` insertion scan (`--exploration-exact-insertion`, default on); small-p, warm and elite seeds keep the windowed kernel. Note that the exact scan costs roughly 5x more per move at k=2000, so configurations differing in insertion policy must be compared at matched **wall time**, not merely at matched SA moves.

**Dense fill at small p.** Explorer slots are filled with `dense_seed` variants when `p <= 0.08` (`--small-p-dense-fill`, default on). Per-restart geometry logging showed uniformly random subsets never contract at small p (32/32 restarts, mean radius 44.7 on a torus of half-width 70.7, no draw below `L/k = 1.20`), while `dense_seed` restarts always do and supply every frontier draw.

**Elite kicks.** `--kick-restarts n` converts the last n scheduled restarts into kicks: seed from an elite member of the completed independent phase, perturb `--kick-fraction` of its members towards KNN neighbours of retained members, and anneal from `--kick-t0` (below `sa_t0`, so the inherited structure is refined rather than melted). The elite snapshot is taken exactly once at the independent→kick boundary and restart waves never straddle it, preserving bitwise invariance to `--restart-threads`.

## Per-restart diagnostics

Each solver restart emits one typed `RestartRecord`: raw length, stable `RestartKind`, sweep, controller role, seed variant, promotion stage, allocated SA iterations, metric-aware centroid, and mean radius. Full-TSP solves participate in the same model: their first record is `tsp-farthest-insertion` and later records are `tsp-nearest-neighbor`. The stable kind code/name table lives in `include/aldous_tsp/restart_kinds.def` and is shared by the public API, schema checks, and Python analysis.

**Deterministic restart racing.** `--racing-candidates` adds a population that is completely separate from the independent diagnostic restarts. Every candidate is screened for `--racing-pilot-iters` iterations using the prefix of the full temperature schedule. Candidates are ranked stably by pilot length, while `--racing-min-jaccard` prefers promoted subsets from different basins; the ranking is then filled by quality if the diversity threshold would leave promotion slots unused. Promoted candidates are rerun from the identical seed and RNG stream at full depth, so their final outcome has ordinary full-restart semantics and does not depend on pilot worker scheduling. Racing is deliberately incompatible with wall-clock anytime mode. Role `raced-production` and the promotion metadata keep these selected, dependent outcomes separate from the independent population used by endpoint analysis.

Each solve records `best_restart`, the index of the lowest restart outcome before post-restart stages such as relinking, oracle polish, and final polish. Instance `p` rows expose it directly and summary rows expose `best_restart_max` (the maximum over instances). With `--second-sweep`, primary/descending records stay first, continuation-only secondary/ascending records are appended, and `best_restart` is recomputed over the combined sequence; `restart_sweeps` and `restart_roles` preserve that provenance. The default supplemental continuation policy never displaces the independently seeded population, and every independent stream is keyed independently of the surrounding `p` grid. If `best_restart_max` repeatedly sits at the last executed restart on a workload, the restart/SA budget is likely too small for that `p`; combined with `--sa-iters-per-k` or `--time-budget-per-p` this makes under-convergence visible instead of inferred.

### Subset candidate table

The full-instance KNN list contains on average only `knn_k * p` subset members, so candidate local search on a subset tour is starved at small and mid `p`. Before each polish (and after membership changes in swap descent and the SA inner polish), the solver builds an exact per-member table of the `m = 16` nearest *subset* members via an expanding-ring search over the instance grid (brute force when no grid is available), in roughly `O(m * N)` per build. Candidate 2-opt and or-opt iterate this table instead of the full KNN list when it is available. Membership is constant within a polish call (2-opt and or-opt only reorder), so one table serves all stages of a polish.

Measured effect: quality-neutral (within a ±1% noise band) at default budgets across `N` from 500 to 8000, including regimes where `k` exceeds the exhaustive-final threshold and KNN lists hold ~4 members. Mid-`p` quality is dominated by the global search budget (restarts × SA iterations), not by candidate precision; the exhaustive final rescue and the SA swap neighborhood compensate for candidate starvation. The table is kept because it is exact, cost-neutral, and a prerequisite for member-restricted segment moves (Or-2/Or-3).

### Candidate 2-opt orientations

Candidate 2-opt scans both edge orientations per anchor: the successor edges of anchor and candidate, and their predecessor edges. With both families, every improving 2-opt move whose shorter new edge connects candidate-list neighbors is discovered; scanning only successor edges (the previous behavior) misses moves anchored on the predecessor side. The candidate loop breaks when the candidate distance reaches the longer of the anchor's two incident edges, and each family is guarded by its own incident-edge bound.

### Or-opt segments (Or-2/Or-3)

Beyond single-node relocation, polish runs a best-improvement descent over segment relocations of length 2 and 3, in both orientations, with candidate insertion edges drawn from the subset candidate table (or the full KNN list) around both segment endpoints plus a small local window. Evaluations are O(1) against the edge cache; applies rebuild the tour in O(k), which is cheap at the descent's low apply rate. Segments crossing the array origin are skipped (at most two of k cyclic segments per length, negligible).

Measured effect at default budgets: quality-neutral, like the candidate-table and orientation changes above (16 wins / 16 losses over 36 per-instance comparisons at `N=500`, within ±1.5%; +0.9%/0.0% at `N=2000`/`N=8000`) at ~0-1% wall-clock cost. The consistent picture across all three neighborhood extensions is that solution quality at these budgets is limited by the global search budget, not by local-search neighborhood reach — which is what the `--sa-iters-per-k` and `--time-budget-per-p` controls address.

## Elite pool

Elite deduplication is collision-safe. Each entry stores both a 64-bit hash and a canonical key. Set-mode keys are sorted node IDs; cycle-mode keys canonicalize rotation and direction. The subset archive protects the complete legacy length-ranked capacity and can retain additional set-diverse entries inside a configurable relative quality window. Diversity is measured by selected-set Jaccard distance, so supplemental slots cannot evict a solution the legacy archive would have kept.

## Membership ejection chains

After ruin/recreate, each full-depth restart runs a bounded variable-depth membership search. A chain proposes an unselected point near a current focus, ejects one selected member, and uses the ejected member as the spatial focus for the next step. Newly added nodes are locked and removed nodes are tabu for the remainder of that chain, so a depth-`d` prefix represents a genuine `d`-for-`d` change rather than cycling the same membership decision.

Each step uses the exact batched swap evaluator over a bounded mix of globally expensive removals and selected neighbours of the proposed additions. The chain may cross a cumulative uphill barrier controlled by `--ejection-chain-max-uphill`, expressed in mean-tour-edge units. Every nonempty prefix is cardinality-valid; the lowest prefix from each start receives route polish and one bounded membership-descent pass. The incumbent is replaced only by a strictly shorter result, making the neighborhood a deterministic quality-only extension of the preceding search.

The default portfolio uses three starts, depth six, 24 incoming candidates, and at most 96 removable members per step. Setting `--disable-ejection-chain` gives a fixed-seed ablation without consuming its RNG stream.

## Exact batched membership exchange

Deterministic one-for-one subset descent, high-p reference-guided exchange, and path relinking share one exact batched evaluator. For an ordered list of admissible `(remove position, add node)` pairs, it computes each removal gain once and performs one batched distance pass per unique add node. The three cheapest insertion edges on the unchanged tour are sufficient: deleting one node invalidates at most the removed node's outgoing edge and its predecessor's outgoing edge, while the newly merged predecessor-to-successor edge is evaluated explicitly. Every pair is then scored in O(1).

For `A` unique add nodes and `E` admissible pairs, evaluation costs `O(A * k + E + k)` rather than `O(E * k)`. The implementation retains the scalar evaluator's strict ordered tie semantics, including the exact insertion predecessor, and materializes only the winning move. Randomized open-square and torus differential tests compare the batched result against exhaustive scalar enumeration, including duplicate, invalid, and exactly tied candidates.

## Path relinking

Relinking walks from one elite subset to another by repeatedly applying the best (remove, add) swap toward the target set. It uses the shared exact membership-exchange decomposition, preserving add-major/remove-minor tie priority. A step therefore costs `O(|add| * k + |remove| * |add| + k)` instead of the naive `O(|remove| * |add| * k)` cross-product, and moves are applied incrementally instead of rebuilding the tour.

Elite pairs whose symmetric difference exceeds 64 nodes are skipped: relink cost grows superlinearly in the difference, while its marginal value over restart/SA search collapses for distant pairs (measured on `N=1000, p=0.5`, where uncapped relinking consumed ~96% of subset wall-clock for ~0.1% quality contribution). Skipped pairs still count as `path_relink_attempts` but not as feasible relinks.

Post-search relinking always includes the ordinary length-ranked prefix selected by `--path-relink-top`, then appends feasible supplemental archive entries whose one-way set difference stays inside that cap. The additional selection is diversity-first and cannot remove any relinking pair the legacy archive would have considered; it only exposes extra basins to the relink operator.

## Search statistics

The result JSON records counters for restarts, TSP ILS iterations, 2-opt, Or-opt, simulated annealing, subset swaps, pair exchange, ruin/recreate, membership ejection chains, path relinking, and wall-clock time by broad solver layer. Region and dense restarts have dedicated counters rather than being reported as random. `elite_restarts` is an aggregate for every elite-seeded restart and therefore includes the exact `kick_restarts` subset.

## Insertion policy (0.9.6)

Three kernels exist. The exact O(k) scan considers every slot. The windowed kernel offers
slots near the removed tour position plus slots adjacent to the incoming node's in-tour
KNN -- but that KNN row is over the FULL point set, so at p = k/N = 0.01 the expected
number of a node's 40 nearest that are in the subset is 0.4, i.e. usually zero. The
spatial kernel (off by default) offers slots adjacent to the incoming node's nearest
CURRENT members, found through a live index, which is populated at any density.

At k=2000 all three make the same decisions (see CHANGELOG 0.9.6), so the policy is chosen
purely on cost: the exact scan is reserved for seeds that must relocate members across the
square (random_subset, high-p), and everything else -- including the dense seeds that
supply every frontier draw at small p -- uses the windowed kernel.
