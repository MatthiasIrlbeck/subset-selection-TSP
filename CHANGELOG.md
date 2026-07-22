# Changelog

## Unreleased -- trust and publication hardening

- Promoted native result output to strict schema 16 with explicit solver, control-reference, aggregation, and complete experiment timing. The CLI emits a digest-bound adjacent receipt for serialization, durable commit, and process end-to-end timing.
- Replaced the same-sample control coefficient with stable two-fold cross-fitting, propagated independent control-reference Monte-Carlo uncertainty, and integrated adjusted observations plus shared reference draws into replicate-block campaign bootstrap.
- Added an exact `N * samples` control-reference work cap and a loss-aware schema-15-to-16 migration.
- Added resolved-configuration and quality-method fingerprints, exact campaign manifests, digest-bound fail-closed resume, and default rejection of mixed-method analysis.
- Publication drivers now require explicit validated N-scaled SA budgets and search policies, refuse silent oracle fallback, and fail on incomplete cells unless partial output is explicitly authorized.

## 0.11.0 -- held-out search-policy presets

- Added `--search-policy legacy-balanced|heldout-balanced|heldout-quality`. The default remains `legacy-balanced`, preserving the 0.10 fixed-seed controller.
- Added a p-aware, fixed-temperature four-candidate SA preset selected on disjoint held-out point and search streams. `heldout-balanced` uses 20,000 SA iterations for `0.02 <= p <= 0.35` when `k >= 40`; `heldout-quality` uses 30,000 iterations, with the same open-square range and a periodic extension through `p = 0.50`.
- Added a held-out full-TSP quality controller. `heldout-quality` screens 16 starts and promotes four into 450-iteration ILS searches while leaving explicit TSP controls authoritative.
- Preset sections activate only when their core SA or TSP controls retain release defaults, staged deterministic search is active, and elapsed-time mode is disabled. Per-restart iteration diagnostics continue to expose the effective allocation.
- Documented the matched-worker study, geometry-specific crossover, scale transfer, rejected generic temperature calibration, and release decision.
- Promoted native result output to strict schema 15 so the selected search-policy preset is explicit. Archived schema 14 remains verifiable, and `scripts/migrate_schema14_to15.py` preserves prior migration provenance while assigning the exact `legacy-balanced` pre-preset controller.

## 0.10.0 -- reproducible release and schema migration

- Synchronized the CMake package and citation metadata at version `0.10.0`.
- Documented native schema 14 as the current strict result format.
- Added `scripts/migrate_schema13_to14.py`, including explicit provenance for
  inferred and unrecoverable historical fields, atomic output replacement,
  schema validation, and idempotency tests.
- Corrected stale documentation about serialized per-oracle-call failures.
- Removed the duplicate persistent KNN squared-distance array, made reverse-KNN adjacency lazy and policy-controlled, and added conservative memory-budget scheduling with serialized requested/resolved/effective concurrency.
- Replaced the dense exact-subset value table with cardinality-indexed masks, rolling value layers, compact reconstruction parents, and a public pre-allocation peak-memory estimate.
- Made AppleClang and MSVC portability jobs blocking, added Ruff correctness, clang-tidy, and GCC ThreadSanitizer jobs, enabled leak detection in ASan CI, and added a pinned historical hot-path performance gate.


## Unreleased — correctness audit repair series

### CLI and build gates

- Static help text is written as data rather than passed to `printf` as a format string, so literal percent signs no longer truncate `--help` or invoke undefined behavior. The documented spatial-insertion default now matches the executable.
- GCC and Clang warning-clean builds use `-Werror` in CI, including the test target, so warning regressions fail before they can reach a release archive.

### Canonical periodic geometry

- Added one authoritative periodic-domain implementation for normalization, minimum-image deltas/distances, circular means, and unique wrapped-grid traversal.
- Instance metrics, exact KNN search, subset candidate tables, the live subset index, seed centroids, restart geometry, and instance-domain lifecycle now use that shared contract. Seam-neighbor misses, duplicate tiny-grid results, stale explicit sides, and out-of-domain metric/index disagreement are covered by deterministic and randomized tests.

### Typed and complete restart diagnostics

- Added public `RestartKind`, `RestartSweep`, and `RestartRecord` types. Stable kind codes and labels now have one source of truth in `include/aldous_tsp/restart_kinds.def`; compile-time checks reject gaps, and schema/Python tests enforce the same table.
- Region and dense restarts have dedicated statistics instead of being reported as random. `elite_restarts` remains the aggregate of all elite-seeded restarts, while `kick_restarts` is the exact scheduled-kick subset.
- Full-TSP restarts now emit diagnostics (`tsp-farthest-insertion`, then `tsp-nearest-neighbor`). Consequently, `p = 1` rows report the TSP restarts that actually ran.
- A second sweep now appends its records after the primary sweep, emits `restart_sweeps`, recomputes `best_restart` over the complete serialized population, and keeps every restart array aligned with `executed_restarts`.
- `scripts/restart_evt.py` obtains kind metadata from the shared definition, includes dense kind `7` by default, validates aligned arrays, and can filter primary versus secondary draws with `--sweeps`.
- Schema version remains `13`; the new fields are optional so historical bundled artifacts remain valid until the next deliberate schema migration.

### Performance observability and exact batched neighborhoods

- Added machine-readable aggregate phase worker-seconds, sampled SA proposal/insertion latency, and a paired hot-path benchmark harness that alternates run order while enforcing both quality and runtime canaries.
- Periodic instances now use canonical minimum-image distance kernels after one-time normalization, including a periodic AVX2 batch path when native compilation is enabled; checked arbitrary-coordinate queries retain the normalization boundary.
- Two-for-two pair exchange preserves the legacy regret-2 neighborhood while batching candidate distances, avoiding candidate-tour reconstruction, and applying a configurable large-`k` safety gate (`pair_exchange_max_k`, default `5000`).
- One-for-one subset descent, high-p exchange, and path relinking now share an exact batched top-three insertion evaluator. Ordered tie semantics and incremental move deltas are differential-tested against scalar enumeration on open and periodic instances.

## 0.9.6 -- what the insertion kernel actually does at production scale

### The finding

At k=2000 (N=200000, p=0.01) the insertion kernel does not change the search at all.
Windowed (0.9.4) and the new spatial kernel produce BIT-IDENTICAL runs: same accepted
moves (36880), same improving moves, same tours, same L/k to every digit -- with a
restart pool that contains the dense explorer seeds (kinds {2: 16, 7: 16}). The exact
O(k) scan returns the same answer too (L/k 0.7040 at 16 restarts; 0.6854 in a 400k-move
probe). They differ only in what they cost:

| kernel      | per-move | k=2000, 1 instance x 16 restarts |
|-------------|----------|----------------------------------|
| exact O(k)  | 25.9 us  | 96.3 s                           |
| windowed    | 4.05 us  | 61.1 s                           |
| spatial     | 1.63 us  | 61.6 s                           |

Why the kernels cannot differ here: candidates are drawn from the KNN rows of the removed
node and its two tour neighbours, so their true spatial neighbours already lie inside the
local tour window -- the window and the spatial slots coincide. The ~24 uniformly random
candidates per move sit 150-200 units from a contracted subset, so their insertion delta
is ~300 whatever slot they are given, and exp(-300/1.4) rejects them under every kernel.
At k=100 the +/-12 window covers a QUARTER of the tour and the kernels genuinely diverge;
at k=2000 it covers 1.25% and they cannot.

Consequence: 0.9.5's exploration-seed insertion fix -- measured at -0.0109 (3.5 sigma) at
k=200 -- does NOTHING at k=2000, and charges +58% of the wall clock for it. It was a
small-k phenomenon that did not transfer. This is now the standing methodological warning
in docs/limitations.md: a result established at k=100-200 is not evidence about k=2000.

### Changed

- Seed kinds split: `dense_seed` is now kind 7; kind 0 means `random_subset` only. They
  shared code 0, which is exactly why "exploration seed" forced the exact scan onto dense
  draws that never needed it -- and why the 0.9.5 geometry analysis had to infer the split
  from the parity of a pool index.
- The exact scan is now reserved for seeds that must relocate members ACROSS THE SQUARE:
  `random_subset` (kind 0) and high-p delete seeds (kind 3). Dense seeds take the fast
  kernel. At k=2000 this is 1.56x less wall for a bit-identical answer (96.3s -> 61.6s).
- `--dense-exact-insertion` restores the 0.9.5 policy. Verified BIT-IDENTICAL to 0.9.5 at
  k=200 (0.618993, 0.603996), so that result remains reproducible.

### Added

- `SubsetIndex` and `find_best_insert_after_remove_spatial` (`--sa-spatial-insertion`,
  `--sa-spatial-neighbors 16`), default OFF. A live doubly-linked bucket grid over the
  CURRENT subset members, refined as the subset contracts, with a bounded ring search. It
  answers the question the kernel actually needs -- "which current members are near this
  candidate?" -- in O(1) at any subset density, which is what the exact scan was
  brute-forcing. 16x cheaper per move than exact, 2.5x cheaper than windowed.

  KEPT OFF BY DEFAULT: a negative result at production scale. At k=2000 it buys nothing
  (identical search; insertion is only ~5% of wall). At k=100 it LOSES to windowed
  (+0.0103 +/- 0.0044). It ships because it is the instrument that proved the kernels are
  equivalent, and because it is the right structure if insertion ever becomes the
  bottleneck. Like region seeds, it is kept as a documented failure rather than deleted.

- Tests: `test_subset_index_matches_bruteforce` (exact m-nearest under periodic wrap and
  add/remove churn), `test_spatial_insertion_saturates_to_exact` (neighbours >= k
  reproduces the exact scan bit-for-bit; restricted queries still report honest deltas
  against a recomputed tour length), `test_dense_seeds_are_not_exploration_seeds` (pins
  the kind-7 policy that the 1.56x depends on).

### Where the wall clock actually goes at k=2000 (1 instance x 16 restarts, spatial)

| stage removed          | wall   | L/k    |
|------------------------|--------|--------|
| baseline               | 61.1 s | 0.7040 |
| - subset-swap passes   | 44.0 s | 0.7044 |
| - ruin-recreate        | 61.1 s | 0.7054 |
| - path relinking       | 62.4 s | 0.7040 |
| - 2-opt                | 64.3 s | 0.7582 |

Subset-swap descent costs 28% of the wall clock for ~0.0004. Path relinking costs nothing
and contributes nothing at this scale. 2-opt is load-bearing. The insertion kernel -- the
thing 0.9.4 and 0.9.5 were both built around -- is about 5% of wall. The next round of
speed work belongs in the descent stages, not the kernel.

## 0.9.5 — sampler diagnostics, exploration-seed fix, elite kicks

The 0.9.4 allocation scans (B=960 and B=1920 at k=2000, 24 paired instances) established that anneals converge by ~60 iterations/candidate and that everything beyond that is pure independent multistart: doubling restarts at fixed depth buys a **constant** −0.0025 (0.6278 → 0.6252 → 0.6228), while doubling depth at fixed restarts buys nothing (−0.0001 ± 0.0009). A best-of-m log-crawl has no plateau, so plateau-hunting as a certification strategy is over. This release turns the solver into an instrument for the two things that replace it: a better sampler, and an extreme-value estimator for the endpoint of the basin-value distribution.

### Critical: windowed insertion crippled exploration seeds (regression introduced in 0.9.4)

0.9.4 made windowed insertion the default kernel (10-11x wall-clock speedup, confirmed). Windowed insertion only offers insertion slots near the removed tour position or near the incoming node's in-tour neighbours. That is right for seeds that start spatially concentrated — but **random and high-p seeds must relocate globally to contract**, and the windowed kernel cannot express that move. Their restarts stalled at L/k ≈ 1.0–1.8 instead of ≈ 0.65.

It escaped detection because every quality A/B in 0.9.4 used `--restarts 4`, whose truncated seed pool contains only small-p seeds — the one kind windowed insertion suits. It surfaced only when per-restart values were logged and the distribution turned out to be nonsense.

Exploration seeds (random, high-p) now use the exact O(k) insertion scan; small-p, warm and elite seeds keep the fast windowed kernel. Measured at k=200, p=0.01, 24 restarts, 4 paired instances: **−0.0109 ± 0.0031 on the final value (3.5σ)**, acting entirely through the left tail of the `dense_seed` draws. `--exploration-exact-insertion=false` restores the 0.9.4 behaviour.

Honest note on existing results: the k=2000 allocation scans (0.6252, 0.6228, …) were produced with the regression active, so they **understate** what independent multistart achieves at those budgets. They remain valid upper bounds; they are not tight ones.

### The small-p restart pool was spending a third of its budget on dead draws

Per-restart geometry logging (toroidal centroid + mean radius of each restart's subset) settled a question that was previously guesswork. The pool's "explorer" slots alternate two constructions, both labelled kind 0. At p=0.01, k=200, 24 restarts, 4 instances:

| construction | best draw | median | mean radius | never contracted |
|---|---|---|---|---|
| `dense_seed` | **0.5858** | 0.6724 | 11.0 | 0/32 |
| `random_subset` | 1.2024 | 1.5060 | 44.7 | **32/32** |

Every frontier draw came from `dense_seed`. Not one uniformly random subset ever contracted, and none came within a factor of two of a usable tour: at small p a random subset is simply too far from any good configuration to be reeled in at realistic budgets. `--small-p-dense-fill` (default **on**, applies at p ≤ 0.08) fills the explorer slots with `dense_seed` variants only. The dominance argument is what justifies the default — a draw that always contracts strictly beats one that never does, so best-of-m can only improve. The measured effect on the final value at 4 instances was +0.0030 ± 0.0068, i.e. **unresolved**: the sandbox needs ~190 instances at k=200 to see a 0.0025 effect. It must be confirmed at k=2000/24 instances, where the paired harness resolves ±0.001. `--small-p-dense-fill=false` restores the old pool.

### Elite-kick restarts

`--kick-restarts <n>` runs the last n scheduled restarts as kicks: each seeds from an elite member of the completed independent phase, perturbed by swapping `--kick-fraction` (default 0.10) of its members to KNN neighbours of retained members, and anneals at the reduced `--kick-t0` (default 0.35) so inherited structure is refined rather than re-melted. The elite snapshot is taken **exactly once**, at the independent→kick boundary, and restart waves never straddle it, so results stay independent of `--restart-threads` (pinned by a test). Default 0 = pure multistart, unchanged.

Budget-matched at k=200, p=0.01 (24 restarts, 4 instances, identical SA move counts): **−0.0047 ± 0.0071**. Favourable, not significant at this sample size. Needs the k=2000 paired test.

### Region seeds (off by default — a negative result worth keeping)

`--region-seeds` replaces pooled random seeds with fresh per-restart local subsets (uniform centre, k points drawn from the `--region-dilation`·k nearest). It never wastes a restart, it is 1.6x faster, its draws are tidy — and it is **worse**: +0.0093 ± 0.0076 at dilation 1, +0.0135 ± 0.0089 at dilation 3. Making the draws reliable narrows the distribution (sd 0.036 vs 0.42), and the minimum lives in the left tail, not the median. Kept as an option because the negative result is the point: a sampler for this problem should maximise left-tail mass, not minimise variance.

### Per-restart logging (input for the endpoint estimator)

With `--include-instance-rows`, each `instance_rows[].p_results[]` now carries `restart_values` (final L/k of every restart, in restart-index order), `restart_kinds` (0=random, 1=warm, 2=small-p, 3=high-p, 4=elite, 5=kick, 6=region), and `restart_centroids_x` / `restart_centroids_y` / `restart_radii` (where in the square each restart settled, and how tightly — toroidal circular mean, not the meaningless arithmetic one). These are approximately iid draws from the search's basin-value distribution; the kinds are logged because that distribution is a **mixture**, and a tail fit on the pooled sample is a fit to a contaminated distribution. `stats.kick_restarts` counts executed kicks. Schema updated.

### Also

- `instance.cpp`: `sx`/`sy`/`side`/`periodic` are now scoped inside the AVX2 block (MSVC C4189).
- `scripts/convergence_study.py`: allocation stage always requests instance rows, and `--solver-arg` (repeatable) passes arbitrary flags through to the solver, so kick/seed-policy arms can be run through the paired harness. `scripts/run_full_study.py` campaigns now emit instance rows.
- Tests: `test_kick_restarts_mechanics` (accounting, tour validity, honest length, thread-count determinism), `test_restart_value_logging`, `test_region_seeds`.

### Reproducibility

`--small-p-dense-fill=false --exploration-exact-insertion=false` reproduces 0.9.4 **bit-for-bit** (verified on a config where the seed-fill loop is active). Defaults intentionally differ: they include the exploration-seed fix and the dense fill.

## 0.9.4 — search-kernel performance and adaptive budgets

### Critical: the batched distance routine ignored periodic boundaries

`dist_many_from` — the SIMD distance kernel used by the SA insertion evaluation on **every move** and by descent scans — computed plain Euclidean distances regardless of `inst.periodic`. On the torus, 43% of random pairs came back wrong (unwrapped), worst error ~1.3x the half-diagonal. This is a long-standing bug that predates this release; it was caught by the new windowed-insertion correctness test, whose delta check refused to reconcile with the applied tour.

Impact on existing torus results, honestly stated: min-image is a minimum, so unwrapped distances only **overestimate**. Reported tour lengths therefore remained valid upper bounds (consistent with observed Held-Karp gaps of ~0.0000 at high budget), but the search was silently biased against boundary-straddling insertions, and the incremental length accounting could drift upward when such moves were nonetheless taken. Expect equal-or-slightly-lower plateaus after the fix; convergence ladders on periodic instances should be re-run (they are cheap under the new kernel).

How it escaped: a test named `test_dist_many_from_matches_scalar` already existed — but it used 6 hand-placed points, open-square only, and 9 ids, **below the `count >= 8` threshold, so it never entered the SIMD body it existed to check**. The test is upgraded to random instances under both boundary conditions with counts that exercise the SIMD body and both scalar tails; `dist_many_from` now applies the minimum-image convention in both the AVX2 and scalar paths, mirroring `Instance::dist`.

### Kernel: per-move cost cut ~3-5x single-threaded, k-linear term eliminated

Profiling the SA loop at `k=2000, N=200000` found ~19 of the ~21.7 us/move in `find_best_insert_after_remove`, which scanned **all k tour positions** per move for the globally best insertion; ~2 us in the per-proposal candidate collection (a ~96-entry list rebuilt per move with an O(list) dedupe scan per push); and O(N) `Tour` copies (~1 MB at N=200000) on every improvement for best-tracking.

- **Windowed insertion (new default).** Evaluates only slots within `--sa-insertion-window` (default 12) tour positions of the removed slot plus slots adjacent to in-tour members of the incoming node's KNN row; distances batched through `dist_many_from`. Identical delta formula and slot indexing to the exact scan; `--sa-exact-insertion` preserves the historical O(k) behavior. Paired-seed validation at matched iteration budgets: identical to exact at `k=2000` low budget (3/3 instances to 4 decimals), `-0.005 +/- 0.005` (windowed better) near the k=500 plateau, `-0.002 +/- 0.002` at ratio 10 after the metric fix — no systematic quality loss anywhere tested.
- **Epoch-stamped candidate dedupe.** O(1) membership instead of the linear scan; verified canonically **bit-identical** to the old binary across periodic, open-square, and oracle configurations (pre-metric-fix, exact-insertion mode).
- **O(k) best-snapshot.** Best-so-far is now nodes+length instead of a full Tour copy; the tracked length is restored on the rebuilt tour so downstream comparisons are bit-identical to the old path.

Measured (sandbox, single thread): `k=500/N=50000` 6.2 -> 2.1 us/move; `k=2000/N=200000` 21.0 -> 4.3 us/move, and per-move cost no longer grows ~linearly in k. Multi-threaded gains are expected to be larger on memory-bandwidth-bound machines (the eliminated O(k) scan and O(N) copies were the main bandwidth consumers); to be confirmed on production hardware. New tests: `test_windowed_insertion_correctness` (applied-move deltas must match recomputed lengths; whole-tour window must equal the exact scan) and the upgraded `test_dist_many_from_matches_scalar`.


Search-kernel performance release plus adaptive budgets: identical or equal-quality results at a fraction of the wall-clock, and new budget controls to reinvest the freed time where multi-`p` curves under-converge.

### Critical: the subset search was not converged, and it fabricated the small-p trend

With the LKH oracle repaired (below), the tours became near-optimal *for the subsets they were given* — and that exposed the real bottleneck. The July 2026 campaign reported `f(0+) = 0.709`, `alpha = 3.01`. Both are artefacts of an under-converged subset search.

`f(p)` is a **minimum over subsets**, so every reported value is an upper bound and an under-converged search biases it *upward*. That bias is not uniform: the candidate pool is `N = k/p`, so the search has strictly more to do at small `p`, and the upward bias is worst exactly where the small-`p` physics lives. An under-converged run therefore does not merely add noise — it manufactures a fake `p`-trend, and `alpha` gets fitted straight out of it.

The budget was `sa_iters = 60000`, **flat in `N`**. In SA proposals per candidate point that is 48 at `(p=0.2, k=250)` but **0.30** at `(p=0.01, k=2000, N=200000)` — the search could not look at each candidate even once, so the 20x larger pool that `p=0.01` buys was never examined.

The evidence, three independent ways:

- **A hard invariant is violated.** At fixed `k`, choosing from a larger pool can only help, so the optimum is non-increasing in `N`. Yet at `k=2000` the pool grew 20x (`N` 10,000 -> 200,000) for a gain of 0.0005, and at `k=1000` the value *rose* by 1.7 sigma from `N=20,000` to `N=50,000`. The true optimum cannot rise.
- **The budget ladder never plateaus.** At `(k=500, N=50000)` — the same iterations-per-candidate as the `p=0.01` batches — raising the budget 8x then 25x moved `L/k` 0.6862 -> 0.6488 -> 0.6276, still falling. The Held-Karp bound on the chosen subsets fell with it, so these are genuinely better *subsets*, not merely better tours.
- **The error dwarfs the signal.** The entire `p`-dependence being measured spans 0.0064. The search error is at least 0.06 — ten times larger, and `p`-dependent.

Corroborating symptoms in the fit itself: `alpha`'s CI was `[0.02, 3.01]`, the whole search range (unidentified); `C = -0.706` was negative, i.e. `f` decreasing in `p`, backwards; and the `f(0+)` CI reached 0.7993, above `beta = 0.7124`, which is impossible.

### `--restarts` is now authoritative (it previously did nothing), with an auto default

The restart count was `max(seed_pool.size(), subset_restarts)`, and the seed builders inject up to 8 small-p seeds at `p <= 0.08`. So the flag could not lower the restart count below the pool size: `--restarts 1` and `--restarts 8` ran identically (verified: 17s vs 16s). The search budget was not controllable at all, which is fatal for a convergence study.

The pool is now truncated to the requested count. Truncation round-robins across seed **kinds** (best-first within each kind, most promising kind first) rather than taking the globally shortest seeds: kind is not cosmetic — it selects specialised operators (high-p exchange, small-p region moves) — so a purely length-ranked truncation could silently disable an entire operator. `test_restarts_flag_is_authoritative` pins the behaviour.

Honoring the flag literally exposed a side effect the audit caught: the historical *default* (3) had never actually run at small p — the seed pool forced 8 — so an honest flag with an unchanged default silently degraded default-quality exactly in the tool's core regime (measured: `L/k` 0.6031 -> 0.6126 at `p=0.02, N=5000`). The default is therefore now **auto** (`-1`): 8 restarts at `p <= 0.08`, 3 otherwise, which reproduces the historical effective behavior bit-for-bit (verified identical to 0.9.1 on the default path). Any explicit `--restarts <n>` remains exactly authoritative.

### New: `--sa-iters-per-n`, a budget that tracks the candidate pool

The subset search picks `k` of `N`, so its budget must be able to scale with `N`, not just `k`. (`--sa-iters-per-k` scales with `k` and is therefore constant across `p` at fixed `k` — it cannot fix this.) The effective budget is now `sa_iters + sa_iters_per_k * k + sa_iters_per_n * N`. The default 0 preserves the historical flat budget; `run_full_study.py` accepts `--sa-iters-per-n` and logs a prominent warning when it is left at 0.

### New: `scripts/convergence_study.py` — validate the budget before trusting a campaign

Two stages, each with a machine-readable verdict:

- **budget** — fix `(p, k)`, sweep the budget, watch for a plateau. It distinguishes *not converged* (still improving: buy budget) from *inconclusive* (steps buried in noise: buy instances — and it tells you how many), because those call for opposite responses.
- **monotone** — the cheap necessary condition. At fixed `k`, verify `L/k` is non-increasing in `N`. It flags significant rises (mathematically impossible) *and* the subtler failure where a much larger pool buys nothing — which is what the `k=2000` batches actually did, and which a rise-only test would have passed.

`--self-test` (wired into CTest) asserts the harness catches the real July 2026 numbers at `k=1000` and `k=2000`, and does *not* false-positive on `k=250`, where the search was converged.

### Audit fixes (0.9.3)

A full-tree audit (fresh review of all recent changes, AddressSanitizer + UBSanitizer over the entire test suite, cross-checks against the 0.9.1 baseline) found and fixed:

- **Default-quality regression at small p** from the now-honest `--restarts` flag; fixed with the auto default described above, verified bit-identical to 0.9.1 defaults.
- **`test_elite_anytime_restarts` assumed a fast machine.** Its 0.4 s wall-clock budget is exceeded by the scheduled restarts alone under sanitizers (0.64 s measured) or plausibly on a loaded laptop, so no anytime wave ever launched and the test failed for reasons unrelated to the logic. The budget is now derived from the measured baseline solve time. This fragility predates 0.9.2 (confirmed by reproducing it on the 0.9.1 baseline under ASan).
- **The robustness sweep never received `--sa-iters-per-n`**, so a budget validated for the campaign would silently not apply to the sweep. It is now passed through to every sweep batch.
- **Version string said 0.8.7** while releases were named 0.9.x, so every result JSON recorded a misleading provenance. Bumped to match the release.
- Sanitizers: the full suite runs clean under ASan+UBSan (zero diagnostics). The MSVC warnings from the Windows build were reviewed: `C4334` (32-bit shift) in `exact_small_tsp_cycle` is guarded by `k <= 16` and benign; `C4146`/`C4244` are benign idioms in the RNG and `std::fill` calls.

### Corrected: the Held-Karp "rigorous lower bracket" on f(0+) was not one

`analyze_campaign.py` extrapolated the Held-Karp bound and printed it as a *"rigorous lower bracket from the Held-Karp bound: >= 0.7030"*. That claim is false. HK lower-bounds the optimal tour through **the subset the solver chose**; it is not a bound on `f(p) = min over subsets`, because a better subset drives the tour *and its HK bound* down together — the "floor" moves whenever the search improves. HK brackets **tour-solving** error only, and says nothing about **subset-selection** error, which is precisely the error that biases `f(0+)` upward. The report and the plot legend now say what the number is and what it is not.

### Critical: the LKH oracle silently failed on every real-sized instance

The oracle produced **zero** successful LKH calls on any instance large enough to matter, and did so silently: the external polish is a best-effort refinement, so each failure fell back to the built-in tour and the run still reported success. A full overnight campaign therefore completed with `ok` on every batch while LKH contributed nothing.

Root cause: LKH stores edge costs in C `int` and multiplies each by its `PRECISION` parameter (default 100), aborting via `eprintf` — stderr plus exit status 1 — when that product overflows (`ReadProblem.c`: `if (N->C[j] * Precision / Precision != N->C[j]) eprintf("PRECISION (= %d) is too large", ...)`). With our default cost scale of `1e6`, any pairwise distance above `INT_MAX / (1e6 * 100) ≈ 21.5` overflows. Every campaign instance exceeds that: even the smallest (`N = 1250`) has side `35.4`. Small hand-checks stayed under the threshold, which is why the failure never surfaced in testing.

This was **not** platform-specific — it would have failed identically on Linux. It escaped the test suite because `tests/reference_lkh.py`, the stand-in used in CI, did not implement LKH's `PRECISION` guard.

- The LKH parameter file now sets `PRECISION = 1`. Our costs are already scaled integers, so LKH's internal ×100 buys no accuracy and only risks overflow.
- `effective_oracle_scale` additionally caps the cost scale per call so the largest cost stays within `INT_MAX / 4`, leaving headroom for the node potentials LKH adds during its ascent. This bounds the cost for arbitrarily large `N`, not just today's campaign sizes. It cannot bias results: only the tour's node *order* is taken from the solver, and its length is recomputed in double precision, so the reported `f(p)` is never quantized by the scale.
- `tests/reference_lkh.py` now faithfully emulates LKH's `PRECISION` overflow abort, and `test_oracle_large_coordinates_precision` exercises the oracle on campaign-scale geometry (`N = 1000`, side `31.6`). The test fails without the fix and passes with it.

Confirmed on the target machine: 480/480 oracle calls solved, 0 failed; the Held-Karp gap at `k=2000` fell from 2.6-3.4% to 0.51-0.57%.

### The oracle now reports why it failed

The failure above was diagnosable only as `external process failed with status 1`, because the child's console output was discarded — on Windows `CREATE_NO_WINDOW` left the child without valid standard handles at all. LKH had been printing `*** Error *** PRECISION (= 100) is too large` on every call, and we were throwing it away.

- `run_external_process` (both platforms) now redirects the child's stdout and stderr to a capture file and surfaces the relevant part — preferring the text after LKH's `*** Error ***` banner — in the recorded error message. On Windows this also hands the child valid standard handles, which it previously lacked.
- The oracle is now **parse-first**: if the solver wrote a usable tour, it is accepted regardless of exit code, since a solver's contract is its tour file and exit-code conventions vary. A call fails only when no usable tour comes back, and the error then reads e.g. `external solver returned no usable tour (exit status 1); solver said: *** Error *** PRECISION (= 100) is too large`.

### Windows support for the external (LKH) oracle

The oracle's process-launch layer was POSIX-only: on Windows `resolve_exec_in_path` was a stub returning empty and the polish path returned "not supported", so `--oracle lkh` failed validation immediately (exit 2) even with a valid LKH binary. This adds the Windows implementations so the oracle works on both platforms.

- `resolve_exec_in_path` now resolves a given path (trying an implicit `.exe`) or searches the `;`-separated `PATH` on Windows.
- `run_external_process` is implemented with `CreateProcessA`, passing the working directory per call via `lpCurrentDirectory` (thread-safe under the parallel instance runner, unlike a global `chdir`) and `CREATE_NO_WINDOW` to avoid a console flash per invocation; it waits with a timeout and returns the child exit code. Argument quoting follows the standard `CommandLineToArgvW` rules.
- `TempWorkDir` was rewritten to create its unique scratch directory with `std::filesystem` instead of POSIX `mkdtemp`, so it is portable; the shared problem/parameter/tour-file logic (already platform-independent) now runs on both platforms rather than being gated out on Windows. `windows.h` is included with `NOMINMAX`/`WIN32_LEAN_AND_MEAN`.
- Verified by cross-compiling the full core library into a Windows PE executable with MinGW-w64 (compiles and links) and by native unit-testing the argument-quoting logic; the POSIX build and all tests are unchanged. Runtime on Windows should be sanity-checked once on-machine (a single small `--oracle lkh` run).
- `run_full_study.py` now logs a failing batch's stderr (previously only stdout), so oracle/argument errors are visible rather than blank.

### One-command study runner — `scripts/run_full_study.py`

A single orchestrator for the whole study: the robustness sweep (both boundary conditions plus a large-N throughput point), the small-p torus campaign, and the analysis. Built to be started once and left alone -- it auto-detects LKH (explicit `--lkh-path` or on PATH) and falls back to the built-in solver with extra restarts when it is absent, is resumable (batches whose output JSON already parses are skipped, so re-running after any interruption continues), and is error-isolated (a failing or timed-out batch is logged and the run proceeds). Progress is timestamped to `<out-dir>/run.log`, and the analysis stage writes f(p), f(0+), and alpha with bootstrap CIs to `<out-dir>/analysis_report.txt` plus a figure.



The final stage of the pipeline: from the per-(p,k) campaign batches it estimates the two quantities Aldous's problem asks for -- f(0+) and the small-p exponent alpha -- with confidence intervals. It (1) extrapolates every p to N -> inf with the torus O(1/N) form, (2) fits `f(p) = f0 + C p^alpha` by profiling the single nonlinear parameter alpha (the model is linear in (f0, C) at fixed alpha, so each step is an exact weighted least squares), and (3) attaches percentile confidence intervals to f(p), f(0+), C, and alpha with a master bootstrap that resamples instances at every (p,k) and reruns the whole chain, so instance noise is propagated through both the finite-size extrapolation and the power-law fit. It also extrapolates the Held-Karp floor to a rigorous lower bracket on f(0+), and reports the per-p tour-to-bound gap so the reader can see whether the underlying solves were converged.

- A `--self-test` (registered as `campaign_analysis_self_test` under `ALDOUS_TSP_ENABLE_PYTHON_TESTS`) generates a synthetic campaign with known (f0, C, alpha) and checks the fit's 95% intervals cover the truth. On a small real torus campaign the script produces a sensible point estimate with honestly wide intervals when the data is sparse (no false precision), and tightens as p-values, instances, and LKH-converged solves are added.
- `run_torus_campaign.py` now also passes `--held-karp` (so the bounds the analysis consumes are produced) and points at `analyze_campaign.py` as the next step.



The two-NN control-variate bound is loose (~0.87 of optimal), so the certified floor on f(p) sat far below the found tour. This adds the Held-Karp (Lagrangian 1-tree) lower bound, which is tight (~0.998 of optimal here), turning the one-sided floor into a razor-thin, rigorous two-sided bracket -- and, via its gap to the found tour, an independent certificate of solver near-optimality.

- New `src/lower_bound.cpp` computes the bound as the minimum 1-tree under modified edge weights `c'(i,j) = d(i,j) + pi_i + pi_j`, minus `2*sum(pi)`, maximized over the node potentials `pi` by subgradient ascent (subgradient `degree_i - 2`, Polyak step toward the found tour length, with a halving schedule). `d` is `inst.dist`, so the bound is exact for the flat torus. Prim's MST over a materialized `O(k^2)` subset matrix; intended for moderate k (guarded above 6000 nodes).
- Correctness and tightness are validated by a new `test_held_karp_bound`: across torus and open instances (k <= 16, exact solver as ground truth) the bound never exceeds the optimum (rigorous) and stays within a few percent of it. An external check over 48 instances measured zero rigor violations, mean bound/optimum = 0.9984 (vs 0.8663 for the two-NN bound), under ASan/UBSan.
- New `--held-karp[=bool]` (with `--hk-iterations`, default 400) computes the bound per solved subset using the found tour as the step-sizing upper bound, and reports per p the mean bound (a sharp floor on f(p)) and the mean gap to the tour (the near-optimality certificate). On a torus run this brackets f(1) as [0.709, 0.720] -- containing beta ~ 0.7124 -- and shows the built-in solver sitting ~0.4% above the bound at p=0.2 and ~1.5% at p=1, quantifying the large-k suboptimality directly. All fields are optional JSON (schema updated); `extrapolate_fpN.py` now prefers the Held-Karp bound over the two-NN bound when extrapolating the lower edge of the bracket.

### LKH oracle on the torus: correctness guard, validation, and campaign runner

The finite-size extrapolation is only as clean as the underlying solves, and the built-in solver leaves a residual suboptimality at large k. LKH removes it -- but only if it is handed the *torus* distances. The existing EXPLICIT full-matrix oracle format already serializes `inst.dist` (torus-aware), so that path is correct; the EUC_2D format hands LKH raw coordinates and would silently optimize open-plane distances while the tour is scored on the torus.

- Added a config guard: `--periodic` with `--oracle-format euc2d` is now rejected at startup with a clear message, so the torus is never solved against the wrong metric.
- Proved the matrix path is torus-correct end to end. A new `tests/reference_lkh.py` speaks LKH's CLI protocol (probed as `--version`, invoked as `LKH run.par`, reads the EXPLICIT matrix, writes a TOUR_FILE) and solves the handed matrix exactly for small k. A new `test_oracle_torus_roundtrip` (guarded by `ALDOUS_TSP_TESTS_DIR`) runs the full oracle path on k=17 torus and open instances and checks the returned tour, scored with the torus metric, equals an independent Held-Karp optimum -- match to ~1e-16, which would break if the matrix used open distances or the tour were mis-scored.
- `scripts/validate_lkh_torus.py`: run on a machine with real LKH, it drives a p=1 torus k-ladder through LKH, checks each point against beta ~ 0.7124, extrapolates via `extrapolate_fpN.py`, and PASS/FAILs on the intercept -- the concrete check that LKH collapses the p=1 anchor onto beta.
- `scripts/run_torus_campaign.py`: the multi-p campaign. For each p and target tour size k it runs one batch at N=k/p on the torus with LKH + control variate, then extrapolates every p to N -> inf and (given f(0+)) fits the small-p exponent alpha. Because the EXPLICIT matrix is O(k^2), k is kept <= ~3000; the torus's O(1/N) convergence makes moderate k sufficient, so the ladder uses small k at large N (fast solves, large domain) -- exactly the regime the torus was adopted for.



Closes the analysis loop: turns a torus k-ladder into an estimate of the limit f(p). Given several runs at the same p and increasing N, it fits `f(p, N) = f(p) + slope/k` (the torus O(1/N) form, equivalent to O(1/k) at fixed p) by weighted least squares and reports the intercept f(p) with a standard error. For contrast it also fits the O(1/sqrt(k)) boundary form and reports both residual RMS values, so the data confirms which form applies -- on real torus ladders the 1/N residual is ~2x smaller than 1/sqrt(N), the empirical signature that the boundary term is gone. An optional `--rescale` applies the Percus-Martin nearest-neighbor rescaling (divide by `1 + 1/(8k)` before fitting) to flatten the residual correction; `--cv` uses the control-variate-corrected mean when present; and when the two-NN subset bound is in the JSON it is extrapolated too, giving a certified floor on f(p) in the limit. A second stage (`--alpha --f0`) fits the small-p exponent in `f(p) - f0 ~ p^alpha` once several extrapolated f(p) are in hand.

The fitter is covered by a `--self-test` (registered as the `extrapolation_self_test` CTest under `ALDOUS_TSP_ENABLE_PYTHON_TESTS`) that checks intercept recovery on synthetic 1/k data and that the 1/k form beats 1/sqrt(k) there. Validation on a torus ladder behaves as expected: at p=0.2 the 1/N form is clearly preferred and extrapolates cleanly; at p=1 the intercept lands ~1.5% above beta because the underlying large-k solves are not fully converged in a single-core sandbox (the raw values *rise* with N -- solver suboptimality, not finite size, and the near-optimal small-k point is consistent with beta). This makes explicit that the extrapolation machinery is ready and the remaining need for the campaign is converged large-k solves (more budget or an LKH oracle); the extrapolated two-NN floor brackets the truth in the meantime.



Follows the torus work: with the boundary confound removed, the remaining obstacle at large N is solver noise/suboptimality, so this adds the Percus-Martin control variate and certified lower bounds.

- New `--control-variate[=bool]` computes, per instance, the two-nearest-neighbor lower bound `L >= (1/2) sum_i (d1_i + d2_i)` on (a) the full point set — whose mean is known, so it acts as a variance-reducing control variate for the reported mean — and (b) the solved subset — a certified lower bound on the found tour that brackets `f(p)` from below and exposes the tour-to-bound gap. The full bound is free from the existing KNN; the subset bound builds a small KNN over just the selected points, reusing the torus metric via a new `Instance::explicit_side` (so the subset's minimum-image distances use the parent torus period, not the subset's bounding box). Off by default; the open-square/default path stays bit-identical (quality canary unchanged).
- `E[B_full]` is pinned by a cheap Monte-Carlo pass (KNN only, no solve; `--cv-mc-samples`, capped at large N) using an independent RNG stream. Empirically `E[B_full]/N = 0.6249` on the torus, matching the analytic Poisson value `(E[d1]+E[d2])/2 = (0.5+0.75)/2 = 0.625` — an independent check on both the bound and the estimator.
- Per p the summary now reports the mean subset lower bound (a bracket on `f(p)`), the mean gap to that bound, and the control-variate-corrected mean with its (reduced) standard error and the achieved variance-reduction fraction (`corr^2`). Measured behavior matches the theory: strong reduction at `p=1` (the `beta` anchor, where the full-set bound and the correlated subset bound coincide), and honestly ~0% at small `p` — an empirical finding in itself, since a cheap density-based subset proxy was tested and does *not* reproduce the optimizer's structure, so the strongly-correlated subset bound has no closed-form mean to exploit there. All quantities are emitted as optional JSON fields (schema updated) and covered by a new `test_control_variate_bounds` (subset bound lower-bounds the tour on every instance; subset bound equals the full-set bound at `p=1`; `E[B_full]/N ~ 0.625` on the torus).



Motivated by the second overnight run, where a k-ladder at fixed `p` revealed that finite-size **boundary** effects dominate the estimate of `f(p)`: on the open square the mean tour length per point converges to its limit only as `O(1/sqrt N)` (a large surface correction; Percus & Martin, PRL 76, 1188, 1996), so a fixed-`k` slice never reaches `f(p)` and the naive small-`p` rise was partly a finite-size artifact. On a flat torus the surface term vanishes, leaving `O(1/N)` corrections, and the limit is provably unchanged (Jaillet 1993) — so the torus reaches `f(p)` at far smaller `k` with a clean `1/N` extrapolation.

- New `--periodic[=bool]` flag generates instances on a flat torus of side `sqrt(N)`. `Instance::dist2` uses the minimum-image convention; the grid KNN gains an exact periodic path (wrapped-ring expansion with a conservative, never-early stopping bound and a brute-force fallback), and `build_grid` snaps the grid to tile `[0, side]^2` exactly so cell-index wrapping matches the torus period. The open-square path is byte-for-byte untouched (the quality canary reproduces its reference means exactly), and `periodic` is echoed into the JSON config (optional schema property).
- Correctness: a new `test_knn_periodic_property` checks the periodic grid KNN against a minimum-image brute force across sizes spanning both the fallback and wrapped-ring regimes, plus an explicit wrap-around-neighbor assertion. Independently, an exhaustive external check compared 644,800 neighbor distances (N up to 6000 x 4 seeds) against a hand-written minimum-image reference with zero mismatches, under ASan/UBSan.
- Validation (full TSP, `p=1`): the torus tracks the BHH constant `beta ~ 0.7124` far more tightly than the square at every N (e.g. at N=1000, torus `L/N = 0.720` vs square `0.739`), and the **difference** square minus torus — the pure boundary term — fits `0.61/sqrt(N)` with R^2 = 0.95, exactly the Percus-Martin surface scaling. (A residual upward drift of the torus values at large N under fixed restarts reflects solver suboptimality, not finite size, and motivates the planned control-variate and budget work.)

### 2-opt shorter-side reversal, tunable SA schedule, portability

- `apply_two_opt` now reverses whichever arc of the cycle is shorter (inner or the complementary wrapping arc), bounding reversal work at k/2 instead of always paying the inner-arc length. Reversing either arc yields the same undirected tour, verified by a new test (`test_two_opt_shorter_side_reversal`) checking valid tour, exact length, and presence of the two new edges for both branches. Measured ~19% faster on far-apart 2-opt applies at k=20000; quality-neutral across seeds (+0.03% over 15 seeds x 4 instances). Benefit concentrated at large k (p=1, scale runs); no effect on the small-p regime where reversals are already short. The changed array layout shifts the deterministic SA trajectory, so the quality canary was re-baselined (and hardened from 3 to 8 instances for stability).
- New `--sa-t0` / `--sa-t1` flags expose the SA temperature schedule endpoints (defaults 1.4 / 0.00005 reproduce the historical hardcoded schedule bit-for-bit). Useful for small-p tuning: the default schedule runs cold there (measured ~1.8% acceptance at p=0.02); a warmer schedule (`--sa-t0 3.0 --sa-t1 0.001`) raised acceptance to ~5.9% and improved quality ~0.4% at p=0.02. Echoed in the JSON config (optional schema properties).
- Portability/provenance: added a compile-time `static_assert(__cplusplus >= 201703L)` guard (turns the earlier MSVC `/Zc:__cplusplus` fix into a hard requirement) and a portable x86 CPUID processor-brand fallback for `cpu_model`, so Windows reports the real CPU instead of `unknown` (the overnight run's metadata showed `cpu_model: unknown` on MSVC).

### SA-inner candidate-table cadence and per-row diagnostics

- The SA loop previously rebuilt the subset candidate table at every 2-opt checkpoint via `maybe_subset_candidates`. At small p this rebuild is a dominant cost (measured ~1000 SA-steps per rebuild at p=0.005, because the sparse subset forces the grid ring search to expand far). Since candidate tables are quality-neutral (the 2-opt recomputes the true delta and only applies improving moves), the table is now built once per restart and reused across checkpoints; added members fall back to the full KNN list. Bit-identical results confirm behavior neutrality; wall-clock impact is small in current configs because the checkpoint fires rarely, but the wasteful repeated rebuild is removed and the change is deterministic.
- New per-p diagnostics: `executed_restarts` and `solve_seconds` on instance p-rows, and `executed_restarts_max` / `solve_seconds_total` on summary rows (all optional schema properties). Combined with `best_restart`/`best_restart_max`, these make per-curve-point convergence directly readable — if `best_restart_max` equals `executed_restarts_max - 1`, that p was still improving when its budget ran out. `solve_seconds_total` shows how wall-clock is distributed across the curve, guiding budget allocation.

### Elite-aware anytime restarts (subset-level ILS)

Motivated by the overnight run, where the p=0.02 row was still improving at cold restart #458 — the anytime (time-budget) tail launched hundreds of from-scratch restarts that never learned from each other.

- In time-budget mode, restarts beyond the scheduled count now alternate between cold seeds and *perturbed elite* seeds: pick one of the top few elite subsets and apply a small (~8%) spatially-coherent ruin-and-recreate kick (each removed member replaced by a KNN neighbor of a retained member), then run the normal polish + SA. This turns the anytime tail into subset-level iterated local search that intensifies around good solutions while alternating cold restarts preserve diversity.
- Only active in time-budget mode (already documented as non-reproducible); the deterministic scheduled-restart path is untouched and restart-thread invariance still holds (verified by test). New `--disable-elite-restarts` flag and `elite_restarts` search-stat (optional schema property) for A/B testing and diagnostics.
- Measured (multi-seed, Aldous small-p regime): net improvement at p=0.01 (-1.0%), p=0.02 (-0.8%), p=0.08 (-0.9%), and mid-p p=0.5 (-0.3%); neutral where the budget is too short for anytime restarts to engage. An initial naive version (20% uniform-random kick) regressed quality by corrupting structure; the gentler spatially-coherent kick fixed it.

### Or-opt engine overhaul (first-improvement + don't-look bits)

Motivated by profiling the 15.8 h overnight production run, which showed or-opt performing 9-47x more candidate evaluations than 2-opt at a 0.0002-0.002% hit rate (up to 1.1 trillion scans in a single phase) — because the old or-opt was best-improvement, rescanning all k nodes every pass to apply a single move, with no don't-look bits and a hard cap of `max_passes` applied moves per call.

- Rewrote `or_opt_1_candidate_descent` and `or_opt_segment_candidate_descent` (Or-2/Or-3) as first-improvement descents with per-node don't-look bits and reverse-KNN wakeups, mirroring the candidate 2-opt kernel. Moves are applied immediately and only affected nodes are re-woken, so retired nodes are not rescanned.
- The descents now run to a true candidate-local optimum instead of stopping after `max_passes` moves. Discovered during testing that reverse-KNN wakeups cover the spatial (KNN) neighborhood but not the `+/-3` positional offsets the descents also scan, which could strand an improving move; fixed with clear-restart confirming passes (repeat all-awake descents until one finds nothing), guaranteeing a candidate-local optimum regardless of wake-set completeness.
- New `test_or_opt_reaches_local_optimum`: independently verifies no improving spatial relocation remains after the descent.
- Measured (fixed work, single thread): at large k (p=1, N=6000) or-opt scans dropped ~2.1x (292M -> 133M) while applying ~2.2x more improving moves and improving quality (0.74910 -> 0.74443, -0.6%); at the Aldous small-p regime (fixed k=600, p=0.005) quality improved ~0.2-0.5% across seeds. The rewrite reaches better local optima *and* scans far less. Quality-canary references shifted slightly (all within tolerance) and were left as-is.

### Restart parallelism, second sweep, and convergence diagnostics

- Subset restarts now run in waves of `--restart-threads` worker threads (CLI default 0 = auto from the leftover thread budget after instance workers; library default 1 = sequential). Each restart uses its own deterministic RNG stream derived from the restart index; outcomes are merged in restart-index order, so results are **bitwise invariant to `restart_threads`** outside time-budget mode (unit test + CLI check). In `--time-budget-per-p` mode, more threads execute more restart waves per unit wall-clock. Real speedup could not be benchmarked in the single-core development sandbox; correctness and invariance are fully verified.
- The per-restart RNG streams replace the previous single shared stream across restarts. **Search trajectories changed once as a result**: measured neutral-to-slightly-better on the quality-canary workload (all three per-p means improved: 0.5735 -> 0.5683, 0.7138 -> 0.6993, 0.6829 -> 0.6764); canary references re-baselined.
- New `--second-sweep`: after the descending warm-start sweep over `p`, an ascending sweep re-solves each `p` seeded from the cheapest-insertion-grown best solution at the next smaller `p` and keeps the better result per row (never-worse by construction; `p = 1` skipped). **This is the one decisively quality-positive search change of this release cycle**: on `N=500` four-point curves (4 instances, seeds 21/99) it improved mid-`p` means by 4-9% (`p=0.2`: 0.63374 vs 0.67671 and 0.62775 vs 0.68660), 18 per-instance wins / 0 losses, at roughly +30% subset wall-clock - consistent with the release-wide finding that quality here is search-diversity/budget-limited rather than neighborhood-limited.
- New convergence diagnostics: `SolveResult::best_restart` (best-producing restart index, also set for full TSP solves), exposed as optional schema-`13` properties `best_restart` on per-instance `p` rows and `best_restart_max` on array-form summary rows. A `best_restart_max` at the last executed restart signals an undersized budget for that `p`.
- `restart_threads` (resolved) and `second_sweep` are echoed in the JSON `config` object as optional schema-`13` properties, following the same artifact-compatibility policy as the budget fields.

### Budget controls (new CLI flags)

- `--sa-iters-per-k <int>`: the effective SA budget for a size-`k` solve becomes `sa_iters + sa_iters_per_k * k` (saturating; default 0 preserves the flat historical budget). Deterministic.
- `--time-budget-per-p <seconds>`: anytime mode; the subset and full-TSP restart loops run at least their configured restarts and then keep launching restarts until the wall-clock budget per `(instance, p)` solve elapses. The final restart runs to completion. Documented as intentionally not bitwise-reproducible across machines; executed restart counts are recorded in `search_stats`.
- Measured on the six-point `N=500` curve (2 instances, seed 7, single thread): defaults 15.0s with means `p=0.1: 0.70727`, `p=0.2: 0.71660`, `p=0.5: 0.68786`; `--sa-iters-per-k 1500` 30.1s improves these to `0.66418 / 0.67032 / 0.64723` (up to -6.1%); `--time-budget-per-p 2.5` 32.6s improves `p=0.05` from `0.60035` to `0.57840` and mid-`p` similarly. Small-`p` and `p=1` endpoints were already converged and are unchanged.
- Both options are echoed in the JSON `config` object. They are declared as *optional* properties within schema `13` so that bundled validation artifacts (including real-LKH oracle runs that require a local LKH installation to regenerate) remain valid. **Release checklist:** the next release should bump the schema to 14, move both fields (and the restart/sweep/diagnostic fields below) into `required`, regenerate `validation_runs/` artifacts with the release binary (requires a local LKH for the real-oracle runs), refresh `docs/known_good_benchmarks.md`, update `VALIDATION_LOCAL.md`, re-baseline `tests/quality_canary.py` if search behavior changed, and flip the macOS/Windows CI jobs to blocking once observed green.

### Subset-restricted candidate lists and dual-orientation 2-opt

- New `SubsetCandidateTable`: exact per-member lists of the 16 nearest *subset* members, built via an expanding-ring grid search in ~`O(m * N)` per build (brute-force fallback without a grid). Candidate 2-opt and or-opt use it in polish, subset swap descent, and the SA inner polish; tables are rebuilt after membership changes.
- Candidate 2-opt now scans both edge orientations (successor and predecessor edges of anchor and candidate), completing the standard neighbor-list discovery guarantee.
- New Or-2/Or-3 segment relocation descent in polish: best-improvement relocation of 2- and 3-node segments in both orientations, candidate insertion edges from the subset table (or KNN) around both segment endpoints. O(1) evaluations against the edge cache; O(k) tour rebuild per applied move.
- **Measured honestly: all three neighborhood extensions are quality-neutral at default budgets.** Tables + orientations: within a ±1% noise band, 12 wins / 24 losses across 36 per-instance comparisons at `N=500`, and -0.9% to +0.4% at `N=2000`-`8000` including regimes with `k` above the exhaustive-final threshold and KNN lists holding ~4 subset members. Or-2/Or-3: 16 wins / 16 losses at `N=500` within ±1.5%, +0.9%/0.0% at `N=2000`/`N=8000`, at ~0-1% wall-clock cost. This contradicts the review's prediction that candidate starvation was a major quality lever: at these budgets, quality is dominated by global search budget, not neighborhood reach. The changes are kept because they are exact (unit-tested against brute force), cost-neutral in wall-clock, and structurally complete the standard candidate-search toolkit.

### Portability and CI

- Fixed: on MSVC the JSON `build_metadata.cplusplus` field reported `199711` (MSVC's legacy `__cplusplus` value), violating the schema's `minimum: 201703`. The core target now sets `/Zc:__cplusplus` on MSVC so the macro is truthful. Found by the first real MSVC production run (15.8 h overnight stress on a Ryzen 5 5600X, which otherwise completed with zero errors).

- Renamed the public `SummaryRow`-level `stderr` member to `stderr_value`: `stderr` is a macro in `<cstdio>` on some platforms (notably MSVC), which made the previous declaration ill-formed there. The JSON field keeps the name `"stderr"`, so output format, schema, and bundled artifacts are unchanged. Library API break for direct users of that member.
- Added non-blocking `macos-appleclang-release` and `windows-msvc-release` CI jobs (`continue-on-error: true` until observed green on real runners, then flip to blocking). The MSVC job exercises the build that the `stderr_value` rename unblocked.
- The two unit tests that spawn a POSIX shell script as a fake LKH binary (`test_oracle_parser_and_fake_lkh`, `test_oracle_top_n_matches_cli_config`) are now registered only on non-Windows hosts, matching the fake-oracle CLI smoke test. Found by the first real Windows `ctest` run (the build and all 33 other unit tests plus `--self-test` passed; only the shell-script oracle tests failed, since Windows cannot execute the `#!/bin/sh` helper). The oracle parsing/config logic is otherwise platform-independent; only the process-launch path is skipped on Windows.
- The fake-oracle CLI smoke test (a POSIX shell script) is now registered only on non-Windows hosts; the oracle parsing logic remains covered on all platforms by the C++ unit tests.
- New `quality_canary` CTest (Python-test suite): a fixed-seed `N=250` three-p workload asserting each per-p mean stays within 6% of recorded references — coarse enough to absorb cross-compiler floating-point trajectory differences, tight enough to catch broken neighborhoods or budget plumbing. Re-baselining instructions are in `tests/quality_canary.py`.
- New `test_solver_matches_exact_enumeration_tiny` unit test: on `N=10, k=5` instances the solver must match the global optimum over all `C(10,5)` subsets x exact cycles — a deterministic micro-scale optimality canary.
- README/algorithm notes now state the measured reality of the native/AVX2 build: `-march=native` was ~38% *slower* than the default build on an AVX-512 GCC host; the kernel stays off by default and should be benchmarked per target.

### Path relinking and hot-path performance

- Rewrote path-relink step selection as an exact decomposition (`path_relink_best_step`): removal gains from the edge cache plus top-3 pre-removal insertion edges per added node give the true best (remove, add) pair in `O(|add| * k + |remove| * |add|)` per step, replacing the naive cross-product that copied the full tour and ran an `O(k)` scan per candidate pair. Moves are now applied incrementally.
- Skip relinking elite pairs with symmetric difference above 64 nodes; such relinks dominated runtime at mid/high `p` while contributing negligible quality. Skips count as attempts but not feasible relinks.
- Removed a dead `O(N)` copy of the membership bitmap that ran on every simulated-annealing move (`collect_add_candidates`), and added an allocation-free `collect_add_candidates_into` used by the SA hot loop.
- `find_best_insert_after_remove`, `evaluate_swap_after_remove`, and `evaluate_move_after_remove` now reuse thread-local scratch buffers; the evaluators compute distances on demand when given a short predecessor candidate list instead of a batched pass over all tour nodes.
- Removed the now-unused tour-copying `best_insert_after_remove` helper.
- Measured on fixed seeds (single thread, Release): `N=1000, p=0.5` defaults 50.3s -> 1.9s at equal quality; six-point `N=500` curve 23.3s -> 14.7s with bit-identical means; `N=4000, p=0.02` 1.76s -> 1.48s; `N=500, p=0.1` at 8 restarts / 200k SA iterations 6.37s -> 5.44s with identical mean. Reinvesting the freed budget (`--restarts 12 --sa-iters 600000`) at the old default wall-clock improves the `N=1000, p=0.5` estimate from 0.67872 to 0.63954.

### Tests

- White-box regression tests: decomposed relink step selection matches brute-force best-pair evaluation; symmetric-difference cap skip/run/counter semantics; subset candidate tables match brute-force m-nearest-member reference on both KNN backends; effective SA budget linearity and overflow saturation; time budget launches additional restarts and never worsens the solution; table-driven 2-opt preserves tour invariants and incremental lengths; segment or-opt preserves permutation/invariants/length accounting; tiny-instance solve matches exhaustive subset enumeration.

## CV/interview polish (source layout and public API)

- Added object-oriented facade classes: `TspSolver`, `SubsetSolver`, and `ExperimentRunner`.
- Moved the executable entry point to `apps/` so `src/` is implementation-focused.
- Moved the generated-version template from `include/` to `cmake/`.
- Added `docs/architecture.md` and a complete `examples/library_usage.cpp`.
- Updated README language from "refactor package" to "solver" and documented the intentional blend of OO domain types with free-function hot search kernels.


## 0.8.7

- Regenerated bundled validation artifacts with the current executable/schema.
- Added `validation_artifacts_schema` regression coverage so stale bundled result JSON cannot silently ship.
- Updated validation-report rendering to clearly separate fake-original routing checks from true upstream solver parity.
- Adjusted benchmark/oracle/profile scripts so bundled validation commands can preserve relative paths instead of environment-specific build paths.

## 0.8.6

- Added structured `build_metadata.effective_optimization_level` to result JSON and bumped the strict schema to version 13.
- Added JSON escaping/output regression coverage for quotes, backslashes, whitespace escapes, control characters, tiny numeric values, and the new build metadata field.
- Added `scripts/render_validation_report.py` and regenerated `docs/known_good_benchmarks.md` from the bundled validation manifests.
- Updated research/output documentation so benchmark, ablation, backend-parity, exhaustive-policy, original-compatibility, and real-oracle status are summarized in tables derived from `validation_runs/`.

## 0.8.5 validation addendum

- Added concrete validation artifacts under `validation_runs/`, including backend parity, all-suite benchmark, exhaustive-policy comparison, profiling smoke, real-oracle skip manifest, original-compatible parity smoke, and upstream clone-attempt log.
- Updated `VALIDATION.md` and `docs/known_good_benchmarks.md` with the observed validation results.
- Validated 19 generated result JSON files against the strict result schema.

## 0.8.5

- Fixed thread default semantics: `--threads 0` now means auto, negative values are rejected, and the default matches the help text.
- Cleaned install rules so `version.hpp.in` is not installed and `cli.hpp` is installed only when CLI support is built.
- Added CLI and install-tree regression tests for the patch-level cleanup.
- Removed split-implementation placeholder source files from the package.

## 0.8.4

- Added explicit boolean parsing for CLI flags. Boolean flags now accept plain presence as true, or `=true`, `=false`, `=yes`, `=no`, `=on`, `=off`, `=1`, and `=0`.
- Fixed `--quick` preset precedence so explicit options override the preset regardless of argument order.
- Added CLI regression coverage for boolean flags, invalid boolean values, and quick preset precedence.
- Added `docs/known_good_benchmarks.md` to collect known-good validation commands and document pending external parity/oracle checks.

## 0.8.3

- Added P2 reproducibility instrumentation: effective compile-flag and target-option metadata, effective KNN backend/cell telemetry, and optional per-instance result rows.
- Bumped result schema to version 12 and tightened schema coverage for the new metadata.
- Added CLI/schema regression coverage for `--include-instance-rows` and effective KNN telemetry.

## 0.8.2

- Made exhaustive 2-opt final-only by default through `--exhaustive-two-opt-policy final-only`. The legacy expensive behavior remains available as `--exhaustive-two-opt-policy all-polish`, and exhaustive 2-opt can be disabled with `never`.
- Added `scripts/benchmark_exhaustive_policy.py` to compare final-only against all-polish on deterministic scenarios and report wall time, solution quality, and two-opt scan deltas.
- Added an `exhaustive_policy_benchmark_smoke` Python CTest target.
- Added a CI low-memory build smoke that configures with `ALDOUS_TSP_LOW_MEMORY_BUILD=ON`, builds the CLI, and runs a dry-run command.
- Bumped result schema to version 11 and records `config.exhaustive_two_opt_policy`.

## 0.8.1

- Made `scripts/benchmark_parity.py` original-prototype compatible via `--baseline-kind original|current|auto`. Original-compatible runs avoid current-only flags such as `--p-values`, `--knn-backend`, and neighborhood ablation options.
- Added an `original_parity_compat_smoke` CTest target using a fake older CLI that fails if current-only flags leak into the baseline command.
- Improved `scripts/oracle_real_smoke.py` so skipped real-oracle checks write an explicit manifest, and real solver runs validate oracle-call records/statuses.
- Added an optional `real_oracle_smoke_optional` CTest target that runs real LKH/Concorde smoke when solvers are available and records a skip manifest otherwise.
- Fixed a sanitizer-detected wraparound bug in high-p reference-neighborhood indexing for tiny/default-p compatibility runs.
- Documented that true original-vs-current parity and real-oracle validation were attempted in the packaging environment but remain dependent on external executables.

## 0.8.0

- Restored optimized Release builds by default. The previous unconditional `-O0 -g0` source-file override is now gated behind `ALDOUS_TSP_LOW_MEMORY_BUILD=ON`.
- Added `release-low-memory` CMake preset for constrained builders while keeping normal Release builds optimized.
- Added build metadata fields for `low_memory_build` and `optimization_profile`.
- Split high-p reference-guided one-for-one exchange counters from generic subset-swap counters with `highp_exchange_scans` and `highp_exchange_improvements`.
- Added `knn_build_seconds` to search statistics for phase-level profiling.
- Added `summary_rows`, an array-form summary with explicit `p`, `key`, `k`, statistics, and values, while retaining the legacy `summary` map for compatibility.
- Added `scripts/profile_run.py` for lightweight phase/counter profiling.
- Added `scripts/oracle_real_smoke.py` for optional real LKH/Concorde smoke checks when external solvers are installed.
- Updated benchmark manifests to include high-p exchange counters and phase timing columns.
- Bumped strict result schema to version 10.

## 0.7.0 - P3 maintainability pass

- Split the monolithic solver implementation into focused translation units for construction, local search, neighborhoods, seeding, subset orchestration, and TSP orchestration.
- Split the CLI implementation into parser, per-instance execution, self-test, and main-run modules.
- Added a small dependency-free JSON writer helper to centralize string escaping, floating-point serialization, and array emission while preserving the project's no-runtime-dependency design.
- Updated build metadata and documentation for the modular source layout.

## 0.6.1

- Added per-oracle-call diagnostics to result JSON, including solver, status, reason, before/after length, gain, and runtime.
- Split path-relink instrumentation into attempts, feasible relinks, elite insertions, and best-solution improvements.
- Added build and runtime metadata fields, including build options, C++ standard, CPU model, and hardware thread count.
- Expanded benchmark manifests with ablation scenarios and updated the strict JSON schema to version 9.

## 0.6.0

- Added an MIT `LICENSE` file.
- Enabled Python regression tests in CI, including schema validation.
- Tightened `schema/results.schema.json` with enum constraints, required config/stat fields, and `additionalProperties: false` for known objects.
- Added a dedicated schema-validation CTest target for generated result JSON.
- Documented original-vs-current parity status and the exact command needed when an original executable is available.

## 0.5.1

- Fixed grid-KNN pruning for very small coordinate scales with scale-aware bounds and safe initial-radius computation.
- Preserved tiny numeric values in JSON output using high-precision non-fixed formatting.
- Changed custom p-value canonicalization to deduplicate only exact duplicates.
- Added tiny-coordinate KNN, JSON precision, and tiny-p-value regression coverage.

## 0.5.0

- Made ablation flags exact for solver-controlled local search, including SA cleanup and subset-swap descent two-opt calls.
- Added a grid-cell safety cap so pathological `--grid-cell` values are enlarged instead of causing huge allocations or integer overflow.
- Made `--oracle-tsp-top` and `--oracle-subset-top` control posthoc oracle polishing over the top elite candidates.
- Added CLI regression tests for exact two-opt ablation, tiny forced grid-cell safety, and oracle top-N behavior.
- Ran internal grid-vs-brute-force parity scenarios and documented that upstream original-prototype parity requires a reachable baseline executable.

## 0.4.0

- Fixed grid KNN for arbitrary finite coordinate ranges supplied by `Instance::set_points()`.
- Reintroduced reverse-KNN/don't-look wakeups in KNN-candidate 2-opt.
- Added batched distance scoring through `dist_many_from()`, including an AVX2 path when available.
- Added incremental tour operations for 2-opt, node moves, and subset swap moves.
- Added property tests for arbitrary-coordinate grid KNN, batched distance scoring, incremental tour mutations, and randomized exact-small-TSP validation.
- Added `scripts/benchmark_parity.py` for grid/brute-force backend parity and optional original-prototype comparison.
- Adjusted Release compile options to keep the dense solver translation unit buildable in constrained CI/container environments.

## 0.3.0

- Re-integrated optional LKH/Concorde oracle post-processing.
- Added TSPLIB matrix/EUC_2D export, process launching, timeout handling, and tour parsing.
- Added fake-LKH unit and CLI smoke tests.
- Added CMake package config generation for `find_package(aldous_tsp CONFIG)`.
- Expanded JSON schema and search statistics with oracle metadata.
- Added a benchmark runner script for repeatable scenario sweeps.

## 0.2.0

- Refactored project into core library, CLI library, and executable.
- Added namespaced public API.
- Added CMake presets, sanitizer option, and CI matrix.
- Added exact brute-force KNN baseline and verification tests.
- Added collision-safe elite deduplication.
- Fixed SA temperature schedule edge case.
- Added configurable p-grid and dry-run support.
- Added atomic JSON output, schema, and plotting improvements.
