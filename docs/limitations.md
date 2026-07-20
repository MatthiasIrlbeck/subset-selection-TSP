# Limitations

- The production solver is heuristic by default and does not certify global subset optimality above the exact solver's hard cap. With `--exact-subset-max-n` enabled, instances through `N = 18` are globally solved under the implemented distance metric and carry explicit proof flags.
- **There is no plateau to find at small p.** Paired allocation scans at k=2000 (B=960, B=1920, 24 instances) show anneals converge by ~60 iterations/candidate, after which the search is pure independent multistart: doubling restarts buys a constant −0.0025 and doubling depth buys nothing (−0.0001 ± 0.0009). Reported small-p values are therefore best-of-m upper bounds that crawl logarithmically with budget, not converged estimates. Interpreting them requires extrapolating the restart-value distribution (see `restart_values` in the result schema), not running a longer ladder.
- Per-restart values are a **mixture** across seed kinds, not iid draws from one distribution. Secondary-sweep, continuation, elite, anytime, and raced-production draws can also be selected or dependent. Any tail fit must filter or stratify by `restart_kinds`, `restart_sweeps`, and `restart_roles`, or it is fitting a contaminated sample.
- The internal high-performance heuristic stack is restored in the cleaner structure. Internal grid/brute-force backend parity is automated; original-prototype parity still requires a supplied baseline executable and should be run before using results in a paper or report.
- Optional LKH/Concorde post-processing is integrated, but external solver behavior depends on the installed binary, TSPLIB interpretation, timeout settings, and integer scaling.
- Oracle output includes aggregate counters and detailed per-call status/error records. External solver correctness still depends on the installed executable and its TSPLIB interpretation.
- The JSON schema validates structure, not scientific adequacy of sample sizes.
- Large runs should report seeds, p-grids, KNN backend, oracle mode, all solver budgets, ablation settings, and `memory_plan`. The planner is conservative but cannot account for every allocator/runtime overhead; leave headroom instead of setting the process limit equal to the estimate.

## Results at small k do not transfer to k=2000

The 0.9.5 exploration-seed insertion fix measured -0.0109 (3.5 sigma) at k=200 and does
exactly nothing at k=2000, where the windowed and spatial kernels produce bit-identical
searches and the exact scan returns the same answer at +58% wall. The mechanism is scale:
the +/-12 insertion window covers 25% of a k=100 tour, 12% of a k=200 tour, and 1.25% of a
k=2000 tour, and the candidates are spatially local to the removal site in every case.
Sandbox A/Bs at k=100-200 resolve ~0.005-0.007 and are the wrong instrument for a k=2000
default. Any policy claim for the production scale must be measured AT the production
scale.

## Historical hotspot measurements are not current budgets

Older profiles in this repository predate the canonical torus kernel, batched
pair and membership exchange, adaptive multi-scale ruin/recreate, and membership
ejection chains. Their phase percentages must not be used to allocate current
campaign budgets. Use `search_stats.phase_timing` and paired fixed-seed canaries
on the intended production geometry and cardinality. In current targeted
canaries, adaptive ruin/recreate and ejection chains both produce measurable
quality gains, so neither should be classified as free or inert without a new
matched-compute ablation.
