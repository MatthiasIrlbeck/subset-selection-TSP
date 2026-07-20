# Result JSON schema

The JSON schema is stored in `schema/results.schema.json`. It is intentionally strict: top-level fields, config fields, search-stat fields, oracle-call records, legacy summary rows, and array-form summary rows reject unknown properties. Current native schema version: `14`. Historical schema-13 documents remain valid against `schema/results-v13.schema.json`.

## Migrating schema 13

Use `scripts/migrate_schema13_to14.py` to convert a historical result without rewriting its original run or build metadata:

```bash
python3 scripts/migrate_schema13_to14.py old.json --output old.v14.json
python3 scripts/migrate_schema13_to14.py old.json --in-place
```

Schema 13 did not serialize every schema-14 field. The migrator records all synthesized values under `migration_metadata.inferred_fields` and records provenance that cannot be reconstructed under `migration_metadata.unrecoverable_fields`. In particular, schema 13 did not reliably identify periodic geometry or stable point/search streams. Supply `--periodic` or `--non-periodic` when that fact is known; otherwise the tool uses the compatibility default `false` and marks the field unrecoverable. Migrated documents are schema-14-valid, but missing historical replicate identities cannot be made suitable for block-bootstrap or multifidelity analysis merely by conversion.

Top-level fields include `schema_version`, `run_metadata`, `build_metadata`, `N`, `done`, `target`, `threads`, `wall_seconds`, `mode`, `distance_backend`, `oracle_status`, `p_values`, `config`, `search_stats`, `oracle_call_records`, `summary`, `summary_rows`, and `instance_rows`. Migrated documents additionally carry `migration_metadata`.

`run_metadata` records project/runtime metadata: project version, detected git commit, compiler string, platform, CPU model, and hardware-thread count.

`build_metadata` records configured build type, CMake generator/version, native/sanitizer/warning/Werror/Python-test build options, whether the low-memory compiler profile was enabled, the named optimization profile, configured/effective C++ flags, target/source compile options, the structured `effective_optimization_level`, and the active `__cplusplus` value.

`config` records the complete resolved user configuration: campaign identity, solver budgets, KNN backend, verification count, exhaustive 2-opt policy, oracle mode/resolution, external executable path/version when available, TSPLIB format, oracle limits, output durability, and ablation flags. `exact_subset_max_n` records the optional global exact-solver threshold (`0` means disabled; the hard maximum is `18`). Schema 14 requires every configuration field emitted by the native executable, while the archived schema-13 definition remains available only for historical artifacts.

`search_stats` contains move counters, phase timing counters, effective KNN backend/cell telemetry, aggregate oracle counters, and restart accounting. `exact_subset_calls`, `exact_subset_solved`, `exact_subset_states`, and `exact_subset_transitions` expose global exact-solver work. Generic subset-swap counters are separated from high-p reference-exchange counters: `subset_swap_scans`, `subset_swap_improvements`, `highp_exchange_scans`, and `highp_exchange_improvements`. The dedicated `region_restarts` and `dense_restarts` counters prevent those seed kinds from being folded into `random_restarts`; historical schema-13 documents may omit them.

The fields `config.pair_exchange_max_k` and `search_stats.pair_exchange_skipped_large_k` record the temporary large-cardinality safety gate for the two-for-two neighborhood. A value of `0` removes the gate; current builds default to `5000`.

`search_stats.phase_timing` reports accumulated worker elapsed-seconds for seed/TSP construction, initial and final polish, annealing, checkpoint polish, post-SA polish, subset swap, high-p exchange, pair exchange, ruin/recreate, membership ejection chains, the exact subset DP, path relinking, TSP ILS, and external-oracle work. `exact_subset_seconds` is the exact solver's elapsed worker time. These values can exceed top-level wall time when restart workers run concurrently. `sa_checkpoint_polish_seconds` is nested inside `sa_seconds`, so phase values are diagnostic rather than a disjoint accounting identity. Proposal and insertion latency are sampled deterministically every 64 SA iterations; use each `*_sample_seconds / *_samples` ratio rather than treating the sampled seconds as total phase time. The field remains optional in archived schema version 13; native output uses schema version 14, whose generated `config` object is strict and complete.

Conditional tour-bound fields use the `conditional_*` prefix. The two-NN and Held-Karp values lower-bound the optimal tour through the subset selected by the heuristic. They do **not** lower-bound the optimum over all size-`k` subsets and therefore do not bracket `f(p)`. Native schema-14 JSON still emits `subset_bound*`, `lower_bound_gap_mean`, and `held_karp_bound*` as deprecated aliases for older analysis consumers.

The schema constrains solver/oracle strings with explicit enums: `mode`, `distance_backend`, `knn_backend`, `exhaustive_two_opt_policy`, `oracle_mode`, `oracle_resolved`, and `oracle_format`.

Path-relink instrumentation is split into separate counters: `path_relink_attempts`, `path_relink_feasible`, `path_relink_elite_insertions`, and `path_relink_best_improvements`. The legacy `path_relink_improvements` field is retained as an alias for best-solution improvements.

The optional diversity-aware subset archive is configured by `elite_diversity_slots`, `elite_min_jaccard`, and `elite_quality_slack`. Its supplemental entries never replace the ordinary length-ranked archive capacity. `elite_diversity_candidates`, `elite_diversity_retained`, and `elite_diversity_rejected` report how unique set candidates passed through that supplemental policy. Relinking still includes the complete ordinary `path_relink_top` prefix before considering any supplemental entry.

Membership-chain configuration is recorded as `ejection_chain_starts`, `ejection_chain_depth`, `ejection_chain_candidates`, `ejection_chain_remove_cap`, and `ejection_chain_max_uphill`. Search telemetry separates attempted starts, nonempty feasible chains, applied steps, exact candidate scans, accepted improvements, and the sum of accepted prefix depths. `ejection_chain_seconds` is accumulated worker elapsed time, like the other neighborhood phase fields.

The top-level `oracle_call_records` array contains one object per attempted external oracle polish. Each record includes the problem type (`tsp` or `subset`), `k`, solver, TSPLIB format, status, executable path, error detail, before/after length, gain, and elapsed seconds.

`summary` is retained for compatibility as an object keyed by full-precision p-value strings. `summary_rows` is the preferred research/analysis shape: it is an array where every entry explicitly contains `p`, the legacy `key`, `k`, `mean`, `std`, `stderr`, `min`, `max`, `n`, and `values`.

`exact_optimal_instances` in each summary entry counts point-set instances whose
complete cardinality-`k` problem was globally proven by the exact DP.
`instance_rows[].p_results[].exact_optimal` carries the corresponding per-row
proof flag. An exact row has `executed_restarts = 0`, `best_restart = -1`, and
omits the parallel restart arrays because no heuristic restart was executed.
This is distinct from a conditional tour lower bound or from exact ordering of
one fixed heuristic subset.

Schema 14 makes the complete generated configuration mandatory and adds campaign/restart provenance used by current analysis. Schema 13 added `build_metadata.effective_optimization_level`, a structured optimization-level field that makes release, low-memory, sanitizer, and debug output easier to compare without interpreting raw compiler flag strings. Version 12 added `instance_rows` for optional per-instance values/statistics, effective KNN backend and cell-size telemetry in `search_stats`, and build metadata fields for effective compile flags and target compile options.

## Per-restart fields

With `--include-instance-rows`, every successful `instance_rows[].p_results[]` entry carries one aligned record per executed restart. Internally, the C++ API stores these as `RestartRecord` objects; JSON retains parallel arrays for compatibility with existing analysis code.

| field | meaning |
|---|---|
| `executed_restarts` | number of executed restarts; equal to the length of every restart array in the row |
| `best_restart` | zero-based index of the lowest pre-postprocessing restart length in the complete serialized record population; `-1` only when no restart ran |
| `restart_values` | final raw restart length divided by `k`, in execution order |
| `restart_kinds` | stable kind code for each record; the authoritative code/name table is `include/aldous_tsp/restart_kinds.def` |
| `restart_sweeps` | `0` for the primary standalone/descending sweep and `1` for the optional ascending secondary sweep |
| `restart_roles` | controller role: `0` independent diagnostic, `1` continuation, `2` elite kick, `3` anytime, `4` raced production |
| `restart_variants` | stable zero-based variant within a seed kind and role; independent of execution order and worker count |
| `restart_promotion_stages` | racing state: `0` ordinary, `1` stopped after the pilot, `2` promoted and rerun at full depth |
| `restart_sa_iterations` | actual SA iterations allocated to the recorded candidate; a promoted raced entry includes its pilot plus full-depth rerun |
| `restart_centroids_x`, `restart_centroids_y` | metric-aware centroid of the selected nodes: circular mean for periodic coordinates and arithmetic mean for an open domain |
| `restart_radii` | mean metric distance of selected nodes to that centroid |

Restart-kind codes are:

| code | label | source |
|---:|---|---|
| 0 | `random` | uniformly sampled subset |
| 1 | `warm` | resized warm start |
| 2 | `small-p` | specialised small-`p` seed |
| 3 | `high-p-delete` | deletion from a larger warm set |
| 4 | `elite` | anytime elite-seeded ILS restart |
| 5 | `kick` | scheduled elite kick |
| 6 | `region` | local region seed |
| 7 | `dense` | dense compact seed |
| 8 | `tsp-farthest-insertion` | first full-TSP restart |
| 9 | `tsp-nearest-neighbor` | later full-TSP restarts |

For an ordinary solve, records are all primary and retain restart-index order. Independent records are generated from streams keyed by instance, cardinality, seed kind, and variant, so their values do not change when other `p` points are added to or reordered in the campaign grid. Under the default `supplemental` continuation policy, warm records are appended without removing an independent draw; `fixed-budget` reserves the configured warm quota inside `--restarts`. Optional deterministic racing is also supplemental: every race candidate receives a fixed pilot, a stable quality-and-diversity rule promotes a fixed number, and promoted candidates are rerun from exactly the same seed and RNG stream at full depth. Each candidate contributes one final record—pilot-only or promoted-full—and role `4` keeps the selected population out of the default independent EVT sample. With `--second-sweep`, the primary records are serialized first and continuation-only secondary records are appended; `best_restart` is then recomputed over that combined sequence. A `p = 1` row contains the full-TSP restart population, so its executed count and arrays are no longer zero/empty when TSP restarts ran.

The kind counters in `search_stats` are derived from the same typed kind assigned to each subset record. `random_restarts`, `warm_restarts`, `smallp_seed_restarts`, `highp_delete_restarts`, `region_restarts`, `dense_restarts`, and `kick_restarts` count their named records exactly. `elite_restarts` intentionally counts all elite-seeded restarts, so it includes both kind `4` and kind `5`; `kick_restarts` is the exact scheduled-kick subset. `racing_pilot_restarts` and `racing_promoted_restarts` expose the staged allocation, while `subset_restarts` counts unique recorded subset candidates rather than counting a promoted rerun twice. `tsp_restarts` remains the full-TSP count.

Restart outcomes are not automatically identically distributed. Seed kinds form a mixture, and continuation, elite-kick, anytime, and raced-production draws are selected or dependent. Tail or endpoint analyses should therefore filter or stratify by `restart_kinds`, `restart_sweeps`, and `restart_roles`; `scripts/restart_evt.py` exposes all three and defaults to role `0` (independent diagnostic). Older files without sweep metadata are interpreted as primary-only; files without role metadata are retained as an explicitly unknown legacy mixture.
