# Generated option reference

This file is generated from `config/options.json`. Do not edit it directly.

## General

| CLI option | JSON key | Default | Constraints | Description |
|---|---|---:|---|---|
| `--help` | — | `—` | — | Show this generated help text and exit. |
| `--version` | — | `—` | — | Print the project version and embedded source provenance, then exit. |
| `--self-test` | — | `—` | — | Run built-in smoke and self tests. |
| `--quick` | — | `—` | — | Apply the small fast-run preset; explicit flags override it regardless of order. |
| `--verbose-p` | `verbose` | `false` | boolean | Print per-instance and per-p progress. |
| `--force` | `force_output` | `false` | boolean | Replace an existing output path. |
| `--dry-run` | `dry_run` | `false` | boolean | Validate and print the resolved configuration without solving. |
| `--dump-config` | `dump_config` | `false` | boolean | Print the resolved configuration before solving. |
| `--output` | `output_path` | `results.json` | minLength=1, maxLength=4096 | Output JSON path. |
| `--output-durability` | `output_durability` | `full` | enum=['none', 'file', 'full'] | Output durability: none, file, or full (file plus parent directory). |

## Simulation and campaign identity

| CLI option | JSON key | Default | Constraints | Description |
|---|---|---:|---|---|
| `--N` | `N` | `500` | minimum=3 | Number of random points. |
| `--instances` | `instances` | `15` | minimum=1 | Monte Carlo point-set instances. |
| `--threads` | `threads` | `0` | minimum=0 | Instance worker threads; zero selects an automatic bounded value. |
| `--memory-budget-mb` | `memory_budget_mb` | `0` | minimum=0 | Conservative peak-memory budget in MiB; zero derives a safe automatic budget from currently available physical/container memory. |
| `--periodic` | `periodic` | `false` | boolean | Use flat-torus periodic boundary conditions. |
| `--campaign-id` | `campaign_id` | `default` | minLength=1, maxLength=256 | Stable campaign identity. |
| `--campaign-shard` | `campaign_shard` | `0` | minimum=0 | Nonnegative campaign shard identity. |
| `--replicate-offset` | `replicate_offset` | `0` | minimum=0 | Campaign-global first replicate identifier. |
| `--point-seed` | `point_seed` | `--seed` | integer | Point-instance seed; defaults to --seed. |
| `--search-seed` | `search_seed` | `--seed` | integer | Heuristic-search seed; defaults to --seed. |
| `--solver-policy-id` | `solver_policy_id` | `default` | minLength=1, maxLength=256 | Stable solver-policy identity. |
| `--fidelity-level` | `fidelity_level` | `strong` | minLength=1, maxLength=256 | Fidelity identity such as cheap or strong. |
| `--p-values` | — | `—` | — | Comma-separated p grid, for example 0.02,0.05,1. |
| `--p-range` | — | `—` | — | Linear p grid from a to b with n values. |
| `--p-file` | — | `—` | — | Read whitespace- or comma-separated p values from a file. |
| `--seed` | `seed` | `2024` | integer | Base random seed. |

## Subset search

| CLI option | JSON key | Default | Constraints | Description |
|---|---|---:|---|---|
| `--second-sweep` | `second_sweep` | `false` | boolean | Run an ascending continuation sweep after the descending sweep. |
| `--exact-subset-max-n` | `exact_subset_max_n` | `0` | minimum=0, maximum=18 | Globally solve subset choice and tour when N is at most this threshold; zero disables it. |
| `--search-policy` | `search_policy_preset` | `legacy-balanced` | enum=['legacy-balanced', 'heldout-balanced', 'heldout-quality'] | Search controller preset. legacy-balanced preserves the 0.10 controller. heldout-balanced uses the held-out four-candidate SA policy through p=0.35; heldout-quality uses a deeper policy, extends periodic runs through p=0.50, and strengthens full-TSP screening. Explicit core SA or TSP population controls take precedence. |
| `--restarts` | `subset_restarts` | `-1` | minimum=-1 | Subset restarts; -1 selects the p-aware automatic policy. |
| `--continuation-restarts` | `continuation_restarts` | `1` | minimum=0 | Warm restarts when a neighboring-p parent exists. |
| `--continuation-policy` | `continuation_policy` | `supplemental` | enum=['supplemental', 'fixed-budget'] | Continuation policy: supplemental or fixed-budget. |
| `--racing-candidates` | `racing_candidates` | `0` | minimum=0 | Supplemental candidates screened by deterministic restart racing; zero disables it. |
| `--racing-survivors` | `racing_survivors` | `2` | minimum=1 | Pilot candidates promoted to a full-depth rerun. |
| `--racing-pilot-iters` | `racing_pilot_iters` | `2000` | minimum=0 | SA iterations used to screen each race candidate. |
| `--racing-min-jaccard` | `racing_min_jaccard` | `0.05` | minimum=0, maximum=1 | Preferred selected-set Jaccard distance between promoted race candidates. |
| `--staged-search` | `staged_search` | `true` | boolean | Reserve expensive post-SA neighborhoods for promoted finalists. |
| `--strong-polish-finalists` | `strong_polish_finalists` | `3` | minimum=1 | Finalists receiving the strong search stage. |
| `--strong-polish-min-jaccard` | `strong_polish_min_jaccard` | `0.02` | minimum=0, maximum=1 | Preferred selected-set distance among strong-polish finalists. |
| `--sa-iters` | `sa_iters` | `60000` | minimum=0 | Base subset-SA iteration budget. |
| `--sa-iters-per-k` | `sa_iters_per_k` | `0` | minimum=0 | Additional SA iterations per selected node k. |
| `--sa-iters-per-n` | `sa_iters_per_n` | `0` | minimum=0 | Additional SA iterations per candidate point N. |
| `--sa-exact-insertion` | `sa_exact_insertion` | `false` | boolean | Use the historical exact O(k) insertion scan for every SA move. |
| `--sa-insertion-window` | `sa_insertion_window` | `12` | minimum=1 | Tour-position radius for windowed SA insertion. |
| `--small-p-dense-fill` | `small_p_dense_fill` | `true` | boolean | At small p, fill the restart pool with dense-seed variants instead of uncontracted random subsets. |
| `--exploration-exact-insertion` | `exploration_exact_insertion` | `true` | boolean | Use exact insertion for spread-out exploration seeds. |
| `--sa-spatial-insertion` | `sa_spatial_insertion` | `—` | boolean | Use insertion slots from a live spatial index (default: false). |
| `--dense-exact-insertion` | `dense_exact_insertion` | `false` | boolean | Force exact insertion on dense seeds; this reproduces the old bit-identical path at k=2000 but costs about +58% wall. |
| `--sa-spatial-neighbors` | `sa_spatial_neighbors` | `16` | minimum=1 | Nearest current members allowed to offer insertion slots. |
| `--region-seeds` | `region_seeds` | `false` | boolean | Use localized region seeds in the explorer pool. |
| `--region-dilation` | `region_dilation` | `3.0` | minimum=1 | Region-seed candidate dilation. |
| `--kick-restarts` | `kick_restarts` | `0` | minimum=0 | Scheduled elite-kick restarts. |
| `--kick-fraction` | `kick_fraction` | `0.1` | exclusiveMinimum=0, exclusiveMaximum=1 | Fraction of members swapped by an elite kick. |
| `--kick-t0` | `kick_t0` | `0.35` | exclusiveMinimum=0 | SA start temperature for elite-kick restarts. |
| `--sa-t0` | `sa_t0` | `1.4` | exclusiveMinimum=0 | Fixed SA start temperature. |
| `--sa-t1` | `sa_t1` | `5e-05` | exclusiveMinimum=0 | Fixed SA end temperature. |
| `--sa-auto-temperature` | `sa_auto_temperature` | `false` | boolean | Calibrate each restart's SA temperatures from sampled uphill move deltas. |
| `--sa-temperature-samples` | `sa_temperature_samples` | `256` | minimum=1, maximum=65536 | Target positive-delta samples for restart-local SA calibration. |
| `--sa-temperature-quantile` | `sa_temperature_quantile` | `0.5` | exclusiveMinimum=0, maximum=1 | Positive-delta quantile used to calibrate SA endpoints. |
| `--sa-initial-uphill-acceptance` | `sa_initial_uphill_acceptance` | `0.6` | exclusiveMinimum=0, exclusiveMaximum=1 | Target initial acceptance for the calibrated uphill-delta quantile. |
| `--sa-final-uphill-acceptance` | `sa_final_uphill_acceptance` | `0.01` | exclusiveMinimum=0, exclusiveMaximum=1 | Target final acceptance for the calibrated uphill-delta quantile. |
| `--sa-candidate-trials` | `sa_candidate_trials` | `1` | minimum=1, maximum=64 | Candidate swaps evaluated per SA iteration; one preserves the historical proposal. |
| `--sa-multiple-try-random-probability` | `sa_multiple_try_random_probability` | `0.1` | minimum=0, maximum=1 | For multiple-try SA, probability of selecting a random valid trial instead of the best. |
| `--restart-threads` | `restart_threads` | `1` | minimum=0 | Parallel subset-restart workers; zero requests automatic allocation. |
| `--time-budget-per-p` | `time_budget_per_p` | `0.0` | minimum=0 | Wall-clock target per instance/p solve; zero disables anytime mode. |
| `--mode` | `mode` | `balanced` | enum=['balanced', 'smallp-region', 'highp-delete', 'hybrid'] | Solver mode: balanced, smallp-region, highp-delete, or hybrid. |

## Full-TSP search

| CLI option | JSON key | Default | Constraints | Description |
|---|---|---:|---|---|
| `--tsp-candidate-starts` | `tsp_candidate_starts` | `12` | minimum=1 | Cheap deterministic TSP starts screened before ILS promotion. |
| `--tsp-farthest-starts` | `tsp_farthest_starts` | `0` | minimum=0, maximum=1 | Opt into at most one cubic farthest-insertion pilot. |
| `--tsp-min-edge-jaccard` | `tsp_min_edge_jaccard` | `0.02` | minimum=0, maximum=1 | Preferred edge-Jaccard distance among promoted TSP starts. |
| `--tsp-restarts` | `tsp_restarts` | `5` | minimum=1 | Full TSP restarts promoted into ILS. |
| `--tsp-ils` | `tsp_ils` | `300` | minimum=0 | Full-TSP ILS perturbation iterations. |
| `--tsp-patience` | `tsp_patience` | `80` | minimum=0 | Full-TSP ILS stagnation patience. |

## Local-search neighborhoods

| CLI option | JSON key | Default | Constraints | Description |
|---|---|---:|---|---|
| `--final-exhaustive-k` | `final_exhaustive_k` | `300` | minimum=0 | Exhaustive final 2-opt cardinality threshold. |
| `--exhaustive-two-opt-policy` | `exhaustive_two_opt_policy` | `final-only` | enum=['never', 'final-only', 'all-polish'] | Exhaustive 2-opt policy: never, final-only, or all-polish. |
| `--subset-swap-passes` | `subset_swap_descent_passes` | `1` | minimum=0 | Deterministic one-for-one subset-swap passes. |
| `--pair-exchange-passes` | `pair_exchange_passes` | `1` | minimum=0 | Two-for-two subset-exchange passes. |
| `--pair-exchange-max-k` | `pair_exchange_max_k` | `5000` | minimum=0 | Skip pair exchange above k; zero removes the gate. |
| `--ruin-recreate-rounds` | `ruin_recreate_rounds` | `4` | minimum=0 | Ruin/recreate LNS rounds. |
| `--adaptive-ruin-recreate` | `adaptive_ruin_recreate` | `true` | boolean | Use the adaptive multi-scale ruin/recreate operator portfolio. |
| `--ruin-recreate-max-fraction` | `ruin_recreate_max_fraction` | `0.05` | minimum=0, maximum=1 | Maximum ruined fraction of k. |
| `--ruin-recreate-max-nodes` | `ruin_recreate_max_nodes` | `96` | minimum=0 | Absolute ruin-size cap; zero removes it. |
| `--ruin-recreate-pool-cap` | `ruin_recreate_pool_cap` | `640` | minimum=1 | Candidate repair-pool cap. |
| `--ejection-chain-starts` | `ejection_chain_starts` | `3` | minimum=0 | Variable-depth membership-chain starts per restart. |
| `--ejection-chain-depth` | `ejection_chain_depth` | `6` | minimum=0 | Maximum membership swaps per ejection chain. |
| `--ejection-chain-candidates` | `ejection_chain_candidates` | `24` | minimum=0 | Incoming-node candidates per chain step. |
| `--ejection-chain-remove-cap` | `ejection_chain_remove_cap` | `96` | minimum=0 | Removable selected-node cap per chain step; zero means all. |
| `--ejection-chain-max-uphill` | `ejection_chain_max_uphill` | `0.75` | minimum=0 | Maximum cumulative uphill excursion in mean-edge units. |
| `--elite-diversity-slots` | `elite_diversity_slots` | `4` | minimum=0 | Supplemental set-diverse elite slots. |
| `--elite-min-jaccard` | `elite_min_jaccard` | `0.02` | minimum=0, maximum=1 | Minimum selected-set Jaccard distance for supplemental elite entries. |
| `--elite-quality-slack` | `elite_quality_slack` | `0.03` | minimum=0 | Quality slack allowed for supplemental diverse entries. |
| `--path-relink-top` | `path_relink_top` | `3` | minimum=0 | Literal maximum archive population exposed to relinking. |
| `--path-relink-diverse-reserve` | `path_relink_diverse_reserve` | `1` | minimum=0 | Diversity-preferred entries inside the relinking node cap. |
| `--path-relink-max-pairs` | `path_relink_max_pairs` | `3` | minimum=0 | Maximum ranked relinking pair attempts; zero is unlimited. |
| `--path-relink-max-removed` | `path_relink_max_removed` | `64` | minimum=0 | Maximum one-way selected-set difference per pair; zero is unlimited. |
| `--path-relink-max-removed-sum` | `path_relink_max_removed_sum` | `128` | minimum=0 | Cumulative selected-set-difference budget; zero is unlimited. |
| `--path-relink-max-candidate-scans` | `path_relink_max_candidate_scans` | `250000` | minimum=0 | Exact relinking candidate-scan budget; zero is unlimited. |

## Geometry and KNN

| CLI option | JSON key | Default | Constraints | Description |
|---|---|---:|---|---|
| `--knn` | `knn_k` | `40` | minimum=0 | Exact KNN candidate count; values are capped to N-1. |
| `--knn-backend` | `knn_backend` | `grid` | enum=['coords_exact_grid_knn', 'coords_exact_bruteforce_knn'] | Exact KNN backend: grid or bruteforce. |
| `--reverse-knn` | `reverse_knn` | `true` | boolean | Build reverse-KNN adjacency lazily when candidate local search needs it; disable to save additional memory. |
| `--verify-knn` | `verify_knn_checks` | `0` | minimum=0 | Sampled exact-KNN verification checks. |
| `--grid-cell` | `grid_cell` | `0.0` | minimum=0 | Force a grid cell size; zero selects an automatic safe value. |

## Statistical diagnostics

| CLI option | JSON key | Default | Constraints | Description |
|---|---|---:|---|---|
| `--control-variate` | `control_variate` | `false` | boolean | Compute the two-nearest-neighbor control variate and fixed-subset bound; the bound measures tour quality conditional on that subset. |
| `--cv-mc-samples` | `cv_mc_samples` | `2000` | minimum=1 | Cheap KNN-only samples used to estimate the full-set control-variate expectation. |
| `--cv-max-point-ops` | `cv_max_point_ops` | `100000000` | minimum=2 | Maximum N times control-reference samples; bounds the Monte Carlo point-generation work exactly. |
| `--held-karp` | `held_karp` | `false` | boolean | Compute the Held-Karp lower bound for each selected subset; this certifies tour quality conditional on that subset, not global subset optimality. |
| `--hk-iterations` | `hk_iterations` | `400` | minimum=1 | Held-Karp subgradient iterations. |
| `--include-instance-rows` | `include_instance_rows` | `false` | boolean | Include per-instance rows and restart diagnostics in JSON. |

## External oracle

| CLI option | JSON key | Default | Constraints | Description |
|---|---|---:|---|---|
| `--oracle` | `oracle_mode` | `none` | enum=['none', 'auto', 'lkh', 'concorde'] | External oracle: none, auto, lkh, or concorde. |
| `--oracle-format` | `oracle_format` | `matrix` | enum=['matrix', 'euc2d'] | Oracle problem format: matrix or euc2d. |
| `--lkh-path` | `lkh_path` | `LKH` | minLength=1, maxLength=4096 | LKH executable path. |
| `--concorde-path` | `concorde_path` | `concorde` | minLength=1, maxLength=4096 | Concorde executable path. |
| `--oracle-time-limit` | `oracle_time_limit_sec` | `0` | minimum=0 | Per-call oracle time limit; zero disables it. |
| `--oracle-scale` | `oracle_scale` | `1000000` | minimum=1 | Distance scaling factor for integer oracle matrices. |
| `--oracle-tsp-top` | `oracle_tsp_top` | `1` | minimum=0 | Top full-TSP solutions sent to the oracle. |
| `--oracle-subset-top` | `oracle_subset_top` | `1` | minimum=0 | Top subset solutions sent to the oracle. |
| `--oracle-min-k` | `oracle_min_k` | `17` | minimum=3 | Minimum cardinality eligible for an external oracle. |
| `--oracle-max-k` | `oracle_max_k` | `2500` | minimum=3 | Maximum cardinality eligible for an external oracle. |
| `--oracle-lkh-runs` | `oracle_lkh_runs` | `6` | minimum=1 | LKH RUNS parameter. |
| `--oracle-lkh-trials` | `oracle_lkh_max_trials` | `0` | minimum=0 | LKH MAX_TRIALS; zero uses the solver default. |
| `--oracle-no-tsp` | `oracle_use_for_tsp` | `true` | boolean | Disable external-oracle use for full TSP. |
| `--oracle-no-subset` | `oracle_use_for_subset` | `true` | boolean | Disable external-oracle use for subset tours. |
| `--oracle-inline-feedback` | `oracle_inline_feedback` | `false` | boolean | Feed oracle-improved cycles back into the in-process elite search. |
| `--oracle-verbose` | `oracle_verbose` | `false` | boolean | Print external-oracle diagnostics. |

## Ablation controls

| CLI option | JSON key | Default | Constraints | Description |
|---|---|---:|---|---|
| `--disable-two-opt` | `disable_two_opt` | `false` | boolean | Disable 2-opt. |
| `--disable-or-opt` | `disable_or_opt` | `false` | boolean | Disable Or-opt. |
| `--disable-subset-swap` | `disable_subset_swap` | `false` | boolean | Disable one-for-one subset swaps. |
| `--disable-pair-exchange` | `disable_pair_exchange` | `false` | boolean | Disable two-for-two subset exchange. |
| `--disable-elite-restarts` | `disable_elite_restarts` | `false` | boolean | Disable elite-seeded anytime restarts. |
| `--disable-ruin-recreate` | `disable_ruin_recreate` | `false` | boolean | Disable ruin/recreate. |
| `--disable-ejection-chain` | `disable_ejection_chain` | `false` | boolean | Disable membership ejection chains. |
| `--disable-path-relink` | `disable_path_relink` | `false` | boolean | Disable path relinking. |
| `--disable-smallp-seeds` | `disable_smallp_seeds` | `false` | boolean | Disable specialized small-p seeds. |
| `--disable-highp-delete` | `disable_highp_delete` | `false` | boolean | Disable high-p deletion seeds and exchange. |
