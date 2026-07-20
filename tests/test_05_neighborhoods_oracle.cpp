#include "test_common.hpp"

namespace {

ALDOUS_TEST(test_oracle_large_coordinates_precision) {
    // Regression: LKH stores costs in `int` and multiplies them by its PRECISION
    // parameter (default 100), aborting when that overflows. With our cost scale
    // of 1e6, any pairwise distance above ~21.5 overflows -- which is EVERY real
    // campaign instance (N=1250 already has side 35). The symptom is silent: the
    // oracle fails on every call and the solver quietly keeps its built-in tour,
    // so an entire study can complete "successfully" with no oracle contribution.
    // The reference stand-in emulates LKH's guard, so this test fails without the
    // PRECISION=1 parameter and the scale cap.
    ExternalOracleConfig cfg;
    cfg.mode = ExternalOracleMode::Lkh;
    cfg.lkh_path = std::string(ALDOUS_TSP_TESTS_DIR) + "/reference_lkh.py";
    cfg.problem_format = OracleProblemFormat::Matrix;
    cfg.min_k = 4;
    cfg.max_k = 5000;
    cfg.scale = 1000000;  // the production default that triggered the overflow
    cfg.lkh_runs = 1;
    OracleContext oracle;
    std::string err;
    if (!build_oracle_context(cfg, oracle, err) || oracle.resolved != ResolvedOracleMode::Lkh) {
        std::fprintf(stderr, "  [skip test_oracle_large_coordinates_precision: reference oracle unavailable]\n");
        return;
    }
    // N=1000 -> side ~31.6, so the farthest pairs exceed the ~21.5 overflow
    // threshold, exactly like the campaign geometry.
    for (bool periodic : {true, false}) {
        Instance inst;
        inst.periodic = periodic;
        Rng rng(periodic ? 909u : 4242u);
        inst.generate(1000, rng);
        inst.build_knn(12, KnnBackend::GridExact);
        require(inst.side > 25.0, "test instance is large enough to overflow LKH's int costs");
        std::vector<int> subset(30);
        std::iota(subset.begin(), subset.end(), 0);
        Tour candidate;
        candidate.init(inst.N);
        candidate.set_tour(subset, inst);
        SearchStats stats;
        external_oracle_polish_tour(candidate, inst, oracle, /*full_tsp=*/false, &stats);
        std::string reason;
        for (const OracleCallRecord& rec : stats.oracle_call_records) {
            if (!rec.error.empty()) {
                reason = rec.error;
            }
        }
        if (stats.oracle_solved == 0) {
            std::fprintf(stderr, "  oracle failed on large coordinates: %s\n", reason.c_str());
        }
        require(stats.oracle_solved > 0, "oracle solves large-coordinate instances (no PRECISION overflow)");
        require(stats.oracle_failed == 0, "no oracle calls fail on large-coordinate instances");
        require(std::isfinite(candidate.length) && candidate.length > 0.0, "oracle tour length is finite");
    }
}

ALDOUS_TEST(test_held_karp_bound) {
    // The Held-Karp 1-tree bound must be a rigorous lower bound on the optimal
    // tour (never exceed it) and be tight (close to it), on both the torus and
    // the open square. Ground truth is the exact solver for k <= 16.
    double worst_ratio = 1.0;
    for (bool periodic : {true, false}) {
        for (int N : {8, 12, 15, 16}) {
            for (unsigned seed = 1; seed <= 4; ++seed) {
                Instance inst;
                inst.periodic = periodic;
                Rng rng(seed * 13u + static_cast<unsigned>(N));
                inst.generate(N, rng);
                inst.build_knn(std::min(N - 1, 10), KnnBackend::GridExact);
                std::vector<int> all(static_cast<std::size_t>(N));
                std::iota(all.begin(), all.end(), 0);
                std::vector<int> cyc;
                double opt = 0.0;
                require(exact_small_tsp_cycle(inst, all, cyc, opt), "exact optimum computed");
                const HeldKarpBound hk = held_karp_bound(inst, all, opt * 1.05, 400);
                require(hk.computed, "Held-Karp bound is computed");
                // Rigorous: never exceeds the true optimum (tiny fp slack).
                require(hk.bound <= opt * (1.0 + 1e-6) + 1e-9,
                        "Held-Karp bound does not exceed the optimum");
                // The subgradient bound is at least the plain 1-tree.
                require(hk.bound >= hk.one_tree - 1e-9,
                        "subgradient bound is at least the plain 1-tree");
                // Tight: within a few percent of optimal.
                require(hk.bound >= 0.9 * opt, "Held-Karp bound is tight (>= 90% of optimum)");
                worst_ratio = std::min(worst_ratio, hk.bound / opt);
            }
        }
    }
    // Across these instances the bound should typically be very tight.
    require(worst_ratio >= 0.9, "worst-case Held-Karp tightness stays high");
}

ALDOUS_TEST(test_control_variate_bounds) {
    // The two-NN subset bound must lower-bound the found tour on every instance;
    // at p=1 the selected subset is the full set so the subset bound equals the
    // full-set bound; and on the torus E[B_full]/N matches the Poisson value 0.625.
    RunOptions opt;
    opt.N = 400;
    opt.instances = 4;
    opt.threads = 1;
    opt.p_values = {0.1, 0.5, 1.0};
    opt.periodic = true;
    opt.control_variate = true;
    opt.cv_mc_samples = 800;
    opt.include_instance_rows = true;
    opt.solver.seed = 4242;
    opt.solver.subset_restarts = 2;
    opt.solver.restart_threads = 1;
    // This test exercises bound construction and aggregation, not heuristic
    // quality. Keep the generated tours valid while removing production-scale
    // search work that made a bookkeeping test dominate the core suite.
    opt.solver.tsp_restarts = 1;
    opt.solver.tsp_ils = 0;
    opt.solver.sa_iters = 0;
    opt.solver.final_exhaustive_k = 0;
    opt.solver.disable_two_opt = true;
    opt.solver.disable_or_opt = true;
    opt.solver.disable_subset_swap = true;
    opt.solver.disable_pair_exchange = true;
    opt.solver.disable_ruin_recreate = true;
    opt.solver.disable_path_relink = true;

    const ResultsDocument doc = ExperimentRunner(opt).run();
    const std::string json = results_to_json(doc);
    require(json.find("\"conditional_two_nn_bound_mean\"") != std::string::npos,
            "JSON exposes the unambiguous conditional two-NN summary field");
    require(json.find("\"subset_bound_mean\"") != std::string::npos,
            "JSON retains the schema-13 two-NN alias");
    require(doc.full_bound_expectation > 0.0, "control variate estimates E[B_full]");
    const double per_point = doc.full_bound_expectation / static_cast<double>(opt.N);
    require(std::fabs(per_point - 0.625) < 0.02,
            "E[B_full]/N matches the Poisson-torus prediction 0.625");

    for (const auto& row : doc.instance_rows) {
        require(row.full_bound >= 0.0, "instance carries a full-set bound");
        for (const auto& pv : row.p_results) {
            require(pv.subset_bound >= 0.0, "instance p-row carries a subset bound");
            require(pv.subset_bound == pv.conditional_two_nn_bound,
                    "legacy and canonical per-instance two-NN names share one value");
            // Subset two-NN bound is a valid lower bound on the found tour.
            require(pv.subset_bound <= pv.value + 1e-6,
                    "subset two-NN bound lower-bounds the found tour length");
            if (pv.k >= opt.N) {
                // p = 1: subset is the full set, so the per-point bounds match.
                require(std::fabs(pv.subset_bound - row.full_bound / static_cast<double>(opt.N)) < 1e-6,
                        "at p=1 the subset bound equals the full-set bound");
            }
        }
    }
    for (const auto& item : doc.summary) {
        const PValueSummary& s = item.second;
        require(s.has_control_variate, "summary marks control variate active");
        require(s.conditional_two_nn_bound_mean <= s.mean + 1e-6,
                "mean conditional subset bound is below the found-tour mean");
        require(s.subset_bound_mean == s.conditional_two_nn_bound_mean,
                "legacy and canonical two-NN summary names share one value");
        require(s.cv_variance_reduction >= 0.0 && s.cv_variance_reduction <= 1.0,
                "variance reduction fraction is in [0,1]");
    }
}

#if defined(ALDOUS_TSP_TESTS_DIR)
// Independent exact TSP (Held-Karp) using the instance's own metric, which is
// torus-aware when inst.periodic is set. Ground truth for the oracle round-trip.
double exact_held_karp(const Instance& inst) {
    const int n = inst.N;
    const int size = 1 << n;
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> dp(static_cast<std::size_t>(size),
                                        std::vector<double>(static_cast<std::size_t>(n), INF));
    dp[1][0] = 0.0;
    for (int mask = 1; mask < size; ++mask) {
        if (!(mask & 1)) { continue; }
        for (int j = 0; j < n; ++j) {
            const double base = dp[static_cast<std::size_t>(mask)][static_cast<std::size_t>(j)];
            if (base == INF) { continue; }
            for (int nx = 0; nx < n; ++nx) {
                if (mask & (1 << nx)) { continue; }
                const int nm = mask | (1 << nx);
                const double c = base + inst.dist(j, nx);
                if (c < dp[static_cast<std::size_t>(nm)][static_cast<std::size_t>(nx)]) {
                    dp[static_cast<std::size_t>(nm)][static_cast<std::size_t>(nx)] = c;
                }
            }
        }
    }
    double best = INF;
    for (int j = 1; j < n; ++j) {
        best = std::min(best, dp[static_cast<std::size_t>(size - 1)][static_cast<std::size_t>(j)] + inst.dist(j, 0));
    }
    return best;
}

ALDOUS_TEST(test_oracle_torus_roundtrip) {
    // The external-oracle path must feed the torus distances to the solver and
    // score the returned tour with the torus metric. With the reference LKH
    // stand-in (which solves the matrix it is handed exactly for this size), the
    // oracle result must equal the independent torus optimum. k must exceed
    // kExactSmallTourLimit for the oracle to engage.
    ExternalOracleConfig cfg;
    cfg.mode = ExternalOracleMode::Lkh;
    cfg.lkh_path = std::string(ALDOUS_TSP_TESTS_DIR) + "/reference_lkh.py";
    cfg.problem_format = OracleProblemFormat::Matrix;
    cfg.min_k = 4;
    cfg.max_k = 5000;
    cfg.tsp_top = 1;
    cfg.lkh_runs = 1;
    OracleContext oracle;
    std::string err;
    if (!build_oracle_context(cfg, oracle, err) || oracle.resolved != ResolvedOracleMode::Lkh) {
        std::fprintf(stderr, "  [skip test_oracle_torus_roundtrip: reference oracle unavailable]\n");
        return;
    }
    const int N = 17;  // smallest k above kExactSmallTourLimit
    for (bool periodic : {true, false}) {
        Instance inst;
        inst.periodic = periodic;
        Rng rng(periodic ? 71u : 131u);
        inst.generate(N, rng);
        inst.build_knn(12, KnnBackend::GridExact);
        const double optimum = exact_held_karp(inst);
        std::vector<int> all(static_cast<std::size_t>(N));
        std::iota(all.begin(), all.end(), 0);
        Tour t;
        t.init(N);
        t.set_tour(all, inst);
        t.ensure_edges(inst);
        const bool improved = external_oracle_polish_tour(t, inst, oracle, true, nullptr, false);
        require(improved, "oracle engages and improves the tour for k > exact limit");
        require(std::fabs(t.length - optimum) <= 1e-6 * std::max(optimum, 1.0),
                "oracle tour matches the independent (torus-aware) exact optimum");
    }
}
#endif

ALDOUS_TEST(test_batched_pair_repair_matches_legacy) {
    auto check_case = [](const Instance& inst,
                         const std::vector<int>& cycle,
                         const std::vector<int>& pool) {
        const PairRepairResult fast = best_two_node_regret_repair(inst, cycle, pool);
        std::vector<unsigned char> banned(static_cast<std::size_t>(inst.N), 0U);
        for (int node : cycle) {
            banned[static_cast<std::size_t>(node)] = 1U;
        }
        double brute_length = std::numeric_limits<double>::infinity();
        std::vector<int> brute_nodes;
        for (int ui = 0; ui < static_cast<int>(pool.size()); ++ui) {
            for (int vi = ui + 1; vi < static_cast<int>(pool.size()); ++vi) {
                std::vector<int> candidate = cycle;
                const std::vector<int> add_pool = {
                    pool[static_cast<std::size_t>(ui)],
                    pool[static_cast<std::size_t>(vi)],
                };
                require(regret_repair_cycle(candidate, inst,
                                            static_cast<int>(cycle.size()) + 2,
                                            add_pool, banned),
                        "legacy two-node regret repair succeeds on a valid pool");
                const double length = cycle_length(inst, candidate);
                if (length < brute_length) {
                    brute_length = length;
                    brute_nodes = std::move(candidate);
                }
            }
        }
        require(fast.valid == !brute_nodes.empty(),
                "batched pair repair validity matches legacy enumeration");
        require(std::fabs(fast.length - brute_length)
                    <= 1e-11 * (1.0 + std::fabs(brute_length)),
                "batched pair repair length matches legacy enumeration");
        require(fast.nodes == brute_nodes,
                "batched pair repair preserves legacy regret ordering and tie breaks");
    };

    for (bool periodic : {false, true}) {
        Instance inst;
        inst.periodic = periodic;
        Rng gen(periodic ? 81231U : 81230U);
        inst.generate(48, gen);
        for (int rep = 0; rep < 80; ++rep) {
            std::vector<int> shuffled = all_nodes(inst.N);
            gen.partial_shuffle(shuffled.begin(), shuffled.end(), 18U);
            const int cycle_size = 8 + (rep % 5);
            std::vector<int> cycle(shuffled.begin(), shuffled.begin() + cycle_size);
            std::vector<int> pool(shuffled.begin() + cycle_size,
                                  shuffled.begin() + cycle_size + 6);
            check_case(inst, cycle, pool);
        }
    }

    Instance tied;
    tied.set_points({{0.0, 0.0}, {4.0, 0.0}, {4.0, 4.0}, {0.0, 4.0},
                     {2.0, 0.0}, {2.0, 0.0}, {2.0, 4.0}, {2.0, 4.0}});
    check_case(tied, {0, 1, 2, 3}, {4, 5, 6, 7});
}

ALDOUS_TEST(test_cached_regret_repair_and_adaptive_lns) {
    for (const bool periodic : {false, true}) {
        Instance inst;
        inst.periodic = periodic;
        Rng rng(periodic ? 39117U : 39116U);
        inst.generate(72, rng);
        inst.build_knn(24, KnnBackend::GridExact);
        for (int rep = 0; rep < 80; ++rep) {
            std::vector<int> shuffled = all_nodes(inst.N);
            rng.partial_shuffle(shuffled.begin(), shuffled.end(), 34U);
            const int start_size = 12 + (rep % 7);
            const int add_count = 3 + (rep % 8);
            std::vector<int> legacy(shuffled.begin(), shuffled.begin() + start_size);
            std::vector<int> cached = legacy;
            std::vector<int> pool(shuffled.begin() + start_size,
                                  shuffled.begin() + start_size + 14);
            std::vector<unsigned char> banned(static_cast<std::size_t>(inst.N), 0U);
            for (const int node : legacy) {
                banned[static_cast<std::size_t>(node)] = 1U;
            }
            require(regret_repair_cycle(legacy, inst, start_size + add_count,
                                        pool, banned),
                    "legacy regret repair succeeds");
            require(cached_regret_repair_cycle(cached, inst,
                                               start_size + add_count,
                                               pool, banned),
                    "cached regret repair succeeds");
            require(cached == legacy,
                    "cached regret repair preserves exact candidate and edge ties");
        }

        Tour tour;
        tour.init(inst.N);
        tour.set_tour(random_subset(inst.N, 36, rng), inst);
        const double before = tour.length;
        SolverOptions options;
        options.ruin_recreate_rounds = 5;
        options.ruin_recreate_max_fraction = 0.20;
        options.ruin_recreate_max_nodes = 12;
        options.ruin_recreate_pool_cap = 96;
        options.final_exhaustive_k = 0;
        SearchStats stats;
        (void)subset_ruin_recreate_lns(tour, inst, rng, options, &stats,
                                       options.ruin_recreate_rounds);
        require(tour.check_invariants() && tour.k == 36,
                "adaptive LNS preserves tour membership invariants");
        require(tour.length <= before + kImprovementEps,
                "adaptive LNS accepts only improving reconstructions");
        require(stats.ruin_recreate_attempts == 10,
                "adaptive LNS records legacy-floor and portfolio rounds");
        require(stats.ruin_recreate_worst_attempts == 3
                    && stats.ruin_recreate_segment_attempts == 4
                    && stats.ruin_recreate_spatial_attempts == 1
                    && stats.ruin_recreate_long_edge_attempts == 1
                    && stats.ruin_recreate_random_attempts == 1,
                "adaptive LNS preserves the legacy floor and cycles through every new operator");
        require(stats.ruin_recreate_removed_nodes >= 15,
                "adaptive LNS records multi-scale ruined cardinalities");
    }
}

ALDOUS_TEST(test_membership_ejection_chain) {
    for (const bool periodic : {false, true}) {
        Rng instance_rng(periodic ? 913712U : 913711U);
        Instance inst;
        inst.periodic = periodic;
        inst.generate(96, instance_rng);
        inst.build_knn(28, KnnBackend::GridExact);

        Tour initial;
        initial.init(inst.N);
        initial.set_tour(random_subset(inst.N, 40, instance_rng), inst);
        SolverOptions options;
        options.ejection_chain_starts = 4;
        options.ejection_chain_depth = 5;
        options.ejection_chain_candidates = 18;
        options.ejection_chain_remove_cap = 36;
        options.ejection_chain_max_uphill = 1.25;
        options.disable_subset_swap = true;
        options.final_exhaustive_k = 0;

        Tour first = initial;
        Tour second = initial;
        Rng first_rng(550019U);
        Rng second_rng(550019U);
        SearchStats first_stats;
        SearchStats second_stats;
        (void)subset_ejection_chain_search(first, inst, first_rng, options,
                                           &first_stats);
        (void)subset_ejection_chain_search(second, inst, second_rng, options,
                                           &second_stats);
        require(first.nodes == second.nodes
                    && std::fabs(first.length - second.length) < 1e-12,
                "membership ejection chains are deterministic for a fixed stream");
        require(first.check_invariants() && first.k == initial.k,
                "membership ejection chains preserve cardinality and uniqueness");
        require(first.length <= initial.length + kImprovementEps,
                "membership ejection chains replace the incumbent only on improvement");
        require(first_stats.ejection_chain_attempts > 0
                    && first_stats.ejection_chain_feasible
                           <= first_stats.ejection_chain_attempts
                    && first_stats.ejection_chain_steps
                           >= first_stats.ejection_chain_feasible
                    && first_stats.ejection_chain_scans > 0,
                "membership ejection chains report bounded search work");
        require(first_stats.ejection_chain_attempts
                    == second_stats.ejection_chain_attempts
                    && first_stats.ejection_chain_feasible
                        == second_stats.ejection_chain_feasible
                    && first_stats.ejection_chain_steps
                        == second_stats.ejection_chain_steps
                    && first_stats.ejection_chain_scans
                        == second_stats.ejection_chain_scans,
                "membership ejection-chain telemetry is deterministic");
        require(first_stats.ejection_chain_improvements == 0
                    || first_stats.ejection_chain_accepted_depth > 0,
                "accepted ejection-chain improvements record a positive prefix depth");
    }
}

ALDOUS_TEST(test_pair_exchange_large_k_gate) {
    Instance inst;
    Rng rng(77123);
    inst.generate(20, rng);
    inst.build_knn(12, KnnBackend::GridExact);
    Tour tour;
    tour.init(inst.N);
    tour.set_tour(random_subset(inst.N, 8, rng), inst);
    SolverOptions options;
    options.pair_exchange_max_k = 7;
    SearchStats stats;
    require(!subset_pair_exchange_descent(tour, inst, rng, options, &stats, 1),
            "large-k pair-exchange gate skips the neighborhood");
    require(stats.pair_exchange_skipped_large_k == 1,
            "large-k pair-exchange gate is explicitly counted");
    require(stats.pair_exchange_scans == 0,
            "large-k pair-exchange gate performs no candidate scans");
}

ALDOUS_TEST(test_batched_swap_matches_scalar) {
    auto check_case = [](const Instance& inst,
                         const Tour& tour,
                         const std::vector<SwapCandidatePair>& candidates) {
        BatchedSwapResult scalar;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const SwapCandidatePair& candidate = candidates[index];
            const SwapMoveEval eval = evaluate_swap_after_remove(inst,
                                                                 tour,
                                                                 candidate.remove_pos,
                                                                 candidate.add_node);
            if (eval.valid && (!scalar.valid || eval.delta < scalar.delta)) {
                scalar.valid = true;
                scalar.delta = eval.delta;
                scalar.remove_pos = candidate.remove_pos;
                scalar.add_node = candidate.add_node;
                scalar.post_remove_pred = eval.post_remove_pred;
                scalar.candidate_index = index;
            }
        }

        const BatchedSwapResult batched = best_batched_swap(inst, tour, candidates);
        require(batched.valid == scalar.valid,
                "batched exact swap validity matches ordered scalar enumeration");
        if (!scalar.valid) { return; }
        require(std::fabs(batched.delta - scalar.delta)
                    <= 1e-12 * (1.0 + std::fabs(scalar.delta)),
                "batched exact swap delta matches scalar enumeration");
        require(batched.remove_pos == scalar.remove_pos,
                "batched exact swap preserves removal tie order");
        require(batched.add_node == scalar.add_node,
                "batched exact swap preserves addition tie order");
        require(batched.post_remove_pred == scalar.post_remove_pred,
                "batched exact swap preserves insertion-edge tie order");
        require(batched.candidate_index == scalar.candidate_index,
                "batched exact swap preserves global candidate order");

        Tour applied = tour;
        applied.apply_swap_post_rem(batched.remove_pos,
                                    batched.post_remove_pred,
                                    batched.add_node,
                                    inst,
                                    batched.delta);
        const double incremental_length = applied.length;
        applied.recompute_length(inst);
        require(std::fabs(incremental_length - applied.length)
                    <= 1e-10 * (1.0 + applied.length),
                "batched exact swap delta agrees with full tour recomputation");
        require(applied.check_invariants(),
                "batched exact swap preserves tour membership invariants");
    };

    for (bool periodic : {false, true}) {
        Rng rng(periodic ? 819991U : 819990U);
        Instance inst;
        inst.periodic = periodic;
        inst.generate(72, rng);
        for (int trial = 0; trial < 100; ++trial) {
            const int k = 6 + rng.randint(18);
            Tour tour;
            tour.init(inst.N);
            tour.set_tour(random_subset(inst.N, k, rng), inst);
            tour.ensure_edges(inst);

            std::vector<int> outside;
            outside.reserve(static_cast<std::size_t>(inst.N - k));
            for (int node = 0; node < inst.N; ++node) {
                if (tour.in_set[static_cast<std::size_t>(node)] == 0U) {
                    outside.push_back(node);
                }
            }

            std::vector<SwapCandidatePair> candidates;
            candidates.reserve(220);
            for (int sample = 0; sample < 200; ++sample) {
                const int remove_pos = rng.randint(k);
                int add_node = outside[static_cast<std::size_t>(rng.randint(static_cast<int>(outside.size())))];
                const int mode = rng.randint(12);
                if (mode == 0) {
                    add_node = tour.nodes[static_cast<std::size_t>(remove_pos)];
                } else if (mode == 1) {
                    add_node = tour.nodes[static_cast<std::size_t>((remove_pos + 1) % k)];
                }
                candidates.push_back({remove_pos, add_node});
                if (mode == 2) {
                    candidates.push_back({remove_pos, add_node});
                }
            }
            candidates.push_back({-1, outside.front()});
            candidates.push_back({k, outside.front()});
            candidates.push_back({0, -1});
            candidates.push_back({0, inst.N});
            check_case(inst, tour, candidates);
        }
    }

    Instance tied;
    tied.set_points({{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                     {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}});
    Tour tied_tour;
    tied_tour.init(tied.N);
    tied_tour.set_tour({0, 1, 2, 3}, tied);
    check_case(tied, tied_tour, {{2, 6}, {1, 5}, {3, 7}, {0, 4}});
}

ALDOUS_TEST(test_path_relink_step_matches_bruteforce) {
    Rng rng(9091);
    Instance inst;
    inst.generate(60, rng);
    inst.build_knn(16, KnnBackend::GridExact);
    for (int trial = 0; trial < 24; ++trial) {
        const int k = 8 + rng.randint(10);
        std::vector<int> nodes = random_subset(inst.N, k, rng);
        Tour tour;
        tour.init(inst.N);
        tour.set_tour(nodes, inst);
        tour.ensure_edges(inst);

        std::vector<int> remove_positions;
        for (int i = 0; i < tour.k; ++i) {
            if (rng.randint(2) == 0) { remove_positions.push_back(i); }
        }
        if (remove_positions.empty()) { remove_positions.push_back(rng.randint(tour.k)); }
        std::vector<int> add_nodes;
        for (int v = 0; v < inst.N && static_cast<int>(add_nodes.size()) < 9; ++v) {
            if (tour.in_set[static_cast<std::size_t>(v)] == 0U && rng.randint(3) == 0) {
                add_nodes.push_back(v);
            }
        }
        if (add_nodes.empty()) { continue; }

        double brute_best = std::numeric_limits<double>::infinity();
        int brute_remove = -1;
        int brute_add = -1;
        // Path relinking has historically used add-major/remove-minor order;
        // strict comparison makes that ordering the deterministic tie break.
        for (int add : add_nodes) {
            for (int ri : remove_positions) {
                const SwapMoveEval eval = evaluate_swap_after_remove(inst, tour, ri, add);
                if (eval.valid && eval.delta < brute_best) {
                    brute_best = eval.delta;
                    brute_remove = ri;
                    brute_add = add;
                }
            }
        }
        const PathRelinkStep step = path_relink_best_step(inst, tour, remove_positions, add_nodes);
        require(step.valid == std::isfinite(brute_best), "relink step validity matches brute force");
        if (step.valid) {
            require(std::abs(step.delta - brute_best) <= 1e-9 * (1.0 + std::abs(brute_best)),
                    "decomposed relink step delta matches brute-force best pair");
            require(step.remove_pos == brute_remove && step.add_node == brute_add,
                    "decomposed relink step preserves add-major/remove-minor tie order");
            const SwapInsertionMove applied = find_best_insert_after_remove(inst, tour, step.remove_pos, step.add_node);
            require(applied.valid, "chosen relink step is applicable");
            require(std::abs(applied.delta - step.delta) <= 1e-9 * (1.0 + std::abs(step.delta)),
                    "applied relink move delta matches selected step delta");
        }
    }
}

ALDOUS_TEST(test_path_relink_distance_cap) {
    Rng rng(4242);
    Instance inst;
    inst.generate(320, rng);
    inst.build_knn(16, KnnBackend::GridExact);
    SolverOptions opt;
    opt.seed = 4242;

    std::vector<int> far_a;
    std::vector<int> far_b;
    for (int i = 0; i < 100; ++i) { far_a.push_back(i); }
    for (int i = 200; i < 300; ++i) { far_b.push_back(i); }
    std::vector<int> out_nodes;
    double out_len = std::numeric_limits<double>::infinity();
    SearchStats stats;
    Rng relink_rng(1);
    require(static_cast<int>(far_a.size()) > kPathRelinkMaxDiff, "test pair exceeds relink cap");
    require(!subset_path_relink_bidirectional(inst, far_a, far_b, relink_rng, opt, out_nodes, out_len, &stats),
            "relink skips elite pairs beyond the symmetric-difference cap");
    require(stats.path_relink_attempts == 1 && stats.path_relink_feasible == 0,
            "skipped relink counts an attempt but not a feasible relink");

    std::vector<int> near_a = far_a;
    std::vector<int> near_b = far_a;
    near_b[3] = 310;
    near_b[40] = 311;
    near_b[77] = 312;
    require(subset_path_relink_bidirectional(inst, near_a, near_b, relink_rng, opt, out_nodes, out_len, &stats),
            "relink runs for close elite pairs");
    require(stats.path_relink_feasible == 1, "close-pair relink is counted feasible");
    require(static_cast<int>(out_nodes.size()) == static_cast<int>(near_a.size()), "relink preserves subset size");
    std::vector<unsigned char> seen(static_cast<std::size_t>(inst.N), 0U);
    for (int v : out_nodes) {
        require(v >= 0 && v < inst.N && seen[static_cast<std::size_t>(v)] == 0U, "relink output is a valid node set");
        seen[static_cast<std::size_t>(v)] = 1U;
    }
    require(std::abs(cycle_length(inst, out_nodes) - out_len) <= 1e-6 * (1.0 + out_len),
            "relink reported length matches its node cycle");
}

ALDOUS_TEST(test_path_relink_counter_semantics) {
    Rng rng(31337);
    Instance inst;
    inst.generate(70, rng);
    inst.build_knn(24, KnnBackend::GridExact);
    SolverOptions opt;
    opt.seed = 31337;
    opt.subset_restarts = 4;
    opt.sa_iters = 20;
    opt.final_exhaustive_k = 80;
    opt.path_relink_top = 3;
    opt.disable_pair_exchange = true;
    opt.disable_ruin_recreate = true;
    opt.disable_smallp_seeds = true;
    opt.disable_highp_delete = true;
    Rng solve_rng(31338);
    SolveResult result = solve_subset(inst, 24, solve_rng, opt);
    require(result.stats.path_relink_attempts > 0, "path relinking attempts are counted");
    require(result.stats.path_relink_feasible <= result.stats.path_relink_attempts, "feasible relinks cannot exceed attempts");
    require(result.stats.path_relink_best_improvements <= result.stats.path_relink_feasible, "best relink improvements cannot exceed feasible relinks");
    require(result.stats.path_relink_improvements == result.stats.path_relink_best_improvements, "legacy relink improvement counter mirrors best improvements");
}


ALDOUS_TEST(test_oracle_parser_and_fake_lkh) {
    std::vector<int> perm;
    require(parse_external_tour_text("NAME : x\nTOUR_SECTION\n1\n3\n2\n-1\nEOF\n", 3, perm), "oracle parser accepts TSPLIB 1-based tour");
    require((perm == std::vector<int>{0, 2, 1}), "oracle parser maps 1-based tour");
    require(!parse_external_tour_text("0 0 1", 3, perm), "oracle parser rejects duplicate nodes");

    Instance inst;
    std::vector<Point> pts;
    constexpr int n = 20;
    pts.reserve(n);
    for (int i = 0; i < n; ++i) {
        const double angle = 2.0 * kPi * static_cast<double>(i) / static_cast<double>(n);
        pts.push_back({std::cos(angle), std::sin(angle)});
    }
    inst.set_points(pts);
    inst.build_knn(n - 1, KnnBackend::GridExact);

    std::vector<int> zigzag;
    for (int i = 0; i < n / 2; ++i) {
        zigzag.push_back(i);
        zigzag.push_back(i + n / 2);
    }
    Tour tour;
    tour.init(n);
    tour.set_tour(zigzag, inst);
    const double before = tour.length;

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "aldous_tsp_fake_lkh_test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    require(!ec, "create fake oracle directory");
    const std::filesystem::path script = dir / "fake_lkh";
    {
        std::ofstream out(script);
        out << "#!/bin/sh\n";
        out << "if [ \"$1\" = \"--version\" ]; then echo fake-lkh-1.0; exit 0; fi\n";
        out << "cat > out.tour <<'TOUR'\n";
        out << "NAME : out\nTYPE : TOUR\nDIMENSION : 20\nTOUR_SECTION\n";
        // The candidate order is 0,10,1,11,...,9,19. This permutation returns 0,1,2,...,19.
        for (int i = 0; i < n / 2; ++i) {
            out << (2 * i + 1) << "\n";
        }
        for (int i = 0; i < n / 2; ++i) {
            out << (2 * i + 2) << "\n";
        }
        out << "-1\nEOF\nTOUR\nexit 0\n";
    }
    std::filesystem::permissions(script,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace,
                                 ec);
    require(!ec, "mark fake LKH executable");

    ExternalOracleConfig cfg;
    cfg.mode = ExternalOracleMode::Lkh;
    cfg.lkh_path = script.string();
    cfg.min_k = 17;
    cfg.max_k = 50;
    cfg.time_limit_sec = 5;
    OracleContext ctx;
    std::string error;
    require(build_oracle_context(cfg, ctx, error), "build fake oracle context");
    require(ctx.resolved == ResolvedOracleMode::Lkh, "fake oracle resolved as LKH");
    require(ctx.version == "fake-lkh-1.0", "fake oracle version captured");
    SearchStats stats;
    const bool improved = external_oracle_polish_tour(tour, inst, ctx, true, &stats);
    require(improved, "fake LKH improves zigzag tour");
    require(tour.length < before - 1e-6, "oracle output shortens tour");
    require(stats.oracle_calls == 1 && stats.oracle_solved == 1 && stats.oracle_improved == 1, "oracle stats updated");
    require(stats.oracle_call_records.size() == 1, "oracle call record captured");
    require(stats.oracle_call_records.front().type == "tsp" && stats.oracle_call_records.front().status == "improved",
            "oracle call record has type and status");
    require(stats.oracle_call_records.front().gain > 0.0, "oracle call record captures gain");
    std::filesystem::remove_all(dir, ec);
}


ALDOUS_TEST(test_grid_cell_safety_for_tiny_forced_cell) {
    std::vector<Point> pts;
    for (int i = 0; i < 48; ++i) {
        const double t = static_cast<double>(i);
        pts.push_back({-1000000.0 + 42000.0 * t, 800000.0 - 31000.0 * (t + (i % 5))});
    }
    Instance inst;
    inst.set_points(pts);
    inst.build_knn(7, KnnBackend::GridExact, 1e-9);
    require(static_cast<std::int64_t>(inst.gx) * static_cast<std::int64_t>(inst.gy) <= kMaxGridCells,
            "forced tiny grid cell is capped safely");
    Rng verify(4444);
    require(inst.verify_knn(48, verify), "safe capped grid remains exact");
}

ALDOUS_TEST(test_solver_ablation_flags_are_exact) {
    Rng rng(123456);
    Instance inst;
    inst.generate(64, rng);
    inst.build_knn(24, KnnBackend::GridExact);

    SolverOptions all_off;
    all_off.seed = 123456;
    all_off.subset_restarts = 2;
    all_off.sa_iters = 0;
    all_off.final_exhaustive_k = 0;
    all_off.disable_two_opt = true;
    all_off.disable_or_opt = true;
    all_off.disable_subset_swap = true;
    all_off.disable_pair_exchange = true;
    all_off.disable_ruin_recreate = true;
    all_off.disable_path_relink = true;
    all_off.disable_smallp_seeds = true;
    all_off.disable_highp_delete = true;
    Rng solve_rng1(1111);
    SolveResult all_off_result = solve_subset(inst, 32, solve_rng1, all_off);
    require(all_off_result.stats.two_opt_scans == 0 && all_off_result.stats.two_opt_improvements == 0,
            "disable-two-opt suppresses all two-opt work with all ablations");
    require(all_off_result.stats.or_opt_scans == 0 && all_off_result.stats.or_opt_improvements == 0,
            "disable-or-opt suppresses all or-opt work");
    require(all_off_result.stats.subset_swap_scans == 0 && all_off_result.stats.subset_swap_improvements == 0,
            "disable-subset-swap suppresses subset-swap work");
    require(all_off_result.stats.pair_exchange_scans == 0 && all_off_result.stats.pair_exchange_improvements == 0,
            "disable-pair-exchange suppresses pair exchange");
    require(all_off_result.stats.ruin_recreate_attempts == 0 && all_off_result.stats.ruin_recreate_improvements == 0,
            "disable-ruin-recreate suppresses LNS");
    require(all_off_result.stats.path_relink_attempts == 0 && all_off_result.stats.path_relink_improvements == 0,
            "disable-path-relink suppresses path relinking");

    SolverOptions no_two_opt;
    no_two_opt.seed = 2222;
    no_two_opt.subset_restarts = 3;
    no_two_opt.sa_iters = 1200;
    no_two_opt.final_exhaustive_k = 0;
    no_two_opt.subset_swap_descent_passes = 2;
    no_two_opt.disable_two_opt = true;
    no_two_opt.disable_pair_exchange = true;
    no_two_opt.disable_ruin_recreate = true;
    no_two_opt.disable_path_relink = true;
    no_two_opt.disable_smallp_seeds = true;
    no_two_opt.disable_highp_delete = true;
    Rng solve_rng2(2222);
    SolveResult no_two_result = solve_subset(inst, 32, solve_rng2, no_two_opt);
    require(no_two_result.stats.two_opt_scans == 0 && no_two_result.stats.two_opt_improvements == 0,
            "disable-two-opt is honored by SA cleanup and subset-swap cleanup");

    SolverOptions highp_accounting;
    highp_accounting.seed = 3333;
    highp_accounting.mode = SolverMode::HighPDelete;
    // --restarts is authoritative now, so ask for enough restarts to reach a
    // high-p seed. The pool holds warm seeds and high-p seeds; truncation
    // round-robins across seed KINDS, so 2 restarts guarantees one of each.
    highp_accounting.subset_restarts = 2;
    highp_accounting.sa_iters = 0;
    highp_accounting.final_exhaustive_k = 0;
    highp_accounting.disable_two_opt = true;
    highp_accounting.disable_or_opt = true;
    highp_accounting.disable_subset_swap = true;
    highp_accounting.disable_pair_exchange = true;
    highp_accounting.disable_ruin_recreate = true;
    highp_accounting.disable_path_relink = true;
    highp_accounting.disable_smallp_seeds = true;
    std::vector<int> warm_full(64);
    std::iota(warm_full.begin(), warm_full.end(), 0);
    Rng solve_rng3(3333);
    SolveResult highp_result = solve_subset(inst, 48, solve_rng3, highp_accounting, &warm_full);
    require(highp_result.stats.subset_swap_scans == 0 && highp_result.stats.subset_swap_improvements == 0,
            "disable-subset-swap suppresses only generic subset-swap counters");
    require(highp_result.stats.highp_exchange_scans > 0,
            "high-p reference exchange has separate scan counters");
}

std::filesystem::path write_fake_lkh_copy_script(const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
    const std::filesystem::path script = dir / "fake_lkh_copy";
    {
        std::ofstream out(script);
        out << "#!/bin/sh\n";
        out << "if [ \"$1\" = \"--version\" ]; then echo fake-lkh-copy-1.0; exit 0; fi\n";
        out << "cp init.tour out.tour\n";
        out << "exit 0\n";
    }
    std::error_code ec;
    std::filesystem::permissions(script,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write |
                                     std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace,
                                 ec);
    require(!ec, "mark fake copy LKH executable");
    return script;
}

ALDOUS_TEST(test_oracle_top_n_matches_cli_config) {
    Rng rng(9090);
    Instance inst;
    inst.generate(32, rng);
    inst.build_knn(24, KnnBackend::GridExact);

    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "aldous_tsp_fake_lkh_topn_test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    const std::filesystem::path script = write_fake_lkh_copy_script(dir);

    ExternalOracleConfig cfg;
    cfg.mode = ExternalOracleMode::Lkh;
    cfg.lkh_path = script.string();
    cfg.min_k = 3;
    cfg.max_k = 64;
    cfg.tsp_top = 3;
    cfg.subset_top = 2;
    cfg.time_limit_sec = 5;
    OracleContext ctx;
    std::string error;
    require(build_oracle_context(cfg, ctx, error), "build top-n oracle context");

    SolverOptions tsp_options;
    tsp_options.oracle = ctx;
    tsp_options.tsp_restarts = 8;
    tsp_options.tsp_ils = 0;
    tsp_options.final_exhaustive_k = 0;
    tsp_options.disable_two_opt = true;
    tsp_options.disable_or_opt = true;
    Rng tsp_rng(9091);
    SolveResult tsp_result = solve_tsp(inst, tsp_rng, tsp_options);
    require(tsp_result.stats.oracle_tsp_calls == 3,
            "oracle-tsp-top controls number of full-TSP posthoc oracle calls");
    require(tsp_result.stats.oracle_call_records.size() == 3,
            "oracle-tsp-top records each posthoc call");

    SolverOptions subset_options;
    subset_options.oracle = ctx;
    subset_options.subset_restarts = 6;
    subset_options.sa_iters = 0;
    subset_options.final_exhaustive_k = 0;
    subset_options.disable_two_opt = true;
    subset_options.disable_or_opt = true;
    subset_options.disable_pair_exchange = true;
    subset_options.disable_ruin_recreate = true;
    subset_options.disable_path_relink = true;
    subset_options.disable_smallp_seeds = true;
    subset_options.disable_highp_delete = true;
    Rng subset_rng(9092);
    SolveResult subset_result = solve_subset(inst, 18, subset_rng, subset_options);
    require(subset_result.stats.oracle_subset_calls == 2,
            "oracle-subset-top controls number of subset posthoc oracle calls");
    require(subset_result.stats.oracle_call_records.size() == 2,
            "oracle-subset-top records each posthoc call");

    std::filesystem::remove_all(dir, ec);
}


ALDOUS_TEST(test_oracle_posix_spawn_timeout_and_concurrency) {
    const std::string unique_suffix = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    // Spaces and shell metacharacters pin that cwd/executable values are passed
    // as positional argv entries rather than interpolated into a shell command.
    const std::filesystem::path dir = std::filesystem::temp_directory_path()
        / ("aldous tsp $spawn test " + unique_suffix);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    require(!ec, "create POSIX spawn test directory");

    auto make_executable = [&](const std::string& name, const std::string& body) {
        const std::filesystem::path script = dir / name;
        {
            std::ofstream out(script);
            require(static_cast<bool>(out), "open POSIX spawn test script");
            out << "#!/bin/sh\n" << body;
            require(static_cast<bool>(out), "write POSIX spawn test script");
        }
        ec.clear();
        std::filesystem::permissions(
            script,
            std::filesystem::perms::owner_read
                | std::filesystem::perms::owner_write
                | std::filesystem::perms::owner_exec,
            std::filesystem::perm_options::replace,
            ec);
        require(!ec, "mark POSIX spawn test script executable");
        return script;
    };

    // Version capture is bounded and truncates a long first line without a
    // busy-spin. The executable also proves PATH-independent absolute launch.
    const std::filesystem::path long_version = make_executable(
        "long_version_lkh",
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  i=0; while [ $i -lt 240 ]; do printf x; i=$((i + 1)); done; printf '\\n'; exit 0\n"
        "fi\n"
        "cp init.tour out.tour\n"
        "exit 0\n");
    ExternalOracleConfig version_cfg;
    version_cfg.mode = ExternalOracleMode::Lkh;
    version_cfg.lkh_path = long_version.string();
    OracleContext version_ctx;
    std::string error;
    require(build_oracle_context(version_cfg, version_ctx, error),
            "build long-version oracle context");
    require(version_ctx.version.size() == 120U,
            "oracle version capture truncates long first lines deterministically");
    require(std::all_of(version_ctx.version.begin(), version_ctx.version.end(),
                        [](char ch) { return ch == 'x'; }),
            "oracle version capture preserves first-line content");

    const std::filesystem::path silent_version = make_executable(
        "silent version lkh",
        "if [ \"$1\" = \"--version\" ]; then exit 0; fi\n"
        "cp init.tour out.tour\n"
        "exit 0\n");
    ExternalOracleConfig silent_cfg;
    silent_cfg.mode = ExternalOracleMode::Lkh;
    silent_cfg.lkh_path = silent_version.string();
    OracleContext silent_ctx;
    require(build_oracle_context(silent_cfg, silent_ctx, error),
            "build silent-version oracle context");
    require(silent_ctx.version == "unknown",
            "silent version probes terminate cleanly and report unknown");

    Instance inst;
    Rng point_rng(7711);
    inst.generate(28, point_rng);
    inst.build_knn(20, KnnBackend::GridExact);

    // A hung solver is killed as a process group and reaped within one deadline.
    const std::filesystem::path hanging = make_executable(
        "hanging_lkh",
        "if [ \"$1\" = \"--version\" ]; then echo hanging-lkh-1.0; exit 0; fi\n"
        "sleep 30\n"
        "exit 0\n");
    ExternalOracleConfig timeout_cfg;
    timeout_cfg.mode = ExternalOracleMode::Lkh;
    timeout_cfg.lkh_path = hanging.string();
    timeout_cfg.min_k = 3;
    timeout_cfg.max_k = 64;
    timeout_cfg.time_limit_sec = 1;
    OracleContext timeout_ctx;
    require(build_oracle_context(timeout_cfg, timeout_ctx, error),
            "build timeout oracle context");
    Tour candidate;
    candidate.init(inst.N);
    std::vector<int> initial(static_cast<std::size_t>(inst.N));
    std::iota(initial.begin(), initial.end(), 0);
    candidate.set_tour(initial, inst);
    SearchStats timeout_stats;
    const auto timeout_start = std::chrono::steady_clock::now();
    require(!external_oracle_polish_tour(
                candidate, inst, timeout_ctx, true, &timeout_stats, false),
            "timed-out oracle cannot improve a tour");
    const double timeout_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - timeout_start).count();
    require(timeout_elapsed >= 0.8 && timeout_elapsed < 4.0,
            "oracle timeout uses one bounded deadline and reaps promptly");
    require(timeout_stats.oracle_calls == 1 && timeout_stats.oracle_failed == 1,
            "timed-out oracle call is recorded as a failure");

    // Resolve a valid executable, then remove it before the actual call. A
    // posix_spawn launch error must become a structured oracle failure rather
    // than leaking descriptors, leaving a child, or throwing through a worker.
    const std::filesystem::path disappearing = make_executable(
        "disappearing lkh",
        "if [ \"$1\" = \"--version\" ]; then echo disappearing-1.0; exit 0; fi\n"
        "cp init.tour out.tour\n"
        "exit 0\n");
    ExternalOracleConfig missing_cfg;
    missing_cfg.mode = ExternalOracleMode::Lkh;
    missing_cfg.lkh_path = disappearing.string();
    missing_cfg.min_k = 3;
    missing_cfg.max_k = 64;
    OracleContext missing_ctx;
    require(build_oracle_context(missing_cfg, missing_ctx, error),
            "build disappearing oracle context");
    ec.clear();
    require(std::filesystem::remove(disappearing, ec) && !ec,
            "remove oracle executable before launch");
    Tour missing_candidate;
    missing_candidate.init(inst.N);
    missing_candidate.set_tour(initial, inst);
    SearchStats missing_stats;
    require(!external_oracle_polish_tour(
                missing_candidate, inst, missing_ctx, true, &missing_stats, false),
            "spawn failure cannot improve a tour");
    require(missing_stats.oracle_calls == 1 && missing_stats.oracle_failed == 1
                && missing_stats.oracle_call_records.size() == 1U,
            "spawn failure is recorded exactly once");
    const std::string& launch_error =
        missing_stats.oracle_call_records.front().error;
    require(launch_error.find("posix_spawn failed") != std::string::npos
                || launch_error.find("oracle executable is not runnable")
                    != std::string::npos,
            "spawn failure retains a deterministic launch diagnostic");

    // Exercise posix_spawn concurrently from restart workers. The fake solver
    // intentionally uses relative paths, pinning the native working-directory
    // action or safe argv-only fallback as well as launch and redirection.
    const std::filesystem::path copy_script = write_fake_lkh_copy_script(dir / "copy");
    ExternalOracleConfig concurrent_cfg;
    concurrent_cfg.mode = ExternalOracleMode::Lkh;
    concurrent_cfg.lkh_path = copy_script.string();
    concurrent_cfg.min_k = 3;
    concurrent_cfg.max_k = 64;
    concurrent_cfg.subset_top = 0;
    concurrent_cfg.inline_feedback = true;
    concurrent_cfg.time_limit_sec = 5;
    OracleContext concurrent_ctx;
    require(build_oracle_context(concurrent_cfg, concurrent_ctx, error),
            "build concurrent oracle context");

    SolverOptions options;
    options.oracle = concurrent_ctx;
    options.subset_restarts = 8;
    options.restart_threads = 4;
    // This test targets concurrent inline oracle calls rather than finalist
    // staging, so retain the all-restarts strong-search contract explicitly.
    options.staged_search = false;
    options.sa_iters = 0;
    options.final_exhaustive_k = 0;
    options.disable_two_opt = true;
    options.disable_or_opt = true;
    options.disable_subset_swap = true;
    options.disable_pair_exchange = true;
    options.disable_ruin_recreate = true;
    options.disable_path_relink = true;
    options.disable_smallp_seeds = true;
    options.disable_highp_delete = true;
    Rng solve_rng(7712);
    const SolveResult result = solve_subset(inst, 20, solve_rng, options);
    require(result.tour.k == 20 && result.tour.check_invariants(),
            "concurrent spawned oracle calls preserve a valid solve");
    require(result.stats.oracle_subset_calls == 8,
            "every concurrent restart performs its inline oracle call");
    require(result.stats.oracle_solved == 8 && result.stats.oracle_failed == 0,
            "concurrent spawned oracle calls all return usable tours");

    std::filesystem::remove_all(dir, ec);
}



} // namespace
