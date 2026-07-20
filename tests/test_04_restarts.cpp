#include "test_common.hpp"

namespace {

ALDOUS_TEST(test_restart_thread_invariance) {
    Rng rng(3535);
    Instance inst;
    inst.generate(150, rng);
    inst.build_knn(16, KnnBackend::GridExact);
    SolverOptions opt;
    opt.seed = 3535;
    opt.subset_restarts = 5;
    opt.sa_iters = 100;
    opt.final_exhaustive_k = 0;
    opt.disable_subset_swap = true;
    opt.disable_pair_exchange = true;
    opt.disable_ruin_recreate = true;
    opt.disable_path_relink = true;

    opt.restart_threads = 1;
    Rng rng_a(777);
    SolveResult a = solve_subset(inst, 40, rng_a, opt);
    opt.restart_threads = 3;
    Rng rng_b(777);
    SolveResult b = solve_subset(inst, 40, rng_b, opt);

    require(a.tour.nodes == b.tour.nodes, "restart parallelism does not change the solution");
    require(std::abs(a.tour.length - b.tour.length) == 0.0, "restart parallelism does not change the length");
    require(a.stats.sa_moves == b.stats.sa_moves && a.stats.sa_accepted == b.stats.sa_accepted,
            "restart parallelism does not change SA statistics");
    require(a.stats.two_opt_improvements == b.stats.two_opt_improvements
                && a.stats.or_opt_improvements == b.stats.or_opt_improvements,
            "restart parallelism does not change local-search statistics");
    require(a.stats.subset_restarts == b.stats.subset_restarts, "restart counts match");
    require(a.best_restart == b.best_restart, "best-restart diagnostic matches");
    require(a.restarts.size() == b.restarts.size(), "restart record counts match");
    for (std::size_t i = 0; i < a.restarts.size(); ++i) {
        require(a.restarts[i].length == b.restarts[i].length
                    && a.restarts[i].kind == b.restarts[i].kind
                    && a.restarts[i].role == b.restarts[i].role
                    && a.restarts[i].seed_variant == b.restarts[i].seed_variant
                    && a.restarts[i].promotion_stage == b.restarts[i].promotion_stage
                    && a.restarts[i].sa_iterations == b.restarts[i].sa_iterations
                    && a.restarts[i].centroid_x == b.restarts[i].centroid_x
                    && a.restarts[i].centroid_y == b.restarts[i].centroid_y
                    && a.restarts[i].radius == b.restarts[i].radius,
                "restart records are invariant to restart parallelism");
    }
}

ALDOUS_TEST(test_best_restart_diagnostic) {
    Rng rng(9494);
    Instance inst;
    inst.generate(120, rng);
    inst.build_knn(16, KnnBackend::GridExact);
    SolverOptions opt;
    opt.seed = 9494;
    opt.subset_restarts = 4;
    opt.sa_iters = 0;
    opt.final_exhaustive_k = 0;
    opt.disable_two_opt = true;
    opt.disable_or_opt = true;
    opt.disable_subset_swap = true;
    opt.disable_pair_exchange = true;
    opt.disable_ruin_recreate = true;
    opt.disable_path_relink = true;
    Rng solve_rng(9495);
    SolveResult result = solve_subset(inst, 30, solve_rng, opt);
    require(result.best_restart >= 0, "best_restart is set after a subset solve");
    require(static_cast<std::size_t>(result.best_restart) < result.restarts.size(),
            "best_restart indexes an executed subset restart record");
    require(result.restarts.size() == result.stats.subset_restarts,
            "subset record count equals executed restart count");
    Rng tsp_rng(9496);
    opt.tsp_restarts = 4;
    SolveResult tsp = solve_tsp(inst, tsp_rng, opt);
    require(tsp.best_restart >= 0
                && static_cast<std::size_t>(tsp.best_restart) < tsp.restarts.size(),
            "best_restart is set for full TSP solves");
    require(tsp.restarts.size() == tsp.stats.tsp_restarts
                && tsp.restarts.size() == 4U,
            "full TSP emits one record per executed restart");
    require(tsp.stats.tsp_candidate_starts
                == static_cast<std::uint64_t>(opt.tsp_candidate_starts),
            "full TSP records every screened candidate start");
    require(tsp.stats.tsp_promoted_restarts == tsp.restarts.size(),
            "full TSP records every promoted ILS restart");
    for (const RestartRecord& record : tsp.restarts) {
        require(record.kind == RestartKind::TspNearestNeighbor,
                "default full-TSP starts use scalable nearest-neighbor construction");
        require(record.strong_polished,
                "promoted full-TSP restarts are marked strongly polished");
    }

    opt.tsp_farthest_starts = 1;
    opt.tsp_candidate_starts = opt.tsp_restarts;
    Rng diagnostic_rng(9497);
    const SolveResult diagnostic = solve_tsp(inst, diagnostic_rng, opt);
    require(std::any_of(diagnostic.restarts.begin(), diagnostic.restarts.end(),
                        [](const RestartRecord& record) {
                            return record.kind == RestartKind::TspFarthestInsertion;
                        }),
            "farthest insertion remains available as an explicit diagnostic start");
}



std::vector<int> legacy_highp_delete_seed_for_test(
    const Instance& inst,
    const std::vector<int>& parent,
    const int target,
    Rng& rng,
    const int mode) {
    std::vector<int> current = parent;
    while (static_cast<int>(current.size()) > target) {
        const int m = static_cast<int>(current.size());
        std::vector<int> order(static_cast<std::size_t>(m));
        std::vector<double> score(static_cast<std::size_t>(m), 0.0);
        std::iota(order.begin(), order.end(), 0);
        for (int i = 0; i < m; ++i) {
            const int before = current[static_cast<std::size_t>((i - 1 + m) % m)];
            const int node = current[static_cast<std::size_t>(i)];
            const int after = current[static_cast<std::size_t>((i + 1) % m)];
            score[static_cast<std::size_t>(i)] =
                inst.dist(before, node) + inst.dist(node, after)
                - inst.dist(before, after);
            if (mode == 2 && inst.knn_k > 0) {
                score[static_cast<std::size_t>(i)] +=
                    0.15 * inst.knn_d_at(node, std::min(inst.knn_k - 1, 10));
            }
        }
        std::sort(order.begin(), order.end(), [&](const int lhs, const int rhs) {
            if (score[static_cast<std::size_t>(lhs)]
                != score[static_cast<std::size_t>(rhs)]) {
                return score[static_cast<std::size_t>(lhs)]
                     > score[static_cast<std::size_t>(rhs)];
            }
            return current[static_cast<std::size_t>(lhs)]
                 < current[static_cast<std::size_t>(rhs)];
        });
        int erase_position = order.front();
        if (mode == 1) {
            erase_position = order[static_cast<std::size_t>(
                rng.randint(std::min(m, 8)))];
        }
        current.erase(current.begin() + erase_position);
    }
    return current;
}

std::vector<int> legacy_grow_seed_for_test(
    const Instance& inst,
    const std::vector<int>& seed,
    const int target,
    const int mode) {
    std::vector<int> current = seed;
    std::vector<unsigned char> in_set(static_cast<std::size_t>(inst.N), 0U);
    for (const int node : current) {
        in_set[static_cast<std::size_t>(node)] = 1U;
    }
    while (static_cast<int>(current.size()) < target) {
        int best_node = -1;
        int best_position = 0;
        double best_cost = std::numeric_limits<double>::infinity();
        std::vector<int> pool;
        pool.reserve(160U);
        for (const int seed_node : current) {
            const int limit = std::min(inst.knn_k, 16 + 4 * mode);
            for (int rank = 0; rank < limit; ++rank) {
                const int candidate = inst.knn_at(seed_node, rank);
                if (candidate >= 0 && candidate < inst.N
                    && in_set[static_cast<std::size_t>(candidate)] == 0U) {
                    push_unique(pool, candidate, nullptr, 160);
                }
            }
            if (static_cast<int>(pool.size()) >= 160) {
                break;
            }
        }
        if (pool.empty() || inst.N <= 600) {
            for (int candidate = 0; candidate < inst.N; ++candidate) {
                if (in_set[static_cast<std::size_t>(candidate)] == 0U) {
                    push_unique(pool, candidate, nullptr, inst.N);
                }
            }
        }
        const int m = static_cast<int>(current.size());
        for (const int candidate : pool) {
            if (m <= 1) {
                best_node = candidate;
                best_position = m;
                best_cost = 0.0;
                break;
            }
            for (int position = 0; position < m; ++position) {
                const int next_position = position + 1 == m ? 0 : position + 1;
                const double cost =
                    inst.dist(current[static_cast<std::size_t>(position)], candidate)
                    + inst.dist(candidate,
                                current[static_cast<std::size_t>(next_position)])
                    - inst.dist(current[static_cast<std::size_t>(position)],
                                current[static_cast<std::size_t>(next_position)]);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_node = candidate;
                    best_position = position + 1;
                }
            }
        }
        require(best_node >= 0, "legacy growth reference finds a candidate");
        current.insert(current.begin() + best_position, best_node);
        in_set[static_cast<std::size_t>(best_node)] = 1U;
    }
    return current;
}

ALDOUS_TEST(test_seed_resize_chain_differential) {
    for (const bool periodic : {false, true}) {
        for (int trial = 0; trial < 12; ++trial) {
            Rng points_rng(static_cast<std::uint64_t>(7000 + 31 * trial
                                                      + (periodic ? 1 : 0)));
            Instance inst;
            inst.periodic = periodic;
            inst.generate(84, points_rng);
            inst.build_knn(24, KnnBackend::GridExact);
            Rng subset_rng(static_cast<std::uint64_t>(8100 + trial));
            std::vector<int> parent = random_subset(inst.N, 68, subset_rng);
            parent = nearest_neighbor_order(inst, parent, trial % 68);

            for (int mode = 0; mode <= 2; ++mode) {
                Rng legacy_rng(static_cast<std::uint64_t>(9000 + 101 * trial + mode));
                Rng heap_rng(static_cast<std::uint64_t>(9000 + 101 * trial + mode));
                const std::vector<int> expected = legacy_highp_delete_seed_for_test(
                    inst, parent, 29, legacy_rng, mode);
                const std::vector<int> actual = highp_delete_seed(
                    inst, parent, 29, heap_rng, mode);
                require(actual == expected,
                        "heap shrink exactly matches the legacy deletion trajectory");
            }

            std::vector<int> small(parent.begin(), parent.begin() + 9);
            for (int mode = 0; mode <= 1; ++mode) {
                Rng grow_rng(static_cast<std::uint64_t>(10000 + trial + mode));
                const std::vector<int> actual = resize_seed(inst, small, 43, grow_rng, mode);
                const std::vector<int> expected = legacy_grow_seed_for_test(
                    inst, small, 43, mode);
                require(actual == expected,
                        "cached growth exactly matches the legacy insertion trajectory");
            }

            Rng chain_rng(static_cast<std::uint64_t>(11000 + trial));
            const auto shrink_snapshots = shrink_seed_chain(
                inst, parent, {55, 41, 27}, chain_rng, 1);
            for (std::size_t i = 0; i < shrink_snapshots.size(); ++i) {
                Rng reference_rng(static_cast<std::uint64_t>(11000 + trial));
                const int target = std::vector<int>({55, 41, 27})[i];
                require(shrink_snapshots[i]
                            == legacy_highp_delete_seed_for_test(
                                inst, parent, target, reference_rng, 1),
                        "one shrink trajectory emits exact reusable snapshots");
            }

            Rng growth_chain_rng(static_cast<std::uint64_t>(12000 + trial));
            const auto growth_snapshots = grow_seed_chain(
                inst, small, {17, 31, 47}, growth_chain_rng, 0);
            const std::vector<int> growth_targets = {17, 31, 47};
            for (std::size_t i = 0; i < growth_snapshots.size(); ++i) {
                require(growth_snapshots[i]
                            == legacy_grow_seed_for_test(
                                inst, small, growth_targets[i], 0),
                        "one growth trajectory emits exact reusable snapshots");
            }
        }
    }
}

ALDOUS_TEST(test_continuation_stream_contract) {
    RunOptions standalone;
    standalone.N = 72;
    standalone.instances = 1;
    standalone.threads = 1;
    standalone.p_values = {0.4};
    standalone.include_instance_rows = true;
    standalone.solver.seed = 424242;
    standalone.solver.subset_restarts = 3;
    standalone.solver.continuation_restarts = 1;
    standalone.solver.sa_iters = 0;
    standalone.solver.restart_threads = 2;
    standalone.solver.final_exhaustive_k = 0;
    standalone.solver.disable_two_opt = true;
    standalone.solver.disable_or_opt = true;
    standalone.solver.disable_subset_swap = true;
    standalone.solver.disable_pair_exchange = true;
    standalone.solver.disable_ruin_recreate = true;
    standalone.solver.disable_path_relink = true;
    standalone.solver.disable_smallp_seeds = true;
    standalone.solver.disable_highp_delete = true;

    RunOptions grid = standalone;
    grid.p_values = {0.8, 0.2, 0.4, 0.4};
    const ResultsDocument single_doc = ExperimentRunner(standalone).run();
    const ResultsDocument grid_doc = ExperimentRunner(grid).run();
    require(grid_doc.p_values == std::vector<double>({0.2, 0.4, 0.8}),
            "ExperimentRunner canonicalizes direct-API p grids");

    const auto find_p = [](const InstanceResultRow& row, const double target)
        -> const InstancePValueRow& {
        const auto found = std::find_if(
            row.p_results.begin(), row.p_results.end(),
            [target](const InstancePValueRow& value) {
                return std::abs(value.p - target) <= 1e-12;
            });
        if (found == row.p_results.end()) {
            throw std::runtime_error("missing p row in continuation contract test");
        }
        return *found;
    };
    const InstancePValueRow& single = find_p(single_doc.instance_rows.front(), 0.4);
    const InstancePValueRow& embedded = find_p(grid_doc.instance_rows.front(), 0.4);
    std::vector<RestartRecord> single_independent;
    std::vector<RestartRecord> embedded_independent;
    for (const RestartRecord& record : single.restarts) {
        if (record.role == RestartRole::IndependentDiagnostic) {
            single_independent.push_back(record);
        }
    }
    for (const RestartRecord& record : embedded.restarts) {
        if (record.role == RestartRole::IndependentDiagnostic
            && record.sweep == RestartSweep::Primary) {
            embedded_independent.push_back(record);
        }
    }
    require(single_independent.size() == 3U
                && embedded_independent.size() == single_independent.size(),
            "the independent restart quota is unchanged by neighboring p values");
    for (std::size_t i = 0; i < single_independent.size(); ++i) {
        const RestartRecord& a = single_independent[i];
        const RestartRecord& b = embedded_independent[i];
        require(a.length == b.length && a.kind == b.kind && a.role == b.role
                    && a.seed_variant == b.seed_variant
                    && a.centroid_x == b.centroid_x
                    && a.centroid_y == b.centroid_y && a.radius == b.radius,
                "independent restart records are p-grid invariant");
    }
    require(embedded.value <= single.value + 1e-12,
            "supplemental continuation cannot worsen the standalone result");
    require(std::count_if(embedded.restarts.begin(), embedded.restarts.end(),
                          [](const RestartRecord& record) {
                              return record.role == RestartRole::Continuation;
                          }) == 1,
            "supplemental mode appends the configured warm quota");

    Rng point_rng(9191);
    Instance inst;
    inst.generate(96, point_rng);
    inst.build_knn(16, KnnBackend::GridExact);
    std::vector<int> warm = all_nodes(inst.N);
    warm.resize(60U);
    SolverOptions fixed = standalone.solver;
    fixed.subset_restarts = 4;
    fixed.continuation_restarts = 1;
    fixed.continuation_policy = ContinuationPolicy::FixedBudget;
    Rng fixed_rng(8181);
    const SolveResult fixed_result = solve_subset(inst, 40, fixed_rng, fixed, &warm);
    const auto count_role = [](const SolveResult& value, const RestartRole role) {
        return static_cast<int>(std::count_if(
            value.restarts.begin(), value.restarts.end(),
            [role](const RestartRecord& record) { return record.role == role; }));
    };
    require(fixed_result.restarts.size() == 4U
                && count_role(fixed_result, RestartRole::IndependentDiagnostic) == 3
                && count_role(fixed_result, RestartRole::Continuation) == 1,
            "fixed-budget continuation reserves an explicit quota");

    fixed.continuation_policy = ContinuationPolicy::Supplemental;
    Rng supplemental_rng(8181);
    const SolveResult supplemental = solve_subset(inst, 40, supplemental_rng, fixed, &warm);
    require(supplemental.restarts.size() == 5U
                && count_role(supplemental, RestartRole::IndependentDiagnostic) == 4
                && count_role(supplemental, RestartRole::Continuation) == 1,
            "supplemental continuation preserves every independent draw");
}

ALDOUS_TEST(test_deterministic_restart_racing) {
    Rng point_rng(515151);
    Instance inst;
    inst.periodic = true;
    inst.generate(128, point_rng);
    inst.build_knn(20, KnnBackend::GridExact);

    SolverOptions base;
    base.subset_restarts = 3;
    base.continuation_restarts = 0;
    base.sa_iters = 80;
    base.restart_threads = 1;
    base.final_exhaustive_k = 0;
    base.disable_or_opt = true;
    base.disable_subset_swap = true;
    base.disable_pair_exchange = true;
    base.disable_ruin_recreate = true;
    base.disable_path_relink = true;

    Rng base_rng(616161);
    const SolveResult baseline = solve_subset(inst, 48, base_rng, base);

    SolverOptions raced = base;
    raced.racing_candidates = 6;
    raced.racing_survivors = 2;
    raced.racing_pilot_iters = 15;
    raced.racing_min_jaccard = 0.05;
    Rng raced_rng(616161);
    const SolveResult one_thread = solve_subset(inst, 48, raced_rng, raced);

    require(one_thread.restarts.size() == 9U
                && one_thread.stats.subset_restarts == 9,
            "racing appends one final record per unique pilot candidate");
    require(one_thread.stats.racing_pilot_restarts == 6
                && one_thread.stats.racing_promoted_restarts == 2,
            "racing exposes exact pilot and promotion counts");
    require(one_thread.tour.length <= baseline.tour.length + 1e-12,
            "supplemental racing cannot worsen the independent result");

    require(baseline.restarts.size() == 3U, "baseline has its independent quota");
    for (std::size_t i = 0; i < baseline.restarts.size(); ++i) {
        const RestartRecord& a = baseline.restarts[i];
        const RestartRecord& b = one_thread.restarts[i];
        require(a.length == b.length && a.kind == b.kind && a.sweep == b.sweep
                    && a.role == b.role && a.seed_variant == b.seed_variant
                    && a.promotion_stage == b.promotion_stage
                    && a.sa_iterations == b.sa_iterations
                    && a.centroid_x == b.centroid_x
                    && a.centroid_y == b.centroid_y && a.radius == b.radius,
                "enabling racing leaves independent diagnostics bit-identical");
    }

    int pilot_only = 0;
    int promoted_full = 0;
    std::set<int> race_variants;
    for (std::size_t i = baseline.restarts.size(); i < one_thread.restarts.size(); ++i) {
        const RestartRecord& record = one_thread.restarts[i];
        require(record.role == RestartRole::RacedProduction,
                "racing records use the separate production role");
        race_variants.insert(record.seed_variant);
        if (record.promotion_stage == RestartPromotionStage::PilotOnly) {
            ++pilot_only;
            require(record.sa_iterations == static_cast<std::uint64_t>(raced.racing_pilot_iters),
                    "non-promoted candidates report their pilot budget");
        } else if (record.promotion_stage == RestartPromotionStage::PromotedFull) {
            ++promoted_full;
            require(record.sa_iterations == static_cast<std::uint64_t>(
                    raced.racing_pilot_iters + raced.sa_iters),
                    "promoted candidates report pilot plus full-depth work");
        } else {
            require(false, "racing records expose a valid promotion stage");
        }
    }
    require(pilot_only == 4 && promoted_full == 2 && race_variants.size() == 6U,
            "the stable promotion rule fills the requested survivor quota");

    raced.restart_threads = 3;
    Rng parallel_rng(616161);
    const SolveResult parallel = solve_subset(inst, 48, parallel_rng, raced);
    require(parallel.tour.nodes == one_thread.tour.nodes
                && parallel.tour.length == one_thread.tour.length
                && parallel.best_restart == one_thread.best_restart,
            "racing result is invariant to restart worker count");
    require(parallel.stats.subset_restarts == one_thread.stats.subset_restarts
                && parallel.stats.racing_pilot_restarts
                    == one_thread.stats.racing_pilot_restarts
                && parallel.stats.racing_promoted_restarts
                    == one_thread.stats.racing_promoted_restarts
                && parallel.stats.sa_moves == one_thread.stats.sa_moves
                && parallel.stats.sa_accepted == one_thread.stats.sa_accepted,
            "racing discrete telemetry is invariant to restart worker count");
    require(parallel.restarts.size() == one_thread.restarts.size(),
            "racing record count is thread invariant");
    for (std::size_t i = 0; i < one_thread.restarts.size(); ++i) {
        const RestartRecord& a = one_thread.restarts[i];
        const RestartRecord& b = parallel.restarts[i];
        require(a.length == b.length && a.kind == b.kind && a.sweep == b.sweep
                    && a.role == b.role && a.seed_variant == b.seed_variant
                    && a.promotion_stage == b.promotion_stage
                    && a.sa_iterations == b.sa_iterations
                    && a.centroid_x == b.centroid_x
                    && a.centroid_y == b.centroid_y && a.radius == b.radius,
                "racing records are invariant to restart worker count");
    }

    SolverOptions invalid = raced;
    invalid.time_budget_per_p = 0.01;
    bool threw = false;
    try {
        Rng invalid_rng(616161);
        (void)solve_subset(inst, 48, invalid_rng, invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "deterministic racing rejects wall-clock anytime mode");
}

ALDOUS_TEST(test_second_sweep_never_worse) {
    RunOptions base;
    base.N = 48;
    base.instances = 1;
    base.threads = 1;
    base.p_values = {0.25, 0.5, 1.0};
    base.include_instance_rows = true;
    base.solver.seed = 6161;
    base.solver.subset_restarts = 2;
    base.solver.sa_iters = 0;
    base.solver.restart_threads = 1;
    base.solver.tsp_restarts = 2;
    base.solver.tsp_ils = 1;
    base.solver.final_exhaustive_k = 0;
    base.solver.disable_two_opt = true;
    base.solver.disable_or_opt = true;
    base.solver.disable_subset_swap = true;
    base.solver.disable_pair_exchange = true;
    base.solver.disable_ruin_recreate = true;
    base.solver.disable_path_relink = true;

    RunOptions swept = base;
    swept.second_sweep = true;

    const ResultsDocument off = ExperimentRunner(base).run();
    const ResultsDocument on = ExperimentRunner(swept).run();
    require(off.summary.size() == on.summary.size(), "second sweep keeps the p grid");
    for (const auto& item : off.summary) {
        const auto found = on.summary.find(item.first);
        require(found != on.summary.end(), "second sweep keeps every p key");
        require(found->second.mean <= item.second.mean + 1e-9,
                "second sweep never worsens a per-p mean");
        require(found->second.best_restart_max >= -1,
                "summary rows carry the best-restart diagnostic");
    }

    require(on.instance_rows.size() == static_cast<std::size_t>(base.instances),
            "second-sweep diagnostics retain every instance row");
    for (const InstanceResultRow& row : on.instance_rows) {
        require(row.p_results.size() == base.p_values.size(),
                "second-sweep diagnostics retain every p row");
        for (std::size_t pi = 0; pi < row.p_results.size(); ++pi) {
            const InstancePValueRow& pv = row.p_results[pi];
            require(pv.executed_restarts == static_cast<int>(pv.restarts.size()),
                    "executed_restarts exactly matches serialized records");
            require(pv.best_restart >= 0
                        && static_cast<std::size_t>(pv.best_restart) < pv.restarts.size(),
                    "best_restart indexes the combined serialized population");

            const bool has_secondary = pi > 0U && pv.k < base.N;
            const std::size_t primary_count =
                off.instance_rows.front().p_results[pi].restarts.size();
            const std::size_t secondary_count = has_secondary
                ? static_cast<std::size_t>(base.solver.continuation_restarts)
                : 0U;
            require(pv.restarts.size() == primary_count + secondary_count,
                    "second sweep appends continuation-only records");
            for (std::size_t ri = 0; ri < pv.restarts.size(); ++ri) {
                const RestartSweep expected_sweep = ri >= primary_count
                    ? RestartSweep::Secondary
                    : RestartSweep::Primary;
                require(pv.restarts[ri].sweep == expected_sweep,
                        "restart sweep tags preserve append order");
                if (ri >= primary_count) {
                    require(pv.restarts[ri].role == RestartRole::Continuation,
                            "secondary sweep records only continuation work");
                }
            }
            if (pv.k == base.N) {
                require(std::all_of(pv.restarts.begin(), pv.restarts.end(),
                                    [](const RestartRecord& record) {
                                        return record.kind == RestartKind::TspNearestNeighbor;
                                    }),
                        "p=1 rows serialize the promoted scalable TSP population");
            }
        }
    }
}


} // namespace
