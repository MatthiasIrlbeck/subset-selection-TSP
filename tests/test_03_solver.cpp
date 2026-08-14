#include "test_common.hpp"

namespace {

ALDOUS_TEST(test_solver_smoke) {
    Rng rng(123);
    Instance inst;
    inst.generate(35, rng);
    inst.build_knn(12);
    SolverOptions opt;
    opt.knn_k = 12;
    opt.tsp_restarts = 1;
    opt.tsp_ils = 5;
    opt.tsp_patience = 2;
    opt.subset_restarts = 1;
    opt.sa_iters = 25;
    opt.final_exhaustive_k = 80;
    Rng solve_rng(456);
    SolveResult tsp = solve_tsp(inst, solve_rng, opt);
    require(tsp.tour.k == inst.N, "TSP solver returns full tour");
    require(std::isfinite(tsp.tour.length), "TSP solver finite length");
    SolveResult subset = solve_subset(inst, 10, solve_rng, opt, &tsp.tour.nodes);
    require(subset.tour.k == 10, "subset solver returns requested k");
    require(std::isfinite(subset.tour.length), "subset solver finite length");
    require(subset.tour.check_invariants(), "subset invariants");
}


ALDOUS_TEST(test_object_oriented_facades) {
    Rng rng(4242);
    Instance inst;
    inst.generate(36, rng);
    inst.build_knn(12);

    SolverOptions options;
    options.knn_k = 12;
    options.tsp_restarts = 1;
    options.tsp_ils = 4;
    options.tsp_patience = 2;
    options.subset_restarts = 1;
    options.sa_iters = 20;
    options.final_exhaustive_k = 80;

    TspSolver tsp_solver(options);
    Rng tsp_rng(4243);
    SolveResult tsp = tsp_solver.solve(inst, tsp_rng);
    require(tsp.tour.k == inst.N, "TspSolver facade returns full tour");

    SubsetSolver subset_solver(options);
    Rng subset_rng(4244);
    SolveResult subset = subset_solver.solve_with_warm_start(inst, 12, subset_rng, tsp.tour.nodes);
    require(subset.tour.k == 12, "SubsetSolver facade returns requested subset size");
    require(subset.tour.check_invariants(), "SubsetSolver facade result has valid invariants");

    RunOptions run_options;
    run_options.N = 30;
    run_options.instances = 1;
    run_options.threads = 1;
    run_options.p_values = {0.5, 1.0};
    run_options.solver = options;
    run_options.solver.knn_k = 10;
    ExperimentRunner runner(run_options);
    ResultsDocument doc = runner.run();
    require(doc.instances_done == 1, "ExperimentRunner facade completes one instance");
    require(doc.summary.size() == 2U, "ExperimentRunner facade creates summaries for requested p-values");
}

ALDOUS_TEST(test_disable_two_opt_ablation_exact) {
    Rng rng(321);
    Instance inst;
    inst.generate(42, rng);
    inst.build_knn(14, KnnBackend::GridExact);
    SolverOptions opt;
    opt.disable_two_opt = true;
    opt.tsp_restarts = 2;
    opt.tsp_ils = 4;
    opt.tsp_patience = 2;
    opt.subset_restarts = 2;
    opt.sa_iters = 1100;
    opt.subset_swap_descent_passes = 2;
    opt.pair_exchange_passes = 1;
    opt.ruin_recreate_rounds = 1;
    opt.path_relink_top = 2;
    opt.final_exhaustive_k = 80;
    Rng solve_rng(654);
    SolveResult tsp = solve_tsp(inst, solve_rng, opt);
    SolveResult subset = solve_subset(inst, 12, solve_rng, opt, &tsp.tour.nodes);
    require(tsp.stats.two_opt_scans == 0 && tsp.stats.two_opt_improvements == 0, "disable-two-opt suppresses TSP two-opt stats");
    require(subset.stats.two_opt_scans == 0 && subset.stats.two_opt_improvements == 0, "disable-two-opt suppresses subset two-opt stats");
}


ALDOUS_TEST(test_or_opt_reaches_local_optimum) {
    // The first-improvement rewrite must still terminate at a candidate-local
    // optimum: after the descent, no single-node relocation onto any candidate
    // insertion position may improve the tour. This is the invariant that both
    // best- and first-improvement schemes must satisfy.
    Rng rng(31337);
    Instance inst;
    inst.generate(200, rng);
    inst.build_knn(16, KnnBackend::GridExact);
    for (int trial = 0; trial < 12; ++trial) {
        const int k = 20 + rng.randint(80);
        std::vector<int> nodes = random_subset(inst.N, k, rng);
        Tour tour;
        tour.init(inst.N);
        tour.set_tour(nodes, inst);
        const SubsetCandidateTable* table = maybe_subset_candidates(inst, tour);
        or_opt_1_candidate_descent(tour, inst, 50, 0, nullptr, table);
        require(tour.check_invariants(), "tour valid after or-opt descent");

        // Independently verify the guaranteed invariant: no improving
        // relocation remains onto a spatially-near insertion position (the
        // KNN/table candidate neighborhood). Reverse-KNN wakeups provably cover
        // this neighborhood. The +/-3 positional offsets the descent also scans
        // are an opportunistic stabilizer and are intentionally not part of the
        // convergence guarantee, so they are excluded here.
        double best_remaining = 0.0;
        for (int i = 0; i < tour.k; ++i) {
            const int node = tour.nodes[static_cast<std::size_t>(i)];
            std::vector<int> preds;
            const int row = table ? table->row_of_node[static_cast<std::size_t>(node)] : -1;
            const int limit = row >= 0 ? table->m : std::min(inst.knn_k, 16);
            for (int r = 0; r < limit; ++r) {
                const int near = row >= 0
                    ? table->ids[static_cast<std::size_t>(row) * static_cast<std::size_t>(table->m) + static_cast<std::size_t>(r)]
                    : inst.knn_at(node, r);
                if (near < 0) { break; }
                const int pos = tour.pos[static_cast<std::size_t>(near)];
                if (pos >= 0 && pos != i) {
                    push_unique(preds, pos);
                    push_unique(preds, (pos == 0 ? tour.k - 1 : pos - 1));
                }
            }
            if (preds.empty()) { continue; }
            const SwapMoveEval eval = evaluate_move_after_remove(inst, tour, i, &preds);
            if (eval.valid) { best_remaining = std::min(best_remaining, eval.delta); }
        }
        require(best_remaining >= -1e-7,
                "no improving spatial relocation remains after or-opt descent (local optimum)");
    }
}

ALDOUS_TEST(test_or_opt_segment_property) {
    Rng rng(5151);
    Instance inst;
    inst.generate(180, rng);
    inst.build_knn(16, KnnBackend::GridExact);
    for (int trial = 0; trial < 10; ++trial) {
        const int k = 15 + rng.randint(60);
        std::vector<int> nodes = random_subset(inst.N, k, rng);
        Tour tour;
        tour.init(inst.N);
        tour.set_tour(nodes, inst);
        const SubsetCandidateTable* table = maybe_subset_candidates(inst, tour);
        for (int L = 2; L <= 3; ++L) {
            const double before = tour.length;
            const int applied = or_opt_segment_candidate_descent(tour, inst, L, 6, 12, nullptr, table);
            require(tour.check_invariants(), "tour invariants hold after segment or-opt");
            require(tour.k == k, "segment or-opt preserves subset size");
            require(tour.length <= before + 1e-9, "segment or-opt never increases length");
            const double recomputed = cycle_length(inst, tour.nodes);
            require(std::abs(tour.length - recomputed) <= 1e-6 * (1.0 + recomputed),
                    "tour length matches recomputation after segment or-opt");
            std::vector<unsigned char> seen(static_cast<std::size_t>(inst.N), 0U);
            for (int v : tour.nodes) {
                require(v >= 0 && v < inst.N && seen[static_cast<std::size_t>(v)] == 0U,
                        "segment or-opt output is a permutation of the subset");
                seen[static_cast<std::size_t>(v)] = 1U;
            }
            (void)applied;
        }
    }
}

ALDOUS_TEST(test_solver_matches_exact_enumeration_tiny) {
    // Quality canary: on tiny instances the solver must find the global
    // optimum over all C(N, k) subsets x optimal cycles. The solver is
    // deterministic for a fixed seed, so this is stable; if a future search
    // change lands on a suboptimal basin here, that is a quality regression
    // worth seeing.
    Rng rng(2718);
    for (int trial = 0; trial < 3; ++trial) {
        Instance inst;
        inst.generate(10, rng);
        inst.build_knn(9, KnnBackend::GridExact);
        const int k = 5;
        // Exact reference: enumerate all subsets.
        std::vector<int> comb(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) { comb[static_cast<std::size_t>(i)] = i; }
        double exact_best = std::numeric_limits<double>::infinity();
        for (;;) {
            std::vector<int> cycle;
            double len = 0.0;
            require(exact_small_tsp_cycle(inst, comb, cycle, len), "exact cycle solves tiny subsets");
            exact_best = std::min(exact_best, len);
            int pos = k - 1;
            while (pos >= 0 && comb[static_cast<std::size_t>(pos)] == inst.N - k + pos) { --pos; }
            if (pos < 0) { break; }
            ++comb[static_cast<std::size_t>(pos)];
            for (int q = pos + 1; q < k; ++q) {
                comb[static_cast<std::size_t>(q)] = comb[static_cast<std::size_t>(q - 1)] + 1;
            }
        }
        SolverOptions opt;
        opt.seed = 2718 + trial;
        opt.subset_restarts = 4;
        opt.sa_iters = 20000;
        Rng solve_rng(static_cast<std::uint64_t>(1000 + trial));
        SolveResult result = solve_subset(inst, k, solve_rng, opt);
        require(std::abs(result.tour.length - exact_best) <= 1e-9 * (1.0 + exact_best),
                "solver finds the enumerated global optimum on tiny instances");
    }
}

ALDOUS_TEST(test_subset_candidate_table_exact) {
    Rng rng(7171);
    for (int backend = 0; backend < 2; ++backend) {
        Instance inst;
        inst.generate(150, rng);
        inst.build_knn(12, backend == 0 ? KnnBackend::GridExact : KnnBackend::BruteForce);
        for (int trial = 0; trial < 10; ++trial) {
            const int k = 10 + rng.randint(60);
            std::vector<int> nodes = random_subset(inst.N, k, rng);
            Tour tour;
            tour.init(inst.N);
            tour.set_tour(nodes, inst);
            SubsetCandidateTable table;
            const int m = 6;
            build_subset_candidates(inst, tour, m, table);
            require(table.m == std::min(m, k - 1), "subset table m respects k");
            for (int i = 0; i < tour.k; ++i) {
                const int a = tour.nodes[static_cast<std::size_t>(i)];
                require(table.row_of_node[static_cast<std::size_t>(a)] == i, "subset table rows map members");
                // Brute-force reference: deterministic (distance, node id).
                std::vector<std::pair<double, int>> ref;
                for (int j = 0; j < tour.k; ++j) {
                    const int v = tour.nodes[static_cast<std::size_t>(j)];
                    if (v != a) { ref.emplace_back(inst.dist(a, v), v); }
                }
                std::sort(ref.begin(), ref.end());
                for (int r = 0; r < table.m; ++r) {
                    const std::size_t slot = static_cast<std::size_t>(i) * static_cast<std::size_t>(table.m) + static_cast<std::size_t>(r);
                    const int c = table.ids[slot];
                    require(c >= 0 && c < inst.N && tour.in_set[static_cast<std::size_t>(c)] != 0U && c != a,
                            "subset table candidates are other subset members");
                    require(c == ref[static_cast<std::size_t>(r)].second,
                            "subset table uses exact deterministic nearest-member ids");
                    require(std::abs(table.dist[slot] - ref[static_cast<std::size_t>(r)].first)
                                <= 1e-9 * (1.0 + ref[static_cast<std::size_t>(r)].first),
                            "subset table distances are the exact m nearest member distances");
                }
            }
        }
    }
}

ALDOUS_TEST(test_effective_sa_iters_scaling) {
    SolverOptions opt;
    opt.sa_iters = 1000;
    opt.sa_iters_per_k = 0;
    opt.sa_iters_per_n = 0;
    require(effective_sa_iters(opt, 500, 10000) == 1000, "flat budget when both scalings are off");
    opt.sa_iters_per_k = 40;
    require(effective_sa_iters(opt, 0, 10000) == 1000, "per-k scaling contributes nothing at k=0");
    require(effective_sa_iters(opt, 250, 10000) == 1000 + 40 * 250, "per-k scaling is linear in k");

    // Per-N scaling: the subset search picks k of N, so its budget must be able
    // to track the CANDIDATE POOL, not just the subset size. Without this the
    // search cannot even propose every candidate once at large N, and the extra
    // candidates a smaller p buys are never examined.
    opt.sa_iters_per_k = 0;
    opt.sa_iters_per_n = 20;
    require(effective_sa_iters(opt, 250, 0) == 1000, "per-N scaling contributes nothing at N=0");
    require(effective_sa_iters(opt, 250, 50000) == 1000 + 20 * 50000, "per-N scaling is linear in N");
    opt.sa_iters_per_k = 40;
    require(effective_sa_iters(opt, 250, 50000) == 1000 + 40 * 250 + 20 * 50000,
            "per-k and per-N scaling compose additively");

    opt.sa_iters = std::numeric_limits<int>::max();
    opt.sa_iters_per_k = std::numeric_limits<int>::max();
    opt.sa_iters_per_n = std::numeric_limits<int>::max();
    require(effective_sa_iters(opt, std::numeric_limits<int>::max(), std::numeric_limits<int>::max()) == std::numeric_limits<int>::max(),
            "effective SA budget saturates instead of overflowing");
}

ALDOUS_TEST(test_restart_worker_exception_propagation) {
    std::atomic<int> active{0};
    bool caught = false;
    try {
        detail::run_parallel_indexed(12, [&](int index) {
            active.fetch_add(1, std::memory_order_relaxed);
            try {
                if (index == 3) {
                    throw std::runtime_error("restart worker failure");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } catch (...) {
                active.fetch_sub(1, std::memory_order_relaxed);
                throw;
            }
            active.fetch_sub(1, std::memory_order_relaxed);
        });
    } catch (const std::runtime_error& error) {
        caught = std::string(error.what()) == "restart worker failure";
    }
    require(caught, "restart worker exception is rethrown on the caller thread");
    require(active.load(std::memory_order_relaxed) == 0,
            "every restart worker is joined before rethrowing");

    std::atomic<int> completed{0};
    detail::run_parallel_indexed(8, [&](int) {
        completed.fetch_add(1, std::memory_order_relaxed);
    });
    require(completed.load(std::memory_order_relaxed) == 8,
            "parallel restart runner remains usable after an exception");
}

ALDOUS_TEST(test_experiment_worker_states_and_callback_exceptions) {
    // Single-threaded first/middle/last failures pin exact state transitions.
    for (const int failure_index : {0, 2, 4}) {
        std::vector<detail::WorkerOutcome<int>> outcomes;
        int completion_calls = 0;
        bool caught = false;
        try {
            (void)detail::run_parallel_work_queue<int>(
                5,
                1,
                outcomes,
                [&](int index) {
                    if (index == failure_index) {
                        throw std::runtime_error("instance worker failure");
                    }
                    return index;
                },
                [&](int, const int&) { ++completion_calls; });
        } catch (const std::runtime_error& error) {
            caught = std::string(error.what()) == "instance worker failure";
        }
        require(caught, "instance worker failure reaches the caller");
        require(outcomes.size() == 5U, "work queue keeps one outcome per target");
        for (int index = 0; index < 5; ++index) {
            const detail::WorkerState state =
                outcomes[static_cast<std::size_t>(index)].state;
            if (index < failure_index) {
                require(state == detail::WorkerState::Success,
                        "instances before a serial failure are successful");
            } else if (index == failure_index) {
                require(state == detail::WorkerState::Failure,
                        "the failing instance has an explicit failure state");
                require(outcomes[static_cast<std::size_t>(index)].exception != nullptr,
                        "the failing instance retains its exception");
            } else {
                require(state == detail::WorkerState::Cancelled,
                        "unstarted instances are explicitly cancelled");
            }
        }
        require(completion_calls == failure_index,
                "only successful instance outcomes reach completion handling");
    }

    // Under parallel scheduling the exact cancelled set is timing-dependent,
    // but every slot must still end in a valid terminal state.
    std::vector<detail::WorkerOutcome<int>> parallel_outcomes;
    std::atomic<int> active{0};
    bool parallel_caught = false;
    try {
        (void)detail::run_parallel_work_queue<int>(
            24,
            6,
            parallel_outcomes,
            [&](int index) {
                active.fetch_add(1, std::memory_order_relaxed);
                try {
                    if (index == 3) {
                        throw std::runtime_error("parallel instance failure");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                } catch (...) {
                    active.fetch_sub(1, std::memory_order_relaxed);
                    throw;
                }
                active.fetch_sub(1, std::memory_order_relaxed);
                return index;
            },
            [](int, const int&) {});
    } catch (const std::runtime_error& error) {
        parallel_caught = std::string(error.what()) == "parallel instance failure";
    }
    require(parallel_caught, "parallel instance failure reaches the caller");
    require(active.load(std::memory_order_relaxed) == 0,
            "parallel instance workers are all joined before rethrow");
    int failures = 0;
    for (const auto& outcome : parallel_outcomes) {
        require(outcome.state != detail::WorkerState::NotStarted
                    && outcome.state != detail::WorkerState::Running,
                "every parallel instance slot has a terminal state");
        if (outcome.state == detail::WorkerState::Failure) {
            ++failures;
        }
    }
    require(failures == 1, "only the injected parallel instance fails");

    // Completion callbacks are marshalled to the caller thread. Their
    // exceptions cancel new work, suppress later callbacks, join all workers,
    // and are then rethrown without std::terminate.
    std::vector<detail::WorkerOutcome<int>> callback_outcomes;
    const std::thread::id caller_thread = std::this_thread::get_id();
    bool callback_on_caller = true;
    int callback_calls = 0;
    bool callback_caught = false;
    try {
        (void)detail::run_parallel_work_queue<int>(
            32,
            4,
            callback_outcomes,
            [](int index) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                return index;
            },
            [&](int, const int&) {
                callback_on_caller = callback_on_caller
                    && std::this_thread::get_id() == caller_thread;
                ++callback_calls;
                throw std::runtime_error("callback failure");
            });
    } catch (const std::runtime_error& error) {
        callback_caught = std::string(error.what()) == "callback failure";
    }
    require(callback_caught, "completion callback exception reaches the caller");
    require(callback_on_caller, "completion callback runs on the caller thread");
    require(callback_calls == 1,
            "no further completion callbacks run after one callback throws");
    for (const auto& outcome : callback_outcomes) {
        require(outcome.state != detail::WorkerState::NotStarted
                    && outcome.state != detail::WorkerState::Running,
                "callback cancellation leaves no incomplete worker states");
    }

    RunOptions options;
    options.N = 12;
    options.instances = 4;
    options.threads = 2;
    options.p_values = {1.0};
    options.solver.knn_k = 4;
    options.solver.tsp_restarts = 1;
    options.solver.tsp_ils = 0;
    options.solver.sa_iters = 0;
    options.solver.final_exhaustive_k = 0;
    options.solver.disable_two_opt = true;
    options.solver.disable_or_opt = true;
    options.solver.oracle.cfg.mode = ExternalOracleMode::None;
    ExperimentRunner runner(options);
    bool runner_callback_caught = false;
    bool runner_callback_on_caller = true;
    try {
        (void)runner.run([&](const ExperimentProgress&) {
            runner_callback_on_caller = runner_callback_on_caller
                && std::this_thread::get_id() == caller_thread;
            throw std::runtime_error("experiment callback failure");
        });
    } catch (const std::runtime_error& error) {
        runner_callback_caught =
            std::string(error.what()) == "experiment callback failure";
    }
    require(runner_callback_caught,
            "ExperimentRunner propagates callback exceptions without terminating");
    require(runner_callback_on_caller,
            "ExperimentRunner progress callbacks run on the caller thread");
}

ALDOUS_TEST(test_elite_kick_near_full) {
    Instance inst;
    Rng point_rng(91);
    inst.generate(100, point_rng);
    inst.build_knn(40, KnnBackend::GridExact);

    auto require_valid_seed = [&](const std::vector<int>& seed, int expected_size) {
        require(static_cast<int>(seed.size()) == expected_size,
                "elite kick preserves subset cardinality");
        std::vector<unsigned char> seen(static_cast<std::size_t>(inst.N), 0U);
        for (const int node : seed) {
            require(node >= 0 && node < inst.N,
                    "elite kick keeps every node in range");
            require(seen[static_cast<std::size_t>(node)] == 0U,
                    "elite kick keeps every node unique");
            seen[static_cast<std::size_t>(node)] = 1U;
        }
    };

    // Exercise the primitive directly over empty, singleton, near-full and full
    // subsets. The full case has no external free node and must still be total.
    for (const int k : {0, 1, inst.N - 1, inst.N}) {
        for (std::uint64_t stream = 0; stream < 128U; ++stream) {
            std::vector<int> seed(static_cast<std::size_t>(k));
            std::iota(seed.begin(), seed.end(), 0);
            Rng kick_rng(make_stream_seed(0x51a7U, stream, static_cast<std::uint64_t>(k)));
            apply_elite_kick(inst, seed, kick_rng, 0.50);
            require_valid_seed(seed, k);
        }
    }

    bool duplicate_rejected = false;
    try {
        std::vector<int> bad{0, 0};
        Rng kick_rng(1);
        apply_elite_kick(inst, bad, kick_rng, 0.5);
    } catch (const std::invalid_argument&) {
        duplicate_rejected = true;
    }
    require(duplicate_rejected, "elite kick rejects duplicate input seeds");

    // Pin the original N=100, k=99 failure shape in both serial and parallel
    // restart waves over many deterministic streams.
    SolverOptions opt;
    opt.subset_restarts = 6;
    opt.subset_kick_restarts = 4;
    opt.kick_fraction = 0.50;
    opt.sa_iters = 0;
    opt.final_exhaustive_k = 0;
    opt.disable_two_opt = true;
    opt.disable_or_opt = true;
    opt.disable_subset_swap = true;
    opt.disable_pair_exchange = true;
    opt.disable_ruin_recreate = true;
    opt.disable_path_relink = true;
    opt.disable_smallp_seeds = true;
    opt.disable_highp_delete = true;
    for (const int restart_threads : {1, 2, 4}) {
        opt.restart_threads = restart_threads;
        for (std::uint64_t stream = 1; stream <= 16U; ++stream) {
            Rng solve_rng(stream);
            const SolveResult result = solve_subset(inst, inst.N - 1, solve_rng, opt);
            require(result.stats.kick_restarts == 4,
                    "near-full solve executes every scheduled kick");
            require(result.tour.k == inst.N - 1 && result.tour.check_invariants(),
                    "near-full kick waves return a valid unique tour");
        }
    }
}

ALDOUS_TEST(test_kick_restarts_mechanics) {
    // Scheduled elite-kick restarts: the last kick_n restarts seed from the
    // best of the independent phase. Checks (a) accounting: exactly kick_n
    // kicks execute; (b) validity of the final tour; (c) DETERMINISM across
    // restart_threads -- the elite snapshot is taken exactly once at the
    // phase boundary, so thread count must not change the result.
    Instance inst;
    inst.periodic = true;
    Rng gen(90210);
    inst.generate(240, gen);
    inst.build_knn(16, KnnBackend::GridExact);
    SolverOptions opt;
    opt.seed = 7;
    opt.subset_restarts = 6;
    opt.subset_kick_restarts = 4;
    opt.sa_iters = 0;
    opt.restart_threads = 1;
    opt.final_exhaustive_k = 0;
    opt.disable_two_opt = true;
    opt.disable_or_opt = true;
    opt.disable_subset_swap = true;
    opt.disable_pair_exchange = true;
    opt.disable_ruin_recreate = true;
    opt.disable_path_relink = true;
    Rng r1(static_cast<std::uint64_t>(opt.seed));
    SolveResult a = solve_subset(inst, 40, r1, opt, nullptr);
    require(a.stats.kick_restarts == 4, "exactly kick_n scheduled kicks ran");
    require(a.stats.subset_restarts == 6, "total restarts unchanged by kicks");
    require(static_cast<int>(a.tour.nodes.size()) == 40 && a.tour.check_invariants(),
            "kick solve returns a valid tour");
    const double got = cycle_length(inst, a.tour.nodes);
    require(std::abs(got - a.tour.length) < 1e-6, "kick solve length is honest");
    opt.restart_threads = 3;
    Rng r2(static_cast<std::uint64_t>(opt.seed));
    SolveResult b = solve_subset(inst, 40, r2, opt, nullptr);
    require(std::abs(a.tour.length - b.tour.length) < 1e-12 && a.tour.nodes == b.tour.nodes,
            "kick restarts are deterministic across restart_threads");
}

ALDOUS_TEST(test_restart_value_logging) {
    Instance inst;
    inst.periodic = true;
    Rng gen(1234);
    inst.generate(240, gen);
    inst.build_knn(16, KnnBackend::GridExact);
    SolverOptions opt;
    opt.seed = 11;
    opt.subset_restarts = 5;
    opt.sa_iters = 0;
    opt.final_exhaustive_k = 0;
    opt.disable_two_opt = true;
    opt.disable_or_opt = true;
    opt.disable_subset_swap = true;
    opt.disable_pair_exchange = true;
    opt.disable_ruin_recreate = true;
    opt.disable_path_relink = true;
    Rng r(static_cast<std::uint64_t>(opt.seed));
    SolveResult res = solve_subset(inst, 30, r, opt, nullptr);
    require(res.restarts.size() == 5U, "one typed record per restart");
    require(res.restarts.size() == res.stats.subset_restarts,
            "subset restart records and accounting agree");
    double lo = std::numeric_limits<double>::infinity();
    for (const RestartRecord& record : res.restarts) {
        require(std::isfinite(record.length) && record.length > 0.0,
                "restart values are finite and positive");
        require(is_valid_restart_kind_code(restart_kind_code(record.kind)),
                "every restart kind is defined by shared metadata");
        require(record.sweep == RestartSweep::Primary,
                "standalone solves label records as primary");
        require(record.promotion_stage == RestartPromotionStage::None
                    && record.sa_iterations == 0,
                "ordinary zero-SA records expose their stage and exact budget");
        require(std::isfinite(record.centroid_x) && std::isfinite(record.centroid_y)
                    && std::isfinite(record.radius) && record.radius >= 0.0,
                "restart geometry is finite");
        lo = std::min(lo, record.length);
    }
    // Post-restart stages (relinking, final polish) may improve further, so the
    // final length is at most the best restart's value.
    require(res.tour.length <= lo + 1e-9, "final length <= best logged restart value");
    require(res.best_restart >= 0
                && static_cast<std::size_t>(res.best_restart) < res.restarts.size(),
            "best_restart indexes the typed record population");
    require(std::abs(res.restarts[static_cast<std::size_t>(res.best_restart)].length - lo)
                <= kImprovementEps,
            "best_restart identifies the best recorded outcome");
}

ALDOUS_TEST(test_region_seeds) {
    // Region seeding replaces the pooled random seeds with fresh per-restart
    // local subsets. It is OFF by default (it measurably underperforms random
    // seeds -- see CHANGELOG); this pins that when enabled it is at least
    // correct and actually reaches the restart pool.
    Instance inst;
    inst.periodic = true;
    Rng gen(4242);
    inst.generate(600, gen);
    inst.build_knn(16, KnnBackend::GridExact);
    SolverOptions opt;
    opt.seed = 3;
    opt.subset_restarts = 7;  // > the p=0.05 seed budget, so fill runs
    opt.sa_iters = 0;
    opt.region_seeds = true;
    opt.region_dilation = 3.0;
    opt.final_exhaustive_k = 0;
    opt.disable_two_opt = true;
    opt.disable_or_opt = true;
    opt.disable_subset_swap = true;
    opt.disable_pair_exchange = true;
    opt.disable_ruin_recreate = true;
    opt.disable_path_relink = true;
    Rng r(static_cast<std::uint64_t>(opt.seed));
    SolveResult res = solve_subset(inst, 30, r, opt, nullptr);
    require(static_cast<int>(res.tour.nodes.size()) == 30 && res.tour.check_invariants(),
            "region-seeded solve returns a valid tour");
    require(std::abs(cycle_length(inst, res.tour.nodes) - res.tour.length) < 1e-6,
            "region-seeded solve reports an honest length");
    const auto region = std::find_if(
        res.restarts.begin(), res.restarts.end(),
        [](const RestartRecord& record) { return record.kind == RestartKind::Region; });
    require(region != res.restarts.end(), "region seeds actually enter the restart pool");
    require(res.stats.region_restarts ==
                static_cast<std::uint64_t>(count_restart_kind(res.restarts, RestartKind::Region)),
            "region restarts have a dedicated, exact counter");
    // Region seeds are local by construction: the subset must be far tighter
    // than the torus itself.
    require(region->radius < 0.25 * inst.side,
            "region-seeded restarts stay localized");
}

ALDOUS_TEST(test_dense_seeds_are_not_exploration_seeds) {
    // The regression this pins: dense_seed and random_subset both used to be
    // labelled Random, so "exploration seed" forced the exact O(k) insertion
    // scan onto dense draws. At k=2000 that scan returns the same search for a
    // dense seed while costing 25.9us/move against 4.05us -- ~58% of the wall
    // clock spent on a foregone conclusion. Dense fill must therefore emit
    // RestartKind::Dense, which must not be treated as globally spread.
    Instance inst;
    inst.periodic = true;
    Rng gen(4242);
    inst.generate(600, gen);
    inst.build_knn(12, KnnBackend::GridExact);
    SolverOptions opt;
    opt.subset_restarts = 7;  // > the p=0.05 seed budget, so fill runs
    opt.sa_iters = 0;
    opt.small_p_dense_fill = true;
    opt.restart_threads = 1;
    opt.final_exhaustive_k = 0;
    opt.disable_two_opt = true;
    opt.disable_or_opt = true;
    opt.disable_subset_swap = true;
    opt.disable_pair_exchange = true;
    opt.disable_ruin_recreate = true;
    opt.disable_path_relink = true;
    Rng srng(99);
    SolveResult res = solve_subset(inst, 30, srng, opt);  // p = 0.05 <= 0.08
    require(res.restarts.size() == 7U, "one typed record per restart");
    const long dense = count_restart_kind(res.restarts, RestartKind::Dense);
    const long spread = count_restart_kind(res.restarts, RestartKind::Random);
    require(dense > 0, "dense fill emits the dense kind");
    require(spread == 0, "dense fill emits no random-subset seeds at small p");
    require(res.stats.dense_restarts == static_cast<std::uint64_t>(dense),
            "dense restarts have a dedicated, exact counter");
    require(res.stats.random_restarts == static_cast<std::uint64_t>(spread),
            "dense restarts are not miscounted as random");

    // With dense fill OFF the pool must contain the spread seeds again, and
    // those are the ones the exact scan exists for.
    SolverOptions opt2 = opt;
    opt2.small_p_dense_fill = false;
    Rng srng2(99);
    SolveResult res2 = solve_subset(inst, 30, srng2, opt2);
    const long spread2 = count_restart_kind(res2.restarts, RestartKind::Random);
    require(spread2 > 0, "without dense fill the pool still draws random-subset seeds");
    require(res2.stats.random_restarts == static_cast<std::uint64_t>(spread2),
            "random restart accounting follows typed kinds");
}

ALDOUS_TEST(test_subset_index_matches_bruteforce) {
    // The index must return exactly the nearest CURRENT members -- under
    // periodic wrap, and after arbitrary add/remove churn. If it silently
    // returns the wrong neighbours, the SA still "works" (it just proposes bad
    // slots), so nothing downstream would fail loudly. Hence a direct test.
    for (const bool periodic : {false, true}) {
        Instance inst;
        inst.periodic = periodic;
        Rng gen(777);
        inst.generate(3000, gen);
        inst.build_knn(16, KnnBackend::GridExact);
        Tour tour;
        tour.init(inst.N);
        std::vector<int> seed = random_subset(inst.N, 120, gen);
        tour.set_tour(seed, inst);
        SubsetIndex index;
        index.build(inst, tour);
        std::vector<int> members(tour.nodes.begin(), tour.nodes.begin() + tour.k);

        for (int step = 0; step < 40; ++step) {
            const int m = 8;
            const int query = gen.randint(inst.N);
            std::vector<int> got;
            index.nearest(inst, query, m, -1, got);
            // brute force over the same membership
            std::vector<std::pair<double, int>> ref;
            ref.reserve(members.size());
            for (const int v : members) {
                if (v == query) { continue; }
                ref.emplace_back(inst.dist(query, v), v);
            }
            std::sort(ref.begin(), ref.end());
            const std::size_t want = std::min<std::size_t>(static_cast<std::size_t>(m), ref.size());
            require(got.size() == want, "index returns the requested neighbour count");
            for (std::size_t i = 0; i < want; ++i) {
                require(got[i] == ref[i].second,
                        "index returns exact nearest ids with deterministic ties");
                require(std::abs(inst.dist(query, got[i]) - ref[i].first) < 1e-9,
                        "index returns the true nearest current members");
            }
            // churn: swap one member out for a non-member
            const int drop_i = gen.randint(static_cast<int>(members.size()));
            const int drop = members[static_cast<std::size_t>(drop_i)];
            int addn = gen.randint(inst.N);
            int guard = 0;
            while (std::find(members.begin(), members.end(), addn) != members.end() && guard < 64) {
                addn = gen.randint(inst.N);
                ++guard;
            }
            if (addn == drop) { continue; }
            index.remove_member(inst, drop);
            index.add_member(inst, addn);
            members[static_cast<std::size_t>(drop_i)] = addn;
            require(index.size() == static_cast<int>(members.size()), "index member count tracks the subset");
        }
    }
}

ALDOUS_TEST(test_spatial_insertion_saturates_to_exact) {
    // With neighbours >= k every slot is gathered, so the spatial kernel must
    // agree with the exact scan exactly. This pins the kernel's semantics: it
    // is a RESTRICTION of the exact scan, never a different objective.
    Instance inst;
    inst.periodic = true;
    Rng gen(31337);
    inst.generate(4000, gen);
    inst.build_knn(24, KnnBackend::GridExact);
    Tour tour;
    tour.init(inst.N);
    std::vector<int> seed = random_subset(inst.N, 60, gen);
    tour.set_tour(seed, inst);
    SubsetIndex index;
    index.build(inst, tour);
    for (int trial = 0; trial < 60; ++trial) {
        const int ri = gen.randint(tour.k);
        int add = gen.randint(inst.N);
        if (tour.in_set[static_cast<std::size_t>(add)] != 0U) { continue; }
        const SwapInsertionMove exact = find_best_insert_after_remove(inst, tour, ri, add);
        const SwapInsertionMove sat =
            find_best_insert_after_remove_spatial(inst, tour, index, ri, add, tour.k + 1, 4);
        require(exact.valid == sat.valid, "saturated spatial kernel agrees on validity");
        if (exact.valid) {
            require(sat.post_pred == exact.post_pred && std::abs(sat.delta - exact.delta) < 1e-12,
                    "saturated spatial kernel reproduces the exact scan");
        }
        // A restricted (neighbours = 12) query must still report an HONEST delta:
        // apply it and confirm the tour length moves by exactly that much.
        const SwapInsertionMove restricted =
            find_best_insert_after_remove_spatial(inst, tour, index, ri, add, 12, 4);
        if (restricted.valid) {
            Tour probe = tour;
            const double before = probe.length;
            const int removed_node = probe.nodes[static_cast<std::size_t>(restricted.remove_pos)];
            probe.apply_swap_post_rem(restricted.remove_pos, restricted.post_pred,
                                      restricted.add_node, inst, restricted.delta);
            require(std::abs(cycle_length(inst, probe.nodes) - (before + restricted.delta)) < 1e-6,
                    "spatial move delta is honest against a recomputed tour length");
            require(probe.check_invariants(), "spatial move leaves a valid tour");
            require(removed_node != restricted.add_node, "spatial move actually exchanges a member");
        }
    }
}

ALDOUS_TEST(test_windowed_insertion_correctness) {
    // The windowed SA insertion must (a) report deltas that match reality when
    // the move is applied, and (b) reduce to the exact O(k) scan whenever the
    // window covers the whole tour. This pins the delta formula, the
    // post-removal predecessor indexing, and the slot-dedupe logic.
    Instance inst;
    inst.periodic = true;
    Rng gen(2718);
    inst.generate(4000, gen);
    inst.build_knn(24, KnnBackend::GridExact);
    Rng rng(31u);
    std::vector<int> subset;
    {
        std::vector<unsigned char> used(static_cast<std::size_t>(inst.N), 0U);
        while (static_cast<int>(subset.size()) < 80) {
            const int c = rng.randint(inst.N);
            if (used[static_cast<std::size_t>(c)] == 0U) { used[static_cast<std::size_t>(c)] = 1U; subset.push_back(c); }
        }
    }
    Tour tour;
    tour.init(inst.N);
    tour.set_tour(subset, inst);
    int applied = 0;
    for (int trial = 0; trial < 200; ++trial) {
        const int ri = rng.randint(tour.k);
        const int add = choose_swap_candidate(inst, tour, ri, rng);
        if (add < 0 || add >= inst.N || tour.in_set[static_cast<std::size_t>(add)] != 0U) { continue; }
        // (b) whole-tour window == exact scan, slot for slot
        const SwapInsertionMove exact = find_best_insert_after_remove(inst, tour, ri, add);
        const SwapInsertionMove wide = find_best_insert_after_remove_windowed(inst, tour, ri, add, tour.k);
        require(exact.valid == wide.valid, "whole-tour window matches exact validity");
        if (exact.valid) {
            require(std::abs(exact.delta - wide.delta) < 1e-9, "whole-tour window matches exact delta");
        }
        // (a) narrow-window move deltas are honest when applied
        const SwapInsertionMove narrow = find_best_insert_after_remove_windowed(inst, tour, ri, add, 6);
        if (!narrow.valid) { continue; }
        Tour probe = tour;
        probe.apply_swap_post_rem(narrow.remove_pos, narrow.post_pred, narrow.add_node, inst, narrow.delta);
        const double recomputed = cycle_length(inst, probe.nodes);
        require(std::abs(recomputed - (tour.length + narrow.delta)) < 1e-6,
                "windowed move delta matches the recomputed tour length");
        require(narrow.delta >= exact.delta - 1e-9,
                "windowed best cannot beat the exact best (subset of slots)");
        tour = probe;
        ++applied;
    }
    require(applied > 50, "windowed insertion test exercised enough applied moves");
}

ALDOUS_TEST(test_heldout_search_policy_resolution) {
    SolverOptions opt;
    opt.search_policy_preset = SearchPolicyPreset::LegacyBalanced;
    const ResolvedSubsetPolicy legacy =
        resolve_subset_policy(opt, 0.20, 400, true, false);
    require(legacy.restarts == 5 && legacy.racing_candidates == 0
                && legacy.sa_iterations < 0
                && legacy.sa_candidate_trials == 1,
            "legacy-balanced preserves the established automatic controller");

    opt.search_policy_preset = SearchPolicyPreset::HeldoutBalanced;
    const ResolvedSubsetPolicy below_p =
        resolve_subset_policy(opt, 0.01, 40, true, false);
    require(!below_p.heldout_sa_policy_applied,
            "held-out SA policy does not extrapolate below its measured p range");
    const ResolvedSubsetPolicy tiny_k =
        resolve_subset_policy(opt, 0.02, 20, true, false);
    require(!tiny_k.heldout_sa_policy_applied,
            "held-out SA policy retains the measured tiny-cardinality guard");
    const ResolvedSubsetPolicy balanced =
        resolve_subset_policy(opt, 0.20, 400, true, false);
    require(balanced.heldout_sa_policy_applied
                && balanced.sa_iterations == 20000
                && balanced.sa_candidate_trials == 4
                && std::abs(balanced.sa_multiple_try_random_probability - 0.1) < 1e-15,
            "balanced held-out policy resolves to the validated four-trial budget");
    const ResolvedSubsetPolicy balanced_edge =
        resolve_subset_policy(opt, 0.35, 700, true, false);
    require(balanced_edge.heldout_sa_policy_applied,
            "balanced policy includes its validated p=0.35 boundary");
    const ResolvedSubsetPolicy balanced_above =
        resolve_subset_policy(opt, 0.40, 800, true, false);
    require(!balanced_above.heldout_sa_policy_applied,
            "balanced policy stops above p=0.35 in both geometries");

    opt.search_policy_preset = SearchPolicyPreset::HeldoutQuality;
    const ResolvedSubsetPolicy quality =
        resolve_subset_policy(opt, 0.40, 800, true, false);
    require(quality.heldout_sa_policy_applied
                && quality.sa_iterations == 30000
                && quality.sa_candidate_trials == 4,
            "quality policy uses the deeper held-out multiple-candidate budget");
    const ResolvedSubsetPolicy quality_periodic_edge =
        resolve_subset_policy(opt, 0.50, 1000, true, false);
    require(quality_periodic_edge.heldout_sa_policy_applied,
            "quality policy includes the validated periodic p=0.50 boundary");
    const ResolvedSubsetPolicy quality_open_above =
        resolve_subset_policy(opt, 0.40, 800, false, false);
    require(!quality_open_above.heldout_sa_policy_applied,
            "quality open-square policy stops above p=0.35");
    const ResolvedSubsetPolicy high =
        resolve_subset_policy(opt, 0.80, 1600, true, false);
    require(!high.heldout_sa_policy_applied,
            "quality policy does not extrapolate into the high-p regime");

    SolverOptions explicit_sa = opt;
    explicit_sa.sa_iters = 30000;
    const ResolvedSubsetPolicy explicit_result =
        resolve_subset_policy(explicit_sa, 0.20, 400, true, false);
    require(!explicit_result.heldout_sa_policy_applied
                && explicit_result.sa_candidate_trials == 1,
            "explicit SA controls take precedence over policy presets");

    SolverOptions explicit_temperature = opt;
    explicit_temperature.sa_t0 = 2.0;
    const ResolvedSubsetPolicy explicit_temperature_result =
        resolve_subset_policy(explicit_temperature, 0.20, 400, true, false);
    require(!explicit_temperature_result.heldout_sa_policy_applied,
            "explicit fixed-temperature controls take precedence over policy presets");

    SolverOptions automatic_temperature = opt;
    automatic_temperature.sa_auto_temperature = true;
    const ResolvedSubsetPolicy calibrated =
        resolve_subset_policy(automatic_temperature, 0.20, 400, true, false);
    require(!calibrated.heldout_sa_policy_applied,
            "automatic temperature experiments are never silently combined with a preset");

    const ResolvedSubsetPolicy continuation =
        resolve_subset_policy(opt, 0.20, 400, true, true);
    require(!continuation.heldout_sa_policy_applied
                && continuation.racing_candidates == 0,
            "continuation-only solves preserve their literal controller");

    SolverOptions balanced_tsp_options;
    balanced_tsp_options.search_policy_preset = SearchPolicyPreset::HeldoutBalanced;
    const ResolvedTspPolicy balanced_tsp =
        resolve_tsp_policy(balanced_tsp_options, true);
    require(balanced_tsp.candidate_starts
                    == balanced_tsp_options.tsp_candidate_starts
                && balanced_tsp.promoted_restarts
                    == balanced_tsp_options.tsp_restarts
                && balanced_tsp.ils_iterations == balanced_tsp_options.tsp_ils,
            "balanced held-out policy preserves the established full-TSP controller");

    const ResolvedTspPolicy quality_tsp = resolve_tsp_policy(opt, true);
    require(quality_tsp.candidate_starts == 16
                && quality_tsp.promoted_restarts == 4
                && quality_tsp.ils_iterations == 450,
            "quality policy exposes the held-out full-TSP screening controller");

    SolverOptions explicit_tsp = opt;
    explicit_tsp.tsp_candidate_starts = 20;
    const ResolvedTspPolicy explicit_tsp_result = resolve_tsp_policy(explicit_tsp, true);
    require(explicit_tsp_result.candidate_starts == 20
                && explicit_tsp_result.promoted_restarts == explicit_tsp.tsp_restarts
                && explicit_tsp_result.ils_iterations == explicit_tsp.tsp_ils,
            "explicit TSP controls take precedence over policy presets");
}

ALDOUS_TEST(test_restarts_flag_is_authoritative) {
    // Regression: the restart count used to be max(seed_pool.size(), restarts),
    // and the seed builders inject up to 8 small-p seeds at p <= 0.08 -- so
    // `--restarts 1` and `--restarts 8` ran identically and the search budget was
    // not controllable at all. That is fatal for a convergence study, where the
    // whole point is to vary the budget and watch for a plateau.
    Instance inst;
    inst.periodic = true;
    Rng gen(31337);
    inst.generate(300, gen);
    inst.build_knn(12, KnnBackend::GridExact);
    const int k = 15;  // p = 0.05 <= 0.08, so the small-p seed pool is populated
    for (const int restarts : {1, 2, 8}) {
        SolverOptions opt;
        opt.subset_restarts = restarts;
        opt.sa_iters = 0;
        opt.final_exhaustive_k = 0;
        opt.disable_two_opt = true;
        opt.disable_or_opt = true;
        opt.disable_subset_swap = true;
        opt.disable_pair_exchange = true;
        opt.disable_ruin_recreate = true;
        opt.disable_path_relink = true;
        Rng rng(99u);
        const SolveResult res = solve_subset(inst, k, rng, opt, nullptr);
        require(res.stats.subset_restarts == static_cast<std::uint64_t>(restarts),
                "the solver runs exactly the requested number of subset restarts");
    }

    // The AUTO default broadens the exploration population under staged search:
    // 12 restarts at p <= 0.08 and 5 above, while reserving expensive strong
    // neighborhoods for promoted finalists. Disabling staged search restores
    // the historical 8/3 population.
    {
        SolverOptions opt;
        opt.sa_iters = 0;
        opt.final_exhaustive_k = 0;
        opt.disable_two_opt = true;
        opt.disable_or_opt = true;
        opt.disable_subset_swap = true;
        opt.disable_pair_exchange = true;
        opt.disable_ruin_recreate = true;
        opt.disable_path_relink = true;
        require(opt.subset_restarts == -1, "default restart count is auto");
        Rng rng(99u);
        const SolveResult smallp = solve_subset(inst, k, rng, opt, nullptr);  // p = 0.05
        require(smallp.stats.subset_restarts == 12,
                "staged auto default explores 12 restarts at p <= 0.08");
        Rng rng2(99u);
        const SolveResult largep = solve_subset(inst, 90, rng2, opt, nullptr);  // p = 0.30
        require(largep.stats.subset_restarts == 5
                    && largep.stats.racing_pilot_restarts == 0,
                "legacy-balanced preserves the former staged automatic population");

        opt.search_policy_preset = SearchPolicyPreset::HeldoutBalanced;
        Rng heldout_policy_rng(99u);
        const SolveResult heldout_policy =
            solve_subset(inst, 90, heldout_policy_rng, opt, nullptr);
        require(heldout_policy.stats.subset_restarts == 5
                    && heldout_policy.stats.racing_pilot_restarts == 0,
                "held-out SA policy preserves the automatic restart population");

        opt.staged_search = false;
        Rng rng3(99u);
        const SolveResult legacy_smallp = solve_subset(inst, k, rng3, opt, nullptr);
        require(legacy_smallp.stats.subset_restarts == 8,
                "disabling staged search restores 8 small-p restarts");
        Rng rng4(99u);
        const SolveResult legacy_largep = solve_subset(inst, 90, rng4, opt, nullptr);
        require(legacy_largep.stats.subset_restarts == 3,
                "disabling staged search restores 3 larger-p restarts");
    }
}

ALDOUS_TEST(test_elite_anytime_restarts) {
    Rng rng(1717);
    Instance inst;
    inst.generate(120, rng);
    inst.build_knn(12, KnnBackend::GridExact);
    SolverOptions opt;
    opt.seed = 1717;
    opt.subset_restarts = 2;
    opt.sa_iters = 0;
    opt.final_exhaustive_k = 0;
    opt.disable_two_opt = true;
    opt.disable_or_opt = true;
    opt.disable_subset_swap = true;
    opt.disable_pair_exchange = true;
    opt.disable_ruin_recreate = true;
    opt.disable_path_relink = true;

    // Without a time budget, no anytime restarts run, so elite restarts must
    // stay at zero and the deterministic path is untouched.
    Rng r0(55);
    const auto t0 = Clock::now();
    SolveResult base = solve_subset(inst, 30, r0, opt);
    const double base_secs = std::chrono::duration<double>(Clock::now() - t0).count();
    require(base.stats.elite_restarts == 0, "no elite restarts without a time budget");

    // With a time budget, anytime restarts run and some are elite-seeded ILS
    // restarts; the result must be a valid size-k subset. The budget is derived
    // from the MEASURED baseline time rather than a wall-clock constant: a fixed
    // 0.4s assumed a fast machine, and under sanitizers (or on a loaded laptop)
    // the scheduled restarts alone exceeded it, so no anytime wave ever launched
    // and the test failed for reasons that had nothing to do with the logic.
    // Scheduled work ~= base_secs, so 6x that leaves ample anytime headroom on
    // any machine.
    opt.time_budget_per_p = std::max(0.05, 6.0 * base_secs);
    Rng r1(55);
    SolveResult budgeted = solve_subset(inst, 30, r1, opt);
    require(budgeted.stats.subset_restarts > base.stats.subset_restarts,
            "time budget launches additional restarts");
    require(budgeted.stats.elite_restarts > 0, "time-budget mode runs elite ILS restarts");
    require(budgeted.tour.check_invariants() && budgeted.tour.k == 30,
            "elite-ILS solve returns a valid size-k subset");

    // Disabling elite restarts keeps anytime mode running cold restarts only.
    opt.disable_elite_restarts = true;
    Rng r2(55);
    SolveResult cold = solve_subset(inst, 30, r2, opt);
    require(cold.stats.elite_restarts == 0, "disable flag suppresses elite restarts");
}

ALDOUS_TEST(test_time_budget_adds_restarts) {
    Rng rng(6060);
    Instance inst;
    inst.generate(120, rng);
    inst.build_knn(16, KnnBackend::GridExact);
    SolverOptions opt;
    opt.seed = 6060;
    opt.subset_restarts = 2;
    opt.sa_iters = 0;
    opt.disable_smallp_seeds = true;
    opt.final_exhaustive_k = 0;
    opt.disable_two_opt = true;
    opt.disable_or_opt = true;
    opt.disable_subset_swap = true;
    opt.disable_pair_exchange = true;
    opt.disable_ruin_recreate = true;
    opt.disable_path_relink = true;

    Rng solve_rng(6061);
    SolveResult base = solve_subset(inst, 30, solve_rng, opt);
    require(base.stats.subset_restarts == 2, "without a budget the configured restarts run");

    opt.time_budget_per_p = 0.02;
    Rng solve_rng2(6061);
    SolveResult budgeted = solve_subset(inst, 30, solve_rng2, opt);
    require(budgeted.stats.subset_restarts > base.stats.subset_restarts,
            "a positive time budget launches additional restarts");
    require(budgeted.tour.length <= base.tour.length + 1e-9,
            "budgeted solve is never worse than the unbudgeted solve");
}

ALDOUS_TEST(test_two_opt_candidate_table_property) {
    Rng rng(8282);
    Instance inst;
    inst.generate(200, rng);
    inst.build_knn(16, KnnBackend::GridExact);
    for (int trial = 0; trial < 8; ++trial) {
        const int k = 20 + rng.randint(60);
        std::vector<int> nodes = random_subset(inst.N, k, rng);
        Tour tour;
        tour.init(inst.N);
        tour.set_tour(nodes, inst);
        const double before = tour.length;
        const SubsetCandidateTable* table = maybe_subset_candidates(inst, tour);
        require(table != nullptr, "subset candidate table is built for k < N");
        two_opt_candidate_descent(tour, inst, 200, 0, nullptr, table);
        require(tour.check_invariants(), "tour invariants hold after table-driven 2-opt");
        require(tour.length <= before + 1e-9, "table-driven 2-opt never increases length");
        const double recomputed = cycle_length(inst, tour.nodes);
        require(std::abs(tour.length - recomputed) <= 1e-6 * (1.0 + recomputed),
                "incremental length matches recomputation after table-driven 2-opt");
    }
}


} // namespace
