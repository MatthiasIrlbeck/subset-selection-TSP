#include "test_common.hpp"

namespace {

ALDOUS_TEST(test_tour_invariants) {
    Instance inst;
    inst.set_points({{0,0}, {1,0}, {1,1}, {0,1}, {2,1}});
    inst.build_knn(4);
    Tour tour;
    tour.init(inst.N);
    tour.set_tour({0, 1, 2, 3}, inst);
    require(tour.check_invariants(), "tour invariants after set_tour");
    require(std::fabs(tour.length - 4.0) < 1e-9, "square tour length");
}

ALDOUS_TEST(test_tour_incremental_mutation_property) {
    Rng rng(777);
    Instance inst;
    inst.generate(60, rng);
    inst.build_knn(16);
    std::vector<int> base(60);
    std::iota(base.begin(), base.end(), 0);
    rng.partial_shuffle(base.begin(), base.end(), 20);
    base.resize(20);
    Tour tour;
    tour.init(inst.N);
    tour.set_tour(base, inst);
    for (int step = 0; step < 120; ++step) {
        const int op = rng.randint(3);
        if (op == 0 && tour.k >= 4) {
            int i = rng.randint(tour.k - 2);
            int j = i + 2 + rng.randint(tour.k - i - 2);
            if (i == 0 && j == tour.k - 1) {
                continue;
            }
            std::vector<int> expected = tour.nodes;
            std::reverse(expected.begin() + i + 1, expected.begin() + j + 1);
            const double delta = cycle_length(inst, expected) - tour.length;
            tour.apply_two_opt(i, j, inst, delta);
        } else if (op == 1 && tour.k < inst.N) {
            std::vector<int> outside;
            for (int v = 0; v < inst.N; ++v) {
                if (tour.in_set[static_cast<std::size_t>(v)] == 0U) { outside.push_back(v); }
            }
            if (outside.empty()) { continue; }
            const int remove_pos = rng.randint(tour.k);
            const int post_pred = rng.randint(tour.k - 1);
            const int add = outside[static_cast<std::size_t>(rng.randint(static_cast<int>(outside.size())))];
            std::vector<int> expected = tour.nodes;
            expected.erase(expected.begin() + remove_pos);
            expected.insert(expected.begin() + post_pred + 1, add);
            const double delta = cycle_length(inst, expected) - tour.length;
            tour.apply_swap_post_rem(remove_pos, post_pred, add, inst, delta);
        } else if (tour.k >= 5) {
            const int remove_pos = rng.randint(tour.k);
            const int post_pred = rng.randint(tour.k - 1);
            std::vector<int> expected = tour.nodes;
            const int node = expected[static_cast<std::size_t>(remove_pos)];
            expected.erase(expected.begin() + remove_pos);
            expected.insert(expected.begin() + post_pred + 1, node);
            const double delta = cycle_length(inst, expected) - tour.length;
            tour.apply_move_post_rem(remove_pos, post_pred, inst, delta);
        }
        require(tour.check_invariants(), "tour invariants after incremental mutation");
        require(std::fabs(tour.length - cycle_length(inst, tour.nodes)) < 1e-8, "incremental tour length matches recompute");
    }
}

ALDOUS_TEST(test_exact_small_tsp) {
    Instance inst;
    inst.set_points({{0,0}, {1,0}, {1,1}, {0,1}});
    inst.build_knn(3);
    std::vector<int> cycle;
    double len = 0.0;
    require(exact_small_tsp_cycle(inst, {0, 1, 2, 3}, cycle, len), "exact small TSP succeeds");
    require(std::fabs(len - 4.0) < 1e-9, "exact square length");
}

ALDOUS_TEST(test_exact_small_tsp_randomized) {
    for (int rep = 0; rep < 12; ++rep) {
        Rng rng(static_cast<std::uint64_t>(300 + rep));
        Instance inst;
        inst.generate(9, rng);
        inst.build_knn(8);
        std::vector<int> nodes = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<int> cycle;
        double exact = 0.0;
        require(exact_small_tsp_cycle(inst, nodes, cycle, exact), "random exact small TSP succeeds");
        std::vector<int> perm = {1, 2, 3, 4, 5, 6, 7};
        double brute = std::numeric_limits<double>::infinity();
        do {
            std::vector<int> cand = {0};
            cand.insert(cand.end(), perm.begin(), perm.end());
            brute = std::min(brute, cycle_length(inst, cand));
        } while (std::next_permutation(perm.begin(), perm.end()));
        require(std::fabs(exact - brute) < 1e-9, "exact small TSP matches brute-force permutation search");
    }
}


ExactSubsetSolution brute_force_exact_subset(const Instance& inst, const int k) {
    ExactSubsetSolution best;
    best.n = inst.N;
    best.k = k;
    if (k == 0) {
        best.solved = true;
        best.proven_optimal = true;
        best.length = 0.0;
        return best;
    }
    std::vector<int> combination(static_cast<std::size_t>(k));
    std::iota(combination.begin(), combination.end(), 0);
    for (;;) {
        std::vector<int> cycle;
        double length = 0.0;
        require(exact_small_tsp_cycle(inst, combination, cycle, length),
                "brute-force subset cycle remains within exact small-TSP limit");
        std::vector<int> selected = canonical_set_key(cycle);
        const std::vector<int> incumbent = canonical_set_key(best.cycle);
        if (!best.solved || length < best.length
            || (length == best.length && selected < incumbent)) {
            best.solved = true;
            best.proven_optimal = true;
            best.length = length;
            best.cycle = std::move(cycle);
        }
        int position = k - 1;
        while (position >= 0
               && combination[static_cast<std::size_t>(position)]
                      == inst.N - k + position) {
            --position;
        }
        if (position < 0) {
            break;
        }
        ++combination[static_cast<std::size_t>(position)];
        for (int next = position + 1; next < k; ++next) {
            combination[static_cast<std::size_t>(next)] =
                combination[static_cast<std::size_t>(next - 1)] + 1;
        }
    }
    return best;
}

ALDOUS_TEST(test_exact_subset_oracle) {
    for (int periodic = 0; periodic < 2; ++periodic) {
        for (int n = 3; n <= 9; ++n) {
            Rng rng(static_cast<std::uint64_t>(91000 + periodic * 100 + n));
            Instance inst;
            inst.periodic = periodic != 0;
            inst.generate(n, rng);
            for (int k = 0; k <= n; ++k) {
                const ExactSubsetSolution exact = exact_subset_cycle(inst, k);
                const ExactSubsetSolution brute = brute_force_exact_subset(inst, k);
                require(exact.solved && exact.proven_optimal,
                        "exact subset oracle proves supported instances");
                require(static_cast<int>(exact.cycle.size()) == k,
                        "exact subset oracle returns the requested cardinality");
                require(std::fabs(exact.length - brute.length)
                            <= 1e-10 * (1.0 + brute.length),
                        "exact subset oracle matches exhaustive subset enumeration");
                require(std::fabs(cycle_length(inst, exact.cycle) - exact.length)
                            <= 1e-10 * (1.0 + exact.length),
                        "exact subset oracle cycle length is self-consistent");
                std::vector<int> selected = exact.cycle;
                std::sort(selected.begin(), selected.end());
                require(std::adjacent_find(selected.begin(), selected.end())
                            == selected.end(),
                        "exact subset oracle cycle contains unique nodes");
                require(selected.empty()
                            || (selected.front() >= 0 && selected.back() < inst.N),
                        "exact subset oracle cycle nodes stay in range");
                const ExactSubsetSolution repeated = exact_subset_cycle(inst, k);
                require(repeated.cycle == exact.cycle
                            && repeated.length == exact.length
                            && repeated.states == exact.states
                            && repeated.transitions == exact.transitions,
                        "exact subset oracle tie resolution is deterministic");
            }
        }
    }

    Instance tied;
    tied.set_points({{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                     {0.0, 0.0}, {0.0, 0.0}});
    const ExactSubsetSolution tie = exact_subset_cycle(tied, 3);
    require(tie.length == 0.0
                && canonical_set_key(tie.cycle) == std::vector<int>({0, 1, 2}),
            "exact subset oracle uses the lowest membership mask on an exact tie");

    Rng large_rng(123);
    Instance unsupported;
    unsupported.generate(kExactSubsetHardLimit + 1, large_rng);
    const ExactSubsetSolution large = exact_subset_cycle(unsupported, 4);
    require(!large.solved && !large.proven_optimal && large.cycle.empty(),
            "exact subset oracle refuses instances above its hard safety limit");

    bool rejected = false;
    try {
        (void)exact_subset_cycle(tied, tied.N + 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "exact subset oracle rejects invalid cardinalities");
}

ALDOUS_TEST(test_exact_subset_solver_integration) {
    Rng point_rng(44001);
    Instance inst;
    inst.periodic = true;
    inst.generate(10, point_rng);
    inst.build_knn(9, KnnBackend::GridExact);
    const ExactSubsetSolution reference = exact_subset_cycle(inst, 5);

    SolverOptions options;
    options.exact_subset_max_n = 10;
    options.subset_restarts = 7;
    options.sa_iters = 5000;
    Rng solve_rng(55001);
    Rng untouched_rng(55001);
    const SolveResult subset = solve_subset(inst, 5, solve_rng, options);
    require(subset.exact_optimal, "subset solver exposes the global exact proof");
    require(subset.restarts.empty() && subset.best_restart == -1,
            "exact subset solve does not fabricate heuristic restart diagnostics");
    require(subset.tour.nodes == reference.cycle
                && subset.tour.length == reference.length,
            "subset solver returns the public exact-oracle solution");
    require(subset.stats.exact_subset_calls == 1
                && subset.stats.exact_subset_solved == 1
                && subset.stats.exact_subset_states == reference.states
                && subset.stats.exact_subset_transitions == reference.transitions,
            "subset solver reports exact-oracle work");
    require(subset.stats.phases.exact_subset_seconds >= 0.0,
            "subset solver reports exact-oracle phase time");
    require(solve_rng.next_u64() == untouched_rng.next_u64(),
            "exact subset solve does not consume the caller RNG stream");

    Rng tsp_rng(66001);
    const SolveResult tsp = solve_tsp(inst, tsp_rng, options);
    const ExactSubsetSolution tsp_reference = exact_subset_cycle(inst, inst.N);
    require(tsp.exact_optimal && tsp.restarts.empty(),
            "full TSP uses the exact subset oracle as the k=N special case");
    require(tsp.tour.nodes == tsp_reference.cycle
                && tsp.tour.length == tsp_reference.length,
            "exact full TSP matches the global oracle");

    SolverOptions disabled = options;
    disabled.exact_subset_max_n = 0;
    disabled.subset_restarts = 1;
    disabled.sa_iters = 0;
    disabled.disable_pair_exchange = true;
    disabled.disable_ruin_recreate = true;
    disabled.disable_ejection_chain = true;
    disabled.disable_path_relink = true;
    Rng heuristic_rng(77001);
    const SolveResult heuristic = solve_subset(inst, 5, heuristic_rng, disabled);
    require(!heuristic.exact_optimal && heuristic.stats.exact_subset_calls == 0,
            "disabled exact oracle leaves heuristic proof status false");

    bool rejected = false;
    try {
        SolverOptions invalid = options;
        invalid.exact_subset_max_n = kExactSubsetHardLimit + 1;
        Rng invalid_rng(1);
        (void)solve_subset(inst, 5, invalid_rng, invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "solver rejects exact-oracle thresholds above the hard cap");

    RunOptions run;
    run.N = 9;
    run.instances = 2;
    run.threads = 1;
    run.p_values = {0.56, 1.0};
    run.include_instance_rows = true;
    run.second_sweep = true;
    run.solver.exact_subset_max_n = 9;
    run.solver.knn_k = 8;
    const ResultsDocument doc = ExperimentRunner(run).run();
    require(doc.stats.exact_subset_calls == 4
                && doc.stats.exact_subset_solved == 4,
            "exact experiment performs one proof per instance/p row even with the second sweep");
    require(doc.summary.at(p_value_key(0.56)).exact_optimal_instances == 2
                && doc.summary.at(p_value_key(1.0)).exact_optimal_instances == 2,
            "experiment summary counts global exact proofs per p");
    require(doc.instance_rows.size() == 2U,
            "exact experiment retains requested instance rows");
    for (const InstanceResultRow& row : doc.instance_rows) {
        require(row.p_results.size() == 2U
                    && row.p_results[0].exact_optimal
                    && row.p_results[1].exact_optimal,
                "instance p rows retain exact proof status");
    }
    const std::string json = results_to_json(doc);
    require(json.find("\"exact_subset_max_n\": 9") != std::string::npos
                && json.find("\"exact_optimal\": true") != std::string::npos
                && json.find("\"exact_optimal_instances\": 2") != std::string::npos,
            "JSON exposes exact-oracle configuration and proof status");
}

ALDOUS_TEST(test_two_opt_shorter_side_reversal) {
    // apply_two_opt reverses whichever arc is shorter. Verify that for edge
    // pairs forcing each branch (inner-shorter and outer-shorter), the result
    // is a valid tour whose length matches an independent recomputation and
    // whose edge set equals the reference 2-opt (remove (a,b),(c,d); add
    // (a,c),(b,d)).
    Rng rng(24601);
    Instance inst;
    inst.generate(400, rng);
    inst.build_knn(10, KnnBackend::GridExact);
    for (int trial = 0; trial < 40; ++trial) {
        const int k = 20 + rng.randint(120);
        std::vector<int> nodes = random_subset(inst.N, k, rng);
        Tour tour;
        tour.init(inst.N);
        tour.set_tour(nodes, inst);
        tour.ensure_edges(inst);
        // Pick two distinct edges i<j with j-i>=2 and not the wraparound pair.
        int i = rng.randint(k);
        int j = rng.randint(k);
        if (i > j) { std::swap(i, j); }
        if (j - i < 2 || (i == 0 && j == k - 1)) { continue; }
        const int a = tour.nodes[static_cast<std::size_t>(i)];
        const int b = tour.nodes[static_cast<std::size_t>(i + 1)];
        const int c = tour.nodes[static_cast<std::size_t>(j)];
        const int d = tour.nodes[static_cast<std::size_t>((j + 1 == k) ? 0 : (j + 1))];
        const double delta = inst.dist(a, c) + inst.dist(b, d)
            - tour.edge_len[static_cast<std::size_t>(i)] - tour.edge_len[static_cast<std::size_t>(j)];
        const double before = tour.length;
        tour.apply_two_opt(i, j, inst, delta);
        require(tour.check_invariants(), "valid tour after shorter-side 2-opt");
        const double recomputed = cycle_length(inst, tour.nodes);
        require(std::abs(tour.length - recomputed) <= 1e-6 * (1.0 + recomputed),
                "incremental length matches recomputation after shorter-side 2-opt");
        require(std::abs(tour.length - (before + delta)) <= 1e-6 * (1.0 + std::abs(before)),
                "length delta is applied exactly");
        // The two new edges (a,c) and (b,d) must be present in the tour.
        const int pa = tour.pos[static_cast<std::size_t>(a)];
        const int pc = tour.pos[static_cast<std::size_t>(c)];
        const bool ac_adj = (tour.nodes[static_cast<std::size_t>((pa + 1) % k)] == c)
                         || (tour.nodes[static_cast<std::size_t>((pa + k - 1) % k)] == c);
        const int pb = tour.pos[static_cast<std::size_t>(b)];
        const bool bd_adj = (tour.nodes[static_cast<std::size_t>((pb + 1) % k)] == d)
                         || (tour.nodes[static_cast<std::size_t>((pb + k - 1) % k)] == d);
        (void)pc;
        require(ac_adj && bd_adj, "the two new 2-opt edges are present after reversal");
    }
}

ALDOUS_TEST(test_two_opt_property) {
    Instance inst;
    inst.set_points({{0,0}, {1,0}, {1,1}, {0,1}});
    inst.build_knn(3);
    Tour tour;
    tour.init(inst.N);
    tour.set_tour({0, 2, 1, 3}, inst);
    const double before = tour.length;
    SearchStats stats;
    const int improvements = two_opt_descent(tour, inst, 10, &stats);
    require(improvements > 0, "two-opt finds crossing improvement");
    require(tour.length < before, "two-opt reduces length");
    require(std::fabs(tour.length - 4.0) < 1e-9, "two-opt reaches square optimum");
    require(tour.check_invariants(), "tour invariants after two-opt");
}


ALDOUS_TEST(test_incremental_tour_mutation_property) {
    Rng rng(777);
    Instance inst;
    inst.generate(80, rng);
    inst.build_knn(20, KnnBackend::GridExact);
    std::vector<int> nodes;
    for (int i = 0; i < 30; ++i) { nodes.push_back(i); }
    Tour tour;
    tour.init(inst.N);
    tour.set_tour(nodes, inst);
    for (int iter = 0; iter < 80; ++iter) {
        if ((iter % 2) == 0) {
            int i = rng.randint(tour.k - 3);
            int j = i + 2 + rng.randint(tour.k - i - 2);
            if (i == 0 && j == tour.k - 1) { j = tour.k - 2; }
            const int a = tour.nodes[static_cast<std::size_t>(i)];
            const int b = tour.nodes[static_cast<std::size_t>(i + 1)];
            const int c = tour.nodes[static_cast<std::size_t>(j)];
            const int d = tour.nodes[static_cast<std::size_t>((j + 1 == tour.k) ? 0 : (j + 1))];
            const double before = tour.length;
            const double delta = inst.dist(a, c) + inst.dist(b, d) - tour.edge_len[static_cast<std::size_t>(i)] - tour.edge_len[static_cast<std::size_t>(j)];
            tour.apply_two_opt(i, j, inst, delta);
            require(tour.check_invariants(), "incremental two-opt preserves invariants");
            require(std::fabs(tour.length - (before + delta)) < 1e-8, "incremental two-opt updates length by delta");
            require(std::fabs(tour.length - cycle_length(inst, tour.nodes)) < 1e-8, "incremental two-opt length matches recompute");
        } else {
            int remove_pos = rng.randint(tour.k);
            int add = -1;
            for (int tries = 0; tries < 200; ++tries) {
                const int candidate = rng.randint(inst.N);
                if (tour.in_set[static_cast<std::size_t>(candidate)] == 0U) {
                    add = candidate;
                    break;
                }
            }
            require(add >= 0, "found add node outside tour");
            std::vector<int> cand;
            cand.reserve(tour.nodes.size());
            for (int i = 0; i < tour.k; ++i) {
                if (i != remove_pos) { cand.push_back(tour.nodes[static_cast<std::size_t>(i)]); }
            }
            const int post_pred = rng.randint(static_cast<int>(cand.size()));
            cand.insert(cand.begin() + post_pred + 1, add);
            const double before = tour.length;
            const double target = cycle_length(inst, cand);
            tour.apply_swap_post_rem(remove_pos, post_pred, add, inst, target - before);
            require(tour.check_invariants(), "incremental subset swap preserves invariants");
            require(tour.nodes == cand, "incremental subset swap creates expected node order");
            require(std::fabs(tour.length - target) < 1e-8, "incremental subset swap length matches recompute");
        }
    }
}

ALDOUS_TEST(test_elite_collision_safe_key) {
    ElitePool pool(4, EliteMode::Set);
    pool.try_add({3, 1, 2}, 10.0);
    pool.try_add({2, 3, 1}, 9.0);
    pool.try_add({4, 5, 6}, 8.0);
    require(pool.entries().size() == 2U, "ElitePool deduplicates same set by canonical key");
    bool found_improved = false;
    for (const EliteEntry& e : pool.entries()) {
        if (e.canonical_key == std::vector<int>({1,2,3}) && std::fabs(e.length - 9.0) < 1e-9) {
            found_improved = true;
        }
    }
    require(found_improved, "ElitePool keeps improved duplicate");

    ElitePool diverse(2, EliteMode::Set, 2, 0.5, 1.0);
    diverse.try_add({0, 1, 2, 3}, 10.0);
    diverse.try_add({0, 1, 2, 4}, 11.0);
    diverse.try_add({4, 5, 6, 7}, 12.0);
    diverse.try_add({0, 1, 2, 5}, 13.0);
    diverse.try_add({8, 9, 10, 11}, 14.0);
    require(diverse.entries().size() == 4U,
            "supplemental diversity slots extend rather than replace the quality archive");
    require(diverse.entries()[0].canonical_key == std::vector<int>({0, 1, 2, 3})
                && diverse.entries()[1].canonical_key == std::vector<int>({0, 1, 2, 4}),
            "diversity pruning protects the complete legacy length-ranked capacity");
    require(diverse.diversity_candidates() == 5
                && diverse.diversity_retained() == 4
                && diverse.diversity_rejected() == 1,
            "diversity archive reports retained and rejected candidates");
    const auto relink_near = diverse.export_relink_nodes(2, 1);
    require(relink_near.size() == 2U,
            "relink export preserves the ordinary quality-ranked prefix");
    const auto relink_wide = diverse.export_relink_nodes(2, 4);
    require(relink_wide.size() == 4U,
            "relink export appends supplemental diverse basins when feasible");
}


} // namespace
