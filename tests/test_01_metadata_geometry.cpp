#include "test_common.hpp"
#include "generated_options.hpp"

namespace {

ALDOUS_TEST(test_generated_defaults_and_presets) {
    RunOptions library_defaults;
    const std::vector<double> expected_grid = {
        0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20,
        0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00,
    };
    require(library_defaults.p_values == expected_grid,
            "the metadata-generated library probability grid is authoritative");
    require(library_defaults.solver.restart_threads == 1,
            "the metadata-generated library restart-thread default stays explicit");

    RunOptions cli_defaults;
    apply_generated_cli_defaults(cli_defaults);
    require(cli_defaults.solver.restart_threads == 0,
            "the metadata-generated CLI default selects automatic restart threading");

    apply_generated_quick_preset(cli_defaults);
    require(cli_defaults.N == 48 && cli_defaults.instances == 2,
            "the generated quick preset applies campaign-size defaults");
    require(cli_defaults.p_values == std::vector<double>({0.1, 0.25, 0.5, 1.0}),
            "the generated quick preset applies its probability grid");
    require(cli_defaults.solver.subset_restarts == 1
                && cli_defaults.solver.tsp_restarts == 1
                && cli_defaults.solver.tsp_ils == 20
                && cli_defaults.solver.tsp_patience == 8
                && cli_defaults.solver.sa_iters == 250
                && cli_defaults.solver.pair_exchange_passes == 0
                && cli_defaults.solver.ruin_recreate_rounds == 1
                && cli_defaults.solver.path_relink_top == 0,
            "all generated quick-preset solver values are applied together");
}

ALDOUS_TEST(test_restart_kind_metadata) {
    require(kRestartKindCount == 10U, "all restart kinds are present");
    for (std::size_t i = 0; i < kRestartKinds.size(); ++i) {
        const RestartKindInfo& info = kRestartKinds[i];
        require(info.code == static_cast<int>(i), "restart kind codes stay contiguous");
        require(restart_kind_code(info.kind) == info.code,
                "restart kind enum and metadata codes agree");
        require(std::string(restart_kind_name(info.kind)) == info.name,
                "restart kind labels come from shared metadata");
        RestartKind parsed = RestartKind::Random;
        require(restart_kind_from_code(info.code, parsed) && parsed == info.kind,
                "restart kind code round-trips");
    }
    RestartKind invalid = RestartKind::Random;
    require(!restart_kind_from_code(-1, invalid)
                && !restart_kind_from_code(static_cast<int>(kRestartKindCount), invalid),
            "invalid restart kind codes are rejected");
    require(restart_sweep_code(RestartSweep::Primary) == 0
                && restart_sweep_code(RestartSweep::Secondary) == 1,
            "restart sweep codes stay stable");
    require(restart_role_code(RestartRole::IndependentDiagnostic) == 0
                && restart_role_code(RestartRole::RacedProduction) == 4
                && is_valid_restart_role_code(3),
            "restart role codes stay stable");
    require(restart_promotion_stage_code(RestartPromotionStage::None) == 0
                && restart_promotion_stage_code(RestartPromotionStage::PilotOnly) == 1
                && restart_promotion_stage_code(RestartPromotionStage::PromotedFull) == 2
                && is_valid_restart_promotion_stage_code(2),
            "restart promotion-stage codes stay stable");
}

ALDOUS_TEST(test_search_phase_timing_add) {
    SearchPhaseTiming total;
    SearchPhaseTiming delta;
    delta.seed_construction_seconds = 1.0;
    delta.sa_seconds = 2.0;
    delta.pair_exchange_seconds = 3.0;
    delta.ejection_chain_seconds = 4.0;
    delta.exact_subset_seconds = 5.0;
    delta.sa_proposal_samples = 4;
    delta.sa_insertion_samples = 5;
    delta.sa_proposal_sample_seconds = 0.006;
    delta.sa_insertion_sample_seconds = 0.007;
    total.add(delta);
    total.add(delta);
    require(total.seed_construction_seconds == 2.0, "phase timings accumulate seed time");
    require(total.sa_seconds == 4.0, "phase timings accumulate SA time");
    require(total.pair_exchange_seconds == 6.0, "phase timings accumulate neighborhood time");
    require(total.ejection_chain_seconds == 8.0,
            "phase timings accumulate ejection-chain time");
    require(total.exact_subset_seconds == 10.0,
            "phase timings accumulate exact-subset time");
    require(total.sa_proposal_samples == 8, "phase timing proposal samples accumulate");
    require(total.sa_insertion_samples == 10, "phase timing insertion samples accumulate");
    require(std::fabs(total.sa_proposal_sample_seconds - 0.012) < 1e-15,
            "phase timing proposal durations accumulate");
    require(std::fabs(total.sa_insertion_sample_seconds - 0.014) < 1e-15,
            "phase timing insertion durations accumulate");
    SearchStats aggregate;
    SearchStats part;
    part.pair_exchange_skipped_large_k = 3;
    part.racing_pilot_restarts = 2;
    part.racing_promoted_restarts = 1;
    part.ruin_recreate_removed_nodes = 7;
    part.ruin_recreate_spatial_attempts = 1;
    part.elite_diversity_candidates = 3;
    part.elite_diversity_retained = 2;
    part.elite_diversity_rejected = 1;
    part.ejection_chain_attempts = 3;
    part.ejection_chain_steps = 7;
    part.ejection_chain_improvements = 1;
    part.ejection_chain_accepted_depth = 4;
    part.exact_subset_calls = 1;
    part.exact_subset_solved = 1;
    part.exact_subset_states = 11;
    part.exact_subset_transitions = 23;
    part.exact_subset_peak_memory_bytes = 12345;
    aggregate.add(part);
    aggregate.add(part);
    require(aggregate.pair_exchange_skipped_large_k == 6,
            "pair-exchange gate telemetry accumulates across workers");
    require(aggregate.racing_pilot_restarts == 4
                && aggregate.racing_promoted_restarts == 2,
            "restart-racing telemetry accumulates across workers");
    require(aggregate.ruin_recreate_removed_nodes == 14
                && aggregate.ruin_recreate_spatial_attempts == 2,
            "adaptive LNS telemetry accumulates across workers");
    require(aggregate.elite_diversity_candidates == 6
                && aggregate.elite_diversity_retained == 4
                && aggregate.elite_diversity_rejected == 2,
            "elite diversity telemetry accumulates across workers");
    require(aggregate.ejection_chain_attempts == 6
                && aggregate.ejection_chain_steps == 14
                && aggregate.ejection_chain_improvements == 2
                && aggregate.ejection_chain_accepted_depth == 8,
            "ejection-chain telemetry accumulates across workers");
    require(aggregate.exact_subset_calls == 2
                && aggregate.exact_subset_solved == 2
                && aggregate.exact_subset_states == 22
                && aggregate.exact_subset_transitions == 46
                && aggregate.exact_subset_peak_memory_bytes == 12345,
            "exact-subset telemetry sums work and keeps the maximum memory estimate");
}

ALDOUS_TEST(test_rng) {
    Rng a(42), b(42), c(43);
    for (int i = 0; i < 8; ++i) {
        require(a.next_u64() == b.next_u64(), "RNG deterministic for same seed");
    }
    bool differs = false;
    Rng d(42);
    for (int i = 0; i < 8; ++i) {
        if (d.next_u64() != c.next_u64()) {
            differs = true;
            break;
        }
    }
    require(differs, "RNG differs for different seed");
}

ALDOUS_TEST(test_periodic_geometry_primitives) {
    const PeriodicDomain domain{10.0};
    require(domain.valid(), "periodic domain accepts a positive finite side");
    require(std::fabs(domain.normalize(21.0) - 1.0) < 1e-12,
            "periodic normalization handles multiple positive wraps");
    require(std::fabs(domain.normalize(-0.25) - 9.75) < 1e-12,
            "periodic normalization handles negative coordinates");
    require(domain.normalize(10.0) == 0.0,
            "periodic normalization keeps the domain half-open");
    require(std::fabs(domain.signed_delta(0.1, 9.9) - 0.2) < 1e-12,
            "wrapped signed delta crosses the seam in the short direction");
    require(std::fabs(domain.delta(9.9, 0.1) - 0.2) < 1e-12,
            "wrapped absolute delta is symmetric");
    require(std::fabs(domain.distance2(Point{0.1, 9.9}, Point{9.9, 0.1}) - 0.08) < 1e-12,
            "periodic point distance wraps both axes");

    require(std::fabs(canonical_periodic_signed_delta(0.1, 9.9, 10.0) - 0.2) < 1e-12,
            "canonical signed delta matches checked seam behavior");
    require(std::fabs(canonical_periodic_distance2(Point{0.1, 9.9},
                                                    Point{9.9, 0.1}, 10.0) - 0.08) < 1e-12,
            "canonical periodic distance wraps without normalization");
    Rng geometry_rng(90210);
    for (int rep = 0; rep < 2000; ++rep) {
        const Point lhs{10.0 * geometry_rng.uniform(), 10.0 * geometry_rng.uniform()};
        const Point rhs{10.0 * geometry_rng.uniform(), 10.0 * geometry_rng.uniform()};
        const double checked = domain.distance2(lhs, rhs);
        const double canonical = canonical_periodic_distance2(lhs, rhs, 10.0);
        require(std::fabs(checked - canonical) < 1e-12,
                "canonical periodic kernel matches checked geometry on canonical points");
    }

    PeriodicMeanAccumulator seam_mean(10.0);
    seam_mean.add(9.8);
    seam_mean.add(0.2);
    const double seam = seam_mean.mean();
    require(std::min(seam, 10.0 - seam) < 1e-12,
            "circular mean keeps seam-straddling coordinates together");

    PeriodicMeanAccumulator antipodal_mean(10.0);
    antipodal_mean.add(0.0);
    antipodal_mean.add(5.0);
    require(std::fabs(antipodal_mean.mean() - 2.5) < 1e-12,
            "degenerate circular mean uses deterministic anchored unwrapping");

    PointMeanAccumulator point_mean(true, 10.0);
    point_mean.add({9.8, 0.2});
    point_mean.add({0.2, 9.8});
    const Point center = point_mean.mean();
    require(std::min(center.x, 10.0 - center.x) < 1e-12
                && std::min(center.y, 10.0 - center.y) < 1e-12,
            "periodic point mean wraps each coordinate consistently");
}

ALDOUS_TEST(test_periodic_grid_primitives) {
    for (const auto& dimensions : {std::pair<int, int>{1, 1}, {2, 2}, {3, 4}, {6, 5}}) {
        const int cells_x = dimensions.first;
        const int cells_y = dimensions.second;
        detail::GenerationMarks marks;
        marks.begin(static_cast<std::size_t>(cells_x * cells_y));
        std::set<int> visited;
        const int max_ring = detail::periodic_max_ring(cells_x, cells_y);
        for (int radius = 0; radius <= max_ring; ++radius) {
            detail::visit_periodic_ring_unique(
                0, 0, radius, cells_x, cells_y, marks,
                [&](int cell) {
                    require(visited.insert(cell).second,
                            "periodic ring traversal visits each wrapped cell once");
                });
        }
        require(static_cast<int>(visited.size()) == cells_x * cells_y,
                "periodic ring traversal covers every cell");
        require(marks.visited() == visited.size(),
                "periodic generation marks track unique cell visits");
    }

    // Validate the analytic stopping lower bound against the exact closest
    // point of every unvisited cell on representative odd/even tiny grids.
    for (int cells : {2, 3, 4, 7}) {
        const double cell = 1.0;
        const double side = static_cast<double>(cells);
        for (double offset : {0.0, 0.125, 0.5, 0.875}) {
            const double qx = offset;
            const double qy = 1.0 - offset;
            detail::GenerationMarks marks;
            marks.begin(static_cast<std::size_t>(cells * cells));
            const int max_ring = detail::periodic_max_ring(cells, cells);
            for (int radius = 0; radius <= max_ring; ++radius) {
                detail::visit_periodic_ring_unique(0, 0, radius, cells, cells, marks,
                                                   [](int) {});
                const long double bound = detail::periodic_unvisited_distance2_lower_bound(
                    qx, qy, cell, cell, cells, cells, radius);
                if (radius == max_ring) {
                    require(std::isinf(bound),
                            "periodic lower bound reports full grid coverage");
                    continue;
                }

                long double exact = std::numeric_limits<long double>::infinity();
                for (int cy = 0; cy < cells; ++cy) {
                    for (int cx = 0; cx < cells; ++cx) {
                        const int dx_raw = std::min(cx, cells - cx);
                        const int dy_raw = std::min(cy, cells - cy);
                        if (std::max(dx_raw, dy_raw) <= radius) {
                            continue;
                        }
                        const long double x0 = static_cast<long double>(cx);
                        const long double x1 = x0 + 1.0L;
                        const long double y0 = static_cast<long double>(cy);
                        const long double y1 = y0 + 1.0L;
                        long double cell_best = std::numeric_limits<long double>::infinity();
                        for (int sx = -1; sx <= 1; ++sx) {
                            for (int sy = -1; sy <= 1; ++sy) {
                                const long double shifted_x0 = x0 + static_cast<long double>(sx) * side;
                                const long double shifted_x1 = x1 + static_cast<long double>(sx) * side;
                                const long double shifted_y0 = y0 + static_cast<long double>(sy) * side;
                                const long double shifted_y1 = y1 + static_cast<long double>(sy) * side;
                                const long double dx =
                                    static_cast<long double>(qx) < shifted_x0
                                        ? shifted_x0 - static_cast<long double>(qx)
                                        : (static_cast<long double>(qx) > shifted_x1
                                               ? static_cast<long double>(qx) - shifted_x1
                                               : 0.0L);
                                const long double dy =
                                    static_cast<long double>(qy) < shifted_y0
                                        ? shifted_y0 - static_cast<long double>(qy)
                                        : (static_cast<long double>(qy) > shifted_y1
                                               ? static_cast<long double>(qy) - shifted_y1
                                               : 0.0L);
                                cell_best = std::min(cell_best, dx * dx + dy * dy);
                            }
                        }
                        exact = std::min(exact, cell_best);
                    }
                }
                require(bound <= exact + 1e-18L,
                        "periodic stopping lower bound is conservative");
            }
        }
    }
}

ALDOUS_TEST(test_periodic_instance_domain_lifecycle) {
    Instance normalized;
    normalized.periodic = true;
    normalized.explicit_side = 10.0;
    normalized.set_points({{0.0, 0.0}, {21.0, 0.0}, {-0.25, 10.25}});
    require(std::fabs(normalized.points[1].x - 1.0) < 1e-12,
            "periodic instance normalizes coordinates beyond multiple domains");
    require(std::fabs(normalized.points[2].x - 9.75) < 1e-12
                && std::fabs(normalized.points[2].y - 0.25) < 1e-12,
            "periodic instance normalizes negative and endpoint-crossing coordinates");
    require(std::fabs(normalized.dist(0, 1) - 1.0) < 1e-12,
            "normalized periodic metric agrees with the declared side");
    normalized.build_knn(2, KnnBackend::GridExact, 1.0);
    Rng verify(17);
    require(normalized.verify_knn(normalized.N, verify),
            "periodic grid and metric agree after coordinate normalization");

    Instance invalid_negative;
    invalid_negative.periodic = true;
    invalid_negative.explicit_side = -1.0;
    bool negative_threw = false;
    try {
        invalid_negative.set_points({{0.0, 0.0}});
    } catch (const std::invalid_argument&) {
        negative_threw = true;
    }
    require(negative_threw, "negative explicit periodic side is rejected");

    Instance invalid_nan;
    invalid_nan.periodic = true;
    invalid_nan.explicit_side = std::numeric_limits<double>::quiet_NaN();
    bool nan_threw = false;
    try {
        invalid_nan.set_points({{0.0, 0.0}});
    } catch (const std::invalid_argument&) {
        nan_threw = true;
    }
    require(nan_threw, "non-finite explicit periodic side is rejected");

    Instance generated;
    generated.periodic = true;
    generated.explicit_side = 100.0;
    Rng generator(1234);
    generated.generate(16, generator);
    require(generated.explicit_side == 0.0,
            "generating a fresh instance clears a stale explicit side");
    require(std::fabs(generated.side - 4.0) < 1e-12,
            "generated instance derives its own domain side");
    generated.build_knn(4, KnnBackend::GridExact);
    require(std::fabs(generated.side - 4.0) < 1e-12,
            "subsequent index construction cannot resurrect a stale side");
}

ALDOUS_TEST(test_periodic_candidate_and_index_edge_cases) {
    Instance inst;
    inst.periodic = true;
    inst.explicit_side = 10.0;
    inst.set_points({
        {0.0, 0.0},   // query
        {9.8, 0.0},   // distance 0.2, lower id
        {0.2, 0.0},   // distance 0.2, higher id
        {4.0, 0.0},
        {5.0, 0.0},
    });
    inst.build_knn(4, KnnBackend::GridExact, 1.0);

    Tour all;
    all.init(inst.N);
    all.set_tour({0, 1, 2, 3, 4}, inst);
    SubsetCandidateTable table;
    build_subset_candidates(inst, all, 2, table);
    const int row = table.row_of_node[0];
    const std::size_t offset = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(table.m);
    require(table.ids[offset] == 1 && table.ids[offset + 1U] == 2,
            "periodic subset candidates cross the seam and resolve ties by node id");
    require(std::fabs(table.dist[offset] - 0.2) < 1e-12
                && std::fabs(table.dist[offset + 1U] - 0.2) < 1e-12,
            "periodic subset candidate distances use the wrapped metric");

    // k=2 intentionally creates a 1x1 live index. Wrapped shells used to visit
    // that sole cell repeatedly and return duplicate node ids.
    Tour tiny;
    tiny.init(inst.N);
    tiny.set_tour({1, 2}, inst);
    SubsetIndex tiny_index;
    tiny_index.build(inst, tiny);
    std::vector<int> nearest;
    tiny_index.nearest(inst, 0, 4, -1, nearest);
    require(nearest == std::vector<int>({1, 2}),
            "periodic 1x1 subset index returns unique nearest members in tie order");

    Tour four;
    four.init(inst.N);
    four.set_tour({1, 2, 3, 4}, inst);
    SubsetIndex two_by_two;
    two_by_two.build(inst, four);
    two_by_two.nearest(inst, 0, 4, -1, nearest);
    require(nearest == std::vector<int>({1, 2, 3, 4}),
            "periodic 2x2 subset index visits each cell and member once");
}

ALDOUS_TEST(test_periodic_seed_metric_and_translation) {
    Instance base;
    base.periodic = true;
    base.explicit_side = 10.0;
    base.set_points({{9.9, 0.0}, {0.1, 0.0}, {4.0, 0.0}, {6.0, 0.0}});
    base.build_knn(3, KnnBackend::GridExact, 1.0);

    Rng seed_rng(91);
    std::vector<int> seed = nearest_to_point_seed(base, 0.0, 0.0, 2, seed_rng);
    std::sort(seed.begin(), seed.end());
    require(seed == std::vector<int>({0, 1}),
            "point-based periodic seed selects both sides of the seam");

    Instance translated;
    translated.periodic = true;
    translated.explicit_side = 10.0;
    translated.set_points({{19.9, -10.0}, {10.1, 10.0}, {14.0, 20.0}, {-4.0, -20.0}});
    translated.build_knn(3, KnnBackend::GridExact, 1.0);
    Rng translated_rng(91);
    std::vector<int> translated_seed =
        nearest_to_point_seed(translated, 10.0, -20.0, 2, translated_rng);
    std::sort(translated_seed.begin(), translated_seed.end());
    require(translated_seed == seed,
            "periodic point seed is invariant under whole-period translations");
}

ALDOUS_TEST(test_knn_edge_cases) {
    for (int n : {3, 4, 5, 16, 64}) {
        Rng rng(static_cast<unsigned>(n));
        Instance inst;
        inst.generate(n, rng);
        for (int k : {1, std::min(3, n - 1), n - 1}) {
            inst.build_knn(k);
            Rng verify(static_cast<unsigned>(100 + n + k));
            require(inst.verify_knn(n, verify), "KNN matches brute force");
        }
    }
    Instance boundary;
    boundary.set_points({{0,0}, {0,0}, {1,0}, {0,1}, {1,1}});
    boundary.build_knn(4);
    Rng verify(9);
    require(boundary.verify_knn(5, verify), "KNN handles duplicate/boundary points");
}


ALDOUS_TEST(test_knn_arbitrary_coordinate_property) {
    for (int rep = 0; rep < 25; ++rep) {
        Rng rng(static_cast<std::uint64_t>(9000 + rep));
        const int n = 25 + rep;
        std::vector<Point> pts;
        pts.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            double x = -50.0 + 100.0 * rng.uniform();
            double y = -80.0 + 160.0 * rng.uniform();
            if (i % 11 == 0) { x = -12.5; }
            if (i % 13 == 0) { y = 7.25; }
            pts.push_back({x, y});
        }
        if (n >= 4) {
            pts[1] = pts[0];
        }
        for (double cell : {0.0, 1.5, 7.0, 500.0}) {
            Instance inst;
            inst.set_points(pts);
            const int k = std::min(n - 1, 7);
            inst.build_knn(k, KnnBackend::GridExact, cell);
            Rng verify(static_cast<std::uint64_t>(12000 + rep));
            require(inst.verify_knn(n, verify), "grid KNN matches brute force for arbitrary coordinates");
        }
    }
}

ALDOUS_TEST(test_knn_periodic_property) {
    // Periodic (torus) KNN must match a brute-force minimum-image search across
    // sizes spanning the brute-force fallback (tiny N) and the wrapped-ring path
    // (larger N). verify_knn uses the same periodic dist2 when inst.periodic is
    // set, giving an independent minimum-image reference.
    for (int n : {5, 12, 40, 120, 400, 1500}) {
        for (unsigned seed = 1; seed <= 3; ++seed) {
            Instance inst;
            inst.periodic = true;
            Rng rng(static_cast<std::uint64_t>(4000u + seed * 17u + static_cast<unsigned>(n)));
            inst.generate(n, rng);
            for (int k : {1, std::min(5, n - 1), std::min(20, n - 1)}) {
                if (k <= 0) { continue; }
                inst.build_knn(k, KnnBackend::GridExact);
                Rng verify(static_cast<std::uint64_t>(55000u + seed + static_cast<unsigned>(n)));
                require(inst.verify_knn(n, verify),
                        "periodic grid KNN matches minimum-image brute force");
            }
        }
    }
    // A neighbor across the wrap must actually be found: two points hugging
    // opposite edges of the torus are near-adjacent under periodicity.
    {
        Instance inst;
        inst.periodic = true;
        Rng rng(1234);
        inst.generate(64, rng);  // establishes side = 8
        const double L = inst.side;
        std::vector<Point> pts = inst.points;
        pts[0] = {0.01, L * 0.5};
        pts[1] = {L - 0.01, L * 0.5};  // torus distance to pts[0] is ~0.02
        inst.set_points(pts);
        inst.periodic = true;  // set_points does not clear the flag, but be explicit
        inst.build_knn(3, KnnBackend::GridExact);
        require(inst.knn_at(0, 0) == 1, "wrap-around neighbor is the nearest on the torus");
    }
}

ALDOUS_TEST(test_knn_tiny_coordinate_property) {
    for (double scale : {1e-6, 1e-9, 1e-10, 1e-12}) {
        for (double forced_cell : {0.0, 1e-12, 1e-10}) {
            for (int rep = 0; rep < 40; ++rep) {
                Rng rng(static_cast<std::uint64_t>(15000 + rep));
                const int n = 30 + (rep % 11);
                const int k = std::min(n - 1, 9);
                const double ox = static_cast<double>((rep % 5) - 2) * scale * 1000.0;
                const double oy = static_cast<double>((rep % 7) - 3) * scale * 1000.0;
                std::vector<Point> pts;
                pts.reserve(static_cast<std::size_t>(n));
                for (int i = 0; i < n; ++i) {
                    double x = ox + scale * (2.0 * rng.uniform() - 1.0);
                    double y = oy + scale * (2.0 * rng.uniform() - 1.0);
                    if (i % 17 == 0) { x = ox; }
                    if (i % 19 == 0) { y = oy; }
                    pts.push_back({x, y});
                }
                if (n > 4) { pts[1] = pts[0]; }
                Instance inst;
                inst.set_points(pts);
                inst.build_knn(k, KnnBackend::GridExact, forced_cell);
                Rng verify(static_cast<std::uint64_t>(17000 + rep));
                require(inst.verify_knn(n, verify), "grid KNN matches brute force for tiny coordinate boxes");
            }
        }
    }
}



ALDOUS_TEST(test_knn_forced_cell_safety_cap) {
    Instance inst;
    inst.set_points({{-10000.0, -10000.0}, {10000.0, 10000.0}, {-10000.0, 10000.0}, {10000.0, -10000.0}, {0.0, 0.0}, {2500.0, -7500.0}});
    inst.build_knn(3, KnnBackend::GridExact, 1e-12);
    const long long cells = static_cast<long long>(inst.gx) * static_cast<long long>(inst.gy);
    require(cells > 0 && cells <= 262144LL, "forced tiny grid cell is capped to a safe grid size");
    Rng verify(4422);
    require(inst.verify_knn(inst.N, verify), "capped forced-cell grid KNN remains exact");
}


ALDOUS_TEST(test_knn_tiny_coordinate_scale_property) {
    auto compare_grid_to_bruteforce = [](const std::vector<Point>& pts, int k, double forced_cell_size) {
        Instance grid;
        grid.set_points(pts);
        grid.build_knn(k, KnnBackend::GridExact, forced_cell_size);
        Instance brute;
        brute.set_points(pts);
        brute.build_knn(k, KnnBackend::BruteForce, 0.0);
        for (int i = 0; i < grid.N; ++i) {
            for (int r = 0; r < k; ++r) {
                if (grid.knn_at(i, r) != brute.knn_at(i, r)) {
                    std::fprintf(stderr,
                                 "tiny-scale KNN mismatch node=%d rank=%d got=%d expected=%d forced=%.17g gx=%d gy=%d cell=%.17g\n",
                                 i, r, grid.knn_at(i, r), brute.knn_at(i, r), forced_cell_size,
                                 grid.gx, grid.gy, grid.cell_size);
                    require(false, "tiny-scale grid KNN matches brute-force ordering");
                }
            }
        }
    };

    for (int rep = 0; rep < 32; ++rep) {
        Rng rng(static_cast<std::uint64_t>(50000 + rep));
        const int n = 30 + (rep % 7);
        const double scale = (rep % 2 == 0) ? 1e-9 : 1e-12;
        std::vector<Point> pts;
        pts.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const double x = scale * (-1.0 + 2.0 * rng.uniform());
            const double y = scale * (-1.0 + 2.0 * rng.uniform());
            pts.push_back({x, y});
        }
        if (n >= 6) {
            pts[1] = pts[0];
            pts[5] = {scale, -scale};
        }
        const int k = std::min(9, n - 1);
        compare_grid_to_bruteforce(pts, k, 0.0);
        compare_grid_to_bruteforce(pts, k, 1e-12);
        compare_grid_to_bruteforce(pts, k, scale / 1024.0);
    }
}

ALDOUS_TEST(test_dist_many_from_matches_scalar) {
    // Regression, upgraded. The original version of this test used 6 hand-placed
    // points, open-square only, and 9 ids -- below the `count >= 8` threshold, so
    // it never entered the SIMD body it existed to check. Meanwhile the SIMD body
    // computed plain Euclidean distances regardless of inst.periodic, so on the
    // torus 43% of random-pair distances were wrong (unwrapped), silently biasing
    // every SA move evaluation against boundary-straddling insertions. The
    // batched routine must implement the SAME metric as Instance::dist on BOTH
    // boundary conditions and across the SIMD body AND the scalar tails.
    {
        Instance inst;
        inst.set_points({{-5.0, 4.0}, {1.0, 0.0}, {2.5, -3.0}, {10.0, 9.0}, {-7.0, 1.5}, {3.0, 3.0}});
        std::vector<int> ids = {1, 2, 3, 4, 5, 0, 2, 4, 1};
        std::vector<double> got(ids.size(), 0.0);
        dist_many_from(inst, 0, ids.data(), static_cast<int>(ids.size()), got.data());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            require(std::fabs(got[i] - inst.dist(0, ids[i])) < 1e-12, "dist_many_from matches scalar distance");
        }
    }
    for (bool periodic : {false, true}) {
        Instance inst;
        inst.periodic = periodic;
        Rng gen(periodic ? 424242u : 121212u);
        inst.generate(3000, gen);
        if (periodic) {
            for (int rep = 0; rep < 200; ++rep) {
                const int node = rep % inst.N;
                const double x = -3.0 * inst.side + 7.0 * inst.side * gen.uniform();
                const double y = -2.0 * inst.side + 5.0 * inst.side * gen.uniform();
                const PeriodicDomain domain{inst.side};
                const Point normalized = domain.normalize(Point{x, y});
                require(std::fabs(inst.dist2_to_point(node, x, y)
                                  - inst.dist2_to_canonical_point(node, normalized.x, normalized.y)) < 1e-12,
                        "checked point queries normalize once before the canonical hot path");
            }
        }
        Rng r(77u);
        for (int rep = 0; rep < 200; ++rep) {
            const int src = r.randint(inst.N);
            const int count = 1 + r.randint(97);  // exercises SIMD body + both tails
            std::vector<int> ids(static_cast<std::size_t>(count));
            for (int i = 0; i < count; ++i) { ids[static_cast<std::size_t>(i)] = r.randint(inst.N); }
            std::vector<double> got(static_cast<std::size_t>(count), -1.0);
            dist_many_from(inst, src, ids.data(), count, got.data());
            for (int i = 0; i < count; ++i) {
                const double want = inst.dist(src, ids[static_cast<std::size_t>(i)]);
                require(std::abs(got[static_cast<std::size_t>(i)] - want) < 1e-9,
                        "dist_many_from matches Instance::dist on both boundary conditions");
            }
        }
    }
}



ALDOUS_TEST(test_knn_memory_representation_and_lazy_reverse) {
    Instance inst;
    Rng rng(0xabc123U);
    inst.generate(96, rng);
    inst.build_knn(20, KnnBackend::GridExact);
    require(inst.knn.size() == 96U * 20U && inst.knn_d.size() == 96U * 20U,
            "KNN retains one node array and one distance array");
    for (int node = 0; node < inst.N; node += 7) {
        for (int rank = 0; rank < inst.knn_k; rank += 3) {
            const double distance = inst.knn_d_at(node, rank);
            require(std::fabs(inst.knn_d2_at(node, rank) - distance * distance) < 1e-15,
                    "squared KNN distance is derived from the persistent distance array");
        }
    }
    require(!inst.has_reverse_knn(), "reverse KNN is absent after ordinary KNN construction");
    std::thread first([&] { inst.ensure_reverse_knn(); });
    std::thread second([&] { inst.ensure_reverse_knn(); });
    first.join();
    second.join();
    require(inst.has_reverse_knn(), "reverse KNN is constructed safely on first use");
    require(inst.rknn_begin.size() == static_cast<std::size_t>(inst.N + 1)
                && inst.rknn_nodes.size() == inst.knn.size(),
            "lazy reverse KNN has exact CSR dimensions");
    inst.release_reverse_knn();
    require(!inst.has_reverse_knn(), "reverse KNN storage can be released explicitly");
}

ALDOUS_TEST(test_memory_budget_planning) {
    RunOptions options;
    options.N = 5000;
    options.instances = 8;
    options.threads = 8;
    options.solver.knn_k = 40;
    options.solver.restart_threads = 2;
    options.solver.reverse_knn = true;

    const MemoryPlan unlimited = estimate_experiment_memory(options, options.threads);
    require(unlimited.resolved_threads == 8 && unlimited.effective_threads == 8
                && !unlimited.limited_by_budget,
            "unlimited memory planning preserves resolved instance concurrency");
    require(unlimited.estimated_instance_bytes > 5000U * 40U * 12U,
            "memory planning includes KNN storage and solver scratch");

    RunOptions without_reverse = options;
    without_reverse.solver.reverse_knn = false;
    const MemoryPlan lean = estimate_experiment_memory(without_reverse, without_reverse.threads);
    require(lean.estimated_instance_bytes < unlimited.estimated_instance_bytes
                && !lean.reverse_knn_enabled,
            "disabling reverse KNN lowers the conservative per-instance estimate");

    RunOptions bounded = options;
    const std::uint64_t target = unlimited.fixed_overhead_bytes
        + 2U * unlimited.estimated_instance_bytes;
    bounded.memory_budget_mb = static_cast<int>((target + (1U << 20U) - 1U) >> 20U);
    const MemoryPlan limited = estimate_experiment_memory(bounded, bounded.threads);
    require(limited.effective_threads >= 1 && limited.effective_threads <= 2
                && limited.limited_by_budget,
            "memory budget reduces instance concurrency before allocation");
    require(limited.estimated_peak_bytes <= limited.budget_bytes,
            "planned peak stays within the configured budget");

    RunOptions impossible = options;
    impossible.memory_budget_mb = 1;
    bool rejected = false;
    try {
        (void)estimate_experiment_memory(impossible, impossible.threads);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "a budget below one-instance demand is rejected before allocation");
}


} // namespace
