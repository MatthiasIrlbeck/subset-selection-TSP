#include "aldous_tsp/config.hpp"
#include "aldous_tsp/experiment.hpp"
#include "aldous_tsp/geometry.hpp"
#include "aldous_tsp/instance.hpp"
#include "aldous_tsp/oracle.hpp"
#include "aldous_tsp/lower_bound.hpp"
#include "aldous_tsp/results.hpp"
#include "aldous_tsp/solver.hpp"
#include "aldous_tsp/tour.hpp"

#include "solver_internal.hpp"
#include "periodic_grid.hpp"
#include "worker.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace aldous_tsp;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::exit(1);
    }
}

long count_restart_kind(const std::vector<RestartRecord>& records,
                        RestartKind kind) {
    return static_cast<long>(std::count_if(
        records.begin(), records.end(),
        [kind](const RestartRecord& record) { return record.kind == kind; }));
}

void test_restart_kind_metadata() {
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
}

void test_search_phase_timing_add() {
    SearchPhaseTiming total;
    SearchPhaseTiming delta;
    delta.seed_construction_seconds = 1.0;
    delta.sa_seconds = 2.0;
    delta.pair_exchange_seconds = 3.0;
    delta.sa_proposal_samples = 4;
    delta.sa_insertion_samples = 5;
    delta.sa_proposal_sample_seconds = 0.006;
    delta.sa_insertion_sample_seconds = 0.007;
    total.add(delta);
    total.add(delta);
    require(total.seed_construction_seconds == 2.0, "phase timings accumulate seed time");
    require(total.sa_seconds == 4.0, "phase timings accumulate SA time");
    require(total.pair_exchange_seconds == 6.0, "phase timings accumulate neighborhood time");
    require(total.sa_proposal_samples == 8, "phase timing proposal samples accumulate");
    require(total.sa_insertion_samples == 10, "phase timing insertion samples accumulate");
    require(std::fabs(total.sa_proposal_sample_seconds - 0.012) < 1e-15,
            "phase timing proposal durations accumulate");
    require(std::fabs(total.sa_insertion_sample_seconds - 0.014) < 1e-15,
            "phase timing insertion durations accumulate");
}

void test_rng() {
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

void test_periodic_geometry_primitives() {
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

void test_periodic_grid_primitives() {
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

void test_periodic_instance_domain_lifecycle() {
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

void test_periodic_candidate_and_index_edge_cases() {
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

void test_periodic_seed_metric_and_translation() {
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

void test_knn_edge_cases() {
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


void test_knn_arbitrary_coordinate_property() {
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

void test_knn_periodic_property() {
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

void test_knn_tiny_coordinate_property() {
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



void test_knn_forced_cell_safety_cap() {
    Instance inst;
    inst.set_points({{-10000.0, -10000.0}, {10000.0, 10000.0}, {-10000.0, 10000.0}, {10000.0, -10000.0}, {0.0, 0.0}, {2500.0, -7500.0}});
    inst.build_knn(3, KnnBackend::GridExact, 1e-12);
    const long long cells = static_cast<long long>(inst.gx) * static_cast<long long>(inst.gy);
    require(cells > 0 && cells <= 262144LL, "forced tiny grid cell is capped to a safe grid size");
    Rng verify(4422);
    require(inst.verify_knn(inst.N, verify), "capped forced-cell grid KNN remains exact");
}


void test_knn_tiny_coordinate_scale_property() {
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

void test_dist_many_from_matches_scalar() {
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


void test_tour_invariants() {
    Instance inst;
    inst.set_points({{0,0}, {1,0}, {1,1}, {0,1}, {2,1}});
    inst.build_knn(4);
    Tour tour;
    tour.init(inst.N);
    tour.set_tour({0, 1, 2, 3}, inst);
    require(tour.check_invariants(), "tour invariants after set_tour");
    require(std::fabs(tour.length - 4.0) < 1e-9, "square tour length");
}

void test_tour_incremental_mutation_property() {
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

void test_exact_small_tsp() {
    Instance inst;
    inst.set_points({{0,0}, {1,0}, {1,1}, {0,1}});
    inst.build_knn(3);
    std::vector<int> cycle;
    double len = 0.0;
    require(exact_small_tsp_cycle(inst, {0, 1, 2, 3}, cycle, len), "exact small TSP succeeds");
    require(std::fabs(len - 4.0) < 1e-9, "exact square length");
}

void test_exact_small_tsp_randomized() {
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

void test_two_opt_shorter_side_reversal() {
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

void test_two_opt_property() {
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


void test_incremental_tour_mutation_property() {
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

void test_elite_collision_safe_key() {
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
}

void test_solver_smoke() {
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


void test_object_oriented_facades() {
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

void test_disable_two_opt_ablation_exact() {
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


void test_or_opt_reaches_local_optimum() {
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

void test_or_opt_segment_property() {
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

void test_solver_matches_exact_enumeration_tiny() {
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

void test_subset_candidate_table_exact() {
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

void test_effective_sa_iters_scaling() {
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

void test_restart_worker_exception_propagation() {
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

void test_experiment_worker_states_and_callback_exceptions() {
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

void test_elite_kick_near_full() {
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

void test_kick_restarts_mechanics() {
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

void test_restart_value_logging() {
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

void test_region_seeds() {
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

void test_dense_seeds_are_not_exploration_seeds() {
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

void test_subset_index_matches_bruteforce() {
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

void test_spatial_insertion_saturates_to_exact() {
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

void test_windowed_insertion_correctness() {
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

void test_restarts_flag_is_authoritative() {
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

    // The AUTO default (-1) must reproduce the historical effective behavior:
    // 8 restarts at p <= 0.08 (the seed pool used to force this) and 3 above.
    // Honoring the flag literally with a default of 3 silently degraded
    // default-quality at small p, which is the tool's core regime.
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
        require(smallp.stats.subset_restarts == 8, "auto default runs 8 restarts at p <= 0.08");
        Rng rng2(99u);
        const SolveResult largep = solve_subset(inst, 90, rng2, opt, nullptr);  // p = 0.30
        require(largep.stats.subset_restarts == 3, "auto default runs 3 restarts at p > 0.08");
    }
}

void test_elite_anytime_restarts() {
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

void test_time_budget_adds_restarts() {
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

void test_two_opt_candidate_table_property() {
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

void test_restart_thread_invariance() {
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
                    && a.restarts[i].centroid_x == b.restarts[i].centroid_x
                    && a.restarts[i].centroid_y == b.restarts[i].centroid_y
                    && a.restarts[i].radius == b.restarts[i].radius,
                "restart records are invariant to restart parallelism");
    }
}

void test_best_restart_diagnostic() {
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
    require(tsp.restarts.front().kind == RestartKind::TspFarthestInsertion,
            "the first TSP restart records farthest insertion");
    for (std::size_t i = 1; i < tsp.restarts.size(); ++i) {
        require(tsp.restarts[i].kind == RestartKind::TspNearestNeighbor,
                "later TSP restarts record nearest-neighbor construction");
    }
}

void test_second_sweep_never_worse() {
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
            const std::size_t expected = has_secondary ? 4U : 2U;
            require(pv.restarts.size() == expected,
                    "second sweep appends rather than replacing restart records");
            for (std::size_t ri = 0; ri < pv.restarts.size(); ++ri) {
                const RestartSweep expected_sweep = (has_secondary && ri >= 2U)
                    ? RestartSweep::Secondary
                    : RestartSweep::Primary;
                require(pv.restarts[ri].sweep == expected_sweep,
                        "restart sweep tags preserve append order");
            }
            if (pv.k == base.N) {
                require(pv.restarts[0].kind == RestartKind::TspFarthestInsertion
                            && pv.restarts[1].kind == RestartKind::TspNearestNeighbor,
                        "p=1 rows serialize the full-TSP restart population");
            }
        }
    }
}

void test_oracle_large_coordinates_precision() {
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

void test_held_karp_bound() {
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

void test_control_variate_bounds() {
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
    require(doc.full_bound_expectation > 0.0, "control variate estimates E[B_full]");
    const double per_point = doc.full_bound_expectation / static_cast<double>(opt.N);
    require(std::fabs(per_point - 0.625) < 0.02,
            "E[B_full]/N matches the Poisson-torus prediction 0.625");

    for (const auto& row : doc.instance_rows) {
        require(row.full_bound >= 0.0, "instance carries a full-set bound");
        for (const auto& pv : row.p_results) {
            require(pv.subset_bound >= 0.0, "instance p-row carries a subset bound");
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
        require(s.subset_bound_mean <= s.mean + 1e-6, "mean subset bound brackets f(p) from below");
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

void test_oracle_torus_roundtrip() {
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

void test_path_relink_step_matches_bruteforce() {
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
        for (int ri : remove_positions) {
            for (int add : add_nodes) {
                const SwapMoveEval eval = evaluate_swap_after_remove(inst, tour, ri, add);
                if (eval.valid && eval.delta < brute_best) {
                    brute_best = eval.delta;
                }
            }
        }
        const PathRelinkStep step = path_relink_best_step(inst, tour, remove_positions, add_nodes);
        require(step.valid == std::isfinite(brute_best), "relink step validity matches brute force");
        if (step.valid) {
            require(std::abs(step.delta - brute_best) <= 1e-9 * (1.0 + std::abs(brute_best)),
                    "decomposed relink step delta matches brute-force best pair");
            const SwapInsertionMove applied = find_best_insert_after_remove(inst, tour, step.remove_pos, step.add_node);
            require(applied.valid, "chosen relink step is applicable");
            require(std::abs(applied.delta - step.delta) <= 1e-9 * (1.0 + std::abs(step.delta)),
                    "applied relink move delta matches selected step delta");
        }
    }
}

void test_path_relink_distance_cap() {
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

void test_path_relink_counter_semantics() {
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


void test_oracle_parser_and_fake_lkh() {
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


void test_grid_cell_safety_for_tiny_forced_cell() {
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

void test_solver_ablation_flags_are_exact() {
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

void test_oracle_top_n_matches_cli_config() {
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


void test_oracle_posix_spawn_timeout_and_concurrency() {
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

void test_json_numeric_precision() {
    ResultsDocument doc;
    doc.N = 12;
    doc.instances_done = 1;
    doc.instances_target = 1;
    doc.options.N = 12;
    doc.options.instances = 1;
    doc.options.solver.grid_cell = 1e-12;
    doc.p_values = {1e-9, 2e-9};
    PValueSummary s1;
    s1.k = 3;
    s1.values = {1e-12};
    s1.mean = 1e-12;
    s1.min = 1e-12;
    s1.max = 1e-12;
    PValueSummary s2 = s1;
    s2.values = {2e-12};
    s2.mean = 2e-12;
    s2.min = 2e-12;
    s2.max = 2e-12;
    doc.summary["1e-09"] = s1;
    doc.summary["2e-09"] = s2;
    const std::string text = results_to_json(doc);
    require(text.find("\"grid_cell\": 0") == std::string::npos, "tiny grid-cell does not serialize as zero");
    require(text.find("e-12") != std::string::npos || text.find("e-13") != std::string::npos,
            "tiny grid-cell is serialized with scientific precision");
    require(text.find("1e-09") != std::string::npos || text.find("1.0000000000000001e-09") != std::string::npos,
            "tiny p-value is serialized with scientific precision");
}


void test_json_escape_regression() {
    const std::string escaped = json_escape("quote\" backslash\\ newline\n tab\t control\x01");
    require(escaped.find("\\\"") != std::string::npos, "JSON escaping handles quotes");
    require(escaped.find("\\\\") != std::string::npos, "JSON escaping handles backslashes");
    require(escaped.find("\\n") != std::string::npos, "JSON escaping handles newlines");
    require(escaped.find("\\t") != std::string::npos, "JSON escaping handles tabs");
    require(escaped.find("\\u0001") != std::string::npos, "JSON escaping handles control characters");
}

void test_json_atomic() {
    ResultsDocument doc;
    doc.N = 10;
    doc.instances_done = 1;
    doc.instances_target = 1;
    doc.options.solver.grid_cell = 1.0e-12;
    doc.p_values = {1.0e-9, 2.0e-9, 1.0};
    PValueSummary s;
    s.k = 10;
    s.values = {0.7};
    s.mean = 0.7;
    s.min = 0.7;
    s.max = 0.7;
    doc.summary["1.0"] = s;
    const std::string text = results_to_json(doc);
    require(text.find("\"schema_version\": 13") != std::string::npos, "JSON schema version present");
    require(text.find("\"build_metadata\"") != std::string::npos, "JSON includes build metadata");
    require(text.find("\"oracle_call_records\"") != std::string::npos, "JSON includes oracle call records");
    require(text.find("\"summary_rows\"") != std::string::npos, "JSON includes array-form summary rows");
    require(text.find("\"knn_build_seconds\"") != std::string::npos, "JSON includes KNN timing stats");
    require(text.find("\"knn_requested_grid_instances\"") != std::string::npos, "JSON includes effective KNN backend stats");
    require(text.find("\"instance_rows\"") != std::string::npos, "JSON includes per-instance row container");
    require(text.find("\"target_compile_options\"") != std::string::npos, "JSON includes target compile options metadata");
    require(text.find("\"effective_optimization_level\"") != std::string::npos, "JSON includes structured optimization-level metadata");
    require(text.find("e-13") != std::string::npos || text.find("e-12") != std::string::npos,
            "JSON preserves tiny grid-cell scale");
    require(text.find("e-09") != std::string::npos || text.find("e-9") != std::string::npos,
            "JSON preserves tiny p-value scale");
    require(text.find("0.0000000000") == std::string::npos, "JSON does not fixed-format tiny values to zero");
    const std::filesystem::path out = std::filesystem::temp_directory_path() / "aldous_tsp_test_results.json";
    std::string err;
    require(write_text_file_atomic(out.string(), text, &err), "atomic write succeeds");
    std::ifstream in(out, std::ios::binary);
    std::string roundtrip((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    require(roundtrip == text, "atomic write roundtrip");
    std::filesystem::remove(out);
}

} // namespace

int main(int argc, char** argv) {
    // Optional substring filter: `aldous_tsp_test_core kick` runs only tests
    // whose name contains "kick". Sanitizer builds are ~20x slower than the
    // release build, so being able to sanitize just the code paths a change
    // touched -- instead of the whole suite -- is the difference between
    // running ASan on every change and not running it at all.
    const char* filter = (argc > 1) ? argv[1] : nullptr;
#define RUN_TEST(name) do { \
        if (filter == nullptr || std::string(#name).find(filter) != std::string::npos) { \
            std::fprintf(stderr, "running %s\n", #name); \
            name(); \
        } \
    } while (false)
    RUN_TEST(test_restart_kind_metadata);
    RUN_TEST(test_search_phase_timing_add);
    RUN_TEST(test_rng);
    RUN_TEST(test_periodic_geometry_primitives);
    RUN_TEST(test_periodic_grid_primitives);
    RUN_TEST(test_periodic_instance_domain_lifecycle);
    RUN_TEST(test_periodic_candidate_and_index_edge_cases);
    RUN_TEST(test_periodic_seed_metric_and_translation);
    RUN_TEST(test_knn_edge_cases);
    RUN_TEST(test_knn_periodic_property);
    RUN_TEST(test_knn_arbitrary_coordinate_property);
    RUN_TEST(test_knn_tiny_coordinate_property);
    RUN_TEST(test_knn_tiny_coordinate_scale_property);
    RUN_TEST(test_knn_forced_cell_safety_cap);
    RUN_TEST(test_grid_cell_safety_for_tiny_forced_cell);
    RUN_TEST(test_dist_many_from_matches_scalar);
    RUN_TEST(test_tour_invariants);
    RUN_TEST(test_tour_incremental_mutation_property);
    RUN_TEST(test_exact_small_tsp);
    RUN_TEST(test_exact_small_tsp_randomized);
    RUN_TEST(test_two_opt_shorter_side_reversal);
    RUN_TEST(test_two_opt_property);
    RUN_TEST(test_incremental_tour_mutation_property);
    RUN_TEST(test_elite_collision_safe_key);
    RUN_TEST(test_solver_smoke);
    RUN_TEST(test_object_oriented_facades);
    RUN_TEST(test_disable_two_opt_ablation_exact);
    RUN_TEST(test_solver_ablation_flags_are_exact);
    RUN_TEST(test_restart_thread_invariance);
    RUN_TEST(test_best_restart_diagnostic);
    RUN_TEST(test_second_sweep_never_worse);
    RUN_TEST(test_control_variate_bounds);
    RUN_TEST(test_held_karp_bound);
    RUN_TEST(test_or_opt_reaches_local_optimum);
    RUN_TEST(test_or_opt_segment_property);
    RUN_TEST(test_solver_matches_exact_enumeration_tiny);
    RUN_TEST(test_subset_candidate_table_exact);
    RUN_TEST(test_effective_sa_iters_scaling);
    RUN_TEST(test_restart_worker_exception_propagation);
    RUN_TEST(test_experiment_worker_states_and_callback_exceptions);
    RUN_TEST(test_elite_kick_near_full);
    RUN_TEST(test_kick_restarts_mechanics);
    RUN_TEST(test_region_seeds);
    RUN_TEST(test_restart_value_logging);
    RUN_TEST(test_dense_seeds_are_not_exploration_seeds);
    RUN_TEST(test_subset_index_matches_bruteforce);
    RUN_TEST(test_spatial_insertion_saturates_to_exact);
    RUN_TEST(test_windowed_insertion_correctness);
    RUN_TEST(test_restarts_flag_is_authoritative);
    RUN_TEST(test_time_budget_adds_restarts);
    RUN_TEST(test_elite_anytime_restarts);
    RUN_TEST(test_two_opt_candidate_table_property);
    RUN_TEST(test_path_relink_step_matches_bruteforce);
#if defined(ALDOUS_TSP_TESTS_DIR)
    RUN_TEST(test_oracle_torus_roundtrip);
    RUN_TEST(test_oracle_large_coordinates_precision);
#endif
    RUN_TEST(test_path_relink_distance_cap);
    RUN_TEST(test_path_relink_counter_semantics);
#if !defined(_WIN32)
    // These tests spawn a POSIX shell script as a fake LKH binary, which cannot
    // execute on Windows. The oracle parsing/config logic they exercise is
    // otherwise platform-independent; only the process-launch path is skipped.
    RUN_TEST(test_oracle_parser_and_fake_lkh);
    RUN_TEST(test_oracle_top_n_matches_cli_config);
    RUN_TEST(test_oracle_posix_spawn_timeout_and_concurrency);
#endif
    RUN_TEST(test_json_numeric_precision);
    RUN_TEST(test_json_escape_regression);
    RUN_TEST(test_json_atomic);
#undef RUN_TEST
    std::printf("core unit tests passed\n");
    return 0;
}
