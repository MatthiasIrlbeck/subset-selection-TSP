#include "aldous_tsp/instance.hpp"

#include "periodic_grid.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace aldous_tsp {

namespace {

constexpr double kTinyGridSpanForBruteforce = 1e-8;

double knn_bound_tol(double a, double b, double cell_size) noexcept {
    // Scale the tolerance to squared-distance magnitudes. A fixed absolute
    // epsilon is unsafe for tiny coordinate boxes because it can dominate all
    // true distances and stop the exact grid search too early.
    const double cell_scale = cell_size * cell_size;
    const double scale = std::max({std::fabs(a), std::fabs(b), std::fabs(cell_scale)});
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        return 0.0;
    }
    return 256.0 * std::numeric_limits<double>::epsilon() * scale;
}

int safe_initial_grid_radius(int knn_k, int max_r, int n, double cell_size,
                             double width, double height) noexcept {
    if (max_r <= 0 || knn_k <= 0 || n <= 0 || !(cell_size > 0.0) || !std::isfinite(cell_size)) {
        return 0;
    }
    const long double w = std::max<long double>(static_cast<long double>(width),
                                                static_cast<long double>(cell_size));
    const long double h = std::max<long double>(static_cast<long double>(height),
                                                static_cast<long double>(cell_size));
    const long double area = std::max<long double>(w * h, std::numeric_limits<long double>::min());
    const long double cell_area = std::max<long double>(static_cast<long double>(cell_size) *
                                                            static_cast<long double>(cell_size),
                                                        std::numeric_limits<long double>::min());
    const long double expected_per_cell = std::max<long double>(
        static_cast<long double>(n) * cell_area / area,
        std::numeric_limits<long double>::min());
    long double raw = std::sqrt(static_cast<long double>(knn_k) /
                                (static_cast<long double>(kPi) * expected_per_cell));
    if (raw != raw) {
        return 0;
    }
    if (raw > static_cast<long double>(max_r)) {
        return max_r;
    }
    raw = std::ceil(raw);
    if (raw <= 0.0L) {
        return 0;
    }
    if (raw >= static_cast<long double>(max_r)) {
        return max_r;
    }
    return static_cast<int>(raw);
}

} // namespace

std::uint64_t mix_hash64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t make_stream_seed(std::uint64_t base_seed, std::uint64_t instance_index, std::uint64_t tag) noexcept {
    const std::uint64_t z = base_seed ^ (0x9e3779b97f4a7c15ULL * (instance_index + 1ULL)) ^ tag;
    return mix_hash64(z);
}

void Instance::generate(int n, Rng& rng) {
    if (n < 0) {
        throw std::invalid_argument("Instance::generate requires n >= 0");
    }
    N = n;
    // A generated instance owns a fresh domain. Do not let an explicit period
    // supplied for a previous point set leak into the new sample.
    explicit_side = 0.0;
    side = std::sqrt(static_cast<double>(N));
    min_x = 0.0;
    min_y = 0.0;
    max_x = side;
    max_y = side;
    points.resize(static_cast<std::size_t>(N));
    for (Point& p : points) {
        p.x = rng.uniform() * side;
        p.y = rng.uniform() * side;
    }
    clear_knn();
}

void Instance::set_points(std::vector<Point> pts) {
    points = std::move(pts);
    N = static_cast<int>(points.size());
    recompute_bounds();
    clear_knn();
}

void Instance::recompute_bounds() {
    if (explicit_side != 0.0
        && (!(explicit_side > 0.0) || !std::isfinite(explicit_side))) {
        throw std::invalid_argument("Instance explicit_side must be zero or a positive finite value");
    }
    if (points.empty()) {
        side = (periodic && explicit_side > 0.0) ? explicit_side : 0.0;
        min_x = max_x = min_y = max_y = 0.0;
        return;
    }

    auto scan_bounds = [&]() {
        min_x = max_x = points.front().x;
        min_y = max_y = points.front().y;
        for (const Point& point : points) {
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                throw std::invalid_argument("Instance coordinates must be finite");
            }
            min_x = std::min(min_x, point.x);
            max_x = std::max(max_x, point.x);
            min_y = std::min(min_y, point.y);
            max_y = std::max(max_y, point.y);
        }
    };

    scan_bounds();
    const double width = std::max(0.0, max_x - min_x);
    const double height = std::max(0.0, max_y - min_y);
    side = std::max({std::sqrt(static_cast<double>(std::max(N, 0))),
                     width,
                     height,
                     1e-12});

    if (periodic) {
        if (explicit_side > 0.0) {
            side = explicit_side;
        }
        const PeriodicDomain domain{side};
        for (Point& point : points) {
            point = domain.normalize(point);
        }
        // Keep metadata truthful after canonicalization. Spatial consumers can
        // now rely on every periodic coordinate belonging to [0, side).
        scan_bounds();
    }
}

double Instance::dist2(int a, int b) const noexcept {
    const Point& lhs = points[static_cast<std::size_t>(a)];
    const Point& rhs = points[static_cast<std::size_t>(b)];
    return periodic ? canonical_periodic_distance2(lhs, rhs, side)
                    : euclidean_distance2(lhs, rhs);
}

double Instance::dist(int a, int b) const noexcept {
    return std::sqrt(dist2(a, b));
}

double Instance::dist2_to_point(int node, double x, double y) const noexcept {
    if (periodic) {
        const PeriodicDomain domain{side};
        return dist2_to_canonical_point(node, domain.normalize(x), domain.normalize(y));
    }
    return euclidean_distance2(points[static_cast<std::size_t>(node)], x, y);
}

double Instance::dist_to_point(int node, double x, double y) const noexcept {
    return std::sqrt(dist2_to_point(node, x, y));
}

double Instance::dist2_to_canonical_point(int node, double x, double y) const noexcept {
    const Point& point = points[static_cast<std::size_t>(node)];
    return periodic ? canonical_periodic_distance2(point, x, y, side)
                    : euclidean_distance2(point, x, y);
}

int Instance::knn_at(int node, int rank) const noexcept {
    return knn[static_cast<std::size_t>(node) * static_cast<std::size_t>(knn_k) + static_cast<std::size_t>(rank)];
}

double Instance::knn_d_at(int node, int rank) const noexcept {
    return knn_d[static_cast<std::size_t>(node) * static_cast<std::size_t>(knn_k) + static_cast<std::size_t>(rank)];
}

double Instance::knn_d2_at(int node, int rank) const noexcept {
    const double distance = knn_d_at(node, rank);
    return distance * distance;
}

void Instance::clear_knn() {
    const std::lock_guard<std::mutex> lock(*reverse_knn_mutex_);
    knn_k = 0;
    knn.clear();
    knn_d.clear();
    rknn_begin.clear();
    rknn_nodes.clear();
    cell_x.clear();
    cell_y.clear();
    cell_begin.clear();
    cell_points.clear();
    grid_min_x = grid_max_x = grid_min_y = grid_max_y = 0.0;
    gx = 0;
    gy = 0;
}

void Instance::build_knn(int k, KnnBackend backend, double forced_cell_size) {
    release_reverse_knn();
    knn_backend = backend;
    last_knn_build = KnnBuildInfo();
    last_knn_build.requested_backend = backend;
    last_knn_build.effective_backend = backend;
    last_knn_build.requested_cell_size = forced_cell_size;
    last_knn_build.coordinate_span = periodic
        ? side
        : std::max(max_x - min_x, max_y - min_y);
    if (N <= 1) {
        clear_knn();
        return;
    }
    k = std::max(0, std::min(k, N - 1));
    if (k == 0) {
        clear_knn();
        return;
    }
    if (backend == KnnBackend::BruteForce) {
        last_knn_build.effective_backend = KnnBackend::BruteForce;
        build_knn_bruteforce(k);
    } else {
        build_knn_grid(k, forced_cell_size);
    }
}

void Instance::build_knn_bruteforce(int k) {
    knn_k = k;
    const auto total = static_cast<std::size_t>(N) * static_cast<std::size_t>(knn_k);
    knn.assign(total, -1);
    knn_d.assign(total, std::numeric_limits<double>::infinity());

    std::vector<std::pair<double, int>> candidates;
    candidates.reserve(static_cast<std::size_t>(std::max(0, N - 1)));
    auto cmp = [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    };

    for (int i = 0; i < N; ++i) {
        candidates.clear();
        for (int j = 0; j < N; ++j) {
            if (j != i) {
                candidates.emplace_back(dist2(i, j), j);
            }
        }
        if (knn_k < static_cast<int>(candidates.size())) {
            std::nth_element(candidates.begin(), candidates.begin() + knn_k, candidates.end(), cmp);
        }
        std::sort(candidates.begin(), candidates.begin() + knn_k, cmp);
        const auto off = static_cast<std::size_t>(i) * static_cast<std::size_t>(knn_k);
        for (int r = 0; r < knn_k; ++r) {
            knn[off + static_cast<std::size_t>(r)] = candidates[static_cast<std::size_t>(r)].second;
            knn_d[off + static_cast<std::size_t>(r)] = std::sqrt(candidates[static_cast<std::size_t>(r)].first);
        }
    }
}

void Instance::insert_best(int* best_idx, double* best_d2, int& count, int idx, double d2) const {
    const int lim = knn_k;
    if (count < lim) {
        int pos = count;
        while (pos > 0 && (d2 < best_d2[pos - 1] || (d2 == best_d2[pos - 1] && idx < best_idx[pos - 1]))) {
            best_d2[pos] = best_d2[pos - 1];
            best_idx[pos] = best_idx[pos - 1];
            --pos;
        }
        best_d2[pos] = d2;
        best_idx[pos] = idx;
        ++count;
        return;
    }
    if (d2 > best_d2[lim - 1] || (d2 == best_d2[lim - 1] && idx >= best_idx[lim - 1])) {
        return;
    }
    int pos = lim - 1;
    while (pos > 0 && (d2 < best_d2[pos - 1] || (d2 == best_d2[pos - 1] && idx < best_idx[pos - 1]))) {
        best_d2[pos] = best_d2[pos - 1];
        best_idx[pos] = best_idx[pos - 1];
        --pos;
    }
    best_d2[pos] = d2;
    best_idx[pos] = idx;
}

void Instance::build_grid(double forced_cell_size) {
    if (N <= 0) {
        return;
    }
    recompute_bounds();
    grid_min_x = min_x;
    grid_min_y = min_y;
    grid_max_x = max_x;
    grid_max_y = max_y;
    if (periodic) {
        // The torus has period `side`. For cell-index wrapping (mod gx, gy) to
        // correspond to the torus metric, the grid must tile [0, side]^2 exactly
        // rather than the tight point bounding box.
        grid_min_x = 0.0;
        grid_min_y = 0.0;
        grid_max_x = side;
        grid_max_y = side;
    }
    double width = grid_max_x - grid_min_x;
    double height = grid_max_y - grid_min_y;
    if (!(width > 0.0)) {
        width = 1e-12;
        grid_max_x = grid_min_x + width;
    }
    if (!(height > 0.0)) {
        height = 1e-12;
        grid_max_y = grid_min_y + height;
    }

    if (forced_cell_size > 0.0 && std::isfinite(forced_cell_size)) {
        cell_size = forced_cell_size;
    } else {
        const double target_occ = std::max(4.0, std::min(16.0, 0.35 * static_cast<double>(std::max(knn_k, 1))));
        const double area_per_point = std::max(width * height / static_cast<double>(std::max(N, 1)), 1e-24);
        cell_size = std::sqrt(target_occ * area_per_point);
    }
    cell_size = std::max(cell_size, 1e-12);
    const double requested_or_initial_cell_size = cell_size;

    auto compute_grid_shape = [&]() {
        auto extent_to_cells = [](double extent, double size) -> std::int64_t {
            if (!(extent > 0.0) || !(size > 0.0) || !std::isfinite(extent) || !std::isfinite(size)) {
                return 1;
            }
            const long double raw = std::ceil(static_cast<long double>(extent) / static_cast<long double>(size));
            if (raw > static_cast<long double>(std::numeric_limits<std::int64_t>::max() / 4)) {
                return std::numeric_limits<std::int64_t>::max() / 4;
            }
            return std::max<std::int64_t>(1, static_cast<std::int64_t>(raw));
        };
        return std::pair<std::int64_t, std::int64_t>{extent_to_cells(width, cell_size), extent_to_cells(height, cell_size)};
    };

    auto too_many_cells = [](std::pair<std::int64_t, std::int64_t> shape) {
        return shape.first <= 0 || shape.second <= 0 || shape.first > kMaxGridCells / std::max<std::int64_t>(shape.second, 1);
    };

    auto shape = compute_grid_shape();
    if (too_many_cells(shape)) {
        const double area = std::max(width * height, 1e-24);
        const double area_bound = std::sqrt(area / static_cast<double>(kMaxGridCells));
        const double x_bound = width / static_cast<double>(kMaxGridCells);
        const double y_bound = height / static_cast<double>(kMaxGridCells);
        cell_size = std::max({cell_size, area_bound, x_bound, y_bound, 1e-12});
        shape = compute_grid_shape();
    }
    while (too_many_cells(shape)) {
        cell_size *= 2.0;
        if (!std::isfinite(cell_size)) {
            throw std::runtime_error("grid KNN backend could not choose a finite safe cell size");
        }
        shape = compute_grid_shape();
    }
    if (shape.first > static_cast<std::int64_t>(std::numeric_limits<int>::max()) ||
        shape.second > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("grid KNN cell dimensions exceed supported integer range");
    }

    gx = static_cast<int>(shape.first);
    gy = static_cast<int>(shape.second);
    if (periodic) {
        // Snap cell size so the grid tiles [0, side]^2 with an exact integer
        // number of cells per axis; then wrapping mod gx/gy matches the torus.
        if (gx > 0) { cell_size = side / static_cast<double>(gx); }
        cell_size = std::max(cell_size, 1e-12);
    }
    last_knn_build.effective_cell_size = cell_size;
    last_knn_build.grid_cell_capped = std::fabs(cell_size - requested_or_initial_cell_size) >
        64.0 * std::numeric_limits<double>::epsilon() * std::max(std::fabs(requested_or_initial_cell_size), 1.0);
    last_knn_build.gx = gx;
    last_knn_build.gy = gy;
    const std::int64_t cells64 = shape.first * shape.second;
    last_knn_build.grid_cells = cells64 > 0 ? static_cast<std::uint64_t>(cells64) : 0ULL;
    if (cells64 > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("grid KNN cell count exceeds supported integer range");
    }
    const int cells = static_cast<int>(cells64);
    cell_x.assign(static_cast<std::size_t>(N), 0);
    cell_y.assign(static_cast<std::size_t>(N), 0);
    std::vector<int> counts(static_cast<std::size_t>(cells), 0);
    for (int i = 0; i < N; ++i) {
        const Point& point = points[static_cast<std::size_t>(i)];
        int cx = static_cast<int>(std::floor((point.x - grid_min_x) / cell_size));
        int cy = static_cast<int>(std::floor((point.y - grid_min_y) / cell_size));
        cx = std::max(0, std::min(cx, gx - 1));
        cy = std::max(0, std::min(cy, gy - 1));
        cell_x[static_cast<std::size_t>(i)] = cx;
        cell_y[static_cast<std::size_t>(i)] = cy;
        ++counts[static_cast<std::size_t>(cy * gx + cx)];
    }
    cell_begin.assign(static_cast<std::size_t>(cells + 1), 0);
    for (int c = 0; c < cells; ++c) {
        cell_begin[static_cast<std::size_t>(c + 1)] = cell_begin[static_cast<std::size_t>(c)] + counts[static_cast<std::size_t>(c)];
    }
    cell_points.assign(static_cast<std::size_t>(N), -1);
    std::vector<int> cursor = cell_begin;
    for (int i = 0; i < N; ++i) {
        const int id = cell_y[static_cast<std::size_t>(i)] * gx + cell_x[static_cast<std::size_t>(i)];
        cell_points[static_cast<std::size_t>(cursor[static_cast<std::size_t>(id)]++)] = i;
    }
}

double Instance::cell_min_d2(double qx, double qy, int cell_id) const noexcept {
    const int ix = cell_id % gx;
    const int iy = cell_id / gx;
    const long double x0 = static_cast<long double>(grid_min_x) + static_cast<long double>(ix) * static_cast<long double>(cell_size);
    const long double x1_raw = static_cast<long double>(grid_min_x) + static_cast<long double>(ix + 1) * static_cast<long double>(cell_size);
    const long double x1 = (ix + 1 == gx) ? static_cast<long double>(grid_max_x)
                                          : std::min(static_cast<long double>(grid_max_x), x1_raw);
    const long double y0 = static_cast<long double>(grid_min_y) + static_cast<long double>(iy) * static_cast<long double>(cell_size);
    const long double y1_raw = static_cast<long double>(grid_min_y) + static_cast<long double>(iy + 1) * static_cast<long double>(cell_size);
    const long double y1 = (iy + 1 == gy) ? static_cast<long double>(grid_max_y)
                                          : std::min(static_cast<long double>(grid_max_y), y1_raw);
    const long double qxl = static_cast<long double>(qx);
    const long double qyl = static_cast<long double>(qy);
    long double dx = 0.0L;
    if (qxl < x0) { dx = x0 - qxl; }
    else if (qxl > x1) { dx = qxl - x1; }
    long double dy = 0.0L;
    if (qyl < y0) { dy = y0 - qyl; }
    else if (qyl > y1) { dy = qyl - y1; }
    const long double d2 = dx * dx + dy * dy;
    if (d2 > static_cast<long double>(std::numeric_limits<double>::max())) {
        return std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(d2);
}

double Instance::outside_bound_d2(int qi, int radius) const noexcept {
    const long double inf = std::numeric_limits<long double>::infinity();
    const int cx = cell_x[static_cast<std::size_t>(qi)];
    const int cy = cell_y[static_cast<std::size_t>(qi)];
    const long double qx = static_cast<long double>(points[static_cast<std::size_t>(qi)].x);
    const long double qy = static_cast<long double>(points[static_cast<std::size_t>(qi)].y);
    long double best_gap = inf;
    const int xmin = cx - radius;
    const int xmax = cx + radius;
    const int ymin = cy - radius;
    const int ymax = cy + radius;
    if (xmin > 0) {
        const long double boundary = static_cast<long double>(grid_min_x) + static_cast<long double>(xmin) * static_cast<long double>(cell_size);
        best_gap = std::min(best_gap, std::max(0.0L, qx - boundary));
    }
    if (xmax < gx - 1) {
        const long double boundary = std::min(static_cast<long double>(grid_max_x),
                                              static_cast<long double>(grid_min_x) + static_cast<long double>(xmax + 1) * static_cast<long double>(cell_size));
        best_gap = std::min(best_gap, std::max(0.0L, boundary - qx));
    }
    if (ymin > 0) {
        const long double boundary = static_cast<long double>(grid_min_y) + static_cast<long double>(ymin) * static_cast<long double>(cell_size);
        best_gap = std::min(best_gap, std::max(0.0L, qy - boundary));
    }
    if (ymax < gy - 1) {
        const long double boundary = std::min(static_cast<long double>(grid_max_y),
                                              static_cast<long double>(grid_min_y) + static_cast<long double>(ymax + 1) * static_cast<long double>(cell_size));
        best_gap = std::min(best_gap, std::max(0.0L, boundary - qy));
    }
    const long double d2 = best_gap * best_gap;
    if (d2 > static_cast<long double>(std::numeric_limits<double>::max())) {
        return std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(d2);
}

void Instance::build_knn_grid(int k, double forced_cell_size) {
    knn_k = k;
    recompute_bounds();
    const double coordinate_span = periodic
        ? side
        : std::max(max_x - min_x, max_y - min_y);
    last_knn_build.coordinate_span = coordinate_span;
    if (!(coordinate_span > kTinyGridSpanForBruteforce)) {
        // Very small coordinate boxes have squared distances close to double
        // roundoff of cell-bound arithmetic. Brute force is exact and cheap for
        // these pathological validation cases, so use it as the safe exact path.
        last_knn_build.effective_backend = KnnBackend::BruteForce;
        last_knn_build.brute_force_fallback = true;
        last_knn_build.effective_cell_size = 0.0;
        last_knn_build.gx = 0;
        last_knn_build.gy = 0;
        last_knn_build.grid_cells = 0;
        build_knn_bruteforce(k);
        return;
    }
    last_knn_build.effective_backend = KnnBackend::GridExact;
    build_grid(forced_cell_size);
    const auto total = static_cast<std::size_t>(N) * static_cast<std::size_t>(knn_k);
    knn.assign(total, -1);
    knn_d.assign(total, std::numeric_limits<double>::infinity());

    std::vector<int> best_idx(static_cast<std::size_t>(knn_k), -1);
    std::vector<double> best_d2(static_cast<std::size_t>(knn_k), std::numeric_limits<double>::infinity());
    const int max_r = std::max(gx, gy);
    const std::size_t kth_index = static_cast<std::size_t>(std::max(0, knn_k - 1));
    const double grid_width = std::max(grid_max_x - grid_min_x, cell_size);
    const double grid_height = std::max(grid_max_y - grid_min_y, cell_size);
    const int guess_r = safe_initial_grid_radius(knn_k, max_r, N, cell_size, grid_width, grid_height);

    auto visit_square = [&](int cx, int cy, int radius, const auto& fn) {
        const int xmin = std::max(0, cx - radius);
        const int xmax = std::min(gx - 1, cx + radius);
        const int ymin = std::max(0, cy - radius);
        const int ymax = std::min(gy - 1, cy + radius);
        for (int iy = ymin; iy <= ymax; ++iy) {
            const int row = iy * gx;
            for (int ix = xmin; ix <= xmax; ++ix) {
                fn(row + ix);
            }
        }
    };
    auto visit_ring = [&](int cx, int cy, int radius, const auto& fn) {
        if (radius == 0) {
            fn(cy * gx + cx);
            return;
        }
        const int top = cy - radius;
        const int bottom = cy + radius;
        const int left = cx - radius;
        const int right = cx + radius;
        const int xmin = std::max(0, left);
        const int xmax = std::min(gx - 1, right);
        if (top >= 0) {
            const int row = top * gx;
            for (int ix = xmin; ix <= xmax; ++ix) { fn(row + ix); }
        }
        if (bottom < gy && bottom != top) {
            const int row = bottom * gx;
            for (int ix = xmin; ix <= xmax; ++ix) { fn(row + ix); }
        }
        const int ymin = std::max(0, top + 1);
        const int ymax = std::min(gy - 1, bottom - 1);
        if (left >= 0) {
            for (int iy = ymin; iy <= ymax; ++iy) { fn(iy * gx + left); }
        }
        if (right < gx && right != left) {
            for (int iy = ymin; iy <= ymax; ++iy) { fn(iy * gx + right); }
        }
    };

    detail::GenerationMarks periodic_cell_marks;
    for (int qi = 0; qi < N; ++qi) {
        int count = 0;
        std::fill(best_idx.begin(), best_idx.end(), -1);
        std::fill(best_d2.begin(), best_d2.end(), std::numeric_limits<double>::infinity());
        const int cx = cell_x[static_cast<std::size_t>(qi)];
        const int cy = cell_y[static_cast<std::size_t>(qi)];
        const double qx = points[static_cast<std::size_t>(qi)].x;
        const double qy = points[static_cast<std::size_t>(qi)].y;
        int radius = 0;

        if (periodic) {
            auto scan_cell_wrapped = [&](int cell_id) {
                for (int p = cell_begin[static_cast<std::size_t>(cell_id)];
                     p < cell_begin[static_cast<std::size_t>(cell_id + 1)];
                     ++p) {
                    const int j = cell_points[static_cast<std::size_t>(p)];
                    if (j != qi) {
                        insert_best(best_idx.data(), best_d2.data(), count, j, dist2(qi, j));
                    }
                }
            };

            periodic_cell_marks.begin(
                static_cast<std::size_t>(gx) * static_cast<std::size_t>(gy));
            const double query_offset_x =
                qx - (grid_min_x + static_cast<double>(cx) * cell_size);
            const double query_offset_y =
                qy - (grid_min_y + static_cast<double>(cy) * cell_size);
            const int max_periodic_ring = detail::periodic_max_ring(gx, gy);
            for (radius = 0; radius <= max_periodic_ring; ++radius) {
                detail::visit_periodic_ring_unique(
                    cx, cy, radius, gx, gy, periodic_cell_marks, scan_cell_wrapped);
                if (periodic_cell_marks.visited()
                    == static_cast<std::size_t>(gx) * static_cast<std::size_t>(gy)) {
                    break;
                }
                if (count >= knn_k) {
                    const long double lower_bound =
                        detail::periodic_unvisited_distance2_lower_bound(
                            query_offset_x,
                            query_offset_y,
                            cell_size,
                            cell_size,
                            gx,
                            gy,
                            radius);
                    // Strict inequality preserves deterministic (distance, id)
                    // ordering when an unseen point can tie the kth distance.
                    if (static_cast<long double>(best_d2[kth_index]) < lower_bound) {
                        break;
                    }
                }
            }
            if (count < knn_k) {
                throw std::logic_error("periodic grid traversal did not cover enough KNN candidates");
            }
            const auto off = static_cast<std::size_t>(qi) * static_cast<std::size_t>(knn_k);
            for (int r = 0; r < knn_k; ++r) {
                knn[off + static_cast<std::size_t>(r)] = best_idx[static_cast<std::size_t>(r)];
                knn_d[off + static_cast<std::size_t>(r)] =
                    std::sqrt(best_d2[static_cast<std::size_t>(r)]);
            }
            continue;
        }

        auto scan_cell = [&](int cell_id) {
            if (count >= knn_k) {
                const double lb = cell_min_d2(qx, qy, cell_id);
                const double best = best_d2[kth_index];
                if (lb > best && (lb - best) > knn_bound_tol(lb, best, cell_size)) {
                    return;
                }
            }
            for (int p = cell_begin[static_cast<std::size_t>(cell_id)]; p < cell_begin[static_cast<std::size_t>(cell_id + 1)]; ++p) {
                const int j = cell_points[static_cast<std::size_t>(p)];
                if (j == qi) {
                    continue;
                }
                insert_best(best_idx.data(), best_d2.data(), count, j, dist2(qi, j));
            }
        };
        if (guess_r > 0) {
            visit_square(cx, cy, guess_r, scan_cell);
            radius = guess_r;
        } else {
            visit_ring(cx, cy, 0, scan_cell);
            radius = 1;
        }
        for (; radius <= max_r; ++radius) {
            if (radius > guess_r || guess_r == 0) {
                visit_ring(cx, cy, radius, scan_cell);
            }
            if (count >= knn_k) {
                const double bound = outside_bound_d2(qi, radius);
                const double best = best_d2[kth_index];
                if (!std::isfinite(bound) || (best < bound && (bound - best) > knn_bound_tol(best, bound, cell_size))) {
                    break;
                }
            }
        }
        if (count < knn_k) {
            for (int j = 0; j < N; ++j) {
                if (j != qi) {
                    insert_best(best_idx.data(), best_d2.data(), count, j, dist2(qi, j));
                }
            }
        }
        const auto off = static_cast<std::size_t>(qi) * static_cast<std::size_t>(knn_k);
        for (int r = 0; r < knn_k; ++r) {
            knn[off + static_cast<std::size_t>(r)] = best_idx[static_cast<std::size_t>(r)];
            knn_d[off + static_cast<std::size_t>(r)] = std::sqrt(best_d2[static_cast<std::size_t>(r)]);
        }
    }
}

void Instance::build_reverse_knn_unlocked() const {
    rknn_begin.assign(static_cast<std::size_t>(N + 1), 0);
    if (knn_k <= 0) {
        rknn_nodes.clear();
        return;
    }
    for (int u = 0; u < N; ++u) {
        const auto off = static_cast<std::size_t>(u) * static_cast<std::size_t>(knn_k);
        for (int r = 0; r < knn_k; ++r) {
            const int v = knn[off + static_cast<std::size_t>(r)];
            if (v >= 0 && v < N) {
                ++rknn_begin[static_cast<std::size_t>(v + 1)];
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        rknn_begin[static_cast<std::size_t>(i + 1)] += rknn_begin[static_cast<std::size_t>(i)];
    }
    rknn_nodes.assign(static_cast<std::size_t>(N) * static_cast<std::size_t>(knn_k), -1);
    std::vector<int> cursor = rknn_begin;
    for (int u = 0; u < N; ++u) {
        const auto off = static_cast<std::size_t>(u) * static_cast<std::size_t>(knn_k);
        for (int r = 0; r < knn_k; ++r) {
            const int v = knn[off + static_cast<std::size_t>(r)];
            if (v >= 0 && v < N) {
                rknn_nodes[static_cast<std::size_t>(cursor[static_cast<std::size_t>(v)]++)] = u;
            }
        }
    }
}

void Instance::ensure_reverse_knn() const {
    const std::lock_guard<std::mutex> lock(*reverse_knn_mutex_);
    const std::size_t expected_begin = static_cast<std::size_t>(N + 1);
    const std::size_t expected_nodes = static_cast<std::size_t>(N)
        * static_cast<std::size_t>(std::max(knn_k, 0));
    if (rknn_begin.size() == expected_begin
        && rknn_nodes.size() == expected_nodes) {
        return;
    }
    build_reverse_knn_unlocked();
}

void Instance::release_reverse_knn() const {
    const std::lock_guard<std::mutex> lock(*reverse_knn_mutex_);
    std::vector<int>().swap(rknn_begin);
    std::vector<int>().swap(rknn_nodes);
}

bool Instance::has_reverse_knn() const {
    const std::lock_guard<std::mutex> lock(*reverse_knn_mutex_);
    return rknn_begin.size() == static_cast<std::size_t>(N + 1)
        && rknn_nodes.size() == static_cast<std::size_t>(N)
            * static_cast<std::size_t>(std::max(knn_k, 0));
}

bool Instance::verify_knn(int checks, Rng& rng) const {
    if (N <= 1 || knn_k <= 0) {
        return true;
    }
    checks = std::max(checks, 0);
    if (checks == 0) {
        return true;
    }
    std::vector<std::pair<double, int>> candidates;
    candidates.reserve(static_cast<std::size_t>(N - 1));
    auto cmp = [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    };
    for (int c = 0; c < checks; ++c) {
        const int i = (checks >= N) ? (c % N) : rng.randint(N);
        candidates.clear();
        for (int j = 0; j < N; ++j) {
            if (j != i) {
                candidates.emplace_back(dist2(i, j), j);
            }
        }
        std::sort(candidates.begin(), candidates.end(), cmp);
        for (int r = 0; r < knn_k; ++r) {
            if (knn_at(i, r) != candidates[static_cast<std::size_t>(r)].second) {
                return false;
            }
            const double expected_d2 = candidates[static_cast<std::size_t>(r)].first;
            const double tolerance = kDistanceEps * std::max(1.0, std::fabs(expected_d2));
            if (std::fabs(knn_d2_at(i, r) - expected_d2) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

void dist_many_from(const Instance& inst, int src, const int* ids, int count, double* out) {
    if (count <= 0 || ids == nullptr || out == nullptr) {
        return;
    }
    const Point& source = inst.points[static_cast<std::size_t>(src)];
#if defined(__AVX2__)
    if (count >= 8) {
        const double sx = source.x;
        const double sy = source.y;
        alignas(32) double xs[4];
        alignas(32) double ys[4];
        int i = 0;
        const __m256d vsx = _mm256_set1_pd(sx);
        const __m256d vsy = _mm256_set1_pd(sy);
        const __m256d vside = _mm256_set1_pd(inst.side);
        const __m256d sign_mask = _mm256_set1_pd(-0.0);
        for (; i + 4 <= count; i += 4) {
            for (int lane = 0; lane < 4; ++lane) {
                const Point& point = inst.points[static_cast<std::size_t>(ids[i + lane])];
                xs[lane] = point.x;
                ys[lane] = point.y;
            }
            const __m256d vx = _mm256_load_pd(xs);
            const __m256d vy = _mm256_load_pd(ys);
            __m256d dx = _mm256_sub_pd(vsx, vx);
            __m256d dy = _mm256_sub_pd(vsy, vy);
            if (inst.periodic) {
                dx = _mm256_andnot_pd(sign_mask, dx);
                dy = _mm256_andnot_pd(sign_mask, dy);
                dx = _mm256_min_pd(dx, _mm256_sub_pd(vside, dx));
                dy = _mm256_min_pd(dy, _mm256_sub_pd(vside, dy));
            }
#if defined(__FMA__)
            const __m256d d2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
#else
            const __m256d d2 = _mm256_add_pd(_mm256_mul_pd(dx, dx), _mm256_mul_pd(dy, dy));
#endif
            _mm256_storeu_pd(out + i, _mm256_sqrt_pd(d2));
        }
        if (inst.periodic) {
            for (; i < count; ++i) {
                const Point& point = inst.points[static_cast<std::size_t>(ids[i])];
                out[i] = std::sqrt(canonical_periodic_distance2(source, point, inst.side));
            }
        } else {
            for (; i < count; ++i) {
                const Point& point = inst.points[static_cast<std::size_t>(ids[i])];
                out[i] = std::sqrt(euclidean_distance2(source, point));
            }
        }
        return;
    }
#endif
    if (inst.periodic) {
        for (int i = 0; i < count; ++i) {
            const Point& point = inst.points[static_cast<std::size_t>(ids[i])];
            out[i] = std::sqrt(canonical_periodic_distance2(source, point, inst.side));
        }
    } else {
        for (int i = 0; i < count; ++i) {
            const Point& point = inst.points[static_cast<std::size_t>(ids[i])];
            out[i] = std::sqrt(euclidean_distance2(source, point));
        }
    }
}

} // namespace aldous_tsp
