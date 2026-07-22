#include "aldous_tsp/validation.hpp"

#include "aldous_tsp/instance.hpp"
#include "aldous_tsp/tour.hpp"
#include "generated_options.hpp"
#include "validation_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace aldous_tsp {
namespace {

bool printable_identity(const std::string& value) {
    return !value.empty() && value.size() <= 256U
        && std::none_of(value.begin(), value.end(), [](const unsigned char ch) {
            return ch < 0x20U || ch == 0x7fU;
        });
}

bool nearly_equal(const double lhs, const double rhs,
                  const double ulp_scale = 128.0) noexcept {
    if (lhs == rhs) {
        return true;
    }
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return false;
    }
    const double scale = std::max({1.0, std::fabs(lhs), std::fabs(rhs)});
    return std::fabs(lhs - rhs)
        <= ulp_scale * std::numeric_limits<double>::epsilon() * scale;
}

bool validate_metric_domain(const Instance& instance, std::string& error) {
    const long double double_max =
        static_cast<long double>(std::numeric_limits<double>::max());
    long double max_d2 = 0.0L;
    if (instance.periodic) {
        if (!std::isfinite(instance.side) || !(instance.side > 0.0)) {
            error = "periodic instance side must be finite and positive";
            return false;
        }
        const long double half = static_cast<long double>(instance.side) / 2.0L;
        max_d2 = 2.0L * half * half;
        for (const Point& point : instance.points) {
            if (!(point.x >= 0.0 && point.x < instance.side)
                || !(point.y >= 0.0 && point.y < instance.side)) {
                error = "periodic instance coordinates must be canonical in [0,side)";
                return false;
            }
        }
    } else {
        long double min_x = static_cast<long double>(instance.points.front().x);
        long double max_x = min_x;
        long double min_y = static_cast<long double>(instance.points.front().y);
        long double max_y = min_y;
        for (const Point& point : instance.points) {
            min_x = std::min(min_x, static_cast<long double>(point.x));
            max_x = std::max(max_x, static_cast<long double>(point.x));
            min_y = std::min(min_y, static_cast<long double>(point.y));
            max_y = std::max(max_y, static_cast<long double>(point.y));
        }
        const long double dx = max_x - min_x;
        const long double dy = max_y - min_y;
        if (!std::isfinite(dx) || !std::isfinite(dy)) {
            error = "instance coordinate span is not representable";
            return false;
        }
        max_d2 = dx * dx + dy * dy;
    }
    if (!std::isfinite(max_d2) || max_d2 > double_max) {
        error = "instance metric exceeds the representable double-precision distance domain";
        return false;
    }
    return true;
}

bool validate_knn_rows(const Instance& instance, std::string& error) {
    if (instance.knn_k == 0) {
        if (!instance.knn.empty() || !instance.knn_d.empty()) {
            error = "instance with knn_k=0 must not retain KNN arrays";
            return false;
        }
        return true;
    }

    const std::size_t expected = static_cast<std::size_t>(instance.N)
        * static_cast<std::size_t>(instance.knn_k);
    if (instance.knn.size() != expected || instance.knn_d.size() != expected) {
        error = "instance KNN arrays are inconsistent with N and knn_k";
        return false;
    }

    std::vector<int> seen(static_cast<std::size_t>(instance.N), -1);
    for (int node = 0; node < instance.N; ++node) {
        double previous_d2 = -1.0;
        int previous_id = -1;
        const std::size_t offset = static_cast<std::size_t>(node)
            * static_cast<std::size_t>(instance.knn_k);
        for (int rank = 0; rank < instance.knn_k; ++rank) {
            const std::size_t index = offset + static_cast<std::size_t>(rank);
            const int neighbor = instance.knn[index];
            const double stored = instance.knn_d[index];
            if (neighbor < 0 || neighbor >= instance.N || neighbor == node) {
                error = "instance KNN row contains an invalid node ID";
                return false;
            }
            if (seen[static_cast<std::size_t>(neighbor)] == node) {
                error = "instance KNN row contains a duplicate node ID";
                return false;
            }
            seen[static_cast<std::size_t>(neighbor)] = node;
            if (!std::isfinite(stored) || stored < 0.0) {
                error = "instance KNN row contains a nonfinite or negative distance";
                return false;
            }
            const double actual_d2 = instance.dist2(node, neighbor);
            if (!std::isfinite(actual_d2) || actual_d2 < 0.0) {
                error = "instance metric produced a nonfinite KNN distance";
                return false;
            }
            const double actual = std::sqrt(actual_d2);
            if (!nearly_equal(stored, actual, 512.0)) {
                error = "instance KNN distance does not agree with the active metric";
                return false;
            }
            if (rank > 0
                && (actual_d2 < previous_d2
                    || (actual_d2 == previous_d2 && neighbor < previous_id))) {
                error = "instance KNN row is not ordered by (distance,node ID)";
                return false;
            }
            previous_d2 = actual_d2;
            previous_id = neighbor;
        }
    }
    return true;
}

bool validate_grid_structure(const Instance& instance, std::string& error) {
    if (instance.knn_k == 0 || instance.knn_backend == KnnBackend::BruteForce
        || instance.last_knn_build.effective_backend == KnnBackend::BruteForce) {
        if (!instance.cell_x.empty() || !instance.cell_y.empty()
            || !instance.cell_begin.empty() || !instance.cell_points.empty()
            || instance.gx != 0 || instance.gy != 0) {
            error = "non-grid instance contains stale grid metadata";
            return false;
        }
        return true;
    }
    if (instance.gx <= 0 || instance.gy <= 0
        || !std::isfinite(instance.cell_size) || !(instance.cell_size > 0.0)
        || !std::isfinite(instance.grid_min_x)
        || !std::isfinite(instance.grid_min_y)
        || !std::isfinite(instance.grid_max_x)
        || !std::isfinite(instance.grid_max_y)
        || !(instance.grid_max_x > instance.grid_min_x)
        || !(instance.grid_max_y > instance.grid_min_y)) {
        error = "instance grid geometry is invalid";
        return false;
    }
    const std::uint64_t cells = static_cast<std::uint64_t>(instance.gx)
        * static_cast<std::uint64_t>(instance.gy);
    if (cells == 0U || cells > static_cast<std::uint64_t>(kMaxGridCells)) {
        error = "instance grid cell count is invalid";
        return false;
    }
    if (instance.cell_x.size() != static_cast<std::size_t>(instance.N)
        || instance.cell_y.size() != static_cast<std::size_t>(instance.N)
        || instance.cell_points.size() != static_cast<std::size_t>(instance.N)
        || instance.cell_begin.size() != static_cast<std::size_t>(cells + 1U)) {
        error = "instance grid arrays have inconsistent sizes";
        return false;
    }
    if (instance.cell_begin.front() != 0
        || instance.cell_begin.back() != instance.N) {
        error = "instance grid prefix table has invalid endpoints";
        return false;
    }
    for (std::size_t i = 1; i < instance.cell_begin.size(); ++i) {
        if (instance.cell_begin[i] < instance.cell_begin[i - 1U]
            || instance.cell_begin[i] < 0
            || instance.cell_begin[i] > instance.N) {
            error = "instance grid prefix table is not monotone";
            return false;
        }
    }
    std::vector<unsigned char> seen(static_cast<std::size_t>(instance.N), 0U);
    for (int i = 0; i < instance.N; ++i) {
        const int cx = instance.cell_x[static_cast<std::size_t>(i)];
        const int cy = instance.cell_y[static_cast<std::size_t>(i)];
        if (cx < 0 || cx >= instance.gx || cy < 0 || cy >= instance.gy) {
            error = "instance grid contains an out-of-range point cell";
            return false;
        }
    }
    for (const int node : instance.cell_points) {
        if (node < 0 || node >= instance.N
            || seen[static_cast<std::size_t>(node)] != 0U) {
            error = "instance grid point permutation is invalid";
            return false;
        }
        seen[static_cast<std::size_t>(node)] = 1U;
    }
    return true;
}

bool validate_reverse_knn_structure(const Instance& instance,
                                    std::string& error) {
    if (instance.rknn_begin.empty() && instance.rknn_nodes.empty()) {
        return true;
    }
    if (instance.rknn_begin.size() != static_cast<std::size_t>(instance.N + 1)
        || instance.rknn_begin.front() != 0
        || instance.rknn_begin.back()
            != static_cast<int>(instance.rknn_nodes.size())) {
        error = "instance reverse-KNN prefix table is invalid";
        return false;
    }
    for (std::size_t i = 1; i < instance.rknn_begin.size(); ++i) {
        if (instance.rknn_begin[i] < instance.rknn_begin[i - 1U]) {
            error = "instance reverse-KNN prefix table is not monotone";
            return false;
        }
    }
    for (const int node : instance.rknn_nodes) {
        if (node < 0 || node >= instance.N) {
            error = "instance reverse-KNN adjacency contains an invalid node";
            return false;
        }
    }
    return true;
}

bool validate_instance_structure(const Instance& instance, std::string& error) {
    if (instance.N < 3) {
        error = "instance must contain at least three points";
        return false;
    }
    if (instance.points.size() != static_cast<std::size_t>(instance.N)) {
        error = "instance point count does not match N";
        return false;
    }
    for (const Point& point : instance.points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            error = "instance coordinates must be finite";
            return false;
        }
    }
    if (instance.explicit_side != 0.0
        && (!std::isfinite(instance.explicit_side)
            || !(instance.explicit_side > 0.0))) {
        error = "instance explicit_side must be zero or finite and positive";
        return false;
    }
    if (!std::isfinite(instance.side) || !(instance.side > 0.0)
        || !std::isfinite(instance.min_x) || !std::isfinite(instance.min_y)
        || !std::isfinite(instance.max_x) || !std::isfinite(instance.max_y)) {
        error = "instance bounds must be finite and side must be positive";
        return false;
    }
    if (!validate_metric_domain(instance, error)) {
        return false;
    }
    if (instance.knn_k < 0 || instance.knn_k >= instance.N) {
        error = "instance knn_k must be in [0,N)";
        return false;
    }
    return validate_knn_rows(instance, error)
        && validate_grid_structure(instance, error)
        && validate_reverse_knn_structure(instance, error);
}

bool same_grid_and_knn(const Instance& source,
                       const Instance& rebuilt,
                       std::string& error) {
    if (!nearly_equal(source.side, rebuilt.side)
        || !nearly_equal(source.min_x, rebuilt.min_x)
        || !nearly_equal(source.min_y, rebuilt.min_y)
        || !nearly_equal(source.max_x, rebuilt.max_x)
        || !nearly_equal(source.max_y, rebuilt.max_y)) {
        error = "instance bound metadata is inconsistent with its coordinates";
        return false;
    }
    if (source.knn != rebuilt.knn || source.knn_d.size() != rebuilt.knn_d.size()) {
        error = "instance KNN nodes are not the exact canonical nearest neighbors";
        return false;
    }
    for (std::size_t i = 0; i < source.knn_d.size(); ++i) {
        if (!nearly_equal(source.knn_d[i], rebuilt.knn_d[i], 512.0)) {
            error = "instance KNN distances differ from a canonical rebuild";
            return false;
        }
    }
    if (source.gx != rebuilt.gx || source.gy != rebuilt.gy
        || !nearly_equal(source.cell_size, rebuilt.cell_size, 512.0)
        || !nearly_equal(source.grid_min_x, rebuilt.grid_min_x)
        || !nearly_equal(source.grid_min_y, rebuilt.grid_min_y)
        || !nearly_equal(source.grid_max_x, rebuilt.grid_max_x)
        || !nearly_equal(source.grid_max_y, rebuilt.grid_max_y)
        || source.cell_x != rebuilt.cell_x || source.cell_y != rebuilt.cell_y
        || source.cell_begin != rebuilt.cell_begin
        || source.cell_points != rebuilt.cell_points) {
        error = "instance grid metadata differs from a canonical rebuild";
        return false;
    }
    if (!source.rknn_begin.empty() || !source.rknn_nodes.empty()) {
        rebuilt.ensure_reverse_knn();
        if (source.rknn_begin != rebuilt.rknn_begin
            || source.rknn_nodes != rebuilt.rknn_nodes) {
            error = "instance reverse-KNN adjacency differs from its canonical form";
            return false;
        }
    }
    return true;
}

bool validate_solver_cross_options(const SolverOptions& solver, std::string& error) {
    if (solver.subset_restarts < 1 && solver.subset_restarts != -1) {
        error = "--restarts must be >= 1, or -1 for the automatic policy";
        return false;
    }
    if (solver.racing_candidates > 0
        && solver.racing_survivors > solver.racing_candidates) {
        error = "--racing-survivors must not exceed --racing-candidates";
        return false;
    }
    if (solver.racing_candidates > 0 && solver.time_budget_per_p > 0.0) {
        error = "deterministic restart racing is incompatible with --time-budget-per-p";
        return false;
    }
    if (!(solver.sa_initial_uphill_acceptance
          > solver.sa_final_uphill_acceptance)) {
        error = "--sa-initial-uphill-acceptance must exceed "
                "--sa-final-uphill-acceptance";
        return false;
    }
    if (solver.path_relink_top > 0
        && solver.path_relink_diverse_reserve > solver.path_relink_top) {
        error = "--path-relink-diverse-reserve must not exceed --path-relink-top";
        return false;
    }
    if (solver.oracle.cfg.max_k < solver.oracle.cfg.min_k) {
        error = "--oracle-max-k must be >= --oracle-min-k";
        return false;
    }
    return true;
}

bool validate_warm_start(const Instance& instance,
                         const int /*k*/,
                         const std::vector<int>* warm_start,
                         std::string& error) {
    if (warm_start == nullptr) {
        return true;
    }
    if (warm_start->size() < 3U
        || warm_start->size() > static_cast<std::size_t>(instance.N)) {
        error = "warm start cardinality must be in [3,N]";
        return false;
    }
    std::vector<unsigned char> seen(static_cast<std::size_t>(instance.N), 0U);
    for (const int node : *warm_start) {
        if (node < 0 || node >= instance.N) {
            error = "warm start contains an out-of-range node";
            return false;
        }
        if (seen[static_cast<std::size_t>(node)] != 0U) {
            error = "warm start contains a duplicate node";
            return false;
        }
        seen[static_cast<std::size_t>(node)] = 1U;
    }
    return true;
}

} // namespace

bool validate_solver_options(const SolverOptions& options, std::string& error) {
    RunOptions probe;
    probe.N = 3;
    probe.instances = 1;
    probe.threads = 1;
    probe.p_values.clear();
    probe.p_values.push_back(1.0);
    probe.solver = options;
    if (!validate_generated_option_ranges(probe, error)) {
        return false;
    }
    return validate_solver_cross_options(options, error);
}

namespace detail {

std::shared_ptr<const Instance> validate_and_canonicalize_instance(
    const Instance& instance,
    std::string& error) {
    if (!validate_instance_structure(instance, error)) {
        return {};
    }

    auto rebuilt = std::make_shared<Instance>();
    rebuilt->periodic = instance.periodic;
    rebuilt->explicit_side = instance.explicit_side;
    try {
        rebuilt->set_points(instance.points);
        if (instance.knn_k > 0) {
            rebuilt->build_knn(
                instance.knn_k,
                instance.knn_backend,
                instance.last_knn_build.requested_cell_size);
        }
    } catch (const std::exception& exception) {
        error = std::string("failed to canonically rebuild instance: ")
            + exception.what();
        return {};
    }
    if (!validate_instance_structure(*rebuilt, error)
        || !same_grid_and_knn(instance, *rebuilt, error)) {
        return {};
    }
    return rebuilt;
}

std::shared_ptr<const Instance> adopt_library_built_instance(
    Instance&& instance,
    std::string& error) {
    if (!validate_instance_structure(instance, error)) {
        return {};
    }
    return std::make_shared<const Instance>(std::move(instance));
}

} // namespace detail

bool validate_instance(const Instance& instance, std::string& error) {
    return static_cast<bool>(
        detail::validate_and_canonicalize_instance(instance, error));
}

bool validate_prepared_instance(const PreparedInstance& instance,
                                std::string& error) {
    if (!instance.valid()) {
        error = "prepared instance is empty or has been moved from";
        return false;
    }
    return true;
}

bool validate_subset_request(const Instance& instance,
                             const int k,
                             const SolverOptions& options,
                             const std::vector<int>* warm_start,
                             std::string& error) {
    if (!validate_instance(instance, error) || !validate_solver_options(options, error)) {
        return false;
    }
    if (k < 3 || k > instance.N) {
        error = "subset cardinality k must be in [3,N]";
        return false;
    }
    return validate_warm_start(instance, k, warm_start, error);
}

bool validate_subset_request(const PreparedInstance& instance,
                             const int k,
                             const SolverOptions& options,
                             const std::vector<int>* warm_start,
                             std::string& error) {
    const Instance& native = instance.instance();
    if (!validate_prepared_instance(instance, error)
        || !validate_solver_options(options, error)) {
        return false;
    }
    if (k < 3 || k > native.N) {
        error = "subset cardinality k must be in [3,N]";
        return false;
    }
    if (native.knn_k <= 0) {
        error = "prepared instance must contain a nonempty exact KNN table";
        return false;
    }
    return validate_warm_start(native, k, warm_start, error);
}

bool validate_tsp_request(const Instance& instance,
                          const SolverOptions& options,
                          std::string& error) {
    return validate_instance(instance, error) && validate_solver_options(options, error);
}

bool validate_tsp_request(const PreparedInstance& instance,
                          const SolverOptions& options,
                          std::string& error) {
    const Instance& native = instance.instance();
    if (!validate_prepared_instance(instance, error)
        || !validate_solver_options(options, error)) {
        return false;
    }
    if (native.knn_k <= 0) {
        error = "prepared instance must contain a nonempty exact KNN table";
        return false;
    }
    return true;
}

bool validate_lower_bound_request(const PreparedInstance& instance,
                                  const std::vector<int>& subset,
                                  const double upper_bound,
                                  const int max_iters,
                                  std::string& error) {
    if (!validate_prepared_instance(instance, error)) {
        return false;
    }
    const Instance& native = instance.instance();
    if (subset.size() < 3U || subset.size() > static_cast<std::size_t>(native.N)) {
        error = "lower-bound subset cardinality must be in [3,N]";
        return false;
    }
    if (max_iters <= 0) {
        error = "Held-Karp iteration count must be positive";
        return false;
    }
    if (std::isnan(upper_bound) || upper_bound < 0.0) {
        error = "Held-Karp upper bound must be nonnegative or positive infinity";
        return false;
    }
    std::vector<unsigned char> seen(static_cast<std::size_t>(native.N), 0U);
    for (const int node : subset) {
        if (node < 0 || node >= native.N) {
            error = "lower-bound subset contains an out-of-range node";
            return false;
        }
        if (seen[static_cast<std::size_t>(node)] != 0U) {
            error = "lower-bound subset contains a duplicate node";
            return false;
        }
        seen[static_cast<std::size_t>(node)] = 1U;
    }
    return true;
}

bool validate_solve_postconditions(const PreparedInstance& prepared,
                                   const int expected_k,
                                   const Tour& tour,
                                   std::string& error) {
    const Instance& instance = prepared.instance();
    if (expected_k < 3 || expected_k > instance.N) {
        error = "internal solve postcondition used an invalid expected cardinality";
        return false;
    }
    if (tour.N != instance.N || tour.k != expected_k
        || tour.nodes.size() != static_cast<std::size_t>(expected_k)) {
        error = "solver returned a tour with the wrong cardinality or instance size";
        return false;
    }
    if (!tour.check_invariants()) {
        error = "solver returned a tour with inconsistent membership/index state";
        return false;
    }
    std::vector<unsigned char> seen(static_cast<std::size_t>(instance.N), 0U);
    for (const int node : tour.nodes) {
        if (node < 0 || node >= instance.N) {
            error = "solver returned an out-of-range tour node";
            return false;
        }
        if (seen[static_cast<std::size_t>(node)] != 0U) {
            error = "solver returned a duplicate tour node";
            return false;
        }
        seen[static_cast<std::size_t>(node)] = 1U;
    }
    if (!std::isfinite(tour.length) || tour.length < 0.0) {
        error = "solver returned a nonfinite or negative tour length";
        return false;
    }
    const double recomputed = cycle_length(instance, tour.nodes);
    if (!std::isfinite(recomputed) || recomputed < 0.0
        || !nearly_equal(tour.length, recomputed, 4096.0)) {
        error = "solver tour length disagrees with a full metric recomputation";
        return false;
    }
    return true;
}

bool validate_run_options(RunOptions& options, std::string& error) {
    if (options.p_values.empty()) {
        options.p_values = default_p_values();
    }
    for (const double p : options.p_values) {
        if (!std::isfinite(p) || !(p > 0.0) || p > 1.0) {
            error = "every probability must be finite and in (0,1]";
            return false;
        }
    }
    std::sort(options.p_values.begin(), options.p_values.end());
    options.p_values.erase(
        std::unique(options.p_values.begin(), options.p_values.end()),
        options.p_values.end());

    if (!validate_generated_option_ranges(options, error)
        || !validate_solver_cross_options(options.solver, error)) {
        return false;
    }
    if (!printable_identity(options.campaign_id)) {
        error = "--campaign-id must contain 1..256 printable characters";
        return false;
    }
    if (!printable_identity(options.solver_policy_id)) {
        error = "--solver-policy-id must contain 1..256 printable characters";
        return false;
    }
    if (!printable_identity(options.fidelity_level)) {
        error = "--fidelity-level must contain 1..256 printable characters";
        return false;
    }
    if (options.control_variate
        && static_cast<long long>(options.cv_max_point_ops)
            < 2LL * static_cast<long long>(options.N)) {
        error = "--cv-max-point-ops must permit at least two N-point reference samples";
        return false;
    }
    if (options.periodic
        && options.solver.oracle.cfg.mode != ExternalOracleMode::None
        && options.solver.oracle.cfg.problem_format == OracleProblemFormat::Euc2d) {
        error = "--periodic requires --oracle-format matrix because euc2d ignores torus wrapping";
        return false;
    }

    if (options.solver.knn_k <= 0) {
        options.solver.knn_k = std::min(40, options.N - 1);
    }
    options.solver.knn_k = std::max(1, std::min(options.solver.knn_k, options.N - 1));

    const unsigned hardware = std::thread::hardware_concurrency();
    const int automatic_threads = hardware == 0U ? 1 : static_cast<int>(hardware);
    if (options.threads == 0) {
        options.threads = automatic_threads;
    }
    options.threads = std::max(1, std::min(options.threads, options.instances));
    if (options.solver.restart_threads == 0) {
        options.solver.restart_threads = std::max(
            1, automatic_threads / std::max(1, options.threads));
    }
    return true;
}

void require_valid_run_options(RunOptions& options) {
    std::string error;
    if (!validate_run_options(options, error)) {
        throw std::invalid_argument(error);
    }
}

void require_valid_instance(const Instance& instance) {
    std::string error;
    if (!validate_instance(instance, error)) {
        throw std::invalid_argument(error);
    }
}

void require_valid_subset_request(const Instance& instance,
                                  const int k,
                                  const SolverOptions& options,
                                  const std::vector<int>* warm_start) {
    std::string error;
    if (!validate_subset_request(instance, k, options, warm_start, error)) {
        throw std::invalid_argument(error);
    }
}

void require_valid_tsp_request(const Instance& instance,
                               const SolverOptions& options) {
    std::string error;
    if (!validate_tsp_request(instance, options, error)) {
        throw std::invalid_argument(error);
    }
}

void require_valid_subset_request(const PreparedInstance& instance,
                                  const int k,
                                  const SolverOptions& options,
                                  const std::vector<int>* warm_start) {
    std::string error;
    if (!validate_subset_request(instance, k, options, warm_start, error)) {
        throw std::invalid_argument(error);
    }
}

void require_valid_tsp_request(const PreparedInstance& instance,
                               const SolverOptions& options) {
    std::string error;
    if (!validate_tsp_request(instance, options, error)) {
        throw std::invalid_argument(error);
    }
}

void require_valid_lower_bound_request(const PreparedInstance& instance,
                                       const std::vector<int>& subset,
                                       const double upper_bound,
                                       const int max_iters) {
    std::string error;
    if (!validate_lower_bound_request(
            instance, subset, upper_bound, max_iters, error)) {
        throw std::invalid_argument(error);
    }
}

void require_valid_solve_postconditions(const PreparedInstance& instance,
                                        const int expected_k,
                                        const Tour& tour) {
    std::string error;
    if (!validate_solve_postconditions(instance, expected_k, tour, error)) {
        throw std::logic_error(error);
    }
}

} // namespace aldous_tsp
