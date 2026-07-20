#include "aldous_tsp/validation.hpp"

#include "aldous_tsp/instance.hpp"
#include "generated_options.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

namespace aldous_tsp {
namespace {

bool printable_identity(const std::string& value) {
    return !value.empty() && value.size() <= 256U
        && std::none_of(value.begin(), value.end(), [](const unsigned char ch) {
            return ch < 0x20U || ch == 0x7fU;
        });
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
    probe.p_values = {1.0};
    probe.solver = options;
    if (!validate_generated_option_ranges(probe, error)) {
        return false;
    }
    return validate_solver_cross_options(options, error);
}

bool validate_instance(const Instance& instance, std::string& error) {
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
    if (instance.periodic && (!std::isfinite(instance.side) || !(instance.side > 0.0))) {
        error = "periodic instance side must be finite and positive";
        return false;
    }
    if (instance.knn_k < 0 || instance.knn_k >= instance.N) {
        error = "instance knn_k must be in [0,N)";
        return false;
    }
    if (instance.knn_k > 0) {
        const std::size_t expected = static_cast<std::size_t>(instance.N)
            * static_cast<std::size_t>(instance.knn_k);
        if (instance.knn.size() != expected || instance.knn_d.size() != expected) {
            error = "instance KNN arrays are inconsistent with N and knn_k";
            return false;
        }
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

bool validate_tsp_request(const Instance& instance,
                          const SolverOptions& options,
                          std::string& error) {
    return validate_instance(instance, error) && validate_solver_options(options, error);
}

bool validate_run_options(RunOptions& options, std::string& error) {
    if (options.p_values.empty()) {
        options.p_values = default_p_values();
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

} // namespace aldous_tsp
