#include "aldous_tsp/memory.hpp"

#include "aldous_tsp/exact_subset.hpp"
#include "aldous_tsp/instance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <thread>

namespace aldous_tsp {
namespace {

std::uint64_t saturating_add(const std::uint64_t lhs,
                             const std::uint64_t rhs) noexcept {
    const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    return lhs > maximum - rhs ? maximum : lhs + rhs;
}

std::uint64_t saturating_multiply(const std::uint64_t lhs,
                                  const std::uint64_t rhs) noexcept {
    if (lhs == 0U || rhs == 0U) {
        return 0U;
    }
    const std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    return lhs > maximum / rhs ? maximum : lhs * rhs;
}

int resolve_thread_count(const int requested, const int instances) noexcept {
    const unsigned hardware = std::thread::hardware_concurrency();
    int threads = requested;
    if (threads <= 0) {
        threads = hardware == 0U ? 1 : static_cast<int>(hardware);
    }
    return std::max(1, std::min(threads, std::max(1, instances)));
}

std::uint64_t exact_memory_bytes(const RunOptions& options) noexcept {
    if (options.solver.exact_subset_max_n <= 0
        || options.N > options.solver.exact_subset_max_n
        || options.N > kExactSubsetHardLimit
        || options.N < 0) {
        return 0U;
    }
    std::uint64_t maximum = 0U;
    if (options.p_values.empty()) {
        return estimate_exact_subset_memory(options.N, options.N)
            .estimated_peak_bytes;
    }
    for (const double p : options.p_values) {
        if (!std::isfinite(p)) {
            continue;
        }
        const int k = std::max(
            0,
            std::min(
                options.N,
                std::max(3, static_cast<int>(std::llround(
                    p * static_cast<double>(options.N))))));
        maximum = std::max(
            maximum,
            estimate_exact_subset_memory(options.N, k).estimated_peak_bytes);
    }
    return maximum;
}

} // namespace

bool solver_uses_reverse_knn(const SolverOptions& options) noexcept {
    return options.reverse_knn
        && (!options.disable_two_opt || !options.disable_or_opt);
}

std::uint64_t estimate_instance_memory_bytes(const RunOptions& options) noexcept {
    const auto n = static_cast<std::uint64_t>(std::max(options.N, 0));
    const auto k = static_cast<std::uint64_t>(
        std::max(0, std::min(options.solver.knn_k, std::max(options.N - 1, 0))));
    const auto nk = saturating_multiply(n, k);
    std::uint64_t bytes = 1U << 20U; // allocator/container and worker reserve

    bytes = saturating_add(bytes, saturating_multiply(n, sizeof(Point)));
    // One persistent exact-KNN node array and one distance array. Squared
    // distances are derived on demand and deliberately absent here.
    bytes = saturating_add(bytes, saturating_multiply(nk, sizeof(int)));
    bytes = saturating_add(bytes, saturating_multiply(nk, sizeof(double)));

    // Grid cell coordinates, point permutation, and capped cell-prefix table.
    bytes = saturating_add(bytes, saturating_multiply(n, 3U * sizeof(int)));
    bytes = saturating_add(
        bytes,
        saturating_multiply(
            static_cast<std::uint64_t>(kMaxGridCells + 1),
            sizeof(int)));

    if (solver_uses_reverse_knn(options.solver)) {
        bytes = saturating_add(bytes, saturating_multiply(nk, sizeof(int)));
        bytes = saturating_add(bytes, saturating_multiply(n + 1U, sizeof(int)));
    }

    // Major per-restart Tour state, candidate buffers, and transient local-
    // search arrays. Deliberately conservative so budgeting fails safe.
    const auto restart_workers = static_cast<std::uint64_t>(
        std::max(1, options.solver.restart_threads));
    bytes = saturating_add(
        bytes,
        saturating_multiply(
            saturating_multiply(n, 96U),
            restart_workers));

    // Quality and supplemental elite archives can retain several full cycles.
    bytes = saturating_add(bytes, saturating_multiply(n, 32U * sizeof(int)));
    bytes = saturating_add(bytes, exact_memory_bytes(options));
    return bytes;
}

MemoryPlan estimate_experiment_memory(const RunOptions& options,
                                      const int requested_threads) {
    MemoryPlan plan;
    plan.requested_threads = requested_threads;
    plan.resolved_threads = resolve_thread_count(requested_threads, options.instances);
    plan.effective_threads = plan.resolved_threads;
    plan.reverse_knn_enabled = solver_uses_reverse_knn(options.solver);
    plan.fixed_overhead_bytes = 16U * 1024U * 1024U;
    plan.estimated_instance_bytes = estimate_instance_memory_bytes(options);

    if (options.memory_budget_mb > 0) {
        plan.budget_bytes = saturating_multiply(
            static_cast<std::uint64_t>(options.memory_budget_mb),
            1024U * 1024U);
        const std::uint64_t one_instance_peak = saturating_add(
            plan.fixed_overhead_bytes,
            plan.estimated_instance_bytes);
        if (plan.budget_bytes < one_instance_peak) {
            throw std::invalid_argument(
                "--memory-budget-mb is below the conservative estimate for one active instance");
        }
        const std::uint64_t available = plan.budget_bytes - plan.fixed_overhead_bytes;
        const std::uint64_t possible = plan.estimated_instance_bytes == 0U
            ? static_cast<std::uint64_t>(plan.resolved_threads)
            : available / plan.estimated_instance_bytes;
        const int bounded = static_cast<int>(std::min<std::uint64_t>(
            possible,
            static_cast<std::uint64_t>(plan.resolved_threads)));
        plan.effective_threads = std::max(1, bounded);
        plan.limited_by_budget = plan.effective_threads < plan.resolved_threads;
    }
    plan.estimated_peak_bytes = saturating_add(
        plan.fixed_overhead_bytes,
        saturating_multiply(
            plan.estimated_instance_bytes,
            static_cast<std::uint64_t>(plan.effective_threads)));
    return plan;
}

} // namespace aldous_tsp
