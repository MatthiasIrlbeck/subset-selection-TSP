#pragma once

#include "aldous_tsp/config.hpp"

#include <cstdint>

namespace aldous_tsp {

// Conservative planning estimate for one experiment invocation. The estimate
// intentionally includes persistent instance data, major solver scratch, elite
// storage, grid metadata, and exact-DP storage when enabled. It is a scheduling
// guard, not a measurement of resident-set size.
struct MemoryPlan {
    std::uint64_t budget_bytes = 0;
    std::uint64_t fixed_overhead_bytes = 0;
    std::uint64_t estimated_instance_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;
    // Raw CLI/library request; zero means automatic hardware-based selection.
    int requested_threads = 0;
    // Requested value after automatic/hardware and instance-count bounds.
    int resolved_threads = 1;
    // Actual instance concurrency after applying the memory budget.
    int effective_threads = 1;
    bool limited_by_budget = false;
    bool reverse_knn_enabled = true;
};

[[nodiscard]] bool solver_uses_reverse_knn(const SolverOptions& options) noexcept;
[[nodiscard]] std::uint64_t estimate_instance_memory_bytes(const RunOptions& options) noexcept;
[[nodiscard]] MemoryPlan estimate_experiment_memory(const RunOptions& options,
                                                     int requested_threads);

} // namespace aldous_tsp
