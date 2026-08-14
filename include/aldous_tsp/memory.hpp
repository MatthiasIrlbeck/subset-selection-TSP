#pragma once

#include "aldous_tsp/config.hpp"

#include <cstdint>
#include <memory>

namespace aldous_tsp {

// Shared blocking limiter used for memory-heavy phases that may occur inside
// otherwise independent workers. Permits are move-only and release
// automatically, including on exceptions.
class ConcurrencyLimiter {
public:
    class Permit {
    public:
        Permit() noexcept = default;
        Permit(Permit&& other) noexcept;
        Permit& operator=(Permit&& other) noexcept;
        ~Permit();

        Permit(const Permit&) = delete;
        Permit& operator=(const Permit&) = delete;

    private:
        struct State;
        explicit Permit(std::shared_ptr<State> state) noexcept;
        std::shared_ptr<State> state_;
        friend class ConcurrencyLimiter;
    };

    explicit ConcurrencyLimiter(int capacity);
    [[nodiscard]] Permit acquire() const;
    [[nodiscard]] int capacity() const noexcept;

private:
    std::shared_ptr<Permit::State> state_;
};

// Conservative phase-aware planning estimate for one experiment invocation.
// It is a scheduling guard, not an RSS measurement. Each phase estimate
// includes retained result rows because completed workers remain resident until
// aggregation finishes.
struct MemoryPlan {
    std::uint64_t budget_bytes = 0;
    std::uint64_t detected_available_bytes = 0;
    bool automatic_budget = false;
    std::uint64_t fixed_overhead_bytes = 0;
    std::uint64_t estimated_instance_bytes = 0;
    std::uint64_t estimated_serialization_bytes = 0;
    std::uint64_t estimated_held_karp_call_bytes = 0;
    std::uint64_t estimated_oracle_call_bytes = 0;
    std::uint64_t estimated_control_reference_bytes = 0;
    std::uint64_t ordinary_phase_peak_bytes = 0;
    std::uint64_t held_karp_phase_peak_bytes = 0;
    std::uint64_t oracle_phase_peak_bytes = 0;
    std::uint64_t control_reference_phase_peak_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;
    // Raw CLI/library request; zero means automatic hardware-based selection.
    int requested_threads = 0;
    // Requested value after automatic/hardware and instance-count bounds.
    int resolved_threads = 1;
    // Actual instance concurrency after applying the memory budget.
    int effective_threads = 1;
    int held_karp_concurrency = 1;
    int oracle_concurrency = 1;
    bool limited_by_budget = false;
    bool reverse_knn_enabled = true;
};

[[nodiscard]] bool solver_uses_reverse_knn(const SolverOptions& options) noexcept;
[[nodiscard]] std::uint64_t detect_available_memory_bytes() noexcept;
[[nodiscard]] std::uint64_t estimate_instance_memory_bytes(const RunOptions& options) noexcept;
[[nodiscard]] std::uint64_t estimate_held_karp_memory_bytes(int k) noexcept;
[[nodiscard]] std::uint64_t estimate_oracle_memory_bytes(int k) noexcept;
[[nodiscard]] std::uint64_t estimate_control_reference_memory_bytes(const RunOptions& options) noexcept;
[[nodiscard]] std::uint64_t estimate_result_storage_bytes(const RunOptions& options) noexcept;
[[nodiscard]] MemoryPlan estimate_experiment_memory(const RunOptions& options,
                                                     int requested_threads);

} // namespace aldous_tsp
