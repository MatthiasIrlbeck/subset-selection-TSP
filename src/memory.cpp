#include "aldous_tsp/memory.hpp"

#include "aldous_tsp/exact_subset.hpp"
#include "aldous_tsp/instance.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

namespace aldous_tsp {

struct ConcurrencyLimiter::Permit::State {
    explicit State(const int requested) : capacity(std::max(1, requested)) {}
    std::mutex mutex;
    std::condition_variable condition;
    int capacity = 1;
    int active = 0;
};

ConcurrencyLimiter::Permit::Permit(std::shared_ptr<State> state) noexcept
    : state_(std::move(state)) {}

ConcurrencyLimiter::Permit::Permit(Permit&& other) noexcept
    : state_(std::move(other.state_)) {}

ConcurrencyLimiter::Permit& ConcurrencyLimiter::Permit::operator=(Permit&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (state_ != nullptr) {
        {
            std::lock_guard<std::mutex> lock(state_->mutex);
            --state_->active;
        }
        state_->condition.notify_one();
    }
    state_ = std::move(other.state_);
    return *this;
}

ConcurrencyLimiter::Permit::~Permit() {
    if (state_ == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        --state_->active;
    }
    state_->condition.notify_one();
}

ConcurrencyLimiter::ConcurrencyLimiter(const int capacity)
    : state_(std::make_shared<Permit::State>(capacity)) {}

ConcurrencyLimiter::Permit ConcurrencyLimiter::acquire() const {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->condition.wait(lock, [&] { return state_->active < state_->capacity; });
    ++state_->active;
    return Permit(state_);
}

int ConcurrencyLimiter::capacity() const noexcept {
    return state_->capacity;
}

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

int cardinality_for(const RunOptions& options, const double p) noexcept {
    if (!std::isfinite(p) || options.N < 3) {
        return 0;
    }
    return std::max(3, std::min(
        options.N,
        static_cast<int>(std::llround(p * static_cast<double>(options.N)))));
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
        const int k = cardinality_for(options, p);
        if (k > 0) {
            maximum = std::max(
                maximum,
                estimate_exact_subset_memory(options.N, k).estimated_peak_bytes);
        }
    }
    return maximum;
}

int maximum_held_karp_k(const RunOptions& options) noexcept {
    if (!options.held_karp) {
        return 0;
    }
    int maximum = 0;
    for (const double p : options.p_values) {
        const int k = cardinality_for(options, p);
        if (k >= 3 && k <= 6000) {
            maximum = std::max(maximum, k);
        }
    }
    return maximum;
}

bool oracle_configured(const RunOptions& options) noexcept {
    return options.solver.oracle.resolved != ResolvedOracleMode::None
        || options.solver.oracle.cfg.mode != ExternalOracleMode::None;
}

int maximum_oracle_k(const RunOptions& options) noexcept {
    if (!oracle_configured(options)) {
        return 0;
    }
    int maximum = 0;
    for (const double p : options.p_values) {
        const int k = cardinality_for(options, p);
        if (k < options.solver.oracle.cfg.min_k
            || k > options.solver.oracle.cfg.max_k) {
            continue;
        }
        const bool full = k == options.N;
        if ((full && options.solver.oracle.cfg.use_for_tsp)
            || (!full && options.solver.oracle.cfg.use_for_subset)) {
            maximum = std::max(maximum, k);
        }
    }
    return maximum;
}

std::uint64_t linux_cgroup_available() noexcept {
#if defined(__linux__)
    auto parse_number = [](const char* path, std::uint64_t& value) {
        std::ifstream input(path);
        std::string text;
        if (!(input >> text) || text == "max") {
            return false;
        }
        try {
            value = static_cast<std::uint64_t>(std::stoull(text));
            return true;
        } catch (...) {
            return false;
        }
    };
    std::uint64_t limit = 0U;
    std::uint64_t current = 0U;
    if (parse_number("/sys/fs/cgroup/memory.max", limit)
        && parse_number("/sys/fs/cgroup/memory.current", current)
        && limit > current && limit < (1ULL << 62U)) {
        return limit - current;
    }
    if (parse_number("/sys/fs/cgroup/memory/memory.limit_in_bytes", limit)
        && parse_number("/sys/fs/cgroup/memory/memory.usage_in_bytes", current)
        && limit > current && limit < (1ULL << 62U)) {
        return limit - current;
    }
#endif
    return 0U;
}

} // namespace

bool solver_uses_reverse_knn(const SolverOptions& options) noexcept {
    return options.reverse_knn
        && (!options.disable_two_opt || !options.disable_or_opt);
}

std::uint64_t detect_available_memory_bytes() noexcept {
    std::uint64_t available = 0U;
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status) != 0) {
        available = static_cast<std::uint64_t>(status.ullAvailPhys);
    }
#elif defined(__APPLE__)
    std::uint64_t total = 0U;
    std::size_t size = sizeof(total);
    if (sysctlbyname("hw.memsize", &total, &size, nullptr, 0) == 0) {
        // The portable sysctl exposes total memory. Reserve a substantial share
        // for the OS and concurrent applications rather than treating it all as
        // allocatable solver memory.
        available = total / 2U;
    }
#elif defined(__linux__)
    std::ifstream input("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0U;
    std::string unit;
    while (input >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            available = saturating_multiply(value, 1024U);
            break;
        }
    }
    if (available == 0U) {
        const long pages = ::sysconf(_SC_AVPHYS_PAGES);
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (pages > 0 && page_size > 0) {
            available = saturating_multiply(
                static_cast<std::uint64_t>(pages),
                static_cast<std::uint64_t>(page_size));
        }
    }
#else
    const long pages = ::sysconf(_SC_AVPHYS_PAGES);
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        available = saturating_multiply(
            static_cast<std::uint64_t>(pages),
            static_cast<std::uint64_t>(page_size));
    }
#endif
    const std::uint64_t cgroup = linux_cgroup_available();
    if (cgroup > 0U) {
        available = available == 0U ? cgroup : std::min(available, cgroup);
    }
    return available;
}

std::uint64_t estimate_instance_memory_bytes(const RunOptions& options) noexcept {
    const auto n = static_cast<std::uint64_t>(std::max(options.N, 0));
    const auto k = static_cast<std::uint64_t>(
        std::max(0, std::min(options.solver.knn_k, std::max(options.N - 1, 0))));
    const auto nk = saturating_multiply(n, k);
    std::uint64_t bytes = std::uint64_t{1} << 20U;

    bytes = saturating_add(bytes, saturating_multiply(n, sizeof(Point)));
    bytes = saturating_add(bytes, saturating_multiply(nk, sizeof(int)));
    bytes = saturating_add(bytes, saturating_multiply(nk, sizeof(double)));
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

    const auto restart_workers = static_cast<std::uint64_t>(
        std::max(1, options.solver.restart_threads));
    bytes = saturating_add(
        bytes,
        saturating_multiply(
            saturating_multiply(n, 96U),
            restart_workers));
    bytes = saturating_add(bytes, saturating_multiply(n, 32U * sizeof(int)));
    bytes = saturating_add(bytes, exact_memory_bytes(options));
    return bytes;
}

std::uint64_t estimate_held_karp_memory_bytes(const int k) noexcept {
    if (k < 3 || k > 6000) {
        return 0U;
    }
    const std::uint64_t n = static_cast<std::uint64_t>(k);
    std::uint64_t bytes = saturating_multiply(
        saturating_multiply(n, n), sizeof(double));
    // Multipliers, keys, parents, degrees, membership flags, and allocator
    // reserve. The matrix dominates, but these terms matter near a hard limit.
    bytes = saturating_add(bytes, saturating_multiply(n, 4U * sizeof(double)));
    bytes = saturating_add(bytes, saturating_multiply(n, 4U * sizeof(int)));
    return saturating_add(bytes, 2U * 1024U * 1024U);
}

std::uint64_t estimate_oracle_memory_bytes(const int k) noexcept {
    if (k < 3) {
        return 0U;
    }
    const std::uint64_t n = static_cast<std::uint64_t>(k);
    // External solvers commonly retain several dense edge/candidate structures.
    // Reserve two doubles per potential edge plus a fixed child-process and I/O
    // allowance. This is deliberately conservative because child RSS is outside
    // the parent allocator and varies by oracle build.
    std::uint64_t bytes = saturating_multiply(
        saturating_multiply(n, n), 2U * sizeof(double));
    bytes = saturating_add(bytes, saturating_multiply(n, 512U));
    return saturating_add(bytes, 64U * 1024U * 1024U);
}

std::uint64_t estimate_control_reference_memory_bytes(const RunOptions& options) noexcept {
    if (!options.control_variate || options.N <= 0) {
        return 0U;
    }
    const std::uint64_t n = static_cast<std::uint64_t>(options.N);
    const std::uint64_t nk = saturating_multiply(n, 2U);
    std::uint64_t bytes = 1U * 1024U * 1024U;
    bytes = saturating_add(bytes, saturating_multiply(n, sizeof(Point)));
    bytes = saturating_add(bytes, saturating_multiply(nk, sizeof(int) + sizeof(double)));
    bytes = saturating_add(bytes, saturating_multiply(n, 3U * sizeof(int)));
    bytes = saturating_add(bytes, saturating_multiply(
        static_cast<std::uint64_t>(kMaxGridCells + 1), sizeof(int)));
    return bytes;
}

std::uint64_t estimate_result_storage_bytes(const RunOptions& options) noexcept {
    const std::uint64_t instances = static_cast<std::uint64_t>(
        std::max(options.instances, 0));
    const std::uint64_t p_count = static_cast<std::uint64_t>(
        std::max<std::size_t>(1U, options.p_values.size()));
    const int ordinary = options.solver.subset_restarts > 0
        ? options.solver.subset_restarts : 12;
    const int records = std::max(
        8,
        ordinary + options.solver.continuation_restarts
            + options.solver.racing_candidates + options.solver.tsp_candidate_starts
            + options.solver.subset_kick_restarts);
    const std::uint64_t per_cell = 512U
        + saturating_multiply(static_cast<std::uint64_t>(records), 128U);
    std::uint64_t bytes = saturating_multiply(
        saturating_multiply(instances, p_count), per_cell);
    if (options.include_instance_rows) {
        // Retained structured rows plus the final JSON string during
        // serialization. Use a second copy and a generous textual expansion.
        bytes = saturating_multiply(bytes, 3U);
    } else {
        bytes = saturating_multiply(bytes, 2U);
    }
    return saturating_add(bytes, 2U * 1024U * 1024U);
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
    plan.estimated_serialization_bytes = estimate_result_storage_bytes(options);
    plan.estimated_held_karp_call_bytes = estimate_held_karp_memory_bytes(
        maximum_held_karp_k(options));
    plan.estimated_oracle_call_bytes = estimate_oracle_memory_bytes(
        maximum_oracle_k(options));
    plan.estimated_control_reference_bytes =
        estimate_control_reference_memory_bytes(options);
    plan.detected_available_bytes = detect_available_memory_bytes();

    if (options.memory_budget_mb > 0) {
        plan.budget_bytes = saturating_multiply(
            static_cast<std::uint64_t>(options.memory_budget_mb),
            1024U * 1024U);
    } else if (plan.detected_available_bytes > 0U) {
        plan.automatic_budget = true;
        // Leave 25% of currently available memory untouched for the OS,
        // compiler/runtime libraries, and estimate error.
        plan.budget_bytes = plan.detected_available_bytes
            - plan.detected_available_bytes / 4U;
    }

    const auto base_for_threads = [&](const int threads) {
        return saturating_add(
            saturating_add(plan.fixed_overhead_bytes,
                           plan.estimated_serialization_bytes),
            saturating_multiply(
                plan.estimated_instance_bytes,
                static_cast<std::uint64_t>(threads)));
    };

    if (plan.budget_bytes > 0U) {
        int selected = 0;
        for (int threads = plan.resolved_threads; threads >= 1; --threads) {
            const std::uint64_t base = base_for_threads(threads);
            const std::uint64_t required = std::max({
                base,
                saturating_add(base, plan.estimated_held_karp_call_bytes),
                saturating_add(base, plan.estimated_oracle_call_bytes),
                saturating_add(
                    saturating_add(plan.fixed_overhead_bytes,
                                   plan.estimated_serialization_bytes),
                    plan.estimated_control_reference_bytes)});
            if (required <= plan.budget_bytes) {
                selected = threads;
                break;
            }
        }
        if (selected == 0) {
            throw std::invalid_argument(
                "the memory budget is below the conservative estimate for one active "
                "instance and its largest enabled analysis phase");
        }
        plan.effective_threads = selected;
        plan.limited_by_budget = selected < plan.resolved_threads;
    }

    const std::uint64_t ordinary_base = base_for_threads(plan.effective_threads);
    auto phase_concurrency = [&](const std::uint64_t per_call) {
        if (per_call == 0U) {
            return plan.effective_threads;
        }
        if (plan.budget_bytes == 0U) {
            return plan.effective_threads;
        }
        const std::uint64_t available = plan.budget_bytes > ordinary_base
            ? plan.budget_bytes - ordinary_base : 0U;
        const std::uint64_t possible = available / per_call;
        return std::max(1, std::min(
            plan.effective_threads,
            static_cast<int>(std::min<std::uint64_t>(
                possible, static_cast<std::uint64_t>(plan.effective_threads)))));
    };
    plan.held_karp_concurrency = phase_concurrency(
        plan.estimated_held_karp_call_bytes);
    plan.oracle_concurrency = phase_concurrency(plan.estimated_oracle_call_bytes);

    plan.ordinary_phase_peak_bytes = ordinary_base;
    plan.held_karp_phase_peak_bytes = saturating_add(
        ordinary_base,
        saturating_multiply(
            plan.estimated_held_karp_call_bytes,
            static_cast<std::uint64_t>(plan.held_karp_concurrency)));
    plan.oracle_phase_peak_bytes = saturating_add(
        ordinary_base,
        saturating_multiply(
            plan.estimated_oracle_call_bytes,
            static_cast<std::uint64_t>(plan.oracle_concurrency)));
    plan.control_reference_phase_peak_bytes = saturating_add(
        saturating_add(plan.fixed_overhead_bytes,
                       plan.estimated_serialization_bytes),
        plan.estimated_control_reference_bytes);
    plan.estimated_peak_bytes = std::max({
        plan.ordinary_phase_peak_bytes,
        plan.held_karp_phase_peak_bytes,
        plan.oracle_phase_peak_bytes,
        plan.control_reference_phase_peak_bytes});
    return plan;
}

} // namespace aldous_tsp
