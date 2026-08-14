#pragma once

#include "aldous_tsp/instance.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace aldous_tsp {

// Conservative element-storage estimate for the exact cardinality-k dynamic
// program on this build. It includes the distance matrix, cardinality-indexed
// mask lists, the mask-to-row index, rolling value layers, and reconstruction
// parents. Container/allocator bookkeeping is included where it is known at
// compile time, but callers should retain ordinary process-memory headroom.
struct ExactSubsetMemoryEstimate {
    bool supported = false;
    int n = 0;
    int k = 0;
    std::uint64_t distance_bytes = 0;
    std::uint64_t mask_index_bytes = 0;
    std::uint64_t mask_storage_bytes = 0;
    std::uint64_t rolling_value_bytes = 0;
    std::uint64_t parent_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;
};

// Exact cardinality-k Euclidean/periodic subset-tour result. The dynamic
// program simultaneously chooses the subset and its optimal cycle, so a solved
// result is a global proof for min_{|S|=k} TSP(S), unlike lower bounds computed
// only on a heuristic's selected subset.
struct ExactSubsetSolution {
    bool solved = false;
    bool proven_optimal = false;
    int n = 0;
    int k = 0;
    double length = std::numeric_limits<double>::infinity();
    std::vector<int> cycle;
    std::uint64_t states = 0;
    std::uint64_t transitions = 0;
    std::uint64_t estimated_peak_memory_bytes = 0;
};

// Returns the exact oracle's cardinality-sensitive peak working-storage
// estimate. Invalid (n,k) pairs and instances above the hard limit report
// supported=false without performing exponential allocation.
[[nodiscard]] ExactSubsetMemoryEstimate estimate_exact_subset_memory(int n,
                                                                      int k) noexcept;

// Solves min_{|S|=k} TSP(S) exactly when inst.N <= kExactSubsetHardLimit.
// Invalid cardinalities throw std::invalid_argument. Larger instances return
// solved=false without allocating the exponential dynamic-programming tables.
[[nodiscard]] ExactSubsetSolution exact_subset_cycle(const Instance& inst, int k);

} // namespace aldous_tsp
