#pragma once

#include "aldous_tsp/instance.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace aldous_tsp {

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
};

// Solves min_{|S|=k} TSP(S) exactly when inst.N <= kExactSubsetHardLimit.
// Invalid cardinalities throw std::invalid_argument. Larger instances return
// solved=false without allocating the exponential dynamic-programming table.
[[nodiscard]] ExactSubsetSolution exact_subset_cycle(const Instance& inst, int k);

} // namespace aldous_tsp
