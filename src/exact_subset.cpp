#include "aldous_tsp/exact_subset.hpp"

#include "aldous_tsp/config.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace aldous_tsp {
namespace {

int least_set_bit(std::uint32_t value) noexcept {
    int bit = 0;
    while ((value & 1U) == 0U) {
        value >>= 1U;
        ++bit;
    }
    return bit;
}

bool better_path(double candidate,
                 int candidate_predecessor,
                 double incumbent,
                 int incumbent_predecessor) noexcept {
    return candidate < incumbent
        || (candidate == incumbent
            && (incumbent_predecessor < 0
                || candidate_predecessor < incumbent_predecessor));
}

bool better_cycle(double candidate,
                  std::uint32_t candidate_mask,
                  int candidate_endpoint,
                  double incumbent,
                  std::uint32_t incumbent_mask,
                  int incumbent_endpoint) noexcept {
    if (candidate != incumbent) {
        return candidate < incumbent;
    }
    if (candidate_mask != incumbent_mask) {
        return candidate_mask < incumbent_mask;
    }
    return incumbent_endpoint < 0 || candidate_endpoint < incumbent_endpoint;
}

} // namespace

ExactSubsetSolution exact_subset_cycle(const Instance& inst, const int k) {
    ExactSubsetSolution result;
    result.n = inst.N;
    result.k = k;

    if (inst.N < 0 || inst.points.size() != static_cast<std::size_t>(inst.N)) {
        throw std::invalid_argument("exact subset oracle requires a consistent instance");
    }
    if (k < 0 || k > inst.N) {
        throw std::invalid_argument("exact subset cardinality must be in [0,N]");
    }
    if (inst.N > kExactSubsetHardLimit) {
        return result;
    }

    if (k == 0) {
        result.solved = true;
        result.proven_optimal = true;
        result.length = 0.0;
        return result;
    }
    if (k == 1) {
        result.solved = true;
        result.proven_optimal = true;
        result.length = 0.0;
        result.cycle = {0};
        result.states = static_cast<std::uint64_t>(inst.N);
        return result;
    }
    if (k == 2) {
        double best_length = std::numeric_limits<double>::infinity();
        std::uint32_t best_mask = 0U;
        int best_first = -1;
        int best_second = -1;
        result.states = static_cast<std::uint64_t>(inst.N);
        for (int first = 0; first < inst.N; ++first) {
            for (int second = first + 1; second < inst.N; ++second) {
                const double edge = inst.dist(first, second);
                if (!std::isfinite(edge)) {
                    throw std::invalid_argument(
                        "exact subset oracle requires finite pairwise distances");
                }
                const double candidate = 2.0 * edge;
                const std::uint32_t mask =
                    (1U << static_cast<unsigned>(first))
                    | (1U << static_cast<unsigned>(second));
                ++result.states;
                ++result.transitions;
                if (better_cycle(candidate, mask, second,
                                 best_length, best_mask, best_second)) {
                    best_length = candidate;
                    best_mask = mask;
                    best_first = first;
                    best_second = second;
                }
            }
        }
        if (best_first >= 0) {
            result.solved = true;
            result.proven_optimal = true;
            result.length = best_length;
            result.cycle = {best_first, best_second};
        }
        return result;
    }

    static_assert(kExactSubsetHardLimit < 31,
                  "exact subset masks require a 32-bit unsigned integer");
    static_assert(kExactSubsetHardLimit <=
                      static_cast<int>(std::numeric_limits<std::int8_t>::max()),
                  "exact subset parents must fit in int8_t");

    const int n = inst.N;
    const std::uint32_t total_masks = 1U << static_cast<unsigned>(n);
    const std::size_t stride = static_cast<std::size_t>(n);
    const double infinity = std::numeric_limits<double>::infinity();

    std::vector<double> distances(stride * stride, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double distance = inst.dist(i, j);
            if (!std::isfinite(distance)) {
                throw std::invalid_argument(
                    "exact subset oracle requires finite pairwise distances");
            }
            distances[static_cast<std::size_t>(i) * stride
                      + static_cast<std::size_t>(j)] = distance;
            distances[static_cast<std::size_t>(j) * stride
                      + static_cast<std::size_t>(i)] = distance;
        }
    }

    std::vector<std::uint8_t> cardinality(static_cast<std::size_t>(total_masks), 0U);
    for (std::uint32_t mask = 1U; mask < total_masks; ++mask) {
        cardinality[static_cast<std::size_t>(mask)] = static_cast<std::uint8_t>(
            cardinality[static_cast<std::size_t>(mask >> 1U)]
            + static_cast<std::uint8_t>(mask & 1U));
    }

    const std::size_t state_count = static_cast<std::size_t>(total_masks) * stride;
    std::vector<double> dp(state_count, infinity);
    std::vector<std::int8_t> parent(state_count, static_cast<std::int8_t>(-1));
    const auto state_index = [stride](const std::uint32_t mask, const int endpoint) {
        return static_cast<std::size_t>(mask) * stride
            + static_cast<std::size_t>(endpoint);
    };

    double best_length = infinity;
    std::uint32_t best_mask = 0U;
    int best_endpoint = -1;

    for (std::uint32_t mask = 1U; mask < total_masks; ++mask) {
        const int count = static_cast<int>(cardinality[static_cast<std::size_t>(mask)]);
        if (count > k) {
            continue;
        }
        const int anchor = least_set_bit(mask);
        if (count == 1) {
            dp[state_index(mask, anchor)] = 0.0;
            ++result.states;
            if (k == 1
                && better_cycle(0.0, mask, anchor,
                                best_length, best_mask, best_endpoint)) {
                best_length = 0.0;
                best_mask = mask;
                best_endpoint = anchor;
            }
            continue;
        }

        std::uint32_t endpoints = mask & ~(1U << static_cast<unsigned>(anchor));
        while (endpoints != 0U) {
            const int endpoint = least_set_bit(endpoints);
            endpoints &= endpoints - 1U;
            const std::uint32_t previous_mask =
                mask ^ (1U << static_cast<unsigned>(endpoint));

            double best_path = infinity;
            int best_predecessor = -1;
            std::uint32_t predecessors = previous_mask;
            while (predecessors != 0U) {
                const int predecessor = least_set_bit(predecessors);
                predecessors &= predecessors - 1U;
                const double prefix = dp[state_index(previous_mask, predecessor)];
                if (!std::isfinite(prefix)) {
                    continue;
                }
                ++result.transitions;
                const double candidate = prefix
                    + distances[static_cast<std::size_t>(predecessor) * stride
                                + static_cast<std::size_t>(endpoint)];
                if (better_path(candidate, predecessor,
                                best_path, best_predecessor)) {
                    best_path = candidate;
                    best_predecessor = predecessor;
                }
            }

            if (best_predecessor < 0) {
                continue;
            }
            const std::size_t index = state_index(mask, endpoint);
            dp[index] = best_path;
            parent[index] = static_cast<std::int8_t>(best_predecessor);
            ++result.states;

            if (count == k) {
                const double cycle_length = best_path
                    + distances[static_cast<std::size_t>(endpoint) * stride
                                + static_cast<std::size_t>(anchor)];
                if (better_cycle(cycle_length, mask, endpoint,
                                 best_length, best_mask, best_endpoint)) {
                    best_length = cycle_length;
                    best_mask = mask;
                    best_endpoint = endpoint;
                }
            }
        }
    }

    if (best_endpoint < 0 || best_mask == 0U || !std::isfinite(best_length)) {
        return result;
    }

    result.cycle.assign(static_cast<std::size_t>(k), -1);
    const int anchor = least_set_bit(best_mask);
    result.cycle[0] = anchor;
    if (k > 1) {
        std::uint32_t mask = best_mask;
        int current = best_endpoint;
        for (int position = k - 1; position >= 1; --position) {
            result.cycle[static_cast<std::size_t>(position)] = current;
            const int predecessor = static_cast<int>(parent[state_index(mask, current)]);
            mask ^= 1U << static_cast<unsigned>(current);
            current = predecessor;
        }
        if (current != anchor) {
            throw std::logic_error("exact subset oracle parent reconstruction failed");
        }
    }

    result.length = best_length;
    result.solved = true;
    result.proven_optimal = true;
    return result;
}

} // namespace aldous_tsp
