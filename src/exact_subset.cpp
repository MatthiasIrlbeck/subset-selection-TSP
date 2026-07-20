#include "aldous_tsp/exact_subset.hpp"

#include "aldous_tsp/config.hpp"

#include <algorithm>
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

int popcount(std::uint32_t value) noexcept {
    int count = 0;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
}

std::uint64_t saturating_add(const std::uint64_t lhs,
                             const std::uint64_t rhs) noexcept {
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    return lhs > maximum - rhs ? maximum : lhs + rhs;
}

std::uint64_t saturating_multiply(const std::uint64_t lhs,
                                  const std::uint64_t rhs) noexcept {
    if (lhs == 0U || rhs == 0U) {
        return 0U;
    }
    const auto maximum = std::numeric_limits<std::uint64_t>::max();
    return lhs > maximum / rhs ? maximum : lhs * rhs;
}

std::uint64_t binomial(const int n, const int k) noexcept {
    if (k < 0 || k > n) {
        return 0U;
    }
    const int smaller = std::min(k, n - k);
    std::uint64_t value = 1U;
    for (int i = 1; i <= smaller; ++i) {
        value = value * static_cast<std::uint64_t>(n - smaller + i)
            / static_cast<std::uint64_t>(i);
    }
    return value;
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

std::size_t state_index(const std::size_t row,
                        const int endpoint,
                        const std::size_t stride) noexcept {
    return row * stride + static_cast<std::size_t>(endpoint);
}

} // namespace

ExactSubsetMemoryEstimate estimate_exact_subset_memory(const int n,
                                                        const int k) noexcept {
    ExactSubsetMemoryEstimate estimate;
    estimate.n = n;
    estimate.k = k;
    if (n < 0 || k < 0 || k > n || n > kExactSubsetHardLimit) {
        return estimate;
    }
    estimate.supported = true;
    if (k <= 2) {
        return estimate;
    }

    const auto nodes = static_cast<std::uint64_t>(n);
    const auto total_masks = std::uint64_t{1} << static_cast<unsigned>(n);
    estimate.distance_bytes = saturating_multiply(
        saturating_multiply(nodes, nodes), sizeof(double));
    estimate.mask_index_bytes = saturating_multiply(total_masks, sizeof(std::int32_t));

    std::uint64_t mask_count = 0U;
    std::uint64_t cumulative_parent_bytes = 0U;
    std::uint64_t maximum_dynamic_bytes = 0U;
    for (int count = 1; count <= k; ++count) {
        const std::uint64_t current_masks = binomial(n, count);
        mask_count = saturating_add(mask_count, current_masks);
        if (count >= 2) {
            cumulative_parent_bytes = saturating_add(
                cumulative_parent_bytes,
                saturating_multiply(
                    saturating_multiply(current_masks, nodes),
                    sizeof(std::int8_t)));
        }
        const std::uint64_t previous_masks = count == 1
            ? 0U
            : binomial(n, count - 1);
        const std::uint64_t rolling_values = saturating_multiply(
            saturating_multiply(
                saturating_add(previous_masks, current_masks),
                nodes),
            sizeof(double));
        maximum_dynamic_bytes = std::max(
            maximum_dynamic_bytes,
            saturating_add(rolling_values, cumulative_parent_bytes));
    }

    estimate.mask_storage_bytes = saturating_add(
        saturating_multiply(mask_count, sizeof(std::uint32_t)),
        saturating_multiply(
            static_cast<std::uint64_t>(k + 1),
            sizeof(std::vector<std::uint32_t>)
                + sizeof(std::vector<std::int8_t>)));
    estimate.parent_bytes = cumulative_parent_bytes;

    std::uint64_t maximum_rolling = 0U;
    for (int count = 1; count <= k; ++count) {
        const std::uint64_t current_masks = binomial(n, count);
        const std::uint64_t previous_masks = count == 1
            ? 0U
            : binomial(n, count - 1);
        maximum_rolling = std::max(
            maximum_rolling,
            saturating_multiply(
                saturating_multiply(
                    saturating_add(previous_masks, current_masks),
                    nodes),
                sizeof(double)));
    }
    estimate.rolling_value_bytes = maximum_rolling;
    estimate.estimated_peak_bytes = saturating_add(
        saturating_add(estimate.distance_bytes, estimate.mask_index_bytes),
        saturating_add(estimate.mask_storage_bytes, maximum_dynamic_bytes));
    return estimate;
}

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
    const ExactSubsetMemoryEstimate memory =
        estimate_exact_subset_memory(inst.N, k);
    result.estimated_peak_memory_bytes = memory.estimated_peak_bytes;
    if (!memory.supported) {
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

    std::vector<std::vector<std::uint32_t>> masks_by_count(
        static_cast<std::size_t>(k + 1));
    for (int count = 1; count <= k; ++count) {
        masks_by_count[static_cast<std::size_t>(count)].reserve(
            static_cast<std::size_t>(binomial(n, count)));
    }
    for (std::uint32_t mask = 1U; mask < total_masks; ++mask) {
        const int count = popcount(mask);
        if (count <= k) {
            masks_by_count[static_cast<std::size_t>(count)].push_back(mask);
        }
    }

    std::vector<std::vector<std::int8_t>> parent_by_count(
        static_cast<std::size_t>(k + 1));
    std::vector<std::int32_t> previous_rows(
        static_cast<std::size_t>(total_masks), -1);
    std::vector<double> previous_values;

    double best_length = infinity;
    std::uint32_t best_mask = 0U;
    int best_endpoint = -1;

    for (int count = 1; count <= k; ++count) {
        const auto& current_masks = masks_by_count[static_cast<std::size_t>(count)];
        std::vector<double> current_values(current_masks.size() * stride, infinity);
        std::vector<std::int8_t> current_parents;
        if (count >= 2) {
            current_parents.assign(
                current_masks.size() * stride,
                static_cast<std::int8_t>(-1));
            const auto& previous_masks =
                masks_by_count[static_cast<std::size_t>(count - 1)];
            for (std::size_t row = 0; row < previous_masks.size(); ++row) {
                previous_rows[static_cast<std::size_t>(previous_masks[row])] =
                    static_cast<std::int32_t>(row);
            }
        }

        for (std::size_t row = 0; row < current_masks.size(); ++row) {
            const std::uint32_t mask = current_masks[row];
            const int anchor = least_set_bit(mask);
            if (count == 1) {
                current_values[state_index(row, anchor, stride)] = 0.0;
                ++result.states;
                continue;
            }

            std::uint32_t endpoints =
                mask & ~(1U << static_cast<unsigned>(anchor));
            while (endpoints != 0U) {
                const int endpoint = least_set_bit(endpoints);
                endpoints &= endpoints - 1U;
                const std::uint32_t previous_mask =
                    mask ^ (1U << static_cast<unsigned>(endpoint));
                const std::int32_t previous_row =
                    previous_rows[static_cast<std::size_t>(previous_mask)];
                if (previous_row < 0) {
                    throw std::logic_error(
                        "exact subset oracle cardinality index is inconsistent");
                }

                double best_path = infinity;
                int best_predecessor = -1;
                std::uint32_t predecessors = previous_mask;
                while (predecessors != 0U) {
                    const int predecessor = least_set_bit(predecessors);
                    predecessors &= predecessors - 1U;
                    const double prefix = previous_values[state_index(
                        static_cast<std::size_t>(previous_row),
                        predecessor,
                        stride)];
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
                const std::size_t index = state_index(row, endpoint, stride);
                current_values[index] = best_path;
                current_parents[index] =
                    static_cast<std::int8_t>(best_predecessor);
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

        if (count >= 2) {
            const auto& previous_masks =
                masks_by_count[static_cast<std::size_t>(count - 1)];
            for (const std::uint32_t mask : previous_masks) {
                previous_rows[static_cast<std::size_t>(mask)] = -1;
            }
            parent_by_count[static_cast<std::size_t>(count)] =
                std::move(current_parents);
        }
        previous_values = std::move(current_values);
    }

    if (best_endpoint < 0 || best_mask == 0U || !std::isfinite(best_length)) {
        return result;
    }

    result.cycle.assign(static_cast<std::size_t>(k), -1);
    const int anchor = least_set_bit(best_mask);
    result.cycle[0] = anchor;
    std::uint32_t mask = best_mask;
    int current = best_endpoint;
    for (int position = k - 1; position >= 1; --position) {
        result.cycle[static_cast<std::size_t>(position)] = current;
        const int count = position + 1;
        const auto& masks = masks_by_count[static_cast<std::size_t>(count)];
        const auto found = std::lower_bound(masks.begin(), masks.end(), mask);
        if (found == masks.end() || *found != mask) {
            throw std::logic_error(
                "exact subset oracle parent mask reconstruction failed");
        }
        const std::size_t row = static_cast<std::size_t>(found - masks.begin());
        const std::int8_t predecessor =
            parent_by_count[static_cast<std::size_t>(count)]
                           [state_index(row, current, stride)];
        if (predecessor < 0) {
            throw std::logic_error(
                "exact subset oracle parent reconstruction failed");
        }
        mask ^= 1U << static_cast<unsigned>(current);
        current = static_cast<int>(predecessor);
    }
    if (current != anchor) {
        throw std::logic_error("exact subset oracle parent reconstruction failed");
    }

    result.length = best_length;
    result.solved = true;
    result.proven_optimal = true;
    return result;
}

} // namespace aldous_tsp
