#include "solver_internal.hpp"

namespace aldous_tsp {

int least_set_bit_index(std::uint32_t value) noexcept {
    int index = 0;
    while ((value & 1U) == 0U) {
        value >>= 1U;
        ++index;
    }
    return index;
}

std::vector<int> nearest_neighbor_order(const Instance& inst, const std::vector<int>& subset, int start_index) {
    const int k = static_cast<int>(subset.size());
    if (k <= 1) {
        return subset;
    }
    start_index = std::max(0, std::min(start_index, k - 1));
    std::vector<int> out;
    out.reserve(subset.size());
    std::vector<unsigned char> used(static_cast<std::size_t>(k), 0U);
    out.push_back(subset[static_cast<std::size_t>(start_index)]);
    used[static_cast<std::size_t>(start_index)] = 1U;
    for (int step = 1; step < k; ++step) {
        const int last = out.back();
        int best = -1;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (int i = 0; i < k; ++i) {
            if (used[static_cast<std::size_t>(i)] != 0U) { continue; }
            const double d2 = inst.dist2(last, subset[static_cast<std::size_t>(i)]);
            if (!std::isfinite(d2) || d2 < 0.0) {
                throw std::domain_error(
                    "nearest_neighbor_order encountered a nonfinite edge cost");
            }
            if (d2 < best_d2
                || (d2 == best_d2
                    && (best < 0
                        || subset[static_cast<std::size_t>(i)]
                            < subset[static_cast<std::size_t>(best)]))) {
                best_d2 = d2;
                best = i;
            }
        }
        if (best < 0) {
            throw std::logic_error(
                "nearest_neighbor_order could not find an unvisited node");
        }
        out.push_back(subset[static_cast<std::size_t>(best)]);
        used[static_cast<std::size_t>(best)] = 1U;
    }
    return out;
}

std::vector<int> nearest_neighbor_full_order(const Instance& inst, int start_node) {
    if (inst.N <= 0) {
        return {};
    }
    start_node = std::max(0, std::min(start_node, inst.N - 1));
    std::vector<int> out;
    out.reserve(static_cast<std::size_t>(inst.N));
    std::vector<unsigned char> used(static_cast<std::size_t>(inst.N), 0U);
    out.push_back(start_node);
    used[static_cast<std::size_t>(start_node)] = 1U;

    for (int step = 1; step < inst.N; ++step) {
        const int last = out.back();
        int best = -1;
        double best_d2 = std::numeric_limits<double>::infinity();

        // Exact fast path: KNN rows are sorted by (distance, node id). If one
        // of the K nearest points is unvisited, the first such point is the
        // globally nearest unvisited point. A complete scan is needed only
        // after the entire retained KNN row has already been consumed.
        for (int rank = 0; rank < inst.knn_k; ++rank) {
            const int node = inst.knn_at(last, rank);
            if (node < 0 || node >= inst.N
                || used[static_cast<std::size_t>(node)] != 0U) {
                continue;
            }
            best = node;
            best_d2 = inst.dist2(last, node);
            break;
        }

        if (best < 0) {
            for (int node = 0; node < inst.N; ++node) {
                if (used[static_cast<std::size_t>(node)] != 0U) {
                    continue;
                }
                const double d2 = inst.dist2(last, node);
                if (!std::isfinite(d2) || d2 < 0.0) {
                    throw std::domain_error(
                        "nearest_neighbor_full_order encountered a nonfinite edge cost");
                }
                if (d2 < best_d2
                    || (d2 == best_d2 && (best < 0 || node < best))) {
                    best = node;
                    best_d2 = d2;
                }
            }
        }
        if (best < 0) {
            throw std::logic_error(
                "nearest_neighbor_full_order could not find an unvisited node");
        }
        out.push_back(best);
        used[static_cast<std::size_t>(best)] = 1U;
    }
    return out;
}

std::vector<int> farthest_insertion_order(const Instance& inst, const std::vector<int>& subset) {
    const int k = static_cast<int>(subset.size());
    if (k <= 3) {
        return subset;
    }
    int ai = 0;
    int bi = 1;
    double best_d2 = -1.0;
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            const double d2 = inst.dist2(subset[static_cast<std::size_t>(i)], subset[static_cast<std::size_t>(j)]);
            if (d2 > best_d2) {
                best_d2 = d2;
                ai = i;
                bi = j;
            }
        }
    }
    std::vector<int> cycle = {subset[static_cast<std::size_t>(ai)], subset[static_cast<std::size_t>(bi)]};
    std::vector<unsigned char> in_cycle(static_cast<std::size_t>(k), 0U);
    in_cycle[static_cast<std::size_t>(ai)] = 1U;
    in_cycle[static_cast<std::size_t>(bi)] = 1U;
    while (static_cast<int>(cycle.size()) < k) {
        int far_i = -1;
        double far_score = -1.0;
        for (int i = 0; i < k; ++i) {
            if (in_cycle[static_cast<std::size_t>(i)] != 0U) { continue; }
            double nearest = std::numeric_limits<double>::infinity();
            for (int node : cycle) {
                nearest = std::min(nearest, inst.dist2(subset[static_cast<std::size_t>(i)], node));
            }
            if (nearest > far_score) {
                far_score = nearest;
                far_i = i;
            }
        }
        const int node = subset[static_cast<std::size_t>(far_i)];
        const int m = static_cast<int>(cycle.size());
        int best_pos = 0;
        double best_cost = std::numeric_limits<double>::infinity();
        for (int i = 0; i < m; ++i) {
            const int next = (i + 1 == m) ? 0 : (i + 1);
            const double cost = inst.dist(cycle[static_cast<std::size_t>(i)], node)
                + inst.dist(node, cycle[static_cast<std::size_t>(next)])
                - inst.dist(cycle[static_cast<std::size_t>(i)], cycle[static_cast<std::size_t>(next)]);
            if (cost < best_cost) {
                best_cost = cost;
                best_pos = i + 1;
            }
        }
        cycle.insert(cycle.begin() + best_pos, node);
        in_cycle[static_cast<std::size_t>(far_i)] = 1U;
    }
    return cycle;
}

bool exact_small_tsp_cycle(const Instance& inst, const std::vector<int>& set_nodes, std::vector<int>& best_cycle, double& best_len) {
    const int k = static_cast<int>(set_nodes.size());
    if (k <= 0) { best_cycle.clear(); best_len = 0.0; return true; }
    if (k == 1) { best_cycle = set_nodes; best_len = 0.0; return true; }
    if (k == 2) { best_cycle = set_nodes; best_len = 2.0 * inst.dist(set_nodes[0], set_nodes[1]); return true; }
    if (k > kExactSmallTourLimit) { return false; }
    const int m = k - 1;
    const auto total = static_cast<std::uint32_t>(1U << m);
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> dm(static_cast<std::size_t>(k * k), 0.0);
    for (int i = 0; i < k; ++i) {
        for (int j = i + 1; j < k; ++j) {
            const double d = inst.dist(set_nodes[static_cast<std::size_t>(i)], set_nodes[static_cast<std::size_t>(j)]);
            dm[static_cast<std::size_t>(i * k + j)] = d;
            dm[static_cast<std::size_t>(j * k + i)] = d;
        }
    }
    std::vector<double> dp(static_cast<std::size_t>(total) * static_cast<std::size_t>(m), inf);
    std::vector<int> parent(static_cast<std::size_t>(total) * static_cast<std::size_t>(m), -1);
    for (int j = 0; j < m; ++j) {
        dp[static_cast<std::size_t>(1U << j) * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)] = dm[static_cast<std::size_t>(j + 1)];
    }
    for (std::uint32_t mask = 1; mask < total; ++mask) {
        std::uint32_t bits = mask;
        while (bits != 0U) {
            const int j = least_set_bit_index(bits);
            bits &= bits - 1U;
            const std::uint32_t prev_mask = mask ^ (1U << j);
            if (prev_mask == 0U) { continue; }
            double best = inf;
            int best_prev = -1;
            std::uint32_t prev_bits = prev_mask;
            while (prev_bits != 0U) {
                const int i = least_set_bit_index(prev_bits);
                prev_bits &= prev_bits - 1U;
                const double cand = dp[static_cast<std::size_t>(prev_mask) * static_cast<std::size_t>(m) + static_cast<std::size_t>(i)]
                    + dm[static_cast<std::size_t>((i + 1) * k + (j + 1))];
                if (cand < best) { best = cand; best_prev = i; }
            }
            dp[static_cast<std::size_t>(mask) * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)] = best;
            parent[static_cast<std::size_t>(mask) * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)] = best_prev;
        }
    }
    const std::uint32_t full = total - 1U;
    best_len = inf;
    int end = -1;
    for (int j = 0; j < m; ++j) {
        const double cand = dp[static_cast<std::size_t>(full) * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)] + dm[static_cast<std::size_t>((j + 1) * k)];
        if (cand < best_len) { best_len = cand; end = j; }
    }
    if (end < 0 || !std::isfinite(best_len)) { return false; }
    std::vector<int> order_index(static_cast<std::size_t>(k), 0);
    std::uint32_t mask = full;
    int cur = end;
    for (int pos = k - 1; pos >= 1; --pos) {
        order_index[static_cast<std::size_t>(pos)] = cur + 1;
        const int prev = parent[static_cast<std::size_t>(mask) * static_cast<std::size_t>(m) + static_cast<std::size_t>(cur)];
        mask ^= (1U << cur);
        cur = prev;
        if (mask == 0U) { break; }
    }
    best_cycle.resize(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        best_cycle[static_cast<std::size_t>(i)] = set_nodes[static_cast<std::size_t>(order_index[static_cast<std::size_t>(i)])];
    }
    return true;
}

} // namespace aldous_tsp
