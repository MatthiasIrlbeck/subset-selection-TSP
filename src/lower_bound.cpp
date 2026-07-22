#include "aldous_tsp/lower_bound.hpp"

#include "aldous_tsp/instance.hpp"
#include "aldous_tsp/validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace aldous_tsp {

namespace {

// Guard: the dense matrix is O(n^2); refuse absurd sizes to avoid OOM. The
// campaign keeps k well under this.
constexpr int kMaxHeldKarpNodes = 6000;

}  // namespace

HeldKarpBound held_karp_bound(const Instance& base,
                              const std::vector<int>& subset,
                              double upper_bound,
                              int max_iters) {
    return held_karp_bound(
        PreparedInstance::from_instance(base), subset, upper_bound, max_iters);
}

HeldKarpBound held_karp_bound(const PreparedInstance& prepared,
                              const std::vector<int>& subset,
                              double upper_bound,
                              int max_iters) {
    require_valid_lower_bound_request(
        prepared, subset, upper_bound, max_iters);
    const Instance& base = prepared.instance();
    HeldKarpBound result;
    const int n = static_cast<int>(subset.size());
    if (n < 3 || n > kMaxHeldKarpNodes) {
        return result;  // not computed
    }
    const auto N = static_cast<std::size_t>(n);

    // Materialize the (symmetric) distance matrix once, using the instance
    // metric so periodic (torus) distances are honored.
    std::vector<double> dist(N * N, 0.0);
    for (int i = 0; i < n; ++i) {
        const int gi = subset[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n; ++j) {
            const double d = base.dist(gi, subset[static_cast<std::size_t>(j)]);
            dist[static_cast<std::size_t>(i) * N + static_cast<std::size_t>(j)] = d;
            dist[static_cast<std::size_t>(j) * N + static_cast<std::size_t>(i)] = d;
        }
    }

    std::vector<double> pi(N, 0.0);
    std::vector<double> key(N, 0.0);
    std::vector<int> parent(N, -1);
    std::vector<char> in_tree(N, 0);
    std::vector<int> degree(N, 0);

    const bool have_ub = std::isfinite(upper_bound);
    double best_bound = -std::numeric_limits<double>::infinity();
    double lambda = 2.0;
    int since_improve = 0;
    const int patience = std::max(5, max_iters / 15);

    int iter = 0;
    for (; iter < max_iters; ++iter) {
        std::fill(degree.begin(), degree.end(), 0);

        // Minimum spanning tree over nodes 1..n-1 (node 0 excluded) via Prim,
        // using modified weights c'(u,v) = dist(u,v) + pi[u] + pi[v].
        std::fill(in_tree.begin(), in_tree.end(), 0);
        double mst_cost = 0.0;
        // Start the tree at node 1.
        for (int v = 1; v < n; ++v) {
            key[static_cast<std::size_t>(v)] = std::numeric_limits<double>::infinity();
            parent[static_cast<std::size_t>(v)] = -1;
        }
        key[1] = 0.0;
        for (int count = 1; count < n; ++count) {
            int u = -1;
            double best_key = std::numeric_limits<double>::infinity();
            for (int v = 1; v < n; ++v) {
                if (!in_tree[static_cast<std::size_t>(v)] && key[static_cast<std::size_t>(v)] < best_key) {
                    best_key = key[static_cast<std::size_t>(v)];
                    u = v;
                }
            }
            if (u < 0 || !std::isfinite(best_key)) {
                throw std::domain_error(
                    "Held-Karp MST construction encountered a nonfinite metric");
            }
            in_tree[static_cast<std::size_t>(u)] = 1;
            mst_cost += key[static_cast<std::size_t>(u)];
            const int pu = parent[static_cast<std::size_t>(u)];
            if (pu != -1) {
                ++degree[static_cast<std::size_t>(u)];
                ++degree[static_cast<std::size_t>(pu)];
            }
            const std::size_t urow = static_cast<std::size_t>(u) * N;
            const double piu = pi[static_cast<std::size_t>(u)];
            for (int v = 1; v < n; ++v) {
                if (in_tree[static_cast<std::size_t>(v)]) {
                    continue;
                }
                const double w = dist[urow + static_cast<std::size_t>(v)] + piu + pi[static_cast<std::size_t>(v)];
                if (w < key[static_cast<std::size_t>(v)]) {
                    key[static_cast<std::size_t>(v)] = w;
                    parent[static_cast<std::size_t>(v)] = u;
                }
            }
        }

        // Attach node 0 with its two cheapest edges (the "1-tree" part).
        int first = -1;
        int second = -1;
        double first_w = std::numeric_limits<double>::infinity();
        double second_w = std::numeric_limits<double>::infinity();
        const double pi0 = pi[0];
        for (int v = 1; v < n; ++v) {
            const double w = dist[static_cast<std::size_t>(v)] + pi0 + pi[static_cast<std::size_t>(v)];
            if (w < first_w) {
                second_w = first_w;
                second = first;
                first_w = w;
                first = v;
            } else if (w < second_w) {
                second_w = w;
                second = v;
            }
        }
        const double one_tree_cost = mst_cost + first_w + second_w;
        if (first < 0 || second < 0 || !std::isfinite(one_tree_cost)) {
            throw std::domain_error(
                "Held-Karp 1-tree construction encountered a nonfinite metric");
        }
        degree[0] = 2;
        ++degree[static_cast<std::size_t>(first)];
        ++degree[static_cast<std::size_t>(second)];

        double pi_sum = 0.0;
        for (double p : pi) {
            pi_sum += p;
        }
        const double lagrangian = one_tree_cost - 2.0 * pi_sum;

        if (iter == 0) {
            result.one_tree = lagrangian;  // pi = 0 here
        }
        if (lagrangian > best_bound) {
            best_bound = lagrangian;
            since_improve = 0;
        } else {
            ++since_improve;
        }

        // Subgradient g_i = degree_i - 2. Zero subgradient == a tour == optimum.
        double gnorm2 = 0.0;
        for (int i = 0; i < n; ++i) {
            const double g = static_cast<double>(degree[static_cast<std::size_t>(i)] - 2);
            gnorm2 += g * g;
        }
        if (gnorm2 == 0.0) {
            result.closed_to_tour = true;
            best_bound = std::max(best_bound, lagrangian);
            ++iter;
            break;
        }

        // Polyak-style step toward the (upper) bound. Without a finite upper
        // bound, fall back to a diminishing step.
        double target_gap;
        if (have_ub) {
            target_gap = upper_bound - lagrangian;
            if (target_gap <= 1e-12) {
                ++iter;
                break;  // bound has met the upper bound; converged
            }
        } else {
            target_gap = std::max(1.0, std::fabs(lagrangian)) * 0.01;
        }
        const double step = lambda * target_gap / gnorm2;
        for (int i = 0; i < n; ++i) {
            pi[static_cast<std::size_t>(i)] += step * static_cast<double>(degree[static_cast<std::size_t>(i)] - 2);
        }

        if (since_improve >= patience) {
            lambda *= 0.5;
            since_improve = 0;
            if (lambda < 1e-4) {
                ++iter;
                break;
            }
        }
    }

    result.bound = best_bound;
    result.iterations = iter;
    result.computed = true;
    if (have_ub) {
        result.gap_to_upper = std::max(0.0, upper_bound - best_bound);
    }
    return result;
}

}  // namespace aldous_tsp
