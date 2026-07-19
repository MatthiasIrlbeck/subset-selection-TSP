#include "solver_internal.hpp"

namespace aldous_tsp {

std::vector<int> nearest_to_point_seed(const Instance& inst, double cx, double cy, int k, Rng& rng) {
    std::vector<int> order(static_cast<std::size_t>(inst.N));
    std::iota(order.begin(), order.end(), 0);
    auto score = [&](int node) {
        return inst.dist2_to_point(node, cx, cy);
    };
    auto cmp = [&](int a, int b) {
        const double da = score(a);
        const double db = score(b);
        if (da != db) {
            return da < db;
        }
        return a < b;
    };
    if (k < inst.N) {
        std::nth_element(order.begin(), order.begin() + k, order.end(), cmp);
    }
    order.resize(static_cast<std::size_t>(k));
    const int start = (k > 0) ? rng.randint(k) : 0;
    return nearest_neighbor_order(inst, order, start);
}

std::vector<int> dense_seed(const Instance& inst, int k, Rng& rng, int rank_offset) {
    if (inst.N <= 0 || k <= 0) {
        return {};
    }
    if (inst.knn_k <= 0) {
        return nearest_neighbor_order(inst, random_subset(inst.N, k, rng), 0);
    }
    const int rank_count = std::max(1, std::min(inst.knn_k, std::max(4, std::min(32 + rank_offset, std::max(4, k / 2)))));
    std::vector<std::pair<double, int>> scored;
    scored.reserve(static_cast<std::size_t>(inst.N));
    for (int node = 0; node < inst.N; ++node) {
        double score = 0.0;
        for (int r = 0; r < rank_count; ++r) {
            score += inst.knn_d_at(node, r);
        }
        score /= static_cast<double>(rank_count);
        score += 0.25 * inst.knn_d_at(node, rank_count - 1);
        scored.emplace_back(score, node);
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first < b.first;
        }
        return a.second < b.second;
    });
    const int choice = std::min(static_cast<int>(scored.size()) - 1, rng.randint(std::max(1, std::min(8, static_cast<int>(scored.size())))));
    const int center = scored[static_cast<std::size_t>(choice)].second;
    PointMeanAccumulator center_mean(inst.periodic, inst.side);
    center_mean.add(inst.points[static_cast<std::size_t>(center)]);
    const int lim = std::min(inst.knn_k, std::max(4, std::min(24, k / 2)));
    for (int r = 0; r < lim; ++r) {
        const int neighbor = inst.knn_at(center, r);
        center_mean.add(inst.points[static_cast<std::size_t>(neighbor)]);
    }
    const Point seed_center = center_mean.mean();
    return nearest_to_point_seed(inst, seed_center.x, seed_center.y, k, rng);
}

std::vector<std::vector<int>> make_smallp_seed_pool(const Instance& inst, int k, Rng& rng, int max_budget) {
    std::vector<std::vector<int>> out;
    const double p = static_cast<double>(k) / static_cast<double>(std::max(1, inst.N));
    if (p > 0.08 || max_budget <= 0) {
        return out;
    }
    const int budget = std::min(max_budget, (p <= 0.03 ? 8 : 5) + (inst.N >= 3000 ? 2 : 0));
    ElitePool dedup(budget, EliteMode::Set);
    auto add_seed = [&](std::vector<int> seed) {
        if (static_cast<int>(out.size()) >= budget || static_cast<int>(seed.size()) != k) {
            return;
        }
        const double len = cycle_length(inst, seed);
        const auto before = dedup.entries().size();
        dedup.try_add(seed, len);
        if (dedup.entries().size() != before) {
            out.push_back(std::move(seed));
        }
    };
    for (int i = 0; i < budget && static_cast<int>(out.size()) < budget; ++i) {
        add_seed(dense_seed(inst, k, rng, 4 * i));
    }
    if (inst.gx > 0 && inst.gy > 0 && static_cast<int>(out.size()) < budget) {
        std::vector<std::pair<int, int>> cells;
        cells.reserve(static_cast<std::size_t>(inst.gx * inst.gy));
        for (int c = 0; c < inst.gx * inst.gy; ++c) {
            const int cnt = inst.cell_begin[static_cast<std::size_t>(c + 1)] - inst.cell_begin[static_cast<std::size_t>(c)];
            if (cnt > 0) {
                cells.emplace_back(-cnt, c);
            }
        }
        std::sort(cells.begin(), cells.end());
        for (const auto& pr : cells) {
            const int c = pr.second;
            const int begin = inst.cell_begin[static_cast<std::size_t>(c)];
            const int end = inst.cell_begin[static_cast<std::size_t>(c + 1)];
            PointMeanAccumulator cell_mean(inst.periodic, inst.side);
            for (int pp = begin; pp < end; ++pp) {
                const int node = inst.cell_points[static_cast<std::size_t>(pp)];
                cell_mean.add(inst.points[static_cast<std::size_t>(node)]);
            }
            const Point center = cell_mean.mean();
            add_seed(nearest_to_point_seed(inst, center.x, center.y, k, rng));
            if (static_cast<int>(out.size()) >= budget) {
                break;
            }
        }
    }
    return out;
}

std::vector<int> highp_delete_seed(const Instance& inst, const std::vector<int>& parent, int k, Rng& rng, int mode) {
    std::vector<int> cur = parent;
    if (static_cast<int>(cur.size()) <= k) {
        return cur;
    }
    while (static_cast<int>(cur.size()) > k) {
        const int m = static_cast<int>(cur.size());
        std::vector<int> ord(static_cast<std::size_t>(m));
        std::vector<double> score(static_cast<std::size_t>(m), 0.0);
        std::iota(ord.begin(), ord.end(), 0);
        for (int i = 0; i < m; ++i) {
            const int prev = cur[static_cast<std::size_t>((i - 1 + m) % m)];
            const int node = cur[static_cast<std::size_t>(i)];
            const int next = cur[static_cast<std::size_t>((i + 1) % m)];
            score[static_cast<std::size_t>(i)] = inst.dist(prev, node) + inst.dist(node, next) - inst.dist(prev, next);
            if (mode == 2 && inst.knn_k > 0) {
                score[static_cast<std::size_t>(i)] += 0.15 * inst.knn_d_at(node, std::min(inst.knn_k - 1, 10));
            }
        }
        std::sort(ord.begin(), ord.end(), [&](int a, int b) {
            if (score[static_cast<std::size_t>(a)] != score[static_cast<std::size_t>(b)]) {
                return score[static_cast<std::size_t>(a)] > score[static_cast<std::size_t>(b)];
            }
            return cur[static_cast<std::size_t>(a)] < cur[static_cast<std::size_t>(b)];
        });
        int erase_pos = ord[0];
        if (mode == 1) {
            const int top = std::min(m, 8);
            erase_pos = ord[static_cast<std::size_t>(rng.randint(top))];
        }
        cur.erase(cur.begin() + erase_pos);
    }
    return cur;
}

std::vector<int> segment_delete_seed(const Instance& inst, const std::vector<int>& parent, int k, Rng& rng) {
    if (static_cast<int>(parent.size()) <= k) {
        return parent;
    }
    const int q = static_cast<int>(parent.size()) - k;
    const int m = static_cast<int>(parent.size());
    const int segment = std::min(q, std::max(2, q / 2));
    std::vector<double> score(static_cast<std::size_t>(m), 0.0);
    std::vector<int> ord(static_cast<std::size_t>(m));
    std::iota(ord.begin(), ord.end(), 0);
    for (int i = 0; i < m; ++i) {
        const int prev = parent[static_cast<std::size_t>((i - 1 + m) % m)];
        const int node = parent[static_cast<std::size_t>(i)];
        const int next = parent[static_cast<std::size_t>((i + 1) % m)];
        score[static_cast<std::size_t>(i)] = inst.dist(prev, node) + inst.dist(node, next) - inst.dist(prev, next);
    }
    std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        if (score[static_cast<std::size_t>(a)] != score[static_cast<std::size_t>(b)]) {
            return score[static_cast<std::size_t>(a)] > score[static_cast<std::size_t>(b)];
        }
        return parent[static_cast<std::size_t>(a)] < parent[static_cast<std::size_t>(b)];
    });
    const int start = ord[static_cast<std::size_t>(rng.randint(std::min(m, 3)))];
    std::vector<unsigned char> del(static_cast<std::size_t>(m), 0U);
    for (int i = 0; i < segment; ++i) {
        del[static_cast<std::size_t>((start + i) % m)] = 1U;
    }
    std::vector<int> cur;
    cur.reserve(static_cast<std::size_t>(m - segment));
    for (int i = 0; i < m; ++i) {
        if (del[static_cast<std::size_t>(i)] == 0U) {
            cur.push_back(parent[static_cast<std::size_t>(i)]);
        }
    }
    return highp_delete_seed(inst, cur, k, rng, 0);
}

std::vector<int> resize_seed(const Instance& inst, const std::vector<int>& seed, int k, Rng& rng, int mode) {
    if (static_cast<int>(seed.size()) == k) {
        return seed;
    }
    if (seed.empty()) {
        return dense_seed(inst, k, rng);
    }
    std::vector<int> cur = seed;
    if (static_cast<int>(cur.size()) > k) {
        return highp_delete_seed(inst, cur, k, rng, mode);
    }

    std::vector<unsigned char> in_set(static_cast<std::size_t>(inst.N), 0U);
    for (int node : cur) {
        if (node >= 0 && node < inst.N) {
            in_set[static_cast<std::size_t>(node)] = 1U;
        }
    }
    while (static_cast<int>(cur.size()) < k) {
        int best_node = -1;
        int best_pos = 0;
        double best_cost = std::numeric_limits<double>::infinity();
        std::vector<int> pool;
        pool.reserve(128);
        for (int seed_node : cur) {
            const int lim = std::min(inst.knn_k, 16 + 4 * mode);
            for (int r = 0; r < lim; ++r) {
                const int v = inst.knn_at(seed_node, r);
                if (v >= 0 && v < inst.N && in_set[static_cast<std::size_t>(v)] == 0U) {
                    push_unique(pool, v, nullptr, 160);
                }
            }
            if (static_cast<int>(pool.size()) >= 160) {
                break;
            }
        }
        if (pool.empty() || inst.N <= 600) {
            for (int v = 0; v < inst.N; ++v) {
                if (in_set[static_cast<std::size_t>(v)] == 0U) {
                    push_unique(pool, v, nullptr, inst.N);
                }
            }
        }
        const int m = static_cast<int>(cur.size());
        for (int candidate : pool) {
            if (m <= 1) {
                best_node = candidate;
                best_pos = m;
                best_cost = 0.0;
                break;
            }
            for (int pos = 0; pos < m; ++pos) {
                const int next_pos = (pos + 1 == m) ? 0 : (pos + 1);
                const double cost = inst.dist(cur[static_cast<std::size_t>(pos)], candidate)
                    + inst.dist(candidate, cur[static_cast<std::size_t>(next_pos)])
                    - inst.dist(cur[static_cast<std::size_t>(pos)], cur[static_cast<std::size_t>(next_pos)]);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_node = candidate;
                    best_pos = pos + 1;
                }
            }
        }
        if (best_node < 0) {
            break;
        }
        cur.insert(cur.begin() + best_pos, best_node);
        in_set[static_cast<std::size_t>(best_node)] = 1U;
    }
    return cur;
}

} // namespace aldous_tsp
