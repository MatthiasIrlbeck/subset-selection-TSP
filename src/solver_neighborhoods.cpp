#include "solver_internal.hpp"

namespace aldous_tsp {

bool highp_delete_exchange_descent(Tour& tour, const Instance& inst, const std::vector<int>& reference, const SolverOptions& options, SearchStats* stats, int passes) {
    if (tour.k < 3 || reference.empty()) {
        return false;
    }
    std::vector<int> ref_pos(static_cast<std::size_t>(inst.N), -1);
    for (int i = 0; i < static_cast<int>(reference.size()); ++i) {
        if (reference[static_cast<std::size_t>(i)] >= 0 && reference[static_cast<std::size_t>(i)] < inst.N) {
            ref_pos[static_cast<std::size_t>(reference[static_cast<std::size_t>(i)])] = i;
        }
    }
    bool any = false;
    tour.ensure_edges(inst);
    std::vector<int> pool;
    pool.reserve(128);
    std::vector<SwapCandidatePair> swap_candidates;
    for (int pass = 0; pass < passes; ++pass) {
        swap_candidates.clear();
        for (int ri = 0; ri < tour.k; ++ri) {
            pool.clear();
            const int rem = tour.nodes[static_cast<std::size_t>(ri)];
            const int rp = (rem >= 0 && rem < inst.N) ? ref_pos[static_cast<std::size_t>(rem)] : -1;
            if (rp >= 0) {
                const int m = static_cast<int>(reference.size());
                auto wrap_ref = [m](int idx) {
                    idx %= m;
                    if (idx < 0) { idx += m; }
                    return idx;
                };
                for (int h = 1; h <= 10; ++h) {
                    push_unique(pool, reference[static_cast<std::size_t>(wrap_ref(rp + h))], &tour.in_set, 96);
                    push_unique(pool, reference[static_cast<std::size_t>(wrap_ref(rp - h))], &tour.in_set, 96);
                }
            }
            if (inst.knn_k > 0) {
                for (int r = 0; r < std::min(inst.knn_k, 24); ++r) {
                    push_unique(pool, inst.knn_at(rem, r), &tour.in_set, 96);
                }
            }
            for (int add : pool) {
                if (stats != nullptr) { ++stats->highp_exchange_scans; }
                swap_candidates.push_back({ri, add});
            }
        }
        const BatchedSwapResult chosen = best_batched_swap(inst, tour, swap_candidates);
        if (!chosen.valid || chosen.delta >= -kImprovementEps) {
            break;
        }
        tour.apply_swap_post_rem(chosen.remove_pos,
                                 chosen.post_remove_pred,
                                 chosen.add_node,
                                 inst,
                                 chosen.delta);
        polish_tour(tour, inst, options, stats, 1);
        any = true;
        if (stats != nullptr) { ++stats->highp_exchange_improvements; }
    }
    return any;
}

bool regret_repair_cycle(std::vector<int>& cycle, const Instance& inst, int target_k, const std::vector<int>& pool, const std::vector<unsigned char>& banned) {
    std::vector<unsigned char> in_set(static_cast<std::size_t>(inst.N), 0U);
    for (int v : cycle) { in_set[static_cast<std::size_t>(v)] = 1U; }
    std::vector<unsigned char> used_pool(pool.size(), 0U);
    while (static_cast<int>(cycle.size()) < target_k) {
        const int m = static_cast<int>(cycle.size());
        int best_node = -1;
        int best_pos = 0;
        int best_pool_idx = -1;
        double best_score = std::numeric_limits<double>::infinity();
        double best_cost = std::numeric_limits<double>::infinity();
        auto consider = [&](int node, int pool_idx) {
            if (node < 0 || node >= inst.N || banned[static_cast<std::size_t>(node)] != 0U || in_set[static_cast<std::size_t>(node)] != 0U) {
                return;
            }
            double c1 = std::numeric_limits<double>::infinity();
            double c2 = std::numeric_limits<double>::infinity();
            int pos1 = 0;
            for (int i = 0; i < m; ++i) {
                const int next = (i + 1 == m) ? 0 : (i + 1);
                const double c = inst.dist(cycle[static_cast<std::size_t>(i)], node)
                    + inst.dist(node, cycle[static_cast<std::size_t>(next)])
                    - inst.dist(cycle[static_cast<std::size_t>(i)], cycle[static_cast<std::size_t>(next)]);
                if (c < c1) { c2 = c1; c1 = c; pos1 = i + 1; }
                else if (c < c2) { c2 = c; }
            }
            if (!std::isfinite(c2)) { c2 = c1; }
            const double score = c1 - 0.35 * (c2 - c1);
            if (score < best_score || (score == best_score && c1 < best_cost)) {
                best_score = score;
                best_cost = c1;
                best_node = node;
                best_pos = pos1;
                best_pool_idx = pool_idx;
            }
        };
        for (int i = 0; i < static_cast<int>(pool.size()); ++i) {
            if (used_pool[static_cast<std::size_t>(i)] == 0U) {
                consider(pool[static_cast<std::size_t>(i)], i);
            }
        }
        if (best_node < 0) {
            for (int node = 0; node < inst.N; ++node) { consider(node, -1); }
        }
        if (best_node < 0) {
            return false;
        }
        cycle.insert(cycle.begin() + best_pos, best_node);
        in_set[static_cast<std::size_t>(best_node)] = 1U;
        if (best_pool_idx >= 0) { used_pool[static_cast<std::size_t>(best_pool_idx)] = 1U; }
    }
    return true;
}

bool cached_regret_repair_cycle(std::vector<int>& cycle,
                                const Instance& inst,
                                int target_k,
                                const std::vector<int>& pool,
                                const std::vector<unsigned char>& banned) {
    if (target_k < static_cast<int>(cycle.size()) || target_k > inst.N
        || banned.size() != static_cast<std::size_t>(inst.N)) {
        return false;
    }
    if (static_cast<int>(cycle.size()) == target_k) {
        return true;
    }
    if (cycle.size() < 2U) {
        return regret_repair_cycle(cycle, inst, target_k, pool, banned);
    }

    struct Profile {
        int node = -1;
        bool active = false;
        int pred[2] = {-1, -1};
        double cost[2] = {std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::infinity()};
        double score = std::numeric_limits<double>::infinity();
    };

    std::vector<unsigned char> in_set(static_cast<std::size_t>(inst.N), 0U);
    std::vector<int> rank(static_cast<std::size_t>(inst.N), -1);
    auto rebuild_rank = [&]() {
        for (int i = 0; i < static_cast<int>(cycle.size()); ++i) {
            rank[static_cast<std::size_t>(cycle[static_cast<std::size_t>(i)])] = i;
        }
    };
    for (const int node : cycle) {
        if (node < 0 || node >= inst.N || in_set[static_cast<std::size_t>(node)] != 0U) {
            return false;
        }
        in_set[static_cast<std::size_t>(node)] = 1U;
    }
    rebuild_rank();

    std::vector<Profile> profiles;
    profiles.reserve(pool.size() + static_cast<std::size_t>(inst.N));
    std::vector<unsigned char> represented(static_cast<std::size_t>(inst.N), 0U);

    auto edge_cost = [&](const int pred, const int node) {
        const int pred_rank = rank[static_cast<std::size_t>(pred)];
        const int next_rank = (pred_rank + 1 == static_cast<int>(cycle.size()))
            ? 0 : pred_rank + 1;
        const int succ = cycle[static_cast<std::size_t>(next_rank)];
        return inst.dist(pred, node) + inst.dist(node, succ) - inst.dist(pred, succ);
    };
    auto better_edge = [&](const double lhs_cost, const int lhs_pred,
                           const double rhs_cost, const int rhs_pred) {
        if (lhs_cost != rhs_cost) {
            return lhs_cost < rhs_cost;
        }
        if (rhs_pred < 0) {
            return true;
        }
        return rank[static_cast<std::size_t>(lhs_pred)]
             < rank[static_cast<std::size_t>(rhs_pred)];
    };
    auto consider_edge = [&](Profile& profile, const int pred, const double cost) {
        if (profile.pred[0] == pred || profile.pred[1] == pred) {
            return;
        }
        if (better_edge(cost, pred, profile.cost[0], profile.pred[0])) {
            profile.cost[1] = profile.cost[0];
            profile.pred[1] = profile.pred[0];
            profile.cost[0] = cost;
            profile.pred[0] = pred;
        } else if (better_edge(cost, pred, profile.cost[1], profile.pred[1])) {
            profile.cost[1] = cost;
            profile.pred[1] = pred;
        }
    };
    auto finish_profile = [](Profile& profile) {
        if (!std::isfinite(profile.cost[1])) {
            profile.cost[1] = profile.cost[0];
            profile.pred[1] = profile.pred[0];
        }
        profile.score = profile.cost[0]
            - 0.35 * (profile.cost[1] - profile.cost[0]);
        profile.active = profile.pred[0] >= 0 && std::isfinite(profile.score);
    };
    auto recompute = [&](Profile& profile) {
        profile.pred[0] = -1;
        profile.pred[1] = -1;
        profile.cost[0] = std::numeric_limits<double>::infinity();
        profile.cost[1] = std::numeric_limits<double>::infinity();
        if (profile.node < 0 || profile.node >= inst.N
            || banned[static_cast<std::size_t>(profile.node)] != 0U
            || in_set[static_cast<std::size_t>(profile.node)] != 0U) {
            profile.active = false;
            profile.score = std::numeric_limits<double>::infinity();
            return;
        }
        for (const int pred : cycle) {
            consider_edge(profile, pred, edge_cost(pred, profile.node));
        }
        finish_profile(profile);
    };
    auto append_profile = [&](const int node) {
        if (node < 0 || node >= inst.N
            || represented[static_cast<std::size_t>(node)] != 0U) {
            return;
        }
        represented[static_cast<std::size_t>(node)] = 1U;
        Profile profile;
        profile.node = node;
        profiles.push_back(profile);
        recompute(profiles.back());
    };
    for (const int node : pool) {
        append_profile(node);
    }

    bool appended_fallback = false;
    while (static_cast<int>(cycle.size()) < target_k) {
        int best_index = -1;
        double best_score = std::numeric_limits<double>::infinity();
        double best_cost = std::numeric_limits<double>::infinity();
        for (int i = 0; i < static_cast<int>(profiles.size()); ++i) {
            const Profile& profile = profiles[static_cast<std::size_t>(i)];
            if (!profile.active) {
                continue;
            }
            if (profile.score < best_score
                || (profile.score == best_score && profile.cost[0] < best_cost)) {
                best_index = i;
                best_score = profile.score;
                best_cost = profile.cost[0];
            }
        }
        if (best_index < 0 && !appended_fallback) {
            appended_fallback = true;
            for (int node = 0; node < inst.N; ++node) {
                append_profile(node);
            }
            continue;
        }
        if (best_index < 0) {
            return false;
        }

        Profile& chosen = profiles[static_cast<std::size_t>(best_index)];
        const int node = chosen.node;
        const int split_pred = chosen.pred[0];
        const int insert_pos = rank[static_cast<std::size_t>(split_pred)] + 1;
        cycle.insert(cycle.begin() + insert_pos, node);
        in_set[static_cast<std::size_t>(node)] = 1U;
        rebuild_rank();
        chosen.active = false;

        for (Profile& profile : profiles) {
            if (!profile.active) {
                continue;
            }
            if (profile.pred[0] == split_pred || profile.pred[1] == split_pred) {
                recompute(profile);
                continue;
            }
            consider_edge(profile, split_pred, edge_cost(split_pred, profile.node));
            consider_edge(profile, node, edge_cost(node, profile.node));
            finish_profile(profile);
        }
    }
    return true;
}

PairRepairResult best_two_node_regret_repair(const Instance& inst,
                                             const std::vector<int>& cycle,
                                             const std::vector<int>& pool) {
    PairRepairResult result;
    const int m = static_cast<int>(cycle.size());
    const int pool_size = static_cast<int>(pool.size());
    if (m < 2 || pool_size < 2 || inst.N <= 0) {
        return result;
    }

    struct InsertProfile {
        bool valid = false;
        int node = -1;
        double best_cost[2] = {std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity()};
        int best_pred[2] = {-1, -1};
        double score = std::numeric_limits<double>::infinity();
    };

    thread_local std::vector<double> edge_length;
    thread_local std::vector<double> distances;
    thread_local std::vector<InsertProfile> profiles;
    thread_local std::vector<unsigned char> in_cycle;
    edge_length.resize(static_cast<std::size_t>(m));
    distances.resize(static_cast<std::size_t>(pool_size) * static_cast<std::size_t>(m));
    profiles.assign(static_cast<std::size_t>(pool_size), InsertProfile{});
    in_cycle.assign(static_cast<std::size_t>(inst.N), 0U);

    for (int i = 0; i < m; ++i) {
        const int node = cycle[static_cast<std::size_t>(i)];
        if (node < 0 || node >= inst.N || in_cycle[static_cast<std::size_t>(node)] != 0U) {
            return result;
        }
        in_cycle[static_cast<std::size_t>(node)] = 1U;
        const int next = (i + 1 == m) ? 0 : (i + 1);
        edge_length[static_cast<std::size_t>(i)] =
            inst.dist(node, cycle[static_cast<std::size_t>(next)]);
    }

    for (int pi = 0; pi < pool_size; ++pi) {
        InsertProfile& profile = profiles[static_cast<std::size_t>(pi)];
        const int node = pool[static_cast<std::size_t>(pi)];
        profile.node = node;
        if (node < 0 || node >= inst.N || in_cycle[static_cast<std::size_t>(node)] != 0U) {
            continue;
        }
        double* row = distances.data()
            + static_cast<std::size_t>(pi) * static_cast<std::size_t>(m);
        dist_many_from(inst, node, cycle.data(), m, row);
        for (int pred = 0; pred < m; ++pred) {
            const int succ = (pred + 1 == m) ? 0 : (pred + 1);
            const double cost = row[static_cast<std::size_t>(pred)]
                + row[static_cast<std::size_t>(succ)]
                - edge_length[static_cast<std::size_t>(pred)];
            // Strict comparisons reproduce regret_repair_cycle's stable scan
            // order, including equal-cost insertion edges.
            if (cost < profile.best_cost[0]) {
                profile.best_cost[1] = profile.best_cost[0];
                profile.best_pred[1] = profile.best_pred[0];
                profile.best_cost[0] = cost;
                profile.best_pred[0] = pred;
            } else if (cost < profile.best_cost[1]) {
                profile.best_cost[1] = cost;
                profile.best_pred[1] = pred;
            }
        }
        if (!std::isfinite(profile.best_cost[1])) {
            profile.best_cost[1] = profile.best_cost[0];
            profile.best_pred[1] = profile.best_pred[0];
        }
        profile.score = profile.best_cost[0]
            - 0.35 * (profile.best_cost[1] - profile.best_cost[0]);
        profile.valid = std::isfinite(profile.score) && profile.best_pred[0] >= 0;
    }

    double best_delta = std::numeric_limits<double>::infinity();
    int best_first = -1;
    int best_second = -1;
    int best_first_pred = -1;
    int best_second_pred = -1;

    for (int ui = 0; ui < pool_size; ++ui) {
        for (int vi = ui + 1; vi < pool_size; ++vi) {
            const InsertProfile& up = profiles[static_cast<std::size_t>(ui)];
            const InsertProfile& vp = profiles[static_cast<std::size_t>(vi)];
            if (!up.valid || !vp.valid || up.node == vp.node) {
                continue;
            }

            int first_index = ui;
            int second_index = vi;
            // regret_repair_cycle considers ui first and replaces it only for
            // a strictly better score, or an equal score with lower c1.
            if (vp.score < up.score
                || (vp.score == up.score && vp.best_cost[0] < up.best_cost[0])) {
                first_index = vi;
                second_index = ui;
            }
            const InsertProfile& first = profiles[static_cast<std::size_t>(first_index)];
            const InsertProfile& second = profiles[static_cast<std::size_t>(second_index)];
            const int split_pred = first.best_pred[0];
            const int split_succ = (split_pred + 1 == m) ? 0 : (split_pred + 1);
            const double* first_dist = distances.data()
                + static_cast<std::size_t>(first_index) * static_cast<std::size_t>(m);
            const double* second_dist = distances.data()
                + static_cast<std::size_t>(second_index) * static_cast<std::size_t>(m);

            double second_cost = std::numeric_limits<double>::infinity();
            int second_pred = -1;
            auto consider_second = [&](double cost, int modified_pred) {
                if (cost < second_cost
                    || (cost == second_cost
                        && (second_pred < 0 || modified_pred < second_pred))) {
                    second_cost = cost;
                    second_pred = modified_pred;
                }
            };

            const int surviving_slot =
                (second.best_pred[0] == split_pred) ? 1 : 0;
            const int surviving_pred = second.best_pred[surviving_slot];
            if (surviving_pred >= 0 && surviving_pred != split_pred) {
                const int modified_pred =
                    (surviving_pred < split_pred) ? surviving_pred : (surviving_pred + 1);
                consider_second(second.best_cost[surviving_slot], modified_pred);
            }

            const double between = inst.dist(first.node, second.node);
            const double before_first = second_dist[static_cast<std::size_t>(split_pred)]
                + between - first_dist[static_cast<std::size_t>(split_pred)];
            consider_second(before_first, split_pred);
            const double after_first = between
                + second_dist[static_cast<std::size_t>(split_succ)]
                - first_dist[static_cast<std::size_t>(split_succ)];
            consider_second(after_first, split_pred + 1);

            if (second_pred < 0 || !std::isfinite(second_cost)) {
                continue;
            }
            const double delta = first.best_cost[0] + second_cost;
            if (delta < best_delta) {
                best_delta = delta;
                best_first = first_index;
                best_second = second_index;
                best_first_pred = split_pred;
                best_second_pred = second_pred;
            }
        }
    }

    if (best_first < 0) {
        return result;
    }
    result.nodes = cycle;
    result.nodes.insert(result.nodes.begin() + best_first_pred + 1,
                        pool[static_cast<std::size_t>(best_first)]);
    result.nodes.insert(result.nodes.begin() + best_second_pred + 1,
                        pool[static_cast<std::size_t>(best_second)]);
    result.length = cycle_length(inst, result.nodes);
    result.valid = std::isfinite(result.length);
    result.first_pool_index = best_first;
    result.second_pool_index = best_second;
    result.first_pred = best_first_pred;
    result.second_pred = best_second_pred;
    return result;
}

namespace {

enum class RuinOperator : int {
    WorstMarginal = 0,
    Segment = 1,
    SpatialCluster = 2,
    LongEdge = 3,
    Random = 4,
};

constexpr int kRuinOperatorCount = 5;

void record_ruin_attempt(SearchStats* stats, const RuinOperator op) noexcept {
    if (stats == nullptr) {
        return;
    }
    switch (op) {
        case RuinOperator::WorstMarginal: ++stats->ruin_recreate_worst_attempts; break;
        case RuinOperator::Segment: ++stats->ruin_recreate_segment_attempts; break;
        case RuinOperator::SpatialCluster: ++stats->ruin_recreate_spatial_attempts; break;
        case RuinOperator::LongEdge: ++stats->ruin_recreate_long_edge_attempts; break;
        case RuinOperator::Random: ++stats->ruin_recreate_random_attempts; break;
    }
}

void record_ruin_improvement(SearchStats* stats, const RuinOperator op) noexcept {
    if (stats == nullptr) {
        return;
    }
    switch (op) {
        case RuinOperator::WorstMarginal: ++stats->ruin_recreate_worst_improvements; break;
        case RuinOperator::Segment: ++stats->ruin_recreate_segment_improvements; break;
        case RuinOperator::SpatialCluster: ++stats->ruin_recreate_spatial_improvements; break;
        case RuinOperator::LongEdge: ++stats->ruin_recreate_long_edge_improvements; break;
        case RuinOperator::Random: ++stats->ruin_recreate_random_improvements; break;
    }
}

int adaptive_ruin_size(const int k, const int round, const SolverOptions& options) noexcept {
    constexpr double scales[] = {0.0025, 0.005, 0.01, 0.02, 0.05};
    const double configured_cap = std::max(0.0, options.ruin_recreate_max_fraction);
    const double fraction = std::min(scales[static_cast<std::size_t>(round % 5)],
                                     configured_cap);
    int size = std::max(3, static_cast<int>(std::ceil(fraction * static_cast<double>(k))));
    if (options.ruin_recreate_max_nodes > 0) {
        size = std::min(size, options.ruin_recreate_max_nodes);
    }
    return std::min(k - 3, size);
}

void fill_removed_positions(const Instance& inst,
                            const Tour& tour,
                            Rng& rng,
                            const RuinOperator op,
                            const int ruin_size,
                            const std::vector<double>& marginal,
                            const std::vector<int>& marginal_order,
                            std::vector<unsigned char>& removed_pos) {
    const int k = tour.k;
    auto mark = [&](const int pos) {
        if (pos >= 0 && pos < k) {
            removed_pos[static_cast<std::size_t>(pos)] = 1U;
        }
    };
    auto count_marked = [&]() {
        return static_cast<int>(std::count(removed_pos.begin(), removed_pos.end(),
                                           static_cast<unsigned char>(1U)));
    };

    switch (op) {
        case RuinOperator::WorstMarginal:
            for (int i = 0; i < ruin_size; ++i) {
                mark(marginal_order[static_cast<std::size_t>(i)]);
            }
            break;

        case RuinOperator::Segment: {
            const int anchor_count = std::max(1, std::min(k, 8));
            const int anchor = marginal_order[static_cast<std::size_t>(rng.randint(anchor_count))];
            const int begin = (anchor - ruin_size / 2 + k) % k;
            for (int offset = 0; offset < ruin_size; ++offset) {
                mark((begin + offset) % k);
            }
            break;
        }

        case RuinOperator::SpatialCluster: {
            const int anchor_count = std::max(1, std::min(k, 8));
            const int anchor_pos = marginal_order[static_cast<std::size_t>(rng.randint(anchor_count))];
            const int anchor_node = tour.nodes[static_cast<std::size_t>(anchor_pos)];
            std::vector<int> positions(static_cast<std::size_t>(k));
            std::iota(positions.begin(), positions.end(), 0);
            std::stable_sort(positions.begin(), positions.end(), [&](const int lhs, const int rhs) {
                const double dl = inst.dist2(anchor_node, tour.nodes[static_cast<std::size_t>(lhs)]);
                const double dr = inst.dist2(anchor_node, tour.nodes[static_cast<std::size_t>(rhs)]);
                if (dl != dr) {
                    return dl < dr;
                }
                return tour.nodes[static_cast<std::size_t>(lhs)]
                     < tour.nodes[static_cast<std::size_t>(rhs)];
            });
            for (int i = 0; i < ruin_size; ++i) {
                mark(positions[static_cast<std::size_t>(i)]);
            }
            break;
        }

        case RuinOperator::LongEdge: {
            std::vector<int> edges(static_cast<std::size_t>(k));
            std::iota(edges.begin(), edges.end(), 0);
            std::stable_sort(edges.begin(), edges.end(), [&](const int lhs, const int rhs) {
                const double dl = tour.edge_len[static_cast<std::size_t>(lhs)];
                const double dr = tour.edge_len[static_cast<std::size_t>(rhs)];
                if (dl != dr) {
                    return dl > dr;
                }
                return tour.nodes[static_cast<std::size_t>(lhs)]
                     < tour.nodes[static_cast<std::size_t>(rhs)];
            });
            for (const int edge : edges) {
                mark(edge);
                if (count_marked() >= ruin_size) { break; }
                mark((edge + 1 == k) ? 0 : edge + 1);
                if (count_marked() >= ruin_size) { break; }
            }
            break;
        }

        case RuinOperator::Random: {
            std::vector<int> positions(static_cast<std::size_t>(k));
            std::iota(positions.begin(), positions.end(), 0);
            for (int i = 0; i < ruin_size; ++i) {
                const int chosen = i + rng.randint(k - i);
                std::swap(positions[static_cast<std::size_t>(i)],
                          positions[static_cast<std::size_t>(chosen)]);
                mark(positions[static_cast<std::size_t>(i)]);
            }
            break;
        }
    }

    // Defensive exact-cardinality fill. It also makes the long-edge operator
    // well-defined when several longest edges share endpoints.
    for (const int pos : marginal_order) {
        if (count_marked() >= ruin_size) {
            break;
        }
        mark(pos);
    }
    (void)marginal;
}

} // namespace

bool run_ruin_recreate_policy(Tour& tour,
                              const Instance& inst,
                              Rng& rng,
                              const SolverOptions& options,
                              SearchStats* stats,
                              const int rounds,
                              const bool adaptive) {
    if (tour.k < 8 || rounds <= 0) {
        return false;
    }
    bool any = false;
    for (int round = 0; round < rounds; ++round) {
        if (stats != nullptr) { ++stats->ruin_recreate_attempts; }
        const int ruin_size = adaptive
            ? adaptive_ruin_size(tour.k, round, options)
            : std::min(tour.k - 3,
                       3 + (round % std::max(1, std::min(8, tour.k / 16))));
        if (ruin_size <= 1) { continue; }
        tour.ensure_edges(inst);
        std::vector<double> score(static_cast<std::size_t>(tour.k), 0.0);
        std::vector<int> ord(static_cast<std::size_t>(tour.k));
        std::iota(ord.begin(), ord.end(), 0);
        for (int i = 0; i < tour.k; ++i) {
            const int prev = (i == 0) ? tour.k - 1 : i - 1;
            const int next = (i + 1 == tour.k) ? 0 : i + 1;
            score[static_cast<std::size_t>(i)] =
                inst.dist(tour.nodes[static_cast<std::size_t>(prev)],
                          tour.nodes[static_cast<std::size_t>(i)])
                + inst.dist(tour.nodes[static_cast<std::size_t>(i)],
                            tour.nodes[static_cast<std::size_t>(next)])
                - inst.dist(tour.nodes[static_cast<std::size_t>(prev)],
                            tour.nodes[static_cast<std::size_t>(next)]);
        }
        std::stable_sort(ord.begin(), ord.end(), [&](const int lhs, const int rhs) {
            if (score[static_cast<std::size_t>(lhs)]
                != score[static_cast<std::size_t>(rhs)]) {
                return score[static_cast<std::size_t>(lhs)]
                     > score[static_cast<std::size_t>(rhs)];
            }
            return tour.nodes[static_cast<std::size_t>(lhs)]
                 < tour.nodes[static_cast<std::size_t>(rhs)];
        });

        const RuinOperator op = adaptive
            ? static_cast<RuinOperator>(round % kRuinOperatorCount)
            : ((round % 2 == 0) ? RuinOperator::Segment
                                : RuinOperator::WorstMarginal);
        record_ruin_attempt(stats, op);
        std::vector<unsigned char> removed_pos(static_cast<std::size_t>(tour.k), 0U);
        fill_removed_positions(inst, tour, rng, op, ruin_size, score, ord, removed_pos);

        std::vector<int> remain;
        std::vector<int> removed;
        remain.reserve(static_cast<std::size_t>(tour.k - ruin_size));
        removed.reserve(static_cast<std::size_t>(ruin_size));
        for (int i = 0; i < tour.k; ++i) {
            if (removed_pos[static_cast<std::size_t>(i)] != 0U) {
                removed.push_back(tour.nodes[static_cast<std::size_t>(i)]);
            } else {
                remain.push_back(tour.nodes[static_cast<std::size_t>(i)]);
            }
        }
        if (static_cast<int>(removed.size()) != ruin_size) {
            throw std::logic_error("ruin operator produced the wrong cardinality");
        }
        if (stats != nullptr) {
            stats->ruin_recreate_removed_nodes += static_cast<std::uint64_t>(removed.size());
        }

        std::vector<unsigned char> banned(static_cast<std::size_t>(inst.N), 0U);
        for (const int node : remain) {
            banned[static_cast<std::size_t>(node)] = 1U;
        }
        const int configured_cap = std::max(ruin_size, options.ruin_recreate_pool_cap);
        const int pool_cap = std::min(inst.N, configured_cap);
        std::vector<int> pool;
        pool.reserve(static_cast<std::size_t>(pool_cap));
        const int neighbor_limit = std::min(inst.knn_k,
                                            std::max(12, std::min(32, 8 + ruin_size / 4)));
        for (const int node : removed) {
            push_unique(pool, node, nullptr, pool_cap);
            for (int rank_index = 0; rank_index < neighbor_limit; ++rank_index) {
                push_unique(pool, inst.knn_at(node, rank_index), nullptr, pool_cap);
            }
            if (static_cast<int>(pool.size()) >= pool_cap) {
                break;
            }
        }
        const int random_trials = std::min(inst.N, 32 + 4 * ruin_size);
        for (int trial = 0;
             trial < random_trials && static_cast<int>(pool.size()) < pool_cap;
             ++trial) {
            push_unique(pool, rng.randint(inst.N), nullptr, pool_cap);
        }

        const bool repaired = cached_regret_repair_cycle(
            remain, inst, tour.k, pool, banned);
        if (!repaired) {
            continue;
        }
        Tour candidate;
        candidate.init(inst.N);
        candidate.set_tour(remain, inst);
        const int polish_strength = ruin_size >= 32 ? 3 : (ruin_size >= 5 ? 2 : 1);
        polish_tour(candidate, inst, options, stats, polish_strength);
        if (candidate.length < tour.length - kImprovementEps) {
            tour = std::move(candidate);
            any = true;
            if (stats != nullptr) { ++stats->ruin_recreate_improvements; }
            record_ruin_improvement(stats, op);
        }
    }
    return any;
}

bool subset_ruin_recreate_lns(Tour& tour,
                              const Instance& inst,
                              Rng& rng,
                              const SolverOptions& options,
                              SearchStats* stats,
                              const int rounds) {
    if (!options.adaptive_ruin_recreate) {
        return run_ruin_recreate_policy(tour, inst, rng, options, stats,
                                        rounds, false);
    }
    if (tour.k < 8 || rounds <= 0) {
        return false;
    }

    // Preserve the complete legacy trajectory as a quality floor. The adaptive
    // portfolio runs from the same starting tour and RNG state; after both
    // branches finish, the shorter result wins and the caller RNG advances as
    // the legacy branch would have advanced. This makes the stronger default
    // incapable of regressing the legacy LNS result on a fixed input while
    // keeping all later stochastic neighborhoods stream-stable.
    const double original_length = tour.length;
    Tour legacy_tour = tour;
    Tour adaptive_tour = tour;
    Rng legacy_rng = rng;
    Rng adaptive_rng = rng;
    SearchStats legacy_stats;
    SearchStats adaptive_stats;
    (void)run_ruin_recreate_policy(legacy_tour, inst, legacy_rng, options,
                                   stats == nullptr ? nullptr : &legacy_stats,
                                   rounds, false);
    (void)run_ruin_recreate_policy(adaptive_tour, inst, adaptive_rng, options,
                                   stats == nullptr ? nullptr : &adaptive_stats,
                                   rounds, true);
    rng = legacy_rng;
    if (stats != nullptr) {
        stats->add(legacy_stats);
        stats->add(adaptive_stats);
    }
    if (adaptive_tour.length < legacy_tour.length - kImprovementEps) {
        tour = std::move(adaptive_tour);
    } else {
        tour = std::move(legacy_tour);
    }
    return tour.length < original_length - kImprovementEps;
}

namespace {

struct ChainRemovalRank {
    double gain = 0.0;
    int position = -1;
    int node = -1;
};

std::vector<ChainRemovalRank> ranked_chain_removals(
    const Instance& inst,
    const Tour& tour,
    const std::vector<unsigned char>* locked_added = nullptr) {
    std::vector<ChainRemovalRank> ranked;
    ranked.reserve(static_cast<std::size_t>(tour.k));
    for (int position = 0; position < tour.k; ++position) {
        const int node = tour.nodes[static_cast<std::size_t>(position)];
        if (locked_added != nullptr
            && (*locked_added)[static_cast<std::size_t>(node)] != 0U) {
            continue;
        }
        const int prev = (position == 0) ? (tour.k - 1) : (position - 1);
        const int next = (position + 1 == tour.k) ? 0 : (position + 1);
        const double gap = inst.dist(
            tour.nodes[static_cast<std::size_t>(prev)],
            tour.nodes[static_cast<std::size_t>(next)]);
        const double gain = tour.edge_len[static_cast<std::size_t>(prev)]
            + tour.edge_len[static_cast<std::size_t>(position)] - gap;
        ranked.push_back({gain, position, node});
    }
    std::stable_sort(ranked.begin(), ranked.end(),
                     [](const ChainRemovalRank& lhs,
                        const ChainRemovalRank& rhs) {
        if (lhs.gain != rhs.gain) {
            return lhs.gain > rhs.gain;
        }
        if (lhs.node != rhs.node) {
            return lhs.node < rhs.node;
        }
        return lhs.position < rhs.position;
    });
    return ranked;
}

std::vector<int> ejection_chain_starts(const Instance& inst,
                                       const Tour& tour,
                                       Rng& rng,
                                       const int requested) {
    std::vector<int> starts;
    if (requested <= 0 || tour.k <= 0) {
        return starts;
    }
    const auto ranked = ranked_chain_removals(inst, tour);
    starts.reserve(static_cast<std::size_t>(std::min(requested, tour.k)));
    auto append = [&](const int node) {
        if (node >= 0
            && std::find(starts.begin(), starts.end(), node) == starts.end()) {
            starts.push_back(node);
        }
    };
    if (!ranked.empty()) {
        append(ranked.front().node);
    }

    int longest_edge = 0;
    for (int edge = 1; edge < tour.k; ++edge) {
        if (tour.edge_len[static_cast<std::size_t>(edge)]
            > tour.edge_len[static_cast<std::size_t>(longest_edge)]) {
            longest_edge = edge;
        }
    }
    append(tour.nodes[static_cast<std::size_t>(
        (longest_edge + 1 == tour.k) ? 0 : longest_edge + 1)]);
    if (requested >= 3) {
        append(tour.nodes[static_cast<std::size_t>(rng.randint(tour.k))]);
    }
    for (const ChainRemovalRank& entry : ranked) {
        if (static_cast<int>(starts.size()) >= requested) {
            break;
        }
        append(entry.node);
    }
    return starts;
}

void collect_ejection_add_candidates(
    const Instance& inst,
    const Tour& tour,
    const int focus_node,
    Rng& rng,
    const int requested,
    const std::vector<unsigned char>& removed_once,
    std::vector<int>& marker,
    const int marker_token,
    std::vector<int>& out) {
    out.clear();
    const int available = inst.N - tour.k;
    const int cap = std::min(std::max(0, requested), available);
    if (cap <= 0) {
        return;
    }
    auto append = [&](const int node) {
        if (node < 0 || node >= inst.N
            || tour.in_set[static_cast<std::size_t>(node)] != 0U
            || removed_once[static_cast<std::size_t>(node)] != 0U
            || marker[static_cast<std::size_t>(node)] == marker_token
            || static_cast<int>(out.size()) >= cap) {
            return;
        }
        marker[static_cast<std::size_t>(node)] = marker_token;
        out.push_back(node);
    };

    if (focus_node >= 0 && focus_node < inst.N && inst.knn_k > 0) {
        const int local = std::min(inst.knn_k, std::max(cap, 12));
        for (int rank = 0; rank < local; ++rank) {
            append(inst.knn_at(focus_node, rank));
        }
        // A small second ring prevents a selected nearest-neighbour shell from
        // starving the chain at high p while keeping the pool spatially tied to
        // the ejected node.
        const int anchors = std::min(inst.knn_k, 4);
        for (int anchor_rank = 0;
             anchor_rank < anchors && static_cast<int>(out.size()) < cap;
             ++anchor_rank) {
            const int anchor = inst.knn_at(focus_node, anchor_rank);
            for (int rank = 0;
                 rank < std::min(inst.knn_k, 6)
                 && static_cast<int>(out.size()) < cap;
                 ++rank) {
                append(inst.knn_at(anchor, rank));
            }
        }
    }

    const int random_trials = std::max(16, cap * 6);
    for (int trial = 0;
         trial < random_trials && static_cast<int>(out.size()) < cap;
         ++trial) {
        append(rng.randint(inst.N));
    }
    // Exact bounded fallback for dense subsets or unlucky rejection samples.
    // The random offset keeps different starts diverse; traversal itself is
    // deterministic for a fixed chain stream.
    const int offset = rng.randint(inst.N);
    for (int scanned = 0;
         scanned < inst.N && static_cast<int>(out.size()) < cap;
         ++scanned) {
        append((offset + scanned) % inst.N);
    }
}

void collect_ejection_remove_positions(
    const Instance& inst,
    const Tour& tour,
    const std::vector<int>& add_nodes,
    const std::vector<unsigned char>& locked_added,
    const int requested,
    std::vector<int>& marker,
    const int marker_token,
    std::vector<int>& out) {
    out.clear();
    const auto ranked = ranked_chain_removals(inst, tour, &locked_added);
    const int eligible = static_cast<int>(ranked.size());
    const int cap = requested <= 0
        ? eligible : std::min(requested, eligible);
    if (cap <= 0) {
        return;
    }
    auto append_position = [&](const int position) {
        if (position < 0 || position >= tour.k
            || marker[static_cast<std::size_t>(position)] == marker_token
            || locked_added[static_cast<std::size_t>(
                   tour.nodes[static_cast<std::size_t>(position)])] != 0U
            || static_cast<int>(out.size()) >= cap) {
            return;
        }
        marker[static_cast<std::size_t>(position)] = marker_token;
        out.push_back(position);
    };

    // Half the budget protects globally expensive deletions; the other half
    // gives local selected neighbours of the proposed additions a chance to
    // propagate a spatial ejection chain.
    const int global_prefix = std::max(1, cap / 2);
    for (int index = 0; index < std::min(global_prefix, eligible); ++index) {
        append_position(ranked[static_cast<std::size_t>(index)].position);
    }
    for (const int add : add_nodes) {
        for (int rank = 0;
             rank < std::min(inst.knn_k, 12)
             && static_cast<int>(out.size()) < cap;
             ++rank) {
            const int member = inst.knn_at(add, rank);
            if (member >= 0 && member < inst.N) {
                append_position(tour.pos[static_cast<std::size_t>(member)]);
            }
        }
        if (static_cast<int>(out.size()) >= cap) {
            break;
        }
    }
    for (const ChainRemovalRank& entry : ranked) {
        if (static_cast<int>(out.size()) >= cap) {
            break;
        }
        append_position(entry.position);
    }
}

} // namespace

bool subset_ejection_chain_search(Tour& tour,
                                  const Instance& inst,
                                  Rng& rng,
                                  const SolverOptions& options,
                                  SearchStats* stats) {
    if (tour.k < 4 || tour.k >= inst.N
        || options.ejection_chain_starts <= 0
        || options.ejection_chain_depth <= 0
        || options.ejection_chain_candidates <= 0) {
        return false;
    }
    tour.ensure_edges(inst);
    const Tour baseline = tour;
    const double baseline_length = baseline.length;
    const double mean_edge = baseline_length / static_cast<double>(baseline.k);
    const double max_length = baseline_length
        + std::max(0.0, options.ejection_chain_max_uphill) * mean_edge;
    const std::vector<int> starts = ejection_chain_starts(
        inst, baseline, rng, options.ejection_chain_starts);

    Tour best_candidate;
    bool have_best = false;
    double best_length = baseline_length;
    int best_depth = 0;
    std::vector<int> add_marker(static_cast<std::size_t>(inst.N), 0);
    std::vector<int> remove_marker(static_cast<std::size_t>(baseline.k), 0);
    int add_token = 0;
    int remove_token = 0;
    std::vector<int> add_nodes;
    std::vector<int> remove_positions;
    std::vector<SwapCandidatePair> pairs;

    for (const int start_node : starts) {
        if (stats != nullptr) {
            ++stats->ejection_chain_attempts;
        }
        Rng chain_rng(rng.next_u64());
        Tour working = baseline;
        std::vector<unsigned char> removed_once(
            static_cast<std::size_t>(inst.N), 0U);
        std::vector<unsigned char> locked_added(
            static_cast<std::size_t>(inst.N), 0U);
        int focus_node = start_node;
        int chain_steps = 0;
        int chain_best_depth = 0;
        double chain_best_length = std::numeric_limits<double>::infinity();
        std::vector<int> chain_best_nodes;

        for (int depth = 1; depth <= options.ejection_chain_depth; ++depth) {
            ++add_token;
            if (add_token == std::numeric_limits<int>::max()) {
                std::fill(add_marker.begin(), add_marker.end(), 0);
                add_token = 1;
            }
            collect_ejection_add_candidates(
                inst, working, focus_node, chain_rng,
                options.ejection_chain_candidates, removed_once,
                add_marker, add_token, add_nodes);
            if (add_nodes.empty()) {
                break;
            }

            ++remove_token;
            if (remove_token == std::numeric_limits<int>::max()) {
                std::fill(remove_marker.begin(), remove_marker.end(), 0);
                remove_token = 1;
            }
            collect_ejection_remove_positions(
                inst, working, add_nodes, locked_added,
                options.ejection_chain_remove_cap,
                remove_marker, remove_token, remove_positions);
            if (remove_positions.empty()) {
                break;
            }

            pairs.clear();
            pairs.reserve(add_nodes.size() * remove_positions.size());
            for (const int add : add_nodes) {
                for (const int remove_pos : remove_positions) {
                    pairs.push_back({remove_pos, add});
                }
            }
            if (stats != nullptr) {
                stats->ejection_chain_scans +=
                    static_cast<std::uint64_t>(pairs.size());
            }
            const BatchedSwapResult chosen = best_batched_swap(inst, working, pairs);
            if (!chosen.valid || !std::isfinite(chosen.delta)) {
                break;
            }
            const double next_length = working.length + chosen.delta;
            if (next_length > max_length + kImprovementEps) {
                break;
            }
            const int removed = working.nodes[
                static_cast<std::size_t>(chosen.remove_pos)];
            working.apply_swap_post_rem(chosen.remove_pos,
                                        chosen.post_remove_pred,
                                        chosen.add_node,
                                        inst,
                                        chosen.delta);
            removed_once[static_cast<std::size_t>(removed)] = 1U;
            locked_added[static_cast<std::size_t>(chosen.add_node)] = 1U;
            focus_node = removed;
            ++chain_steps;
            if (stats != nullptr) {
                ++stats->ejection_chain_steps;
            }
            if (working.length < chain_best_length - kImprovementEps) {
                chain_best_length = working.length;
                chain_best_nodes = working.nodes;
                chain_best_depth = depth;
            }
        }

        if (chain_steps == 0 || chain_best_nodes.empty()) {
            continue;
        }
        if (stats != nullptr) {
            ++stats->ejection_chain_feasible;
        }
        Tour candidate;
        candidate.init(inst.N);
        candidate.set_tour(chain_best_nodes, inst);
        polish_tour(candidate, inst, options, stats, 1);
        // A changed membership basin deserves one exact local membership pass;
        // this is intentionally bounded to one pass per start.
        if (!options.disable_subset_swap) {
            (void)subset_swap_descent_impl(candidate, inst, 1,
                                           !options.disable_two_opt, stats);
        }
        if (candidate.length < best_length - kImprovementEps) {
            best_length = candidate.length;
            best_candidate = std::move(candidate);
            best_depth = chain_best_depth;
            have_best = true;
        }
    }

    if (!have_best || best_length >= baseline_length - kImprovementEps) {
        return false;
    }
    tour = std::move(best_candidate);
    if (stats != nullptr) {
        ++stats->ejection_chain_improvements;
        stats->ejection_chain_accepted_depth +=
            static_cast<std::uint64_t>(best_depth);
    }
    return true;
}

bool subset_pair_exchange_descent(Tour& tour, const Instance& inst, Rng& rng, const SolverOptions& options, SearchStats* stats, int passes) {
    if (tour.k < 6 || passes <= 0) {
        return false;
    }
    if (options.pair_exchange_max_k > 0 && tour.k > options.pair_exchange_max_k) {
        if (stats != nullptr) {
            ++stats->pair_exchange_skipped_large_k;
        }
        return false;
    }
    bool any = false;
    for (int pass = 0; pass < passes; ++pass) {
        tour.ensure_edges(inst);
        std::vector<double> score(static_cast<std::size_t>(tour.k), 0.0);
        std::vector<int> ord(static_cast<std::size_t>(tour.k));
        std::iota(ord.begin(), ord.end(), 0);
        for (int i = 0; i < tour.k; ++i) {
            const int prev = (i == 0) ? tour.k - 1 : i - 1;
            const int next = (i + 1 == tour.k) ? 0 : i + 1;
            score[static_cast<std::size_t>(i)] = inst.dist(tour.nodes[static_cast<std::size_t>(prev)], tour.nodes[static_cast<std::size_t>(i)])
                + inst.dist(tour.nodes[static_cast<std::size_t>(i)], tour.nodes[static_cast<std::size_t>(next)])
                - inst.dist(tour.nodes[static_cast<std::size_t>(prev)], tour.nodes[static_cast<std::size_t>(next)]);
        }
        std::sort(ord.begin(), ord.end(), [&](int a, int b) {
            if (score[static_cast<std::size_t>(a)] != score[static_cast<std::size_t>(b)]) {
                return score[static_cast<std::size_t>(a)] > score[static_cast<std::size_t>(b)];
            }
            return tour.nodes[static_cast<std::size_t>(a)] < tour.nodes[static_cast<std::size_t>(b)];
        });
        const int top_rm = std::min(tour.k, 10);
        double best_len = tour.length - kImprovementEps;
        std::vector<int> best_nodes;
        for (int aa = 0; aa < top_rm; ++aa) {
            for (int bb = aa + 1; bb < top_rm; ++bb) {
                const int ri = ord[static_cast<std::size_t>(aa)];
                const int rj = ord[static_cast<std::size_t>(bb)];
                std::vector<int> remain;
                remain.reserve(static_cast<std::size_t>(tour.k - 2));
                std::vector<unsigned char> banned(static_cast<std::size_t>(inst.N), 0U);
                for (int i = 0; i < tour.k; ++i) {
                    if (i != ri && i != rj) {
                        const int node = tour.nodes[static_cast<std::size_t>(i)];
                        remain.push_back(node);
                        banned[static_cast<std::size_t>(node)] = 1U;
                    }
                }
                std::vector<int> pool;
                pool.reserve(80);
                for (int rem_pos : {ri, rj}) {
                    const int rem = tour.nodes[static_cast<std::size_t>(rem_pos)];
                    if (inst.knn_k > 0) {
                        for (int r = 0; r < std::min(inst.knn_k, 18); ++r) {
                            push_unique(pool, inst.knn_at(rem, r), &banned, 80);
                        }
                    }
                }
                for (int trial = 0; trial < 24 && static_cast<int>(pool.size()) < 80; ++trial) {
                    push_unique(pool, rng.randint(inst.N), &banned, 80);
                }
                if (pool.size() < 2U) {
                    continue;
                }
                if (stats != nullptr) {
                    const std::uint64_t count = static_cast<std::uint64_t>(pool.size());
                    stats->pair_exchange_scans += count * (count - 1U) / 2U;
                }
                PairRepairResult repaired = best_two_node_regret_repair(inst, remain, pool);
                if (repaired.valid && repaired.length < best_len) {
                    best_len = repaired.length;
                    best_nodes = std::move(repaired.nodes);
                }
            }
        }
        if (best_nodes.empty()) {
            break;
        }
        tour.set_tour(best_nodes, inst);
        polish_tour(tour, inst, options, stats, 1);
        any = true;
        if (stats != nullptr) { ++stats->pair_exchange_improvements; }
    }
    return any;
}

PathRelinkStep path_relink_best_step(const Instance& inst,
                                     const Tour& tour,
                                     const std::vector<int>& remove_positions,
                                     const std::vector<int>& add_nodes,
                                     SearchStats* stats) {
    PathRelinkStep best;
    if (tour.k < 4 || !tour.edge_valid || remove_positions.empty() || add_nodes.empty()) {
        return best;
    }

    // Preserve the legacy add-major/remove-minor ordering: it is the stable
    // tie priority used by path relinking on geometrically symmetric inputs.
    std::vector<SwapCandidatePair> candidates;
    candidates.reserve(add_nodes.size() * remove_positions.size());
    for (int add : add_nodes) {
        for (int remove_pos : remove_positions) {
            candidates.push_back({remove_pos, add});
        }
    }
    if (stats != nullptr) {
        stats->path_relink_candidate_scans +=
            static_cast<std::uint64_t>(candidates.size());
    }
    const BatchedSwapResult chosen = best_batched_swap(inst, tour, candidates);
    if (chosen.valid) {
        best.valid = true;
        best.delta = chosen.delta;
        best.remove_pos = chosen.remove_pos;
        best.add_node = chosen.add_node;
    }
    return best;
}

bool subset_path_relink_oneway(const Instance& inst, const std::vector<int>& src, const std::vector<int>& dst, Rng& rng, const SolverOptions& options, std::vector<int>& best_nodes, double& best_len, SearchStats* stats) {
    (void)rng;
    if (src.empty() || src.size() != dst.size() || static_cast<int>(src.size()) < 4) {
        return false;
    }
    Tour cur;
    cur.init(inst.N);
    cur.set_tour(src, inst);
    polish_tour(cur, inst, options, stats, 1);
    std::vector<unsigned char> target(static_cast<std::size_t>(inst.N), 0U);
    for (int v : dst) { target[static_cast<std::size_t>(v)] = 1U; }
    best_nodes = cur.nodes;
    best_len = cur.length;
    bool moved = false;
    std::vector<int> remove_positions;
    std::vector<int> add_nodes;
    for (int step = 0; step < cur.k; ++step) {
        cur.ensure_edges(inst);
        remove_positions.clear();
        add_nodes.clear();
        for (int i = 0; i < cur.k; ++i) {
            if (target[static_cast<std::size_t>(cur.nodes[static_cast<std::size_t>(i)])] == 0U) {
                remove_positions.push_back(i);
            }
        }
        for (int v : dst) {
            if (cur.in_set[static_cast<std::size_t>(v)] == 0U) {
                add_nodes.push_back(v);
            }
        }
        if (remove_positions.empty() || add_nodes.empty()) {
            break;
        }
        const PathRelinkStep chosen = path_relink_best_step(
            inst, cur, remove_positions, add_nodes, stats);
        if (!chosen.valid) {
            break;
        }
        const SwapInsertionMove move = find_best_insert_after_remove(inst, cur, chosen.remove_pos, chosen.add_node);
        if (!move.valid) {
            break;
        }
        cur.apply_swap_post_rem(move.remove_pos, move.post_pred, move.add_node, inst, move.delta);
        moved = true;
        if ((step + 1) % 3 == 0) {
            polish_tour(cur, inst, options, stats, 1);
        }
        if (cur.length < best_len - kImprovementEps) {
            best_len = cur.length;
            best_nodes = cur.nodes;
        }
    }
    if (!moved) {
        return false;
    }
    Tour fin;
    fin.init(inst.N);
    fin.set_tour(best_nodes, inst);
    polish_tour(fin, inst, options, stats, 1);
    best_nodes = fin.nodes;
    best_len = fin.length;
    return true;
}

bool subset_path_relink_bidirectional(const Instance& inst, const std::vector<int>& a, const std::vector<int>& b, Rng& rng, const SolverOptions& options, std::vector<int>& best_nodes, double& best_len, SearchStats* stats) {
    if (stats != nullptr) { ++stats->path_relink_attempts; }
    if (a.empty() || a.size() != b.size()) {
        return false;
    }
    // Relink cost scales superlinearly with the symmetric difference while its
    // value over restart/SA search collapses for distant pairs, so distant
    // elite pairs are skipped (measured: for N=1000, p=0.5 uncapped relinking
    // was ~96% of solve wall-clock at ~0.1% quality contribution).
    {
        std::vector<unsigned char> in_b(static_cast<std::size_t>(inst.N), 0U);
        for (int v : b) {
            if (v >= 0 && v < inst.N) { in_b[static_cast<std::size_t>(v)] = 1U; }
        }
        int diff = 0;
        for (int v : a) {
            if (v < 0 || v >= inst.N || in_b[static_cast<std::size_t>(v)] == 0U) { ++diff; }
        }
        if (diff == 0
            || (options.path_relink_max_removed > 0
                && diff > options.path_relink_max_removed)) {
            return false;
        }
    }
    std::vector<int> n1, n2;
    double l1 = std::numeric_limits<double>::infinity();
    double l2 = std::numeric_limits<double>::infinity();
    const bool ok1 = subset_path_relink_oneway(inst, a, b, rng, options, n1, l1, stats);
    const bool ok2 = subset_path_relink_oneway(inst, b, a, rng, options, n2, l2, stats);
    if (ok1 && (!ok2 || l1 <= l2)) {
        best_nodes = std::move(n1);
        best_len = l1;
        if (stats != nullptr) { ++stats->path_relink_feasible; }
        return true;
    }
    if (ok2) {
        best_nodes = std::move(n2);
        best_len = l2;
        if (stats != nullptr) { ++stats->path_relink_feasible; }
        return true;
    }
    return false;
}

} // namespace aldous_tsp
