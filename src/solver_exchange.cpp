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


} // namespace aldous_tsp
