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
    for (int pass = 0; pass < passes; ++pass) {
        double best_delta = -kImprovementEps;
        int best_remove = -1;
        int best_add = -1;
        int best_post = 0;
        for (int ri = 0; ri < tour.k; ++ri) {
            std::vector<int> pool;
            pool.reserve(128);
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
                const SwapMoveEval eval = evaluate_swap_after_remove(inst, tour, ri, add);
                if (eval.valid && eval.delta < best_delta) {
                    best_delta = eval.delta;
                    best_remove = ri;
                    best_add = add;
                    best_post = eval.post_remove_pred;
                }
            }
        }
        if (best_remove < 0) {
            break;
        }
        tour.apply_swap_post_rem(best_remove, best_post, best_add, inst, best_delta);
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

bool subset_ruin_recreate_lns(Tour& tour, const Instance& inst, Rng& rng, const SolverOptions& options, SearchStats* stats, int rounds) {
    if (tour.k < 8 || rounds <= 0) {
        return false;
    }
    bool any = false;
    for (int round = 0; round < rounds; ++round) {
        if (stats != nullptr) { ++stats->ruin_recreate_attempts; }
        const int ruin_size = std::min(tour.k - 3, 3 + (round % std::max(1, std::min(8, tour.k / 16))));
        if (ruin_size <= 1) { continue; }
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
        std::vector<unsigned char> removed_pos(static_cast<std::size_t>(tour.k), 0U);
        const int anchor = ord[static_cast<std::size_t>(rng.randint(std::min(tour.k, 6)))];
        if (round % 2 == 0) {
            const int start = (anchor - ruin_size / 2 + tour.k) % tour.k;
            for (int t = 0; t < ruin_size; ++t) { removed_pos[static_cast<std::size_t>((start + t) % tour.k)] = 1U; }
        } else {
            for (int t = 0; t < ruin_size; ++t) { removed_pos[static_cast<std::size_t>(ord[static_cast<std::size_t>(t)])] = 1U; }
        }
        std::vector<int> remain;
        std::vector<int> removed;
        remain.reserve(static_cast<std::size_t>(tour.k - ruin_size));
        for (int i = 0; i < tour.k; ++i) {
            if (removed_pos[static_cast<std::size_t>(i)] != 0U) { removed.push_back(tour.nodes[static_cast<std::size_t>(i)]); }
            else { remain.push_back(tour.nodes[static_cast<std::size_t>(i)]); }
        }
        std::vector<unsigned char> banned(static_cast<std::size_t>(inst.N), 0U);
        for (int v : remain) { banned[static_cast<std::size_t>(v)] = 1U; }
        std::vector<int> pool;
        for (int v : removed) {
            push_unique(pool, v, nullptr, 160);
            if (inst.knn_k > 0) {
                for (int r = 0; r < std::min(inst.knn_k, 20); ++r) {
                    push_unique(pool, inst.knn_at(v, r), nullptr, 160);
                }
            }
        }
        for (int trial = 0; trial < 32 && static_cast<int>(pool.size()) < 160; ++trial) {
            push_unique(pool, rng.randint(inst.N), nullptr, 160);
        }
        if (!regret_repair_cycle(remain, inst, tour.k, pool, banned)) {
            continue;
        }
        Tour cand;
        cand.init(inst.N);
        cand.set_tour(remain, inst);
        polish_tour(cand, inst, options, stats, ruin_size >= 5 ? 2 : 1);
        if (cand.length < tour.length - kImprovementEps) {
            tour = std::move(cand);
            any = true;
            if (stats != nullptr) { ++stats->ruin_recreate_improvements; }
        }
    }
    return any;
}

bool subset_pair_exchange_descent(Tour& tour, const Instance& inst, Rng& rng, const SolverOptions& options, SearchStats* stats, int passes) {
    if (tour.k < 6 || passes <= 0) {
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
                std::vector<unsigned char> banned(static_cast<std::size_t>(inst.N), 0U);
                for (int i = 0; i < tour.k; ++i) {
                    if (i != ri && i != rj) {
                        remain.push_back(tour.nodes[static_cast<std::size_t>(i)]);
                        banned[static_cast<std::size_t>(tour.nodes[static_cast<std::size_t>(i)])] = 1U;
                    }
                }
                std::vector<int> pool;
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
                if (static_cast<int>(pool.size()) < 2) { continue; }
                for (int ui = 0; ui < static_cast<int>(pool.size()); ++ui) {
                    for (int vi = ui + 1; vi < static_cast<int>(pool.size()); ++vi) {
                        if (stats != nullptr) { ++stats->pair_exchange_scans; }
                        std::vector<int> cand = remain;
                        std::vector<unsigned char> ban2 = banned;
                        std::vector<int> add_pool = {pool[static_cast<std::size_t>(ui)], pool[static_cast<std::size_t>(vi)]};
                        if (!regret_repair_cycle(cand, inst, tour.k, add_pool, ban2)) { continue; }
                        const double len = cycle_length(inst, cand);
                        if (len < best_len) {
                            best_len = len;
                            best_nodes = std::move(cand);
                        }
                    }
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
                                     const std::vector<int>& add_nodes) {
    PathRelinkStep best;
    if (tour.k < 4 || !tour.edge_valid || remove_positions.empty() || add_nodes.empty()) {
        return best;
    }
    const int k = tour.k;

    // Removal gains and merged-edge metadata are independent of the added node.
    struct RemoveCand {
        int ri = -1;
        int prev = -1;
        int next = -1;
        double gap = 0.0;
        double gain = 0.0;
    };
    thread_local std::vector<RemoveCand> removes;
    removes.clear();
    removes.reserve(remove_positions.size());
    for (int ri : remove_positions) {
        if (ri < 0 || ri >= k) { continue; }
        RemoveCand rc;
        rc.ri = ri;
        rc.prev = (ri == 0) ? (k - 1) : (ri - 1);
        rc.next = (ri + 1 == k) ? 0 : (ri + 1);
        rc.gap = inst.dist(tour.nodes[static_cast<std::size_t>(rc.prev)], tour.nodes[static_cast<std::size_t>(rc.next)]);
        rc.gain = tour.edge_len[static_cast<std::size_t>(rc.prev)] + tour.edge_len[static_cast<std::size_t>(ri)] - rc.gap;
        removes.push_back(rc);
    }
    if (removes.empty()) {
        return best;
    }

    thread_local std::vector<double> add_dist;
    add_dist.assign(static_cast<std::size_t>(k), 0.0);
    for (int add : add_nodes) {
        if (add < 0 || add >= inst.N || tour.in_set[static_cast<std::size_t>(add)] != 0U) { continue; }
        dist_many_from(inst, add, tour.nodes.data(), k, add_dist.data());

        // Top-3 insertion edges over the unmodified tour. Any removal
        // invalidates at most two predecessor slots ({ri, prev(ri)}), so the
        // best valid pre-removal edge for every removal is among these three.
        double cand_cost[3] = {std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::infinity()};
        int cand_pred[3] = {-1, -1, -1};
        for (int pred = 0; pred < k; ++pred) {
            const int succ = (pred + 1 == k) ? 0 : (pred + 1);
            const double cost = add_dist[static_cast<std::size_t>(pred)]
                + add_dist[static_cast<std::size_t>(succ)]
                - tour.edge_len[static_cast<std::size_t>(pred)];
            if (cost < cand_cost[2]) {
                cand_cost[2] = cost;
                cand_pred[2] = pred;
                if (cand_cost[2] < cand_cost[1]) {
                    std::swap(cand_cost[1], cand_cost[2]);
                    std::swap(cand_pred[1], cand_pred[2]);
                }
                if (cand_cost[1] < cand_cost[0]) {
                    std::swap(cand_cost[0], cand_cost[1]);
                    std::swap(cand_pred[0], cand_pred[1]);
                }
            }
        }

        for (const RemoveCand& rc : removes) {
            double insert_cost = add_dist[static_cast<std::size_t>(rc.prev)]
                + add_dist[static_cast<std::size_t>(rc.next)] - rc.gap;
            for (int c = 0; c < 3; ++c) {
                if (cand_pred[c] >= 0 && cand_pred[c] != rc.ri && cand_pred[c] != rc.prev) {
                    insert_cost = std::min(insert_cost, cand_cost[c]);
                    break;
                }
            }
            const double delta = insert_cost - rc.gain;
            if (delta < best.delta) {
                best.valid = true;
                best.delta = delta;
                best.remove_pos = rc.ri;
                best.add_node = add;
            }
        }
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
        const PathRelinkStep chosen = path_relink_best_step(inst, cur, remove_positions, add_nodes);
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
        if (diff == 0 || diff > kPathRelinkMaxDiff) {
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
