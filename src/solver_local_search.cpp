#include "solver_internal.hpp"

namespace aldous_tsp {

int two_opt_candidate_descent(Tour& tour, const Instance& inst, int max_passes, int cand_cap, SearchStats* stats, const SubsetCandidateTable* table) {
    tour.ensure_edges(inst);
    if (tour.k < 4 || max_passes <= 0 || (inst.knn_k <= 0 && table == nullptr)) {
        return 0;
    }
    const int kk = std::min(std::max(inst.knn_k, 0), cand_cap <= 0 ? std::max(inst.knn_k, 0) : cand_cap);
    const int scan_limit = static_cast<int>(std::min<std::int64_t>(static_cast<std::int64_t>(max_passes) * static_cast<std::int64_t>(tour.k), static_cast<std::int64_t>(std::numeric_limits<int>::max())));
    int improvements = 0;
    int scans = 0;
    std::vector<unsigned char> dont_look(static_cast<std::size_t>(tour.N), 0U);

    auto wake_node = [&](int node) {
        if (node < 0 || node >= tour.N) { return; }
        dont_look[static_cast<std::size_t>(node)] = 0U;
        if (!inst.rknn_begin.empty()) {
            for (int p = inst.rknn_begin[static_cast<std::size_t>(node)]; p < inst.rknn_begin[static_cast<std::size_t>(node + 1)]; ++p) {
                const int parent = inst.rknn_nodes[static_cast<std::size_t>(p)];
                if (parent >= 0 && parent < tour.N && tour.in_set[static_cast<std::size_t>(parent)] != 0U) {
                    dont_look[static_cast<std::size_t>(parent)] = 0U;
                }
            }
        }
    };

    // Evaluates and, when improving, applies the 2-opt move removing the edges
    // at positions pi and pj. Both scan orientations reduce to this: successor
    // edges of the anchor/candidate, or their predecessor edges.
    auto try_edge_pair = [&](int pi, int pj) -> bool {
        if (pi == pj) { return false; }
        const int ii = std::min(pi, pj);
        const int jj = std::max(pi, pj);
        if (jj - ii < 2 || (ii == 0 && jj == tour.k - 1)) {
            return false;
        }
        if (stats != nullptr) { ++stats->two_opt_scans; }
        const int aa = tour.nodes[static_cast<std::size_t>(ii)];
        const int bb = tour.nodes[static_cast<std::size_t>(ii + 1)];
        const int cc = tour.nodes[static_cast<std::size_t>(jj)];
        const int dd = tour.nodes[static_cast<std::size_t>((jj + 1 == tour.k) ? 0 : (jj + 1))];
        const double delta = inst.dist(aa, cc) + inst.dist(bb, dd)
            - tour.edge_len[static_cast<std::size_t>(ii)] - tour.edge_len[static_cast<std::size_t>(jj)];
        if (delta >= -kImprovementEps) {
            return false;
        }
        tour.apply_two_opt(ii, jj, inst, delta);
        ++improvements;
        if (stats != nullptr) { ++stats->two_opt_improvements; }
        const int left = tour.nodes[static_cast<std::size_t>((ii == 0) ? (tour.k - 1) : (ii - 1))];
        const int e0 = tour.nodes[static_cast<std::size_t>(ii)];
        const int e1 = tour.nodes[static_cast<std::size_t>(ii + 1)];
        const int e2 = tour.nodes[static_cast<std::size_t>(jj)];
        const int right = tour.nodes[static_cast<std::size_t>((jj + 1 == tour.k) ? 0 : (jj + 1))];
        wake_node(left);
        wake_node(e0);
        wake_node(e1);
        wake_node(e2);
        wake_node(right);
        if (jj - ii <= 256) {
            for (int t = ii + 1; t <= jj; ++t) {
                wake_node(tour.nodes[static_cast<std::size_t>(t)]);
            }
        }
        return true;
    };

    bool any_improved = true;
    while (any_improved && scans < scan_limit) {
        any_improved = false;
        for (int idx = 0; idx < tour.k && scans < scan_limit; ++idx) {
            const int a = tour.nodes[static_cast<std::size_t>(idx)];
            if (dont_look[static_cast<std::size_t>(a)] != 0U) {
                continue;
            }
            ++scans;
            const int prev_idx = (idx == 0) ? (tour.k - 1) : (idx - 1);
            const double old_ab = tour.edge_len[static_cast<std::size_t>(idx)];
            const double old_za = tour.edge_len[static_cast<std::size_t>(prev_idx)];
            const double old_max = std::max(old_ab, old_za);
            const int row = (table != nullptr && a >= 0 && a < static_cast<int>(table->row_of_node.size()))
                ? table->row_of_node[static_cast<std::size_t>(a)] : -1;
            const int limit = (row >= 0)
                ? std::min(table->m, cand_cap <= 0 ? table->m : cand_cap)
                : kk;
            bool improved_here = false;
            for (int r = 0; r < limit; ++r) {
                int c = -1;
                double dac = 0.0;
                if (row >= 0) {
                    const std::size_t slot = static_cast<std::size_t>(row) * static_cast<std::size_t>(table->m) + static_cast<std::size_t>(r);
                    c = table->ids[slot];
                    if (c < 0) { break; }
                    dac = table->dist[slot];
                } else {
                    c = inst.knn_at(a, r);
                    dac = inst.knn_d_at(a, r);
                }
                if (dac >= old_max) {
                    break;
                }
                const int j = tour.pos[static_cast<std::size_t>(c)];
                if (j < 0 || j == idx) {
                    continue;
                }
                // Successor-edge orientation: remove (a, succ a) and (c, succ c).
                if (dac < old_ab && try_edge_pair(idx, j)) {
                    improved_here = true;
                    any_improved = true;
                    break;
                }
                // Predecessor-edge orientation: remove (pred a, a) and (pred c, c).
                const int j_prev = (j == 0) ? (tour.k - 1) : (j - 1);
                if (dac < old_za && try_edge_pair(prev_idx, j_prev)) {
                    improved_here = true;
                    any_improved = true;
                    break;
                }
            }
            if (!improved_here) {
                dont_look[static_cast<std::size_t>(a)] = 1U;
            } else {
                break;
            }
        }
    }
    return improvements;
}

int or_opt_1_candidate_descent(Tour& tour, const Instance& inst, int max_passes, int cand_cap, SearchStats* stats, const SubsetCandidateTable* table) {
    tour.ensure_edges(inst);
    if (tour.k < 5 || max_passes <= 0) {
        return 0;
    }
    const int kk = std::min(std::max(inst.knn_k, 0), cand_cap <= 0 ? std::max(inst.knn_k, 0) : cand_cap);
    const int scan_limit = static_cast<int>(std::min<std::int64_t>(
        static_cast<std::int64_t>(max_passes) * static_cast<std::int64_t>(tour.k),
        static_cast<std::int64_t>(std::numeric_limits<int>::max())));
    int improvements = 0;
    int scans = 0;
    std::vector<unsigned char> dont_look(static_cast<std::size_t>(tour.N), 0U);
    thread_local std::vector<int> pred_positions;

    // First-improvement relocation with don't-look bits and reverse-KNN
    // wakeups, mirroring the candidate 2-opt structure. A node is retired
    // (don't-look set) once it has no improving relocation; it is woken again
    // only when a nearby move changes its neighborhood. This replaces the old
    // best-improvement scheme that re-scanned all k nodes to apply one move.
    auto wake_node = [&](int node) {
        if (node < 0 || node >= tour.N) { return; }
        dont_look[static_cast<std::size_t>(node)] = 0U;
        if (!inst.rknn_begin.empty()) {
            for (int p = inst.rknn_begin[static_cast<std::size_t>(node)]; p < inst.rknn_begin[static_cast<std::size_t>(node + 1)]; ++p) {
                const int parent = inst.rknn_nodes[static_cast<std::size_t>(p)];
                if (parent >= 0 && parent < tour.N && tour.in_set[static_cast<std::size_t>(parent)] != 0U) {
                    dont_look[static_cast<std::size_t>(parent)] = 0U;
                }
            }
        }
    };

    bool found_since_clear = true;
    while (found_since_clear && scans < scan_limit) {
        // Clear-restart: a full sweep with all nodes awake, then wakeup-driven
        // descent to quiescence. Reverse-KNN wakeups cover the spatial
        // neighborhood but not the positional pos-1 offsets, so a wakeup-only
        // descent can strand a move; repeating from an all-awake state until a
        // full descent finds nothing guarantees a true candidate-local optimum.
        std::fill(dont_look.begin(), dont_look.end(), static_cast<unsigned char>(0));
        found_since_clear = false;
        bool any_improved = true;
        while (any_improved && scans < scan_limit) {
            any_improved = false;
            for (int idx = 0; idx < tour.k && scans < scan_limit; ++idx) {
                const int node = tour.nodes[static_cast<std::size_t>(idx)];
                if (dont_look[static_cast<std::size_t>(node)] != 0U) {
                    continue;
                }
                ++scans;
                pred_positions.clear();
                pred_positions.reserve(64);
                for (int off = -3; off <= 3; ++off) {
                    int pred = idx + off;
                    while (pred < 0) { pred += tour.k; }
                    while (pred >= tour.k) { pred -= tour.k; }
                    if (pred != idx) { push_unique(pred_positions, pred); }
                }
                const int row = (table != nullptr && node >= 0 && node < static_cast<int>(table->row_of_node.size()))
                    ? table->row_of_node[static_cast<std::size_t>(node)] : -1;
                const int limit = (row >= 0)
                    ? std::min(table->m, cand_cap <= 0 ? table->m : cand_cap)
                    : kk;
                for (int r = 0; r < limit; ++r) {
                    int near = -1;
                    if (row >= 0) {
                        near = table->ids[static_cast<std::size_t>(row) * static_cast<std::size_t>(table->m) + static_cast<std::size_t>(r)];
                        if (near < 0) { break; }
                    } else {
                        near = inst.knn_at(node, r);
                    }
                    const int pos = tour.pos[static_cast<std::size_t>(near)];
                    if (pos >= 0 && pos != idx) {
                        push_unique(pred_positions, pos);
                        push_unique(pred_positions, (pos == 0 ? tour.k - 1 : pos - 1));
                    }
                }
                if (stats != nullptr) { stats->or_opt_scans += static_cast<std::uint64_t>(pred_positions.size()); }
                const SwapMoveEval eval = evaluate_move_after_remove(inst, tour, idx, &pred_positions);
                if (eval.valid && eval.delta < -kImprovementEps) {
                    const int old_prev = tour.nodes[static_cast<std::size_t>((idx == 0) ? (tour.k - 1) : (idx - 1))];
                    const int old_next = tour.nodes[static_cast<std::size_t>((idx + 1 == tour.k) ? 0 : (idx + 1))];
                    tour.apply_move_post_rem(idx, eval.post_remove_pred, inst, eval.delta);
                    ++improvements;
                    any_improved = true;
                    found_since_clear = true;
                    if (stats != nullptr) { ++stats->or_opt_improvements; }
                    const int newpos = tour.pos[static_cast<std::size_t>(node)];
                    const int new_prev = tour.nodes[static_cast<std::size_t>((newpos == 0) ? (tour.k - 1) : (newpos - 1))];
                    const int new_next = tour.nodes[static_cast<std::size_t>((newpos + 1 == tour.k) ? 0 : (newpos + 1))];
                    wake_node(node);
                    wake_node(old_prev);
                    wake_node(old_next);
                    wake_node(new_prev);
                    wake_node(new_next);
                } else {
                    dont_look[static_cast<std::size_t>(node)] = 1U;
                }
            }
        }
    }
    return improvements;
}

int or_opt_segment_candidate_descent(Tour& tour, const Instance& inst, int seg_len, int max_passes, int cand_cap, SearchStats* stats, const SubsetCandidateTable* table) {
    tour.ensure_edges(inst);
    const int L = seg_len;
    if (L < 2 || L > 3 || tour.k < L + 4 || max_passes <= 0) {
        return 0;
    }
    const int kk = std::min(std::max(inst.knn_k, 0), cand_cap <= 0 ? std::max(inst.knn_k, 0) : cand_cap);
    const int scan_limit = static_cast<int>(std::min<std::int64_t>(
        static_cast<std::int64_t>(max_passes) * static_cast<std::int64_t>(tour.k),
        static_cast<std::int64_t>(std::numeric_limits<int>::max())));
    int improvements = 0;
    int scans = 0;
    std::vector<unsigned char> dont_look(static_cast<std::size_t>(tour.N), 0U);
    thread_local std::vector<int> pred_positions;
    thread_local std::vector<int> rebuilt;

    // First-improvement segment relocation with don't-look bits keyed on the
    // segment's start node, mirroring or_opt_1. Retiring a start node avoids
    // re-scanning every segment each pass; a node is woken when a nearby move
    // changes its neighborhood.
    auto wake_node = [&](int nd) {
        if (nd < 0 || nd >= tour.N) { return; }
        dont_look[static_cast<std::size_t>(nd)] = 0U;
        if (!inst.rknn_begin.empty()) {
            for (int p = inst.rknn_begin[static_cast<std::size_t>(nd)]; p < inst.rknn_begin[static_cast<std::size_t>(nd + 1)]; ++p) {
                const int parent = inst.rknn_nodes[static_cast<std::size_t>(p)];
                if (parent >= 0 && parent < tour.N && tour.in_set[static_cast<std::size_t>(parent)] != 0U) {
                    dont_look[static_cast<std::size_t>(parent)] = 0U;
                }
            }
        }
    };

    // Clear-restart confirming passes, as in or_opt_1: repeat all-awake
    // descents until one finds nothing, guaranteeing a candidate-local optimum
    // despite the reverse-KNN wake set not covering positional offsets.
    bool found_since_clear = true;
    while (found_since_clear && scans < scan_limit) {
      std::fill(dont_look.begin(), dont_look.end(), static_cast<unsigned char>(0));
      found_since_clear = false;
      bool any_improved = true;
      while (any_improved && scans < scan_limit) {
        any_improved = false;
        for (int s = 0; s + L - 1 < tour.k && scans < scan_limit; ++s) {
            const int u = tour.nodes[static_cast<std::size_t>(s)];
            if (dont_look[static_cast<std::size_t>(u)] != 0U) {
                continue;
            }
            ++scans;
            const int v = tour.nodes[static_cast<std::size_t>(s + L - 1)];
            const int p_idx = (s == 0) ? (tour.k - 1) : (s - 1);
            const int q_idx = (s + L == tour.k) ? 0 : (s + L);
            const double gain = tour.edge_len[static_cast<std::size_t>(p_idx)]
                + tour.edge_len[static_cast<std::size_t>(s + L - 1)]
                - inst.dist(tour.nodes[static_cast<std::size_t>(p_idx)], tour.nodes[static_cast<std::size_t>(q_idx)]);

            // Candidate insertion edges: near both segment endpoints, plus a
            // small window of local offsets for stability.
            pred_positions.clear();
            for (int off = -3; off <= 3; ++off) {
                int pred = s + off;
                while (pred < 0) { pred += tour.k; }
                while (pred >= tour.k) { pred -= tour.k; }
                push_unique(pred_positions, pred);
            }
            auto add_near = [&](int endpoint) {
                const int row = (table != nullptr && endpoint >= 0 && endpoint < static_cast<int>(table->row_of_node.size()))
                    ? table->row_of_node[static_cast<std::size_t>(endpoint)] : -1;
                const int limit = (row >= 0)
                    ? std::min(table->m, cand_cap <= 0 ? table->m : cand_cap)
                    : kk;
                for (int r = 0; r < limit; ++r) {
                    int near = -1;
                    if (row >= 0) {
                        near = table->ids[static_cast<std::size_t>(row) * static_cast<std::size_t>(table->m) + static_cast<std::size_t>(r)];
                        if (near < 0) { break; }
                    } else {
                        near = inst.knn_at(endpoint, r);
                    }
                    const int pos = tour.pos[static_cast<std::size_t>(near)];
                    if (pos >= 0) {
                        push_unique(pred_positions, pos);
                        push_unique(pred_positions, (pos == 0 ? tour.k - 1 : pos - 1));
                    }
                }
            };
            add_near(u);
            add_near(v);

            double best_delta = -kImprovementEps;
            int best_pred = -1;
            bool best_reversed = false;
            for (int j : pred_positions) {
                // Exclude the merged edge (null move) and edges inside or
                // leaving the segment, which do not exist after removal.
                if (j == p_idx || (j >= s && j <= s + L - 1)) {
                    continue;
                }
                if (stats != nullptr) { ++stats->or_opt_scans; }
                const int a = tour.nodes[static_cast<std::size_t>(j)];
                const int b = tour.nodes[static_cast<std::size_t>((j + 1 == tour.k) ? 0 : (j + 1))];
                const double base = tour.edge_len[static_cast<std::size_t>(j)];
                const double fwd = inst.dist(a, u) + inst.dist(v, b) - base;
                const double rev = inst.dist(a, v) + inst.dist(u, b) - base;
                const bool reversed = rev < fwd;
                const double delta = (reversed ? rev : fwd) - gain;
                if (delta < best_delta) {
                    best_delta = delta;
                    best_pred = j;
                    best_reversed = reversed;
                }
            }
            if (best_pred < 0) {
                dont_look[static_cast<std::size_t>(u)] = 1U;
                continue;
            }
            const int old_prev = tour.nodes[static_cast<std::size_t>(p_idx)];
            const int old_next = tour.nodes[static_cast<std::size_t>(q_idx)];
            const int ins_a = tour.nodes[static_cast<std::size_t>(best_pred)];
            const int ins_b = tour.nodes[static_cast<std::size_t>((best_pred + 1 == tour.k) ? 0 : (best_pred + 1))];
            rebuilt.clear();
            rebuilt.reserve(static_cast<std::size_t>(tour.k));
            for (int t = 0; t < tour.k; ++t) {
                if (t >= s && t <= s + L - 1) {
                    continue;
                }
                rebuilt.push_back(tour.nodes[static_cast<std::size_t>(t)]);
                if (t == best_pred) {
                    if (best_reversed) {
                        for (int e = s + L - 1; e >= s; --e) {
                            rebuilt.push_back(tour.nodes[static_cast<std::size_t>(e)]);
                        }
                    } else {
                        for (int e = s; e <= s + L - 1; ++e) {
                            rebuilt.push_back(tour.nodes[static_cast<std::size_t>(e)]);
                        }
                    }
                }
            }
            tour.set_tour(rebuilt, inst);
            ++improvements;
            any_improved = true;
            found_since_clear = true;
            if (stats != nullptr) { ++stats->or_opt_improvements; }
            wake_node(u);
            wake_node(v);
            wake_node(old_prev);
            wake_node(old_next);
            wake_node(ins_a);
            wake_node(ins_b);
        }
      }
    }
    return improvements;
}

bool use_all_polish_exhaustive_two_opt(const SolverOptions& options, int k) noexcept {
    return options.exhaustive_two_opt_policy == ExhaustiveTwoOptPolicy::AllPolish
        && options.final_exhaustive_k > 0
        && k <= options.final_exhaustive_k;
}

void polish_tour(Tour& tour, const Instance& inst, const SolverOptions& options, SearchStats* stats, int strength) {
    if (tour.k <= kExactSmallTourLimit) {
        std::vector<int> exact;
        double exact_len = 0.0;
        if (exact_small_tsp_cycle(inst, tour.nodes, exact, exact_len)) {
            tour.set_tour(exact, inst);
        }
    }
    const bool exhaustive = use_all_polish_exhaustive_two_opt(options, tour.k);
    // Membership is constant within a polish call (2-opt and or-opt only
    // reorder), so one subset candidate table serves every stage below.
    const SubsetCandidateTable* table = exhaustive ? nullptr : maybe_subset_candidates(inst, tour);
    if (!options.disable_two_opt) {
        if (exhaustive) { two_opt_descent(tour, inst, 10000, stats); }
        else { two_opt_candidate_descent(tour, inst, 120 + 40 * strength, 32 + 8 * strength, stats, table); }
    }
    if (!options.disable_or_opt) {
        if (exhaustive && tour.k <= 180) { or_opt_1_descent(tour, inst, 12 + 3 * strength, stats); }
        else { or_opt_1_candidate_descent(tour, inst, 5 + strength, 18 + 4 * strength, stats, table); }
        or_opt_segment_candidate_descent(tour, inst, 2, 2 + strength, 12, stats, table);
        or_opt_segment_candidate_descent(tour, inst, 3, 2 + strength, 12, stats, table);
    }
    if (!options.disable_two_opt) {
        if (exhaustive) { two_opt_descent(tour, inst, 10000, stats); }
        else { two_opt_candidate_descent(tour, inst, 80 + 30 * strength, 40 + 8 * strength, stats, table); }
    }
}

void final_polish_tour(Tour& tour, const Instance& inst, const SolverOptions& options, SearchStats* stats, int strength) {
    polish_tour(tour, inst, options, stats, strength);
    if (!options.disable_two_opt
        && options.exhaustive_two_opt_policy != ExhaustiveTwoOptPolicy::Never
        && options.final_exhaustive_k > 0
        && tour.k <= options.final_exhaustive_k) {
        two_opt_descent(tour, inst, 10000, stats);
    }
}


int two_opt_descent(Tour& tour, const Instance& inst, int max_passes, SearchStats* stats) {
    tour.ensure_edges(inst);
    if (tour.k < 4 || max_passes <= 0) { return 0; }
    int improvements = 0;
    for (int pass = 0; pass < max_passes; ++pass) {
        bool improved = false;
        for (int i = 0; i < tour.k - 2 && !improved; ++i) {
            const int a = tour.nodes[static_cast<std::size_t>(i)];
            const int b = tour.nodes[static_cast<std::size_t>(i + 1)];
            for (int j = i + 2; j < tour.k; ++j) {
                if (i == 0 && j == tour.k - 1) { continue; }
                if (stats != nullptr) { ++stats->two_opt_scans; }
                const int c = tour.nodes[static_cast<std::size_t>(j)];
                const int d = tour.nodes[static_cast<std::size_t>((j + 1 == tour.k) ? 0 : (j + 1))];
                const double delta = inst.dist(a, c) + inst.dist(b, d)
                    - tour.edge_len[static_cast<std::size_t>(i)] - tour.edge_len[static_cast<std::size_t>(j)];
                if (delta < -kImprovementEps) {
                    tour.apply_two_opt(i, j, inst, delta);
                    improved = true;
                    ++improvements;
                    if (stats != nullptr) { ++stats->two_opt_improvements; }
                    break;
                }
            }
        }
        if (!improved) { break; }
    }
    return improvements;
}

int or_opt_1_descent(Tour& tour, const Instance& inst, int max_passes, SearchStats* stats) {
    tour.ensure_edges(inst);
    if (tour.k < 5 || max_passes <= 0) { return 0; }
    int improvements = 0;
    for (int pass = 0; pass < max_passes; ++pass) {
        double best_delta = -kImprovementEps;
        int best_remove = -1;
        int best_post = 0;
        for (int i = 0; i < tour.k; ++i) {
            if (stats != nullptr) { stats->or_opt_scans += static_cast<std::uint64_t>(std::max(0, tour.k - 2)); }
            const SwapMoveEval eval = evaluate_move_after_remove(inst, tour, i);
            if (eval.valid && eval.delta < best_delta) {
                best_delta = eval.delta;
                best_remove = i;
                best_post = eval.post_remove_pred;
            }
        }
        if (best_remove < 0) { break; }
        tour.apply_move_post_rem(best_remove, best_post, inst, best_delta);
        ++improvements;
        if (stats != nullptr) { ++stats->or_opt_improvements; }
    }
    return improvements;
}

int subset_swap_descent_impl(Tour& tour, const Instance& inst, int max_passes, bool enable_two_opt, SearchStats* stats) {
    tour.ensure_edges(inst);
    if (tour.k < 3 || tour.k >= inst.N || max_passes <= 0) { return 0; }
    int improvements = 0;
    Rng local_rng(subset_hash_nodes(tour.nodes));
    std::vector<int> add_candidates;
    std::vector<SwapCandidatePair> swap_candidates;
    const int candidate_cap = (inst.N <= 800) ? inst.N : 160;
    for (int pass = 0; pass < max_passes; ++pass) {
        swap_candidates.clear();
        for (int ri = 0; ri < tour.k; ++ri) {
            collect_add_candidates_into(inst, tour, ri, local_rng, candidate_cap, add_candidates);
            for (int add : add_candidates) {
                if (add < 0 || add >= inst.N || tour.in_set[static_cast<std::size_t>(add)] != 0U) { continue; }
                if (stats != nullptr) { ++stats->subset_swap_scans; }
                swap_candidates.push_back({ri, add});
            }
        }
        const BatchedSwapResult chosen = best_batched_swap(inst, tour, swap_candidates);
        if (!chosen.valid || chosen.delta >= -kImprovementEps) { break; }
        tour.apply_swap_post_rem(chosen.remove_pos,
                                 chosen.post_remove_pred,
                                 chosen.add_node,
                                 inst,
                                 chosen.delta);
        if (enable_two_opt) {
            // Membership changed: build a fresh subset candidate table.
            two_opt_candidate_descent(tour, inst, 120, 40, stats, maybe_subset_candidates(inst, tour));
        }
        ++improvements;
        if (stats != nullptr) { ++stats->subset_swap_improvements; }
    }
    return improvements;
}

int subset_swap_descent(Tour& tour, const Instance& inst, int max_passes, SearchStats* stats) {
    return subset_swap_descent_impl(tour, inst, max_passes, true, stats);
}

} // namespace aldous_tsp
