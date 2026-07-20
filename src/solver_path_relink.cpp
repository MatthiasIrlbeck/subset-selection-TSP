#include "solver_internal.hpp"

namespace aldous_tsp {

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
