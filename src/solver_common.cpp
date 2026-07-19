#include "solver_internal.hpp"

#include "periodic_grid.hpp"

namespace aldous_tsp {

std::vector<int> all_nodes(int n) {
    std::vector<int> nodes(static_cast<std::size_t>(n));
    std::iota(nodes.begin(), nodes.end(), 0);
    return nodes;
}

std::vector<int> random_subset(int n, int k, Rng& rng) {
    std::vector<int> nodes = all_nodes(n);
    rng.partial_shuffle(nodes.begin(), nodes.end(), static_cast<std::size_t>(k));
    nodes.resize(static_cast<std::size_t>(k));
    return nodes;
}

RestartRecord make_restart_record(const Instance& inst,
                                  const std::vector<int>& nodes,
                                  double length,
                                  RestartKind kind) {
    RestartRecord record;
    record.length = length;
    record.kind = kind;
    if (nodes.empty()) {
        return record;
    }

    PointMeanAccumulator centroid(inst.periodic, inst.side);
    for (const int node : nodes) {
        centroid.add(inst.points[static_cast<std::size_t>(node)]);
    }
    const Point center = centroid.mean();
    record.centroid_x = center.x;
    record.centroid_y = center.y;
    double radius_sum = 0.0;
    for (const int node : nodes) {
        radius_sum += inst.dist_to_point(node, center.x, center.y);
    }
    record.radius = radius_sum / static_cast<double>(nodes.size());
    return record;
}

void push_unique(std::vector<int>& values, int value, const std::vector<unsigned char>* banned, int cap) {
    if (static_cast<int>(values.size()) >= cap) {
        return;
    }
    if (banned != nullptr && value >= 0 && value < static_cast<int>(banned->size()) && (*banned)[static_cast<std::size_t>(value)] != 0U) {
        return;
    }
    for (int x : values) {
        if (x == value) {
            return;
        }
    }
    values.push_back(value);
}


int next_live_index(int idx, int removed, int n) noexcept {
    int next = idx + 1;
    if (next >= n) { next = 0; }
    if (next == removed) {
        ++next;
        if (next >= n) { next = 0; }
    }
    return next;
}

// Windowed variant of find_best_insert_after_remove. The exact version scans
// every tour position -- O(k) per SA move, which measured as ~19 us of the
// ~21.7 us total move cost at k=2000 and makes move cost grow linearly in k.
// This variant evaluates only the slots that geometry says can matter:
//   (a) predecessors within `window` tour positions of the removed slot. The
//       incoming candidate is KNN-local to the removed node, and at small p
//       the tour spacing (~p^-1/2) exceeds the KNN radius, so its competitive
//       insertion slots are the vacated slot and its tour-neighborhood.
//   (b) slots adjacent to in-tour members of the incoming node's own KNN row
//       -- the far-relocation slots, whenever they exist (they usually do at
//       large p, rarely at small p).
// The delta formula, the post-removal predecessor indexing, and tie-breaking
// direction are identical to the exact version; only the candidate slot set
// is restricted. Validated against the exact version by budget-matched
// convergence ladders (see convergence_study.py); the exact scan remains
// available via --sa-exact-insertion.
SwapInsertionMove find_best_insert_after_remove_windowed(const Instance& inst, const Tour& tour, int remove_pos, int add_node, int window) {
    SwapInsertionMove move;
    move.remove_pos = remove_pos;
    move.add_node = add_node;
    const int n = tour.k;
    if (n < 2 || remove_pos < 0 || remove_pos >= n || add_node < 0 || add_node >= inst.N) {
        return move;
    }
    if (n <= 2 * window + 2) {
        // Window covers the whole tour: identical to the exact scan, so use it.
        return find_best_insert_after_remove(inst, tour, remove_pos, add_node);
    }
    const int prev_idx = (remove_pos == 0) ? (n - 1) : (remove_pos - 1);
    const int next_idx = (remove_pos + 1 == n) ? 0 : (remove_pos + 1);
    const int prev = tour.nodes[static_cast<std::size_t>(prev_idx)];
    const int removed = tour.nodes[static_cast<std::size_t>(remove_pos)];
    const int next = tour.nodes[static_cast<std::size_t>(next_idx)];
    const double remove_gain = inst.dist(prev, removed) + inst.dist(removed, next) - inst.dist(prev, next);

    double best_insert = std::numeric_limits<double>::infinity();
    int best_post = -1;

    // Per-move slot dedupe: stamp array sized to the tour, epoch-invalidated.
    thread_local std::vector<std::uint32_t> slot_stamp;
    thread_local std::uint32_t slot_epoch = 0U;
    if (static_cast<int>(slot_stamp.size()) < n) {
        slot_stamp.assign(static_cast<std::size_t>(n), 0U);
        slot_epoch = 0U;
    }
    ++slot_epoch;
    if (slot_epoch == 0U) {  // wrapped: reset stamps
        std::fill(slot_stamp.begin(), slot_stamp.end(), 0U);
        slot_epoch = 1U;
    }

    // Gather the candidate slots first, then evaluate all endpoint distances in
    // one dist_many_from batch (SIMD + far better locality than one-at-a-time).
    thread_local std::vector<int> w_pred;
    thread_local std::vector<int> w_endpoints;
    thread_local std::vector<double> w_base;
    thread_local std::vector<double> w_dist;
    w_pred.clear(); w_endpoints.clear(); w_base.clear();

    auto gather_pred = [&](int pred) {
        if (pred < 0) { pred += n; }
        if (pred >= n) { pred -= n; }
        if (pred == remove_pos) { return; }
        if (slot_stamp[static_cast<std::size_t>(pred)] == slot_epoch) { return; }
        slot_stamp[static_cast<std::size_t>(pred)] = slot_epoch;
        const int succ = next_live_index(pred, remove_pos, n);
        if (succ == pred || succ == remove_pos) { return; }
        const int a = tour.nodes[static_cast<std::size_t>(pred)];
        const int b = tour.nodes[static_cast<std::size_t>(succ)];
        double base;
        if (pred == prev_idx) {
            base = inst.dist(prev, next);
        } else if ((pred + 1 == n ? 0 : pred + 1) == succ && tour.edge_valid && static_cast<int>(tour.edge_len.size()) == n) {
            base = tour.edge_len[static_cast<std::size_t>(pred)];
        } else {
            base = inst.dist(a, b);
        }
        w_pred.push_back(pred);
        w_endpoints.push_back(a);
        w_endpoints.push_back(b);
        w_base.push_back(base);
    };

    // (a) the local window around the removed slot (includes the vacated slot).
    for (int off = -window; off <= window; ++off) {
        gather_pred(remove_pos + off);
    }
    // (b) slots adjacent to in-tour members of the incoming node's KNN row.
    if (inst.knn_k > 0) {
        for (int r = 0; r < inst.knn_k; ++r) {
            const int w = inst.knn_at(add_node, r);
            if (w < 0 || w >= inst.N || tour.in_set[static_cast<std::size_t>(w)] == 0U) { continue; }
            const int wp = tour.pos[static_cast<std::size_t>(w)];
            if (wp < 0 || wp >= n) { continue; }
            gather_pred(wp);       // insert after w
            gather_pred(wp - 1);   // insert before w
        }
    }

    if (!w_pred.empty()) {
        w_dist.assign(w_endpoints.size(), 0.0);
        dist_many_from(inst, add_node, w_endpoints.data(), static_cast<int>(w_endpoints.size()), w_dist.data());
        for (int e = 0; e < static_cast<int>(w_pred.size()); ++e) {
            const double insert_cost = w_dist[static_cast<std::size_t>(2 * e)] + w_dist[static_cast<std::size_t>(2 * e + 1)] - w_base[static_cast<std::size_t>(e)];
            if (insert_cost < best_insert) {
                best_insert = insert_cost;
                const int pred = w_pred[static_cast<std::size_t>(e)];
                best_post = (pred < remove_pos) ? pred : (pred - 1);
            }
        }
    }

    if (best_post >= 0 && std::isfinite(best_insert)) {
        move.valid = true;
        move.post_pred = best_post;
        move.delta = best_insert - remove_gain;
        move.new_length = tour.length + move.delta;
    }
    return move;
}


SwapInsertionMove find_best_insert_after_remove(const Instance& inst, const Tour& tour, int remove_pos, int add_node) {
    SwapInsertionMove move;
    move.remove_pos = remove_pos;
    move.add_node = add_node;
    if (tour.k < 2 || remove_pos < 0 || remove_pos >= tour.k || add_node < 0 || add_node >= inst.N) {
        return move;
    }

    const int n = tour.k;
    const int prev_idx = (remove_pos == 0) ? (n - 1) : (remove_pos - 1);
    const int next_idx = (remove_pos + 1 == n) ? 0 : (remove_pos + 1);
    const int prev = tour.nodes[static_cast<std::size_t>(prev_idx)];
    const int removed = tour.nodes[static_cast<std::size_t>(remove_pos)];
    const int next = tour.nodes[static_cast<std::size_t>(next_idx)];
    const double remove_gain = inst.dist(prev, removed) + inst.dist(removed, next) - inst.dist(prev, next);

    if (n == 2) {
        move.valid = true;
        move.post_pred = 0;
        move.delta = inst.dist(removed, add_node) + inst.dist(add_node, removed) - tour.length;
        move.new_length = tour.length + move.delta;
        return move;
    }

    thread_local std::vector<int> pred_index;
    thread_local std::vector<int> endpoints;
    thread_local std::vector<double> base_edge;
    pred_index.clear();
    endpoints.clear();
    base_edge.clear();
    pred_index.reserve(static_cast<std::size_t>(n - 1));
    endpoints.reserve(static_cast<std::size_t>(2 * (n - 1)));
    base_edge.reserve(static_cast<std::size_t>(n - 1));
    for (int pred = 0; pred < n; ++pred) {
        if (pred == remove_pos) {
            continue;
        }
        const int succ = next_live_index(pred, remove_pos, n);
        if (succ == pred || succ == remove_pos) {
            continue;
        }
        const int a = tour.nodes[static_cast<std::size_t>(pred)];
        const int b = tour.nodes[static_cast<std::size_t>(succ)];
        pred_index.push_back(pred);
        endpoints.push_back(a);
        endpoints.push_back(b);
        if (pred == prev_idx) {
            base_edge.push_back(inst.dist(prev, next));
        } else if ((pred + 1 == n ? 0 : pred + 1) == succ && tour.edge_valid && static_cast<int>(tour.edge_len.size()) == n) {
            base_edge.push_back(tour.edge_len[static_cast<std::size_t>(pred)]);
        } else {
            base_edge.push_back(inst.dist(a, b));
        }
    }
    if (pred_index.empty()) {
        return move;
    }

    thread_local std::vector<double> endpoint_dist;
    endpoint_dist.assign(endpoints.size(), 0.0);
    dist_many_from(inst, add_node, endpoints.data(), static_cast<int>(endpoints.size()), endpoint_dist.data());

    double best_insert = std::numeric_limits<double>::infinity();
    int best_post = -1;
    for (int e = 0; e < static_cast<int>(pred_index.size()); ++e) {
        const double insert_cost = endpoint_dist[static_cast<std::size_t>(2 * e)] + endpoint_dist[static_cast<std::size_t>(2 * e + 1)] - base_edge[static_cast<std::size_t>(e)];
        if (insert_cost < best_insert) {
            best_insert = insert_cost;
            const int pred = pred_index[static_cast<std::size_t>(e)];
            best_post = (pred < remove_pos) ? pred : (pred - 1);
        }
    }
    if (best_post >= 0 && std::isfinite(best_insert)) {
        move.valid = true;
        move.post_pred = best_post;
        move.delta = best_insert - remove_gain;
        move.new_length = tour.length + move.delta;
    }
    return move;
}

int next_live_index_after_remove(int idx, int removed, int n) noexcept {
    int next = idx + 1;
    if (next >= n) { next = 0; }
    if (next == removed) {
        ++next;
        if (next >= n) { next = 0; }
    }
    return next;
}

int post_index_from_current(int current_index, int removed) noexcept {
    return (current_index < removed) ? current_index : (current_index - 1);
}

SwapMoveEval evaluate_swap_after_remove(const Instance& inst, const Tour& tour, int remove_pos, int add_node, const std::vector<int>* pred_positions) {
    SwapMoveEval best;
    if (tour.k < 3 || remove_pos < 0 || remove_pos >= tour.k || add_node < 0 || add_node >= inst.N) {
        return best;
    }
    const int removed = tour.nodes[static_cast<std::size_t>(remove_pos)];
    if (add_node != removed && tour.in_set[static_cast<std::size_t>(add_node)] != 0U) {
        return best;
    }
    const int prev_idx = (remove_pos == 0) ? (tour.k - 1) : (remove_pos - 1);
    const int next_idx = (remove_pos + 1 == tour.k) ? 0 : (remove_pos + 1);
    const double gap_len = inst.dist(tour.nodes[static_cast<std::size_t>(prev_idx)], tour.nodes[static_cast<std::size_t>(next_idx)]);
    const double remove_gain = tour.edge_len[static_cast<std::size_t>(prev_idx)] + tour.edge_len[static_cast<std::size_t>(remove_pos)] - gap_len;

    // Full scans amortize one batched distance pass over all tour nodes; short
    // candidate lists are cheaper with direct per-endpoint distances.
    thread_local std::vector<double> add_dist;
    const bool batched = (pred_positions == nullptr);
    if (batched) {
        add_dist.assign(static_cast<std::size_t>(tour.k), 0.0);
        dist_many_from(inst, add_node, tour.nodes.data(), tour.k, add_dist.data());
    }
    auto dist_to = [&](int idx) {
        return batched ? add_dist[static_cast<std::size_t>(idx)]
                       : inst.dist(add_node, tour.nodes[static_cast<std::size_t>(idx)]);
    };

    auto consider_pred = [&](int pred_idx) {
        if (pred_idx < 0 || pred_idx >= tour.k || pred_idx == remove_pos) {
            return;
        }
        const int succ_idx = next_live_index_after_remove(pred_idx, remove_pos, tour.k);
        if (succ_idx == pred_idx || succ_idx == remove_pos) {
            return;
        }
        const double base = (pred_idx == prev_idx) ? gap_len : tour.edge_len[static_cast<std::size_t>(pred_idx)];
        const double insert_cost = dist_to(pred_idx) + dist_to(succ_idx) - base;
        const double delta = insert_cost - remove_gain;
        if (!best.valid || delta < best.delta) {
            best.valid = true;
            best.delta = delta;
            best.post_remove_pred = post_index_from_current(pred_idx, remove_pos);
        }
    };

    if (pred_positions == nullptr) {
        for (int pred = 0; pred < tour.k; ++pred) {
            consider_pred(pred);
        }
    } else {
        for (int pred : *pred_positions) {
            consider_pred(pred);
        }
    }
    return best;
}

SwapMoveEval evaluate_move_after_remove(const Instance& inst, const Tour& tour, int remove_pos, const std::vector<int>* pred_positions) {
    SwapMoveEval best;
    if (tour.k < 5 || remove_pos < 0 || remove_pos >= tour.k) {
        return best;
    }
    const int node = tour.nodes[static_cast<std::size_t>(remove_pos)];
    const int prev_idx = (remove_pos == 0) ? (tour.k - 1) : (remove_pos - 1);
    const int next_idx = (remove_pos + 1 == tour.k) ? 0 : (remove_pos + 1);
    const double gap_len = inst.dist(tour.nodes[static_cast<std::size_t>(prev_idx)], tour.nodes[static_cast<std::size_t>(next_idx)]);
    const double remove_gain = tour.edge_len[static_cast<std::size_t>(prev_idx)] + tour.edge_len[static_cast<std::size_t>(remove_pos)] - gap_len;

    thread_local std::vector<double> node_dist;
    const bool batched = (pred_positions == nullptr);
    if (batched) {
        node_dist.assign(static_cast<std::size_t>(tour.k), 0.0);
        dist_many_from(inst, node, tour.nodes.data(), tour.k, node_dist.data());
    }
    auto dist_to = [&](int idx) {
        return batched ? node_dist[static_cast<std::size_t>(idx)]
                       : inst.dist(node, tour.nodes[static_cast<std::size_t>(idx)]);
    };

    auto consider_pred = [&](int pred_idx) {
        if (pred_idx < 0 || pred_idx >= tour.k || pred_idx == remove_pos || pred_idx == prev_idx) {
            return;
        }
        const int succ_idx = next_live_index_after_remove(pred_idx, remove_pos, tour.k);
        if (succ_idx == pred_idx || succ_idx == remove_pos) {
            return;
        }
        const double base = (pred_idx == prev_idx) ? gap_len : tour.edge_len[static_cast<std::size_t>(pred_idx)];
        const double insert_cost = dist_to(pred_idx) + dist_to(succ_idx) - base;
        const double delta = insert_cost - remove_gain;
        if (!best.valid || delta < best.delta) {
            best.valid = true;
            best.delta = delta;
            best.post_remove_pred = post_index_from_current(pred_idx, remove_pos);
        }
    };

    if (pred_positions == nullptr) {
        for (int pred = 0; pred < tour.k; ++pred) {
            consider_pred(pred);
        }
    } else {
        for (int pred : *pred_positions) {
            consider_pred(pred);
        }
    }
    return best;
}


void collect_add_candidates_into(const Instance& inst, const Tour& tour, int remove_pos, Rng& rng, int cap, std::vector<int>& candidates) {
    candidates.clear();
    if (static_cast<int>(candidates.capacity()) < cap) {
        candidates.reserve(static_cast<std::size_t>(cap));
    }
    const int removed = tour.nodes[static_cast<std::size_t>(remove_pos)];
    // O(1) dedupe via an epoch-stamped array. Bit-identical to the previous
    // push_unique linear scan (same membership predicate, same insertion
    // order, same cap behavior, same rng consumption below), but removes the
    // O(list) scan per push -- which at ~80 pushes per SA move was a leading
    // term of the proposal cost.
    thread_local std::vector<std::uint32_t> seen_stamp;
    thread_local std::uint32_t seen_epoch = 0U;
    if (static_cast<int>(seen_stamp.size()) < inst.N) {
        seen_stamp.assign(static_cast<std::size_t>(inst.N), 0U);
        seen_epoch = 0U;
    }
    ++seen_epoch;
    if (seen_epoch == 0U) {
        std::fill(seen_stamp.begin(), seen_stamp.end(), 0U);
        seen_epoch = 1U;
    }
    auto push = [&](int node) {
        if (node >= 0 && node < inst.N && tour.in_set[static_cast<std::size_t>(node)] == 0U) {
            if (static_cast<int>(candidates.size()) >= cap) {
                return;
            }
            if (seen_stamp[static_cast<std::size_t>(node)] == seen_epoch) {
                return;
            }
            seen_stamp[static_cast<std::size_t>(node)] = seen_epoch;
            candidates.push_back(node);
        }
    };
    if (inst.knn_k > 0) {
        const int lim = std::min(inst.knn_k, 24);
        for (int r = 0; r < lim; ++r) { push(inst.knn_at(removed, r)); }
        const int prev_pos = (remove_pos == 0) ? (tour.k - 1) : (remove_pos - 1);
        const int next_pos = (remove_pos + 1 == tour.k) ? 0 : (remove_pos + 1);
        const int prev = tour.nodes[static_cast<std::size_t>(prev_pos)];
        const int next = tour.nodes[static_cast<std::size_t>(next_pos)];
        for (int r = 0; r < std::min(inst.knn_k, 16); ++r) {
            push(inst.knn_at(prev, r));
            push(inst.knn_at(next, r));
        }
    }
    for (int trial = 0; trial < 24 && static_cast<int>(candidates.size()) < cap; ++trial) {
        push(rng.randint(inst.N));
    }
    if (candidates.empty() || inst.N <= 400) {
        for (int node = 0; node < inst.N && static_cast<int>(candidates.size()) < cap; ++node) {
            push(node);
        }
    }
    if (candidates.empty()) {
        candidates.push_back(removed);
    }
}

std::vector<int> collect_add_candidates(const Instance& inst, const Tour& tour, int remove_pos, Rng& rng, int cap) {
    std::vector<int> candidates;
    collect_add_candidates_into(inst, tour, remove_pos, rng, cap, candidates);
    return candidates;
}

int choose_swap_candidate(const Instance& inst, const Tour& tour, int remove_pos, Rng& rng) {
    thread_local std::vector<int> candidates;
    collect_add_candidates_into(inst, tour, remove_pos, rng, 96, candidates);
    return candidates[static_cast<std::size_t>(rng.randint(static_cast<int>(candidates.size())))];
}

void polish_elite_with_oracle(ElitePool& elite, const Instance& inst, const SolverOptions& options, bool full_tsp, int top_keep, SearchStats* stats) {
    if (top_keep <= 0) { return; }
    const auto elite_nodes = elite.export_nodes();
    const int limit = std::min(static_cast<int>(elite_nodes.size()), top_keep);
    for (int i = 0; i < limit; ++i) {
        Tour candidate;
        candidate.init(inst.N);
        candidate.set_tour(elite_nodes[static_cast<std::size_t>(i)], inst);
        if (external_oracle_polish_tour(candidate, inst, options.oracle, full_tsp, stats, !options.disable_two_opt)) {
            elite.try_add(candidate.nodes, candidate.length);
        }
    }
}

namespace {

inline void subset_insert_best(int* ids, double* d2s, int m, int& count, int node, double d2) {
    if (count == m) {
        const int worst = count - 1;
        if (d2 > d2s[worst] || (d2 == d2s[worst] && node >= ids[worst])) {
            return;
        }
    }
    int pos = (count < m) ? count : (m - 1);
    while (pos > 0
           && (d2 < d2s[pos - 1]
               || (d2 == d2s[pos - 1] && node < ids[pos - 1]))) {
        ids[pos] = ids[pos - 1];
        d2s[pos] = d2s[pos - 1];
        --pos;
    }
    ids[pos] = node;
    d2s[pos] = d2;
    if (count < m) {
        ++count;
    }
}

inline bool instance_has_grid(const Instance& inst) noexcept {
    return inst.gx > 0 && inst.gy > 0 && inst.cell_size > 0.0
        && !inst.cell_points.empty()
        && static_cast<int>(inst.cell_x.size()) == inst.N
        && static_cast<int>(inst.cell_y.size()) == inst.N
        && static_cast<int>(inst.cell_begin.size()) == inst.gx * inst.gy + 1;
}

} // namespace

void build_subset_candidates(const Instance& inst, const Tour& tour, int m, SubsetCandidateTable& table) {
    table.m = std::max(0, std::min(m, std::max(0, tour.k - 1)));
    table.row_of_node.assign(static_cast<std::size_t>(inst.N), -1);
    const std::size_t rows = static_cast<std::size_t>(std::max(0, tour.k));
    table.ids.assign(rows * static_cast<std::size_t>(table.m), -1);
    table.dist.assign(rows * static_cast<std::size_t>(table.m),
                      std::numeric_limits<double>::infinity());
    if (table.m <= 0) {
        return;
    }
    for (int i = 0; i < tour.k; ++i) {
        table.row_of_node[static_cast<std::size_t>(tour.nodes[static_cast<std::size_t>(i)])] = i;
    }

    const bool have_grid = instance_has_grid(inst);
    detail::GenerationMarks periodic_cell_marks;
    for (int i = 0; i < tour.k; ++i) {
        const int a = tour.nodes[static_cast<std::size_t>(i)];
        int* ids = &table.ids[static_cast<std::size_t>(i) * static_cast<std::size_t>(table.m)];
        double* d2s = &table.dist[static_cast<std::size_t>(i) * static_cast<std::size_t>(table.m)];
        int count = 0;
        if (have_grid) {
            const int cx = inst.cell_x[static_cast<std::size_t>(a)];
            const int cy = inst.cell_y[static_cast<std::size_t>(a)];
            auto scan_cell = [&](int cell) {
                for (int p = inst.cell_begin[static_cast<std::size_t>(cell)];
                     p < inst.cell_begin[static_cast<std::size_t>(cell + 1)];
                     ++p) {
                    const int v = inst.cell_points[static_cast<std::size_t>(p)];
                    if (v == a || tour.in_set[static_cast<std::size_t>(v)] == 0U) {
                        continue;
                    }
                    subset_insert_best(ids, d2s, table.m, count, v, inst.dist2(a, v));
                }
            };

            if (inst.periodic) {
                const std::size_t cells =
                    static_cast<std::size_t>(inst.gx) * static_cast<std::size_t>(inst.gy);
                periodic_cell_marks.begin(cells);
                const Point& query = inst.points[static_cast<std::size_t>(a)];
                const double offset_x =
                    query.x - (inst.grid_min_x + static_cast<double>(cx) * inst.cell_size);
                const double offset_y =
                    query.y - (inst.grid_min_y + static_cast<double>(cy) * inst.cell_size);
                const int max_ring = detail::periodic_max_ring(inst.gx, inst.gy);
                for (int radius = 0; radius <= max_ring; ++radius) {
                    detail::visit_periodic_ring_unique(
                        cx,
                        cy,
                        radius,
                        inst.gx,
                        inst.gy,
                        periodic_cell_marks,
                        scan_cell);
                    if (periodic_cell_marks.visited() == cells) {
                        break;
                    }
                    if (count == table.m) {
                        const long double lower_bound =
                            detail::periodic_unvisited_distance2_lower_bound(
                                offset_x,
                                offset_y,
                                inst.cell_size,
                                inst.cell_size,
                                inst.gx,
                                inst.gy,
                                radius);
                        if (static_cast<long double>(d2s[table.m - 1]) < lower_bound) {
                            break;
                        }
                    }
                }
            } else {
                auto visit_cell = [&](int vx, int vy) {
                    if (vx < 0 || vx >= inst.gx || vy < 0 || vy >= inst.gy) {
                        return;
                    }
                    scan_cell(vy * inst.gx + vx);
                };
                const int max_ring = std::max(inst.gx, inst.gy);
                for (int radius = 0; radius <= max_ring; ++radius) {
                    if (count == table.m) {
                        // Every point in an unvisited ring is at least this far
                        // from the query cell. Use a strict comparison so a
                        // distance tie can still be resolved by node id.
                        const double bound =
                            (static_cast<double>(radius) - 1.0) * inst.cell_size;
                        if (bound > 0.0 && d2s[table.m - 1] < bound * bound) {
                            break;
                        }
                    }
                    const int xmin = cx - radius;
                    const int xmax = cx + radius;
                    const int ymin = cy - radius;
                    const int ymax = cy + radius;
                    if (radius == 0) {
                        visit_cell(cx, cy);
                    } else {
                        for (int x = xmin; x <= xmax; ++x) {
                            visit_cell(x, ymin);
                            visit_cell(x, ymax);
                        }
                        for (int y = ymin + 1; y <= ymax - 1; ++y) {
                            visit_cell(xmin, y);
                            visit_cell(xmax, y);
                        }
                    }
                    if (radius > 0
                        && xmin < 0 && xmax >= inst.gx
                        && ymin < 0 && ymax >= inst.gy) {
                        break;
                    }
                }
            }
        } else {
            for (int j = 0; j < tour.k; ++j) {
                const int v = tour.nodes[static_cast<std::size_t>(j)];
                if (v != a) {
                    subset_insert_best(ids, d2s, table.m, count, v, inst.dist2(a, v));
                }
            }
        }
        if (count != table.m) {
            throw std::logic_error("subset candidate search did not find enough members");
        }
        for (int rank = 0; rank < count; ++rank) {
            d2s[rank] = std::sqrt(d2s[rank]);
        }
    }
}

const SubsetCandidateTable* maybe_subset_candidates(const Instance& inst, const Tour& tour) {
    if (tour.k >= inst.N || tour.k < 5) {
        return nullptr;
    }
    if (!instance_has_grid(inst) && tour.k > 2048) {
        return nullptr;
    }
    constexpr int kSubsetCandM = 16;
    thread_local SubsetCandidateTable table;
    build_subset_candidates(inst, tour, kSubsetCandM, table);
    return &table;
}

int effective_sa_iters(const SolverOptions& options, int k, int N) noexcept {
    const long long base = static_cast<long long>(options.sa_iters);
    const long long extra = static_cast<long long>(options.sa_iters_per_k) * static_cast<long long>(std::max(0, k));
    const long long pool = static_cast<long long>(options.sa_iters_per_n) * static_cast<long long>(std::max(0, N));
    const long long total = base + extra + pool;
    if (total < 0) {
        return 0;
    }
    if (total > static_cast<long long>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(total);
}

void perturb_three_cut(Tour& tour, Rng& rng) {
    if (tour.k < 8) {
        return;
    }
    int c[3] = {rng.randint(tour.k), rng.randint(tour.k), rng.randint(tour.k)};
    std::sort(c, c + 3);
    if (c[0] == c[1] || c[1] == c[2]) {
        return;
    }
    std::vector<int> next;
    next.reserve(tour.nodes.size());
    for (int i = 0; i <= c[0]; ++i) { next.push_back(tour.nodes[static_cast<std::size_t>(i)]); }
    for (int i = c[1] + 1; i <= c[2]; ++i) { next.push_back(tour.nodes[static_cast<std::size_t>(i)]); }
    for (int i = c[0] + 1; i <= c[1]; ++i) { next.push_back(tour.nodes[static_cast<std::size_t>(i)]); }
    for (int i = c[2] + 1; i < tour.k; ++i) { next.push_back(tour.nodes[static_cast<std::size_t>(i)]); }
    tour.set_tour_only(next);
}


// ---------------------------------------------------------------------------
// SubsetIndex: live spatial index over the current subset members.
// ---------------------------------------------------------------------------

int SubsetIndex::cell_of(const Instance& inst, int node) const noexcept {
    const Point& point = inst.points[static_cast<std::size_t>(node)];
    const double origin_x = inst.periodic ? 0.0 : inst.min_x;
    const double origin_y = inst.periodic ? 0.0 : inst.min_y;
    const double relative_x = point.x - origin_x;
    const double relative_y = point.y - origin_y;
    int cell_x = static_cast<int>(relative_x / cell_len_);
    int cell_y = static_cast<int>(relative_y / cell_len_);
    cell_x = std::clamp(cell_x, 0, cells_side_ - 1);
    cell_y = std::clamp(cell_y, 0, cells_side_ - 1);
    return cell_y * cells_side_ + cell_x;
}

void SubsetIndex::relink(const Instance& inst) {
    head_.assign(static_cast<std::size_t>(cells_side_) * static_cast<std::size_t>(cells_side_), -1);
    nonempty_ = 0;
    for (const int node : members_) {
        const int c = cell_of(inst, node);
        if (head_[static_cast<std::size_t>(c)] < 0) { ++nonempty_; }
        next_[static_cast<std::size_t>(node)] = head_[static_cast<std::size_t>(c)];
        prev_[static_cast<std::size_t>(node)] = -1;
        if (head_[static_cast<std::size_t>(c)] >= 0) {
            prev_[static_cast<std::size_t>(head_[static_cast<std::size_t>(c)])] = node;
        }
        head_[static_cast<std::size_t>(c)] = node;
    }
}

void SubsetIndex::build(const Instance& inst, const Tour& tour) {
    const int k = tour.k;
    next_.assign(static_cast<std::size_t>(inst.N), -1);
    prev_.assign(static_cast<std::size_t>(inst.N), -1);
    members_.assign(tour.nodes.begin(), tour.nodes.begin() + k);
    count_ = k;
    // One member per cell for a subset spread over the whole domain. The cap
    // bounds memory when the subset contracts and the grid is refined.
    const int base = std::max(1, static_cast<int>(std::lround(std::sqrt(static_cast<double>(std::max(1, k))))));
    cells_side_ = base;
    cells_side_cap_ = std::min(4096, base * 8);
    const double span = std::max(inst.side, 1e-12);
    cell_len_ = span / static_cast<double>(cells_side_);
    relink(inst);
}

void SubsetIndex::add_member(const Instance& inst, int node) {
    const int c = cell_of(inst, node);
    if (head_[static_cast<std::size_t>(c)] < 0) { ++nonempty_; }
    next_[static_cast<std::size_t>(node)] = head_[static_cast<std::size_t>(c)];
    prev_[static_cast<std::size_t>(node)] = -1;
    if (head_[static_cast<std::size_t>(c)] >= 0) {
        prev_[static_cast<std::size_t>(head_[static_cast<std::size_t>(c)])] = node;
    }
    head_[static_cast<std::size_t>(c)] = node;
    ++count_;
    // As the subset contracts, members pile into few cells and queries degrade
    // towards a linear scan. Refine the grid when the average occupancy of a
    // live cell gets high; the rebuild is O(k) and happens O(log) times.
    if (nonempty_ > 0 && cells_side_ < cells_side_cap_
        && count_ > 3 * nonempty_) {
        members_.clear();
        members_.reserve(static_cast<std::size_t>(count_));
        const int cells = cells_side_ * cells_side_;
        for (int c2 = 0; c2 < cells; ++c2) {
            for (int v = head_[static_cast<std::size_t>(c2)]; v >= 0; v = next_[static_cast<std::size_t>(v)]) {
                members_.push_back(v);
            }
        }
        cells_side_ = std::min(cells_side_cap_, cells_side_ * 2);
        cell_len_ = std::max(inst.side, 1e-12) / static_cast<double>(cells_side_);
        relink(inst);
    }
}

void SubsetIndex::remove_member(const Instance& inst, int node) {
    const int c = cell_of(inst, node);
    const int pv = prev_[static_cast<std::size_t>(node)];
    const int nx = next_[static_cast<std::size_t>(node)];
    if (pv >= 0) { next_[static_cast<std::size_t>(pv)] = nx; }
    else { head_[static_cast<std::size_t>(c)] = nx; }
    if (nx >= 0) { prev_[static_cast<std::size_t>(nx)] = pv; }
    prev_[static_cast<std::size_t>(node)] = -1;
    next_[static_cast<std::size_t>(node)] = -1;
    if (head_[static_cast<std::size_t>(c)] < 0) { --nonempty_; }
    --count_;
}

void SubsetIndex::nearest(const Instance& inst, int query_node, int m, int exclude,
                          std::vector<int>& out, int max_rings) const {
    out.clear();
    if (m <= 0 || count_ <= 0 || query_node < 0 || query_node >= inst.N) {
        return;
    }

    const int query_cell = cell_of(inst, query_node);
    const int query_cell_x = query_cell % cells_side_;
    const int query_cell_y = query_cell / cells_side_;
    const Point& query = inst.points[static_cast<std::size_t>(query_node)];
    const double origin_x = inst.periodic ? 0.0 : inst.min_x;
    const double origin_y = inst.periodic ? 0.0 : inst.min_y;
    const double query_offset_x =
        query.x - (origin_x + static_cast<double>(query_cell_x) * cell_len_);
    const double query_offset_y =
        query.y - (origin_y + static_cast<double>(query_cell_y) * cell_len_);

    // Best-m by (squared distance, id): m is small (<= 64), so insertion
    // sorting a flat array is faster than a heap and gives deterministic ties.
    thread_local std::vector<double> best_distance2;
    thread_local std::vector<int> best_node;
    thread_local detail::GenerationMarks cell_marks;
    thread_local detail::GenerationMarks node_marks;
    best_distance2.clear();
    best_node.clear();
    node_marks.begin(static_cast<std::size_t>(inst.N));

    auto consider = [&](int node) {
        if (!node_marks.mark(static_cast<std::size_t>(node))) {
            return;
        }
        if (node == exclude || node == query_node) {
            return;
        }
        const double distance2 = inst.dist2(query_node, node);
        const int have = static_cast<int>(best_node.size());
        if (have >= m) {
            const int worst = have - 1;
            if (distance2 > best_distance2[static_cast<std::size_t>(worst)]
                || (distance2 == best_distance2[static_cast<std::size_t>(worst)]
                    && node >= best_node[static_cast<std::size_t>(worst)])) {
                return;
            }
        }

        int position = have;
        if (have < m) {
            best_distance2.push_back(distance2);
            best_node.push_back(node);
        } else {
            position = have - 1;
            best_distance2[static_cast<std::size_t>(position)] = distance2;
            best_node[static_cast<std::size_t>(position)] = node;
        }
        while (position > 0) {
            const std::size_t previous = static_cast<std::size_t>(position - 1);
            const std::size_t current = static_cast<std::size_t>(position);
            const bool swap_needed =
                best_distance2[current] < best_distance2[previous]
                || (best_distance2[current] == best_distance2[previous]
                    && best_node[current] < best_node[previous]);
            if (!swap_needed) {
                break;
            }
            std::swap(best_distance2[current], best_distance2[previous]);
            std::swap(best_node[current], best_node[previous]);
            --position;
        }
    };

    auto scan_cell = [&](int cell) {
        for (int node = head_[static_cast<std::size_t>(cell)];
             node >= 0;
             node = next_[static_cast<std::size_t>(node)]) {
            consider(node);
        }
    };

    if (inst.periodic) {
        const std::size_t cell_count =
            static_cast<std::size_t>(cells_side_) * static_cast<std::size_t>(cells_side_);
        cell_marks.begin(cell_count);
        int last_ring = detail::periodic_max_ring(cells_side_, cells_side_);
        if (max_rings >= 0) {
            last_ring = std::min(last_ring, max_rings);
        }
        for (int radius = 0; radius <= last_ring; ++radius) {
            detail::visit_periodic_ring_unique(
                query_cell_x,
                query_cell_y,
                radius,
                cells_side_,
                cells_side_,
                cell_marks,
                scan_cell);
            if (cell_marks.visited() == cell_count) {
                break;
            }
            if (static_cast<int>(best_node.size()) >= m) {
                const long double lower_bound =
                    detail::periodic_unvisited_distance2_lower_bound(
                        query_offset_x,
                        query_offset_y,
                        cell_len_,
                        cell_len_,
                        cells_side_,
                        cells_side_,
                        radius);
                if (static_cast<long double>(best_distance2[static_cast<std::size_t>(m - 1)])
                    < lower_bound) {
                    break;
                }
            }
        }
    } else {
        int last_ring = std::max({query_cell_x,
                                  cells_side_ - 1 - query_cell_x,
                                  query_cell_y,
                                  cells_side_ - 1 - query_cell_y});
        if (max_rings >= 0) {
            last_ring = std::min(last_ring, max_rings);
        }
        for (int radius = 0; radius <= last_ring; ++radius) {
            const int left = query_cell_x - radius;
            const int right = query_cell_x + radius;
            const int top = query_cell_y - radius;
            const int bottom = query_cell_y + radius;
            if (radius == 0) {
                scan_cell(query_cell);
            } else {
                const int x_min = std::max(0, left);
                const int x_max = std::min(cells_side_ - 1, right);
                if (top >= 0) {
                    const int row = top * cells_side_;
                    for (int x = x_min; x <= x_max; ++x) {
                        scan_cell(row + x);
                    }
                }
                if (bottom < cells_side_ && bottom != top) {
                    const int row = bottom * cells_side_;
                    for (int x = x_min; x <= x_max; ++x) {
                        scan_cell(row + x);
                    }
                }
                const int y_min = std::max(0, top + 1);
                const int y_max = std::min(cells_side_ - 1, bottom - 1);
                if (left >= 0) {
                    for (int y = y_min; y <= y_max; ++y) {
                        scan_cell(y * cells_side_ + left);
                    }
                }
                if (right < cells_side_ && right != left) {
                    for (int y = y_min; y <= y_max; ++y) {
                        scan_cell(y * cells_side_ + right);
                    }
                }
            }

            if (static_cast<int>(best_node.size()) >= m) {
                long double gap = std::numeric_limits<long double>::infinity();
                const long double cell_length = static_cast<long double>(cell_len_);
                const long double offset_x = static_cast<long double>(query_offset_x);
                const long double offset_y = static_cast<long double>(query_offset_y);
                if (query_cell_x - radius > 0) {
                    gap = std::min(gap,
                                   static_cast<long double>(radius) * cell_length + offset_x);
                }
                if (query_cell_x + radius < cells_side_ - 1) {
                    gap = std::min(gap,
                                   static_cast<long double>(radius + 1) * cell_length - offset_x);
                }
                if (query_cell_y - radius > 0) {
                    gap = std::min(gap,
                                   static_cast<long double>(radius) * cell_length + offset_y);
                }
                if (query_cell_y + radius < cells_side_ - 1) {
                    gap = std::min(gap,
                                   static_cast<long double>(radius + 1) * cell_length - offset_y);
                }
                if (std::isfinite(gap) && gap > 0.0L) {
                    gap = std::nextafter(gap, 0.0L);
                }
                const long double lower_bound = gap * gap;
                if (!std::isfinite(gap)
                    || static_cast<long double>(
                           best_distance2[static_cast<std::size_t>(m - 1)]) < lower_bound) {
                    break;
                }
            }
        }
    }

    out.assign(best_node.begin(), best_node.end());
}



SwapInsertionMove find_best_insert_after_remove_spatial(const Instance& inst, const Tour& tour,
                                                        const SubsetIndex& index, int remove_pos,
                                                        int add_node, int neighbors, int window) {
    SwapInsertionMove move;
    move.remove_pos = remove_pos;
    move.add_node = add_node;
    const int n = tour.k;
    if (n < 2 || remove_pos < 0 || remove_pos >= n || add_node < 0 || add_node >= inst.N) {
        return move;
    }
    if (neighbors >= n || n <= 2 * window + 2) {
        // Saturated: every slot would be gathered anyway. Use the exact scan --
        // this is also the invariant the unit test pins.
        return find_best_insert_after_remove(inst, tour, remove_pos, add_node);
    }
    const int prev_idx = (remove_pos == 0) ? (n - 1) : (remove_pos - 1);
    const int next_idx = (remove_pos + 1 == n) ? 0 : (remove_pos + 1);
    const int prev = tour.nodes[static_cast<std::size_t>(prev_idx)];
    const int removed = tour.nodes[static_cast<std::size_t>(remove_pos)];
    const int next = tour.nodes[static_cast<std::size_t>(next_idx)];
    const double remove_gain = inst.dist(prev, removed) + inst.dist(removed, next) - inst.dist(prev, next);

    double best_insert = std::numeric_limits<double>::infinity();
    int best_post = -1;

    thread_local std::vector<std::uint32_t> sp_stamp;
    thread_local std::uint32_t sp_epoch = 0U;
    if (static_cast<int>(sp_stamp.size()) < n) {
        sp_stamp.assign(static_cast<std::size_t>(n), 0U);
        sp_epoch = 0U;
    }
    ++sp_epoch;
    if (sp_epoch == 0U) {
        std::fill(sp_stamp.begin(), sp_stamp.end(), 0U);
        sp_epoch = 1U;
    }

    thread_local std::vector<int> sp_pred;
    thread_local std::vector<int> sp_endpoints;
    thread_local std::vector<double> sp_base;
    thread_local std::vector<double> sp_dist;
    thread_local std::vector<int> sp_near;
    sp_pred.clear(); sp_endpoints.clear(); sp_base.clear();

    auto gather_pred = [&](int pred) {
        if (pred < 0) { pred += n; }
        if (pred >= n) { pred -= n; }
        if (pred == remove_pos) { return; }
        if (sp_stamp[static_cast<std::size_t>(pred)] == sp_epoch) { return; }
        sp_stamp[static_cast<std::size_t>(pred)] = sp_epoch;
        const int succ = next_live_index(pred, remove_pos, n);
        if (succ == pred || succ == remove_pos) { return; }
        const int a = tour.nodes[static_cast<std::size_t>(pred)];
        const int b = tour.nodes[static_cast<std::size_t>(succ)];
        double base;
        if (pred == prev_idx) {
            base = inst.dist(prev, next);
        } else if ((pred + 1 == n ? 0 : pred + 1) == succ && tour.edge_valid
                   && static_cast<int>(tour.edge_len.size()) == n) {
            base = tour.edge_len[static_cast<std::size_t>(pred)];
        } else {
            base = inst.dist(a, b);
        }
        sp_pred.push_back(pred);
        sp_endpoints.push_back(a);
        sp_endpoints.push_back(b);
        sp_base.push_back(base);
    };

    // (a) the vacated slot and a small local window around it.
    for (int off = -window; off <= window; ++off) {
        gather_pred(remove_pos + off);
    }
    // (b) THE POINT OF THIS KERNEL: the tour edges touching the nearest CURRENT
    // members of the incoming node. Unlike the static KNN row, this is populated
    // at any subset density, so a candidate drawn uniformly at random from the
    // whole square -- the only kind of proposal that can relocate a member out
    // of a bad region -- gets offered the slots where it actually belongs.
    index.nearest(inst, add_node, neighbors, removed, sp_near, 3);
    for (const int w : sp_near) {
        const int wp = tour.pos[static_cast<std::size_t>(w)];
        if (wp < 0 || wp >= n) { continue; }
        gather_pred(wp);      // insert after w
        gather_pred(wp - 1);  // insert before w
    }

    if (!sp_pred.empty()) {
        sp_dist.assign(sp_endpoints.size(), 0.0);
        dist_many_from(inst, add_node, sp_endpoints.data(), static_cast<int>(sp_endpoints.size()), sp_dist.data());
        for (int e = 0; e < static_cast<int>(sp_pred.size()); ++e) {
            const double insert_cost = sp_dist[static_cast<std::size_t>(2 * e)]
                                       + sp_dist[static_cast<std::size_t>(2 * e + 1)]
                                       - sp_base[static_cast<std::size_t>(e)];
            if (insert_cost < best_insert) {
                best_insert = insert_cost;
                const int pred = sp_pred[static_cast<std::size_t>(e)];
                best_post = (pred < remove_pos) ? pred : (pred - 1);
            }
        }
    }
    if (best_post < 0 || !std::isfinite(best_insert)) {
        return move;
    }
    move.post_pred = best_post;
    move.delta = best_insert - remove_gain;
    move.valid = true;
    return move;
}

} // namespace aldous_tsp
