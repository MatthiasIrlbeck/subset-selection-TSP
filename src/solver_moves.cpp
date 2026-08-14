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

BatchedSwapResult best_batched_swap(const Instance& inst,
                                    const Tour& tour,
                                    const std::vector<SwapCandidatePair>& candidates) {
    BatchedSwapResult best;
    const int k = tour.k;
    if (k < 3 || !tour.edge_valid || candidates.empty() || inst.N <= 0) {
        return best;
    }

    struct RemovalProfile {
        int prev = -1;
        int next = -1;
        double gap = 0.0;
        double gain = 0.0;
    };
    thread_local std::vector<RemovalProfile> removals;
    removals.resize(static_cast<std::size_t>(k));
    for (int ri = 0; ri < k; ++ri) {
        RemovalProfile& removal = removals[static_cast<std::size_t>(ri)];
        removal.prev = (ri == 0) ? (k - 1) : (ri - 1);
        removal.next = (ri + 1 == k) ? 0 : (ri + 1);
        removal.gap = inst.dist(tour.nodes[static_cast<std::size_t>(removal.prev)],
                                tour.nodes[static_cast<std::size_t>(removal.next)]);
        removal.gain = tour.edge_len[static_cast<std::size_t>(removal.prev)]
            + tour.edge_len[static_cast<std::size_t>(ri)] - removal.gap;
    }

    // Intrusive per-add linked lists avoid a vector allocation per group while
    // retaining every candidate's original index for deterministic tie order.
    thread_local std::vector<int> group_head;
    thread_local std::vector<int> group_tail;
    thread_local std::vector<int> next_candidate;
    thread_local std::vector<int> unique_adds;
    // Reset only add-node slots touched by the preceding invocation. Clearing
    // all N slots would reintroduce an O(N) term for sparse candidate sets.
    for (int add_node : unique_adds) {
        group_head[static_cast<std::size_t>(add_node)] = -1;
        group_tail[static_cast<std::size_t>(add_node)] = -1;
    }
    unique_adds.clear();
    const std::size_t node_count = static_cast<std::size_t>(inst.N);
    if (group_head.size() < node_count) {
        group_head.resize(node_count, -1);
        group_tail.resize(node_count, -1);
    }
    next_candidate.assign(candidates.size(), -1);
    unique_adds.reserve(std::min<std::size_t>(candidates.size(), node_count));
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const SwapCandidatePair& candidate = candidates[index];
        if (candidate.remove_pos < 0 || candidate.remove_pos >= k
            || candidate.add_node < 0 || candidate.add_node >= inst.N) {
            continue;
        }
        const int removed = tour.nodes[static_cast<std::size_t>(candidate.remove_pos)];
        if (candidate.add_node != removed
            && tour.in_set[static_cast<std::size_t>(candidate.add_node)] != 0U) {
            continue;
        }
        int& head = group_head[static_cast<std::size_t>(candidate.add_node)];
        int& tail = group_tail[static_cast<std::size_t>(candidate.add_node)];
        const int current = static_cast<int>(index);
        if (head < 0) {
            head = current;
            unique_adds.push_back(candidate.add_node);
        } else {
            next_candidate[static_cast<std::size_t>(tail)] = current;
        }
        tail = current;
    }

    thread_local std::vector<double> add_dist;
    add_dist.resize(static_cast<std::size_t>(k));
    for (int add_node : unique_adds) {
        dist_many_from(inst, add_node, tour.nodes.data(), k, add_dist.data());

        double top_cost[3] = {std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity()};
        int top_pred[3] = {-1, -1, -1};
        for (int pred = 0; pred < k; ++pred) {
            const int succ = (pred + 1 == k) ? 0 : (pred + 1);
            const double cost = add_dist[static_cast<std::size_t>(pred)]
                + add_dist[static_cast<std::size_t>(succ)]
                - tour.edge_len[static_cast<std::size_t>(pred)];
            if (cost < top_cost[2]) {
                top_cost[2] = cost;
                top_pred[2] = pred;
                if (top_cost[2] < top_cost[1]) {
                    std::swap(top_cost[1], top_cost[2]);
                    std::swap(top_pred[1], top_pred[2]);
                }
                if (top_cost[1] < top_cost[0]) {
                    std::swap(top_cost[0], top_cost[1]);
                    std::swap(top_pred[0], top_pred[1]);
                }
            }
        }

        for (int index = group_head[static_cast<std::size_t>(add_node)];
             index >= 0;
             index = next_candidate[static_cast<std::size_t>(index)]) {
            const SwapCandidatePair& candidate = candidates[static_cast<std::size_t>(index)];
            const RemovalProfile& removal = removals[static_cast<std::size_t>(candidate.remove_pos)];
            double insert_cost = add_dist[static_cast<std::size_t>(removal.prev)]
                + add_dist[static_cast<std::size_t>(removal.next)] - removal.gap;
            int insert_pred = removal.prev;
            for (int rank = 0; rank < 3; ++rank) {
                const int pred = top_pred[rank];
                if (pred < 0 || pred == candidate.remove_pos || pred == removal.prev) {
                    continue;
                }
                const double cost = top_cost[rank];
                if (cost < insert_cost || (cost == insert_cost && pred < insert_pred)) {
                    insert_cost = cost;
                    insert_pred = pred;
                }
                break;
            }
            const double delta = insert_cost - removal.gain;
            const std::size_t candidate_index = static_cast<std::size_t>(index);
            if (!best.valid || delta < best.delta
                || (delta == best.delta && candidate_index < best.candidate_index)) {
                best.valid = true;
                best.delta = delta;
                best.remove_pos = candidate.remove_pos;
                best.add_node = add_node;
                best.post_remove_pred = post_index_from_current(insert_pred,
                                                                candidate.remove_pos);
                best.candidate_index = candidate_index;
            }
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


} // namespace aldous_tsp
