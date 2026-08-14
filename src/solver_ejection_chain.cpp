#include "solver_internal.hpp"

namespace aldous_tsp {

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


} // namespace aldous_tsp
