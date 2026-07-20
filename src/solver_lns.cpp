#include "solver_internal.hpp"

namespace aldous_tsp {

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


} // namespace aldous_tsp
