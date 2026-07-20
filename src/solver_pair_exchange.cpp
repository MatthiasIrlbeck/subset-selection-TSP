#include "solver_internal.hpp"

namespace aldous_tsp {

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


} // namespace aldous_tsp
