#include "solver_internal.hpp"

namespace aldous_tsp {

SolveResult solve_tsp(const Instance& inst, Rng& rng, const SolverOptions& options) {
    const auto start = Clock::now();
    SolveResult result;
    result.tour.init(inst.N);
    if (inst.N <= 0) { return result; }
    ElitePool elite(std::max(4, std::min(24, options.tsp_restarts + 8)), EliteMode::Cycle);
    const std::vector<int> all = all_nodes(inst.N);
    const int restarts = std::max(1, options.tsp_restarts);
    const double time_budget = options.time_budget_per_p;
    double best_tsp_len = std::numeric_limits<double>::infinity();
    for (int r = 0;; ++r) {
        if (r >= restarts) {
            if (time_budget <= 0.0) { break; }
            if (std::chrono::duration<double>(Clock::now() - start).count() >= time_budget) { break; }
        }
        std::vector<int> seed = (r == 0) ? farthest_insertion_order(inst, all) : nearest_neighbor_order(inst, all, rng.randint(inst.N));
        Tour tour;
        tour.init(inst.N);
        tour.set_tour(seed, inst);
        polish_tour(tour, inst, options, &result.stats, 2);
        Tour best_restart = tour;
        int no_improve = 0;
        const int ils = std::max(0, options.tsp_ils);
        for (int it = 0; it < ils; ++it) {
            Tour cand = best_restart;
            perturb_three_cut(cand, rng);
            cand.recompute_length(inst);
            polish_tour(cand, inst, options, &result.stats, 1);
            ++result.stats.tsp_ils_iterations;
            if (cand.length + kImprovementEps < best_restart.length) {
                best_restart = std::move(cand);
                no_improve = 0;
            } else if (++no_improve > std::max(0, options.tsp_patience)) {
                break;
            }
        }
        if (options.oracle.cfg.inline_feedback) {
            (void)external_oracle_polish_tour(best_restart, inst, options.oracle, true, &result.stats, !options.disable_two_opt);
        }
        const RestartKind kind = (r == 0)
            ? RestartKind::TspFarthestInsertion
            : RestartKind::TspNearestNeighbor;
        result.restarts.push_back(
            make_restart_record(inst, best_restart.nodes, best_restart.length, kind));
        if (best_restart.length < best_tsp_len - kImprovementEps) {
            best_tsp_len = best_restart.length;
            result.best_restart = static_cast<int>(result.restarts.size()) - 1;
        }
        elite.try_add(best_restart.nodes, best_restart.length);
        ++result.stats.tsp_restarts;
    }
    polish_elite_with_oracle(elite, inst, options, true, options.oracle.cfg.tsp_top, &result.stats);
    const auto nodes = elite.export_nodes();
    if (!nodes.empty()) {
        result.tour.set_tour(nodes.front(), inst);
        final_polish_tour(result.tour, inst, options, &result.stats, 2);
    }
    result.stats.tsp_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

} // namespace aldous_tsp
