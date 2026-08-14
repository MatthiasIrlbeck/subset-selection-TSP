#include "aldous_tsp/solver.hpp"

#include <utility>

namespace aldous_tsp {

TspSolver::TspSolver(SolverOptions options) : options_(std::move(options)) {}

SolveResult TspSolver::solve(const Instance& inst, Rng& rng) const {
    return solve_tsp(inst, rng, options_);
}

SolveResult TspSolver::solve(const PreparedInstance& inst, Rng& rng) const {
    return solve_tsp(inst, rng, options_);
}

SubsetSolver::SubsetSolver(SolverOptions options) : options_(std::move(options)) {}

SolveResult SubsetSolver::solve(const Instance& inst, int k, Rng& rng) const {
    return solve_subset(inst, k, rng, options_, nullptr, {});
}

SolveResult SubsetSolver::solve(const PreparedInstance& inst,
                                const int k,
                                Rng& rng) const {
    return solve_subset(inst, k, rng, options_, nullptr, {});
}

SolveResult SubsetSolver::solve_with_warm_start(const Instance& inst, int k, Rng& rng, const std::vector<int>& warm_start) const {
    return solve_subset(inst, k, rng, options_, &warm_start, {});
}

SolveResult SubsetSolver::solve_with_warm_start(
    const PreparedInstance& inst,
    const int k,
    Rng& rng,
    const std::vector<int>& warm_start) const {
    return solve_subset(inst, k, rng, options_, &warm_start, {});
}

} // namespace aldous_tsp
