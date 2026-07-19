#pragma once

#include "aldous_tsp/config.hpp"
#include "aldous_tsp/restart.hpp"
#include "aldous_tsp/tour.hpp"

#include <vector>

namespace aldous_tsp {

struct SolveResult {
    Tour tour;
    SearchStats stats;
    // True only when the complete cardinality-k subset-selection problem was
    // solved exactly. Conditional tour polishing/lower bounds do not set it.
    bool exact_optimal = false;
    // One typed record per executed restart, in restart-index order.
    // `length` is in raw distance units. Kind codes/names are defined once in
    // restart_kinds.def and shared by the API, JSON schema, and analysis tools.
    std::vector<RestartRecord> restarts;
    // Index of the restart whose outcome had the best length before the
    // post-restart stages (relinking, oracle, final polish); -1 when no
    // restart ran. If this repeatedly equals the last executed restart on a
    // workload, the restart/SA budget is likely too small.
    int best_restart = -1;
};

// Optional search-controller request used by ExperimentRunner. The default
// public solve performs the independent population plus any configured warm
// restarts. A secondary p-sweep sets continuation_only so it does not repeat
// independent diagnostic draws under a different sweep seed.
struct SubsetSolveRequest {
    bool continuation_only = false;
};

std::vector<int> nearest_neighbor_order(const Instance& inst, const std::vector<int>& subset, int start_index);
std::vector<int> farthest_insertion_order(const Instance& inst, const std::vector<int>& subset);

bool exact_small_tsp_cycle(const Instance& inst, const std::vector<int>& set_nodes, std::vector<int>& best_cycle, double& best_len);

int two_opt_descent(Tour& tour, const Instance& inst, int max_passes, SearchStats* stats = nullptr);
int or_opt_1_descent(Tour& tour, const Instance& inst, int max_passes, SearchStats* stats = nullptr);
int subset_swap_descent(Tour& tour, const Instance& inst, int max_passes, SearchStats* stats = nullptr);

SolveResult solve_tsp(const Instance& inst, Rng& rng, const SolverOptions& options);
SolveResult solve_subset(const Instance& inst,
                         int k,
                         Rng& rng,
                         const SolverOptions& options,
                         const std::vector<int>* warm_start = nullptr,
                         const SubsetSolveRequest& request = {});

class TspSolver {
public:
    explicit TspSolver(SolverOptions options = {});

    [[nodiscard]] const SolverOptions& options() const noexcept { return options_; }
    [[nodiscard]] SolveResult solve(const Instance& inst, Rng& rng) const;

private:
    SolverOptions options_;
};

class SubsetSolver {
public:
    explicit SubsetSolver(SolverOptions options = {});

    [[nodiscard]] const SolverOptions& options() const noexcept { return options_; }
    [[nodiscard]] SolveResult solve(const Instance& inst, int k, Rng& rng) const;
    [[nodiscard]] SolveResult solve_with_warm_start(const Instance& inst, int k, Rng& rng, const std::vector<int>& warm_start) const;

private:
    SolverOptions options_;
};

} // namespace aldous_tsp
