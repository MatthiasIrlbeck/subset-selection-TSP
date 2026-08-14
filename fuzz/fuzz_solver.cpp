#include "aldous_tsp/solver.hpp"
#include "fuzz_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size > 16384U) {
        return 0;
    }
    aldous_tsp::fuzz::Reader reader(data, size);
    const int n = reader.bounded_int(3, 24);
    aldous_tsp::Instance instance;
    instance.periodic = (reader.byte() & 1U) != 0U;
    aldous_tsp::Rng point_rng(reader.u64());
    instance.generate(n, point_rng);
    instance.build_knn(std::min(reader.bounded_int(2, 12), n - 1));

    aldous_tsp::SolverOptions options;
    options.seed = reader.u64();
    options.subset_restarts = reader.bounded_int(1, 2);
    options.restart_threads = 1;
    options.sa_iters = reader.bounded_int(0, 48);
    options.sa_iters_per_k = 0;
    options.sa_iters_per_n = 0;
    options.tsp_candidate_starts = reader.bounded_int(1, 3);
    options.tsp_restarts = reader.bounded_int(1, 2);
    options.tsp_ils = reader.bounded_int(0, 4);
    options.tsp_patience = 2;
    options.final_exhaustive_k = 32;
    options.subset_swap_descent_passes = reader.bounded_int(0, 1);
    options.pair_exchange_passes = reader.bounded_int(0, 1);
    options.pair_exchange_max_k = 64;
    options.ruin_recreate_rounds = reader.bounded_int(0, 1);
    options.ejection_chain_starts = reader.bounded_int(0, 1);
    options.ejection_chain_depth = reader.bounded_int(1, 3);
    options.path_relink_top = 0;
    options.racing_candidates = 0;
    options.staged_search = (reader.byte() & 1U) != 0U;
    options.strong_polish_finalists = 1;

    aldous_tsp::Rng search_rng(reader.u64());
    const bool full = (reader.byte() & 3U) == 0U;
    const int k = full ? n : reader.bounded_int(3, n);
    const aldous_tsp::SolveResult result = full
        ? aldous_tsp::solve_tsp(instance, search_rng, options)
        : aldous_tsp::solve_subset(instance, k, search_rng, options);
    if (result.tour.k != k || !result.tour.check_invariants()
        || !std::isfinite(result.tour.length)) {
        __builtin_trap();
    }
    const double exact = aldous_tsp::cycle_length(instance, result.tour.nodes);
    if (std::fabs(result.tour.length - exact) > 1.0e-7) {
        __builtin_trap();
    }
    return 0;
}
