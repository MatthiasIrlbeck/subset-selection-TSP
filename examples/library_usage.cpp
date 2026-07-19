#include <aldous_tsp/experiment.hpp>
#include <aldous_tsp/instance.hpp>
#include <aldous_tsp/solver.hpp>

#include <iostream>

int main() {
    aldous_tsp::Rng rng(2024);

    aldous_tsp::Instance instance;
    instance.generate(120, rng);
    instance.build_knn(32, aldous_tsp::KnnBackend::GridExact);

    aldous_tsp::SolverOptions options;
    options.tsp_restarts = 2;
    options.tsp_ils = 20;
    options.subset_restarts = 2;
    options.sa_iters = 500;
    options.final_exhaustive_k = 120;

    aldous_tsp::TspSolver tsp_solver(options);
    aldous_tsp::SolveResult tsp = tsp_solver.solve(instance, rng);

    aldous_tsp::SubsetSolver subset_solver(options);
    aldous_tsp::SolveResult subset = subset_solver.solve_with_warm_start(instance, 40, rng, tsp.tour.nodes);

    aldous_tsp::RunOptions run_options;
    run_options.N = 80;
    run_options.instances = 1;
    run_options.threads = 1;
    run_options.p_values = {0.25, 1.0};
    run_options.solver = options;
    run_options.solver.knn_k = 24;

    aldous_tsp::ExperimentRunner runner(run_options);
    aldous_tsp::ResultsDocument results = runner.run();

    std::cout << "full TSP length: " << tsp.tour.length << '\n';
    std::cout << "subset length: " << subset.tour.length << '\n';
    std::cout << "experiment p-values: " << results.summary.size() << '\n';
    return (tsp.tour.check_invariants() && subset.tour.check_invariants()) ? 0 : 1;
}
