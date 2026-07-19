#include "cli_internal.hpp"

#include "aldous_tsp/experiment.hpp"

namespace aldous_tsp {

int run_self_test() {
    int failed = 0;
    auto check = [&](const char* name, bool ok) {
        std::printf("  %-24s %s\n", name, ok ? "OK" : "FAILED");
        if (!ok) { ++failed; }
    };
    std::printf("Running self-tests...\n");

    check("rng-determinism", []() {
        Rng a(123), b(123);
        for (int i = 0; i < 20; ++i) {
            if (a.next_u64() != b.next_u64()) { return false; }
        }
        return true;
    }());

    check("knn-bruteforce", []() {
        Rng rng(1);
        Instance inst;
        inst.generate(40, rng);
        inst.build_knn(8, KnnBackend::GridExact);
        Rng verify(2);
        return inst.verify_knn(40, verify);
    }());

    check("exact-small-tsp", []() {
        Instance inst;
        inst.set_points({{0,0}, {1,0}, {1,1}, {0,1}});
        inst.build_knn(3, KnnBackend::GridExact);
        std::vector<int> cycle;
        double len = 0.0;
        if (!exact_small_tsp_cycle(inst, {0, 1, 2, 3}, cycle, len)) { return false; }
        return std::fabs(len - 4.0) < 1e-9;
    }());

    check("two-opt-crossing", []() {
        Instance inst;
        inst.set_points({{0,0}, {1,0}, {1,1}, {0,1}});
        inst.build_knn(3, KnnBackend::GridExact);
        Tour tour;
        tour.init(4);
        tour.set_tour({0, 2, 1, 3}, inst);
        const double before = tour.length;
        two_opt_descent(tour, inst, 10, nullptr);
        return tour.length < before - 1e-9 && std::fabs(tour.length - 4.0) < 1e-9 && tour.check_invariants();
    }());

    check("json-escape", []() {
        return json_escape("a\\b\"c\n") == "a\\\\b\\\"c\\n";
    }());

    check("solver-smoke", []() {
        RunOptions opt;
        opt.N = 30;
        opt.instances = 1;
        opt.threads = 1;
        opt.p_values = {0.2, 1.0};
        opt.solver.knn_k = 10;
        opt.solver.knn_backend = KnnBackend::GridExact;
        opt.solver.sa_iters = 10;
        opt.solver.subset_restarts = 1;
        opt.solver.tsp_restarts = 1;
        ExperimentRunner runner(opt);
        ResultsDocument doc = runner.run();
        return doc.instances_done == 1 && doc.summary.size() == 2U;
    }());

    if (failed == 0) {
        std::printf("All self-tests passed.\n");
    } else {
        std::printf("%d self-test(s) failed.\n", failed);
    }
    return failed == 0 ? 0 : 1;
}

} // namespace aldous_tsp
