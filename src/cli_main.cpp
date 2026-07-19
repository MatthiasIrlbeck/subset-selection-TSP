#include "cli_internal.hpp"

#include "aldous_tsp/experiment.hpp"

namespace aldous_tsp {

int cli_main(int argc, char** argv) {
    RunOptions opt;
    // CLI default: auto-derive restart parallelism from the leftover thread
    // budget (library default stays sequential). Results are invariant to
    // restart_threads outside time-budget mode.
    opt.solver.restart_threads = 0;
    bool self_test = false;
    if (!parse_args(argc, argv, opt, self_test)) {
        return 1;
    }
    if (self_test) {
        return run_self_test();
    }
    std::string err;
    if (!validate_options(opt, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (!build_oracle_context(opt.solver.oracle.cfg, opt.solver.oracle, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    if (opt.dump_config || opt.dry_run) {
        std::printf("Resolved config: %s\n", config_summary(opt).c_str());
    }
    if (opt.dry_run) {
        return 0;
    }
    if (std::filesystem::exists(opt.output_path) && !opt.force_output) {
        std::fprintf(stderr, "Refusing to overwrite existing output file %s (use --force or --output)\n", opt.output_path.c_str());
        return 1;
    }

    std::printf("========================================================================\n");
    std::printf("  ALDOUS SUBSET-SELECTION TSP · solver %s\n", kProjectVersion);
    std::printf("  %s\n", config_summary(opt).c_str());
    std::printf("========================================================================\n");
    std::fflush(stdout);

    ExperimentRunner runner(opt);
    ResultsDocument doc = runner.run([&](const ExperimentProgress& progress) {
        std::printf("  [done %d/%d] instance %d in %.2fs  elapsed %.0fs  ETA %.0fs\n",
                    progress.completed,
                    progress.total,
                    progress.instance_index + 1,
                    progress.instance_seconds,
                    progress.elapsed_seconds,
                    progress.eta_seconds);
        std::fflush(stdout);
    });

    if (doc.instances_done != opt.instances) {
        std::fprintf(stderr, "Only %d/%d instances completed successfully\n", doc.instances_done, opt.instances);
        return 2;
    }

    std::string write_err;
    if (!write_text_file_atomic(opt.output_path, results_to_json(doc), &write_err)) {
        std::fprintf(stderr, "%s\n", write_err.c_str());
        return 2;
    }

    std::printf("\nFINAL (N=%d, instances=%d, wall=%.1fs)\n", opt.N, opt.instances, doc.wall_seconds);
    std::printf("  %18s %5s %10s %10s %10s %10s\n", "p", "k", "mean", "stderr", "min", "max");
    for (double p : opt.p_values) {
        const PValueSummary& summary = doc.summary[p_value_key(p)];
        std::printf("  %18.10g %5d %10.5f %10.5f %10.5f %10.5f\n",
                    p,
                    summary.k,
                    summary.mean,
                    summary.stderr_value,
                    summary.min,
                    summary.max);
    }
    std::printf("Results written to %s\n", opt.output_path.c_str());
    return 0;
}

} // namespace aldous_tsp
