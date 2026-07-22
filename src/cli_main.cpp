#include "cli_internal.hpp"

#include "aldous_tsp/experiment.hpp"
#include "generated_options.hpp"

#include <exception>
#include <new>

namespace aldous_tsp {
namespace {

int cli_main_impl(int argc, char** argv) {
    RunOptions opt;
    apply_generated_cli_defaults(opt);
    bool self_test = false;
    std::string parse_error;
    const CliParseOutcome parse_outcome =
        parse_args(argc, argv, opt, self_test, parse_error);
    if (parse_outcome == CliParseOutcome::ExitSuccess) {
        return 0;
    }
    if (parse_outcome == CliParseOutcome::Error) {
        std::fprintf(stderr, "%s\n", parse_error.c_str());
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
        std::fprintf(stderr, "Only %d/%d instances completed successfully\n",
                     doc.instances_done, opt.instances);
        return 2;
    }

    const ReplacePolicy replace_policy = opt.force_output
        ? ReplacePolicy::ReplaceExisting
        : ReplacePolicy::NoReplace;
    const AtomicWriteResult write_result = write_text_file_atomic(
        opt.output_path, results_to_json(doc), replace_policy, opt.output_durability);
    if (!write_result.satisfies(opt.output_durability)) {
        if (!write_result.message.empty()) {
            std::fprintf(stderr, "%s\n", write_result.message.c_str());
        } else if (write_result.committed()) {
            std::fprintf(stderr,
                         "Output was committed, but the requested durability level was not achieved\n");
        } else {
            std::fprintf(stderr, "Output could not be committed\n");
        }
        return write_result.committed() ? 2 : 1;
    }
    if (!write_result.message.empty()) {
        std::fprintf(stderr, "Output warning: %s\n", write_result.message.c_str());
    }

    std::printf("\nFINAL (N=%d, instances=%d, wall=%.1fs)\n",
                opt.N, opt.instances, doc.wall_seconds);
    std::printf("  %18s %5s %10s %10s %10s %10s\n",
                "p", "k", "mean", "stderr", "min", "max");
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

} // namespace

int cli_main(int argc, char** argv) {
    try {
        return cli_main_impl(argc, argv);
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "Fatal error: memory allocation failed\n");
        return 2;
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "Fatal error: %s\n", exception.what());
        return 2;
    } catch (...) {
        std::fprintf(stderr, "Fatal error: non-standard exception\n");
        return 2;
    }
}

} // namespace aldous_tsp
