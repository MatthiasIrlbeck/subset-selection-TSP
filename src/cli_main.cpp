#include "cli_internal.hpp"

#include "aldous_tsp/experiment.hpp"
#include "generated_options.hpp"
#include "json_writer.hpp"
#include "sha256.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <new>
#include <sstream>

namespace aldous_tsp {
namespace {

using CliClock = std::chrono::steady_clock;

const char* commit_state_name(const OutputCommitState state) noexcept {
    switch (state) {
        case OutputCommitState::NotCommitted: return "not-committed";
        case OutputCommitState::Committed: return "committed";
        case OutputCommitState::FileDurable: return "file-durable";
        case OutputCommitState::FullyDurable: return "fully-durable";
    }
    return "unknown";
}

std::string make_output_receipt(const RunOptions& opt,
                                const ResultsDocument& doc,
                                const std::string& result_text,
                                const double serialization_seconds,
                                const AtomicWriteResult& write_result,
                                const double end_to_end_seconds) {
    std::ostringstream out;
    JsonWriter writer(out);
    out << "{\n  \"receipt_schema_version\": 1,\n"
        << "  \"result_path\": ";
    writer.string(opt.output_path);
    out << ",\n  \"result_sha256\": ";
    writer.string(detail::sha256_hex(result_text));
    out << ",\n  \"result_bytes\": " << result_text.size()
        << ",\n  \"requested_durability\": ";
    writer.string(output_durability_name(opt.output_durability));
    out << ",\n  \"commit_state\": ";
    writer.string(commit_state_name(write_result.state));
    out << ",\n  \"timing\": {\n"
        << "    \"solver_wall_seconds\": ";
    writer.number(doc.solver_wall_seconds);
    out << ",\n    \"control_reference_seconds\": ";
    writer.number(doc.control_reference_seconds);
    out << ",\n    \"aggregation_seconds\": ";
    writer.number(doc.aggregation_seconds);
    out << ",\n    \"experiment_wall_seconds\": ";
    writer.number(doc.experiment_wall_seconds);
    out << ",\n    \"serialization_seconds\": ";
    writer.number(serialization_seconds);
    out << ",\n    \"output_write_seconds\": ";
    writer.number(write_result.write_seconds);
    out << ",\n    \"output_synchronization_seconds\": ";
    writer.number(write_result.synchronization_seconds);
    out << ",\n    \"output_commit_seconds\": ";
    writer.number(write_result.commit_seconds);
    out << ",\n    \"output_total_seconds\": ";
    writer.number(write_result.total_seconds);
    out << ",\n    \"process_end_to_end_seconds\": ";
    writer.number(end_to_end_seconds);
    out << "\n  }\n}\n";
    return out.str();
}

int cli_main_impl(int argc, char** argv) {
    const auto process_start = CliClock::now();
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
    const auto serialization_start = CliClock::now();
    const std::string result_text = results_to_json(doc);
    const double serialization_seconds = std::chrono::duration<double>(
        CliClock::now() - serialization_start).count();
    const AtomicWriteResult write_result = write_text_file_atomic(
        opt.output_path, result_text, replace_policy, opt.output_durability);
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

    const double end_to_end_seconds = std::chrono::duration<double>(
        CliClock::now() - process_start).count();
    const std::string receipt_path = opt.output_path + ".receipt";
    const AtomicWriteResult receipt_result = write_text_file_atomic(
        receipt_path,
        make_output_receipt(opt, doc, result_text, serialization_seconds,
                            write_result, end_to_end_seconds),
        ReplacePolicy::ReplaceExisting,
        opt.output_durability);
    if (!receipt_result.satisfies(opt.output_durability)) {
        std::fprintf(stderr,
                     "Output receipt warning: primary result is committed, but %s could not "
                     "be written at the requested durability: %s\n",
                     receipt_path.c_str(), receipt_result.message.c_str());
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
    std::printf("Timing receipt written to %s (end-to-end %.3fs)\n",
                receipt_path.c_str(), end_to_end_seconds);
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
