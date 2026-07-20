#include "cli_internal.hpp"

#include "generated_options.hpp"

namespace aldous_tsp {

CliParseOutcome parse_args(const int argc,
                           char** argv,
                           RunOptions& opt,
                           bool& self_test,
                           std::string& error) {
    self_test = false;
    bool quick = false;
    if (!generated_quick_requested(argc, argv, quick, error)) {
        return CliParseOutcome::Error;
    }
    if (quick) {
        apply_quick_preset(opt);
    }

    for (int index = 1; index < argc; ++index) {
        const GeneratedCliParseResult result = parse_generated_cli_option(
            index, argc, argv, opt, self_test, error);
        switch (result) {
            case GeneratedCliParseResult::Matched:
                break;
            case GeneratedCliParseResult::ExitSuccess:
                return CliParseOutcome::ExitSuccess;
            case GeneratedCliParseResult::Error:
                return CliParseOutcome::Error;
            case GeneratedCliParseResult::NoMatch:
                error = "unknown argument: "
                    + std::string(argv[index] != nullptr ? argv[index] : "<null>");
                return CliParseOutcome::Error;
        }
    }
    return CliParseOutcome::Run;
}

} // namespace aldous_tsp
