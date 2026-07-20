#include "cli_internal.hpp"
#include "fuzz_common.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size > 16384U) {
        return 0;
    }
    std::vector<std::string> storage;
    storage.emplace_back("aldous_tsp");
    std::string current;
    current.reserve(128U);
    for (std::size_t i = 0; i < size && storage.size() < 65U; ++i) {
        const unsigned char ch = data[i];
        if (ch == 0U || ch == static_cast<unsigned char>('\n')) {
            if (!current.empty()) {
                storage.push_back(current);
                current.clear();
            }
        } else if (current.size() < 512U) {
            current.push_back(static_cast<char>(ch));
        }
    }
    if (!current.empty() && storage.size() < 65U) {
        storage.push_back(current);
    }
    // File-backed p grids are tested by ordinary unit tests. Never allow a
    // fuzzer mutation to open a device, FIFO, or enormous pseudo-file.
    for (std::size_t i = 1; i < storage.size(); ++i) {
        if (storage[i].rfind("--p-file", 0) == 0) {
            storage[i] = "--p-values=0.5";
        }
    }

    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (std::string& argument : storage) {
        argv.push_back(argument.data());
    }

    aldous_tsp::RunOptions options;
    options.solver.restart_threads = 0;
    bool self_test = false;
    std::string error;
    const aldous_tsp::CliParseOutcome outcome = aldous_tsp::parse_args(
        static_cast<int>(argv.size()), argv.data(), options, self_test, error);
    if (outcome == aldous_tsp::CliParseOutcome::Run) {
        (void)aldous_tsp::validate_options(options, error);
    }
    return 0;
}
