#include "test_support.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <string>

int main(int argc, char** argv) {
    const char* filter = (argc > 1) ? argv[1] : nullptr;
    auto tests = aldous_tsp::test::registry();
    std::stable_sort(tests.begin(), tests.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.source != rhs.source) {
            return lhs.source < rhs.source;
        }
        return lhs.line < rhs.line;
    });

    int executed = 0;
    for (const aldous_tsp::test::TestCase& test : tests) {
        if (filter != nullptr
            && test.name.find(filter) == std::string::npos
            && test.source.find(filter) == std::string::npos) {
            continue;
        }
        ++executed;
        std::fprintf(stderr, "running %s\n", test.name.c_str());
        try {
            test.function();
        } catch (const std::exception& exception) {
            std::fprintf(stderr, "FAILED: %s: %s\n", test.name.c_str(), exception.what());
            return 1;
        } catch (...) {
            std::fprintf(stderr, "FAILED: %s: non-standard exception\n", test.name.c_str());
            return 1;
        }
    }
    if (executed == 0) {
        std::fprintf(stderr, "no tests matched the requested filter\n");
        return 2;
    }
    std::printf("core unit tests passed (%d tests)\n", executed);
    return 0;
}
