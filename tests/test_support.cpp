#include "test_support.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace aldous_tsp::test {

std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

Registrar::Registrar(const char* name,
                     TestFunction function,
                     const char* source,
                     const int line) {
    registry().push_back(TestCase{name, function, source, line});
}

void require(const bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message == nullptr ? "test assertion failed" : message);
    }
}

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

long count_restart_kind(const std::vector<RestartRecord>& records,
                        const RestartKind kind) {
    return static_cast<long>(std::count_if(
        records.begin(), records.end(),
        [kind](const RestartRecord& record) { return record.kind == kind; }));
}

} // namespace aldous_tsp::test
