#pragma once

#include "aldous_tsp/restart.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace aldous_tsp::test {

using TestFunction = void (*)();

struct TestCase {
    std::string name;
    TestFunction function = nullptr;
    std::string source;
    int line = 0;
};

std::vector<TestCase>& registry();

class Registrar {
public:
    Registrar(const char* name, TestFunction function, const char* source, int line);
};

void require(bool condition, const char* message);
void require(bool condition, const std::string& message);
long count_restart_kind(const std::vector<RestartRecord>& records, RestartKind kind);

} // namespace aldous_tsp::test

#define ALDOUS_TEST(name)                                                        \
    static void name();                                                          \
    static const ::aldous_tsp::test::Registrar name##_registrar(                 \
        #name, &name, __FILE__, __LINE__);                                       \
    static void name()
