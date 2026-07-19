#pragma once

#include "aldous_tsp/cli.hpp"

#include "aldous_tsp/config.hpp"
#include "aldous_tsp/instance.hpp"
#include "aldous_tsp/oracle.hpp"
#include "aldous_tsp/results.hpp"
#include "aldous_tsp/solver.hpp"
#include "aldous_tsp/version.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace aldous_tsp {

using Clock = std::chrono::steady_clock;

bool parse_int(const std::string& text, int& out);
bool parse_double(const std::string& text, double& out);
std::vector<std::string> split(const std::string& text, char delim);
bool parse_p_values(const std::string& text, std::vector<double>& out);
bool parse_p_range(const std::string& text, std::vector<double>& out);
bool read_p_file(const std::string& path, std::vector<double>& out, std::string& err);
void canonicalize_p_values(std::vector<double>& values);
void print_help(const char* argv0);
std::string config_summary(const RunOptions& opt);
bool validate_options(RunOptions& opt, std::string& err);
bool parse_args(int argc, char** argv, RunOptions& opt, bool& self_test);

int run_self_test();

} // namespace aldous_tsp
