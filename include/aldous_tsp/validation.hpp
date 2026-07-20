#pragma once

#include "aldous_tsp/config.hpp"

#include <string>
#include <vector>

namespace aldous_tsp {

class Instance;

// Applies runtime defaults (p grid, KNN count, worker counts), validates every
// metadata-defined range, and enforces cross-option invariants. On failure the
// input is left in a valid-to-destroy but otherwise unspecified partially
// normalized state and `error` contains an actionable message.
bool validate_run_options(RunOptions& options, std::string& error);

// Validates a standalone solver configuration for direct library use. This is
// the same field-local and cross-option contract used by validate_run_options,
// without resolving run-level worker counts.
bool validate_solver_options(const SolverOptions& options, std::string& error);

// Validates the public mathematical input independently of CLI parsing.
bool validate_instance(const Instance& instance, std::string& error);
bool validate_subset_request(const Instance& instance,
                             int k,
                             const SolverOptions& options,
                             const std::vector<int>* warm_start,
                             std::string& error);
bool validate_tsp_request(const Instance& instance,
                          const SolverOptions& options,
                          std::string& error);

// Throwing wrappers used at public C++ API boundaries.
void require_valid_run_options(RunOptions& options);
void require_valid_subset_request(const Instance& instance,
                                  int k,
                                  const SolverOptions& options,
                                  const std::vector<int>* warm_start = nullptr);
void require_valid_tsp_request(const Instance& instance,
                               const SolverOptions& options);

} // namespace aldous_tsp
