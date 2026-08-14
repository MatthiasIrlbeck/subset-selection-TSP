#pragma once

#include "aldous_tsp/config.hpp"
#include "aldous_tsp/tour.hpp"

#include <string>
#include <vector>

namespace aldous_tsp {

bool external_oracle_applicable(const OracleContext& oracle, int k, bool full_tsp) noexcept;
bool build_oracle_context(const ExternalOracleConfig& cfg, OracleContext& oracle, std::string& error);
bool external_oracle_polish_tour(Tour& candidate,
                                 const Instance& inst,
                                 const OracleContext& oracle,
                                 bool full_tsp,
                                 SearchStats* stats = nullptr,
                                 bool enable_internal_two_opt = true);

// Exposed for focused tests and for users who want to validate custom oracle wrappers.
bool parse_external_tour_text(const std::string& text, int k, std::vector<int>& permutation);

} // namespace aldous_tsp
