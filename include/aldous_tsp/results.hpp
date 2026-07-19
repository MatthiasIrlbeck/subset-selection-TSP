#pragma once

#include "aldous_tsp/config.hpp"
#include "aldous_tsp/instance.hpp"
#include "aldous_tsp/restart.hpp"

#include <map>
#include <string>
#include <vector>

namespace aldous_tsp {

struct PValueSummary {
    int k = 0;
    double mean = 0.0;
    double stddev = 0.0;
    // Standard error of the mean. Named stderr_value because `stderr` is a
    // macro in <cstdio> on some platforms (notably MSVC), which makes a member
    // named `stderr` ill-formed; the JSON field keeps the name "stderr".
    double stderr_value = 0.0;
    double min = 0.0;
    double max = 0.0;
    // Maximum over instances of the best-producing restart index for this p;
    // convergence diagnostic (see SolveResult::best_restart).
    int best_restart_max = -1;
    // Aggregate diagnostics over instances for this p: max executed restarts
    // and total subset-solve wall-seconds. Make under-convergence and time
    // distribution visible per curve point.
    int executed_restarts_max = -1;
    double solve_seconds_total = 0.0;
    // Control-variate outputs (populated only when --control-variate is set).
    // subset_bound_mean: mean over instances of the two-NN lower bound on the
    // solved subset, divided by k -- a certified lower bound bracketing f(p)
    // from below. cv_mean/cv_stderr: full-set control-variate-corrected mean
    // and its standard error. cv_variance_reduction: fraction of variance
    // removed (corr^2 of value vs full-set bound; large at p=1).
    bool has_control_variate = false;
    double subset_bound_mean = 0.0;
    // Mean of (value - subset_bound): the gap from the found tour to its two-NN
    // lower bound. Dominated by the intrinsic looseness of the bound (which is
    // ~0.087 below optimal even at p=1, where B_full/N -> 0.625 vs beta ~ 0.712),
    // so it is NOT a pure solver-suboptimality measure; changes in it at fixed p
    // and N do track solver quality.
    double lower_bound_gap_mean = 0.0;
    double cv_mean = 0.0;
    double cv_stderr = 0.0;
    double cv_variance_reduction = 0.0;
    // Held-Karp lower bound outputs (populated only when --held-karp is set).
    // held_karp_bound_mean: mean over instances of the tight 1-tree lower bound
    // divided by k -- a sharp floor bracketing f(p). held_karp_gap_mean: mean of
    // (value - held_karp_bound)/k, a near-optimality certificate for the solver
    // (small => the found tour is provably close to optimal).
    bool has_held_karp = false;
    double held_karp_bound_mean = 0.0;
    double held_karp_gap_mean = 0.0;
    std::vector<double> values;
};

struct InstancePValueRow {
    double p = 0.0;
    std::string key;
    int k = 0;
    double value = 0.0;
    // Index into `restarts`; under a second sweep this is the combined primary
    // then secondary population, not merely the sweep that supplied `value`.
    int best_restart = -1;
    int executed_restarts = 0;
    double solve_seconds = 0.0;
    // Two-NN lower bound on this instance's solved subset, divided by k
    // (control variate; -1 when not computed).
    double subset_bound = -1.0;
    // Typed per-restart diagnostics in restart-index order. Lengths remain
    // raw here; results_to_json derives the backward-compatible L/k arrays.
    std::vector<RestartRecord> restarts;
    // Held-Karp (1-tree) lower bound on this instance's solved subset, divided
    // by k (-1 when not computed).
    double held_karp_bound = -1.0;
};

struct InstanceResultRow {
    bool ok = true;
    int index = -1;
    double wall_seconds = 0.0;
    std::vector<double> values;
    std::vector<InstancePValueRow> p_results;
    SearchStats stats;
    KnnBuildInfo knn_info;
    // Two-NN lower bound on this instance's full point set (control variate;
    // -1 when not computed). Same for every p, so stored once per instance.
    double full_bound = -1.0;
};

struct ResultsDocument {
    int schema_version = kDefaultSchemaVersion;
    int N = 0;
    int instances_done = 0;
    int instances_target = 0;
    int threads = 0;
    double wall_seconds = 0.0;
    RunOptions options;
    SearchStats stats;
    std::vector<double> p_values;
    std::map<std::string, PValueSummary> summary;
    std::vector<InstanceResultRow> instance_rows;
    // Monte-Carlo estimate of E[B_full] (two-NN lower bound over N points on the
    // same domain), the known mean of the control variate. -1 when not computed.
    double full_bound_expectation = -1.0;
    double full_bound_expectation_stderr = -1.0;
    int full_bound_expectation_samples = 0;
};

std::string json_escape(const std::string& input);
std::string p_value_key(double p);
PValueSummary summarize_p_values(int N, double p, const std::vector<double>& values);
std::string results_to_json(const ResultsDocument& doc);
bool write_text_file_atomic(const std::string& path, const std::string& text, std::string* error = nullptr);

} // namespace aldous_tsp
