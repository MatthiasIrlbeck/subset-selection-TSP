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
    // Number of point-set instances for which this cardinality-k result was
    // globally proven optimal by the exact subset oracle.
    int exact_optimal_instances = 0;
    // Control-variate outputs (populated only when --control-variate is set).
    // conditional_two_nn_bound_mean is the mean two-NN lower bound on the tour
    // through the subset selected by the heuristic, divided by k. It is a
    // rigorous bound on TSP(S_found), not on min_{|S|=k} TSP(S): improving the
    // selected subset can lower both the tour and this conditional bound.
    // cv_mean/cv_stderr are the full-set control-variate-corrected mean and its
    // standard error. cv_variance_reduction is the removed variance fraction.
    bool has_control_variate = false;
    union {
        double conditional_two_nn_bound_mean = 0.0;
        // Deprecated source-compatible alias. JSON also emits this legacy name.
        double subset_bound_mean;
    };
    // Mean of value - conditional_two_nn_bound. This diagnoses tour-ordering
    // quality for the chosen subsets; it does not measure subset-selection
    // error and is also affected by the intrinsic looseness of the two-NN bound.
    union {
        double conditional_two_nn_gap_mean = 0.0;
        // Deprecated source-compatible alias. JSON also emits this legacy name.
        double lower_bound_gap_mean;
    };
    double cv_mean = 0.0;
    double cv_stderr = 0.0;
    double cv_variance_reduction = 0.0;
    // Held-Karp outputs (populated only when --held-karp is set). These are also
    // conditional on S_found. A small gap certifies that the tour ordering for
    // S_found is close to optimal; it cannot certify that S_found is the best
    // cardinality-k subset.
    bool has_held_karp = false;
    union {
        double conditional_held_karp_bound_mean = 0.0;
        // Deprecated source-compatible alias. JSON also emits this legacy name.
        double held_karp_bound_mean;
    };
    union {
        double conditional_held_karp_gap_mean = 0.0;
        // Deprecated source-compatible alias. JSON also emits this legacy name.
        double held_karp_gap_mean;
    };
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
    bool exact_optimal = false;
    // Two-NN lower bound on the tour through this instance's selected subset,
    // divided by k (-1 when not computed). This is conditional on S_found.
    union {
        double conditional_two_nn_bound = -1.0;
        // Deprecated source-compatible alias. JSON also emits this legacy name.
        double subset_bound;
    };
    // Typed per-restart diagnostics in restart-index order. Lengths remain
    // raw here; results_to_json derives the backward-compatible L/k arrays.
    std::vector<RestartRecord> restarts;
    // Held-Karp (1-tree) lower bound on the tour through this selected subset,
    // divided by k (-1 when not computed). This is conditional on S_found.
    union {
        double conditional_held_karp_bound = -1.0;
        // Deprecated source-compatible alias. JSON also emits this legacy name.
        double held_karp_bound;
    };
};

struct InstanceResultRow {
    bool ok = true;
    int index = -1;
    // Stable campaign-global point-set identity. `index` remains local to this
    // result shard; replicate_id is offset by RunOptions::replicate_offset.
    std::uint64_t replicate_id = 0;
    // Deterministic 64-bit stream fingerprints. JSON emits fixed-width lower-
    // case hexadecimal strings so JavaScript consumers do not lose precision.
    std::uint64_t point_stream_id = 0;
    std::uint64_t search_stream_id = 0;
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
bool write_text_file_atomic(const std::string& path,
                            const std::string& text,
                            OutputDurability durability,
                            std::string* error = nullptr);
bool write_text_file_atomic(const std::string& path,
                            const std::string& text,
                            std::string* error = nullptr);

} // namespace aldous_tsp
