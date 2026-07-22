#include "aldous_tsp/results.hpp"

#include <cstring>
#if defined(_MSC_VER)
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

#include "aldous_tsp/version.hpp"

#include "json_writer.hpp"
#include "generated_options.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <thread>

#ifndef ALDOUS_TSP_BUILD_TYPE
#define ALDOUS_TSP_BUILD_TYPE "unknown"
#endif
#ifndef ALDOUS_TSP_BUILD_ENABLE_NATIVE
#define ALDOUS_TSP_BUILD_ENABLE_NATIVE "unknown"
#endif
#ifndef ALDOUS_TSP_BUILD_ENABLE_SANITIZERS
#define ALDOUS_TSP_BUILD_ENABLE_SANITIZERS "unknown"
#endif
#ifndef ALDOUS_TSP_BUILD_ENABLE_WARNINGS
#define ALDOUS_TSP_BUILD_ENABLE_WARNINGS "unknown"
#endif
#ifndef ALDOUS_TSP_BUILD_ENABLE_WERROR
#define ALDOUS_TSP_BUILD_ENABLE_WERROR "unknown"
#endif
#ifndef ALDOUS_TSP_BUILD_CXX_FLAGS
#define ALDOUS_TSP_BUILD_CXX_FLAGS ""
#endif
#ifndef ALDOUS_TSP_BUILD_LOW_MEMORY
#define ALDOUS_TSP_BUILD_LOW_MEMORY "unknown"
#endif
#ifndef ALDOUS_TSP_BUILD_SOURCE_COMPILE_OPTIONS
#define ALDOUS_TSP_BUILD_SOURCE_COMPILE_OPTIONS ""
#endif
#ifndef ALDOUS_TSP_BUILD_TARGET_COMPILE_OPTIONS
#define ALDOUS_TSP_BUILD_TARGET_COMPILE_OPTIONS ""
#endif
#ifndef ALDOUS_TSP_BUILD_EFFECTIVE_COMPILE_OPTIONS
#define ALDOUS_TSP_BUILD_EFFECTIVE_COMPILE_OPTIONS ""
#endif

namespace aldous_tsp {

std::string json_escape(const std::string& input) {
    return json_escape_text(input);
}

std::string p_value_key(double p) {
    if (!std::isfinite(p)) {
        throw std::invalid_argument("p-value keys must be finite");
    }
    return json_number_text(p);
}

PValueSummary summarize_p_values(int N, double p, const std::vector<double>& values) {
    PValueSummary s;
    s.k = std::max(3, std::min(N, static_cast<int>(std::llround(p * static_cast<double>(N)))));
    s.values = values;
    if (values.empty()) {
        return s;
    }
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    double var = 0.0;
    if (values.size() > 1U) {
        for (double value : values) {
            const double d = value - s.mean;
            var += d * d;
        }
        var /= static_cast<double>(values.size() - 1U);
    }
    s.stddev = std::sqrt(var);
    s.stderr_value = values.size() > 1U ? s.stddev / std::sqrt(static_cast<double>(values.size())) : 0.0;
    const auto minmax = std::minmax_element(values.begin(), values.end());
    s.min = *minmax.first;
    s.max = *minmax.second;
    return s;
}

namespace {

std::string platform_string() {
#if defined(__linux__)
    return "linux";
#elif defined(__APPLE__)
    return "macos";
#elif defined(_WIN32)
    return "windows";
#elif defined(__FreeBSD__)
    return "freebsd";
#else
    return "unknown";
#endif
}

std::string compiler_string() {
    std::ostringstream oss;
#if defined(__clang__)
    oss << "Clang " << __clang_version__;
#elif defined(__GNUC__)
    oss << "GCC " << __VERSION__;
#elif defined(_MSC_FULL_VER)
    oss << "MSVC " << _MSC_FULL_VER;
#else
    oss << "unknown";
#endif
    oss << " (C++" << __cplusplus << ')';
    return oss.str();
}

std::string trim_left(std::string value) {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
        value.erase(value.begin());
    }
    return value;
}

std::string cpuid_brand_string() {
    // x86 processor brand string from CPUID leaves 0x80000002-0x80000004.
    // Portable across MSVC and GCC/Clang; empty on non-x86 or unsupported.
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4];
    __cpuid(regs, static_cast<int>(0x80000000));
    if (static_cast<unsigned>(regs[0]) < 0x80000004u) { return ""; }
    char brand[49] = {0};
    for (unsigned leaf = 0; leaf < 3; ++leaf) {
        __cpuid(regs, static_cast<int>(0x80000002u + leaf));
        std::memcpy(brand + leaf * 16U, regs, 16U);
    }
    std::string s(brand);
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(0x80000000u, &eax, &ebx, &ecx, &edx) == 0 || eax < 0x80000004u) { return ""; }
    char brand[49] = {0};
    for (unsigned leaf = 0; leaf < 3; ++leaf) {
        __get_cpuid(0x80000002u + leaf, &eax, &ebx, &ecx, &edx);
        std::memcpy(brand + leaf * 16U + 0U, &eax, 4U);
        std::memcpy(brand + leaf * 16U + 4U, &ebx, 4U);
        std::memcpy(brand + leaf * 16U + 8U, &ecx, 4U);
        std::memcpy(brand + leaf * 16U + 12U, &edx, 4U);
    }
    std::string s(brand);
#else
    std::string s;
#endif
    // Collapse surrounding whitespace (brand strings are commonly padded).
    const std::size_t first = s.find_first_not_of(" \t");
    const std::size_t last = s.find_last_not_of(" \t\0");
    if (first == std::string::npos) { return ""; }
    return s.substr(first, last - first + 1U);
}

std::string cpu_model_string() {
#if defined(__linux__)
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        for (const char* key : {"model name", "Hardware"}) {
            const std::size_t key_len = std::char_traits<char>::length(key);
            if (line.compare(0, key_len, key) == 0) {
                const std::size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    std::string value = trim_left(line.substr(colon + 1U));
                    if (!value.empty()) {
                        return value;
                    }
                }
            }
        }
    }
#endif
    const std::string brand = cpuid_brand_string();
    if (!brand.empty()) {
        return brand;
    }
    return "unknown";
}

bool cmake_option_enabled(const char* value) {
    const std::string text(value == nullptr ? "" : value);
    return text == "ON" || text == "TRUE" || text == "1" || text == "YES";
}

void write_json_double(std::ostream& out, double value) {
    JsonWriter(out).number(value);
}

void write_double_array(std::ostream& out, const std::vector<double>& values) {
    JsonWriter(out).double_array(values);
}

template <typename ValueFn>
void write_restart_double_array(std::ostream& out,
                                const std::vector<RestartRecord>& records,
                                ValueFn value_of) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i != 0U) { out << ", "; }
        write_json_double(out, value_of(records[i]));
    }
    out << ']';
}

void write_restart_kind_array(std::ostream& out,
                              const std::vector<RestartRecord>& records) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i != 0U) { out << ", "; }
        out << restart_kind_code(records[i].kind);
    }
    out << ']';
}

void write_restart_sweep_array(std::ostream& out,
                               const std::vector<RestartRecord>& records) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i != 0U) { out << ", "; }
        out << restart_sweep_code(records[i].sweep);
    }
    out << ']';
}

void write_restart_role_array(std::ostream& out,
                              const std::vector<RestartRecord>& records) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i != 0U) { out << ", "; }
        out << restart_role_code(records[i].role);
    }
    out << ']';
}

void write_restart_variant_array(std::ostream& out,
                                 const std::vector<RestartRecord>& records) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i != 0U) { out << ", "; }
        out << records[i].seed_variant;
    }
    out << ']';
}

void write_restart_promotion_stage_array(
    std::ostream& out,
    const std::vector<RestartRecord>& records) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i != 0U) { out << ", "; }
        out << restart_promotion_stage_code(records[i].promotion_stage);
    }
    out << ']';
}

void write_restart_sa_iteration_array(
    std::ostream& out,
    const std::vector<RestartRecord>& records) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i != 0U) { out << ", "; }
        out << records[i].sa_iterations;
    }
    out << ']';
}

void write_restart_sa_temperature_array(
    std::ostream& out,
    const std::vector<RestartRecord>& records,
    const bool start) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i != 0U) { out << ", "; }
        write_json_double(out, start ? records[i].sa_t0 : records[i].sa_t1);
    }
    out << ']';
}

void write_restart_sa_temperature_sample_array(
    std::ostream& out,
    const std::vector<RestartRecord>& records) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i != 0U) { out << ", "; }
        out << records[i].sa_temperature_samples;
    }
    out << ']';
}

void write_restart_sa_temperature_calibrated_array(
    std::ostream& out,
    const std::vector<RestartRecord>& records) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i != 0U) { out << ", "; }
        out << (records[i].sa_temperature_calibrated ? "true" : "false");
    }
    out << ']';
}

void write_restart_strong_polished_array(
    std::ostream& out,
    const std::vector<RestartRecord>& records) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (i != 0U) { out << ", "; }
        out << (records[i].strong_polished ? "true" : "false");
    }
    out << ']';
}

std::string hex_u64(const std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::nouppercase << std::setw(16)
        << std::setfill('0') << value;
    return out.str();
}

std::string summary_key(double p) {
    return p_value_key(p);
}

void write_phase_timing(std::ostream& out, const SearchPhaseTiming& phases, const std::string& indent) {
    out << indent << "{\n";
    auto write_seconds = [&](const char* name, double value, bool comma = true) {
        out << indent << "  \"" << name << "\": ";
        write_json_double(out, value);
        out << (comma ? ",\n" : "\n");
    };
    write_seconds("seed_construction_seconds", phases.seed_construction_seconds);
    write_seconds("tsp_construction_seconds", phases.tsp_construction_seconds);
    write_seconds("initial_polish_seconds", phases.initial_polish_seconds);
    write_seconds("sa_seconds", phases.sa_seconds);
    write_seconds("sa_checkpoint_polish_seconds", phases.sa_checkpoint_polish_seconds);
    write_seconds("post_sa_polish_seconds", phases.post_sa_polish_seconds);
    write_seconds("subset_swap_seconds", phases.subset_swap_seconds);
    write_seconds("highp_exchange_seconds", phases.highp_exchange_seconds);
    write_seconds("pair_exchange_seconds", phases.pair_exchange_seconds);
    write_seconds("ruin_recreate_seconds", phases.ruin_recreate_seconds);
    write_seconds("ejection_chain_seconds", phases.ejection_chain_seconds);
    write_seconds("exact_subset_seconds", phases.exact_subset_seconds);
    write_seconds("path_relink_seconds", phases.path_relink_seconds);
    write_seconds("tsp_ils_seconds", phases.tsp_ils_seconds);
    write_seconds("final_polish_seconds", phases.final_polish_seconds);
    write_seconds("oracle_seconds", phases.oracle_seconds);
    out << indent << "  \"sa_proposal_samples\": " << phases.sa_proposal_samples << ",\n"
        << indent << "  \"sa_insertion_samples\": " << phases.sa_insertion_samples << ",\n";
    write_seconds("sa_proposal_sample_seconds", phases.sa_proposal_sample_seconds);
    write_seconds("sa_insertion_sample_seconds", phases.sa_insertion_sample_seconds, false);
    out << indent << '}';
}

template <std::size_t Size>
void write_u64_array(std::ostream& out,
                     const std::array<std::uint64_t, Size>& values) {
    out << '[';
    for (std::size_t i = 0U; i < Size; ++i) {
        if (i != 0U) { out << ", "; }
        out << values[i];
    }
    out << ']';
}

void write_stats(std::ostream& out, const SearchStats& stats, const std::string& indent) {
    out << indent << "{\n"
        << indent << "  \"exact_subset_calls\": " << stats.exact_subset_calls << ",\n"
        << indent << "  \"exact_subset_solved\": " << stats.exact_subset_solved << ",\n"
        << indent << "  \"exact_subset_states\": " << stats.exact_subset_states << ",\n"
        << indent << "  \"exact_subset_transitions\": " << stats.exact_subset_transitions << ",\n"
        << indent << "  \"exact_subset_peak_memory_bytes\": "
        << stats.exact_subset_peak_memory_bytes << ",\n"
        << indent << "  \"tsp_candidate_starts\": " << stats.tsp_candidate_starts << ",\n"
        << indent << "  \"tsp_promoted_restarts\": " << stats.tsp_promoted_restarts << ",\n"
        << indent << "  \"tsp_restarts\": " << stats.tsp_restarts << ",\n"
        << indent << "  \"tsp_ils_iterations\": " << stats.tsp_ils_iterations << ",\n"
        << indent << "  \"subset_restarts\": " << stats.subset_restarts << ",\n"
        << indent << "  \"smallp_seed_restarts\": " << stats.smallp_seed_restarts << ",\n"
        << indent << "  \"highp_delete_restarts\": " << stats.highp_delete_restarts << ",\n"
        << indent << "  \"warm_restarts\": " << stats.warm_restarts << ",\n"
        << indent << "  \"random_restarts\": " << stats.random_restarts << ",\n"
        << indent << "  \"region_restarts\": " << stats.region_restarts << ",\n"
        << indent << "  \"dense_restarts\": " << stats.dense_restarts << ",\n"
        << indent << "  \"racing_pilot_restarts\": " << stats.racing_pilot_restarts << ",\n"
        << indent << "  \"racing_promoted_restarts\": " << stats.racing_promoted_restarts << ",\n"
        << indent << "  \"strong_polish_candidates\": " << stats.strong_polish_candidates << ",\n"
        << indent << "  \"strong_polish_finalists\": " << stats.strong_polish_finalists << ",\n"
        << indent << "  \"strong_polish_improvements\": " << stats.strong_polish_improvements << ",\n"
        << indent << "  \"elite_restarts\": " << stats.elite_restarts << ",\n"
        << indent << "  \"kick_restarts\": " << stats.kick_restarts << ",\n"
        << indent << "  \"elite_diversity_candidates\": " << stats.elite_diversity_candidates << ",\n"
        << indent << "  \"elite_diversity_retained\": " << stats.elite_diversity_retained << ",\n"
        << indent << "  \"elite_diversity_rejected\": " << stats.elite_diversity_rejected << ",\n"
        << indent << "  \"two_opt_scans\": " << stats.two_opt_scans << ",\n"
        << indent << "  \"two_opt_improvements\": " << stats.two_opt_improvements << ",\n"
        << indent << "  \"or_opt_scans\": " << stats.or_opt_scans << ",\n"
        << indent << "  \"or_opt_improvements\": " << stats.or_opt_improvements << ",\n"
        << indent << "  \"sa_moves\": " << stats.sa_moves << ",\n"
        << indent << "  \"sa_accepted\": " << stats.sa_accepted << ",\n"
        << indent << "  \"sa_improving\": " << stats.sa_improving << ",\n"
        << indent << "  \"sa_candidate_evaluations\": " << stats.sa_candidate_evaluations << ",\n"
        << indent << "  \"sa_multiple_try_iterations\": " << stats.sa_multiple_try_iterations << ",\n"
        << indent << "  \"sa_temperature_schedules\": " << stats.sa_temperature_schedules << ",\n"
        << indent << "  \"sa_temperature_calibrations\": " << stats.sa_temperature_calibrations << ",\n"
        << indent << "  \"sa_temperature_fallbacks\": " << stats.sa_temperature_fallbacks << ",\n"
        << indent << "  \"sa_calibration_attempts\": " << stats.sa_calibration_attempts << ",\n"
        << indent << "  \"sa_calibration_uphill_samples\": " << stats.sa_calibration_uphill_samples << ",\n";
    out << indent << "  \"sa_temperature_t0_sum\": ";
    write_json_double(out, stats.sa_temperature_t0_sum);
    out << ",\n" << indent << "  \"sa_temperature_t1_sum\": ";
    write_json_double(out, stats.sa_temperature_t1_sum);
    out << ",\n" << indent << "  \"sa_temperature_t0_min\": ";
    write_json_double(out, stats.sa_temperature_schedules > 0U
        ? stats.sa_temperature_t0_min : 0.0);
    out << ",\n" << indent << "  \"sa_temperature_t0_max\": ";
    write_json_double(out, stats.sa_temperature_t0_max);
    out << ",\n" << indent << "  \"sa_temperature_t1_min\": ";
    write_json_double(out, stats.sa_temperature_schedules > 0U
        ? stats.sa_temperature_t1_min : 0.0);
    out << ",\n" << indent << "  \"sa_temperature_t1_max\": ";
    write_json_double(out, stats.sa_temperature_t1_max);
    out << ",\n" << indent << "  \"sa_decile_moves\": ";
    write_u64_array(out, stats.sa_decile_moves);
    out << ",\n" << indent << "  \"sa_decile_uphill_moves\": ";
    write_u64_array(out, stats.sa_decile_uphill_moves);
    out << ",\n" << indent << "  \"sa_decile_accepted\": ";
    write_u64_array(out, stats.sa_decile_accepted);
    out << ",\n" << indent << "  \"sa_decile_uphill_accepted\": ";
    write_u64_array(out, stats.sa_decile_uphill_accepted);
    out << ",\n"
        << indent << "  \"subset_swap_scans\": " << stats.subset_swap_scans << ",\n"
        << indent << "  \"subset_swap_improvements\": " << stats.subset_swap_improvements << ",\n"
        << indent << "  \"highp_exchange_scans\": " << stats.highp_exchange_scans << ",\n"
        << indent << "  \"highp_exchange_improvements\": " << stats.highp_exchange_improvements << ",\n"
        << indent << "  \"pair_exchange_scans\": " << stats.pair_exchange_scans << ",\n"
        << indent << "  \"pair_exchange_improvements\": " << stats.pair_exchange_improvements << ",\n"
        << indent << "  \"pair_exchange_skipped_large_k\": " << stats.pair_exchange_skipped_large_k << ",\n"
        << indent << "  \"ruin_recreate_attempts\": " << stats.ruin_recreate_attempts << ",\n"
        << indent << "  \"ruin_recreate_improvements\": " << stats.ruin_recreate_improvements << ",\n"
        << indent << "  \"ruin_recreate_removed_nodes\": " << stats.ruin_recreate_removed_nodes << ",\n"
        << indent << "  \"ruin_recreate_worst_attempts\": " << stats.ruin_recreate_worst_attempts << ",\n"
        << indent << "  \"ruin_recreate_segment_attempts\": " << stats.ruin_recreate_segment_attempts << ",\n"
        << indent << "  \"ruin_recreate_segment_improvements\": " << stats.ruin_recreate_segment_improvements << ",\n"
        << indent << "  \"ruin_recreate_spatial_attempts\": " << stats.ruin_recreate_spatial_attempts << ",\n"
        << indent << "  \"ruin_recreate_spatial_improvements\": " << stats.ruin_recreate_spatial_improvements << ",\n"
        << indent << "  \"ruin_recreate_long_edge_attempts\": " << stats.ruin_recreate_long_edge_attempts << ",\n"
        << indent << "  \"ruin_recreate_long_edge_improvements\": " << stats.ruin_recreate_long_edge_improvements << ",\n"
        << indent << "  \"ruin_recreate_random_attempts\": " << stats.ruin_recreate_random_attempts << ",\n"
        << indent << "  \"ruin_recreate_random_improvements\": " << stats.ruin_recreate_random_improvements << ",\n"
        << indent << "  \"ejection_chain_attempts\": " << stats.ejection_chain_attempts << ",\n"
        << indent << "  \"ejection_chain_feasible\": " << stats.ejection_chain_feasible << ",\n"
        << indent << "  \"ejection_chain_steps\": " << stats.ejection_chain_steps << ",\n"
        << indent << "  \"ejection_chain_scans\": " << stats.ejection_chain_scans << ",\n"
        << indent << "  \"ejection_chain_improvements\": " << stats.ejection_chain_improvements << ",\n"
        << indent << "  \"ejection_chain_accepted_depth\": " << stats.ejection_chain_accepted_depth << ",\n"
        << indent << "  \"path_relink_pairs_considered\": " << stats.path_relink_pairs_considered << ",\n"
        << indent << "  \"path_relink_pairs_skipped_distance\": " << stats.path_relink_pairs_skipped_distance << ",\n"
        << indent << "  \"path_relink_pairs_skipped_budget\": " << stats.path_relink_pairs_skipped_budget << ",\n"
        << indent << "  \"path_relink_attempts\": " << stats.path_relink_attempts << ",\n"
        << indent << "  \"path_relink_feasible\": " << stats.path_relink_feasible << ",\n"
        << indent << "  \"path_relink_removed_sum\": " << stats.path_relink_removed_sum << ",\n"
        << indent << "  \"path_relink_candidate_scans\": " << stats.path_relink_candidate_scans << ",\n"
        << indent << "  \"path_relink_elite_insertions\": " << stats.path_relink_elite_insertions << ",\n"
        << indent << "  \"path_relink_best_improvements\": " << stats.path_relink_best_improvements << ",\n"
        << indent << "  \"path_relink_improvements\": " << stats.path_relink_improvements << ",\n";
    out << indent << "  \"knn_build_seconds\": ";
    write_json_double(out, stats.knn_build_seconds);
    out << ",\n"
        << indent << "  \"knn_requested_grid_instances\": " << stats.knn_requested_grid_instances << ",\n"
        << indent << "  \"knn_requested_bruteforce_instances\": " << stats.knn_requested_bruteforce_instances << ",\n"
        << indent << "  \"knn_effective_grid_instances\": " << stats.knn_effective_grid_instances << ",\n"
        << indent << "  \"knn_effective_bruteforce_instances\": " << stats.knn_effective_bruteforce_instances << ",\n"
        << indent << "  \"knn_bruteforce_fallback_instances\": " << stats.knn_bruteforce_fallback_instances << ",\n"
        << indent << "  \"knn_grid_cell_capped_instances\": " << stats.knn_grid_cell_capped_instances << ",\n"
        << indent << "  \"knn_grid_cell_samples\": " << stats.knn_grid_cell_samples << ",\n"
        << indent << "  \"knn_grid_cells_max\": " << stats.knn_grid_cells_max << ",\n"
        << indent << "  \"knn_grid_cells_sum\": " << stats.knn_grid_cells_sum << ",\n"
        << indent << "  \"grid_cell_effective_min\": ";
    write_json_double(out, stats.grid_cell_effective_min);
    out << ",\n" << indent << "  \"grid_cell_effective_max\": ";
    write_json_double(out, stats.grid_cell_effective_max);
    out << ",\n" << indent << "  \"grid_cell_effective_sum\": ";
    write_json_double(out, stats.grid_cell_effective_sum);
    out << ",\n" << indent << "  \"tsp_seconds\": ";
    write_json_double(out, stats.tsp_seconds);
    out << ",\n" << indent << "  \"subset_seconds\": ";
    write_json_double(out, stats.subset_seconds);
    out << ",\n"
        << indent << "  \"oracle_calls\": " << stats.oracle_calls << ",\n"
        << indent << "  \"oracle_solved\": " << stats.oracle_solved << ",\n"
        << indent << "  \"oracle_improved\": " << stats.oracle_improved << ",\n"
        << indent << "  \"oracle_failed\": " << stats.oracle_failed << ",\n"
        << indent << "  \"oracle_tsp_calls\": " << stats.oracle_tsp_calls << ",\n"
        << indent << "  \"oracle_subset_calls\": " << stats.oracle_subset_calls << ",\n"
        << indent << "  \"oracle_gain\": ";
    write_json_double(out, stats.oracle_gain);
    out << ",\n" << indent << "  \"phase_timing\": ";
    write_phase_timing(out, stats.phases, indent + "  ");
    out << "\n" << indent << '}';
}

void write_oracle_call_records(std::ostream& out, const std::vector<OracleCallRecord>& records, const std::string& indent) {
    out << '[';
    for (std::size_t i = 0; i < records.size(); ++i) {
        const OracleCallRecord& r = records[i];
        if (i != 0U) {
            out << ',';
        }
        out << "\n" << indent << "  {\n"
            << indent << "    \"type\": \"" << json_escape(r.type) << "\",\n"
            << indent << "    \"k\": " << r.k << ",\n"
            << indent << "    \"solver\": \"" << json_escape(r.solver) << "\",\n"
            << indent << "    \"format\": \"" << json_escape(r.format) << "\",\n"
            << indent << "    \"status\": \"" << json_escape(r.status) << "\",\n"
            << indent << "    \"exec_path\": \"" << json_escape(r.exec_path) << "\",\n"
            << indent << "    \"exec_sha256\": \"" << json_escape(r.exec_sha256) << "\",\n"
            << indent << "    \"solver_version\": \"" << json_escape(r.solver_version) << "\",\n"
            << indent << "    \"error\": \"" << json_escape(r.error) << "\",\n"
            << indent << "    \"before_length\": ";
        write_json_double(out, r.before_length);
        out << ",\n" << indent << "    \"after_length\": ";
        write_json_double(out, r.after_length);
        out << ",\n" << indent << "    \"gain\": ";
        write_json_double(out, r.gain);
        out << ",\n" << indent << "    \"seconds\": ";
        write_json_double(out, r.seconds);
        out << "\n" << indent << "  }";
    }
    if (!records.empty()) {
        out << '\n' << indent;
    }
    out << ']';
}

void write_memory_plan(std::ostream& out, const MemoryPlan& plan, const std::string& indent) {
    out << indent << "{\n"
        << indent << "  \"budget_bytes\": " << plan.budget_bytes << ",\n"
        << indent << "  \"fixed_overhead_bytes\": " << plan.fixed_overhead_bytes << ",\n"
        << indent << "  \"estimated_instance_bytes\": " << plan.estimated_instance_bytes << ",\n"
        << indent << "  \"estimated_peak_bytes\": " << plan.estimated_peak_bytes << ",\n"
        << indent << "  \"requested_threads\": " << plan.requested_threads << ",\n"
        << indent << "  \"resolved_threads\": " << plan.resolved_threads << ",\n"
        << indent << "  \"effective_threads\": " << plan.effective_threads << ",\n"
        << indent << "  \"limited_by_budget\": "
        << (plan.limited_by_budget ? "true" : "false") << ",\n"
        << indent << "  \"reverse_knn_enabled\": "
        << (plan.reverse_knn_enabled ? "true" : "false") << "\n"
        << indent << '}';
}

void write_knn_info(std::ostream& out, const KnnBuildInfo& info, const std::string& indent) {
    out << indent << "{\n"
        << indent << "  \"requested_backend\": \"" << knn_backend_name(info.requested_backend) << "\",\n"
        << indent << "  \"effective_backend\": \"" << knn_backend_name(info.effective_backend) << "\",\n"
        << indent << "  \"bruteforce_fallback\": " << (info.brute_force_fallback ? "true" : "false") << ",\n"
        << indent << "  \"grid_cell_capped\": " << (info.grid_cell_capped ? "true" : "false") << ",\n"
        << indent << "  \"requested_cell_size\": ";
    write_json_double(out, info.requested_cell_size);
    out << ",\n" << indent << "  \"effective_cell_size\": ";
    write_json_double(out, info.effective_cell_size);
    out << ",\n"
        << indent << "  \"gx\": " << info.gx << ",\n"
        << indent << "  \"gy\": " << info.gy << ",\n"
        << indent << "  \"grid_cells\": " << info.grid_cells << ",\n"
        << indent << "  \"coordinate_span\": ";
    write_json_double(out, info.coordinate_span);
    out << "\n" << indent << '}';
}

void write_instance_rows(std::ostream& out, const std::vector<InstanceResultRow>& rows) {
    out << '[';
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const InstanceResultRow& row = rows[i];
        if (i != 0U) {
            out << ',';
        }
        out << "\n    {\n"
            << "      \"index\": " << row.index << ",\n"
            << "      \"replicate_id\": " << row.replicate_id << ",\n"
            << "      \"point_stream_id\": \"" << hex_u64(row.point_stream_id) << "\",\n"
            << "      \"search_stream_id\": \"" << hex_u64(row.search_stream_id) << "\",\n"
            << "      \"ok\": " << (row.ok ? "true" : "false") << ",\n"
            << "      \"wall_seconds\": ";
        write_json_double(out, row.wall_seconds);
        out << ",\n      \"values\": ";
        write_double_array(out, row.values);
        if (row.full_bound >= 0.0) {
            out << ",\n      \"full_bound\": ";
            write_json_double(out, row.full_bound);
        }
        out << ",\n      \"p_results\": [";
        for (std::size_t j = 0; j < row.p_results.size(); ++j) {
            const InstancePValueRow& pv = row.p_results[j];
            if (j != 0U) {
                out << ',';
            }
            out << "\n        {\n"
                << "          \"p\": ";
            write_json_double(out, pv.p);
            out << ",\n          \"key\": \"" << json_escape(pv.key) << "\",\n"
                << "          \"k\": " << pv.k << ",\n"
                << "          \"value\": ";
            write_json_double(out, pv.value);
            out << ",\n          \"best_restart\": " << pv.best_restart;
            out << ",\n          \"executed_restarts\": " << pv.executed_restarts;
            if (!pv.restarts.empty()) {
                const double inv_k = 1.0 / static_cast<double>(pv.k);
                out << ",\n          \"restart_values\": ";
                write_restart_double_array(out, pv.restarts,
                    [inv_k](const RestartRecord& record) { return record.length * inv_k; });
                out << ",\n          \"restart_kinds\": ";
                write_restart_kind_array(out, pv.restarts);
                out << ",\n          \"restart_sweeps\": ";
                write_restart_sweep_array(out, pv.restarts);
                out << ",\n          \"restart_roles\": ";
                write_restart_role_array(out, pv.restarts);
                out << ",\n          \"restart_variants\": ";
                write_restart_variant_array(out, pv.restarts);
                out << ",\n          \"restart_promotion_stages\": ";
                write_restart_promotion_stage_array(out, pv.restarts);
                out << ",\n          \"restart_sa_iterations\": ";
                write_restart_sa_iteration_array(out, pv.restarts);
                out << ",\n          \"restart_sa_t0\": ";
                write_restart_sa_temperature_array(out, pv.restarts, true);
                out << ",\n          \"restart_sa_t1\": ";
                write_restart_sa_temperature_array(out, pv.restarts, false);
                out << ",\n          \"restart_sa_temperature_samples\": ";
                write_restart_sa_temperature_sample_array(out, pv.restarts);
                out << ",\n          \"restart_sa_temperature_calibrated\": ";
                write_restart_sa_temperature_calibrated_array(out, pv.restarts);
                out << ",\n          \"restart_strong_polished\": ";
                write_restart_strong_polished_array(out, pv.restarts);
                out << ",\n          \"restart_centroids_x\": ";
                write_restart_double_array(out, pv.restarts,
                    [](const RestartRecord& record) { return record.centroid_x; });
                out << ",\n          \"restart_centroids_y\": ";
                write_restart_double_array(out, pv.restarts,
                    [](const RestartRecord& record) { return record.centroid_y; });
                out << ",\n          \"restart_radii\": ";
                write_restart_double_array(out, pv.restarts,
                    [](const RestartRecord& record) { return record.radius; });
            }
            out << ",\n          \"solve_seconds\": ";
            write_json_double(out, pv.solve_seconds);
            out << ",\n          \"exact_optimal\": "
                << (pv.exact_optimal ? "true" : "false");
            if (pv.conditional_two_nn_bound >= 0.0) {
                out << ",\n          \"conditional_two_nn_bound\": ";
                write_json_double(out, pv.conditional_two_nn_bound);
                out << ",\n          \"subset_bound\": ";
                write_json_double(out, pv.conditional_two_nn_bound);
            }
            if (pv.conditional_held_karp_bound >= 0.0) {
                out << ",\n          \"conditional_held_karp_bound\": ";
                write_json_double(out, pv.conditional_held_karp_bound);
                out << ",\n          \"held_karp_bound\": ";
                write_json_double(out, pv.conditional_held_karp_bound);
            }
            out << "\n        }";
        }
        if (!row.p_results.empty()) {
            out << '\n';
        }
        out << "      ],\n      \"knn_build\": ";
        write_knn_info(out, row.knn_info, "      ");
        out << ",\n      \"search_stats\": ";
        write_stats(out, row.stats, "      ");
        out << ",\n      \"oracle_call_records\": ";
        write_oracle_call_records(out, row.stats.oracle_call_records, "      ");
        out << "\n    }";
    }
    if (!rows.empty()) {
        out << '\n';
    }
    out << "  ]";
}

} // namespace

std::string results_to_json(const ResultsDocument& doc) {
    std::ostringstream out;
    out << "{\n"
        << "  \"schema_version\": " << doc.schema_version << ",\n"
        << "  \"run_metadata\": {\n"
        << "    \"project_version\": \"" << json_escape(kProjectVersion) << "\",\n"
        << "    \"git_commit\": \"" << json_escape(kGitCommit) << "\",\n"
        << "    \"compiler\": \"" << json_escape(compiler_string()) << "\",\n"
        << "    \"platform\": \"" << json_escape(platform_string()) << "\",\n"
        << "    \"cpu_model\": \"" << json_escape(cpu_model_string()) << "\",\n"
        << "    \"hardware_threads\": " << std::max(1U, std::thread::hardware_concurrency()) << "\n"
        << "  },\n"
        << "  \"build_metadata\": {\n"
        << "    \"build_type\": \"" << json_escape(ALDOUS_TSP_BUILD_TYPE) << "\",\n"
        << "    \"configured_build_type\": \"" << json_escape(kConfiguredBuildType) << "\",\n"
        << "    \"cmake_generator\": \"" << json_escape(kCMakeGenerator) << "\",\n"
        << "    \"cmake_version\": \"" << json_escape(kCMakeVersion) << "\",\n"
        << "    \"enable_native\": " << (cmake_option_enabled(ALDOUS_TSP_BUILD_ENABLE_NATIVE) ? "true" : "false") << ",\n"
        << "    \"enable_sanitizers\": " << (cmake_option_enabled(ALDOUS_TSP_BUILD_ENABLE_SANITIZERS) ? "true" : "false") << ",\n"
        << "    \"enable_warnings\": " << (cmake_option_enabled(ALDOUS_TSP_BUILD_ENABLE_WARNINGS) ? "true" : "false") << ",\n"
        << "    \"enable_werror\": " << (cmake_option_enabled(ALDOUS_TSP_BUILD_ENABLE_WERROR) ? "true" : "false") << ",\n"
        << "    \"enable_python_tests\": " << (cmake_option_enabled(kBuildEnablePythonTests) ? "true" : "false") << ",\n"
        << "    \"low_memory_build\": " << (cmake_option_enabled(kBuildLowMemory) ? "true" : "false") << ",\n"
        << "    \"optimization_profile\": \"" << json_escape(kOptimizationProfile) << "\",\n"
        << "    \"effective_optimization_level\": \"" << json_escape(kEffectiveOptimizationLevel) << "\",\n"
        << "    \"cxx_flags\": \"" << json_escape(ALDOUS_TSP_BUILD_CXX_FLAGS) << "\",\n"
        << "    \"cxx_flags_configured\": \"" << json_escape(kCxxFlagsConfigured) << "\",\n"
        << "    \"cxx_flags_effective_configured\": \"" << json_escape(kCxxFlagsEffectiveConfigured) << "\",\n"
        << "    \"core_target_compile_options\": \"" << json_escape(kCoreTargetCompileOptions) << "\",\n"
        << "    \"cli_target_compile_options\": \"" << json_escape(kCliTargetCompileOptions) << "\",\n"
        << "    \"source_compile_options\": \"" << json_escape(ALDOUS_TSP_BUILD_SOURCE_COMPILE_OPTIONS) << "\",\n"
        << "    \"target_compile_options\": \"" << json_escape(ALDOUS_TSP_BUILD_TARGET_COMPILE_OPTIONS) << "\",\n"
        << "    \"effective_compile_options\": \"" << json_escape(ALDOUS_TSP_BUILD_EFFECTIVE_COMPILE_OPTIONS) << "\",\n"
        << "    \"cplusplus\": " << static_cast<long long>(__cplusplus) << "\n"
        << "  },\n"
        << "  \"campaign_metadata\": {\n"
        << "    \"campaign_id\": \"" << json_escape(doc.options.campaign_id) << "\",\n"
        << "    \"campaign_shard\": " << doc.options.campaign_shard << ",\n"
        << "    \"replicate_offset\": " << doc.options.replicate_offset << ",\n"
        << "    \"point_seed\": " << effective_point_seed(doc.options) << ",\n"
        << "    \"search_seed\": " << effective_search_seed(doc.options) << ",\n"
        << "    \"solver_policy_id\": \"" << json_escape(doc.options.solver_policy_id) << "\",\n"
        << "    \"fidelity_level\": \"" << json_escape(doc.options.fidelity_level) << "\"\n"
        << "  },\n"
        << "  \"N\": " << doc.N << ",\n"
        << "  \"done\": " << doc.instances_done << ",\n"
        << "  \"target\": " << doc.instances_target << ",\n";
    if (doc.full_bound_expectation >= 0.0) {
        out << "  \"full_bound_expectation\": ";
        write_json_double(out, doc.full_bound_expectation);
        out << ",\n  \"full_bound_expectation_stderr\": ";
        write_json_double(out, doc.full_bound_expectation_stderr);
        out << ",\n  \"full_bound_expectation_samples\": " << doc.full_bound_expectation_samples << ",\n";
    }
    out << "  \"threads\": " << doc.threads << ",\n"
        << "  \"memory_plan\": ";
    write_memory_plan(out, doc.memory_plan, "  ");
    out << ",\n  \"wall_seconds\": ";
    write_json_double(out, doc.wall_seconds);
    out << ",\n"
        << "  \"mode\": \"" << solver_mode_name(doc.options.solver.mode) << "\",\n"
        << "  \"distance_backend\": \"" << knn_backend_name(doc.options.solver.knn_backend) << "\",\n"
        << "  \"oracle_status\": \"" << json_escape(doc.options.solver.oracle.status) << "\",\n"
        << "  \"p_values\": ";
    write_double_array(out, doc.p_values);
    out << ",\n  \"config\": ";
    write_generated_config(out, doc.options, "  ");
    out << ",\n  \"search_stats\": ";
    write_stats(out, doc.stats, "  ");
    out << ",\n  \"oracle_call_records\": ";
    write_oracle_call_records(out, doc.stats.oracle_call_records, "  ");
    out << ",\n  \"summary\": {\n";

    bool first = true;
    for (const auto& item : doc.summary) {
        if (!first) {
            out << ",\n";
        }
        first = false;
        const PValueSummary& s = item.second;
        out << "    \"" << json_escape(item.first) << "\": {\n"
            << "      \"k\": " << s.k << ",\n"
            << "      \"mean\": ";
        write_json_double(out, s.mean);
        out << ",\n      \"std\": ";
        write_json_double(out, s.stddev);
        out << ",\n      \"stderr\": ";
        write_json_double(out, s.stderr_value);
        out << ",\n      \"min\": ";
        write_json_double(out, s.min);
        out << ",\n      \"max\": ";
        write_json_double(out, s.max);
        out << ",\n      \"exact_optimal_instances\": "
            << s.exact_optimal_instances;
        out << ",\n      \"n\": " << s.values.size() << ",\n      \"values\": ";
        write_double_array(out, s.values);
        out << "\n    }";
    }
    out << "\n  },\n  \"summary_rows\": [\n";
    bool first_row = true;
    for (std::size_t pi = 0; pi < doc.p_values.size(); ++pi) {
        const double p_value = doc.p_values[pi];
        const auto found = doc.summary.find(summary_key(p_value));
        if (found == doc.summary.end()) {
            continue;
        }
        if (!first_row) {
            out << ",\n";
        }
        first_row = false;
        const PValueSummary& s = found->second;
        out << "    {\n      \"p\": ";
        write_json_double(out, p_value);
        out << ",\n      \"key\": \"" << json_escape(found->first) << "\",\n"
            << "      \"k\": " << s.k << ",\n      \"mean\": ";
        write_json_double(out, s.mean);
        out << ",\n      \"std\": ";
        write_json_double(out, s.stddev);
        out << ",\n      \"stderr\": ";
        write_json_double(out, s.stderr_value);
        out << ",\n      \"min\": ";
        write_json_double(out, s.min);
        out << ",\n      \"max\": ";
        write_json_double(out, s.max);
        out << ",\n      \"best_restart_max\": " << s.best_restart_max;
        out << ",\n      \"executed_restarts_max\": " << s.executed_restarts_max;
        out << ",\n      \"solve_seconds_total\": ";
        write_json_double(out, s.solve_seconds_total);
        out << ",\n      \"exact_optimal_instances\": "
            << s.exact_optimal_instances;
        if (s.has_control_variate) {
            out << ",\n      \"conditional_two_nn_bound_mean\": ";
            write_json_double(out, s.conditional_two_nn_bound_mean);
            out << ",\n      \"conditional_two_nn_gap_mean\": ";
            write_json_double(out, s.conditional_two_nn_gap_mean);
            // Schema-13 compatibility aliases. These names are retained for
            // old analysis consumers but do not imply a global subset bound.
            out << ",\n      \"subset_bound_mean\": ";
            write_json_double(out, s.conditional_two_nn_bound_mean);
            out << ",\n      \"lower_bound_gap_mean\": ";
            write_json_double(out, s.conditional_two_nn_gap_mean);
            out << ",\n      \"cv_mean\": ";
            write_json_double(out, s.cv_mean);
            out << ",\n      \"cv_stderr\": ";
            write_json_double(out, s.cv_stderr);
            out << ",\n      \"cv_variance_reduction\": ";
            write_json_double(out, s.cv_variance_reduction);
        }
        if (s.has_held_karp) {
            out << ",\n      \"conditional_held_karp_bound_mean\": ";
            write_json_double(out, s.conditional_held_karp_bound_mean);
            out << ",\n      \"conditional_held_karp_gap_mean\": ";
            write_json_double(out, s.conditional_held_karp_gap_mean);
            out << ",\n      \"held_karp_bound_mean\": ";
            write_json_double(out, s.conditional_held_karp_bound_mean);
            out << ",\n      \"held_karp_gap_mean\": ";
            write_json_double(out, s.conditional_held_karp_gap_mean);
        }
        out << ",\n      \"n\": " << s.values.size() << ",\n      \"values\": ";
        write_double_array(out, s.values);
        out << "\n    }";
    }
    out << "\n  ],\n  \"instance_rows\": ";
    write_instance_rows(out, doc.instance_rows);
    out << "\n}\n";
    return out.str();
}

} // namespace aldous_tsp
