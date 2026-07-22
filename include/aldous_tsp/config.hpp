#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace aldous_tsp {

static_assert(__cplusplus >= 201703L,
    "aldous_tsp requires C++17 or newer. On MSVC, build with /Zc:__cplusplus "
    "so the __cplusplus macro is reported correctly.");

inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr double kImprovementEps = 1e-10;
inline constexpr double kDistanceEps = 1e-12;
inline constexpr double kGeomBoundEps = 1e-15;
inline constexpr double kBhhReference = 0.7124;
inline constexpr int kExactSmallTourLimit = 16;
// Exponential exact cardinality-k subset-tour oracle. The implementation uses
// cardinality layers and reports a pre-allocation memory estimate; raising this
// limit still requires explicit time/memory benchmarks and wider mask storage.
inline constexpr int kExactSubsetHardLimit = 18;
#include "aldous_tsp/generated/options_schema_version.inc"
inline constexpr std::int64_t kMaxGridCells = 262144;
inline constexpr std::size_t kSaTemperatureBins = 10U;

struct OracleCallRecord {
    std::string type;
    int k = 0;
    std::string solver;
    std::string format;
    std::string status;
    double before_length = 0.0;
    double after_length = std::numeric_limits<double>::quiet_NaN();
    double gain = 0.0;
    double seconds = 0.0;
    std::string exec_path;
    std::string exec_sha256;
    std::string solver_version;
    std::string error;
};

// Accumulated phase timings. Values are elapsed-seconds summed across restart
// workers, not necessarily wall-clock seconds; nested fields (for example
// sa_checkpoint_polish_seconds within sa_seconds) are deliberately reported
// separately and must not be summed as disjoint phases.
struct SearchPhaseTiming {
    double seed_construction_seconds = 0.0;
    double tsp_construction_seconds = 0.0;
    double initial_polish_seconds = 0.0;
    double sa_seconds = 0.0;
    double sa_checkpoint_polish_seconds = 0.0;
    double post_sa_polish_seconds = 0.0;
    double subset_swap_seconds = 0.0;
    double highp_exchange_seconds = 0.0;
    double pair_exchange_seconds = 0.0;
    double ruin_recreate_seconds = 0.0;
    double ejection_chain_seconds = 0.0;
    double exact_subset_seconds = 0.0;
    double path_relink_seconds = 0.0;
    double tsp_ils_seconds = 0.0;
    double final_polish_seconds = 0.0;
    double oracle_seconds = 0.0;

    // Low-overhead deterministic samples taken every 64 SA iterations. The
    // sample durations are raw measured seconds; divide by the corresponding
    // sample count to obtain average proposal/insertion latency without putting
    // a clock read on every move.
    std::uint64_t sa_proposal_samples = 0;
    std::uint64_t sa_insertion_samples = 0;
    double sa_proposal_sample_seconds = 0.0;
    double sa_insertion_sample_seconds = 0.0;

    void add(const SearchPhaseTiming& other) noexcept;
};

struct SearchStats {
    // Exact subset oracle telemetry. A solved call globally proves both the
    // selected cardinality-k subset and its cycle optimal.
    std::uint64_t exact_subset_calls = 0;
    std::uint64_t exact_subset_solved = 0;
    std::uint64_t exact_subset_states = 0;
    std::uint64_t exact_subset_transitions = 0;
    // Maximum cardinality-sensitive working-storage estimate among exact calls.
    std::uint64_t exact_subset_peak_memory_bytes = 0;
    // Full-TSP construction/racing telemetry. Candidate starts receive the
    // cheap construction + initial-polish pilot; promoted starts receive the
    // configured ILS budget and produce restart records.
    std::uint64_t tsp_candidate_starts = 0;
    std::uint64_t tsp_promoted_restarts = 0;
    std::uint64_t tsp_restarts = 0;
    std::uint64_t tsp_ils_iterations = 0;
    std::uint64_t subset_restarts = 0;
    std::uint64_t smallp_seed_restarts = 0;
    std::uint64_t highp_delete_restarts = 0;
    std::uint64_t warm_restarts = 0;
    std::uint64_t random_restarts = 0;
    std::uint64_t region_restarts = 0;
    std::uint64_t dense_restarts = 0;
    // Deterministic supplemental restart racing. Every race candidate receives
    // the fixed pilot budget; promoted candidates are then rerun from the same
    // seed and RNG stream at the full budget. `subset_restarts` counts unique
    // candidates, while these counters expose the actual staged allocation.
    std::uint64_t racing_pilot_restarts = 0;
    std::uint64_t racing_promoted_restarts = 0;
    // Staged subset-search funnel. Candidates receive seed construction, local
    // ordering, SA, and post-SA polish. Only selected finalists receive the
    // expensive exact membership neighborhoods and optional inline oracle.
    std::uint64_t strong_polish_candidates = 0;
    std::uint64_t strong_polish_finalists = 0;
    std::uint64_t strong_polish_improvements = 0;
    // All elite-seeded restarts. Scheduled kicks are included here and are
    // additionally counted by kick_restarts for exact kick accounting.
    std::uint64_t elite_restarts = 0;
    std::uint64_t kick_restarts = 0;
    std::uint64_t elite_diversity_candidates = 0;
    std::uint64_t elite_diversity_retained = 0;
    std::uint64_t elite_diversity_rejected = 0;
    std::uint64_t two_opt_scans = 0;
    std::uint64_t two_opt_improvements = 0;
    std::uint64_t or_opt_scans = 0;
    std::uint64_t or_opt_improvements = 0;
    std::uint64_t sa_moves = 0;
    std::uint64_t sa_accepted = 0;
    std::uint64_t sa_improving = 0;
    // Experimental SA-controller telemetry. Defaults preserve the historical
    // one-candidate, fixed-temperature trajectory exactly.
    std::uint64_t sa_candidate_evaluations = 0;
    std::uint64_t sa_multiple_try_iterations = 0;
    std::uint64_t sa_temperature_schedules = 0;
    std::uint64_t sa_temperature_calibrations = 0;
    std::uint64_t sa_temperature_fallbacks = 0;
    std::uint64_t sa_calibration_attempts = 0;
    std::uint64_t sa_calibration_uphill_samples = 0;
    double sa_temperature_t0_sum = 0.0;
    double sa_temperature_t1_sum = 0.0;
    double sa_temperature_t0_min = std::numeric_limits<double>::infinity();
    double sa_temperature_t0_max = 0.0;
    double sa_temperature_t1_min = std::numeric_limits<double>::infinity();
    double sa_temperature_t1_max = 0.0;
    std::array<std::uint64_t, kSaTemperatureBins> sa_decile_moves{};
    std::array<std::uint64_t, kSaTemperatureBins> sa_decile_uphill_moves{};
    std::array<std::uint64_t, kSaTemperatureBins> sa_decile_accepted{};
    std::array<std::uint64_t, kSaTemperatureBins> sa_decile_uphill_accepted{};
    std::uint64_t subset_swap_scans = 0;
    std::uint64_t subset_swap_improvements = 0;
    std::uint64_t highp_exchange_scans = 0;
    std::uint64_t highp_exchange_improvements = 0;
    std::uint64_t pair_exchange_scans = 0;
    std::uint64_t pair_exchange_improvements = 0;
    std::uint64_t pair_exchange_skipped_large_k = 0;
    std::uint64_t ruin_recreate_attempts = 0;
    std::uint64_t ruin_recreate_improvements = 0;
    std::uint64_t ruin_recreate_removed_nodes = 0;
    std::uint64_t ruin_recreate_worst_attempts = 0;
    std::uint64_t ruin_recreate_worst_improvements = 0;
    std::uint64_t ruin_recreate_segment_attempts = 0;
    std::uint64_t ruin_recreate_segment_improvements = 0;
    std::uint64_t ruin_recreate_spatial_attempts = 0;
    std::uint64_t ruin_recreate_spatial_improvements = 0;
    std::uint64_t ruin_recreate_long_edge_attempts = 0;
    std::uint64_t ruin_recreate_long_edge_improvements = 0;
    std::uint64_t ruin_recreate_random_attempts = 0;
    std::uint64_t ruin_recreate_random_improvements = 0;
    std::uint64_t ejection_chain_attempts = 0;
    std::uint64_t ejection_chain_feasible = 0;
    std::uint64_t ejection_chain_steps = 0;
    std::uint64_t ejection_chain_scans = 0;
    std::uint64_t ejection_chain_improvements = 0;
    std::uint64_t ejection_chain_accepted_depth = 0;
    std::uint64_t path_relink_pairs_considered = 0;
    std::uint64_t path_relink_pairs_skipped_distance = 0;
    std::uint64_t path_relink_pairs_skipped_budget = 0;
    std::uint64_t path_relink_attempts = 0;
    std::uint64_t path_relink_feasible = 0;
    std::uint64_t path_relink_removed_sum = 0;
    std::uint64_t path_relink_candidate_scans = 0;
    std::uint64_t path_relink_elite_insertions = 0;
    std::uint64_t path_relink_best_improvements = 0;
    std::uint64_t path_relink_improvements = 0; // Backward-compatible alias for best improvements.
    double knn_build_seconds = 0.0;
    std::uint64_t knn_requested_grid_instances = 0;
    std::uint64_t knn_requested_bruteforce_instances = 0;
    std::uint64_t knn_effective_grid_instances = 0;
    std::uint64_t knn_effective_bruteforce_instances = 0;
    std::uint64_t knn_bruteforce_fallback_instances = 0;
    std::uint64_t knn_grid_cell_capped_instances = 0;
    std::uint64_t knn_grid_cell_samples = 0;
    std::uint64_t knn_grid_cells_sum = 0;
    std::uint64_t knn_grid_cells_max = 0;
    double grid_cell_effective_min = 0.0;
    double grid_cell_effective_max = 0.0;
    double grid_cell_effective_sum = 0.0;
    double tsp_seconds = 0.0;
    double subset_seconds = 0.0;
    std::uint64_t oracle_calls = 0;
    std::uint64_t oracle_solved = 0;
    std::uint64_t oracle_improved = 0;
    std::uint64_t oracle_failed = 0;
    std::uint64_t oracle_tsp_calls = 0;
    std::uint64_t oracle_subset_calls = 0;
    double oracle_gain = 0.0;
    std::vector<OracleCallRecord> oracle_call_records;
    SearchPhaseTiming phases;

    void add(const SearchStats& other);
};

enum class SolverMode {
    Balanced,
    SmallPRegion,
    HighPDelete,
    Hybrid
};

// How warm/continuation restarts interact with the configured restart budget.
// Supplemental preserves the complete independent population and appends warm
// restarts, so adding neighboring p-values cannot worsen a fixed-budget result.
// FixedBudget reserves an explicit quota inside --restarts for matched-compute
// studies; the independent prefix remains deterministic and visible.
enum class ContinuationPolicy {
    Supplemental,
    FixedBudget,
};

// Selects an evidence-backed automatic search controller. The legacy value
// preserves the published release behavior; held-out presets alter only the
// SA/TSP sections whose affected controls remain at release defaults.
enum class SearchPolicyPreset {
    LegacyBalanced,
    HeldoutBalanced,
    HeldoutQuality,
};

enum class KnnBackend {
    BruteForce,
    GridExact
};

enum class ExhaustiveTwoOptPolicy {
    Never,
    FinalOnly,
    AllPolish
};

enum class ExternalOracleMode {
    None,
    Auto,
    Lkh,
    Concorde
};

enum class ResolvedOracleMode {
    None,
    Lkh,
    Concorde
};

enum class OracleProblemFormat {
    Matrix,
    Euc2d
};

// Durability contract for atomic result replacement. `File` flushes the
// temporary file before replacement; `Full` additionally synchronizes the
// containing directory on platforms that expose that operation.
enum class OutputDurability {
    None,
    File,
    Full,
};

struct ExternalOracleConfig {
#include "aldous_tsp/generated/external_oracle_fields.inc"
};

struct OracleContext {
    ExternalOracleConfig cfg;
    ResolvedOracleMode resolved = ResolvedOracleMode::None;
    std::string exec_path;
    std::string exec_sha256 = "unknown";
    std::string version = "unknown";
    std::string status = "disabled";
};

struct SolverOptions {
#include "aldous_tsp/generated/solver_options_fields.inc"

    OracleContext oracle;
};

struct RunOptions {
#include "aldous_tsp/generated/run_options_fields.inc"

    SolverOptions solver;
};

inline int effective_point_seed(const RunOptions& options) noexcept {
    return options.point_seed.value_or(options.solver.seed);
}

inline int effective_search_seed(const RunOptions& options) noexcept {
    return options.search_seed.value_or(options.solver.seed);
}

const char* solver_mode_name(SolverMode mode) noexcept;
bool parse_solver_mode(const std::string& text, SolverMode& out) noexcept;
const char* continuation_policy_name(ContinuationPolicy policy) noexcept;
bool parse_continuation_policy(const std::string& text, ContinuationPolicy& out) noexcept;
const char* search_policy_preset_name(SearchPolicyPreset policy) noexcept;
bool parse_search_policy_preset(const std::string& text, SearchPolicyPreset& out) noexcept;
const char* knn_backend_name(KnnBackend backend) noexcept;
bool parse_knn_backend(const std::string& text, KnnBackend& out) noexcept;
const char* exhaustive_two_opt_policy_name(ExhaustiveTwoOptPolicy policy) noexcept;
bool parse_exhaustive_two_opt_policy(const std::string& text, ExhaustiveTwoOptPolicy& out) noexcept;
const char* external_oracle_mode_name(ExternalOracleMode mode) noexcept;
bool parse_external_oracle_mode(const std::string& text, ExternalOracleMode& out) noexcept;
const char* oracle_problem_format_name(OracleProblemFormat format) noexcept;
bool parse_oracle_problem_format(const std::string& text, OracleProblemFormat& out) noexcept;
const char* resolved_oracle_mode_name(ResolvedOracleMode mode) noexcept;
const char* output_durability_name(OutputDurability durability) noexcept;
bool parse_output_durability(const std::string& text, OutputDurability& out) noexcept;
std::vector<double> default_p_values();

} // namespace aldous_tsp
