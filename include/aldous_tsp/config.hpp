#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
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
// Exponential exact cardinality-k subset-tour oracle. N=18 requires about
// 43 MiB for its DP and parent tables; raising this limit should be accompanied
// by explicit memory/time benchmarks and a wider mask/parent representation.
inline constexpr int kExactSubsetHardLimit = 18;
inline constexpr int kDefaultSchemaVersion = 13;
inline constexpr std::int64_t kMaxGridCells = 262144;

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
    std::uint64_t path_relink_attempts = 0;
    std::uint64_t path_relink_feasible = 0;
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

struct ExternalOracleConfig {
    ExternalOracleMode mode = ExternalOracleMode::None;
    OracleProblemFormat problem_format = OracleProblemFormat::Matrix;
    std::string lkh_path = "LKH";
    std::string concorde_path = "concorde";
    int time_limit_sec = 0;
    int scale = 1000000;
    int tsp_top = 1;
    int subset_top = 1;
    int min_k = kExactSmallTourLimit + 1;
    int max_k = 2500;
    int lkh_runs = 6;
    int lkh_max_trials = 0;
    bool use_for_tsp = true;
    bool use_for_subset = true;
    bool inline_feedback = false;
    bool verbose = false;
};

struct OracleContext {
    ExternalOracleConfig cfg;
    ResolvedOracleMode resolved = ResolvedOracleMode::None;
    std::string exec_path;
    std::string version = "unknown";
    std::string status = "disabled";
};

struct SolverOptions {
    int knn_k = 40;
    KnnBackend knn_backend = KnnBackend::GridExact;
    int verify_knn_checks = 0;
    double grid_cell = 0.0;

    // Use the global exact subset-tour dynamic program whenever N is at most
    // this threshold. Zero disables it. The hard safety cap is 18.
    int exact_subset_max_n = 0;
    int tsp_restarts = 5;
    int tsp_ils = 300;
    int tsp_patience = 80;
    // Subset restarts. Values >= 1 are authoritative: the solver runs exactly
    // this many. The default -1 means AUTO, which reproduces the historical
    // effective behavior: 8 restarts when p <= 0.08 (the small-p seed pool used
    // to force this regardless of the flag) and 3 otherwise. Before 0.9.2 the
    // flag could not lower the count below the seed-pool size, so honoring it
    // literally would have silently degraded default-quality at small p.
    int subset_restarts = -1;
    // Number of warm/continuation restarts when a parent solution is available.
    // Under Supplemental these are appended to --restarts. Under FixedBudget
    // they occupy an explicit quota within --restarts. A secondary sweep runs
    // continuation restarts only rather than repeating the independent search.
    int continuation_restarts = 1;
    ContinuationPolicy continuation_policy = ContinuationPolicy::Supplemental;
    // Optional deterministic supplemental restart racing. Independent
    // diagnostic restarts remain unchanged; race candidates are a separate
    // population screened with a fixed SA pilot and a diversity-aware stable
    // promotion rule. Zero candidates disables racing (the default).
    int racing_candidates = 0;
    int racing_survivors = 2;
    int racing_pilot_iters = 2000;
    double racing_min_jaccard = 0.05;
    int sa_iters = 60000;
    // Additional SA iterations per subset element: the effective SA budget for
    // a size-k solve is sa_iters + sa_iters_per_k * k. The default 0 keeps the
    // historical flat budget.
    int sa_iters_per_k = 0;

    // Extra SA iterations per CANDIDATE point N. The subset search chooses k of
    // N points, so the *pool* it must explore scales with N, not k: with a budget
    // flat in N the search cannot even propose each candidate once (at N=200000
    // and sa_iters=60000 it sees 30% of them), and the extra candidates that a
    // smaller p buys are simply never looked at. That silently biases f(p) upward
    // by more at small p than at large p -- i.e. it corrupts exactly the p-trend
    // the study measures. Set this to make the budget scale with the pool; a
    // ratio of ~10-25 iterations per candidate is where L/k has been observed to
    // stop improving. The default 0 preserves the historical flat budget.
    int sa_iters_per_n = 0;

    // SA move insertion policy. The historical move searched ALL k tour
    // positions for the globally best insertion of the incoming node -- an
    // O(k) scan per SA move (~19 us of the measured ~21.7 us per move at
    // k=2000), making move cost grow linearly in k. The windowed policy
    // evaluates only (a) tour positions within sa_insertion_window of the
    // removed slot (spatially local slots -- where a KNN-sourced candidate's
    // best insertion lives, since tour spacing p^-1/2 exceeds the KNN radius
    // at small p) and (b) slots adjacent to in-tour members of the incoming
    // node's KNN row (the far-relocation slots, when they exist). Exact
    // semantics are preserved behind sa_exact_insertion for A/B validation.
    bool sa_exact_insertion = false;
    int sa_insertion_window = 12;

    // Elite-kick restarts. The B=960/B=1920 allocation scans at k=2000 showed
    // pure independent multistart: anneals converge by ~60 iters/candidate and
    // the frontier then improves a CONSTANT ~0.0025 per DOUBLING of restarts
    // (Gumbel-like best-of-m behavior) -- no plateau, ever. Kicks replace the
    // last subset_kick_restarts of the scheduled restarts: each seeds from an
    // elite member of the INDEPENDENT phase, perturbed by swapping
    // kick_fraction of its members to KNN neighbors of retained members, and
    // anneals at the reduced temperature kick_t0 so the inherited structure is
    // refined rather than re-melted. The elite snapshot is taken exactly once,
    // at the independent->kick boundary, so results are independent of
    // restart_threads. Default 0 preserves pure multistart.
    // Region seeds. Diagnostics at p=0.01 (per-restart geometry logging) showed
    // that random seeds -- the ONLY kind that produces frontier draws, because
    // each one samples a different region of the square -- fail to contract in
    // ~half of their restarts, leaving a spread-out subset at L/k ~ 1.0-1.8.
    // A region seed samples the same region space directly: pick a uniform
    // center, take the k nearest points, order them. It starts where a random
    // seed spends half its budget trying to arrive, and being compact from the
    // outset it can use the fast windowed insertion kernel.
    // Small-p seed fill. The restart pool is topped up to --restarts with
    // "explorer" seeds, historically ALTERNATING dense_seed and random_subset.
    // Per-restart geometry logging at p=0.01 (k=200, 24 restarts, 4 instances)
    // showed the alternation is half dead weight: every dense_seed restart
    // contracted (mean radius 11) and produced every frontier draw (best
    // 0.5858), while 32/32 random_subset restarts NEVER contracted (mean radius
    // 44.7) and none got below L/k = 1.20 -- a third of the total SA budget
    // spent on tours that are not competitive by a factor of two. At small p a
    // uniformly random subset is simply too far from any good configuration to
    // be reeled in at realistic budgets. With this on, the fill uses dense_seed
    // variants only when p <= 0.08. Set false to restore the alternation.
    bool small_p_dense_fill = true;

    // Exploration seeds (random / high-p) use the exact O(k) insertion scan
    // instead of the windowed kernel: windowed insertion only offers slots near
    // the removed position or near the incoming node's in-tour neighbors, which
    // is fatal for a subset that must relocate globally to contract. Turning
    // this off restores the 0.9.4 behavior (windowed for everyone) -- measured
    // at +0.0109 +/- 0.0031 WORSE on the final value (k=200, p=0.01), so it is
    // on by default. It is a flag because with small_p_dense_fill the explorer
    // seeds are compact by construction, so the cheap kernel may now suffice;
    // that trade (quality per move vs moves per second) has to be settled at
    // k=2000, where the exact scan is ~5x more expensive per move.
    bool exploration_exact_insertion = true;

    // Spatial insertion kernel: candidate slots are the tour edges adjacent to
    // the nearest CURRENT subset members of the incoming node, found through a
    // live spatial index rather than through tour positions (which sit around
    // the wrong node) or the static full-set KNN row (which is empty at small
    // p). This is what the exact O(k) scan was being used to emulate, at a
    // fraction of the cost. Overridden by --sa-exact-insertion.
    bool sa_spatial_insertion = false;

    // 0.9.5 forced the exact O(k) insertion scan on dense seeds too (they shared
    // kind code 0 with random_subset). At k=200 that measured -0.0109 (3.5 sigma);
    // at k=2000 it returns a BIT-IDENTICAL search at +58% wall. Kept as a flag so
    // the k=200 result stays reproducible, but off by default.
    bool dense_exact_insertion = false;
    int sa_spatial_neighbors = 16;

    bool region_seeds = false;
    // Dilation of a region seed: take the (dilation * k) nearest points to the
    // center, then choose k of them at random. Dilation 1 gives the compact
    // disc of k nearest points -- which measurably UNDERPERFORMS random seeds:
    // it removes the cherry-picking freedom that lets a subset beat the
    // beta ~ 0.71 "take every point in a small disc" regime. Larger dilation
    // restores that freedom while keeping the region localized.
    double region_dilation = 3.0;

    int subset_kick_restarts = 0;
    double kick_fraction = 0.10;
    double kick_t0 = 0.35;
    // Simulated-annealing temperature schedule endpoints (geometric). The move
    // acceptance probability is exp(-max(0,delta)/T). Defaults reproduce the
    // historical hardcoded schedule.
    double sa_t0 = 1.4;
    double sa_t1 = 0.00005;
    // Number of worker threads for parallel subset restarts within one
    // (instance, p) solve. 1 = sequential. Per-restart RNG streams make
    // results invariant to this value outside time-budget mode; in budget
    // mode more threads execute more restarts per unit wall-clock.
    int restart_threads = 1;
    // Wall-clock target in seconds per (instance, p) solve. 0 disables. When
    // set, the solver runs at least its configured restarts and then keeps
    // launching additional restarts until the budget has elapsed; the final
    // restart always runs to completion. Restart counts become wall-clock
    // dependent, so bitwise reproducibility across machines is intentionally
    // traded for anytime behavior.
    double time_budget_per_p = 0.0;
    int final_exhaustive_k = 300;
    ExhaustiveTwoOptPolicy exhaustive_two_opt_policy = ExhaustiveTwoOptPolicy::FinalOnly;
    int subset_swap_descent_passes = 1;
    int pair_exchange_passes = 1;
    // Pair exchange is intentionally bounded until its O(k * pool) working
    // set and full candidate generation have been validated at larger scales.
    // 0 removes the gate.
    int pair_exchange_max_k = 5000;
    int ruin_recreate_rounds = 4;
    // Adaptive large-neighborhood search cycles through worst-marginal,
    // contiguous-segment, spatial-cluster, long-edge, and random ruin
    // operators while increasing the ruined cardinality across rounds.
    // The absolute and fractional caps keep the repair working set bounded at
    // very large k; 0 for max_nodes removes only the absolute cap.
    bool adaptive_ruin_recreate = true;
    double ruin_recreate_max_fraction = 0.05;
    int ruin_recreate_max_nodes = 96;
    int ruin_recreate_pool_cap = 640;
    // Variable-depth membership chains can traverse a sequence of temporarily
    // uphill swaps and retain the best locally polished prefix. Added nodes are
    // locked and removed nodes are tabu within one chain, so depth d explores
    // a true d-for-d membership change rather than cycling one node in and out.
    int ejection_chain_starts = 3;
    int ejection_chain_depth = 6;
    int ejection_chain_candidates = 24;
    int ejection_chain_remove_cap = 96;
    // Maximum cumulative uphill excursion, measured in mean-tour-edge units.
    double ejection_chain_max_uphill = 0.75;
    // Additional archive slots selected for set diversity. The ordinary
    // length-ranked `keep` entries remain protected, so enabling these slots
    // cannot evict a solution the legacy archive would have retained.
    int elite_diversity_slots = 4;
    double elite_min_jaccard = 0.02;
    double elite_quality_slack = 0.03;
    int path_relink_top = 3;
    int seed = 2024;
    SolverMode mode = SolverMode::Balanced;

    bool disable_two_opt = false;
    bool disable_or_opt = false;
    bool disable_subset_swap = false;
    bool disable_pair_exchange = false;
    // In time-budget (anytime) mode, seed restarts beyond the scheduled count
    // from perturbed elite members (subset-level ILS) instead of cold seeds.
    bool disable_elite_restarts = false;
    bool disable_ruin_recreate = false;
    bool disable_ejection_chain = false;
    bool disable_path_relink = false;
    bool disable_smallp_seeds = false;
    bool disable_highp_delete = false;
    OracleContext oracle;
};

struct RunOptions {
    int N = 500;
    int instances = 15;
    // 0 means auto-detect/cap to the instance count during validation.
    int threads = 0;
    // When true, generate instances on a flat torus (periodic boundary
    // conditions) instead of the open square. Removes the O(1/sqrt N) boundary
    // correction to f(p,N), leaving O(1/N) (Percus & Martin 1996); the limit
    // f(p) is unchanged (Jaillet 1993). Enables clean small-k extrapolation.
    bool periodic = false;
    // Compute the two-nearest-neighbor lower bound (Percus-Martin control
    // variate) per instance: on the full point set (analytic/MC-known mean,
    // used to variance-reduce the reported mean) and on the selected subset.
    // The latter rigorously lower-bounds only the optimal tour through that
    // selected subset; it does not lower-bound the optimum over all size-k
    // subsets. Off by default; adds a cheap subset KNN per solve.
    bool control_variate = false;
    // Cheap Monte-Carlo instances used to pin E[B_full] for the control
    // variate (KNN only, no solve). Capped internally for very large N.
    int cv_mc_samples = 2000;
    // Compute the Held-Karp (Lagrangian 1-tree) lower bound per selected subset.
    // Its gap certifies tour-ordering quality conditional on that subset; it is
    // not a global lower bound on the optimum over all size-k subsets. Off by
    // default; O(k^2) per solve, so intended for moderate k.
    bool held_karp = false;
    int hk_iterations = 400;
    bool verbose = false;
    bool include_instance_rows = false;
    // After the descending warm-start sweep over p, run a second ascending
    // sweep seeding each p from the grown best solution at the next smaller p
    // and keep the better result per p. Roughly doubles subset wall-clock.
    bool second_sweep = false;
    bool force_output = false;
    bool dry_run = false;
    bool dump_config = false;
    std::string output_path = "results.json";
    std::vector<double> p_values;
    SolverOptions solver;
};

const char* solver_mode_name(SolverMode mode) noexcept;
bool parse_solver_mode(const std::string& text, SolverMode& out) noexcept;
const char* continuation_policy_name(ContinuationPolicy policy) noexcept;
bool parse_continuation_policy(const std::string& text, ContinuationPolicy& out) noexcept;
const char* knn_backend_name(KnnBackend backend) noexcept;
bool parse_knn_backend(const std::string& text, KnnBackend& out) noexcept;
const char* exhaustive_two_opt_policy_name(ExhaustiveTwoOptPolicy policy) noexcept;
bool parse_exhaustive_two_opt_policy(const std::string& text, ExhaustiveTwoOptPolicy& out) noexcept;
const char* external_oracle_mode_name(ExternalOracleMode mode) noexcept;
bool parse_external_oracle_mode(const std::string& text, ExternalOracleMode& out) noexcept;
const char* oracle_problem_format_name(OracleProblemFormat format) noexcept;
bool parse_oracle_problem_format(const std::string& text, OracleProblemFormat& out) noexcept;
const char* resolved_oracle_mode_name(ResolvedOracleMode mode) noexcept;
std::vector<double> default_p_values();

} // namespace aldous_tsp
