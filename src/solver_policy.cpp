#include "solver_internal.hpp"

#include <algorithm>
#include <cmath>

namespace aldous_tsp {
namespace {

constexpr double kPolicyBoundaryEps = 1e-12;
constexpr double kHeldoutMinP = 0.02;
constexpr double kHeldoutOpenMaxP = 0.35;
constexpr double kHeldoutPeriodicBalancedMaxP = 0.35;
constexpr double kHeldoutPeriodicQualityMaxP = 0.50;
constexpr int kHeldoutMinK = 40;
constexpr int kHeldoutBalancedSaIterations = 20000;
constexpr int kHeldoutQualitySaIterations = 30000;
constexpr int kHeldoutCandidateTrials = 4;
constexpr double kHeldoutRandomSelectionProbability = 0.10;
constexpr int kHeldoutQualityTspCandidateStarts = 16;
constexpr int kHeldoutQualityTspPromotedRestarts = 4;

bool in_closed_range(const double value,
                     const double lower,
                     const double upper) noexcept {
    return value + kPolicyBoundaryEps >= lower
        && value <= upper + kPolicyBoundaryEps;
}

bool has_untouched_sa_controller(const SolverOptions& options) noexcept {
    const SolverOptions defaults;
    return options.sa_iters == defaults.sa_iters
        && options.sa_iters_per_k == defaults.sa_iters_per_k
        && options.sa_iters_per_n == defaults.sa_iters_per_n
        && options.sa_candidate_trials == defaults.sa_candidate_trials
        && options.sa_multiple_try_random_probability
               == defaults.sa_multiple_try_random_probability
        && options.sa_t0 == defaults.sa_t0
        && options.sa_t1 == defaults.sa_t1
        && options.sa_auto_temperature == defaults.sa_auto_temperature;
}

} // namespace

ResolvedSubsetPolicy resolve_subset_policy(const SolverOptions& options,
                                           const double p,
                                           const int k,
                                           const bool periodic,
                                           const bool continuation_only) noexcept {
    ResolvedSubsetPolicy resolved;
    resolved.restarts = options.subset_restarts >= 1
        ? options.subset_restarts
        : (options.staged_search
               ? (p <= 0.08 ? 12 : 5)
               : (p <= 0.08 ? 8 : 3));
    resolved.strong_polish_finalists = options.strong_polish_finalists;
    resolved.racing_candidates = continuation_only ? 0 : options.racing_candidates;
    resolved.racing_survivors = options.racing_survivors;
    resolved.racing_pilot_iters = options.racing_pilot_iters;
    resolved.sa_candidate_trials = options.sa_candidate_trials;
    resolved.sa_multiple_try_random_probability =
        options.sa_multiple_try_random_probability;

    // The held-out proposal policies were evaluated only for ordinary,
    // deterministic staged restarts. Explicit SA controls, continuation-only
    // solves, elapsed-time search, and non-staged trajectories retain their
    // literal behavior. The k guard avoids extrapolating into the tiny-tour
    // regime where the held-out policy was neutral.
    if (options.search_policy_preset == SearchPolicyPreset::LegacyBalanced
        || continuation_only
        || !options.staged_search
        || options.time_budget_per_p > 0.0
        || !has_untouched_sa_controller(options)
        || k < kHeldoutMinK) {
        return resolved;
    }

    double maximum_p = kHeldoutOpenMaxP;
    int sa_iterations = kHeldoutBalancedSaIterations;
    if (options.search_policy_preset == SearchPolicyPreset::HeldoutQuality) {
        sa_iterations = kHeldoutQualitySaIterations;
        if (periodic) {
            maximum_p = kHeldoutPeriodicQualityMaxP;
        }
    } else if (periodic) {
        maximum_p = kHeldoutPeriodicBalancedMaxP;
    }

    if (!in_closed_range(p, kHeldoutMinP, maximum_p)) {
        return resolved;
    }

    resolved.sa_iterations = sa_iterations;
    resolved.sa_candidate_trials = kHeldoutCandidateTrials;
    resolved.sa_multiple_try_random_probability =
        kHeldoutRandomSelectionProbability;
    resolved.heldout_sa_policy_applied = true;
    return resolved;
}

ResolvedTspPolicy resolve_tsp_policy(const SolverOptions& options,
                                     const bool periodic) noexcept {
    (void)periodic;
    ResolvedTspPolicy resolved;
    resolved.candidate_starts = options.tsp_candidate_starts;
    resolved.promoted_restarts = options.tsp_restarts;
    resolved.ils_iterations = options.tsp_ils;

    const SolverOptions defaults;
    const bool untouched_tsp_controller =
        options.tsp_candidate_starts == defaults.tsp_candidate_starts
        && options.tsp_restarts == defaults.tsp_restarts
        && options.tsp_ils == defaults.tsp_ils;
    if (!untouched_tsp_controller
        || options.search_policy_preset != SearchPolicyPreset::HeldoutQuality) {
        return resolved;
    }

    // Held-out open and periodic campaigns found a consistent quality gain
    // from screening sixteen starts and promoting four into a deeper ILS.
    resolved.candidate_starts = kHeldoutQualityTspCandidateStarts;
    resolved.promoted_restarts = kHeldoutQualityTspPromotedRestarts;
    resolved.ils_iterations = 450;
    return resolved;
}

} // namespace aldous_tsp
