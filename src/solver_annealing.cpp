#include "solver_internal.hpp"

#include <algorithm>
#include <cmath>

namespace aldous_tsp {
namespace {

SwapInsertionMove evaluate_sa_candidate(const Instance& inst,
                                         const Tour& tour,
                                         const int remove_pos,
                                         const int add_node,
                                         const SolverOptions& options,
                                         const bool exact_insertion,
                                         const SubsetIndex* spatial_index,
                                         const int spatial_neighbors) {
    if (add_node < 0 || add_node >= inst.N
        || tour.in_set[static_cast<std::size_t>(add_node)] != 0U) {
        return {};
    }
    if (exact_insertion) {
        return find_best_insert_after_remove(inst, tour, remove_pos, add_node);
    }
    if (spatial_index != nullptr) {
        return find_best_insert_after_remove_spatial(
            inst, tour, *spatial_index, remove_pos, add_node,
            spatial_neighbors, std::max(1, options.sa_insertion_window));
    }
    return find_best_insert_after_remove_windowed(
        inst, tour, remove_pos, add_node,
        std::max(1, options.sa_insertion_window));
}

SaTemperatureSchedule fixed_schedule(const SolverOptions& options,
                                     const bool elite_seed) {
    SaTemperatureSchedule schedule;
    const double base_t0 = options.sa_t0 > 0.0 ? options.sa_t0 : 1.4;
    schedule.t0 = elite_seed
        ? std::min(base_t0, options.kick_t0 > 0.0 ? options.kick_t0 : 0.35)
        : base_t0;
    schedule.t1 = options.sa_t1 > 0.0 ? options.sa_t1 : 0.00005;
    if (!(schedule.t1 < schedule.t0)) {
        schedule.t1 = std::max(1e-12, schedule.t0 * 1e-3);
    }
    return schedule;
}

} // namespace

SaProposal propose_sa_move(const Instance& inst,
                           const Tour& tour,
                           Rng& rng,
                           const SolverOptions& options,
                           const bool exact_insertion,
                           const SubsetIndex* spatial_index,
                           const int spatial_neighbors,
                           const bool measure_timing) {
    SaProposal proposal;
    const int trials = std::max(1, options.sa_candidate_trials);
    thread_local std::vector<SwapInsertionMove> valid;
    valid.clear();
    valid.reserve(static_cast<std::size_t>(trials));

    for (int trial = 0; trial < trials; ++trial) {
        const Clock::time_point proposal_start =
            measure_timing ? Clock::now() : Clock::time_point{};
        const int remove_pos = rng.randint(tour.k);
        const int add_node = choose_swap_candidate(inst, tour, remove_pos, rng);
        if (measure_timing) {
            proposal.proposal_seconds +=
                std::chrono::duration<double>(Clock::now() - proposal_start).count();
        }
        if (add_node < 0 || add_node >= inst.N
            || tour.in_set[static_cast<std::size_t>(add_node)] != 0U) {
            continue;
        }
        ++proposal.candidate_evaluations;
        const Clock::time_point insertion_start =
            measure_timing ? Clock::now() : Clock::time_point{};
        SwapInsertionMove move = evaluate_sa_candidate(
            inst, tour, remove_pos, add_node, options, exact_insertion,
            spatial_index, spatial_neighbors);
        if (measure_timing) {
            proposal.insertion_seconds +=
                std::chrono::duration<double>(Clock::now() - insertion_start).count();
        }
        if (move.valid) {
            valid.push_back(move);
        }
    }
    if (valid.empty()) {
        return proposal;
    }
    if (valid.size() == 1U || trials == 1) {
        proposal.move = valid.front();
        return proposal;
    }

    const double random_probability = std::clamp(
        options.sa_multiple_try_random_probability, 0.0, 1.0);
    if (random_probability > 0.0 && rng.uniform() < random_probability) {
        proposal.move = valid[static_cast<std::size_t>(
            rng.randint(static_cast<int>(valid.size())))];
        return proposal;
    }

    const auto best = std::min_element(
        valid.begin(), valid.end(), [](const SwapInsertionMove& lhs,
                                      const SwapInsertionMove& rhs) {
            return lhs.delta < rhs.delta;
        });
    proposal.move = *best;
    return proposal;
}

SaTemperatureSchedule resolve_sa_temperature_schedule(
    const Instance& inst,
    const Tour& tour,
    Rng calibration_rng,
    const SolverOptions& options,
    const bool elite_seed,
    const bool exact_insertion,
    const SubsetIndex* spatial_index,
    const int spatial_neighbors,
    const int sa_iterations) {
    SaTemperatureSchedule schedule = fixed_schedule(options, elite_seed);
    if (!options.sa_auto_temperature || sa_iterations <= 0) {
        return schedule;
    }

    const int target = std::max(1, options.sa_temperature_samples);
    const int minimum = std::min(target, std::max(8, target / 4));
    const int max_attempts = std::max(64, target * 16);
    std::vector<double> uphill;
    uphill.reserve(static_cast<std::size_t>(target));

    for (int attempt = 0;
         attempt < max_attempts && static_cast<int>(uphill.size()) < target;
         ++attempt) {
        ++schedule.attempts;
        SaProposal proposal = propose_sa_move(
            inst, tour, calibration_rng, options, exact_insertion,
            spatial_index, spatial_neighbors);
        if (!proposal.move.valid || !(proposal.move.delta > kImprovementEps)
            || !std::isfinite(proposal.move.delta)) {
            continue;
        }
        uphill.push_back(proposal.move.delta);
    }
    schedule.uphill_samples = static_cast<std::uint64_t>(uphill.size());
    if (static_cast<int>(uphill.size()) < minimum) {
        return schedule;
    }

    const double quantile = std::clamp(options.sa_temperature_quantile, 0.0, 1.0);
    const std::size_t index = static_cast<std::size_t>(std::floor(
        quantile * static_cast<double>(uphill.size() - 1U)));
    std::nth_element(uphill.begin(), uphill.begin() + static_cast<std::ptrdiff_t>(index),
                     uphill.end());
    const double representative = uphill[index];
    const double initial_acceptance = options.sa_initial_uphill_acceptance;
    const double final_acceptance = options.sa_final_uphill_acceptance;
    const double calibrated_t0 = representative / -std::log(initial_acceptance);
    const double calibrated_t1 = representative / -std::log(final_acceptance);
    if (!std::isfinite(calibrated_t0) || !std::isfinite(calibrated_t1)
        || !(calibrated_t0 > calibrated_t1) || !(calibrated_t1 > 0.0)) {
        return schedule;
    }

    schedule.t0 = calibrated_t0;
    schedule.t1 = calibrated_t1;
    if (elite_seed) {
        const double kick_cap = options.kick_t0 > 0.0 ? options.kick_t0 : 0.35;
        schedule.t0 = std::min(schedule.t0, kick_cap);
        schedule.t1 = std::min(schedule.t1, schedule.t0 * 0.5);
    }
    if (!(schedule.t0 > schedule.t1) || !(schedule.t1 > 0.0)) {
        return fixed_schedule(options, elite_seed);
    }
    schedule.calibrated = true;
    return schedule;
}

} // namespace aldous_tsp
