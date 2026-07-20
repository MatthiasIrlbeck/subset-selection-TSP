#include "solver_internal.hpp"
#include "aldous_tsp/validation.hpp"
#include "worker.hpp"

#include "aldous_tsp/exact_subset.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace aldous_tsp {
namespace {

std::uint64_t undirected_edge_key(const int a, const int b) noexcept {
    const auto lo = static_cast<std::uint32_t>(std::min(a, b));
    const auto hi = static_cast<std::uint32_t>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32U)
         | static_cast<std::uint64_t>(hi);
}

std::vector<std::uint64_t> canonical_cycle_edges(const std::vector<int>& cycle) {
    std::vector<std::uint64_t> edges;
    if (cycle.size() < 2U) {
        return edges;
    }
    edges.reserve(cycle.size());
    for (std::size_t i = 0; i < cycle.size(); ++i) {
        const std::size_t next = (i + 1U == cycle.size()) ? 0U : i + 1U;
        edges.push_back(undirected_edge_key(cycle[i], cycle[next]));
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
}

double edge_jaccard_distance(const std::vector<std::uint64_t>& lhs,
                             const std::vector<std::uint64_t>& rhs) noexcept {
    std::size_t i = 0U;
    std::size_t j = 0U;
    std::size_t intersection = 0U;
    while (i < lhs.size() && j < rhs.size()) {
        if (lhs[i] < rhs[j]) {
            ++i;
        } else if (rhs[j] < lhs[i]) {
            ++j;
        } else {
            ++intersection;
            ++i;
            ++j;
        }
    }
    const std::size_t set_union = lhs.size() + rhs.size() - intersection;
    return set_union == 0U
        ? 0.0
        : 1.0 - static_cast<double>(intersection)
                    / static_cast<double>(set_union);
}

struct TspCandidate {
    int variant = 0;
    RestartKind kind = RestartKind::TspNearestNeighbor;
    std::vector<int> nodes;
    std::vector<std::uint64_t> edges;
    double length = std::numeric_limits<double>::infinity();
    SearchStats stats;
};

struct TspOutcome {
    int variant = 0;
    RestartKind kind = RestartKind::TspNearestNeighbor;
    std::vector<int> nodes;
    RestartRecord record;
    SearchStats stats;
};

std::vector<int> select_tsp_candidates(const std::vector<TspCandidate>& candidates,
                                       const int target,
                                       const double min_edge_jaccard) {
    std::vector<int> ranked(candidates.size());
    std::iota(ranked.begin(), ranked.end(), 0);
    std::stable_sort(ranked.begin(), ranked.end(), [&](const int lhs, const int rhs) {
        const TspCandidate& a = candidates[static_cast<std::size_t>(lhs)];
        const TspCandidate& b = candidates[static_cast<std::size_t>(rhs)];
        if (a.length != b.length) { return a.length < b.length; }
        if (a.kind != b.kind) {
            return restart_kind_code(a.kind) < restart_kind_code(b.kind);
        }
        return a.variant < b.variant;
    });

    std::vector<int> selected;
    selected.reserve(static_cast<std::size_t>(target));
    for (const int candidate_index : ranked) {
        bool diverse = true;
        for (const int retained_index : selected) {
            if (edge_jaccard_distance(
                    candidates[static_cast<std::size_t>(candidate_index)].edges,
                    candidates[static_cast<std::size_t>(retained_index)].edges)
                + kDistanceEps < min_edge_jaccard) {
                diverse = false;
                break;
            }
        }
        if (diverse) {
            selected.push_back(candidate_index);
            if (static_cast<int>(selected.size()) == target) {
                break;
            }
        }
    }
    // Diversity is a preference, not a reason to leave a full ILS slot idle.
    for (const int candidate_index : ranked) {
        if (static_cast<int>(selected.size()) == target) { break; }
        if (std::find(selected.begin(), selected.end(), candidate_index)
            == selected.end()) {
            selected.push_back(candidate_index);
        }
    }
    // Merge and serialize in stable seed-variant order, independent of ranking.
    std::sort(selected.begin(), selected.end(), [&](const int lhs, const int rhs) {
        return candidates[static_cast<std::size_t>(lhs)].variant
             < candidates[static_cast<std::size_t>(rhs)].variant;
    });
    return selected;
}

} // namespace

SolveResult solve_tsp(const Instance& inst, Rng& rng, const SolverOptions& options) {
    require_valid_tsp_request(inst, options);
    const auto start = Clock::now();
    SolveResult result;
    result.tour.init(inst.N);
    if (options.exact_subset_max_n < 0
        || options.exact_subset_max_n > kExactSubsetHardLimit) {
        throw std::invalid_argument(
            "exact_subset_max_n must be in [0,kExactSubsetHardLimit]");
    }
    if (options.tsp_candidate_starts < 1) {
        throw std::invalid_argument("tsp_candidate_starts must be >= 1");
    }
    if (options.tsp_farthest_starts < 0 || options.tsp_farthest_starts > 1) {
        throw std::invalid_argument("tsp_farthest_starts must be 0 or 1");
    }
    if (!std::isfinite(options.tsp_min_edge_jaccard)
        || options.tsp_min_edge_jaccard < 0.0
        || options.tsp_min_edge_jaccard > 1.0) {
        throw std::invalid_argument(
            "tsp_min_edge_jaccard must be finite and in [0,1]");
    }
    if (options.exact_subset_max_n > 0
        && inst.N <= options.exact_subset_max_n) {
        ExactSubsetSolution exact;
        {
            ScopedPhaseTimer phase_timer(result.stats.phases.exact_subset_seconds);
            ++result.stats.exact_subset_calls;
            exact = exact_subset_cycle(inst, inst.N);
        }
        if (!exact.solved || !exact.proven_optimal) {
            throw std::logic_error(
                "exact subset oracle did not solve an enabled full-TSP instance");
        }
        ++result.stats.exact_subset_solved;
        result.stats.exact_subset_states += exact.states;
        result.stats.exact_subset_transitions += exact.transitions;
        result.tour.set_tour(exact.cycle, inst);
        result.exact_optimal = true;
        result.stats.tsp_seconds =
            std::chrono::duration<double>(Clock::now() - start).count();
        return result;
    }

    const int promoted_target = std::max(1, options.tsp_restarts);
    const int candidate_count = std::max(promoted_target,
                                         options.tsp_candidate_starts);
    const int restart_threads = std::max(1, options.restart_threads);
    const double time_budget = options.time_budget_per_p;
    const std::uint64_t solve_stream_base = rng.next_u64();
    const std::vector<int> all = all_nodes(inst.N);
    ElitePool elite(std::max(4, std::min(24, promoted_target + 8)),
                    EliteMode::Cycle);

    auto run_candidate = [&](const int variant) {
        TspCandidate candidate;
        candidate.variant = variant;
        candidate.kind = variant < options.tsp_farthest_starts
            ? RestartKind::TspFarthestInsertion
            : RestartKind::TspNearestNeighbor;
        Rng candidate_rng(make_stream_seed(
            solve_stream_base,
            static_cast<std::uint64_t>(variant),
            0x6a09e667f3bcc909ULL));
        {
            ScopedPhaseTimer phase_timer(
                candidate.stats.phases.tsp_construction_seconds);
            if (candidate.kind == RestartKind::TspFarthestInsertion) {
                candidate.nodes = farthest_insertion_order(inst, all);
            } else {
                const int start_index = variant == 0 ? 0 : candidate_rng.randint(inst.N);
                candidate.nodes = nearest_neighbor_full_order(inst, start_index);
            }
        }
        Tour tour;
        tour.init(inst.N);
        tour.set_tour(candidate.nodes, inst);
        {
            ScopedPhaseTimer phase_timer(
                candidate.stats.phases.initial_polish_seconds);
            polish_tour(tour, inst, options, &candidate.stats, 2);
        }
        candidate.nodes = tour.nodes;
        candidate.length = tour.length;
        candidate.edges = canonical_cycle_edges(candidate.nodes);
        candidate.stats.tsp_candidate_starts = 1;
        return candidate;
    };

    std::vector<TspCandidate> candidates(static_cast<std::size_t>(candidate_count));
    for (int begin = 0; begin < candidate_count; begin += restart_threads) {
        const int wave = std::min(restart_threads, candidate_count - begin);
        detail::run_parallel_indexed(wave, [&](const int offset) {
            const int variant = begin + offset;
            candidates[static_cast<std::size_t>(variant)] = run_candidate(variant);
        });
    }
    for (const TspCandidate& candidate : candidates) {
        result.stats.add(candidate.stats);
    }

    const std::vector<int> promoted = select_tsp_candidates(
        candidates,
        promoted_target,
        options.tsp_min_edge_jaccard);

    auto run_full = [&](const TspCandidate& candidate) {
        TspOutcome outcome;
        outcome.variant = candidate.variant;
        outcome.kind = candidate.kind;
        Rng restart_rng(make_stream_seed(
            solve_stream_base,
            static_cast<std::uint64_t>(candidate.variant),
            0xbb67ae8584caa73bULL));
        Tour best_restart;
        best_restart.init(inst.N);
        best_restart.set_tour(candidate.nodes, inst);
        // Preserve the pilot's exact incrementally-polished value.
        best_restart.length = candidate.length;
        int no_improve = 0;
        const int ils = std::max(0, options.tsp_ils);
        {
            ScopedPhaseTimer phase_timer(outcome.stats.phases.tsp_ils_seconds);
            for (int it = 0; it < ils; ++it) {
                Tour cand = best_restart;
                perturb_three_cut(cand, restart_rng);
                cand.recompute_length(inst);
                polish_tour(cand, inst, options, &outcome.stats, 1);
                ++outcome.stats.tsp_ils_iterations;
                if (cand.length + kImprovementEps < best_restart.length) {
                    best_restart = std::move(cand);
                    no_improve = 0;
                } else if (++no_improve > std::max(0, options.tsp_patience)) {
                    break;
                }
            }
        }
        if (options.oracle.cfg.inline_feedback) {
            ScopedPhaseTimer phase_timer(outcome.stats.phases.oracle_seconds);
            (void)external_oracle_polish_tour(
                best_restart, inst, options.oracle, true, &outcome.stats,
                !options.disable_two_opt);
        }
        outcome.nodes = best_restart.nodes;
        outcome.record = make_restart_record(
            inst, outcome.nodes, best_restart.length, candidate.kind);
        outcome.record.role = candidate_count > promoted_target
            ? RestartRole::RacedProduction
            : RestartRole::IndependentDiagnostic;
        outcome.record.seed_variant = candidate.variant;
        outcome.record.promotion_stage = candidate_count > promoted_target
            ? RestartPromotionStage::PromotedFull
            : RestartPromotionStage::None;
        outcome.record.strong_polished = true;
        outcome.stats.tsp_promoted_restarts = 1;
        outcome.stats.tsp_restarts = 1;
        return outcome;
    };

    double best_tsp_len = std::numeric_limits<double>::infinity();
    auto merge_outcome = [&](TspOutcome& outcome) {
        result.stats.add(outcome.stats);
        elite.try_add(outcome.nodes, outcome.record.length);
        result.restarts.push_back(outcome.record);
        if (outcome.record.length < best_tsp_len - kImprovementEps) {
            best_tsp_len = outcome.record.length;
            result.best_restart = static_cast<int>(result.restarts.size()) - 1;
        }
    };

    for (int begin = 0; begin < static_cast<int>(promoted.size());
         begin += restart_threads) {
        const int wave = std::min(
            restart_threads, static_cast<int>(promoted.size()) - begin);
        std::vector<TspOutcome> outcomes(static_cast<std::size_t>(wave));
        detail::run_parallel_indexed(wave, [&](const int offset) {
            const int candidate_index = promoted[static_cast<std::size_t>(begin + offset)];
            outcomes[static_cast<std::size_t>(offset)] = run_full(
                candidates[static_cast<std::size_t>(candidate_index)]);
        });
        for (TspOutcome& outcome : outcomes) {
            merge_outcome(outcome);
        }
    }

    // Elapsed-time mode remains intentionally machine-dependent. Additional
    // restarts are independent nearest-neighbor starts run in bounded waves.
    int anytime_variant = candidate_count;
    while (time_budget > 0.0
           && std::chrono::duration<double>(Clock::now() - start).count()
                  < time_budget) {
        const int wave = restart_threads;
        std::vector<TspCandidate> extra_candidates(static_cast<std::size_t>(wave));
        detail::run_parallel_indexed(wave, [&](const int offset) {
            extra_candidates[static_cast<std::size_t>(offset)] =
                run_candidate(anytime_variant + offset);
        });
        std::vector<TspOutcome> outcomes(static_cast<std::size_t>(wave));
        detail::run_parallel_indexed(wave, [&](const int offset) {
            outcomes[static_cast<std::size_t>(offset)] =
                run_full(extra_candidates[static_cast<std::size_t>(offset)]);
            outcomes[static_cast<std::size_t>(offset)].record.role =
                RestartRole::Anytime;
            outcomes[static_cast<std::size_t>(offset)].record.promotion_stage =
                RestartPromotionStage::None;
        });
        for (int offset = 0; offset < wave; ++offset) {
            result.stats.add(extra_candidates[static_cast<std::size_t>(offset)].stats);
            merge_outcome(outcomes[static_cast<std::size_t>(offset)]);
        }
        anytime_variant += wave;
    }

    {
        ScopedPhaseTimer phase_timer(result.stats.phases.oracle_seconds);
        polish_elite_with_oracle(elite, inst, options, true,
                                 options.oracle.cfg.tsp_top, &result.stats);
    }
    const auto nodes = elite.export_nodes();
    if (!nodes.empty()) {
        result.tour.set_tour(nodes.front(), inst);
        ScopedPhaseTimer phase_timer(result.stats.phases.final_polish_seconds);
        final_polish_tour(result.tour, inst, options, &result.stats, 2);
    }
    result.stats.tsp_seconds =
        std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

} // namespace aldous_tsp
