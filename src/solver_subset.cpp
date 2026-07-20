#include "solver_internal.hpp"
#include "aldous_tsp/validation.hpp"
#include "worker.hpp"

#include "aldous_tsp/exact_subset.hpp"

#include <algorithm>
#include <map>
#include <numeric>

namespace aldous_tsp {
namespace {

struct SeedCandidate {
    std::vector<int> nodes;
    RestartKind kind = RestartKind::Random;
    RestartRole role = RestartRole::IndependentDiagnostic;
    int variant = 0;
    double length = std::numeric_limits<double>::infinity();
};

std::uint64_t seed_stream(const std::uint64_t solve_stream_base,
                          const RestartRole role,
                          const RestartKind kind,
                          const int variant,
                          const std::uint64_t purpose) {
    const std::uint64_t role_kind =
        (static_cast<std::uint64_t>(restart_role_code(role)) << 32U)
        ^ static_cast<std::uint64_t>(restart_kind_code(kind));
    return make_stream_seed(
        solve_stream_base,
        mix_hash64(role_kind ^ purpose),
        mix_hash64(static_cast<std::uint64_t>(variant) ^ 0x9e3779b97f4a7c15ULL));
}

// Fixed kind quotas are more stable than globally ranking raw seed-cycle
// lengths: kind selects materially different search behavior. Round-robin by
// kind, best-first within each kind, and repeat deterministically only when a
// caller explicitly requests more restarts than there are distinct candidates.
std::vector<SeedCandidate> select_seed_candidates(std::vector<SeedCandidate> candidates,
                                                  const int target) {
    if (target <= 0 || candidates.empty()) {
        return {};
    }
    std::map<RestartKind, std::vector<SeedCandidate>> by_kind;
    for (SeedCandidate& candidate : candidates) {
        by_kind[candidate.kind].push_back(std::move(candidate));
    }
    std::vector<std::vector<SeedCandidate>*> kinds;
    kinds.reserve(by_kind.size());
    for (auto& entry : by_kind) {
        std::stable_sort(entry.second.begin(), entry.second.end(),
                         [](const SeedCandidate& lhs, const SeedCandidate& rhs) {
            if (lhs.length != rhs.length) { return lhs.length < rhs.length; }
            if (lhs.variant != rhs.variant) { return lhs.variant < rhs.variant; }
            return lhs.nodes < rhs.nodes;
        });
        kinds.push_back(&entry.second);
    }
    std::stable_sort(kinds.begin(), kinds.end(), [](const auto* lhs, const auto* rhs) {
        const SeedCandidate& a = lhs->front();
        const SeedCandidate& b = rhs->front();
        if (a.length != b.length) { return a.length < b.length; }
        return restart_kind_code(a.kind) < restart_kind_code(b.kind);
    });

    std::vector<SeedCandidate> selected;
    selected.reserve(static_cast<std::size_t>(target));
    for (std::size_t round = 0U; static_cast<int>(selected.size()) < target; ++round) {
        bool progressed = false;
        for (std::vector<SeedCandidate>* group : kinds) {
            if (group->empty()) { continue; }
            const SeedCandidate& source = (*group)[round % group->size()];
            selected.push_back(source);
            progressed = true;
            if (static_cast<int>(selected.size()) == target) { break; }
        }
        if (!progressed) { break; }
    }
    return selected;
}

std::vector<SeedCandidate> select_continuation_candidates(
    std::vector<SeedCandidate> candidates,
    const int target) {
    if (target <= 0 || candidates.empty()) {
        return {};
    }
    std::map<RestartKind, std::vector<SeedCandidate>> by_kind;
    for (SeedCandidate& candidate : candidates) {
        by_kind[candidate.kind].push_back(std::move(candidate));
    }
    for (auto& entry : by_kind) {
        std::stable_sort(entry.second.begin(), entry.second.end(),
                         [](const SeedCandidate& lhs, const SeedCandidate& rhs) {
            if (lhs.length != rhs.length) { return lhs.length < rhs.length; }
            if (lhs.variant != rhs.variant) { return lhs.variant < rhs.variant; }
            return lhs.nodes < rhs.nodes;
        });
    }

    // At high p the deletion trajectory is the specialized continuation
    // operator; keep it before the generic resized warm seed. Round-robin then
    // guarantees both kinds when the caller reserves at least two draws.
    constexpr RestartKind priority[] = {
        RestartKind::HighPDelete,
        RestartKind::Warm,
    };
    std::vector<SeedCandidate> selected;
    selected.reserve(static_cast<std::size_t>(target));
    for (std::size_t round = 0U; static_cast<int>(selected.size()) < target; ++round) {
        bool progressed = false;
        for (const RestartKind kind : priority) {
            auto found = by_kind.find(kind);
            if (found == by_kind.end() || found->second.empty()) { continue; }
            selected.push_back(found->second[round % found->second.size()]);
            progressed = true;
            if (static_cast<int>(selected.size()) == target) { break; }
        }
        if (!progressed) { break; }
    }
    return selected;
}

int sorted_set_removed_count(const std::vector<int>& lhs,
                             const std::vector<int>& rhs) noexcept {
    std::size_t li = 0U;
    std::size_t ri = 0U;
    int intersection = 0;
    while (li < lhs.size() && ri < rhs.size()) {
        if (lhs[li] < rhs[ri]) {
            ++li;
        } else if (rhs[ri] < lhs[li]) {
            ++ri;
        } else {
            ++intersection;
            ++li;
            ++ri;
        }
    }
    return static_cast<int>(lhs.size()) - intersection;
}

std::vector<std::uint64_t> canonical_undirected_edges(
    const std::vector<int>& cycle) {
    std::vector<std::uint64_t> edges;
    edges.reserve(cycle.size());
    for (std::size_t i = 0; i < cycle.size(); ++i) {
        const std::uint32_t a = static_cast<std::uint32_t>(
            std::min(cycle[i], cycle[(i + 1U) % cycle.size()]));
        const std::uint32_t b = static_cast<std::uint32_t>(
            std::max(cycle[i], cycle[(i + 1U) % cycle.size()]));
        edges.push_back((static_cast<std::uint64_t>(a) << 32U)
                        | static_cast<std::uint64_t>(b));
    }
    std::sort(edges.begin(), edges.end());
    return edges;
}

double cycle_edge_jaccard_distance(const std::vector<int>& lhs,
                                   const std::vector<int>& rhs) {
    const std::vector<std::uint64_t> left = canonical_undirected_edges(lhs);
    const std::vector<std::uint64_t> right = canonical_undirected_edges(rhs);
    std::size_t li = 0U;
    std::size_t ri = 0U;
    std::size_t intersection = 0U;
    while (li < left.size() && ri < right.size()) {
        if (left[li] < right[ri]) {
            ++li;
        } else if (right[ri] < left[li]) {
            ++ri;
        } else {
            ++intersection;
            ++li;
            ++ri;
        }
    }
    const std::size_t edge_union = left.size() + right.size() - intersection;
    return edge_union == 0U
        ? 0.0
        : 1.0 - static_cast<double>(intersection)
                    / static_cast<double>(edge_union);
}

std::uint64_t estimated_bidirectional_relink_scans(const int removed) noexcept {
    if (removed <= 0) { return 0U; }
    // Two directions, each evaluating r^2 candidate swaps for r=d..1:
    // 2 * sum(r^2) = d(d+1)(2d+1)/3. Divide before multiplying and saturate.
    std::uint64_t a = static_cast<std::uint64_t>(removed);
    std::uint64_t b = a + 1U;
    std::uint64_t c = 2U * a + 1U;
    if (a % 3U == 0U) {
        a /= 3U;
    } else if (b % 3U == 0U) {
        b /= 3U;
    } else {
        c /= 3U;
    }
    constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
    if (a != 0U && b > max / a) { return max; }
    const std::uint64_t ab = a * b;
    if (ab != 0U && c > max / ab) { return max; }
    return ab * c;
}

void record_subset_restart_kind(SearchStats& stats, RestartKind kind) noexcept {
    switch (kind) {
        case RestartKind::Random: ++stats.random_restarts; break;
        case RestartKind::Warm: ++stats.warm_restarts; break;
        case RestartKind::SmallP: ++stats.smallp_seed_restarts; break;
        case RestartKind::HighPDelete: ++stats.highp_delete_restarts; break;
        case RestartKind::Elite: ++stats.elite_restarts; break;
        case RestartKind::Kick:
            ++stats.elite_restarts;
            ++stats.kick_restarts;
            break;
        case RestartKind::Region: ++stats.region_restarts; break;
        case RestartKind::Dense: ++stats.dense_restarts; break;
        case RestartKind::TspFarthestInsertion:
        case RestartKind::TspNearestNeighbor:
            break;
    }
}

} // namespace

SolveResult solve_subset(const Instance& inst,
                         int k,
                         Rng& rng,
                         const SolverOptions& options,
                         const std::vector<int>* warm_start,
                         const SubsetSolveRequest& request) {
    require_valid_subset_request(inst, k, options, warm_start);
    const auto start = Clock::now();
    SolveResult result;
    result.tour.init(inst.N);
    if (options.exact_subset_max_n < 0
        || options.exact_subset_max_n > kExactSubsetHardLimit) {
        throw std::invalid_argument(
            "exact_subset_max_n must be in [0,kExactSubsetHardLimit]");
    }
    if (k == inst.N) { return solve_tsp(inst, rng, options); }

    const double p = static_cast<double>(k) / static_cast<double>(std::max(1, inst.N));
    if (options.racing_candidates < 0) {
        throw std::invalid_argument("racing_candidates must be >= 0");
    }
    if (options.racing_survivors < 1) {
        throw std::invalid_argument("racing_survivors must be >= 1");
    }
    if (options.racing_candidates > 0
        && options.racing_survivors > options.racing_candidates) {
        throw std::invalid_argument("racing_survivors must not exceed racing_candidates");
    }
    if (options.racing_pilot_iters < 0) {
        throw std::invalid_argument("racing_pilot_iters must be >= 0");
    }
    if (!std::isfinite(options.racing_min_jaccard)
        || options.racing_min_jaccard < 0.0
        || options.racing_min_jaccard > 1.0) {
        throw std::invalid_argument("racing_min_jaccard must be finite and in [0,1]");
    }
    if (options.strong_polish_finalists < 1) {
        throw std::invalid_argument("strong_polish_finalists must be >= 1");
    }
    if (!std::isfinite(options.strong_polish_min_jaccard)
        || options.strong_polish_min_jaccard < 0.0
        || options.strong_polish_min_jaccard > 1.0) {
        throw std::invalid_argument(
            "strong_polish_min_jaccard must be finite and in [0,1]");
    }
    if (options.racing_candidates > 0 && options.time_budget_per_p > 0.0) {
        throw std::invalid_argument(
            "deterministic restart racing is incompatible with time_budget_per_p");
    }
    if (options.elite_diversity_slots < 0) {
        throw std::invalid_argument("elite_diversity_slots must be >= 0");
    }
    if (!std::isfinite(options.elite_min_jaccard)
        || options.elite_min_jaccard < 0.0
        || options.elite_min_jaccard > 1.0) {
        throw std::invalid_argument("elite_min_jaccard must be finite and in [0,1]");
    }
    if (!std::isfinite(options.elite_quality_slack)
        || options.elite_quality_slack < 0.0) {
        throw std::invalid_argument("elite_quality_slack must be finite and >= 0");
    }
    if (options.path_relink_top < 0
        || options.path_relink_diverse_reserve < 0
        || options.path_relink_max_pairs < 0
        || options.path_relink_max_removed < 0
        || options.path_relink_max_removed_sum < 0
        || options.path_relink_max_candidate_scans < 0) {
        throw std::invalid_argument("path-relink limits must be >= 0");
    }
    if (options.ejection_chain_starts < 0
        || options.ejection_chain_depth < 0
        || options.ejection_chain_candidates < 0
        || options.ejection_chain_remove_cap < 0) {
        throw std::invalid_argument("ejection-chain integer options must be >= 0");
    }
    if (!std::isfinite(options.ejection_chain_max_uphill)
        || options.ejection_chain_max_uphill < 0.0) {
        throw std::invalid_argument(
            "ejection_chain_max_uphill must be finite and >= 0");
    }
    // Explicit values configure the base population. AUTO reproduces the
    // historical effective count. Supplemental continuation is deliberately
    // outside this population so adding neighboring p-values cannot remove an
    // independent draw or worsen the best-of-restarts result.
    const int resolved_restarts = options.subset_restarts >= 1
        ? options.subset_restarts
        : (options.staged_search
               ? (p <= 0.08 ? 12 : 5)
               : (p <= 0.08 ? 8 : 3));
    const bool has_warm = warm_start != nullptr && !warm_start->empty();
    if (request.continuation_only && !has_warm) {
        throw std::invalid_argument("continuation-only subset solve requires a warm start");
    }

    if (options.exact_subset_max_n > 0
        && inst.N <= options.exact_subset_max_n) {
        ExactSubsetSolution exact;
        {
            ScopedPhaseTimer phase_timer(result.stats.phases.exact_subset_seconds);
            ++result.stats.exact_subset_calls;
            exact = exact_subset_cycle(inst, k);
        }
        if (!exact.solved || !exact.proven_optimal) {
            throw std::logic_error(
                "exact subset oracle did not solve an enabled subset instance");
        }
        ++result.stats.exact_subset_solved;
        result.stats.exact_subset_states += exact.states;
        result.stats.exact_subset_transitions += exact.transitions;
        result.tour.set_tour(exact.cycle, inst);
        result.exact_optimal = true;
        result.stats.subset_seconds =
            std::chrono::duration<double>(Clock::now() - start).count();
        return result;
    }

    // Consume exactly one caller draw. Every seed family, variant, restart, and
    // post-processing stream is derived from this stable base, so constructing
    // warm candidates cannot perturb independent candidates or their anneals.
    const std::uint64_t solve_stream_base = rng.next_u64();

    const int kick_n = (request.continuation_only || options.disable_elite_restarts)
        ? 0
        : std::max(0, std::min(options.subset_kick_restarts, resolved_restarts - 1));
    int continuation_n = 0;
    int indep_restarts = 0;
    if (request.continuation_only) {
        continuation_n = options.continuation_restarts;
    } else if (has_warm && options.continuation_restarts > 0
               && options.continuation_policy == ContinuationPolicy::FixedBudget) {
        const int non_kick_budget = resolved_restarts - kick_n;
        continuation_n = std::min(options.continuation_restarts,
                                  std::max(0, non_kick_budget - 1));
        indep_restarts = non_kick_budget - continuation_n;
    } else {
        indep_restarts = resolved_restarts - kick_n;
        if (has_warm && options.continuation_restarts > 0) {
            continuation_n = options.continuation_restarts;
        }
    }
    const int kick_begin = indep_restarts;
    const int continuation_begin = kick_begin + kick_n;
    const int total_restarts = continuation_begin + continuation_n;
    const int racing_n = request.continuation_only ? 0 : options.racing_candidates;
    if (total_restarts <= 0) {
        result.stats.subset_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        return result;
    }

    const auto seed_pool_start = Clock::now();
    auto make_candidate = [&](std::vector<int> nodes,
                              const RestartKind kind,
                              const RestartRole role,
                              const int variant) -> SeedCandidate {
        SeedCandidate candidate;
        if (static_cast<int>(nodes.size()) != k) {
            return candidate;
        }
        Tour tour;
        tour.init(inst.N);
        tour.set_tour(nodes, inst);
        candidate.nodes = std::move(nodes);
        candidate.kind = kind;
        candidate.role = role;
        candidate.variant = variant;
        candidate.length = tour.length;
        return candidate;
    };
    auto append_valid = [&](std::vector<SeedCandidate>& destination,
                            SeedCandidate candidate) {
        if (static_cast<int>(candidate.nodes.size()) == k
            && std::isfinite(candidate.length)) {
            destination.push_back(std::move(candidate));
        }
    };

    std::vector<SeedCandidate> independent_candidates;
    if (indep_restarts > 0) {
        if (!options.disable_smallp_seeds
            && (options.mode == SolverMode::SmallPRegion
                || options.mode == SolverMode::Hybrid || p <= 0.08)) {
            Rng smallp_rng(seed_stream(solve_stream_base,
                                       RestartRole::IndependentDiagnostic,
                                       RestartKind::SmallP,
                                       0,
                                       0x2d98c47d86e6f2adULL));
            auto seeds = make_smallp_seed_pool(inst, k, smallp_rng, 8);
            for (std::size_t i = 0; i < seeds.size(); ++i) {
                append_valid(independent_candidates,
                             make_candidate(std::move(seeds[i]),
                                            RestartKind::SmallP,
                                            RestartRole::IndependentDiagnostic,
                                            static_cast<int>(i)));
            }
        }

        const bool dense_fill = options.small_p_dense_fill && p <= 0.08;
        int fill_variant = 0;
        while (static_cast<int>(independent_candidates.size()) < indep_restarts) {
            const bool use_dense = dense_fill || ((fill_variant & 1) == 0);
            const RestartKind kind = use_dense ? RestartKind::Dense : RestartKind::Random;
            Rng seed_rng_local(seed_stream(solve_stream_base,
                                           RestartRole::IndependentDiagnostic,
                                           kind,
                                           fill_variant,
                                           0x7137449123ef65cdULL));
            std::vector<int> seed;
            if (use_dense) {
                seed = dense_seed(inst, k, seed_rng_local, fill_variant);
            } else {
                seed = random_subset(inst.N, k, seed_rng_local);
                seed = (k > 36)
                    ? nearest_neighbor_order(inst, seed, seed_rng_local.randint(k))
                    : farthest_insertion_order(inst, seed);
            }
            append_valid(independent_candidates,
                         make_candidate(std::move(seed), kind,
                                        RestartRole::IndependentDiagnostic,
                                        fill_variant));
            ++fill_variant;
        }
    }
    std::vector<SeedCandidate> independent_seeds =
        select_seed_candidates(std::move(independent_candidates), indep_restarts);

    std::vector<SeedCandidate> continuation_candidates;
    if (continuation_n > 0 && has_warm) {
        // Build two deterministic warm constructors even for one executed
        // continuation restart, then choose the better seed within that kind.
        // Seed construction is not an executed draw, and the stateful resize
        // kernels make this diversity inexpensive while preserving historical
        // continuation quality and deterministic variant metadata.
        const int warm_variants = std::max(2, continuation_n);
        for (int variant = 0; variant < warm_variants; ++variant) {
            Rng warm_rng(seed_stream(solve_stream_base,
                                     RestartRole::Continuation,
                                     RestartKind::Warm,
                                     variant,
                                     0xb5c0fbcfec4d3b2fULL));
            append_valid(continuation_candidates,
                         make_candidate(resize_seed(inst, *warm_start, k, warm_rng,
                                                    variant & 1),
                                        RestartKind::Warm,
                                        RestartRole::Continuation,
                                        variant));
        }
        if (!options.disable_highp_delete && p >= 0.50
            && (options.mode == SolverMode::HighPDelete
                || options.mode == SolverMode::Hybrid
                || options.mode == SolverMode::Balanced)
            && static_cast<int>(warm_start->size()) > k) {
            for (int variant = 0; variant < 2; ++variant) {
                Rng delete_rng(seed_stream(solve_stream_base,
                                           RestartRole::Continuation,
                                           RestartKind::HighPDelete,
                                           variant,
                                           0x8f1bbcdcb7a56463ULL));
                append_valid(continuation_candidates,
                             make_candidate(highp_delete_seed(inst, *warm_start, k,
                                                              delete_rng, variant),
                                            RestartKind::HighPDelete,
                                            RestartRole::Continuation,
                                            variant));
            }
            Rng segment_rng(seed_stream(solve_stream_base,
                                        RestartRole::Continuation,
                                        RestartKind::HighPDelete,
                                        2,
                                        0x8f1bbcdcb7a56463ULL));
            append_valid(continuation_candidates,
                         make_candidate(segment_delete_seed(inst, *warm_start, k,
                                                            segment_rng),
                                        RestartKind::HighPDelete,
                                        RestartRole::Continuation,
                                        2));
        }
    }
    std::vector<SeedCandidate> continuation_seeds =
        select_continuation_candidates(std::move(continuation_candidates), continuation_n);

    // Racing is a separate supplemental population. Its role-specific streams
    // cannot consume or perturb an independent diagnostic draw. At small p the
    // proven compact dense seeds fill the race; elsewhere dense and random
    // candidates alternate to preserve a broad basin sample.
    std::vector<SeedCandidate> racing_candidates;
    racing_candidates.reserve(static_cast<std::size_t>(racing_n));
    for (int variant = 0; variant < racing_n; ++variant) {
        const bool use_dense = (options.small_p_dense_fill && p <= 0.08)
                               || ((variant & 1) == 0);
        const RestartKind kind = use_dense ? RestartKind::Dense : RestartKind::Random;
        Rng race_seed_rng(seed_stream(solve_stream_base,
                                      RestartRole::RacedProduction,
                                      kind,
                                      variant,
                                      0x3c6ef372fe94f82bULL));
        std::vector<int> seed;
        if (use_dense) {
            seed = dense_seed(inst, k, race_seed_rng, variant);
        } else {
            seed = random_subset(inst.N, k, race_seed_rng);
            seed = (k > 36)
                ? nearest_neighbor_order(inst, seed, race_seed_rng.randint(k))
                : farthest_insertion_order(inst, seed);
        }
        append_valid(racing_candidates,
                     make_candidate(std::move(seed), kind,
                                    RestartRole::RacedProduction, variant));
    }
    std::vector<SeedCandidate> racing_seeds =
        select_seed_candidates(std::move(racing_candidates), racing_n);
    if (static_cast<int>(independent_seeds.size()) != indep_restarts
        || static_cast<int>(continuation_seeds.size()) != continuation_n
        || static_cast<int>(racing_seeds.size()) != racing_n) {
        throw std::runtime_error("failed to construct the requested restart seed population");
    }

    // Keep the scheduled-search archive capacity unchanged when racing is
    // enabled. This preserves kick and continuation behavior bit-for-bit;
    // raced outcomes are merged only after the complete scheduled population.
    ElitePool elite(std::max(8, total_restarts + 8),
                    EliteMode::Set,
                    options.elite_diversity_slots,
                    options.elite_min_jaccard,
                    options.elite_quality_slack);
    // Prime only from independent seeds before the kick snapshot. Continuation
    // seeds are deliberately excluded so scheduled kicks are invariant to the
    // presence of neighboring p-values.
    for (const SeedCandidate& candidate : independent_seeds) {
        elite.try_add(candidate.nodes, candidate.length);
    }
    if (request.continuation_only) {
        for (const SeedCandidate& candidate : continuation_seeds) {
            elite.try_add(candidate.nodes, candidate.length);
        }
    }

    result.stats.phases.seed_construction_seconds +=
        std::chrono::duration<double>(Clock::now() - seed_pool_start).count();
    const int sa_iters_eff = effective_sa_iters(options, k, inst.N);
    const double time_budget = options.time_budget_per_p;
    const double sa_t0 = options.sa_t0 > 0.0 ? options.sa_t0 : 1.4;
    const double sa_t1 = options.sa_t1 > 0.0 ? options.sa_t1 : 0.00005;
    const int restart_threads = std::max(1, options.restart_threads);
    // One base draw, then a derived stream per restart: restart results are a
    // pure function of (options, instance, restart index), so the outcome is
    // deterministic and invariant to restart_threads outside budget mode.
    const std::uint64_t restart_stream_base =
        make_stream_seed(solve_stream_base,
                         0x9b05688c2b3e6c1fULL,
                         0x452821e638d01377ULL);

    struct RestartOutcome {
        std::vector<int> nodes;
        RestartRecord record;
        Rng rng_state;
        bool elite_seed = false;
        bool strong_eligible = false;
        SearchStats stats;
    };

    // Elite snapshot for anytime (time-budget) ILS restarts. Refreshed once
    // per wave from the current elite pool; restarts beyond the scheduled count
    // seed from a perturbed elite member instead of a cold seed, turning the
    // anytime tail into subset-level iterated local search. Only active in
    // time-budget mode, which is already documented as non-reproducible, so the
    // deterministic scheduled-restart path is unaffected.
    std::vector<std::vector<int>> elite_seeds;

    auto run_restart = [&](const int restart,
                           const SeedCandidate* forced_candidate,
                           const int sa_iterations_override,
                           const RestartPromotionStage promotion_stage,
                           const bool pilot_only) -> RestartOutcome {
        RestartOutcome out;
        const std::uint64_t restart_seed = forced_candidate == nullptr
            ? make_stream_seed(restart_stream_base,
                               static_cast<std::uint64_t>(restart),
                               0x452821e638d01377ULL)
            : seed_stream(solve_stream_base,
                          forced_candidate->role,
                          forced_candidate->kind,
                          forced_candidate->variant,
                          0xbb67ae8584caa73bULL);
        Rng rrng(restart_seed);
        std::vector<int> seed;
        RestartKind kind = RestartKind::Random;
        RestartRole role = RestartRole::IndependentDiagnostic;
        int seed_variant = 0;
        const auto restart_seed_start = Clock::now();
        const bool scheduled_kick = forced_candidate == nullptr
                                    && restart >= kick_begin
                                    && restart < continuation_begin
                                    && !elite_seeds.empty();
        const bool anytime_restart = forced_candidate == nullptr
                                     && restart >= total_restarts;
        const bool elite_ils = scheduled_kick
                               || (anytime_restart && !options.disable_elite_restarts
                                   && !elite_seeds.empty()
                                   && (((restart - total_restarts) & 1) == 0));
        if (elite_ils) {
            // Pick among the top few elite members and apply a small, spatially
            // coherent ruin-and-recreate kick. Scheduled kicks are drawn only
            // from the completed independent phase; continuation seeds cannot
            // perturb their stream or elite snapshot.
            const int pool = std::min<int>(4, static_cast<int>(elite_seeds.size()));
            seed = elite_seeds[static_cast<std::size_t>(rrng.randint(pool))];
            apply_elite_kick(inst, seed, rrng, options.kick_fraction);
            kind = scheduled_kick ? RestartKind::Kick : RestartKind::Elite;
            role = scheduled_kick ? RestartRole::EliteKick : RestartRole::Anytime;
            seed_variant = scheduled_kick ? restart - kick_begin
                                          : restart - total_restarts;
            out.elite_seed = true;
        } else {
            const SeedCandidate* candidate = forced_candidate;
            if (candidate != nullptr) {
                // A race pilot and its promoted rerun use the identical seed and
                // RNG stream. The pilot therefore screens a prefix of the exact
                // full-depth restart rather than a differently randomized proxy.
            } else if (restart < kick_begin) {
                candidate = &independent_seeds[static_cast<std::size_t>(restart)];
            } else if (restart >= continuation_begin && restart < total_restarts) {
                candidate = &continuation_seeds[
                    static_cast<std::size_t>(restart - continuation_begin)];
            } else if (anytime_restart) {
                // The cold half of the anytime tail is deterministic given the
                // completed scheduled populations. Prefer independent draws;
                // a continuation-only secondary sweep falls back to its warm
                // population.
                const std::vector<SeedCandidate>& fallback =
                    independent_seeds.empty() ? continuation_seeds : independent_seeds;
                if (!fallback.empty()) {
                    candidate = &fallback[static_cast<std::size_t>(
                        (restart - total_restarts) % static_cast<int>(fallback.size()))];
                }
            }
            if (candidate == nullptr) {
                throw std::logic_error("restart schedule has no seed candidate");
            }
            seed = candidate->nodes;
            kind = candidate->kind;
            role = anytime_restart ? RestartRole::Anytime : candidate->role;
            seed_variant = anytime_restart ? restart - total_restarts
                                           : candidate->variant;
            if (options.region_seeds
                && role == RestartRole::IndependentDiagnostic
                && (kind == RestartKind::Random || kind == RestartKind::Dense)) {
                // Fresh region seed per independent restart: uniform center, k
                // sampled nodes from a dilated local neighborhood, then a
                // nearest-neighbor cycle. Its random stream is restart-local.
                const int center = rrng.randint(inst.N);
                const double dil = options.region_dilation >= 1.0
                    ? options.region_dilation : 3.0;
                const int pool_n = std::min(
                    inst.N,
                    std::max(k, static_cast<int>(std::lround(
                        dil * static_cast<double>(k)))));
                std::vector<std::pair<double, int>> by_dist;
                by_dist.reserve(static_cast<std::size_t>(inst.N));
                for (int i = 0; i < inst.N; ++i) {
                    by_dist.emplace_back(inst.dist(center, i), i);
                }
                if (pool_n < inst.N) {
                    std::nth_element(by_dist.begin(),
                                     by_dist.begin() + pool_n,
                                     by_dist.end());
                }
                std::vector<int> candidates;
                candidates.reserve(static_cast<std::size_t>(pool_n));
                for (int i = 0; i < pool_n; ++i) {
                    candidates.push_back(by_dist[static_cast<std::size_t>(i)].second);
                }
                for (int i = 0; i < k; ++i) {
                    const int j = i + rrng.randint(pool_n - i);
                    std::swap(candidates[static_cast<std::size_t>(i)],
                              candidates[static_cast<std::size_t>(j)]);
                }
                std::vector<int> region(candidates.begin(), candidates.begin() + k);
                seed = nearest_neighbor_order(inst, region, rrng.randint(k));
                kind = RestartKind::Region;
            }
        }
        out.stats.phases.seed_construction_seconds +=
            std::chrono::duration<double>(Clock::now() - restart_seed_start).count();
        // The complete typed record is finalized after local search.
        // Elite-seeded restarts (scheduled kicks and anytime ILS) anneal at the
        // reduced kick_t0: a full-melt t0 would erase the inherited structure
        // and reduce the kick to an independent restart with a biased seed.
        const double restart_t0 = out.elite_seed
            ? std::min(sa_t0, options.kick_t0 > 0.0 ? options.kick_t0 : 0.35)
            : sa_t0;
        const double restart_log_ratio = std::log(sa_t1 / restart_t0);
        const int restart_sa_iters = sa_iterations_override >= 0
            ? sa_iterations_override : sa_iters_eff;
        // Insertion policy is per seed kind. Windowed insertion only offers
        // slots near the removed position or near the incoming node's in-tour
        // KNN -- perfect for seeds that start spatially concentrated (smallp,
        // warm, elite), but it removes the global relocations that RANDOM and
        // HIGHP seeds need to contract a spread-out subset at small p: their
        // restarts were observed stalling at L/k ~ 1.0-1.7 instead of ~0.65.
        // Exploration seeds therefore keep the exact O(k) insertion scan;
        // that is what makes them explorers.
        // Only seeds that must relocate members ACROSS THE SQUARE need the exact
        // scan: a random_subset (kind 0) is spread over the whole domain, and a
        // HighPDelete likewise. A Dense seed starts
        // compact and only ever refines locally -- at k=2000 the exact scan
        // returns a bit-identical search for it (same accepted moves, same tour)
        // while costing 25.9us/move against 4.05us, so forcing it there spent
        // ~58% of the wall clock on a foregone conclusion.
        const bool exploration_seed =
            (!out.elite_seed && kind == RestartKind::Random)
            || kind == RestartKind::HighPDelete
            || (options.dense_exact_insertion && !out.elite_seed
                && kind == RestartKind::Dense);
        const bool use_spatial = options.sa_spatial_insertion && !options.sa_exact_insertion;
        const bool restart_exact_insertion = options.sa_exact_insertion
                                             || (!use_spatial && exploration_seed
                                                 && options.exploration_exact_insertion);
        Tour tour;
        tour.init(inst.N);
        tour.set_tour(seed, inst);
        {
            ScopedPhaseTimer phase_timer(out.stats.phases.initial_polish_seconds);
            polish_tour(tour, inst, options, &out.stats,
                        kind == RestartKind::HighPDelete ? 2 : 1);
        }
        if (kind == RestartKind::HighPDelete) {
            ScopedPhaseTimer phase_timer(out.stats.phases.highp_exchange_seconds);
            highp_delete_exchange_descent(tour, inst, warm_start == nullptr ? seed : *warm_start, options, &out.stats, 2);
        }
        // Best-so-far snapshot. A full Tour copy drags the O(N) pos/in_set
        // arrays along (~1 MB per copy at N=200000, on every improvement);
        // the SA only ever needs the node order and the length.
        std::vector<int> best_nodes = tour.nodes;
        double best_length = tour.length;
        // Build the subset candidate table once for the SA loop and reuse it at
        // every 2-opt checkpoint. Membership drifts as SA swaps members, so the
        // table becomes stale, but candidate tables are quality-neutral (the
        // 2-opt recomputes the true delta and only applies improving moves), and
        // rebuilding it every checkpoint is a dominant cost at small p (measured
        // ~1000 SA-steps per rebuild at p=0.005). Added members not in the table
        // fall back to the full KNN list inside the descent.
        const bool sa_use_tables = !use_all_polish_exhaustive_two_opt(options, tour.k);
        const SubsetCandidateTable* sa_table = nullptr;
        SubsetCandidateTable sa_table_storage;
        if (!options.disable_two_opt && sa_use_tables) {
            build_subset_candidates(inst, tour, 16, sa_table_storage);
            sa_table = &sa_table_storage;
        }
        // The index tracks subset membership, which inside the SA loop changes
        // ONLY through accepted swap moves (the periodic 2-opt descents reorder
        // but never re-select), so an O(1) update per acceptance keeps it exact.
        SubsetIndex sindex;
        if (use_spatial) { sindex.build(inst, tour); }
        const int spatial_neighbors = std::max(1, options.sa_spatial_neighbors);
        {
            ScopedPhaseTimer sa_timer(out.stats.phases.sa_seconds);
            for (int it = 0; it < restart_sa_iters; ++it) {
                // A pilot follows the prefix of the full schedule; it does not
                // compress the temperature range into its shorter budget.
                const double frac = (sa_iters_eff <= 1)
                    ? 0.0
                    : static_cast<double>(it) / static_cast<double>(sa_iters_eff - 1);
                const double temperature = restart_t0 * std::exp(restart_log_ratio * frac);
                const bool timing_sample = (it & 63) == 0;
                const Clock::time_point proposal_start = timing_sample ? Clock::now() : Clock::time_point{};
                const int ri = rrng.randint(tour.k);
                const int add = choose_swap_candidate(inst, tour, ri, rrng);
                if (timing_sample) {
                    ++out.stats.phases.sa_proposal_samples;
                    out.stats.phases.sa_proposal_sample_seconds +=
                        std::chrono::duration<double>(Clock::now() - proposal_start).count();
                }
                if (tour.in_set[static_cast<std::size_t>(add)] != 0U) { continue; }
                const Clock::time_point insertion_start = timing_sample ? Clock::now() : Clock::time_point{};
                SwapInsertionMove move =
                    restart_exact_insertion
                        ? find_best_insert_after_remove(inst, tour, ri, add)
                        : (use_spatial
                               ? find_best_insert_after_remove_spatial(inst, tour, sindex, ri, add,
                                                                       spatial_neighbors,
                                                                       std::max(1, options.sa_insertion_window))
                               : find_best_insert_after_remove_windowed(inst, tour, ri, add,
                                                                        std::max(1, options.sa_insertion_window)));
                if (timing_sample) {
                    ++out.stats.phases.sa_insertion_samples;
                    out.stats.phases.sa_insertion_sample_seconds +=
                        std::chrono::duration<double>(Clock::now() - insertion_start).count();
                }
                if (!move.valid) { continue; }
                const double delta = move.delta;
                ++out.stats.sa_moves;
                const bool improving = delta < -kImprovementEps;
                const bool accept = improving || rrng.uniform() < std::exp(-std::max(0.0, delta) / std::max(temperature, 1e-12));
                if (accept) {
                    const int removed_node = tour.nodes[static_cast<std::size_t>(move.remove_pos)];
                    tour.apply_swap_post_rem(move.remove_pos, move.post_pred, move.add_node, inst, move.delta);
                    if (use_spatial) {
                        sindex.remove_member(inst, removed_node);
                        sindex.add_member(inst, move.add_node);
                    }
                    ++out.stats.sa_accepted;
                    if (improving) { ++out.stats.sa_improving; }
                    if (!options.disable_two_opt && (it + 1) % 1000 == 0) {
                        ScopedPhaseTimer checkpoint_timer(out.stats.phases.sa_checkpoint_polish_seconds);
                        if (use_all_polish_exhaustive_two_opt(options, tour.k)) { two_opt_descent(tour, inst, 100, &out.stats); }
                        else { two_opt_candidate_descent(tour, inst, 40, 32, &out.stats, sa_table); }
                    }
                    if (tour.length < best_length - kImprovementEps) {
                        best_nodes = tour.nodes;
                        best_length = tour.length;
                    }
                }
            }
        }
        tour.set_tour(best_nodes, inst);
        // set_tour recomputes the length from scratch; restore the value the
        // search actually tracked so downstream comparisons (elite pool,
        // best-of-restarts) see bit-identical numbers to the old Tour-copy
        // path, whose length was the incrementally maintained one.
        tour.length = best_length;
        {
            ScopedPhaseTimer phase_timer(out.stats.phases.post_sa_polish_seconds);
            polish_tour(tour, inst, options, &out.stats, pilot_only ? 1 : 2);
        }
        const bool run_strong_inline = !pilot_only && !options.staged_search;
        if (run_strong_inline && !options.disable_subset_swap) {
            ScopedPhaseTimer phase_timer(out.stats.phases.subset_swap_seconds);
            subset_swap_descent_impl(tour, inst, options.subset_swap_descent_passes,
                                     !options.disable_two_opt, &out.stats);
            polish_tour(tour, inst, options, &out.stats, 1);
        }
        if (run_strong_inline && !options.disable_pair_exchange) {
            ScopedPhaseTimer phase_timer(out.stats.phases.pair_exchange_seconds);
            subset_pair_exchange_descent(tour, inst, rrng, options, &out.stats,
                                          options.pair_exchange_passes);
        }
        if (run_strong_inline && !options.disable_ruin_recreate) {
            ScopedPhaseTimer phase_timer(out.stats.phases.ruin_recreate_seconds);
            subset_ruin_recreate_lns(tour, inst, rrng, options, &out.stats,
                                     options.ruin_recreate_rounds);
        }
        if (run_strong_inline && !options.disable_ejection_chain) {
            ScopedPhaseTimer phase_timer(out.stats.phases.ejection_chain_seconds);
            (void)subset_ejection_chain_search(tour, inst, rrng, options,
                                               &out.stats);
        }
        if (run_strong_inline && options.oracle.cfg.inline_feedback) {
            ScopedPhaseTimer phase_timer(out.stats.phases.oracle_seconds);
            (void)external_oracle_polish_tour(
                tour, inst, options.oracle, false, &out.stats,
                !options.disable_two_opt);
        }
        out.nodes = tour.nodes;
        out.record = make_restart_record(inst, out.nodes, tour.length, kind);
        out.record.role = role;
        out.record.seed_variant = seed_variant;
        out.record.promotion_stage = promotion_stage;
        out.record.sa_iterations = static_cast<std::uint64_t>(restart_sa_iters);
        out.record.strong_polished = run_strong_inline;
        out.rng_state = rrng;
        out.strong_eligible = !pilot_only;
        return out;
    };

    struct StagedCandidate {
        std::size_t record_index = 0U;
        std::vector<int> nodes;
        Rng rng_state;
    };

    int launched = 0;
    bool kick_snapshot_taken = false;
    double best_outcome_len = std::numeric_limits<double>::infinity();
    std::vector<RestartOutcome> outcomes;
    std::vector<std::vector<int>> restart_nodes;
    std::vector<StagedCandidate> staged_candidates;
    auto merge_outcome = [&](RestartOutcome& outcome) {
        result.stats.add(outcome.stats);
        elite.try_add(outcome.nodes, outcome.record.length);
        ++result.stats.subset_restarts;
        record_subset_restart_kind(result.stats, outcome.record.kind);
        const std::size_t record_index = result.restarts.size();
        result.restarts.push_back(outcome.record);
        restart_nodes.push_back(outcome.nodes);
        if (options.staged_search && outcome.strong_eligible) {
            StagedCandidate staged;
            staged.record_index = record_index;
            staged.nodes = outcome.nodes;
            staged.rng_state = outcome.rng_state;
            staged_candidates.push_back(std::move(staged));
            ++result.stats.strong_polish_candidates;
        }
        if (outcome.record.length < best_outcome_len - kImprovementEps) {
            best_outcome_len = outcome.record.length;
            result.best_restart = static_cast<int>(record_index);
        }
    };
    for (;;) {
        int wave = 0;
        if (launched < kick_begin) {
            // Waves never straddle role boundaries. Scheduled kicks observe the
            // complete independent phase, and continuation starts only after
            // the kick phase has merged deterministically.
            wave = std::min(restart_threads, kick_begin - launched);
        } else if (launched < continuation_begin) {
            wave = std::min(restart_threads, continuation_begin - launched);
        } else if (launched < total_restarts) {
            wave = std::min(restart_threads, total_restarts - launched);
        } else if (time_budget > 0.0
                   && std::chrono::duration<double>(Clock::now() - start).count() < time_budget) {
            // Anytime mode: keep launching restart waves until the wall-clock
            // budget for this (instance, p) solve has elapsed.
            wave = restart_threads;
        } else {
            break;
        }
        outcomes.assign(static_cast<std::size_t>(wave), RestartOutcome{});
        if (!options.disable_elite_restarts) {
            if (launched >= total_restarts) {
                // Anytime: refresh once per wave (documented non-reproducible).
                elite_seeds = elite.export_nodes();
            } else if (launched >= kick_begin
                       && launched < continuation_begin
                       && !kick_snapshot_taken) {
                // Scheduled kicks: snapshot EXACTLY ONCE at the phase boundary,
                // so kick seeds are independent of restart_threads and the
                // deterministic-path guarantee survives.
                elite_seeds = elite.export_nodes();
                kick_snapshot_taken = true;
            }
        }
        detail::run_parallel_indexed(wave, [&](int index) {
            outcomes[static_cast<std::size_t>(index)] = run_restart(
                launched + index, nullptr, -1,
                RestartPromotionStage::None, false);
        });
        // Merge strictly in restart-index order so elite content, stats, and
        // diagnostics are independent of thread scheduling.
        for (int i = 0; i < wave; ++i) {
            RestartOutcome& outcome = outcomes[static_cast<std::size_t>(i)];
            merge_outcome(outcome);
        }
        launched += wave;
    }

    if (racing_n > 0) {
        const int pilot_iters = std::min(options.racing_pilot_iters, sa_iters_eff);
        std::vector<RestartOutcome> raced_outcomes(static_cast<std::size_t>(racing_n));

        // Run bounded waves, then retain the original candidate order. Promotion
        // and final merge are therefore invariant to worker completion order.
        for (int begin = 0; begin < racing_n; begin += restart_threads) {
            const int wave = std::min(restart_threads, racing_n - begin);
            detail::run_parallel_indexed(wave, [&](const int offset) {
                const int index = begin + offset;
                RestartOutcome pilot = run_restart(
                    total_restarts + index,
                    &racing_seeds[static_cast<std::size_t>(index)],
                    pilot_iters,
                    RestartPromotionStage::PilotOnly,
                    true);
                pilot.stats.racing_pilot_restarts = 1;
                raced_outcomes[static_cast<std::size_t>(index)] = std::move(pilot);
            });
        }

        std::vector<int> ranked(static_cast<std::size_t>(racing_n));
        std::iota(ranked.begin(), ranked.end(), 0);
        std::stable_sort(ranked.begin(), ranked.end(), [&](const int lhs, const int rhs) {
            const RestartRecord& a = raced_outcomes[static_cast<std::size_t>(lhs)].record;
            const RestartRecord& b = raced_outcomes[static_cast<std::size_t>(rhs)].record;
            if (a.length != b.length) { return a.length < b.length; }
            if (a.kind != b.kind) {
                return restart_kind_code(a.kind) < restart_kind_code(b.kind);
            }
            if (a.seed_variant != b.seed_variant) {
                return a.seed_variant < b.seed_variant;
            }
            return lhs < rhs;
        });

        const int survivor_count = std::min(options.racing_survivors, racing_n);
        std::vector<int> promoted;
        promoted.reserve(static_cast<std::size_t>(survivor_count));
        std::vector<unsigned char> membership(static_cast<std::size_t>(inst.N), 0U);
        auto jaccard_distance = [&](const std::vector<int>& lhs,
                                    const std::vector<int>& rhs) {
            for (const int node : lhs) {
                membership[static_cast<std::size_t>(node)] = 1U;
            }
            int intersection = 0;
            for (const int node : rhs) {
                intersection += membership[static_cast<std::size_t>(node)] != 0U ? 1 : 0;
            }
            for (const int node : lhs) {
                membership[static_cast<std::size_t>(node)] = 0U;
            }
            const int set_union = static_cast<int>(lhs.size() + rhs.size()) - intersection;
            return set_union > 0
                ? 1.0 - static_cast<double>(intersection) / static_cast<double>(set_union)
                : 0.0;
        };
        for (const int candidate_index : ranked) {
            bool diverse = true;
            for (const int selected_index : promoted) {
                if (jaccard_distance(
                        raced_outcomes[static_cast<std::size_t>(candidate_index)].nodes,
                        raced_outcomes[static_cast<std::size_t>(selected_index)].nodes)
                    + kDistanceEps < options.racing_min_jaccard) {
                    diverse = false;
                    break;
                }
            }
            if (diverse) {
                promoted.push_back(candidate_index);
                if (static_cast<int>(promoted.size()) == survivor_count) { break; }
            }
        }
        // Diversity is a preference, not a reason to leave compute unused.
        for (const int candidate_index : ranked) {
            if (static_cast<int>(promoted.size()) == survivor_count) { break; }
            if (std::find(promoted.begin(), promoted.end(), candidate_index) == promoted.end()) {
                promoted.push_back(candidate_index);
            }
        }

        std::vector<RestartOutcome> promoted_outcomes(promoted.size());
        for (int begin = 0; begin < static_cast<int>(promoted.size()); begin += restart_threads) {
            const int wave = std::min(restart_threads,
                                      static_cast<int>(promoted.size()) - begin);
            detail::run_parallel_indexed(wave, [&](const int offset) {
                const int promoted_pos = begin + offset;
                const int candidate_index = promoted[static_cast<std::size_t>(promoted_pos)];
                promoted_outcomes[static_cast<std::size_t>(promoted_pos)] = run_restart(
                    total_restarts + candidate_index,
                    &racing_seeds[static_cast<std::size_t>(candidate_index)],
                    sa_iters_eff,
                    RestartPromotionStage::PromotedFull,
                    false);
            });
        }
        for (std::size_t i = 0; i < promoted.size(); ++i) {
            const int candidate_index = promoted[i];
            RestartOutcome& full = promoted_outcomes[i];
            full.stats.racing_promoted_restarts = 1;
            full.stats.add(raced_outcomes[static_cast<std::size_t>(candidate_index)].stats);
            full.record.sa_iterations += static_cast<std::uint64_t>(pilot_iters);
            raced_outcomes[static_cast<std::size_t>(candidate_index)] = std::move(full);
        }

        for (RestartOutcome& outcome : raced_outcomes) {
            merge_outcome(outcome);
        }
    }

    if (options.staged_search && !staged_candidates.empty()) {
        std::vector<unsigned char> membership(static_cast<std::size_t>(inst.N), 0U);
        auto jaccard_distance = [&](const std::vector<int>& lhs,
                                    const std::vector<int>& rhs) {
            for (const int node : lhs) {
                membership[static_cast<std::size_t>(node)] = 1U;
            }
            int intersection = 0;
            for (const int node : rhs) {
                intersection += membership[static_cast<std::size_t>(node)] != 0U ? 1 : 0;
            }
            for (const int node : lhs) {
                membership[static_cast<std::size_t>(node)] = 0U;
            }
            const int set_union = static_cast<int>(lhs.size() + rhs.size()) - intersection;
            return set_union > 0
                ? 1.0 - static_cast<double>(intersection) / static_cast<double>(set_union)
                : 0.0;
        };

        auto select_diverse = [&](std::vector<std::size_t> pool,
                                  const int requested) {
            const int target = std::min(requested, static_cast<int>(pool.size()));
            std::stable_sort(pool.begin(), pool.end(), [&](const std::size_t lhs,
                                                            const std::size_t rhs) {
                const RestartRecord& a = result.restarts[
                    staged_candidates[lhs].record_index];
                const RestartRecord& b = result.restarts[
                    staged_candidates[rhs].record_index];
                if (a.length != b.length) { return a.length < b.length; }
                if (a.role != b.role) {
                    return restart_role_code(a.role) < restart_role_code(b.role);
                }
                if (a.kind != b.kind) {
                    return restart_kind_code(a.kind) < restart_kind_code(b.kind);
                }
                if (a.seed_variant != b.seed_variant) {
                    return a.seed_variant < b.seed_variant;
                }
                return staged_candidates[lhs].record_index
                       < staged_candidates[rhs].record_index;
            });

            std::vector<std::size_t> selected;
            selected.reserve(static_cast<std::size_t>(target));
            for (const std::size_t candidate : pool) {
                bool diverse = true;
                for (const std::size_t prior : selected) {
                    if (jaccard_distance(staged_candidates[candidate].nodes,
                                         staged_candidates[prior].nodes)
                        + kDistanceEps < options.strong_polish_min_jaccard) {
                        diverse = false;
                        break;
                    }
                }
                if (diverse) {
                    selected.push_back(candidate);
                    if (static_cast<int>(selected.size()) == target) { break; }
                }
            }
            // Diversity is a preference; never leave a configured finalist slot
            // idle when a quality-ranked candidate remains available.
            for (const std::size_t candidate : pool) {
                if (static_cast<int>(selected.size()) == target) { break; }
                if (std::find(selected.begin(), selected.end(), candidate)
                    == selected.end()) {
                    selected.push_back(candidate);
                }
            }
            return selected;
        };

        std::vector<std::size_t> core_pool;
        std::vector<std::size_t> continuation_pool;
        std::vector<std::size_t> raced_pool;
        for (std::size_t i = 0; i < staged_candidates.size(); ++i) {
            const RestartRole role = result.restarts[
                staged_candidates[i].record_index].role;
            if (role == RestartRole::RacedProduction) {
                raced_pool.push_back(i);
            } else if (role == RestartRole::Continuation) {
                continuation_pool.push_back(i);
            } else {
                core_pool.push_back(i);
            }
        }

        std::vector<std::size_t> finalists;
        auto append_unique = [&](const std::vector<std::size_t>& chosen) {
            for (const std::size_t candidate : chosen) {
                if (std::find(finalists.begin(), finalists.end(), candidate)
                    == finalists.end()) {
                    finalists.push_back(candidate);
                }
            }
        };
        if (request.continuation_only) {
            std::vector<std::size_t> all(staged_candidates.size());
            std::iota(all.begin(), all.end(), 0U);
            append_unique(select_diverse(std::move(all),
                                         options.strong_polish_finalists));
        } else if (options.continuation_policy == ContinuationPolicy::FixedBudget) {
            core_pool.insert(core_pool.end(), continuation_pool.begin(),
                             continuation_pool.end());
            append_unique(select_diverse(std::move(core_pool),
                                         options.strong_polish_finalists));
        } else {
            // Supplemental continuation and racing cannot displace the stable
            // independent population. They receive small explicit finalist
            // reserves in addition to the configured core quota.
            append_unique(select_diverse(std::move(core_pool),
                                         options.strong_polish_finalists));
            append_unique(select_diverse(std::move(continuation_pool), 1));
        }
        append_unique(select_diverse(
            std::move(raced_pool), std::max(1, options.racing_survivors)));
        std::stable_sort(finalists.begin(), finalists.end(),
                         [&](const std::size_t lhs, const std::size_t rhs) {
            return staged_candidates[lhs].record_index
                   < staged_candidates[rhs].record_index;
        });

        struct StrongOutcome {
            std::size_t staged_index = 0U;
            std::vector<int> nodes;
            RestartRecord record;
            SearchStats stats;
        };
        std::vector<StrongOutcome> strong_outcomes(finalists.size());
        auto run_strong = [&](const std::size_t staged_index) {
            StrongOutcome out;
            out.staged_index = staged_index;
            const StagedCandidate& candidate = staged_candidates[staged_index];
            const RestartRecord old_record = result.restarts[candidate.record_index];
            Tour tour;
            tour.init(inst.N);
            tour.set_tour(candidate.nodes, inst);
            // Resume the exact incrementally maintained value and RNG state at
            // the post-SA boundary. With every candidate promoted, this is
            // bit-equivalent to the legacy inline strong-search path.
            tour.length = old_record.length;
            Rng strong_rng = candidate.rng_state;
            if (!options.disable_subset_swap) {
                ScopedPhaseTimer phase_timer(out.stats.phases.subset_swap_seconds);
                subset_swap_descent_impl(tour, inst,
                                         options.subset_swap_descent_passes,
                                         !options.disable_two_opt, &out.stats);
                polish_tour(tour, inst, options, &out.stats, 1);
            }
            if (!options.disable_pair_exchange) {
                ScopedPhaseTimer phase_timer(out.stats.phases.pair_exchange_seconds);
                subset_pair_exchange_descent(tour, inst, strong_rng, options,
                                              &out.stats,
                                              options.pair_exchange_passes);
            }
            if (!options.disable_ruin_recreate) {
                ScopedPhaseTimer phase_timer(out.stats.phases.ruin_recreate_seconds);
                subset_ruin_recreate_lns(tour, inst, strong_rng, options,
                                         &out.stats,
                                         options.ruin_recreate_rounds);
            }
            if (!options.disable_ejection_chain) {
                ScopedPhaseTimer phase_timer(out.stats.phases.ejection_chain_seconds);
                (void)subset_ejection_chain_search(tour, inst, strong_rng,
                                                   options, &out.stats);
            }
            if (options.oracle.cfg.inline_feedback) {
                ScopedPhaseTimer phase_timer(out.stats.phases.oracle_seconds);
                (void)external_oracle_polish_tour(
                    tour, inst, options.oracle, false, &out.stats,
                    !options.disable_two_opt);
            }
            if (tour.length > old_record.length + kImprovementEps) {
                throw std::logic_error("strong polish worsened a restart");
            }
            out.nodes = tour.nodes;
            out.record = make_restart_record(inst, out.nodes, tour.length,
                                             old_record.kind);
            out.record.role = old_record.role;
            out.record.seed_variant = old_record.seed_variant;
            out.record.promotion_stage = old_record.promotion_stage;
            out.record.sa_iterations = old_record.sa_iterations;
            out.record.strong_polished = true;
            out.stats.strong_polish_finalists = 1;
            if (tour.length < old_record.length - kImprovementEps) {
                out.stats.strong_polish_improvements = 1;
            }
            return out;
        };

        for (int begin = 0; begin < static_cast<int>(finalists.size());
             begin += restart_threads) {
            const int wave = std::min(restart_threads,
                                      static_cast<int>(finalists.size()) - begin);
            detail::run_parallel_indexed(wave, [&](const int offset) {
                const int position = begin + offset;
                strong_outcomes[static_cast<std::size_t>(position)] =
                    run_strong(finalists[static_cast<std::size_t>(position)]);
            });
        }
        for (StrongOutcome& outcome : strong_outcomes) {
            const StagedCandidate& candidate = staged_candidates[outcome.staged_index];
            result.stats.add(outcome.stats);
            result.restarts[candidate.record_index] = outcome.record;
            restart_nodes[candidate.record_index] = outcome.nodes;
        }

        // Rebuild the archive from each restart's final stage. This prevents a
        // finalist's weaker pre-polish snapshot from occupying an elite or
        // relinking slot alongside its refined version.
        const std::uint64_t early_diversity_candidates = elite.diversity_candidates();
        const std::uint64_t early_diversity_retained = elite.diversity_retained();
        const std::uint64_t early_diversity_rejected = elite.diversity_rejected();
        ElitePool final_elite(std::max(8, total_restarts + 8),
                              EliteMode::Set,
                              options.elite_diversity_slots,
                              options.elite_min_jaccard,
                              options.elite_quality_slack);
        for (std::size_t i = 0; i < result.restarts.size(); ++i) {
            final_elite.try_add(restart_nodes[i], result.restarts[i].length);
        }
        result.stats.elite_diversity_candidates += early_diversity_candidates;
        result.stats.elite_diversity_retained += early_diversity_retained;
        result.stats.elite_diversity_rejected += early_diversity_rejected;
        elite = std::move(final_elite);

        best_outcome_len = std::numeric_limits<double>::infinity();
        result.best_restart = -1;
        for (std::size_t i = 0; i < result.restarts.size(); ++i) {
            if (result.restarts[i].length < best_outcome_len - kImprovementEps) {
                best_outcome_len = result.restarts[i].length;
                result.best_restart = static_cast<int>(i);
            }
        }
    }

    if (!options.disable_path_relink && options.path_relink_top >= 2) {
        ScopedPhaseTimer phase_timer(result.stats.phases.path_relink_seconds);
        Rng relink_rng(make_stream_seed(solve_stream_base,
                                        0x510e527fade682d1ULL,
                                        0x1f83d9abfb41bd6bULL));
        const std::vector<EliteEntry> relink_entries = elite.export_relink_entries(
            options.path_relink_top,
            options.path_relink_diverse_reserve,
            options.path_relink_max_removed);

        struct PairCandidate {
            int first = -1;
            int second = -1;
            int removed = 0;
            std::uint64_t estimated_scans = 0U;
            double utility = -std::numeric_limits<double>::infinity();
            double worst_length = std::numeric_limits<double>::infinity();
        };
        std::vector<PairCandidate> pairs;
        const double best_length = relink_entries.empty()
            ? std::numeric_limits<double>::infinity()
            : relink_entries.front().length;
        for (int i = 0; i < static_cast<int>(relink_entries.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(relink_entries.size()); ++j) {
                ++result.stats.path_relink_pairs_considered;
                const EliteEntry& a = relink_entries[static_cast<std::size_t>(i)];
                const EliteEntry& b = relink_entries[static_cast<std::size_t>(j)];
                const int removed = sorted_set_removed_count(
                    a.canonical_key, b.canonical_key);
                if (removed == 0
                    || (options.path_relink_max_removed > 0
                        && removed > options.path_relink_max_removed)) {
                    ++result.stats.path_relink_pairs_skipped_distance;
                    continue;
                }
                const std::uint64_t estimated_scans =
                    estimated_bidirectional_relink_scans(removed);
                const double set_union = static_cast<double>(a.nodes.size())
                    + static_cast<double>(removed);
                const double set_distance = set_union > 0.0
                    ? 2.0 * static_cast<double>(removed) / set_union
                    : 0.0;
                const double edge_distance = cycle_edge_jaccard_distance(
                    a.nodes, b.nodes);
                const double scale = std::max(std::fabs(best_length), kDistanceEps);
                const double quality_gap = std::max(
                    0.0, (0.5 * (a.length + b.length) - best_length) / scale);
                const double work_penalty = 1.0
                    + static_cast<double>(estimated_scans) / 100000.0;
                const double anchor_bonus = (i == 0 || j == 0) ? 1.20 : 1.0;
                PairCandidate pair;
                pair.first = i;
                pair.second = j;
                pair.removed = removed;
                pair.estimated_scans = estimated_scans;
                pair.worst_length = std::max(a.length, b.length);
                pair.utility = anchor_bonus * (set_distance + 0.5 * edge_distance)
                    / ((1.0 + 50.0 * quality_gap) * work_penalty);
                pairs.push_back(pair);
            }
        }
        std::stable_sort(pairs.begin(), pairs.end(), [](const PairCandidate& lhs,
                                                        const PairCandidate& rhs) {
            if (lhs.utility != rhs.utility) { return lhs.utility > rhs.utility; }
            if (lhs.worst_length != rhs.worst_length) {
                return lhs.worst_length < rhs.worst_length;
            }
            if (lhs.estimated_scans != rhs.estimated_scans) {
                return lhs.estimated_scans < rhs.estimated_scans;
            }
            if (lhs.first != rhs.first) { return lhs.first < rhs.first; }
            return lhs.second < rhs.second;
        });

        int selected_pairs = 0;
        std::uint64_t removed_sum = 0U;
        std::uint64_t estimated_scan_sum = 0U;
        for (const PairCandidate& pair : pairs) {
            const bool exceeds_pairs = options.path_relink_max_pairs > 0
                && selected_pairs >= options.path_relink_max_pairs;
            const std::uint64_t pair_removed =
                static_cast<std::uint64_t>(pair.removed);
            const std::uint64_t removed_limit =
                static_cast<std::uint64_t>(options.path_relink_max_removed_sum);
            const bool exceeds_removed = options.path_relink_max_removed_sum > 0
                && (pair_removed > removed_limit
                    || removed_sum > removed_limit - pair_removed);
            const bool exceeds_scans = options.path_relink_max_candidate_scans > 0
                && (pair.estimated_scans
                        > static_cast<std::uint64_t>(options.path_relink_max_candidate_scans)
                    || estimated_scan_sum
                        > static_cast<std::uint64_t>(options.path_relink_max_candidate_scans)
                            - pair.estimated_scans);
            if (exceeds_pairs || exceeds_removed || exceeds_scans) {
                ++result.stats.path_relink_pairs_skipped_budget;
                continue;
            }
            ++selected_pairs;
            removed_sum += static_cast<std::uint64_t>(pair.removed);
            estimated_scan_sum += pair.estimated_scans;

            std::vector<int> rel_nodes;
            double rel_len = std::numeric_limits<double>::infinity();
            if (subset_path_relink_bidirectional(
                    inst,
                    relink_entries[static_cast<std::size_t>(pair.first)].nodes,
                    relink_entries[static_cast<std::size_t>(pair.second)].nodes,
                    relink_rng, options, rel_nodes, rel_len, &result.stats)) {
                const auto before_entries = elite.entries().size();
                const double before_best = elite.entries().empty()
                    ? std::numeric_limits<double>::infinity()
                    : elite.entries().front().length;
                const double before_worst = elite.entries().empty()
                    ? std::numeric_limits<double>::infinity()
                    : elite.entries().back().length;
                elite.try_add(rel_nodes, rel_len);
                const auto after_entries = elite.entries().size();
                const double after_best = elite.entries().empty()
                    ? std::numeric_limits<double>::infinity()
                    : elite.entries().front().length;
                if (after_entries > before_entries
                    || rel_len < before_worst - kImprovementEps) {
                    ++result.stats.path_relink_elite_insertions;
                }
                if (after_best < before_best - kImprovementEps) {
                    ++result.stats.path_relink_best_improvements;
                    ++result.stats.path_relink_improvements;
                }
            }
        }
        result.stats.path_relink_removed_sum += removed_sum;
        if (options.path_relink_max_candidate_scans > 0
            && result.stats.path_relink_candidate_scans
                > static_cast<std::uint64_t>(options.path_relink_max_candidate_scans)) {
            throw std::logic_error("path relinking exceeded its candidate-scan budget");
        }
    }

    {
        ScopedPhaseTimer phase_timer(result.stats.phases.oracle_seconds);
        polish_elite_with_oracle(elite, inst, options, false, options.oracle.cfg.subset_top, &result.stats);
    }
    const auto nodes = elite.export_nodes();
    if (!nodes.empty()) {
        result.tour.set_tour(nodes.front(), inst);
        ScopedPhaseTimer phase_timer(result.stats.phases.final_polish_seconds);
        final_polish_tour(result.tour, inst, options, &result.stats, 2);
    }
    result.stats.elite_diversity_candidates += elite.diversity_candidates();
    result.stats.elite_diversity_retained += elite.diversity_retained();
    result.stats.elite_diversity_rejected += elite.diversity_rejected();
    result.stats.subset_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

} // namespace aldous_tsp
