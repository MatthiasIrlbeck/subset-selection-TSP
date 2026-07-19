#include "solver_internal.hpp"
#include "worker.hpp"

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
    const auto start = Clock::now();
    SolveResult result;
    result.tour.init(inst.N);
    k = std::max(0, std::min(k, inst.N));
    if (k <= 0) { return result; }
    if (k >= inst.N) { return solve_tsp(inst, rng, options); }

    const double p = static_cast<double>(k) / static_cast<double>(std::max(1, inst.N));
    // Explicit values configure the base population. AUTO reproduces the
    // historical effective count. Supplemental continuation is deliberately
    // outside this population so adding neighboring p-values cannot remove an
    // independent draw or worsen the best-of-restarts result.
    const int resolved_restarts = options.subset_restarts >= 1
        ? options.subset_restarts
        : (p <= 0.08 ? 8 : 3);
    const bool has_warm = warm_start != nullptr && !warm_start->empty();
    if (request.continuation_only && !has_warm) {
        throw std::invalid_argument("continuation-only subset solve requires a warm start");
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
    if (static_cast<int>(independent_seeds.size()) != indep_restarts
        || static_cast<int>(continuation_seeds.size()) != continuation_n) {
        throw std::runtime_error("failed to construct the requested restart seed population");
    }

    ElitePool elite(std::max(8, total_restarts + 8), EliteMode::Set);
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
        bool elite_seed = false;
        SearchStats stats;
    };

    // Elite snapshot for anytime (time-budget) ILS restarts. Refreshed once
    // per wave from the current elite pool; restarts beyond the scheduled count
    // seed from a perturbed elite member instead of a cold seed, turning the
    // anytime tail into subset-level iterated local search. Only active in
    // time-budget mode, which is already documented as non-reproducible, so the
    // deterministic scheduled-restart path is unaffected.
    std::vector<std::vector<int>> elite_seeds;

    auto run_restart = [&](int restart) -> RestartOutcome {
        RestartOutcome out;
        Rng rrng(make_stream_seed(restart_stream_base,
                                  static_cast<std::uint64_t>(restart),
                                  0x452821e638d01377ULL));
        std::vector<int> seed;
        RestartKind kind = RestartKind::Random;
        RestartRole role = RestartRole::IndependentDiagnostic;
        int seed_variant = 0;
        const auto restart_seed_start = Clock::now();
        const bool scheduled_kick = restart >= kick_begin
                                    && restart < continuation_begin
                                    && !elite_seeds.empty();
        const bool anytime_restart = restart >= total_restarts;
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
            const SeedCandidate* candidate = nullptr;
            if (restart < kick_begin) {
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
            for (int it = 0; it < sa_iters_eff; ++it) {
                const double frac = (sa_iters_eff <= 1) ? 0.0 : static_cast<double>(it) / static_cast<double>(sa_iters_eff - 1);
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
            polish_tour(tour, inst, options, &out.stats, 2);
        }
        if (!options.disable_subset_swap) {
            ScopedPhaseTimer phase_timer(out.stats.phases.subset_swap_seconds);
            subset_swap_descent_impl(tour, inst, options.subset_swap_descent_passes, !options.disable_two_opt, &out.stats);
            polish_tour(tour, inst, options, &out.stats, 1);
        }
        if (!options.disable_pair_exchange) {
            ScopedPhaseTimer phase_timer(out.stats.phases.pair_exchange_seconds);
            subset_pair_exchange_descent(tour, inst, rrng, options, &out.stats, options.pair_exchange_passes);
        }
        if (!options.disable_ruin_recreate) {
            ScopedPhaseTimer phase_timer(out.stats.phases.ruin_recreate_seconds);
            subset_ruin_recreate_lns(tour, inst, rrng, options, &out.stats, options.ruin_recreate_rounds);
        }
        if (options.oracle.cfg.inline_feedback) {
            ScopedPhaseTimer phase_timer(out.stats.phases.oracle_seconds);
            (void)external_oracle_polish_tour(tour, inst, options.oracle, false, &out.stats, !options.disable_two_opt);
        }
        out.nodes = tour.nodes;
        out.record = make_restart_record(inst, out.nodes, tour.length, kind);
        out.record.role = role;
        out.record.seed_variant = seed_variant;
        return out;
    };

    int launched = 0;
    bool kick_snapshot_taken = false;
    double best_outcome_len = std::numeric_limits<double>::infinity();
    std::vector<RestartOutcome> outcomes;
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
            outcomes[static_cast<std::size_t>(index)] = run_restart(launched + index);
        });
        // Merge strictly in restart-index order so elite content, stats, and
        // diagnostics are independent of thread scheduling.
        for (int i = 0; i < wave; ++i) {
            RestartOutcome& outcome = outcomes[static_cast<std::size_t>(i)];
            result.stats.add(outcome.stats);
            elite.try_add(outcome.nodes, outcome.record.length);
            ++result.stats.subset_restarts;
            record_subset_restart_kind(result.stats, outcome.record.kind);
            result.restarts.push_back(outcome.record);
            if (outcome.record.length < best_outcome_len - kImprovementEps) {
                best_outcome_len = outcome.record.length;
                result.best_restart = static_cast<int>(result.restarts.size()) - 1;
            }
        }
        launched += wave;
    }

    if (!options.disable_path_relink) {
        ScopedPhaseTimer phase_timer(result.stats.phases.path_relink_seconds);
        Rng relink_rng(make_stream_seed(solve_stream_base,
                                        0x510e527fade682d1ULL,
                                        0x1f83d9abfb41bd6bULL));
        auto elite_nodes = elite.export_nodes();
        const int top = std::min(static_cast<int>(elite_nodes.size()), std::max(0, options.path_relink_top));
        for (int i = 0; i < top; ++i) {
            for (int j = i + 1; j < top; ++j) {
                std::vector<int> rel_nodes;
                double rel_len = std::numeric_limits<double>::infinity();
                if (subset_path_relink_bidirectional(inst, elite_nodes[static_cast<std::size_t>(i)], elite_nodes[static_cast<std::size_t>(j)], relink_rng, options, rel_nodes, rel_len, &result.stats)) {
                    const auto before_entries = elite.entries().size();
                    const double before_best = elite.entries().empty() ? std::numeric_limits<double>::infinity() : elite.entries().front().length;
                    const double before_worst = elite.entries().empty() ? std::numeric_limits<double>::infinity() : elite.entries().back().length;
                    elite.try_add(rel_nodes, rel_len);
                    const auto after_entries = elite.entries().size();
                    const double after_best = elite.entries().empty() ? std::numeric_limits<double>::infinity() : elite.entries().front().length;
                    if (after_entries > before_entries || rel_len < before_worst - kImprovementEps) {
                        ++result.stats.path_relink_elite_insertions;
                    }
                    if (after_best < before_best - kImprovementEps) {
                        ++result.stats.path_relink_best_improvements;
                        ++result.stats.path_relink_improvements;
                    }
                }
            }
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
    result.stats.subset_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

} // namespace aldous_tsp
