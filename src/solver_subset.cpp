#include "solver_internal.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <thread>

namespace aldous_tsp {
namespace {

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

SolveResult solve_subset(const Instance& inst, int k, Rng& rng, const SolverOptions& options, const std::vector<int>* warm_start) {
    const auto start = Clock::now();
    SolveResult result;
    result.tour.init(inst.N);
    k = std::max(0, std::min(k, inst.N));
    if (k <= 0) { return result; }
    if (k >= inst.N) { return solve_tsp(inst, rng, options); }

    const double p = static_cast<double>(k) / static_cast<double>(std::max(1, inst.N));
    // Resolve the restart count. Explicit values (>= 1) are authoritative; the
    // AUTO default (-1) reproduces the historical effective behavior, in which
    // the small-p seed pool forced 8 restarts at p <= 0.08 regardless of the
    // flag, and 3 ran otherwise. Without this, fixing the flag to be honest
    // would have silently cut default-quality at small p (measured: L/k
    // 0.6031 -> 0.6126 at p=0.02, N=5000).
    const int resolved_restarts = options.subset_restarts >= 1
        ? options.subset_restarts
        : (p <= 0.08 ? 8 : 3);
    ElitePool elite(std::max(8, resolved_restarts + 8), EliteMode::Set);
    std::vector<std::vector<int>> seed_pool;
    std::vector<RestartKind> seed_kind;
    std::vector<double> seed_length;
    auto add_seed = [&](std::vector<int> seed, RestartKind kind) {
        if (static_cast<int>(seed.size()) != k) { return; }
        Tour t;
        t.init(inst.N);
        t.set_tour(seed, inst);
        elite.try_add(seed, t.length);
        seed_pool.push_back(std::move(seed));
        seed_kind.push_back(kind);
        seed_length.push_back(t.length);
    };

    if (warm_start != nullptr && !warm_start->empty()) {
        add_seed(resize_seed(inst, *warm_start, k, rng, 0), RestartKind::Warm);
        add_seed(resize_seed(inst, *warm_start, k, rng, 1), RestartKind::Warm);
        if (!options.disable_highp_delete && p >= 0.50 && (options.mode == SolverMode::HighPDelete || options.mode == SolverMode::Hybrid || options.mode == SolverMode::Balanced) && static_cast<int>(warm_start->size()) > k) {
            add_seed(highp_delete_seed(inst, *warm_start, k, rng, 0), RestartKind::HighPDelete);
            add_seed(highp_delete_seed(inst, *warm_start, k, rng, 1), RestartKind::HighPDelete);
            add_seed(segment_delete_seed(inst, *warm_start, k, rng), RestartKind::HighPDelete);
        }
    }
    if (!options.disable_smallp_seeds && (options.mode == SolverMode::SmallPRegion || options.mode == SolverMode::Hybrid || p <= 0.08)) {
        auto seeds = make_smallp_seed_pool(inst, k, rng, 8);
        for (auto& s : seeds) { add_seed(std::move(s), RestartKind::SmallP); }
    }
    const bool dense_fill = options.small_p_dense_fill && p <= 0.08;
    while (static_cast<int>(seed_pool.size()) < resolved_restarts) {
        std::vector<int> seed;
        RestartKind fill_kind = RestartKind::Random;
        if (dense_fill || static_cast<int>(seed_pool.size()) % 2 == 0) {
            seed = dense_seed(inst, k, rng, static_cast<int>(seed_pool.size()));
            fill_kind = RestartKind::Dense;  // compact, local refinement
        } else {
            seed = random_subset(inst.N, k, rng);
            seed = (k > 36) ? nearest_neighbor_order(inst, seed, rng.randint(k)) : farthest_insertion_order(inst, seed);
            fill_kind = RestartKind::Random;  // spread over the whole domain
        }
        add_seed(std::move(seed), fill_kind);
    }

    // --restarts is authoritative. Previously the restart count was
    //   max(seed_pool.size(), subset_restarts)
    // and the seed builders inject up to 8 small-p seeds at p <= 0.08, so the
    // flag could not lower the restart count below the pool size: `--restarts 1`
    // and `--restarts 8` ran identically. That made the search budget
    // uncontrollable, which is fatal for a convergence study.
    //
    // When the pool is larger than the requested count we round-robin across seed
    // KINDS (best-first within each kind, most promising kind first) rather than
    // simply taking the globally shortest seeds. Kind is not cosmetic -- it
    // selects specialised operators (high-p exchange, small-p region moves) -- so
    // a purely length-ranked truncation could silently disable an entire operator.
    const int total_restarts = resolved_restarts;
    // Scheduled elite-kick restarts occupy the LAST kick_n slots; at least one
    // independent restart always runs so the kick phase has an incumbent.
    const int kick_n = std::max(0, std::min(options.subset_kick_restarts, total_restarts - 1));
    const int indep_restarts = total_restarts - kick_n;
    bool kick_snapshot_taken = false;
    if (static_cast<int>(seed_pool.size()) > total_restarts) {
        std::map<RestartKind, std::vector<int>> by_kind;
        for (std::size_t i = 0; i < seed_pool.size(); ++i) {
            by_kind[seed_kind[i]].push_back(static_cast<int>(i));
        }
        std::vector<std::vector<int>*> kinds;
        for (auto& entry : by_kind) {
            std::sort(entry.second.begin(), entry.second.end(), [&](int a, int b) {
                return seed_length[static_cast<std::size_t>(a)] < seed_length[static_cast<std::size_t>(b)];
            });
            kinds.push_back(&entry.second);
        }
        // Most promising kind first, judged by its best seed.
        std::stable_sort(kinds.begin(), kinds.end(), [&](const std::vector<int>* a, const std::vector<int>* b) {
            return seed_length[static_cast<std::size_t>(a->front())] < seed_length[static_cast<std::size_t>(b->front())];
        });
        std::vector<int> chosen;
        chosen.reserve(static_cast<std::size_t>(total_restarts));
        for (std::size_t round = 0; static_cast<int>(chosen.size()) < total_restarts; ++round) {
            bool progressed = false;
            for (std::vector<int>* group : kinds) {
                if (round >= group->size()) { continue; }
                chosen.push_back((*group)[round]);
                progressed = true;
                if (static_cast<int>(chosen.size()) == total_restarts) { break; }
            }
            if (!progressed) { break; }
        }
        std::vector<std::vector<int>> kept_pool;
        std::vector<RestartKind> kept_kind;
        kept_pool.reserve(chosen.size());
        kept_kind.reserve(chosen.size());
        for (const int idx : chosen) {
            kept_pool.push_back(std::move(seed_pool[static_cast<std::size_t>(idx)]));
            kept_kind.push_back(seed_kind[static_cast<std::size_t>(idx)]);
        }
        seed_pool = std::move(kept_pool);
        seed_kind = std::move(kept_kind);
    }
    const int sa_iters_eff = effective_sa_iters(options, k, inst.N);
    const double time_budget = options.time_budget_per_p;
    const double sa_t0 = options.sa_t0 > 0.0 ? options.sa_t0 : 1.4;
    const double sa_t1 = options.sa_t1 > 0.0 ? options.sa_t1 : 0.00005;
    const int restart_threads = std::max(1, options.restart_threads);
    // One base draw, then a derived stream per restart: restart results are a
    // pure function of (options, instance, restart index), so the outcome is
    // deterministic and invariant to restart_threads outside budget mode.
    const std::uint64_t restart_stream_base = rng.next_u64();

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
        Rng rrng(make_stream_seed(restart_stream_base, static_cast<std::uint64_t>(restart), 0x452821e638d01377ULL));
        std::vector<int> seed;
        RestartKind kind = RestartKind::Random;
        const bool scheduled_kick = (restart >= indep_restarts) && (restart < total_restarts)
                                    && !options.disable_elite_restarts && !elite_seeds.empty();
        const bool elite_ils = scheduled_kick
                               || ((restart >= total_restarts) && !options.disable_elite_restarts
                                   && !elite_seeds.empty()
                                   && (((restart - total_restarts) & 1) == 0));  // anytime: alternate elite/cold
        if (elite_ils) {
            // Pick among the top few elite members and apply a small, spatially
            // coherent ruin-and-recreate kick: swap out ~8% of members, each
            // replaced by a KNN neighbor of a retained member (falling back to
            // random). Keeping the perturbation local preserves the good
            // structure so polish + SA intensify around it.
            const int pool = std::min<int>(4, static_cast<int>(elite_seeds.size()));
            seed = elite_seeds[static_cast<std::size_t>(rrng.randint(pool))];
            const double frac = (options.kick_fraction > 0.0 && options.kick_fraction < 1.0) ? options.kick_fraction : 0.10;
            const int kick = std::max(1, static_cast<int>(std::lround(frac * static_cast<double>(seed.size()))));
            thread_local std::vector<unsigned char> inset;
            inset.assign(static_cast<std::size_t>(inst.N), 0U);
            for (int v : seed) { inset[static_cast<std::size_t>(v)] = 1U; }
            for (int t = 0; t < kick; ++t) {
                const int ri = rrng.randint(static_cast<int>(seed.size()));
                inset[static_cast<std::size_t>(seed[static_cast<std::size_t>(ri)])] = 0U;
                // Re-add near a random retained member for spatial coherence.
                int add = -1;
                const int anchor = seed[static_cast<std::size_t>(rrng.randint(static_cast<int>(seed.size())))];
                if (inst.knn_k > 0) {
                    const int kstart = rrng.randint(inst.knn_k);
                    for (int off = 0; off < inst.knn_k; ++off) {
                        const int cand = inst.knn_at(anchor, (kstart + off) % inst.knn_k);
                        if (cand >= 0 && inset[static_cast<std::size_t>(cand)] == 0U) { add = cand; break; }
                    }
                }
                if (add < 0) {
                    add = rrng.randint(inst.N);
                    int guard = 0;
                    while (inset[static_cast<std::size_t>(add)] != 0U && guard < 64) { add = rrng.randint(inst.N); ++guard; }
                }
                inset[static_cast<std::size_t>(add)] = 1U;
                seed[static_cast<std::size_t>(ri)] = add;
            }
            kind = scheduled_kick ? RestartKind::Kick : RestartKind::Elite;
            out.elite_seed = true;
        } else {
            seed = seed_pool[static_cast<std::size_t>(restart % static_cast<int>(seed_pool.size()))];
            kind = seed_kind[static_cast<std::size_t>(restart % static_cast<int>(seed_kind.size()))];
            if (options.region_seeds
                && (kind == RestartKind::Random || kind == RestartKind::Dense)) {
                // Fresh region seed per restart: uniform center, k nearest
                // points, nearest-neighbor order. Replaces the pooled random
                // seed (which is reused across the restart cycle anyway, and
                // whose contraction phase fails about half the time).
                const int center = rrng.randint(inst.N);
                const double dil = options.region_dilation >= 1.0 ? options.region_dilation : 3.0;
                const int pool_n = std::min(inst.N, std::max(k, static_cast<int>(std::lround(dil * static_cast<double>(k)))));
                std::vector<std::pair<double, int>> by_dist;
                by_dist.reserve(static_cast<std::size_t>(inst.N));
                for (int i = 0; i < inst.N; ++i) {
                    by_dist.emplace_back(inst.dist(center, i), i);
                }
                std::nth_element(by_dist.begin(), by_dist.begin() + pool_n, by_dist.end());
                std::vector<int> candidates;
                candidates.reserve(static_cast<std::size_t>(pool_n));
                for (int i = 0; i < pool_n; ++i) { candidates.push_back(by_dist[static_cast<std::size_t>(i)].second); }
                // Random k of the dilated neighborhood: local, but free to skip
                // awkward points instead of being forced to take every one.
                for (int i = 0; i < k; ++i) {
                    const int j = i + rrng.randint(pool_n - i);
                    std::swap(candidates[static_cast<std::size_t>(i)], candidates[static_cast<std::size_t>(j)]);
                }
                std::vector<int> region(candidates.begin(), candidates.begin() + k);
                seed = nearest_neighbor_order(inst, region, rrng.randint(k));
                kind = RestartKind::Region;  // compact: windowed insertion is enough
            }
        }
        out.record.kind = kind;
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
        polish_tour(tour, inst, options, &out.stats,
                    kind == RestartKind::HighPDelete ? 2 : 1);
        if (kind == RestartKind::HighPDelete) {
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
        for (int it = 0; it < sa_iters_eff; ++it) {
            const double frac = (sa_iters_eff <= 1) ? 0.0 : static_cast<double>(it) / static_cast<double>(sa_iters_eff - 1);
            const double temperature = restart_t0 * std::exp(restart_log_ratio * frac);
            const int ri = rrng.randint(tour.k);
            const int add = choose_swap_candidate(inst, tour, ri, rrng);
            if (tour.in_set[static_cast<std::size_t>(add)] != 0U) { continue; }
            SwapInsertionMove move =
                restart_exact_insertion
                    ? find_best_insert_after_remove(inst, tour, ri, add)
                    : (use_spatial
                           ? find_best_insert_after_remove_spatial(inst, tour, sindex, ri, add,
                                                                   spatial_neighbors,
                                                                   std::max(1, options.sa_insertion_window))
                           : find_best_insert_after_remove_windowed(inst, tour, ri, add,
                                                                    std::max(1, options.sa_insertion_window)));
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
                    if (use_all_polish_exhaustive_two_opt(options, tour.k)) { two_opt_descent(tour, inst, 100, &out.stats); }
                    else { two_opt_candidate_descent(tour, inst, 40, 32, &out.stats, sa_table); }
                }
                if (tour.length < best_length - kImprovementEps) {
                    best_nodes = tour.nodes;
                    best_length = tour.length;
                }
            }
        }
        tour.set_tour(best_nodes, inst);
        // set_tour recomputes the length from scratch; restore the value the
        // search actually tracked so downstream comparisons (elite pool,
        // best-of-restarts) see bit-identical numbers to the old Tour-copy
        // path, whose length was the incrementally maintained one.
        tour.length = best_length;
        polish_tour(tour, inst, options, &out.stats, 2);
        if (!options.disable_subset_swap) {
            subset_swap_descent_impl(tour, inst, options.subset_swap_descent_passes, !options.disable_two_opt, &out.stats);
            polish_tour(tour, inst, options, &out.stats, 1);
        }
        if (!options.disable_pair_exchange) {
            subset_pair_exchange_descent(tour, inst, rrng, options, &out.stats, options.pair_exchange_passes);
        }
        if (!options.disable_ruin_recreate) {
            subset_ruin_recreate_lns(tour, inst, rrng, options, &out.stats, options.ruin_recreate_rounds);
        }
        if (options.oracle.cfg.inline_feedback) {
            (void)external_oracle_polish_tour(tour, inst, options.oracle, false, &out.stats, !options.disable_two_opt);
        }
        out.nodes = tour.nodes;
        out.record = make_restart_record(inst, out.nodes, tour.length, kind);
        return out;
    };

    int launched = 0;
    double best_outcome_len = std::numeric_limits<double>::infinity();
    std::vector<RestartOutcome> outcomes;
    for (;;) {
        int wave = 0;
        if (launched < indep_restarts) {
            // Waves never straddle the independent->kick boundary: every kick
            // must see the elite pool of the COMPLETED independent phase.
            wave = std::min(restart_threads, indep_restarts - launched);
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
            } else if (launched >= indep_restarts && !kick_snapshot_taken) {
                // Scheduled kicks: snapshot EXACTLY ONCE at the phase boundary,
                // so kick seeds are independent of restart_threads and the
                // deterministic-path guarantee survives.
                elite_seeds = elite.export_nodes();
                kick_snapshot_taken = true;
            }
        }
        if (wave == 1) {
            outcomes[0] = run_restart(launched);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(wave));
            for (int i = 0; i < wave; ++i) {
                workers.emplace_back([&outcomes, &run_restart, launched, i]() {
                    outcomes[static_cast<std::size_t>(i)] = run_restart(launched + i);
                });
            }
            for (std::thread& worker : workers) { worker.join(); }
        }
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
        auto elite_nodes = elite.export_nodes();
        const int top = std::min(static_cast<int>(elite_nodes.size()), std::max(0, options.path_relink_top));
        for (int i = 0; i < top; ++i) {
            for (int j = i + 1; j < top; ++j) {
                std::vector<int> rel_nodes;
                double rel_len = std::numeric_limits<double>::infinity();
                if (subset_path_relink_bidirectional(inst, elite_nodes[static_cast<std::size_t>(i)], elite_nodes[static_cast<std::size_t>(j)], rng, options, rel_nodes, rel_len, &result.stats)) {
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

    polish_elite_with_oracle(elite, inst, options, false, options.oracle.cfg.subset_top, &result.stats);
    const auto nodes = elite.export_nodes();
    if (!nodes.empty()) {
        result.tour.set_tour(nodes.front(), inst);
        final_polish_tour(result.tour, inst, options, &result.stats, 2);
    }
    result.stats.subset_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return result;
}

} // namespace aldous_tsp
