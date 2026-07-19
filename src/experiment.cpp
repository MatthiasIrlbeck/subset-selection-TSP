#include "aldous_tsp/experiment.hpp"

#include "aldous_tsp/lower_bound.hpp"

#include "aldous_tsp/solver.hpp"

#include "worker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace aldous_tsp {
namespace {

using Clock = std::chrono::steady_clock;

void append_restart_records(std::vector<RestartRecord>& destination,
                            const std::vector<RestartRecord>& source,
                            RestartSweep sweep) {
    destination.reserve(destination.size() + source.size());
    for (RestartRecord record : source) {
        record.sweep = sweep;
        destination.push_back(record);
    }
}

int best_restart_index(const std::vector<RestartRecord>& records) noexcept {
    int best_index = -1;
    double best_length = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < records.size(); ++i) {
        if (records[i].length < best_length - kImprovementEps) {
            best_length = records[i].length;
            best_index = static_cast<int>(i);
        }
    }
    return best_index;
}

// Two-nearest-neighbor lower bound on the optimal tour through a point set:
// L >= (1/2) sum_i (d1_i + d2_i), where d1_i, d2_i are the distances from i to
// its nearest and second-nearest neighbors. Valid because each city contributes
// two tour edges, each at least its nearest available edge (Held-Karp-style
// degree bound). Requires the instance KNN to hold at least two neighbors.
double two_nn_bound_from_knn(const Instance& inst) {
    double sum = 0.0;
    for (int i = 0; i < inst.N; ++i) {
        sum += inst.knn_d_at(i, 0) + inst.knn_d_at(i, 1);
    }
    return 0.5 * sum;
}

// Two-NN lower bound on a selected subset that lives on `base`'s domain. Builds
// a small KNN over just the subset points, using the parent's torus period via
// explicit_side so the minimum-image metric matches. Returns raw length units.
double two_nn_bound_subset(const Instance& base, const std::vector<int>& subset) {
    if (subset.size() < 3U) {
        return 0.0;
    }
    Instance sub;
    sub.periodic = base.periodic;
    if (base.periodic) {
        sub.explicit_side = base.side;
    }
    std::vector<Point> pts;
    pts.reserve(subset.size());
    for (int node : subset) {
        pts.push_back(base.points[static_cast<std::size_t>(node)]);
    }
    sub.set_points(std::move(pts));
    sub.build_knn(2, KnnBackend::GridExact);
    return two_nn_bound_from_knn(sub);
}

// Precise Monte-Carlo estimate of E[B_full] (the two-NN bound over N points on
// the same domain) -- the known mean of the control variate. KNN only, no
// solve, so far cheaper than one expensive instance. Uses an independent RNG
// stream so it does not overlap the measured instances.
void estimate_full_bound_expectation(const RunOptions& opt, ResultsDocument& doc) {
    int samples = opt.cv_mc_samples;
    // Cap so total cheap-KNN work stays bounded at very large N.
    const int cap = std::max(500, static_cast<int>(100000000LL / std::max(opt.N, 1)));
    samples = std::min(samples, cap);
    samples = std::max(samples, 2);
    double sum = 0.0;
    double sum2 = 0.0;
    for (int s = 0; s < samples; ++s) {
        Instance inst;
        inst.periodic = opt.periodic;
        Rng rng(make_stream_seed(static_cast<std::uint64_t>(opt.solver.seed),
                                 0xC0DEC0DEC0DEC0DEULL,
                                 static_cast<std::uint64_t>(s) ^ 0x9E3779B97F4A7C15ULL));
        inst.generate(opt.N, rng);
        inst.build_knn(2, KnnBackend::GridExact);
        const double b = two_nn_bound_from_knn(inst);
        sum += b;
        sum2 += b * b;
    }
    const double mean = sum / static_cast<double>(samples);
    const double var = std::max(0.0, sum2 / static_cast<double>(samples) - mean * mean);
    doc.full_bound_expectation = mean;
    doc.full_bound_expectation_stderr = std::sqrt(var / static_cast<double>(samples));
    doc.full_bound_expectation_samples = samples;
}

void record_knn_build_stats(SearchStats& stats, const KnnBuildInfo& info) {
    if (info.requested_backend == KnnBackend::GridExact) {
        ++stats.knn_requested_grid_instances;
    } else {
        ++stats.knn_requested_bruteforce_instances;
    }
    if (info.effective_backend == KnnBackend::GridExact) {
        ++stats.knn_effective_grid_instances;
        ++stats.knn_grid_cell_samples;
        stats.knn_grid_cells_sum += info.grid_cells;
        stats.knn_grid_cells_max = std::max(stats.knn_grid_cells_max, info.grid_cells);
        if (stats.knn_effective_grid_instances == 1U) {
            stats.grid_cell_effective_min = info.effective_cell_size;
            stats.grid_cell_effective_max = info.effective_cell_size;
        } else {
            stats.grid_cell_effective_min = std::min(stats.grid_cell_effective_min, info.effective_cell_size);
            stats.grid_cell_effective_max = std::max(stats.grid_cell_effective_max, info.effective_cell_size);
        }
        stats.grid_cell_effective_sum += info.effective_cell_size;
    } else {
        ++stats.knn_effective_bruteforce_instances;
    }
    if (info.brute_force_fallback) {
        ++stats.knn_bruteforce_fallback_instances;
    }
    if (info.grid_cell_capped) {
        ++stats.knn_grid_cell_capped_instances;
    }
}

struct CoreInstanceRunResult {
    int index = -1;
    double wall_seconds = 0.0;
    std::vector<double> values;
    std::vector<std::vector<RestartRecord>> restarts;  // per p, raw typed records
    std::vector<int> best_restarts;
    std::vector<int> restarts_used;
    std::vector<double> solve_seconds;
    std::vector<unsigned char> exact_optimal;
    SearchStats stats;
    KnnBuildInfo knn_info;
    // Control-variate bounds (raw length units; -1/empty when not requested).
    double full_bound = -1.0;
    std::vector<double> subset_bounds;
    std::vector<double> held_karp_bounds;
};

CoreInstanceRunResult run_one_instance_core(int index, const RunOptions& opt) {
    const auto start = Clock::now();
    CoreInstanceRunResult out;
    out.index = index;
    out.values.assign(opt.p_values.size(), std::numeric_limits<double>::quiet_NaN());

    Rng point_rng(make_stream_seed(static_cast<std::uint64_t>(opt.solver.seed), static_cast<std::uint64_t>(index), 0x243f6a8885a308d3ULL));
    Instance inst;
    inst.periodic = opt.periodic;
    inst.generate(opt.N, point_rng);

    const auto knn_start = Clock::now();
    inst.build_knn(opt.solver.knn_k, opt.solver.knn_backend, opt.solver.grid_cell);
    out.stats.knn_build_seconds += std::chrono::duration<double>(Clock::now() - knn_start).count();
    out.knn_info = inst.last_knn_build;
    record_knn_build_stats(out.stats, inst.last_knn_build);

    if (opt.control_variate && opt.solver.knn_k >= 2 && inst.N >= 3) {
        out.full_bound = two_nn_bound_from_knn(inst);
    }

    if (opt.solver.verify_knn_checks > 0) {
        Rng verify_rng(make_stream_seed(static_cast<std::uint64_t>(opt.solver.seed), static_cast<std::uint64_t>(index), 0x13198a2e03707344ULL));
        if (!inst.verify_knn(opt.solver.verify_knn_checks, verify_rng)) {
            throw std::runtime_error(
                "KNN verification failed for instance " + std::to_string(index));
        }
    }

    const std::size_t np = opt.p_values.size();
    out.restarts.resize(np);
    out.best_restarts.assign(np, -1);
    out.restarts_used.assign(np, 0);
    out.solve_seconds.assign(np, 0.0);
    out.exact_optimal.assign(np, 0U);
    std::vector<std::vector<int>> sweep_nodes(opt.second_sweep ? np : 0U);
    std::vector<double> sweep_lengths(opt.second_sweep ? np : 0U, std::numeric_limits<double>::infinity());
    std::vector<std::vector<int>> final_nodes((opt.control_variate || opt.held_karp) ? np : 0U);
    std::vector<int> warm;
    for (int pi = static_cast<int>(np) - 1; pi >= 0; --pi) {
        const double p = opt.p_values[static_cast<std::size_t>(pi)];
        const int k = std::max(3, std::min(opt.N, static_cast<int>(std::llround(p * static_cast<double>(opt.N)))));
        Rng local_rng(make_stream_seed(static_cast<std::uint64_t>(opt.solver.seed),
                                       static_cast<std::uint64_t>(index),
                                       mix_hash64(static_cast<std::uint64_t>(std::llround(p * 1000000.0)))));
        const auto solve_start = Clock::now();
        SolveResult solved = solve_subset(inst, k, local_rng, opt.solver, warm.empty() ? nullptr : &warm);
        const double solve_s = std::chrono::duration<double>(Clock::now() - solve_start).count();
        out.stats.add(solved.stats);
        out.values[static_cast<std::size_t>(pi)] = solved.tour.length / static_cast<double>(k);
        std::vector<RestartRecord>& records = out.restarts[static_cast<std::size_t>(pi)];
        append_restart_records(records, solved.restarts, RestartSweep::Primary);
        out.best_restarts[static_cast<std::size_t>(pi)] = best_restart_index(records);
        out.restarts_used[static_cast<std::size_t>(pi)] = static_cast<int>(records.size());
        out.solve_seconds[static_cast<std::size_t>(pi)] = solve_s;
        out.exact_optimal[static_cast<std::size_t>(pi)] =
            solved.exact_optimal ? 1U : 0U;
        warm = solved.tour.nodes;
        if (opt.control_variate || opt.held_karp) {
            final_nodes[static_cast<std::size_t>(pi)] = solved.tour.nodes;
        }
        if (opt.second_sweep) {
            sweep_nodes[static_cast<std::size_t>(pi)] = solved.tour.nodes;
            sweep_lengths[static_cast<std::size_t>(pi)] = solved.tour.length;
        }
    }

    if (opt.second_sweep && np > 1U && opt.solver.continuation_restarts > 0) {
        // Ascending sweep: seed each p from the (grown) best solution at the
        // next smaller p and keep the better result per p. Catches descending
        // rows that landed in a poor basin.
        std::vector<int> grow = sweep_nodes[0];
        for (std::size_t pi = 1; pi < np; ++pi) {
            const double p = opt.p_values[pi];
            const int k = std::max(3, std::min(opt.N, static_cast<int>(std::llround(p * static_cast<double>(opt.N)))));
            if (k >= opt.N) {
                // p = 1 rows delegate to the full TSP solver, which ignores
                // subset warm starts; skip and keep chaining upward.
                continue;
            }
            if (out.exact_optimal[pi] != 0U) {
                // The descending pass already proved the global optimum for
                // this cardinality. Re-running the exponential dynamic
                // program from a different warm start cannot improve it.
                grow = sweep_nodes[pi];
                continue;
            }
            Rng up_rng(make_stream_seed(static_cast<std::uint64_t>(opt.solver.seed),
                                        static_cast<std::uint64_t>(index),
                                        mix_hash64(static_cast<std::uint64_t>(std::llround(p * 1000000.0)) ^ 0x2b7e151628aed2a6ULL)));
            const auto solve_start = Clock::now();
            SubsetSolveRequest continuation_request;
            continuation_request.continuation_only = true;
            SolveResult solved = solve_subset(inst, k, up_rng, opt.solver,
                                              grow.empty() ? nullptr : &grow,
                                              continuation_request);
            out.solve_seconds[pi] += std::chrono::duration<double>(Clock::now() - solve_start).count();
            append_restart_records(out.restarts[pi], solved.restarts, RestartSweep::Secondary);
            out.restarts_used[pi] = static_cast<int>(out.restarts[pi].size());
            out.best_restarts[pi] = best_restart_index(out.restarts[pi]);
            out.stats.add(solved.stats);
            if (solved.exact_optimal) {
                out.exact_optimal[pi] = 1U;
            }
            if (solved.tour.length < sweep_lengths[pi] - 1e-12) {
                sweep_lengths[pi] = solved.tour.length;
                sweep_nodes[pi] = solved.tour.nodes;
                out.values[pi] = solved.tour.length / static_cast<double>(k);
                if (opt.control_variate || opt.held_karp) {
                    final_nodes[pi] = solved.tour.nodes;
                }
            }
            grow = sweep_nodes[pi];
        }
    }

    if (opt.control_variate) {
        out.subset_bounds.assign(np, -1.0);
        for (std::size_t pi = 0; pi < np; ++pi) {
            if (!final_nodes[pi].empty()) {
                out.subset_bounds[pi] = two_nn_bound_subset(inst, final_nodes[pi]);
            }
        }
    }

    if (opt.held_karp) {
        out.held_karp_bounds.assign(np, -1.0);
        for (std::size_t pi = 0; pi < np; ++pi) {
            if (final_nodes[pi].size() >= 3U) {
                const double ub = cycle_length(inst, final_nodes[pi]);
                const HeldKarpBound hk = held_karp_bound(inst, final_nodes[pi], ub, opt.hk_iterations);
                if (hk.computed) {
                    out.held_karp_bounds[pi] = hk.bound;
                }
            }
        }
    }

    out.wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return out;
}

int resolve_thread_count(int requested, int instances) {
    const unsigned hw = std::thread::hardware_concurrency();
    int threads = requested;
    if (threads <= 0) {
        threads = hw == 0U ? 1 : static_cast<int>(hw);
    }
    return std::max(1, std::min(threads, std::max(1, instances)));
}

} // namespace

ExperimentRunner::ExperimentRunner(RunOptions options) : options_(std::move(options)) {
    if (options_.p_values.empty()) {
        options_.p_values = default_p_values();
    }
    // The public API accepts RunOptions directly, bypassing the CLI validator.
    // Canonicalize here as well so sweep direction and per-p seed identities do
    // not depend on caller ordering or duplicate entries.
    std::sort(options_.p_values.begin(), options_.p_values.end());
    options_.p_values.erase(
        std::unique(options_.p_values.begin(), options_.p_values.end()),
        options_.p_values.end());
    const int requested_threads = options_.threads;
    options_.threads = resolve_thread_count(requested_threads, options_.instances);
    if (options_.solver.restart_threads <= 0) {
        // Auto: split the requested (or hardware) thread budget across the
        // instance workers; leftover parallelism goes to restart waves.
        const unsigned hw = std::thread::hardware_concurrency();
        const int budget = requested_threads <= 0 ? (hw == 0U ? 1 : static_cast<int>(hw)) : requested_threads;
        options_.solver.restart_threads = std::max(1, budget / std::max(1, options_.threads));
    }
}

ResultsDocument ExperimentRunner::run(const ExperimentProgressCallback& progress) const {
    RunOptions opt = options_;
    opt.threads = resolve_thread_count(opt.threads, opt.instances);

    ResultsDocument doc;
    doc.N = opt.N;
    doc.instances_target = opt.instances;
    doc.threads = opt.threads;
    doc.options = opt;
    doc.p_values = opt.p_values;

    const auto global_start = Clock::now();
    std::vector<detail::WorkerOutcome<CoreInstanceRunResult>> outcomes;
    int progress_completed = 0;
    const detail::WorkerSummary worker_summary =
        detail::run_parallel_work_queue<CoreInstanceRunResult>(
            opt.instances,
            opt.threads,
            outcomes,
            [&](int index) {
                return run_one_instance_core(index, opt);
            },
            [&](int index, const CoreInstanceRunResult& result) {
                ++progress_completed;
                if (!progress) {
                    return;
                }
                const double elapsed =
                    std::chrono::duration<double>(Clock::now() - global_start).count();
                ExperimentProgress event;
                event.completed = progress_completed;
                event.total = opt.instances;
                event.instance_index = index;
                event.instance_seconds = result.wall_seconds;
                event.elapsed_seconds = elapsed;
                event.eta_seconds = progress_completed > 0
                    ? elapsed / static_cast<double>(progress_completed)
                        * static_cast<double>(opt.instances - progress_completed)
                    : 0.0;
                progress(event);
            });

    doc.wall_seconds = std::chrono::duration<double>(Clock::now() - global_start).count();
    doc.instances_done = worker_summary.succeeded;

    std::vector<std::vector<double>> by_p(opt.p_values.size());
    for (std::vector<double>& values : by_p) {
        values.reserve(static_cast<std::size_t>(opt.instances));
    }
    std::vector<int> best_restart_max(opt.p_values.size(), -1);
    std::vector<int> executed_restarts_max(opt.p_values.size(), -1);
    std::vector<double> solve_seconds_total(opt.p_values.size(), 0.0);
    std::vector<int> exact_optimal_instances(opt.p_values.size(), 0);
    // Control-variate arrays, aligned by instance order per p.
    std::vector<std::vector<double>> cv_sub(opt.p_values.size());
    std::vector<std::vector<double>> cv_full(opt.p_values.size());
    std::vector<std::vector<double>> cv_hk(opt.p_values.size());

    for (const detail::WorkerOutcome<CoreInstanceRunResult>& outcome : outcomes) {
        if (outcome.state != detail::WorkerState::Success || !outcome.value.has_value()) {
            continue;
        }
        const CoreInstanceRunResult& r = *outcome.value;
        doc.stats.add(r.stats);
        for (std::size_t pi = 0; pi < opt.p_values.size(); ++pi) {
            by_p[pi].push_back(r.values[pi]);
            if (pi < r.best_restarts.size()) {
                best_restart_max[pi] = std::max(best_restart_max[pi], r.best_restarts[pi]);
            }
            if (pi < r.restarts_used.size()) {
                executed_restarts_max[pi] = std::max(executed_restarts_max[pi], r.restarts_used[pi]);
            }
            if (pi < r.solve_seconds.size()) {
                solve_seconds_total[pi] += r.solve_seconds[pi];
            }
            if (pi < r.exact_optimal.size() && r.exact_optimal[pi] != 0U) {
                ++exact_optimal_instances[pi];
            }
            if (opt.control_variate) {
                const int k = std::max(3, std::min(opt.N, static_cast<int>(std::llround(opt.p_values[pi] * static_cast<double>(opt.N)))));
                const double kd = static_cast<double>(k);
                if (pi < r.subset_bounds.size() && r.subset_bounds[pi] >= 0.0) {
                    cv_sub[pi].push_back(r.subset_bounds[pi] / kd);
                }
                if (r.full_bound >= 0.0) {
                    cv_full[pi].push_back(r.full_bound / kd);
                }
            }
            if (opt.held_karp) {
                const int k = std::max(3, std::min(opt.N, static_cast<int>(std::llround(opt.p_values[pi] * static_cast<double>(opt.N)))));
                if (pi < r.held_karp_bounds.size() && r.held_karp_bounds[pi] >= 0.0) {
                    cv_hk[pi].push_back(r.held_karp_bounds[pi] / static_cast<double>(k));
                }
            }
        }
        if (opt.include_instance_rows) {
            InstanceResultRow row;
            row.ok = true;
            row.index = r.index;
            row.wall_seconds = r.wall_seconds;
            row.values = r.values;
            row.stats = r.stats;
            row.knn_info = r.knn_info;
            row.full_bound = r.full_bound;
            row.p_results.reserve(opt.p_values.size());
            for (std::size_t pi = 0; pi < opt.p_values.size(); ++pi) {
                InstancePValueRow pvrow;
                pvrow.p = opt.p_values[pi];
                pvrow.key = p_value_key(opt.p_values[pi]);
                pvrow.k = std::max(3, std::min(opt.N, static_cast<int>(std::llround(opt.p_values[pi] * static_cast<double>(opt.N)))));
                pvrow.value = r.values[pi];
                pvrow.best_restart = pi < r.best_restarts.size() ? r.best_restarts[pi] : -1;
                pvrow.executed_restarts = pi < r.restarts_used.size() ? r.restarts_used[pi] : 0;
                pvrow.solve_seconds = pi < r.solve_seconds.size() ? r.solve_seconds[pi] : 0.0;
                pvrow.exact_optimal =
                    pi < r.exact_optimal.size() && r.exact_optimal[pi] != 0U;
                if (pi < r.restarts.size()) { pvrow.restarts = r.restarts[pi]; }
                if (pi < r.subset_bounds.size() && r.subset_bounds[pi] >= 0.0) {
                    pvrow.conditional_two_nn_bound =
                        r.subset_bounds[pi] / static_cast<double>(pvrow.k);
                }
                if (pi < r.held_karp_bounds.size() && r.held_karp_bounds[pi] >= 0.0) {
                    pvrow.conditional_held_karp_bound =
                        r.held_karp_bounds[pi] / static_cast<double>(pvrow.k);
                }
                row.p_results.push_back(std::move(pvrow));
            }
            doc.instance_rows.push_back(std::move(row));
        }
    }

    if (opt.control_variate) {
        estimate_full_bound_expectation(opt, doc);
    }

    for (std::size_t pi = 0; pi < opt.p_values.size(); ++pi) {
        PValueSummary summary = summarize_p_values(opt.N, opt.p_values[pi], by_p[pi]);
        summary.best_restart_max = best_restart_max[pi];
        summary.executed_restarts_max = executed_restarts_max[pi];
        summary.solve_seconds_total = solve_seconds_total[pi];
        summary.exact_optimal_instances = exact_optimal_instances[pi];
        if (opt.control_variate) {
            summary.has_control_variate = true;
            const int k = std::max(3, std::min(opt.N, static_cast<int>(std::llround(opt.p_values[pi] * static_cast<double>(opt.N)))));
            const std::vector<double>& sub = cv_sub[pi];
            if (!sub.empty()) {
                double s = 0.0;
                for (double v : sub) { s += v; }
                summary.conditional_two_nn_bound_mean = s / static_cast<double>(sub.size());
                summary.conditional_two_nn_gap_mean =
                    summary.mean - summary.conditional_two_nn_bound_mean;
            }
            // Full-set control variate with known mean E[B_full]/k.
            const std::vector<double>& y = by_p[pi];
            const std::vector<double>& x = cv_full[pi];
            summary.cv_mean = summary.mean;
            summary.cv_stderr = summary.stderr_value;
            if (doc.full_bound_expectation >= 0.0 && x.size() == y.size() && y.size() >= 2U) {
                const double mu_x = doc.full_bound_expectation / static_cast<double>(k);
                const auto M = static_cast<double>(y.size());
                double my = 0.0;
                double mx = 0.0;
                for (double v : y) { my += v; }
                for (double v : x) { mx += v; }
                my /= M;
                mx /= M;
                double sxx = 0.0;
                double sxy = 0.0;
                double syy = 0.0;
                for (std::size_t i = 0; i < y.size(); ++i) {
                    const double dx = x[i] - mx;
                    const double dy = y[i] - my;
                    sxx += dx * dx;
                    sxy += dx * dy;
                    syy += dy * dy;
                }
                if (sxx > 0.0) {
                    const double lambda = sxy / sxx;
                    summary.cv_mean = my - lambda * (mx - mu_x);
                    double rss = 0.0;
                    for (std::size_t i = 0; i < y.size(); ++i) {
                        const double resid = (y[i] - lambda * x[i]) - (my - lambda * mx);
                        rss += resid * resid;
                    }
                    const double resid_var = rss / (M - 1.0);
                    summary.cv_stderr = std::sqrt(std::max(0.0, resid_var) / M);
                    const double rho2 = (syy > 0.0) ? (sxy * sxy) / (sxx * syy) : 0.0;
                    summary.cv_variance_reduction = std::min(std::max(rho2, 0.0), 1.0);
                }
            }
        }
        if (opt.held_karp) {
            summary.has_held_karp = true;
            const std::vector<double>& hk = cv_hk[pi];
            if (!hk.empty()) {
                double s = 0.0;
                for (double v : hk) { s += v; }
                summary.conditional_held_karp_bound_mean =
                    s / static_cast<double>(hk.size());
                summary.conditional_held_karp_gap_mean =
                    summary.mean - summary.conditional_held_karp_bound_mean;
            }
        }
        doc.summary[p_value_key(opt.p_values[pi])] = std::move(summary);
    }

    return doc;
}

} // namespace aldous_tsp
