#include "aldous_tsp/experiment.hpp"
#include "aldous_tsp/validation.hpp"

#include "aldous_tsp/lower_bound.hpp"
#include "aldous_tsp/memory.hpp"

#include "aldous_tsp/solver.hpp"

#include "worker.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
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
    const int operation_cap = opt.cv_max_point_ops / std::max(opt.N, 1);
    if (operation_cap < 2) {
        throw std::invalid_argument(
            "control-reference work budget permits fewer than two samples");
    }
    const int samples = std::min(opt.cv_mc_samples, operation_cap);
    double mean = 0.0;
    double m2 = 0.0;
    for (int sample = 0; sample < samples; ++sample) {
        Rng rng(make_stream_seed(static_cast<std::uint64_t>(effective_point_seed(opt)),
                                 0xC0DEC0DEC0DEC0DEULL,
                                 static_cast<std::uint64_t>(sample)
                                     ^ 0x9E3779B97F4A7C15ULL));
        PreparedInstance prepared = InstanceBuilder()
            .periodic(opt.periodic)
            .generate(opt.N, rng)
            .build(2, KnnBackend::GridExact);
        const double value = two_nn_bound_from_knn(prepared.instance());
        const double delta = value - mean;
        mean += delta / static_cast<double>(sample + 1);
        m2 += delta * (value - mean);
    }
    const double variance = samples > 1
        ? std::max(0.0, m2 / static_cast<double>(samples - 1)) : 0.0;
    doc.full_bound_expectation = mean;
    doc.full_bound_expectation_stddev = std::sqrt(variance);
    doc.full_bound_expectation_stderr =
        std::sqrt(variance / static_cast<double>(samples));
    doc.full_bound_expectation_samples = samples;
    doc.full_bound_expectation_point_operations =
        static_cast<std::uint64_t>(samples)
        * static_cast<std::uint64_t>(std::max(opt.N, 0));
}

struct CrossFittedControlVariate {
    double mean = 0.0;
    double total_stderr = 0.0;
    double sampling_stderr = 0.0;
    double reference_stderr = 0.0;
    double lambda = 0.0;
    double lambda_fold0 = 0.0;
    double lambda_fold1 = 0.0;
    double variance_reduction = 0.0;
    std::vector<double> adjusted;
    std::vector<double> applied_lambda;
};

double fitted_control_coefficient(const std::vector<double>& y,
                                  const std::vector<double>& x,
                                  const std::vector<std::uint64_t>& ids,
                                  const int excluded_fold,
                                  bool& available) {
    double mean_y = 0.0;
    double mean_x = 0.0;
    std::size_t count = 0U;
    for (std::size_t i = 0; i < y.size(); ++i) {
        if (excluded_fold >= 0
            && static_cast<int>(ids[i] & 1U) == excluded_fold) {
            continue;
        }
        ++count;
        mean_y += (y[i] - mean_y) / static_cast<double>(count);
        mean_x += (x[i] - mean_x) / static_cast<double>(count);
    }
    if (count < 2U) {
        available = false;
        return 0.0;
    }
    double sxx = 0.0;
    double sxy = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        if (excluded_fold >= 0
            && static_cast<int>(ids[i] & 1U) == excluded_fold) {
            continue;
        }
        const double dx = x[i] - mean_x;
        sxx += dx * dx;
        sxy += dx * (y[i] - mean_y);
    }
    available = std::isfinite(sxx) && std::isfinite(sxy) && sxx > 0.0;
    return available ? sxy / sxx : 0.0;
}

CrossFittedControlVariate estimate_cross_fitted_control_variate(
    const std::vector<double>& y,
    const std::vector<double>& x,
    const std::vector<std::uint64_t>& ids,
    const double reference_mean,
    const double reference_stderr) {
    CrossFittedControlVariate result;
    if (y.size() != x.size() || y.size() != ids.size() || y.empty()) {
        return result;
    }
    bool global_available = false;
    const double global = fitted_control_coefficient(
        y, x, ids, -1, global_available);
    bool fold0_available = false;
    bool fold1_available = false;
    // The coefficient applied to fold 0 is fitted using fold 1, and vice versa.
    result.lambda_fold0 = fitted_control_coefficient(
        y, x, ids, 0, fold0_available);
    result.lambda_fold1 = fitted_control_coefficient(
        y, x, ids, 1, fold1_available);
    if (!fold0_available) {
        result.lambda_fold0 = global_available ? global : 0.0;
    }
    if (!fold1_available) {
        result.lambda_fold1 = global_available ? global : 0.0;
    }

    result.adjusted.resize(y.size());
    result.applied_lambda.resize(y.size());
    double mean = 0.0;
    double mean_lambda = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double lambda = (ids[i] & 1U) == 0U
            ? result.lambda_fold0 : result.lambda_fold1;
        result.applied_lambda[i] = lambda;
        result.adjusted[i] = y[i] - lambda * (x[i] - reference_mean);
        mean += (result.adjusted[i] - mean) / static_cast<double>(i + 1U);
        mean_lambda += (lambda - mean_lambda) / static_cast<double>(i + 1U);
    }
    result.mean = mean;
    result.lambda = mean_lambda;

    double adjusted_m2 = 0.0;
    double raw_mean = 0.0;
    double raw_m2 = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double adjusted_delta = result.adjusted[i] - result.mean;
        adjusted_m2 += adjusted_delta * adjusted_delta;
        const double raw_delta = y[i] - raw_mean;
        raw_mean += raw_delta / static_cast<double>(i + 1U);
        raw_m2 += raw_delta * (y[i] - raw_mean);
    }
    const double count = static_cast<double>(y.size());
    const double adjusted_variance = y.size() > 1U
        ? adjusted_m2 / static_cast<double>(y.size() - 1U) : 0.0;
    const double raw_variance = y.size() > 1U
        ? raw_m2 / static_cast<double>(y.size() - 1U) : 0.0;
    result.sampling_stderr = std::sqrt(
        std::max(0.0, adjusted_variance) / count);
    result.reference_stderr = std::fabs(mean_lambda) * reference_stderr;
    result.total_stderr = std::hypot(
        result.sampling_stderr, result.reference_stderr);
    result.variance_reduction = raw_variance > 0.0
        ? std::clamp(1.0 - adjusted_variance / raw_variance, 0.0, 1.0)
        : 0.0;
    return result;
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
    std::uint64_t replicate_id = 0;
    std::uint64_t point_stream_id = 0;
    std::uint64_t search_stream_id = 0;
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

CoreInstanceRunResult run_one_instance_core(
    int index, const RunOptions& opt,
    const std::shared_ptr<ConcurrencyLimiter>& held_karp_limiter) {
    const auto start = Clock::now();
    CoreInstanceRunResult out;
    out.index = index;
    out.replicate_id = static_cast<std::uint64_t>(opt.replicate_offset)
        + static_cast<std::uint64_t>(index);
    out.point_stream_id = make_stream_seed(
        static_cast<std::uint64_t>(effective_point_seed(opt)),
        out.replicate_id,
        0x243f6a8885a308d3ULL);
    out.search_stream_id = make_stream_seed(
        static_cast<std::uint64_t>(effective_search_seed(opt)),
        out.replicate_id,
        0x13198a2e03707344ULL);
    out.values.assign(opt.p_values.size(), std::numeric_limits<double>::quiet_NaN());

    Rng point_rng(out.point_stream_id);
    const auto knn_start = Clock::now();
    PreparedInstance prepared = InstanceBuilder()
        .periodic(opt.periodic)
        .generate(opt.N, point_rng)
        .build(opt.solver.knn_k, opt.solver.knn_backend, opt.solver.grid_cell);
    const Instance& inst = prepared.instance();
    out.stats.knn_build_seconds += std::chrono::duration<double>(Clock::now() - knn_start).count();
    out.knn_info = inst.last_knn_build;
    record_knn_build_stats(out.stats, inst.last_knn_build);

    if (opt.control_variate && opt.solver.knn_k >= 2 && inst.N >= 3) {
        out.full_bound = two_nn_bound_from_knn(inst);
    }

    if (opt.solver.verify_knn_checks > 0) {
        Rng verify_rng(make_stream_seed(out.point_stream_id,
                                       0x9e3779b97f4a7c15ULL,
                                       0x94d049bb133111ebULL));
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
        Rng local_rng(make_stream_seed(out.search_stream_id,
                                       out.replicate_id,
                                       mix_hash64(static_cast<std::uint64_t>(std::llround(p * 1000000.0)))));
        const auto solve_start = Clock::now();
        SolveResult solved = solve_subset(
            prepared, k, local_rng, opt.solver,
            warm.empty() ? nullptr : &warm);
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
            Rng up_rng(make_stream_seed(out.search_stream_id,
                                        out.replicate_id,
                                        mix_hash64(static_cast<std::uint64_t>(std::llround(p * 1000000.0)) ^ 0x2b7e151628aed2a6ULL)));
            const auto solve_start = Clock::now();
            SubsetSolveRequest continuation_request;
            continuation_request.continuation_only = true;
            SolveResult solved = solve_subset(prepared, k, up_rng, opt.solver,
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
                std::optional<ConcurrencyLimiter::Permit> permit;
                if (held_karp_limiter != nullptr) {
                    permit.emplace(held_karp_limiter->acquire());
                }
                const HeldKarpBound hk = held_karp_bound(
                    prepared, final_nodes[pi], ub, opt.hk_iterations);
                if (hk.computed) {
                    out.held_karp_bounds[pi] = hk.bound;
                }
            }
        }
    }

    out.wall_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return out;
}

} // namespace

ExperimentRunner::ExperimentRunner(RunOptions options)
    : requested_threads_(options.threads), options_(std::move(options)) {
    require_valid_run_options(options_);
}

ResultsDocument ExperimentRunner::run(const ExperimentProgressCallback& progress) const {
    RunOptions opt = options_;
    const MemoryPlan memory_plan = estimate_experiment_memory(opt, requested_threads_);
    opt.threads = memory_plan.effective_threads;
    const std::shared_ptr<ConcurrencyLimiter> held_karp_limiter =
        opt.held_karp
            ? std::make_shared<ConcurrencyLimiter>(memory_plan.held_karp_concurrency)
            : nullptr;
    if (opt.solver.oracle.resolved != ResolvedOracleMode::None) {
        opt.solver.oracle.concurrency_limiter =
            std::make_shared<ConcurrencyLimiter>(memory_plan.oracle_concurrency);
    }

    ResultsDocument doc;
    doc.N = opt.N;
    doc.instances_target = opt.instances;
    doc.threads = opt.threads;
    doc.memory_plan = memory_plan;
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
                return run_one_instance_core(index, opt, held_karp_limiter);
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

    doc.solver_wall_seconds =
        std::chrono::duration<double>(Clock::now() - global_start).count();
    doc.instances_done = worker_summary.succeeded;

    if (opt.control_variate) {
        const auto reference_start = Clock::now();
        estimate_full_bound_expectation(opt, doc);
        doc.control_reference_seconds =
            std::chrono::duration<double>(Clock::now() - reference_start).count();
    }
    const auto aggregation_start = Clock::now();

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
    std::vector<std::vector<std::uint64_t>> cv_ids(opt.p_values.size());
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
                    cv_ids[pi].push_back(r.replicate_id);
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
            row.replicate_id = r.replicate_id;
            row.point_stream_id = r.point_stream_id;
            row.search_stream_id = r.search_stream_id;
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

    std::vector<CrossFittedControlVariate> cv_estimates(opt.p_values.size());
    for (std::size_t pi = 0; pi < opt.p_values.size(); ++pi) {
        PValueSummary summary = summarize_p_values(opt.N, opt.p_values[pi], by_p[pi]);
        summary.best_restart_max = best_restart_max[pi];
        summary.executed_restarts_max = executed_restarts_max[pi];
        summary.solve_seconds_total = solve_seconds_total[pi];
        summary.exact_optimal_instances = exact_optimal_instances[pi];
        if (opt.control_variate) {
            summary.has_control_variate = true;
            const int k = std::max(
                3,
                std::min(
                    opt.N,
                    static_cast<int>(std::llround(
                        opt.p_values[pi] * static_cast<double>(opt.N)))));
            const std::vector<double>& sub = cv_sub[pi];
            if (!sub.empty()) {
                double bound_mean = 0.0;
                for (std::size_t i = 0; i < sub.size(); ++i) {
                    bound_mean += (sub[i] - bound_mean)
                        / static_cast<double>(i + 1U);
                }
                summary.conditional_two_nn_bound_mean = bound_mean;
                summary.conditional_two_nn_gap_mean = summary.mean - bound_mean;
            }

            summary.cv_mean = summary.mean;
            summary.cv_stderr = summary.stderr_value;
            summary.cv_sampling_stderr = summary.stderr_value;
            if (doc.full_bound_expectation >= 0.0
                && cv_full[pi].size() == by_p[pi].size()
                && cv_ids[pi].size() == by_p[pi].size()
                && by_p[pi].size() >= 2U) {
                const double kd = static_cast<double>(k);
                cv_estimates[pi] = estimate_cross_fitted_control_variate(
                    by_p[pi], cv_full[pi], cv_ids[pi],
                    doc.full_bound_expectation / kd,
                    doc.full_bound_expectation_stderr / kd);
                const CrossFittedControlVariate& estimate = cv_estimates[pi];
                summary.cv_mean = estimate.mean;
                summary.cv_stderr = estimate.total_stderr;
                summary.cv_sampling_stderr = estimate.sampling_stderr;
                summary.cv_reference_stderr = estimate.reference_stderr;
                summary.cv_lambda = estimate.lambda;
                summary.cv_lambda_fold0 = estimate.lambda_fold0;
                summary.cv_lambda_fold1 = estimate.lambda_fold1;
                summary.cv_variance_reduction = estimate.variance_reduction;
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

    if (opt.control_variate && opt.include_instance_rows) {
        for (std::size_t row_index = 0; row_index < doc.instance_rows.size(); ++row_index) {
            InstanceResultRow& row = doc.instance_rows[row_index];
            for (std::size_t pi = 0; pi < row.p_results.size(); ++pi) {
                CrossFittedControlVariate& estimate = cv_estimates[pi];
                if (row_index >= estimate.adjusted.size()) {
                    continue;
                }
                InstancePValueRow& p_row = row.p_results[pi];
                if (p_row.k > 0 && row.full_bound >= 0.0) {
                    p_row.control_variate_x =
                        row.full_bound / static_cast<double>(p_row.k);
                    p_row.cv_adjusted_value = estimate.adjusted[row_index];
                    p_row.cv_lambda = estimate.applied_lambda[row_index];
                }
            }
        }
    }

    doc.aggregation_seconds =
        std::chrono::duration<double>(Clock::now() - aggregation_start).count();
    doc.experiment_wall_seconds =
        std::chrono::duration<double>(Clock::now() - global_start).count();
    doc.wall_seconds = doc.experiment_wall_seconds;
    return doc;
}

} // namespace aldous_tsp
