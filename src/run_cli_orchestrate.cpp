#include "run_cli_internal.hpp"
#include "tsp_solver.hpp"
#include "subset_solver.hpp"

namespace run_cli_detail {

InstanceResult run_one_instance(int ii,const RunConfig& cfg,const std::vector<double>& pv){
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    InstanceResult out;
    out.index = ii;
    out.fvals.assign(pv.size(), std::numeric_limits<double>::quiet_NaN());

    std::ostringstream oss;
    if(cfg.verbose_p){
        oss << "══════════════════════════════════════════════════════════════════════════\n";
        oss << "  Instance " << (ii + 1) << " / " << cfg.n_inst << "\n";
        oss << "══════════════════════════════════════════════════════════════════════════\n";
    }

    Rng point_rng;
    point_rng.seed(make_stream_seed((uint64_t)cfg.seed, (uint64_t)ii, 0x243f6a8885a308d3ULL));
    Instance ins;
    ins.generate(cfg.N, point_rng);
    ins.build_knn(cfg.knn_k, cfg.grid_cell);

    if(cfg.verify_knn_checks > 0){
        Rng verify_rng;
        verify_rng.seed(make_stream_seed((uint64_t)cfg.seed, (uint64_t)ii, 0x13198a2e03707344ULL));
        bool ok = ins.verify_knn(cfg.verify_knn_checks, verify_rng);
        if(cfg.verbose_p){
            oss << "  KNN verify: " << (ok ? "OK" : "FAILED") << " (" << cfg.verify_knn_checks << " sampled queries)\n";
        }
        if(!ok){
            out.ok = false;
            out.log = oss.str();
            out.wall_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
            return out;
        }
    }

    ChainState low_chain, mid_chain, high_chain;

    for(int pi=(int)pv.size()-1; pi>=0; --pi){
        double p = pv[pi];
        int k = std::max(3, (int)std::round(p * cfg.N));
        auto tp0 = Clock::now();

        Tour result;
        std::vector<std::vector<int>> elite_here;
        bool full_tsp = (k >= cfg.N);
        RunRegime regime = select_run_regime(cfg.mode, p, full_tsp);

        ChainState* chain = nullptr;
        bool enable_geom_master = false;
        switch(regime){
            case RunRegime::FULL: chain = &high_chain; break;
            case RunRegime::LOW: chain = &low_chain; enable_geom_master = true; break;
            case RunRegime::MID: chain = &mid_chain; break;
            case RunRegime::HIGH: chain = &high_chain; break;
        }

        Rng local_rng;
        local_rng.seed(make_regime_p_seed((uint64_t)cfg.seed, (uint64_t)ii, regime, p));

        if(full_tsp){
            solve_tsp(ins, local_rng, result, cfg.oracle, cfg.tsp_restarts, cfg.tsp_ils, cfg.tsp_patience, &elite_here);
            // seed the non-low chains from the exact same p=1 tour family
            update_chain_state(high_chain, result, elite_here, k);
            update_chain_state(mid_chain, result, elite_here, k);
        } else if(regime == RunRegime::HIGH){
            int dk = (chain->prev_k >= 0) ? std::abs(chain->prev_k - k) : k;
            int scaled = (int)std::min<int64_t>(((int64_t)cfg.sa_iters * (int64_t)dk) / std::max(k, 1), (int64_t)std::numeric_limits<int>::max());
            int ai = std::min(cfg.sa_iters, std::max(std::min(cfg.sa_iters, 1500), scaled));
            int ar = cfg.restarts;
            solve_subset_highp_delete(ins, k, local_rng, result, ai, ar, &chain->prev_elite, &elite_here);
            update_chain_state(*chain, result, elite_here, k);
        } else {
            int dk = (chain->prev_k >= 0) ? std::abs(chain->prev_k - k) : k;
            int scaled = (int)std::min<int64_t>(((int64_t)cfg.sa_iters * (int64_t)dk) / std::max(k, 1), (int64_t)std::numeric_limits<int>::max());
            int ai = std::min(cfg.sa_iters, std::max(std::min(cfg.sa_iters, 1500), scaled));
            int ar = cfg.restarts;
            std::vector<std::vector<int>> warm_pool;
            std::vector<std::vector<int>> guide_pool;
            if(!chain->prev_elite.empty()) warm_pool = make_warm_pool_from_elite(ins, chain->prev_elite, k, local_rng, &guide_pool);
            solve_subset(ins, k, local_rng, result, cfg.oracle, ai, ar,
                         warm_pool.empty() ? nullptr : &warm_pool,
                         guide_pool.empty() ? nullptr : &guide_pool,
                         &elite_here,
                         enable_geom_master);
            update_chain_state(*chain, result, elite_here, k);
        }

        Tour reported;
        OracleStats pstats;
        int posthoc_top = full_tsp ? cfg.oracle.cfg.tsp_top : cfg.oracle.cfg.subset_top;
        maybe_posthoc_oracle_select_best(ins, result, elite_here, cfg.oracle, full_tsp, posthoc_top, reported, &pstats);
        out.oracle.add(pstats);
        double dt = std::chrono::duration<double>(Clock::now() - tp0).count();
        double fv = reported.length / k;
        out.fvals[pi] = fv;
        int ridx = run_regime_index(regime);
        out.regime_counts[(size_t)ridx] += 1;
        out.regime_seconds[(size_t)ridx] += dt;

        if(cfg.verbose_p){
            char bhh_tag[48] = "";
            if(p >= 0.999) std::snprintf(bhh_tag, sizeof(bhh_tag), "  ◀ BHH≈%.3f", BHH_REFERENCE);
            char line[384];
            double imp = result.length - reported.length;
            std::snprintf(line, sizeof(line), "  p=%.3f k=%4d len=%9.2f rep=%9.2f f=%.4f [%6.1fs]  regime=%s%s%s\n",
                          p, k, result.length, reported.length, fv, dt, run_regime_name(regime),
                          imp > IMPROVEMENT_EPS ? "  oracle+" : "",
                          bhh_tag);
            oss << line;
        }
    }

    out.wall_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
    if(cfg.verbose_p){
        char tail[320];
        std::snprintf(tail, sizeof(tail),
                      "  oracle: calls=%llu solved=%llu improved=%llu gain=%.6f  [tsp %llu/%llu, subset %llu/%llu]\n"
                      "  regimes: full=%d/%.1fs low=%d/%.1fs mid=%d/%.1fs high=%d/%.1fs\n"
                      "  ── instance wall: %.1fs ──\n\n",
                      (unsigned long long)out.oracle.calls,
                      (unsigned long long)out.oracle.solved,
                      (unsigned long long)out.oracle.improved,
                      out.oracle.exact_gain,
                      (unsigned long long)out.oracle.tsp_improved,
                      (unsigned long long)out.oracle.tsp_calls,
                      (unsigned long long)out.oracle.subset_improved,
                      (unsigned long long)out.oracle.subset_calls,
                      out.regime_counts[0], out.regime_seconds[0],
                      out.regime_counts[1], out.regime_seconds[1],
                      out.regime_counts[2], out.regime_seconds[2],
                      out.regime_counts[3], out.regime_seconds[3],
                      out.wall_seconds);
        oss << tail;
        out.log = oss.str();
    }
    return out;
}

int run_cli_execute(const RunConfig& cfg,
                    const std::vector<double>& pv,
                    const std::string& output_path,
                    bool force_output,
                    const ResultsMetadata& results_meta){
    {
        std::error_code ec;
        bool exists = std::filesystem::exists(output_path, ec);
        if(ec){
            std::fprintf(stderr, "Failed to inspect output path %s: %s\n", output_path.c_str(), ec.message().c_str());
            return 1;
        }
        if(exists && !force_output){
            std::fprintf(stderr, "Refusing to overwrite existing output file %s (use --force or --output <newfile>)\n", output_path.c_str());
            return 1;
        }
    }

    std::printf("========================================================================\n");
    std::printf("  ALDOUS SUBSET-SELECTION TSP · %s\n", banner_title_for_mode(cfg.mode).c_str());
    std::printf("  Exact Euclidean model · coordinate backend · exact grid KNN · edge cache · %s\n", banner_desc_for_mode(cfg.mode).c_str());
    std::printf("========================================================================\n");
    std::printf("  N=%d, %d p-vals, %d inst, %d subset restarts, SA=%d, knn=%d, threads=%d\n",
                cfg.N, (int)pv.size(), cfg.n_inst, cfg.restarts, cfg.sa_iters, cfg.knn_k, cfg.threads);
    std::printf("  TSP restarts=%d, ILS=%d, patience=%d%s\n",
                cfg.tsp_restarts, cfg.tsp_ils, cfg.tsp_patience, cfg.verbose_p ? ", verbose-p" : "");
    std::printf("  Mode: %s\n", solver_mode_name(cfg.mode));
    std::printf("  Mode status: %s\n", solver_mode_stability_label(cfg.mode));
    if(solver_mode_is_experimental(cfg.mode)){
        std::printf("  NOTE: balanced remains the default mode; this path is experimental.\n");
    }
    std::printf("  External oracle: %s\n\n", cfg.oracle.status.c_str());
    std::fflush(stdout);

    using Clock = std::chrono::steady_clock;
    auto t_global = Clock::now();
    auto elapsed = [&]()->double{ return std::chrono::duration<double>(Clock::now() - t_global).count(); };

    std::vector<InstanceResult> results(cfg.n_inst);
    std::atomic<int> next_idx{0};
    std::atomic<int> done{0};
    std::atomic<int> fail_idx{-1};
    std::mutex io_mu;

    auto worker = [&](){
        while(true){
            if(fail_idx.load(std::memory_order_relaxed) >= 0) break;
            int ii = next_idx.fetch_add(1, std::memory_order_relaxed);
            if(ii >= cfg.n_inst) break;
            InstanceResult res = run_one_instance(ii, cfg, pv);
            results[ii] = std::move(res);
            if(!results[ii].ok){
                int expect = -1;
                fail_idx.compare_exchange_strong(expect, ii, std::memory_order_relaxed);
            }
            int completed = done.fetch_add(1, std::memory_order_relaxed) + 1;
            double el = elapsed();
            double eta = (completed > 0) ? (el / completed * (cfg.n_inst - completed)) : 0.0;
            std::lock_guard<std::mutex> lk(io_mu);
            if(cfg.verbose_p && !results[ii].log.empty()) std::fputs(results[ii].log.c_str(), stdout);
            if(results[ii].ok){
                std::printf("  [done %2d/%d] instance %d finished in %.1fs  elapsed %.0fs  ETA %.0fs\n",
                            completed, cfg.n_inst, ii + 1, results[ii].wall_seconds, el, eta);
            } else {
                std::printf("  [FAIL %2d/%d] instance %d failed KNN verification after %.1fs\n",
                            completed, cfg.n_inst, ii + 1, results[ii].wall_seconds);
            }
            std::fflush(stdout);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve((size_t)cfg.threads);
    for(int t=0; t<cfg.threads; ++t) pool.emplace_back(worker);
    for(auto& th : pool) th.join();

    if(fail_idx.load(std::memory_order_relaxed) >= 0){
        int bad = fail_idx.load(std::memory_order_relaxed);
        std::fprintf(stderr, "KNN verification failed on instance %d\n", bad + 1);
        return 2;
    }

    std::vector<std::vector<double>> all_f(pv.size());
    for(auto& v : all_f) v.reserve(cfg.n_inst);
    OracleStats oracle_total;
    std::array<int,4> regime_counts{{0,0,0,0}};
    std::array<double,4> regime_seconds{{0.0,0.0,0.0,0.0}};
    for(int ii=0; ii<cfg.n_inst; ++ii){
        oracle_total.add(results[ii].oracle);
        for(int r=0; r<4; ++r){
            regime_counts[(size_t)r] += results[ii].regime_counts[(size_t)r];
            regime_seconds[(size_t)r] += results[ii].regime_seconds[(size_t)r];
        }
        for(int pi=0; pi<(int)pv.size(); ++pi){
            all_f[pi].push_back(results[ii].fvals[pi]);
        }
    }

    double wt = elapsed();
    if(!write_json(output_path.c_str(), cfg.N, pv, all_f, cfg.n_inst, cfg.n_inst, cfg.restarts, cfg.sa_iters, cfg.seed, cfg.threads, wt,
                   oracle_total, cfg.oracle.status, distance_backend_name(), results_meta,
                   regime_counts, regime_seconds)){
        std::fprintf(stderr, "Failed to write JSON results to %s\n", output_path.c_str());
        return 2;
    }

    std::printf("\n══════════════════════════════════════════════════════════════════════════\n");
    std::printf("  FINAL (N=%d, %d inst, %.0fs = %.1fmin, %d threads)\n", cfg.N, cfg.n_inst, wt, wt/60.0, cfg.threads);
    std::printf("══════════════════════════════════════════════════════════════════════════\n");
    std::printf("  %6s %4s %9s %8s %8s %8s %8s\n", "p", "k", "mean_f", "±se", "std", "min", "max");
    std::printf("  ──────────────────────────────────────────────────────────────────\n");
    for(int pi=0; pi<(int)pv.size(); pi++){
        double p = pv[pi];
        int k = std::max(3, (int)std::round(p * cfg.N));
        const auto& v = all_f[pi];
        if(v.empty()) continue;
        double mean = 0.0;
        for(double x : v) mean += x;
        mean /= static_cast<double>(v.size());
        double var = 0.0;
        if(v.size() > 1){
            for(double x : v) var += (x - mean) * (x - mean);
            var /= static_cast<double>(v.size() - 1);
        }
        double sd = std::sqrt(var);
        double se = v.size() > 1 ? sd / std::sqrt(static_cast<double>(v.size())) : 0.0;
        auto mm = std::minmax_element(v.begin(), v.end());
        char bhh_suffix[64] = "";
        if(p >= 0.999) std::snprintf(bhh_suffix, sizeof(bhh_suffix), "  (BHH asymptotic ref ≈ %.4f)", BHH_REFERENCE);
        std::printf("  %6.3f %4d %9.4f %8.4f %8.4f %8.4f %8.4f%s\n",
                    p, k, mean, se, sd, *mm.first, *mm.second, bhh_suffix);
    }
    if(!all_f.empty() && !all_f.back().empty()){
        double mean = 0.0;
        for(double x : all_f.back()) mean += x;
        mean /= static_cast<double>(all_f.back().size());
        std::printf("\n  f(1) = %.4f  (vs BHH asymptotic ref %.4f; finite-N and heuristic suboptimality may keep this higher)\n", mean, BHH_REFERENCE);
    }
    std::printf("  Regime usage (count / cumulative seconds):\n");
    for(int r=0; r<4; ++r){
        std::printf("    %-4s %5d / %7.1fs\n",
                    run_regime_label_from_index(r),
                    regime_counts[(size_t)r],
                    regime_seconds[(size_t)r]);
    }
    std::printf("  Total wall time: %.0fs (%.1f min)\n", wt, wt/60.0);
    return 0;
}

} // namespace run_cli_detail
