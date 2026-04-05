#include "subset_solver_internal.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>


static std::string resolve_exec_in_path(const std::string& prog){
    // Direct PATH lookup.
    if(prog.empty()) return std::string();
    if(prog.find('/') != std::string::npos){
        return access(prog.c_str(), X_OK) == 0 ? prog : std::string();
    }
    const char* env = std::getenv("PATH");
    if(!env) return std::string();
    std::string path(env);
    size_t start = 0;
    while(start <= path.size()){
        size_t end = path.find(':', start);
        std::string dir = (end == std::string::npos) ? path.substr(start) : path.substr(start, end - start);
        if(dir.empty()) dir = ".";
        std::string cand = dir + "/" + prog;
        if(access(cand.c_str(), X_OK) == 0) return cand;
        if(end == std::string::npos) break;
        start = end + 1;
    }
    return std::string();
}

bool build_oracle_context(const ExternalOracleConfig& cfg,OracleContext& oracle,std::string& err){
    err.clear();
    oracle = OracleContext();
    oracle.cfg = cfg;
    if(oracle.cfg.mode == ExternalOracleMode::NONE){
        oracle.status = "disabled";
        return true;
    }
    std::string lkh = resolve_exec_in_path(oracle.cfg.lkh_path);
    std::string con = resolve_exec_in_path(oracle.cfg.concorde_path);
    if(oracle.cfg.mode == ExternalOracleMode::AUTO){
        if(!lkh.empty()){
            oracle.resolved = ResolvedOracleMode::LKH;
            oracle.exec_path = lkh;
        } else if(!con.empty()){
            oracle.resolved = ResolvedOracleMode::CONCORDE;
            oracle.exec_path = con;
        } else {
            oracle.status = "auto: no supported external solver found on PATH";
            return true;
        }
    } else if(oracle.cfg.mode == ExternalOracleMode::LKH){
        if(lkh.empty()){
            err = "Requested --oracle lkh, but executable was not found: " + oracle.cfg.lkh_path;
            return false;
        }
        oracle.resolved = ResolvedOracleMode::LKH;
        oracle.exec_path = lkh;
    } else if(oracle.cfg.mode == ExternalOracleMode::CONCORDE){
        if(con.empty()){
            err = "Requested --oracle concorde, but executable was not found: " + oracle.cfg.concorde_path;
            return false;
        }
        oracle.resolved = ResolvedOracleMode::CONCORDE;
        oracle.exec_path = con;
    }

    OracleProblemFormat fmt = resolved_oracle_problem_format(oracle);
    std::ostringstream oss;
    oss << resolved_mode_name(oracle.resolved) << " @ " << oracle.exec_path
        << " (format=" << oracle_problem_format_name(fmt)
        << ", scale=" << oracle.cfg.scale
        << ", tsp-top=" << oracle.cfg.tsp_top
        << ", subset-top=" << oracle.cfg.subset_top
        << ", k-range=[" << oracle.cfg.min_k << ',' << oracle.cfg.max_k << "]"
        << (oracle.cfg.inline_feedback ? ", inline-feedback" : ", posthoc-only");
    if(oracle.cfg.verbose) oss << ", verbose";
    oss << ")";
    oracle.status = oss.str();
    return true;
}

static bool write_tsplib_full_matrix(const std::string& fn,const Instance& inst,const std::vector<int>& base_nodes,int scale){
    int k = (int)base_nodes.size();
    FILE* f = std::fopen(fn.c_str(), "w");
    if(!f) return false;
    static thread_local std::array<char, TSPLIB_IO_BUFFER_BYTES> buf;
    std::setvbuf(f, buf.data(), _IOFBF, buf.size());
    std::fprintf(f, "NAME : aldous_oracle\nTYPE : TSP\nDIMENSION : %d\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n", k);
    for(int i=0; i<k; ++i){
        int ai = base_nodes[i];
        for(int j=0; j<k; ++j){
            long long w = 0;
            if(i != j){
                double d = inst.dist(ai, base_nodes[j]);
                w = (long long)std::llround((double)scale * d);
            }
            std::fprintf(f, "%lld%c", w, (j + 1 == k) ? '\n' : ' ');
        }
    }
    std::fprintf(f, "EOF\n");
    std::fclose(f);
    return true;
}

static bool write_tsplib_euc2d(const std::string& fn,const Instance& inst,const std::vector<int>& base_nodes,int scale){
    int k = (int)base_nodes.size();
    FILE* f = std::fopen(fn.c_str(), "w");
    if(!f) return false;
    static thread_local std::array<char, TSPLIB_IO_BUFFER_BYTES> buf;
    std::setvbuf(f, buf.data(), _IOFBF, buf.size());
    std::fprintf(f, "NAME : aldous_oracle\nTYPE : TSP\nDIMENSION : %d\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n", k);
    for(int i=0; i<k; ++i){
        long long xi = (long long)std::llround((double)scale * inst.x[base_nodes[i]]);
        long long yi = (long long)std::llround((double)scale * inst.y[base_nodes[i]]);
        std::fprintf(f, "%d %lld %lld\n", i + 1, xi, yi);
    }
    std::fprintf(f, "EOF\n");
    std::fclose(f);
    return true;
}

static bool write_tsplib_identity_tour(const std::string& fn,int k){
    FILE* f = std::fopen(fn.c_str(), "w");
    if(!f) return false;
    std::fprintf(f, "NAME : init\nTYPE : TOUR\nDIMENSION : %d\nTOUR_SECTION\n", k);
    for(int i=1; i<=k; ++i) std::fprintf(f, "%d\n", i);
    std::fprintf(f, "-1\nEOF\n");
    std::fclose(f);
    return true;
}

static std::vector<long long> extract_all_ints(const std::string& text){
    std::vector<long long> vals;
    const char* s = text.c_str();
    while(*s){
        char* e = nullptr;
        long long v = std::strtoll(s, &e, 10);
        if(e != s){
            vals.push_back(v);
            s = e;
        } else ++s;
    }
    return vals;
}

static bool parse_perm_window(const std::vector<long long>& vals,size_t start,int k,bool one_based,std::vector<int>& perm){
    if(start + (size_t)k > vals.size()) return false;
    perm.assign(k, -1);
    std::vector<uint8_t> seen((size_t)k, 0);
    for(int i=0; i<k; ++i){
        long long raw = vals[start + (size_t)i];
        long long v = one_based ? (raw - 1) : raw;
        if(v < 0 || v >= k) return false;
        if(seen[(size_t)v]) return false;
        seen[(size_t)v] = 1;
        perm[i] = (int)v;
    }
    return true;
}

static bool parse_external_tour_file(const std::string& fn,int k,std::vector<int>& perm){
    std::ifstream in(fn);
    if(!in) return false;
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto try_vals = [&](const std::vector<long long>& vals)->bool{
        std::vector<int> tmp;
        for(size_t start=0; start + (size_t)k <= vals.size(); ++start){
            if(parse_perm_window(vals, start, k, false, tmp)){ perm = tmp; return true; }
            if(parse_perm_window(vals, start, k, true, tmp)){ perm = tmp; return true; }
        }
        return false;
    };
    size_t ts = text.find("TOUR_SECTION");
    if(ts != std::string::npos){
        if(try_vals(extract_all_ints(text.substr(ts)))) return true;
    }
    return try_vals(extract_all_ints(text));
}

static int run_external_process(const std::vector<std::string>& argv,const std::string& cwd,
                                int timeout_sec,bool verbose){
    if(argv.empty()) return -1;
    pid_t pid = fork();
    if(pid == 0){
        setpgid(0, 0);
        if(!cwd.empty()) { if(chdir(cwd.c_str()) != 0) _exit(126); }
        if(verbose){
            dup2(STDERR_FILENO, STDOUT_FILENO);
        } else {
            int fd = open("/dev/null", O_WRONLY);
            if(fd >= 0){
                dup2(fd, STDOUT_FILENO);
                dup2(fd, STDERR_FILENO);
                if(fd > STDERR_FILENO) close(fd);
            }
        }
        std::vector<char*> args;
        args.reserve(argv.size() + 1);
        for(const auto& s : argv) args.push_back(const_cast<char*>(s.c_str()));
        args.push_back(nullptr);
        execvp(args[0], args.data());
        _exit(127);
    }
    if(pid < 0) return -1;
    int status = 0;
    if(timeout_sec <= 0){
        if(waitpid(pid, &status, 0) != pid) return -1;
        if(WIFEXITED(status) && WEXITSTATUS(status) == 0) return 0;
        return status ? status : -1;
    }
    auto t0 = std::chrono::steady_clock::now();
    while(true){
        pid_t w = waitpid(pid, &status, WNOHANG);
        if(w == pid){
            if(WIFEXITED(status) && WEXITSTATUS(status) == 0) return 0;
            return status ? status : -1;
        }
        if(w < 0) return -1;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if(dt > timeout_sec){
            kill(-pid, SIGKILL);
            waitpid(pid, &status, 0);
            return 124;
        }
        // Oracle calls are long enough that coarse polling is fine here.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

static bool external_oracle_polish_nodes(const Instance& inst,const OracleContext& oracle,
                                         const std::vector<int>& input_nodes,
                                         std::vector<int>& out_nodes,double& out_len,
                                         uint64_t seed,int post_strength){
    int k = (int)input_nodes.size();
    if(k < 3) return false;
    TempWorkDir tmp("aldous_oracle");
    if(!tmp.ok) return false;
    std::string tsp = tmp.path + "/problem.tsp";
    std::string init = tmp.path + "/init.tour";
    std::string par = tmp.path + "/run.par";
    std::string out = tmp.path + "/out.tour";
    OracleProblemFormat fmt = resolved_oracle_problem_format(oracle);
    bool wrote = false;
    if(fmt == OracleProblemFormat::MATRIX) wrote = write_tsplib_full_matrix(tsp, inst, input_nodes, oracle.cfg.scale);
    else wrote = write_tsplib_euc2d(tsp, inst, input_nodes, oracle.cfg.scale);
    if(!wrote) return false;
    std::vector<std::string> argv;
    if(oracle.resolved == ResolvedOracleMode::LKH){
        if(!write_tsplib_identity_tour(init, k)) return false;
        std::ofstream pf(par);
        if(!pf) return false;
        pf << "PROBLEM_FILE = problem.tsp\n";
        pf << "INITIAL_TOUR_FILE = init.tour\n";
        pf << "TOUR_FILE = out.tour\n";
        pf << "RUNS = " << std::max(1, oracle.cfg.lkh_runs) << "\n";
        if(oracle.cfg.lkh_max_trials > 0) pf << "MAX_TRIALS = " << oracle.cfg.lkh_max_trials << "\n";
        if(oracle.cfg.time_limit_sec > 0) pf << "TIME_LIMIT = " << oracle.cfg.time_limit_sec << "\n";
        pf << "SEED = " << (unsigned)(seed & 0x7fffffffULL) << "\n";
        pf << "TRACE_LEVEL = " << (oracle.cfg.verbose ? 1 : 0) << "\n";
        pf.close();
        argv = {oracle.exec_path, par};
    } else if(oracle.resolved == ResolvedOracleMode::CONCORDE){
        argv = {oracle.exec_path, "-o", out, tsp};
    } else return false;
    int rc = run_external_process(argv, tmp.path, oracle.cfg.time_limit_sec, oracle.cfg.verbose);
    if(rc != 0) return false;
    std::vector<int> perm;
    if(!parse_external_tour_file(out, k, perm)) {
        if(oracle.resolved != ResolvedOracleMode::CONCORDE) return false;
        std::string fallback = tmp.path + "/problem.sol";
        if(!parse_external_tour_file(fallback, k, perm)) return false;
    }
    out_nodes.resize(k);
    for(int i=0; i<k; ++i) out_nodes[i] = input_nodes[perm[i]];
    Tour t;
    t.init(inst.N);
    t.set_tour_only(out_nodes.data(), k);
    t.recompute_length(inst);
    polish_fixed_subset_tour(t, inst, post_strength);
    out_nodes = t.nodes;
    out_len = t.length;
    return true;
}

bool external_oracle_polish_tour(Tour& cand,const Instance& inst,const OracleContext& oracle,
                                        bool full_tsp,int post_strength,OracleStats* stats){
    cand.ensure_edges(inst);
    if(!external_oracle_applicable(oracle, cand.k, full_tsp)) return false;
    if(stats){
        ++stats->calls;
        if(full_tsp) ++stats->tsp_calls; else ++stats->subset_calls;
    }
    double before = cand.length;
    std::vector<int> out_nodes;
    double out_len = std::numeric_limits<double>::infinity();
    uint64_t seed = mix_hash64(subset_hash_nodes(cand.nodes) ^ ((uint64_t)cand.k << 32)
                               ^ (full_tsp ? 0x6a09e667f3bcc909ULL : 0xbb67ae8584caa73bULL));
    if(!external_oracle_polish_nodes(inst, oracle, cand.nodes, out_nodes, out_len, seed, post_strength)){
        if(stats) ++stats->failed;
        return false;
    }
    if(stats){
        ++stats->solved;
        if(full_tsp) ++stats->tsp_solved; else ++stats->subset_solved;
    }
    if(out_len + IMPROVEMENT_EPS < before){
        cand.set_tour_only(out_nodes.data(), cand.k);
        cand.recompute_length(inst);
        if(stats){
            ++stats->improved;
            double gain = before - cand.length;
            stats->exact_gain += gain;
            if(full_tsp){ ++stats->tsp_improved; stats->tsp_gain += gain; }
            else { ++stats->subset_improved; stats->subset_gain += gain; }
        }
        return true;
    }
    return false;
}
