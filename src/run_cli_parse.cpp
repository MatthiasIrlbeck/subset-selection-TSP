#include "run_cli_internal.hpp"

namespace {

struct CliOptions {
    int N = 500;
    int n_inst = 15;
    int restarts = 3;
    int sa_iters = 60000;
    int seed = 2024;
    bool self_test = false;
    int tsp_restarts = -1;
    int tsp_ils = -1;
    int tsp_patience = -1;
    bool user_set_restarts = false;
    bool user_set_tsp_restarts = false;
    bool user_set_tsp_ils = false;
    bool user_set_tsp_patience = false;
    int verify_knn_checks = 0;
    double grid_cell = 0.0;
    bool user_set_grid_cell = false;
    int threads = 0;
    bool user_set_threads = false;
    int knn_override = -1;
    bool user_set_knn = false;
    bool verbose_p = false;
    run_cli_detail::SolverMode mode = run_cli_detail::SolverMode::BALANCED;
    std::string output_path = "results.json";
    bool force_output = false;
    ExternalOracleMode oracle_mode = ExternalOracleMode::NONE;
    OracleProblemFormat oracle_format = OracleProblemFormat::MATRIX;
    std::string lkh_path = "LKH";
    std::string concorde_path = "concorde";
    int oracle_time_limit = 0;
    int oracle_scale = 1000000;
    int oracle_tsp_top = 3;
    int oracle_subset_top = 2;
    int oracle_min_k = EXACT_SMALL_TSP_K + 1;
    int oracle_max_k = 2500;
    int oracle_lkh_runs = 6;
    int oracle_lkh_trials = 0;
    bool oracle_no_tsp = false;
    bool oracle_no_subset = false;
    bool oracle_inline_feedback = false;
    bool oracle_verbose = false;
};

bool parse_int_arg(const std::string& s,int& out){
    try{
        size_t pos = 0;
        int v = std::stoi(s, &pos);
        if(pos != s.size()) return false;
        out = v;
        return true;
    } catch(const std::exception&){
        return false;
    }
}

bool parse_double_arg(const std::string& s,double& out){
    try{
        size_t pos = 0;
        double v = std::stod(s, &pos);
        if(pos != s.size()) return false;
        out = v;
        return true;
    } catch(const std::exception&){
        return false;
    }
}

void print_help(const char* prog){
    const char* exe = (prog && *prog) ? prog : "aldous_tsp";
    std::printf(R"(Usage: %s [options]

General options:
  --help                      Show this help message and exit
  --quick                     Run a smaller quick configuration
  --self-test                 Run built-in self-tests and exit
  --mode <name>               Solver mode
                              balanced       default mode for full-curve runs
                              smallp-region  experimental low-p search path
                              highp-delete   experimental high-p deletion path
                              hybrid         experimental combined path
  --output <file>             Write JSON results to <file> (default: results.json)
  --force                     Allow overwriting an existing output file
  --verbose-p                 Print per-instance per-p progress details

Simulation options:
  --N <int>                   Number of points (default: 500)
  --instances <int>           Number of instances (default: 15)
  --restarts <int>            Subset-solver restarts (default: 3)
  --sa-iters <int>            Subset SA iterations budget (default: 60000)
  --seed <int>                Base RNG seed (default: 2024)
  --threads <int>             Worker threads (default: auto)
  --knn <int>                 Override KNN candidate count
  --grid-cell <float>         Force grid cell size for exact KNN build
  --verify-knn <int>          Sampled exact-KNN verification checks

TSP options:
  --tsp-restarts <int>        Full-TSP restart count
  --tsp-ils <int>             Full-TSP ILS iterations
  --tsp-patience <int>        Full-TSP stagnation patience

External oracle options:
  --oracle <name>             Oracle mode: %s (default: none)
                              none keeps runs fully internal and reproducible
                              auto probes PATH for LKH first, then Concorde
  --oracle-format <name>      TSPLIB export format: %s (default: matrix)
  --lkh-path <path>           Path or executable name for LKH (default: LKH)
  --concorde-path <path>      Path or executable name for Concorde (default: concorde)
  --oracle-time-limit <int>   External-oracle time limit in seconds
  --oracle-scale <int>        Integer scaling for TSPLIB export (default: 1000000)
  --oracle-tsp-top <int>      Number of full-TSP elite tours to post-process (default: 3)
  --oracle-subset-top <int>   Number of subset elite tours to post-process (default: 2)
  --oracle-min-k <int>        Minimum k for external oracle use
  --oracle-max-k <int>        Maximum k for external oracle use
  --oracle-lkh-runs <int>     LKH RUNS parameter (default: 6)
  --oracle-lkh-trials <int>   Optional LKH MAX_TRIALS override
  --oracle-no-tsp             Disable oracle use on full TSP instances
  --oracle-no-subset          Disable oracle use on subset instances
  --oracle-inline-feedback    Apply oracle polishing inside elite refinement
  --oracle-verbose            Forward external solver logs to stderr

Notes:
  balanced is the default mode.
  External solver support is optional; --oracle none is the default.
  smallp-region, highp-delete, and hybrid are experimental modes.
)",
        exe,
        run_cli_detail::external_oracle_mode_choices(),
        run_cli_detail::oracle_problem_format_choices());
}

std::vector<double> build_default_p_values(){
    return {0.02,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.80,1.00};
}

} // namespace

namespace run_cli_detail {

int tsp_main_cli_impl(int argc,char** argv){
    CliOptions opt;

    auto need_value = [&](int i,const std::string& flag)->bool{
        if(i + 1 < argc) return true;
        std::fprintf(stderr, "Missing value for %s\n", flag.c_str());
        return false;
    };
    auto parse_next_int = [&](int& i,const std::string& flag,int& out)->bool{
        if(!need_value(i, flag)) return false;
        std::string s = argv[++i];
        if(!parse_int_arg(s, out)){
            std::fprintf(stderr, "Invalid integer for %s: %s\n", flag.c_str(), s.c_str());
            return false;
        }
        return true;
    };
    auto parse_next_double = [&](int& i,const std::string& flag,double& out)->bool{
        if(!need_value(i, flag)) return false;
        std::string s = argv[++i];
        if(!parse_double_arg(s, out)){
            std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), s.c_str());
            return false;
        }
        return true;
    };
    auto parse_next_string = [&](int& i,const std::string& flag,std::string& out)->bool{
        if(!need_value(i, flag)) return false;
        out = argv[++i];
        return true;
    };
    auto apply_quick_preset = [&](){
        opt.N = 200;
        opt.n_inst = 3;
        opt.restarts = 2;
        opt.sa_iters = 25000;
        opt.user_set_restarts = false;
    };

    for(int i=1;i<argc;i++){
        std::string a = argv[i];
        if(a == "--help") { print_help(argv[0]); return 0; }
        else if(a == "--quick") apply_quick_preset();
        else if(a == "--self-test") opt.self_test = true;
        else if(a == "--mode") {
            std::string s;
            if(!parse_next_string(i, a, s)) return 1;
            if(!try_parse_solver_mode(s, opt.mode)){
                std::fprintf(stderr, "Invalid value for %s: %s (expected one of: %s)\n",
                             a.c_str(), s.c_str(), solver_mode_choices());
                return 1;
            }
        }
        else if(a == "--N") { if(!parse_next_int(i, a, opt.N)) return 1; }
        else if(a == "--instances") { if(!parse_next_int(i, a, opt.n_inst)) return 1; }
        else if(a == "--restarts") { if(!parse_next_int(i, a, opt.restarts)) return 1; opt.user_set_restarts = true; }
        else if(a == "--sa-iters") { if(!parse_next_int(i, a, opt.sa_iters)) return 1; }
        else if(a == "--tsp-restarts") { if(!parse_next_int(i, a, opt.tsp_restarts)) return 1; opt.user_set_tsp_restarts = true; }
        else if(a == "--tsp-ils") { if(!parse_next_int(i, a, opt.tsp_ils)) return 1; opt.user_set_tsp_ils = true; }
        else if(a == "--tsp-patience") { if(!parse_next_int(i, a, opt.tsp_patience)) return 1; opt.user_set_tsp_patience = true; }
        else if(a == "--seed") { if(!parse_next_int(i, a, opt.seed)) return 1; }
        else if(a == "--verify-knn") { if(!parse_next_int(i, a, opt.verify_knn_checks)) return 1; }
        else if(a == "--grid-cell") { if(!parse_next_double(i, a, opt.grid_cell)) return 1; opt.user_set_grid_cell = true; }
        else if(a == "--threads") { if(!parse_next_int(i, a, opt.threads)) return 1; opt.user_set_threads = true; }
        else if(a == "--knn") { if(!parse_next_int(i, a, opt.knn_override)) return 1; opt.user_set_knn = true; }
        else if(a == "--verbose-p") opt.verbose_p = true;
        else if(a == "--output") { if(!parse_next_string(i, a, opt.output_path)) return 1; }
        else if(a == "--force") opt.force_output = true;
        else if(a == "--oracle") {
            std::string s;
            if(!parse_next_string(i, a, s)) return 1;
            if(!try_parse_external_oracle_mode(s, opt.oracle_mode)){
                std::fprintf(stderr, "Invalid value for %s: %s (expected one of: %s)\n",
                             a.c_str(), s.c_str(), external_oracle_mode_choices());
                return 1;
            }
        }
        else if(a == "--oracle-format") {
            std::string s;
            if(!parse_next_string(i, a, s)) return 1;
            if(!try_parse_oracle_problem_format(s, opt.oracle_format)){
                std::fprintf(stderr, "Invalid value for %s: %s (expected one of: %s)\n",
                             a.c_str(), s.c_str(), oracle_problem_format_choices());
                return 1;
            }
        }
        else if(a == "--lkh-path") { if(!parse_next_string(i, a, opt.lkh_path)) return 1; }
        else if(a == "--concorde-path") { if(!parse_next_string(i, a, opt.concorde_path)) return 1; }
        else if(a == "--oracle-time-limit") { if(!parse_next_int(i, a, opt.oracle_time_limit)) return 1; }
        else if(a == "--oracle-scale") { if(!parse_next_int(i, a, opt.oracle_scale)) return 1; }
        else if(a == "--oracle-tsp-top") { if(!parse_next_int(i, a, opt.oracle_tsp_top)) return 1; }
        else if(a == "--oracle-subset-top") { if(!parse_next_int(i, a, opt.oracle_subset_top)) return 1; }
        else if(a == "--oracle-min-k") { if(!parse_next_int(i, a, opt.oracle_min_k)) return 1; }
        else if(a == "--oracle-max-k") { if(!parse_next_int(i, a, opt.oracle_max_k)) return 1; }
        else if(a == "--oracle-lkh-runs") { if(!parse_next_int(i, a, opt.oracle_lkh_runs)) return 1; }
        else if(a == "--oracle-lkh-trials") { if(!parse_next_int(i, a, opt.oracle_lkh_trials)) return 1; }
        else if(a == "--oracle-no-tsp") opt.oracle_no_tsp = true;
        else if(a == "--oracle-no-subset") opt.oracle_no_subset = true;
        else if(a == "--oracle-inline-feedback") opt.oracle_inline_feedback = true;
        else if(a == "--oracle-verbose") opt.oracle_verbose = true;
        else{
            std::fprintf(stderr, "Unknown arg: %s\n", a.c_str());
            return 1;
        }
    }

    if(opt.self_test){
        return run_self_test(argv[0]);
    }

    if(opt.N < 3){
        std::fprintf(stderr, "Invalid value for --N: %d (expected >= 3)\n", opt.N);
        return 1;
    }
    if(opt.n_inst < 1){
        std::fprintf(stderr, "Invalid value for --instances: %d (expected >= 1)\n", opt.n_inst);
        return 1;
    }
    if(opt.restarts < 1){
        std::fprintf(stderr, "Invalid value for --restarts: %d (expected >= 1)\n", opt.restarts);
        return 1;
    }
    if(opt.sa_iters < 1){
        std::fprintf(stderr, "Invalid value for --sa-iters: %d (expected >= 1)\n", opt.sa_iters);
        return 1;
    }
    if(opt.user_set_tsp_restarts && opt.tsp_restarts < 1){
        std::fprintf(stderr, "Invalid value for --tsp-restarts: %d (expected >= 1)\n", opt.tsp_restarts);
        return 1;
    }
    if(opt.user_set_tsp_ils && opt.tsp_ils < 0){
        std::fprintf(stderr, "Invalid value for --tsp-ils: %d (expected >= 0)\n", opt.tsp_ils);
        return 1;
    }
    if(opt.user_set_tsp_patience && opt.tsp_patience < 0){
        std::fprintf(stderr, "Invalid value for --tsp-patience: %d (expected >= 0)\n", opt.tsp_patience);
        return 1;
    }
    if(opt.verify_knn_checks < 0){
        std::fprintf(stderr, "Invalid value for --verify-knn: %d (expected >= 0)\n", opt.verify_knn_checks);
        return 1;
    }
    if(opt.user_set_grid_cell && (!std::isfinite(opt.grid_cell) || opt.grid_cell <= 0.0)){
        std::fprintf(stderr, "Invalid value for --grid-cell: %.17g (expected a finite value > 0)\n", opt.grid_cell);
        return 1;
    }
    if(opt.user_set_threads && opt.threads < 1){
        std::fprintf(stderr, "Invalid value for --threads: %d (expected >= 1)\n", opt.threads);
        return 1;
    }
    if(opt.user_set_knn && opt.knn_override < 1){
        std::fprintf(stderr, "Invalid value for --knn: %d (expected >= 1)\n", opt.knn_override);
        return 1;
    }
    if(opt.user_set_knn && opt.knn_override >= opt.N){
        std::fprintf(stderr, "Invalid value for --knn: %d (expected between 1 and N-1=%d)\n", opt.knn_override, opt.N - 1);
        return 1;
    }
    if(opt.oracle_time_limit < 0){
        std::fprintf(stderr, "Invalid value for --oracle-time-limit: %d (expected >= 0)\n", opt.oracle_time_limit);
        return 1;
    }
    if(opt.oracle_scale < 1){
        std::fprintf(stderr, "Invalid value for --oracle-scale: %d (expected >= 1)\n", opt.oracle_scale);
        return 1;
    }
    if(opt.oracle_tsp_top < 0){
        std::fprintf(stderr, "Invalid value for --oracle-tsp-top: %d (expected >= 0)\n", opt.oracle_tsp_top);
        return 1;
    }
    if(opt.oracle_subset_top < 0){
        std::fprintf(stderr, "Invalid value for --oracle-subset-top: %d (expected >= 0)\n", opt.oracle_subset_top);
        return 1;
    }
    if(opt.oracle_min_k < EXACT_SMALL_TSP_K + 1){
        std::fprintf(stderr, "Invalid value for --oracle-min-k: %d (expected >= %d)\n", opt.oracle_min_k, EXACT_SMALL_TSP_K + 1);
        return 1;
    }
    if(opt.oracle_max_k < opt.oracle_min_k){
        std::fprintf(stderr, "Invalid value for --oracle-max-k: %d (expected >= --oracle-min-k=%d)\n", opt.oracle_max_k, opt.oracle_min_k);
        return 1;
    }
    if(opt.oracle_lkh_runs < 1){
        std::fprintf(stderr, "Invalid value for --oracle-lkh-runs: %d (expected >= 1)\n", opt.oracle_lkh_runs);
        return 1;
    }
    if(opt.oracle_lkh_trials < 0){
        std::fprintf(stderr, "Invalid value for --oracle-lkh-trials: %d (expected >= 0)\n", opt.oracle_lkh_trials);
        return 1;
    }

    int knn_k = (opt.knn_override > 0) ? std::min(opt.knn_override, std::max(1, opt.N - 1)) : std::min(40, opt.N / 3);
    std::vector<double> pv = build_default_p_values();

    unsigned hw = std::thread::hardware_concurrency();
    int auto_threads = (hw > 0) ? (int)hw : 1;
    if(opt.threads <= 0) opt.threads = auto_threads;
    opt.threads = std::max(1, std::min(opt.threads, opt.n_inst));

    RunConfig cfg;
    cfg.N = opt.N;
    cfg.n_inst = opt.n_inst;
    cfg.restarts = opt.restarts;
    cfg.sa_iters = opt.sa_iters;
    cfg.seed = opt.seed;
    cfg.tsp_restarts = (opt.tsp_restarts >= 0) ? opt.tsp_restarts : (opt.user_set_restarts ? opt.restarts : std::max(opt.restarts, 5));
    cfg.tsp_ils = (opt.tsp_ils >= 0) ? opt.tsp_ils : std::max(800, 200 + opt.N/2);
    cfg.tsp_patience = (opt.tsp_patience >= 0) ? opt.tsp_patience : std::max(80, opt.N/10);
    cfg.knn_k = knn_k;
    cfg.verify_knn_checks = opt.verify_knn_checks;
    cfg.threads = opt.threads;
    cfg.grid_cell = opt.grid_cell;
    cfg.verbose_p = opt.verbose_p;
    cfg.mode = opt.mode;

    ExternalOracleConfig oracle_cfg;
    oracle_cfg.mode = opt.oracle_mode;
    oracle_cfg.problem_format = opt.oracle_format;
    oracle_cfg.lkh_path = opt.lkh_path;
    oracle_cfg.concorde_path = opt.concorde_path;
    oracle_cfg.time_limit_sec = std::max(0, opt.oracle_time_limit);
    oracle_cfg.scale = std::max(1, opt.oracle_scale);
    oracle_cfg.tsp_top = std::max(0, opt.oracle_tsp_top);
    oracle_cfg.subset_top = std::max(0, opt.oracle_subset_top);
    oracle_cfg.min_k = std::max(EXACT_SMALL_TSP_K + 1, opt.oracle_min_k);
    oracle_cfg.max_k = std::max(oracle_cfg.min_k, opt.oracle_max_k);
    oracle_cfg.lkh_runs = std::max(1, opt.oracle_lkh_runs);
    oracle_cfg.lkh_max_trials = std::max(0, opt.oracle_lkh_trials);
    oracle_cfg.use_for_tsp = !opt.oracle_no_tsp;
    oracle_cfg.use_for_subset = !opt.oracle_no_subset;
    oracle_cfg.inline_feedback = opt.oracle_inline_feedback;
    oracle_cfg.verbose = opt.oracle_verbose;

    std::string oracle_err;
    if(!build_oracle_context(oracle_cfg, cfg.oracle, oracle_err)){
        std::fprintf(stderr, "%s\n", oracle_err.c_str());
        return 2;
    }

    ResultsMetadata results_meta = build_results_metadata(argc, argv, opt.mode);
    return run_cli_execute(cfg, pv, opt.output_path, opt.force_output, results_meta);
}

} // namespace run_cli_detail
