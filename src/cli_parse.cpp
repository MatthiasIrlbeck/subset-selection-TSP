#include "cli_internal.hpp"

namespace aldous_tsp {

bool parse_int(const std::string& text, int& out) {
    try {
        std::size_t pos = 0;
        const int value = std::stoi(text, &pos);
        if (pos != text.size()) {
            return false;
        }
        out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_double(const std::string& text, double& out) {
    try {
        std::size_t pos = 0;
        const double value = std::stod(text, &pos);
        if (pos != text.size() || !std::isfinite(value)) {
            return false;
        }
        out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::vector<std::string> split(const std::string& text, char delim) {
    std::vector<std::string> parts;
    std::string cur;
    std::istringstream in(text);
    while (std::getline(in, cur, delim)) {
        parts.push_back(cur);
    }
    return parts;
}

bool parse_p_values(const std::string& text, std::vector<double>& out) {
    std::vector<double> values;
    for (const std::string& part : split(text, ',')) {
        if (part.empty()) {
            continue;
        }
        double p = 0.0;
        if (!parse_double(part, p) || !(p > 0.0 && p <= 1.0)) {
            return false;
        }
        values.push_back(p);
    }
    if (values.empty()) {
        return false;
    }
    out = std::move(values);
    return true;
}

bool parse_p_range(const std::string& text, std::vector<double>& out) {
    const auto parts = split(text, ':');
    if (parts.size() != 3U) {
        return false;
    }
    double first = 0.0;
    double last = 0.0;
    int count = 0;
    if (!parse_double(parts[0], first) || !parse_double(parts[1], last) || !parse_int(parts[2], count)) {
        return false;
    }
    if (!(first > 0.0 && first <= 1.0 && last > 0.0 && last <= 1.0) || count < 1) {
        return false;
    }
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(count));
    if (count == 1) {
        values.push_back(first);
    } else {
        for (int i = 0; i < count; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(count - 1);
            values.push_back(first + (last - first) * t);
        }
    }
    out = std::move(values);
    return true;
}

bool read_p_file(const std::string& path, std::vector<double>& out, std::string& err) {
    std::ifstream in(path);
    if (!in) {
        err = "failed to open p-file: " + path;
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    for (char& ch : text) {
        if (ch == ',') {
            ch = ' ';
        }
    }
    std::istringstream tokens(text);
    std::vector<double> values;
    std::string token;
    while (tokens >> token) {
        double p = 0.0;
        if (!parse_double(token, p) || !(p > 0.0 && p <= 1.0)) {
            err = "invalid p-value in p-file: " + token;
            return false;
        }
        values.push_back(p);
    }
    if (values.empty()) {
        err = "p-file contains no values: " + path;
        return false;
    }
    out = std::move(values);
    return true;
}

void canonicalize_p_values(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}



void apply_quick_preset(RunOptions& opt) {
    opt.N = 48;
    opt.instances = 2;
    opt.p_values = {0.10, 0.25, 0.50, 1.00};
    opt.solver.subset_restarts = 1;
    opt.solver.tsp_restarts = 1;
    opt.solver.tsp_ils = 20;
    opt.solver.tsp_patience = 8;
    opt.solver.sa_iters = 250;
    opt.solver.pair_exchange_passes = 0;
    opt.solver.ruin_recreate_rounds = 1;
    opt.solver.path_relink_top = 0;
}

bool parse_bool_literal(const std::string& text, bool& out) {
    std::string value;
    value.reserve(text.size());
    for (char ch : text) {
        if (ch >= 'A' && ch <= 'Z') {
            value.push_back(static_cast<char>(ch - 'A' + 'a'));
        } else {
            value.push_back(ch);
        }
    }
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        out = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        out = false;
        return true;
    }
    return false;
}

bool bool_flag_value(const std::string& arg, const std::string& flag, bool& matched, bool& value) {
    matched = false;
    value = true;
    if (arg == flag) {
        matched = true;
        return true;
    }
    const std::string prefix = flag + "=";
    if (arg.rfind(prefix, 0) != 0) {
        return true;
    }
    matched = true;
    const std::string literal = arg.substr(prefix.size());
    if (!parse_bool_literal(literal, value)) {
        std::fprintf(stderr,
                     "Invalid boolean value for %s: %s (expected true/false, yes/no, on/off, or 1/0)\n",
                     flag.c_str(),
                     literal.c_str());
        return false;
    }
    return true;
}
std::string p_key(double p) {
    std::ostringstream out;
    out << std::setprecision(17) << p;
    return out.str();
}

void print_help(const char* argv0) {
    const char* exe = (argv0 != nullptr && *argv0 != '\0') ? argv0 : "aldous_tsp";
    std::fprintf(stdout, "Usage: %s [options]\n\n", exe);
    std::fputs(R"(General:
  --help                         Show this help text and exit
  --self-test                    Run built-in smoke/self tests
  --quick[=true|false]             Small preset for fast runs. Explicit flags override it regardless of order
  --dry-run                      Print resolved config and exit
  --dump-config                  Print resolved config before running
  --output <file>                Output JSON path (default: results.json)
  --force[=true|false]             Overwrite an existing output file
  --verbose-p[=true|false]         Print per-instance p-value progress
  --include-instance-rows[=true|false]
                                  Include per-instance values and stats in JSON

Simulation:
  --N <int>                      Number of random points (default: 500)
  --instances <int>              Monte Carlo instances (default: 15)
  --threads <int>                Worker threads (default: auto; 0 = auto)
  --seed <int>                   Base random seed (default: 2024)
  --p-values <csv>               Comma-separated p grid, e.g. 0.02,0.05,1
  --p-range <a:b:n>              Linear p grid from a to b with n values
  --p-file <file>                Read whitespace/comma separated p values

Solver:
  --mode <name>                  balanced | smallp-region | highp-delete | hybrid
  --knn <int>                    Exact KNN candidate count (default: min(40,N-1))
  --knn-backend <name>           grid | bruteforce (default: grid)
  --grid-cell <float>            Force grid cell size for grid KNN
  --verify-knn <int>             Sampled KNN verification checks
  --restarts <int>               Subset restarts. Values >= 1 run exactly that
                                 many. Default: auto = 8 when p <= 0.08, else 3
                                 (the historical effective behavior)
  --continuation-restarts <int>  Warm restarts when a neighboring-p parent exists
                                 (default: 1)
  --continuation-policy <name>   supplemental | fixed-budget (default: supplemental).
                                 Supplemental appends warm work without replacing
                                 independent draws; fixed-budget reserves a quota.
  --racing-candidates <int>      Supplemental candidates screened by deterministic
                                 restart racing (default: 0 = disabled)
  --racing-survivors <int>       Pilot candidates promoted to a full-depth rerun
                                 (default: 2)
  --racing-pilot-iters <int>     SA iterations used to screen each race candidate;
                                 capped at the full restart budget (default: 2000)
  --racing-min-jaccard <float>   Minimum selected-set Jaccard distance preferred
                                 between promoted candidates (default: 0.05)
  --sa-iters <int>               Subset SA iteration budget (default: 60000)
  --sa-iters-per-k <int>         Extra SA iterations per subset element k (default: 0)
  --sa-iters-per-n <int>         Extra SA iterations per CANDIDATE point N (default: 0).
                                 The subset search picks k of N, so its budget must scale
                                 with N; ~10-25 is where L/k stops improving.
  --sa-exact-insertion[=true|false]  Use the historical O(k) global-best insertion
                                 scan per SA move (default: false = windowed O(1))
  --sa-insertion-window <int>    Tour-position radius for windowed insertion (default: 12)
  --dense-exact-insertion[=true|false]
                                 Force the exact O(k) scan on dense seeds too, as 0.9.5 did
                                 (bit-identical search at k=2000, +58% wall; default: false)
  --sa-spatial-insertion[=true|false]
                                 Insertion slots come from the incoming node's nearest CURRENT
                                 subset members via a live spatial index (default: false)
  --sa-spatial-neighbors <int>   How many nearest members offer slots (default: 16)
  --exploration-exact-insertion[=true|false]
                                 Exploration seeds (random/high-p) use the exact O(k)
                                 insertion scan; windowed insertion cannot relocate a
                                 spread-out subset (default: true)
  --small-p-dense-fill[=true|false]
                                 At p<=0.08 fill the restart pool with dense_seed variants
                                 instead of alternating in random subsets, which never
                                 contract at small p (default: true)
  --region-seeds[=true|false]     Replace pooled random seeds with fresh per-restart region
                                 seeds: uniform center, k nearest points (default: false)
  --region-dilation <float>      Region seed draws k points from the (dilation*k) nearest
                                 to the center (default: 3.0)
  --kick-restarts <int>          Of the scheduled restarts, run the LAST n as elite-kick
                                 restarts: seed = perturbed best of the independent phase,
                                 annealed at --kick-t0 (default: 0 = pure multistart)
  --kick-fraction <float>        Fraction of members swapped by a kick (default: 0.10)
  --kick-t0 <float>              SA start temperature for elite-seeded restarts (default: 0.35)
  --sa-t0 <float>                SA start temperature (default: 1.4)
  --sa-t1 <float>                SA end temperature (default: 0.00005)
  --time-budget-per-p <sec>      Wall-clock target per (instance, p) solve; keeps
                                 launching restarts until elapsed (default: 0 = off)
  --restart-threads <int>        Threads for parallel subset restarts within one
                                 (instance, p) solve; 0 = auto from leftover thread
                                 budget, 1 = sequential (default: 0). Results are
                                 invariant to this outside --time-budget-per-p mode.
  --second-sweep[=bool]          After the descending warm-start sweep over p, run an
                                 ascending sweep seeded from smaller p and keep the
                                 better result per p (default: off; ~2x subset time)
  --periodic[=bool]              Generate instances on a flat torus (periodic boundary
                                 conditions) instead of the open square. Removes the
                                 O(1/sqrt N) boundary term in f(p,N); same limit f(p),
                                 O(1/N) convergence (default: off)
  --control-variate[=bool]       Compute the two-nearest-neighbor lower bound per
                                 instance: on the full set (known-mean control variate,
                                 variance-reduces the mean; large effect at p=1) and on
                                 the selected subset. The subset value bounds only the
                                 tour through that subset, not the optimum over all
                                 size-k subsets (default: off)
  --cv-mc-samples <int>          Cheap Monte-Carlo instances used to pin E[B_full] for
                                 the control variate (default: 2000; capped at large N)
  --held-karp[=bool]             Compute the Held-Karp (Lagrangian 1-tree) lower bound
                                 per selected subset. Its gap certifies tour-ordering
                                 quality conditional on that subset; it is not a global
                                 bound on the best size-k subset. O(k^2) per solve;
                                 intended for moderate k (default: off)
  --hk-iterations <int>          Subgradient iterations for the Held-Karp bound
                                 (default: 400)
  --exact-subset-max-n <int>     Globally solve subset choice and tour when N is at
                                 most this value (default: 0 = off; hard max: 18)
  --tsp-restarts <int>           Full TSP restarts (default: 5)
  --tsp-ils <int>                Full-TSP ILS perturbation iterations
  --tsp-patience <int>           Full-TSP ILS stagnation patience
  --final-exhaustive-k <int>     Exhaustive final 2-opt threshold (default: 300)
  --exhaustive-two-opt-policy <p> never | final-only | all-polish (default: final-only)
  --subset-swap-passes <int>     Deterministic subset swap descent passes
  --pair-exchange-passes <int>   Two-for-two subset exchange passes
  --pair-exchange-max-k <int>    Skip pair exchange above k (default: 5000; 0 = unlimited)
  --ruin-recreate-rounds <int>   LNS ruin/recreate rounds
  --adaptive-ruin-recreate[=bool]
                                 Use multi-scale operator portfolio (default: true)
  --ruin-recreate-max-fraction <x>
                                 Maximum ruined fraction of k (default: 0.05)
  --ruin-recreate-max-nodes <int>
                                 Absolute ruin-size cap (default: 96; 0 = unlimited)
  --ruin-recreate-pool-cap <int> Candidate repair pool cap (default: 640)
  --ejection-chain-starts <int>  Variable-depth chain starts per restart (default: 3)
  --ejection-chain-depth <int>   Maximum membership swaps per chain (default: 6)
  --ejection-chain-candidates <int>
                                 Incoming-node candidates per chain step (default: 24)
  --ejection-chain-remove-cap <int>
                                 Removable members per chain step (default: 96; 0 = all)
  --ejection-chain-max-uphill <x>
                                 Cumulative uphill allowance in mean-edge units (default: 0.75)
  --elite-diversity-slots <int>  Supplemental set-diverse archive slots (default: 4)
  --elite-min-jaccard <x>        Minimum Jaccard distance for diverse slots (default: 0.02)
  --elite-quality-slack <x>      Relative quality window for diverse slots (default: 0.03)
  --path-relink-top <int>        Elite-pool path relinking width

External oracle:
  --oracle <name>                none | auto | lkh | concorde (default: none)
  --oracle-format <name>         matrix | euc2d (default: matrix)
  --lkh-path <path>              LKH executable path/name (default: LKH)
  --concorde-path <path>         Concorde executable path/name (default: concorde)
  --oracle-time-limit <int>      Per-call timeout in seconds (0 = no timeout)
  --oracle-scale <int>           TSPLIB integer scaling factor
  --oracle-tsp-top <int>         Number of full-TSP elite candidates to post-process
  --oracle-subset-top <int>      Number of subset elite candidates to post-process
  --oracle-min-k <int>           Minimum k for oracle use
  --oracle-max-k <int>           Maximum k for oracle use
  --oracle-lkh-runs <int>        LKH RUNS parameter
  --oracle-lkh-trials <int>      Optional LKH MAX_TRIALS override
  --oracle-no-tsp[=true|false]    Disable oracle for p=1/full TSP
  --oracle-no-subset[=true|false] Disable oracle for subset TSP
  --oracle-inline-feedback[=true|false]
                                  Also polish elite candidates during search
  --oracle-verbose[=true|false]   Forward external solver output to stderr

Ablation / diagnostics:
  --disable-two-opt[=true|false]
  --disable-or-opt[=true|false]
  --disable-subset-swap[=true|false]
  --disable-pair-exchange[=true|false]
  --disable-ruin-recreate[=true|false]
  --disable-ejection-chain[=true|false]
  --disable-path-relink[=true|false]
  --disable-smallp-seeds[=true|false]
  --disable-highp-delete[=true|false]

Boolean flags accept plain presence as true, or explicit =true/=false
(using true/false, yes/no, on/off, or 1/0).

)", stdout);
}

std::string config_summary(const RunOptions& opt) {
    std::ostringstream out;
    out << "N=" << opt.N
        << ", instances=" << opt.instances
        << ", threads=" << opt.threads
        << ", seed=" << opt.solver.seed
        << ", mode=" << solver_mode_name(opt.solver.mode)
        << ", knn=" << opt.solver.knn_k
        << ", knn_backend=" << knn_backend_name(opt.solver.knn_backend)
        << ", verify_knn_checks=" << opt.solver.verify_knn_checks
        << ", subset_restarts=" << (opt.solver.subset_restarts >= 1 ? std::to_string(opt.solver.subset_restarts) : std::string("auto"))
        << ", continuation_restarts=" << opt.solver.continuation_restarts
        << ", continuation_policy=" << continuation_policy_name(opt.solver.continuation_policy)
        << ", racing_candidates=" << opt.solver.racing_candidates
        << ", racing_survivors=" << opt.solver.racing_survivors
        << ", racing_pilot_iters=" << opt.solver.racing_pilot_iters
        << ", racing_min_jaccard=" << opt.solver.racing_min_jaccard
        << ", sa_iters=" << opt.solver.sa_iters
        << ", sa_iters_per_k=" << opt.solver.sa_iters_per_k
        << ", sa_iters_per_n=" << opt.solver.sa_iters_per_n
        << ", sa_insertion=" << (opt.solver.sa_exact_insertion ? std::string("exact") : ("window" + std::to_string(opt.solver.sa_insertion_window)))
        << ", dense_exact_insertion=" << (opt.solver.dense_exact_insertion ? "on" : "off")
        << ", sa_spatial_insertion=" << (opt.solver.sa_spatial_insertion ? "on" : "off")
        << ", sa_spatial_neighbors=" << opt.solver.sa_spatial_neighbors
        << ", exploration_insertion=" << (opt.solver.exploration_exact_insertion ? "exact" : "windowed")
        << ", small_p_dense_fill=" << (opt.solver.small_p_dense_fill ? "on" : "off")
        << ", region_seeds=" << (opt.solver.region_seeds ? "on" : "off")
        << ", region_dilation=" << opt.solver.region_dilation
        << ", kick_restarts=" << opt.solver.subset_kick_restarts
        << ", kick_fraction=" << opt.solver.kick_fraction
        << ", kick_t0=" << opt.solver.kick_t0
        << ", sa_t0=" << opt.solver.sa_t0
        << ", sa_t1=" << opt.solver.sa_t1
        << ", time_budget_per_p=" << opt.solver.time_budget_per_p
        << ", restart_threads=" << (opt.solver.restart_threads <= 0 ? std::string("auto") : std::to_string(opt.solver.restart_threads))
        << ", second_sweep=" << (opt.second_sweep ? "true" : "false")
        << ", periodic=" << (opt.periodic ? "true" : "false")
        << ", control_variate=" << (opt.control_variate ? "true" : "false")
        << ", cv_mc_samples=" << opt.cv_mc_samples
        << ", held_karp=" << (opt.held_karp ? "true" : "false")
        << ", hk_iterations=" << opt.hk_iterations
        << ", exact_subset_max_n=" << opt.solver.exact_subset_max_n
        << ", tsp_restarts=" << opt.solver.tsp_restarts
        << ", final_exhaustive_k=" << opt.solver.final_exhaustive_k
        << ", pair_exchange_max_k=" << opt.solver.pair_exchange_max_k
        << ", ejection_chain_starts=" << opt.solver.ejection_chain_starts
        << ", ejection_chain_depth=" << opt.solver.ejection_chain_depth
        << ", exhaustive_two_opt_policy=" << exhaustive_two_opt_policy_name(opt.solver.exhaustive_two_opt_policy)
        << ", oracle=" << opt.solver.oracle.status
        << ", p_values=";
    for (std::size_t i = 0; i < opt.p_values.size(); ++i) {
        if (i != 0U) {
            out << ',';
        }
        out << std::setprecision(17) << opt.p_values[i];
    }
    return out.str();
}

bool validate_options(RunOptions& opt, std::string& err) {
    if (opt.p_values.empty()) {
        opt.p_values = default_p_values();
    }
    canonicalize_p_values(opt.p_values);
    if (opt.N < 3) { err = "--N must be >= 3"; return false; }
    if (opt.instances < 1) { err = "--instances must be >= 1"; return false; }
    if (opt.solver.subset_restarts < 1 && opt.solver.subset_restarts != -1) { err = "--restarts must be >= 1 (or omit it for auto)"; return false; }
    if (opt.solver.continuation_restarts < 0) { err = "--continuation-restarts must be >= 0"; return false; }
    if (opt.solver.racing_candidates < 0) { err = "--racing-candidates must be >= 0"; return false; }
    if (opt.solver.racing_survivors < 1) { err = "--racing-survivors must be >= 1"; return false; }
    if (opt.solver.racing_candidates > 0
        && opt.solver.racing_survivors > opt.solver.racing_candidates) {
        err = "--racing-survivors must not exceed --racing-candidates";
        return false;
    }
    if (opt.solver.racing_pilot_iters < 0) { err = "--racing-pilot-iters must be >= 0"; return false; }
    if (!std::isfinite(opt.solver.racing_min_jaccard)
        || opt.solver.racing_min_jaccard < 0.0
        || opt.solver.racing_min_jaccard > 1.0) {
        err = "--racing-min-jaccard must be finite and in [0,1]";
        return false;
    }
    if (opt.solver.exact_subset_max_n < 0
        || opt.solver.exact_subset_max_n > kExactSubsetHardLimit) {
        err = "--exact-subset-max-n must be in [0,18]";
        return false;
    }
    if (opt.solver.tsp_restarts < 1) { err = "--tsp-restarts must be >= 1"; return false; }
    if (opt.solver.sa_iters < 0) { err = "--sa-iters must be >= 0"; return false; }
    if (opt.solver.sa_iters_per_k < 0) { err = "--sa-iters-per-k must be >= 0"; return false; }
    if (opt.solver.sa_iters_per_n < 0) { err = "--sa-iters-per-n must be >= 0"; return false; }
    if (opt.solver.sa_insertion_window < 1) { err = "--sa-insertion-window must be >= 1"; return false; }
    if (opt.solver.sa_spatial_neighbors < 1) { err = "--sa-spatial-neighbors must be >= 1"; return false; }
    if (opt.solver.region_dilation < 1.0) { err = "--region-dilation must be >= 1"; return false; }
    if (opt.solver.subset_kick_restarts < 0) { err = "--kick-restarts must be >= 0"; return false; }
    if (!(opt.solver.kick_fraction > 0.0 && opt.solver.kick_fraction < 1.0)) { err = "--kick-fraction must be in (0,1)"; return false; }
    if (!(opt.solver.kick_t0 > 0.0)) { err = "--kick-t0 must be > 0"; return false; }
    if (opt.solver.sa_t0 <= 0.0 || !std::isfinite(opt.solver.sa_t0)) { err = "--sa-t0 must be finite and > 0"; return false; }
    if (opt.solver.sa_t1 <= 0.0 || !std::isfinite(opt.solver.sa_t1)) { err = "--sa-t1 must be finite and > 0"; return false; }
    if (opt.solver.time_budget_per_p < 0.0 || !std::isfinite(opt.solver.time_budget_per_p)) { err = "--time-budget-per-p must be finite and >= 0"; return false; }
    if (opt.solver.racing_candidates > 0 && opt.solver.time_budget_per_p > 0.0) {
        err = "deterministic restart racing is incompatible with --time-budget-per-p";
        return false;
    }
    if (opt.solver.restart_threads < 0) { err = "--restart-threads must be >= 0"; return false; }
    if (opt.solver.final_exhaustive_k < 0) { err = "--final-exhaustive-k must be >= 0"; return false; }
    if (opt.solver.subset_swap_descent_passes < 0) { err = "--subset-swap-passes must be >= 0"; return false; }
    if (opt.solver.pair_exchange_passes < 0) { err = "--pair-exchange-passes must be >= 0"; return false; }
    if (opt.solver.pair_exchange_max_k < 0) { err = "--pair-exchange-max-k must be >= 0"; return false; }
    if (opt.solver.ruin_recreate_rounds < 0) { err = "--ruin-recreate-rounds must be >= 0"; return false; }
    if (!std::isfinite(opt.solver.ruin_recreate_max_fraction)
        || opt.solver.ruin_recreate_max_fraction < 0.0
        || opt.solver.ruin_recreate_max_fraction > 1.0) {
        err = "--ruin-recreate-max-fraction must be finite and in [0,1]";
        return false;
    }
    if (opt.solver.ruin_recreate_max_nodes < 0) { err = "--ruin-recreate-max-nodes must be >= 0"; return false; }
    if (opt.solver.ruin_recreate_pool_cap < 1) { err = "--ruin-recreate-pool-cap must be >= 1"; return false; }
    if (opt.solver.ejection_chain_starts < 0) { err = "--ejection-chain-starts must be >= 0"; return false; }
    if (opt.solver.ejection_chain_depth < 0) { err = "--ejection-chain-depth must be >= 0"; return false; }
    if (opt.solver.ejection_chain_candidates < 0) { err = "--ejection-chain-candidates must be >= 0"; return false; }
    if (opt.solver.ejection_chain_remove_cap < 0) { err = "--ejection-chain-remove-cap must be >= 0"; return false; }
    if (!std::isfinite(opt.solver.ejection_chain_max_uphill)
        || opt.solver.ejection_chain_max_uphill < 0.0) {
        err = "--ejection-chain-max-uphill must be finite and >= 0";
        return false;
    }
    if (opt.solver.elite_diversity_slots < 0) { err = "--elite-diversity-slots must be >= 0"; return false; }
    if (!std::isfinite(opt.solver.elite_min_jaccard)
        || opt.solver.elite_min_jaccard < 0.0
        || opt.solver.elite_min_jaccard > 1.0) {
        err = "--elite-min-jaccard must be finite and in [0,1]";
        return false;
    }
    if (!std::isfinite(opt.solver.elite_quality_slack)
        || opt.solver.elite_quality_slack < 0.0) {
        err = "--elite-quality-slack must be finite and >= 0";
        return false;
    }
    if (opt.solver.path_relink_top < 0) { err = "--path-relink-top must be >= 0"; return false; }
    if (opt.solver.verify_knn_checks < 0) { err = "--verify-knn must be >= 0"; return false; }
    if (opt.solver.grid_cell < 0.0 || !std::isfinite(opt.solver.grid_cell)) { err = "--grid-cell must be finite and >= 0"; return false; }
    if (opt.solver.tsp_ils < 0) { err = "--tsp-ils must be >= 0"; return false; }
    if (opt.solver.tsp_patience < 0) { err = "--tsp-patience must be >= 0"; return false; }
    if (opt.solver.oracle.cfg.time_limit_sec < 0) { err = "--oracle-time-limit must be >= 0"; return false; }
    if (opt.solver.oracle.cfg.scale < 1) { err = "--oracle-scale must be >= 1"; return false; }
    if (opt.solver.oracle.cfg.tsp_top < 0) { err = "--oracle-tsp-top must be >= 0"; return false; }
    if (opt.solver.oracle.cfg.subset_top < 0) { err = "--oracle-subset-top must be >= 0"; return false; }
    if (opt.solver.oracle.cfg.min_k < 3) { err = "--oracle-min-k must be >= 3"; return false; }
    if (opt.solver.oracle.cfg.max_k < opt.solver.oracle.cfg.min_k) { err = "--oracle-max-k must be >= --oracle-min-k"; return false; }
    if (opt.solver.oracle.cfg.lkh_runs < 1) { err = "--oracle-lkh-runs must be >= 1"; return false; }
    if (opt.solver.oracle.cfg.lkh_max_trials < 0) { err = "--oracle-lkh-trials must be >= 0"; return false; }
    if (opt.hk_iterations < 1) { err = "--hk-iterations must be >= 1"; return false; }
    if (opt.periodic
        && opt.solver.oracle.cfg.mode != ExternalOracleMode::None
        && opt.solver.oracle.cfg.problem_format == OracleProblemFormat::Euc2d) {
        // EUC_2D hands the external solver raw coordinates, so it would optimize
        // open-plane distances -- ignoring the torus wrap -- while the tour is
        // scored with the periodic metric. Only the EXPLICIT full matrix (which
        // serializes the torus distances) is correct under --periodic.
        err = "--periodic requires --oracle-format matrix (euc2d ignores the torus wrap)";
        return false;
    }
    if (opt.solver.knn_k <= 0) {
        opt.solver.knn_k = std::min(40, opt.N - 1);
    }
    opt.solver.knn_k = std::max(1, std::min(opt.solver.knn_k, opt.N - 1));
    if (opt.threads < 0) { err = "--threads must be >= 0 (0 means auto)"; return false; }
    const unsigned hw = std::thread::hardware_concurrency();
    const int auto_threads = hw == 0U ? 1 : static_cast<int>(hw);
    if (opt.threads == 0) {
        opt.threads = auto_threads;
    }
    opt.threads = std::max(1, std::min(opt.threads, opt.instances));
    return true;
}

bool parse_args(int argc, char** argv, RunOptions& opt, bool& self_test) {
    self_test = false;

    bool quick_requested = false;
    for (int qi = 1; qi < argc; ++qi) {
        bool matched = false;
        bool value = true;
        if (!bool_flag_value(argv[qi], "--quick", matched, value)) {
            return false;
        }
        if (matched) {
            quick_requested = value;
        }
    }
    if (quick_requested) {
        apply_quick_preset(opt);
    }

    auto value_for = [&](int& i, const std::string& flag, std::string& value) -> bool {
        const std::string arg = argv[i];
        const std::string prefix = flag + "=";
        if (arg.rfind(prefix, 0) == 0) {
            value = arg.substr(prefix.size());
            return true;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Missing value for %s\n", flag.c_str());
            return false;
        }
        value = argv[++i];
        return true;
    };
    auto parse_int_flag = [&](int& i, const std::string& flag, int& target) -> bool {
        std::string value;
        if (!value_for(i, flag, value)) { return false; }
        if (!parse_int(value, target)) {
            std::fprintf(stderr, "Invalid integer for %s: %s\n", flag.c_str(), value.c_str());
            return false;
        }
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string flag = arg;
        const std::size_t eq = flag.find('=');
        if (eq != std::string::npos) {
            flag = flag.substr(0, eq);
        }
        auto handle_bool = [&](const std::string& name, bool& target) -> int {
            bool matched = false;
            bool value = true;
            if (!bool_flag_value(arg, name, matched, value)) {
                return -1;
            }
            if (matched) {
                target = value;
                return 1;
            }
            return 0;
        };
        auto handle_inverse_bool = [&](const std::string& name, bool& target) -> int {
            bool matched = false;
            bool value = true;
            if (!bool_flag_value(arg, name, matched, value)) {
                return -1;
            }
            if (matched) {
                target = !value;
                return 1;
            }
            return 0;
        };
        auto consume_bool = [&](int result) -> bool {
            if (result < 0) {
                throw std::runtime_error("invalid boolean flag");
            }
            return result > 0;
        };

        bool matched_bool = false;
        bool bool_value = true;
        if (!bool_flag_value(arg, "--help", matched_bool, bool_value)) { return false; }
        if (matched_bool) { if (bool_value) { print_help(argv[0]); std::exit(0); } continue; }
        if (!bool_flag_value(arg, "--self-test", matched_bool, bool_value)) { return false; }
        if (matched_bool) { self_test = bool_value; continue; }
        if (!bool_flag_value(arg, "--quick", matched_bool, bool_value)) { return false; }
        if (matched_bool) { continue; }
        try {
            if (consume_bool(handle_bool("--dry-run", opt.dry_run))) { continue; }
            if (consume_bool(handle_bool("--dump-config", opt.dump_config))) { continue; }
            if (consume_bool(handle_bool("--force", opt.force_output))) { continue; }
            if (consume_bool(handle_bool("--verbose-p", opt.verbose))) { continue; }
            if (consume_bool(handle_bool("--dense-exact-insertion", opt.solver.dense_exact_insertion))) { continue; }
            if (consume_bool(handle_bool("--sa-spatial-insertion", opt.solver.sa_spatial_insertion))) { continue; }
            if (consume_bool(handle_bool("--exploration-exact-insertion", opt.solver.exploration_exact_insertion))) { continue; }
            if (consume_bool(handle_bool("--small-p-dense-fill", opt.solver.small_p_dense_fill))) { continue; }
            if (consume_bool(handle_bool("--region-seeds", opt.solver.region_seeds))) { continue; }
            if (consume_bool(handle_bool("--include-instance-rows", opt.include_instance_rows))) { continue; }
            if (consume_bool(handle_bool("--second-sweep", opt.second_sweep))) { continue; }
            if (consume_bool(handle_bool("--periodic", opt.periodic))) { continue; }
            if (consume_bool(handle_bool("--control-variate", opt.control_variate))) { continue; }
            if (consume_bool(handle_bool("--held-karp", opt.held_karp))) { continue; }
            if (consume_bool(handle_bool("--disable-two-opt", opt.solver.disable_two_opt))) { continue; }
            if (consume_bool(handle_bool("--disable-or-opt", opt.solver.disable_or_opt))) { continue; }
            if (consume_bool(handle_bool("--disable-subset-swap", opt.solver.disable_subset_swap))) { continue; }
            if (consume_bool(handle_bool("--disable-pair-exchange", opt.solver.disable_pair_exchange))) { continue; }
            if (consume_bool(handle_bool("--disable-elite-restarts", opt.solver.disable_elite_restarts))) { continue; }
            if (consume_bool(handle_bool("--disable-ruin-recreate", opt.solver.disable_ruin_recreate))) { continue; }
            if (consume_bool(handle_bool("--adaptive-ruin-recreate", opt.solver.adaptive_ruin_recreate))) { continue; }
            if (consume_bool(handle_bool("--disable-ejection-chain", opt.solver.disable_ejection_chain))) { continue; }
            if (consume_bool(handle_bool("--disable-path-relink", opt.solver.disable_path_relink))) { continue; }
            if (consume_bool(handle_bool("--disable-smallp-seeds", opt.solver.disable_smallp_seeds))) { continue; }
            if (consume_bool(handle_bool("--disable-highp-delete", opt.solver.disable_highp_delete))) { continue; }
        } catch (const std::runtime_error&) {
            return false;
        }

        if (flag == "--N") { if (!parse_int_flag(i, flag, opt.N)) { return false; } continue; }
        if (flag == "--instances") { if (!parse_int_flag(i, flag, opt.instances)) { return false; } continue; }
        if (flag == "--threads") { if (!parse_int_flag(i, flag, opt.threads)) { return false; } continue; }
        if (flag == "--seed") { if (!parse_int_flag(i, flag, opt.solver.seed)) { return false; } continue; }
        if (flag == "--knn") { if (!parse_int_flag(i, flag, opt.solver.knn_k)) { return false; } continue; }
        if (flag == "--knn-backend") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_knn_backend(value, opt.solver.knn_backend)) {
                std::fprintf(stderr, "Invalid --knn-backend: %s\n", value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--grid-cell") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_double(value, opt.solver.grid_cell)) {
                std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--restarts") { if (!parse_int_flag(i, flag, opt.solver.subset_restarts)) { return false; } continue; }
        if (flag == "--continuation-restarts") {
            if (!parse_int_flag(i, flag, opt.solver.continuation_restarts)) { return false; }
            continue;
        }
        if (flag == "--continuation-policy") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_continuation_policy(value, opt.solver.continuation_policy)) {
                std::fprintf(stderr, "Invalid --continuation-policy: %s\n", value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--racing-candidates") {
            if (!parse_int_flag(i, flag, opt.solver.racing_candidates)) { return false; }
            continue;
        }
        if (flag == "--racing-survivors") {
            if (!parse_int_flag(i, flag, opt.solver.racing_survivors)) { return false; }
            continue;
        }
        if (flag == "--racing-pilot-iters") {
            if (!parse_int_flag(i, flag, opt.solver.racing_pilot_iters)) { return false; }
            continue;
        }
        if (flag == "--racing-min-jaccard") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_double(value, opt.solver.racing_min_jaccard)) {
                std::fprintf(stderr, "Invalid floating-point value for %s: %s\n",
                             flag.c_str(), value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--cv-mc-samples") { if (!parse_int_flag(i, flag, opt.cv_mc_samples)) { return false; } continue; }
        if (flag == "--hk-iterations") { if (!parse_int_flag(i, flag, opt.hk_iterations)) { return false; } continue; }
        if (flag == "--sa-iters") { if (!parse_int_flag(i, flag, opt.solver.sa_iters)) { return false; } continue; }
        if (flag == "--sa-iters-per-k") { if (!parse_int_flag(i, flag, opt.solver.sa_iters_per_k)) { return false; } continue; }
        if (flag == "--sa-iters-per-n") { if (!parse_int_flag(i, flag, opt.solver.sa_iters_per_n)) { return false; } continue; }
        if (flag == "--sa-insertion-window") { if (!parse_int_flag(i, flag, opt.solver.sa_insertion_window)) { return false; } continue; }
        if (flag == "--region-dilation") { std::string v; if (!value_for(i, flag, v)) { return false; } if (!parse_double(v, opt.solver.region_dilation)) { std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), v.c_str()); return false; } continue; }
        if (flag == "--sa-spatial-neighbors") { if (!parse_int_flag(i, flag, opt.solver.sa_spatial_neighbors)) { return false; } continue; }
        if (flag == "--kick-restarts") { if (!parse_int_flag(i, flag, opt.solver.subset_kick_restarts)) { return false; } continue; }
        if (flag == "--kick-fraction") { std::string v; if (!value_for(i, flag, v)) { return false; } if (!parse_double(v, opt.solver.kick_fraction)) { std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), v.c_str()); return false; } continue; }
        if (flag == "--kick-t0") { std::string v; if (!value_for(i, flag, v)) { return false; } if (!parse_double(v, opt.solver.kick_t0)) { std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), v.c_str()); return false; } continue; }
        try {
            if (consume_bool(handle_bool("--sa-exact-insertion", opt.solver.sa_exact_insertion))) { continue; }
        } catch (const std::runtime_error&) {
            return false;
        }
        if (flag == "--sa-t0") { std::string v; if (!value_for(i, flag, v)) { return false; } if (!parse_double(v, opt.solver.sa_t0)) { std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), v.c_str()); return false; } continue; }
        if (flag == "--sa-t1") { std::string v; if (!value_for(i, flag, v)) { return false; } if (!parse_double(v, opt.solver.sa_t1)) { std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), v.c_str()); return false; } continue; }
        if (flag == "--restart-threads") { if (!parse_int_flag(i, flag, opt.solver.restart_threads)) { return false; } continue; }
        if (flag == "--time-budget-per-p") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_double(value, opt.solver.time_budget_per_p)) {
                std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--exact-subset-max-n") { if (!parse_int_flag(i, flag, opt.solver.exact_subset_max_n)) { return false; } continue; }
        if (flag == "--tsp-restarts") { if (!parse_int_flag(i, flag, opt.solver.tsp_restarts)) { return false; } continue; }
        if (flag == "--tsp-ils") { if (!parse_int_flag(i, flag, opt.solver.tsp_ils)) { return false; } continue; }
        if (flag == "--tsp-patience") { if (!parse_int_flag(i, flag, opt.solver.tsp_patience)) { return false; } continue; }
        if (flag == "--final-exhaustive-k") { if (!parse_int_flag(i, flag, opt.solver.final_exhaustive_k)) { return false; } continue; }
        if (flag == "--exhaustive-two-opt-policy") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_exhaustive_two_opt_policy(value, opt.solver.exhaustive_two_opt_policy)) {
                std::fprintf(stderr, "Invalid --exhaustive-two-opt-policy: %s\n", value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--subset-swap-passes") { if (!parse_int_flag(i, flag, opt.solver.subset_swap_descent_passes)) { return false; } continue; }
        if (flag == "--pair-exchange-passes") { if (!parse_int_flag(i, flag, opt.solver.pair_exchange_passes)) { return false; } continue; }
        if (flag == "--pair-exchange-max-k") { if (!parse_int_flag(i, flag, opt.solver.pair_exchange_max_k)) { return false; } continue; }
        if (flag == "--ruin-recreate-rounds") { if (!parse_int_flag(i, flag, opt.solver.ruin_recreate_rounds)) { return false; } continue; }
        if (flag == "--ruin-recreate-max-fraction") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_double(value, opt.solver.ruin_recreate_max_fraction)) {
                std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--ruin-recreate-max-nodes") { if (!parse_int_flag(i, flag, opt.solver.ruin_recreate_max_nodes)) { return false; } continue; }
        if (flag == "--ruin-recreate-pool-cap") { if (!parse_int_flag(i, flag, opt.solver.ruin_recreate_pool_cap)) { return false; } continue; }
        if (flag == "--ejection-chain-starts") { if (!parse_int_flag(i, flag, opt.solver.ejection_chain_starts)) { return false; } continue; }
        if (flag == "--ejection-chain-depth") { if (!parse_int_flag(i, flag, opt.solver.ejection_chain_depth)) { return false; } continue; }
        if (flag == "--ejection-chain-candidates") { if (!parse_int_flag(i, flag, opt.solver.ejection_chain_candidates)) { return false; } continue; }
        if (flag == "--ejection-chain-remove-cap") { if (!parse_int_flag(i, flag, opt.solver.ejection_chain_remove_cap)) { return false; } continue; }
        if (flag == "--ejection-chain-max-uphill") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_double(value, opt.solver.ejection_chain_max_uphill)) {
                std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--elite-diversity-slots") { if (!parse_int_flag(i, flag, opt.solver.elite_diversity_slots)) { return false; } continue; }
        if (flag == "--elite-min-jaccard") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_double(value, opt.solver.elite_min_jaccard)) {
                std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--elite-quality-slack") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_double(value, opt.solver.elite_quality_slack)) {
                std::fprintf(stderr, "Invalid floating-point value for %s: %s\n", flag.c_str(), value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--path-relink-top") { if (!parse_int_flag(i, flag, opt.solver.path_relink_top)) { return false; } continue; }
        if (flag == "--verify-knn") {
            if (!parse_int_flag(i, flag, opt.solver.verify_knn_checks)) { return false; }
            continue;
        }
        if (flag == "--mode") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_solver_mode(value, opt.solver.mode)) {
                std::fprintf(stderr, "Invalid --mode: %s\n", value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--output") {
            if (!value_for(i, flag, opt.output_path)) { return false; }
            continue;
        }
        if (flag == "--p-values") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_p_values(value, opt.p_values)) {
                std::fprintf(stderr, "Invalid --p-values: %s\n", value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--p-range") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_p_range(value, opt.p_values)) {
                std::fprintf(stderr, "Invalid --p-range: %s\n", value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--p-file") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            std::string err;
            if (!read_p_file(value, opt.p_values, err)) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--oracle") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_external_oracle_mode(value, opt.solver.oracle.cfg.mode)) {
                std::fprintf(stderr, "Invalid --oracle: %s\n", value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--oracle-format") {
            std::string value;
            if (!value_for(i, flag, value)) { return false; }
            if (!parse_oracle_problem_format(value, opt.solver.oracle.cfg.problem_format)) {
                std::fprintf(stderr, "Invalid --oracle-format: %s\n", value.c_str());
                return false;
            }
            continue;
        }
        if (flag == "--lkh-path") { if (!value_for(i, flag, opt.solver.oracle.cfg.lkh_path)) { return false; } continue; }
        if (flag == "--concorde-path") { if (!value_for(i, flag, opt.solver.oracle.cfg.concorde_path)) { return false; } continue; }
        if (flag == "--oracle-time-limit") { if (!parse_int_flag(i, flag, opt.solver.oracle.cfg.time_limit_sec)) { return false; } continue; }
        if (flag == "--oracle-scale") { if (!parse_int_flag(i, flag, opt.solver.oracle.cfg.scale)) { return false; } continue; }
        if (flag == "--oracle-tsp-top") { if (!parse_int_flag(i, flag, opt.solver.oracle.cfg.tsp_top)) { return false; } continue; }
        if (flag == "--oracle-subset-top") { if (!parse_int_flag(i, flag, opt.solver.oracle.cfg.subset_top)) { return false; } continue; }
        if (flag == "--oracle-min-k") { if (!parse_int_flag(i, flag, opt.solver.oracle.cfg.min_k)) { return false; } continue; }
        if (flag == "--oracle-max-k") { if (!parse_int_flag(i, flag, opt.solver.oracle.cfg.max_k)) { return false; } continue; }
        if (flag == "--oracle-lkh-runs") { if (!parse_int_flag(i, flag, opt.solver.oracle.cfg.lkh_runs)) { return false; } continue; }
        if (flag == "--oracle-lkh-trials") { if (!parse_int_flag(i, flag, opt.solver.oracle.cfg.lkh_max_trials)) { return false; } continue; }
        try {
            if (consume_bool(handle_inverse_bool("--oracle-no-tsp", opt.solver.oracle.cfg.use_for_tsp))) { continue; }
            if (consume_bool(handle_inverse_bool("--oracle-no-subset", opt.solver.oracle.cfg.use_for_subset))) { continue; }
            if (consume_bool(handle_bool("--oracle-inline-feedback", opt.solver.oracle.cfg.inline_feedback))) { continue; }
            if (consume_bool(handle_bool("--oracle-verbose", opt.solver.oracle.cfg.verbose))) { continue; }
        } catch (const std::runtime_error&) {
            return false;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
        return false;
    }
    return true;
}

} // namespace aldous_tsp
