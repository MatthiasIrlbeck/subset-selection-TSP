#include "run_cli_internal.hpp"
#include "subset_solver_internal.hpp"

#include <cerrno>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace run_cli_detail {

static bool self_test_knn(){
    Rng rng;
    rng.seed(1);
    for(int rep=0; rep<3; ++rep){
        Instance ins;
        ins.generate(64 + rep * 7, rng);
        ins.build_knn(10, 0.0);
        Rng q;
        q.seed(100 + rep);
        if(!ins.verify_knn(16, q)) return false;
    }
    return true;
}

static bool self_test_knn_boundary(){
    Rng rng;
    rng.seed(11);
    Instance ins;
    ins.generate(21, rng);
    ins.build_knn(20, 0.0);
    Rng q;
    q.seed(211);
    return ins.verify_knn(21, q);
}

static bool self_test_regime_dispatch(){
    if(select_run_regime(SolverMode::BALANCED, 1.0, true) != RunRegime::FULL) return false;
    if(select_run_regime(SolverMode::BALANCED, 0.03, false) != RunRegime::MID) return false;
    if(select_run_regime(SolverMode::SMALLP_REGION, 0.03, false) != RunRegime::LOW) return false;
    if(select_run_regime(SolverMode::SMALLP_REGION, 0.10, false) != RunRegime::MID) return false;
    if(select_run_regime(SolverMode::HIGHP_DELETE, 0.60, false) != RunRegime::HIGH) return false;
    if(select_run_regime(SolverMode::HYBRID, 0.03, false) != RunRegime::LOW) return false;
    if(select_run_regime(SolverMode::HYBRID, 0.20, false) != RunRegime::MID) return false;
    if(select_run_regime(SolverMode::HYBRID, 0.60, false) != RunRegime::HIGH) return false;
    return true;
}

static bool self_test_exact_small_tsp(){
    Rng rng;
    rng.seed(2);
    Instance ins;
    ins.generate(12, rng);
    ins.build_knn(8, 0.0);
    std::vector<int> set_nodes = {0,1,2,3,4,5,6};
    std::vector<int> cycle;
    double exact_len = 0.0;
    if(!exact_small_tsp_cycle(ins, set_nodes, cycle, exact_len)) return false;

    std::vector<int> perm;
    for(int i=1; i<(int)set_nodes.size(); ++i) perm.push_back(i);
    double brute = std::numeric_limits<double>::infinity();
    do{
        double cur = 0.0;
        int prev = 0;
        for(int idx : perm){
            cur += ins.dist(set_nodes[prev], set_nodes[idx]);
            prev = idx;
        }
        cur += ins.dist(set_nodes[prev], set_nodes[0]);
        if(cur < brute) brute = cur;
    } while(std::next_permutation(perm.begin(), perm.end()));
    return std::fabs(brute - exact_len) <= 1e-9;
}

static bool self_test_two_opt(){
    Rng rng;
    rng.seed(3);
    Instance ins;
    ins.generate(24, rng);
    ins.build_knn(12, 0.0);
    int raw[] = {0,1,2,3,4,5,6,7};
    Tour t;
    t.init(ins.N);
    t.set_tour(raw, 8, ins);
    Tour brute;
    brute.init(ins.N);
    std::vector<int> nodes = t.nodes;
    int ii = 1, jj = 5;
    std::reverse(nodes.begin() + ii + 1, nodes.begin() + jj + 1);
    brute.set_tour(nodes.data(), (int)nodes.size(), ins);

    double delta = ins.dist(t.nodes[ii], t.nodes[jj]) + ins.dist(t.nodes[ii+1], t.nodes[(jj+1)%t.k]) - t.edge_len[ii] - t.edge_len[jj];
    t.apply_two_opt(ii, jj, ins, delta);
    if(!check_tour_invariants(t)) return false;
    if(std::fabs(t.length - brute.length) > 1e-9) return false;
    return t.nodes == brute.nodes;
}

static bool self_test_swap_post_rem(){
    Rng rng;
    rng.seed(4);
    Instance ins;
    ins.generate(30, rng);
    ins.build_knn(12, 0.0);
    int raw[] = {0,1,2,3,4,5,6,7};
    Tour t;
    t.init(ins.N);
    t.set_tour(raw, 8, ins);

    int ri = 3;
    int add = 10;
    int post_pred = 1;

    std::vector<int> brute_nodes = t.nodes;
    brute_nodes.erase(brute_nodes.begin() + ri);
    brute_nodes.insert(brute_nodes.begin() + (post_pred + 1), add);

    Tour brute;
    brute.init(ins.N);
    brute.set_tour(brute_nodes.data(), (int)brute_nodes.size(), ins);
    double delta = brute.length - t.length;
    t.apply_swap_post_rem(ri, post_pred, add, ins, delta);

    if(!check_tour_invariants(t)) return false;
    if(std::fabs(t.length - brute.length) > 1e-9) return false;
    return t.nodes == brute.nodes;
}

static bool read_file_to_string(const std::filesystem::path& path,std::string& out){
    std::ifstream in(path, std::ios::binary);
    if(!in) return false;
    std::ostringstream oss;
    oss << in.rdbuf();
    out = oss.str();
    return in.good() || in.eof();
}

static bool parse_hex_digit(char c,unsigned& out){
    if(c >= '0' && c <= '9'){ out = (unsigned)(c - '0'); return true; }
    if(c >= 'a' && c <= 'f'){ out = 10u + (unsigned)(c - 'a'); return true; }
    if(c >= 'A' && c <= 'F'){ out = 10u + (unsigned)(c - 'A'); return true; }
    return false;
}

static bool parse_json_string_literal(const std::string& s,size_t& pos,std::string& out){
    if(pos >= s.size() || s[pos] != '"') return false;
    ++pos;
    out.clear();
    while(pos < s.size()){
        unsigned char c = (unsigned char)s[pos++];
        if(c == '"') return true;
        if(c < 0x20) return false;
        if(c != '\\'){
            out.push_back((char)c);
            continue;
        }
        if(pos >= s.size()) return false;
        char esc = s[pos++];
        switch(esc){
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if(pos + 4 > s.size()) return false;
                unsigned value = 0;
                for(int i=0; i<4; ++i){
                    unsigned nibble = 0;
                    if(!parse_hex_digit(s[pos++], nibble)) return false;
                    value = (value << 4) | nibble;
                }
                if(value > 0x7f) return false;
                out.push_back((char)value);
                break;
            }
            default:
                return false;
        }
    }
    return false;
}

static bool json_find_string_field(const std::string& doc,const std::string& key,std::string& out){
    std::string needle = '"' + key + '"';
    size_t pos = doc.find(needle);
    if(pos == std::string::npos) return false;
    pos = doc.find(':', pos + needle.size());
    if(pos == std::string::npos) return false;
    ++pos;
    while(pos < doc.size()){
        char c = doc[pos];
        bool space = c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
        if(!space) break;
        ++pos;
    }
    return parse_json_string_literal(doc, pos, out);
}

struct ScopedEnvVar {
    std::string name;
    bool had_old = false;
    std::string old_value;
    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;
    ScopedEnvVar(ScopedEnvVar&&) = delete;
    ScopedEnvVar& operator=(ScopedEnvVar&&) = delete;

    explicit ScopedEnvVar(const char* env_name) : name(env_name ? env_name : ""){
        const char* cur = name.empty() ? nullptr : std::getenv(name.c_str());
        if(cur){
            had_old = true;
            old_value = cur;
        }
    }

    bool set(const std::string& value) const {
        return !name.empty() && setenv(name.c_str(), value.c_str(), 1) == 0;
    }

    ~ScopedEnvVar(){
        if(name.empty()) return;
        if(had_old) setenv(name.c_str(), old_value.c_str(), 1);
        else unsetenv(name.c_str());
    }
};

struct ProcessResult {
    bool exited = false;
    int exit_code = -1;
    bool signaled = false;
    int term_signal = 0;
    std::string output;
};

static bool run_cli_subprocess(const char* prog,const std::vector<std::string>& args,
                               ProcessResult& out,bool capture_output=false){
    out = ProcessResult();
    if(!prog || !*prog) return false;

    int pipefd[2] = {-1, -1};
    if(capture_output && pipe(pipefd) != 0) return false;

    pid_t pid = fork();
    if(pid < 0){
        if(pipefd[0] >= 0) close(pipefd[0]);
        if(pipefd[1] >= 0) close(pipefd[1]);
        return false;
    }
    if(pid == 0){
        if(capture_output){
            close(pipefd[0]);
            dup2(pipefd[1], STDOUT_FILENO);
            dup2(pipefd[1], STDERR_FILENO);
            if(pipefd[1] > STDERR_FILENO) close(pipefd[1]);
        } else {
            int devnull = open("/dev/null", O_WRONLY);
            if(devnull >= 0){
                dup2(devnull, STDOUT_FILENO);
                dup2(devnull, STDERR_FILENO);
                if(devnull > STDERR_FILENO) close(devnull);
            }
        }
        std::vector<char*> child_argv;
        child_argv.reserve(args.size() + 2);
        child_argv.push_back(const_cast<char*>(prog));
        for(const std::string& arg : args) child_argv.push_back(const_cast<char*>(arg.c_str()));
        child_argv.push_back(nullptr);
        execvp(prog, child_argv.data());
        _exit(127);
    }
    if(capture_output) close(pipefd[1]);

    int status = 0;
    if(capture_output){
        std::string buf;
        char chunk[4096];
        while(true){
            ssize_t n = read(pipefd[0], chunk, sizeof(chunk));
            if(n < 0){
                if(errno == EINTR) continue;
                close(pipefd[0]);
                (void)waitpid(pid, &status, 0);
                return false;
            }
            if(n == 0) break;
            buf.append(chunk, chunk + n);
        }
        close(pipefd[0]);
        out.output = std::move(buf);
    }
    if(waitpid(pid, &status, 0) < 0) return false;
    if(WIFEXITED(status)){
        out.exited = true;
        out.exit_code = WEXITSTATUS(status);
        return true;
    }
    if(WIFSIGNALED(status)){
        out.signaled = true;
        out.term_signal = WTERMSIG(status);
        return true;
    }
    return false;
}

static bool self_test_json_output_plumbing(){
    TempWorkDir tmp("aldous_cli");
    if(!tmp.ok) return false;

    std::filesystem::path good_path = std::filesystem::path(tmp.path) / "escaped-results.json";
    std::filesystem::path bad_path = std::filesystem::path(tmp.path) / "missing" / "results.json";

    ResultsMetadata meta;
    meta.schema_version = 4;
    meta.cli_command = "cmd --flag=\"alpha\" 'beta'\nnext\tline";
    meta.mode_name = "balanced";
    meta.git_commit.clear();
    meta.compiler = "Compiler \\ \"quoted\"";

    OracleStats oracle_stats;
    std::vector<double> pv = {0.5, 1.0};
    std::vector<std::vector<double>> af = {{0.81, 0.82}, {0.71, 0.72}};
    std::array<int,4> regime_counts{{1,2,3,4}};
    std::array<double,4> regime_seconds{{1.5,2.5,3.5,4.5}};
    std::string oracle_status = "oracle \\ \"quoted\"\nline\tend";
    std::string backend = "coords\\grid\"backend";

    std::string good_path_str = good_path.string();
    if(!write_json(good_path_str.c_str(), 32, pv, af, 2, 2, 1, 20, 17, 1, 0.25,
                   oracle_stats, oracle_status, backend, meta,
                   regime_counts, regime_seconds)) return false;
    if(!std::filesystem::exists(good_path)) return false;

    std::string doc;
    if(!read_file_to_string(good_path, doc)) return false;
    if(doc.find("\"schema_version\": 4") == std::string::npos) return false;
    if(doc.find("\"git_commit\": null") == std::string::npos) return false;
    if(doc.find("\"hostname\":") != std::string::npos) return false;
    if(doc.find("\n    \"mode\":") != std::string::npos) return false;
    if(doc.find("\n  \"mode\": \"balanced\"") == std::string::npos) return false;

    std::string value;
    if(!json_find_string_field(doc, "cli_command", value) || value != meta.cli_command) return false;
    if(!json_find_string_field(doc, "mode", value) || value != meta.mode_name) return false;
    if(!json_find_string_field(doc, "compiler", value) || value != meta.compiler) return false;
    if(!json_find_string_field(doc, "oracle_status", value) || value != oracle_status) return false;
    if(!json_find_string_field(doc, "distance_backend", value) || value != backend) return false;

    std::string bad_path_str = bad_path.string();
    if(write_json(bad_path_str.c_str(), 32, pv, af, 2, 2, 1, 20, 17, 1, 0.25,
                  oracle_stats, oracle_status, backend, meta,
                  regime_counts, regime_seconds)) return false;
    return true;
}

static bool self_test_oracle_default_disabled(){
    TempWorkDir tmp("aldous_cli");
    if(!tmp.ok) return false;

    std::filesystem::path script = std::filesystem::path(tmp.path) / "dummy_lkh";
    {
        std::ofstream out(script);
        if(!out) return false;
        out << "#!/bin/sh\nexit 0\n";
    }
    std::error_code ec;
    std::filesystem::permissions(script,
                                 std::filesystem::perms::owner_read |
                                 std::filesystem::perms::owner_write |
                                 std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace,
                                 ec);
    if(ec) return false;

    ScopedEnvVar path_env("PATH");
    if(!path_env.set(tmp.path)) return false;

    ExternalOracleConfig cfg;
    OracleContext oracle;
    std::string err;
    if(!build_oracle_context(cfg, oracle, err)) return false;
    if(!err.empty()) return false;
    if(oracle.resolved != ResolvedOracleMode::NONE) return false;
    if(!oracle.exec_path.empty()) return false;
    return oracle.status.find("disabled") != std::string::npos;
}

static bool self_test_oracle_dummy_exec_resolution(){
    TempWorkDir tmp("aldous_cli");
    if(!tmp.ok) return false;

    std::filesystem::path script = std::filesystem::path(tmp.path) / "dummy_lkh";
    {
        std::ofstream out(script);
        if(!out) return false;
        out << "#!/bin/sh\nexit 0\n";
    }
    std::error_code ec;
    std::filesystem::permissions(script,
                                 std::filesystem::perms::owner_read |
                                 std::filesystem::perms::owner_write |
                                 std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace,
                                 ec);
    if(ec) return false;

    ScopedEnvVar path_env("PATH");
    const char* old_path = std::getenv("PATH");
    std::string merged_path = tmp.path;
    if(old_path && *old_path) merged_path += ':' + std::string(old_path);
    if(!path_env.set(merged_path)) return false;

    ExternalOracleConfig cfg;
    cfg.mode = ExternalOracleMode::AUTO;
    cfg.lkh_path = "dummy_lkh";

    OracleContext oracle;
    std::string err;
    if(!build_oracle_context(cfg, oracle, err)) return false;
    if(!err.empty()) return false;
    if(oracle.resolved != ResolvedOracleMode::LKH) return false;
    if(oracle.exec_path != script.string()) return false;
    return oracle.status.find("lkh @ " + script.string()) != std::string::npos;
}

static bool self_test_cli_rejects_bad_input(const char* prog){
    const std::vector<std::vector<std::string>> bad_cases = {
        {"--mode", "hybird"},
        {"--oracle", "autp"},
        {"--oracle-format", "auto"},
        {"--N", "-5"},
        {"--N", "2"},
        {"--instances", "0"},
        {"--restarts", "-1"},
        {"--threads", "0"},
        {"--knn", "500"},
        {"--grid-cell", "0"},
        {"--tsp-restarts", "0"},
        {"--oracle-scale", "0"},
        {"--oracle-max-k", "1"}
    };
    for(const auto& args : bad_cases){
        ProcessResult proc;
        if(!run_cli_subprocess(prog, args, proc)) return false;
        if(!proc.exited || proc.exit_code != 1) return false;
        if(proc.signaled) return false;
    }
    return true;
}

static bool self_test_quick_preset_precedence(const char* prog){
    TempWorkDir tmp("aldous_cli");
    if(!tmp.ok) return false;

    std::filesystem::path out_path = std::filesystem::path(tmp.path) / "quick-precedence.json";
    std::vector<std::string> args = {
        "--quick",
        "--N", "12",
        "--instances", "1",
        "--restarts", "1",
        "--sa-iters", "1",
        "--tsp-restarts", "1",
        "--tsp-ils", "1",
        "--tsp-patience", "1",
        "--threads", "1",
        "--mode", "balanced",
        "--oracle", "none",
        "--output", out_path.string()
    };

    ProcessResult proc;
    if(!run_cli_subprocess(prog, args, proc)) return false;
    if(!proc.exited || proc.exit_code != 0) return false;
    if(!std::filesystem::exists(out_path)) return false;

    std::string doc;
    if(!read_file_to_string(out_path, doc)) return false;
    if(doc.find("\"N\": 12") == std::string::npos) return false;
    if(doc.find("\"target\": 1") == std::string::npos) return false;
    if(doc.find("\"restarts\": 1") == std::string::npos) return false;
    if(doc.find("\"sa_iters\": 1") == std::string::npos) return false;
    return true;
}


static bool self_test_oracle_verbose_passthrough(const char* prog){
    TempWorkDir tmp("aldous_cli");
    if(!tmp.ok) return false;

    std::filesystem::path script = std::filesystem::path(tmp.path) / "dummy_lkh_verbose";
    {
        std::ofstream out(script);
        if(!out) return false;
        out << "#!/bin/sh\n";
        out << "echo ORACLE_VERBOSE_MARKER 1>&2\n";
        out << "cp init.tour out.tour\n";
        out << "exit 0\n";
    }
    std::error_code ec;
    std::filesystem::permissions(script,
                                 std::filesystem::perms::owner_read |
                                 std::filesystem::perms::owner_write |
                                 std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace,
                                 ec);
    if(ec) return false;

    auto base_args = std::vector<std::string>{
        "--N", "20",
        "--instances", "1",
        "--restarts", "1",
        "--sa-iters", "1",
        "--tsp-restarts", "1",
        "--tsp-ils", "1",
        "--tsp-patience", "1",
        "--threads", "1",
        "--mode", "balanced",
        "--oracle", "lkh",
        "--lkh-path", script.string(),
        "--oracle-no-subset",
        "--oracle-tsp-top", "1"
    };

    std::filesystem::path quiet_path = std::filesystem::path(tmp.path) / "oracle-quiet.json";
    std::vector<std::string> quiet_args = base_args;
    quiet_args.push_back("--output");
    quiet_args.push_back(quiet_path.string());

    ProcessResult quiet_proc;
    if(!run_cli_subprocess(prog, quiet_args, quiet_proc, true)) return false;
    if(!quiet_proc.exited || quiet_proc.exit_code != 0) return false;
    if(quiet_proc.output.find("ORACLE_VERBOSE_MARKER") != std::string::npos) return false;

    std::filesystem::path verbose_path = std::filesystem::path(tmp.path) / "oracle-verbose.json";
    std::vector<std::string> verbose_args = base_args;
    verbose_args.push_back("--oracle-verbose");
    verbose_args.push_back("--output");
    verbose_args.push_back(verbose_path.string());

    ProcessResult verbose_proc;
    if(!run_cli_subprocess(prog, verbose_args, verbose_proc, true)) return false;
    if(!verbose_proc.exited || verbose_proc.exit_code != 0) return false;
    return verbose_proc.output.find("ORACLE_VERBOSE_MARKER") != std::string::npos;
}

static bool self_test_cli_output_flag(const char* prog){
    TempWorkDir tmp("aldous_cli");
    if(!tmp.ok) return false;

    auto base_args = std::vector<std::string>{
        "--N", "12",
        "--instances", "1",
        "--restarts", "1",
        "--sa-iters", "1",
        "--tsp-restarts", "1",
        "--tsp-ils", "1",
        "--tsp-patience", "1",
        "--threads", "1",
        "--mode", "balanced",
        "--oracle", "none"
    };

    std::filesystem::path good_path = std::filesystem::path(tmp.path) / "cli-output.json";
    std::vector<std::string> good_args = base_args;
    good_args.push_back("--output");
    good_args.push_back(good_path.string());

    ProcessResult good_proc;
    if(!run_cli_subprocess(prog, good_args, good_proc)) return false;
    if(!good_proc.exited || good_proc.exit_code != 0) return false;
    if(!std::filesystem::exists(good_path)) return false;

    std::string doc;
    if(!read_file_to_string(good_path, doc)) return false;
    if(doc.find("\"summary\"") == std::string::npos) return false;

    std::filesystem::path bad_path = std::filesystem::path(tmp.path) / "missing" / "cli-output.json";
    std::vector<std::string> bad_args = base_args;
    bad_args.push_back("--output");
    bad_args.push_back(bad_path.string());

    ProcessResult bad_proc;
    if(!run_cli_subprocess(prog, bad_args, bad_proc)) return false;
    return bad_proc.exited && bad_proc.exit_code != 0;
}


static bool self_test_output_overwrite_guard(const char* prog){
    TempWorkDir tmp("aldous_cli");
    if(!tmp.ok) return false;

    std::filesystem::path out_path = std::filesystem::path(tmp.path) / "overwrite-guard.json";
    auto base_args = std::vector<std::string>{
        "--N", "12",
        "--instances", "1",
        "--restarts", "1",
        "--sa-iters", "1",
        "--tsp-restarts", "1",
        "--tsp-ils", "1",
        "--tsp-patience", "1",
        "--threads", "1",
        "--mode", "balanced",
        "--oracle", "none",
        "--output", out_path.string()
    };

    ProcessResult first_proc;
    if(!run_cli_subprocess(prog, base_args, first_proc)) return false;
    if(!first_proc.exited || first_proc.exit_code != 0) return false;
    if(!std::filesystem::exists(out_path)) return false;

    ProcessResult second_proc;
    if(!run_cli_subprocess(prog, base_args, second_proc)) return false;
    if(!second_proc.exited || second_proc.exit_code == 0) return false;

    std::vector<std::string> force_args = base_args;
    force_args.push_back("--force");
    ProcessResult force_proc;
    if(!run_cli_subprocess(prog, force_args, force_proc)) return false;
    return force_proc.exited && force_proc.exit_code == 0;
}

static bool self_test_smoke_modes(){
    std::vector<double> pv = {0.05, 0.10, 0.50, 1.00};
    for(SolverMode mode : {SolverMode::BALANCED, SolverMode::SMALLP_REGION, SolverMode::HIGHP_DELETE, SolverMode::HYBRID}){
        RunConfig cfg;
        cfg.N = 24;
        cfg.n_inst = 1;
        cfg.restarts = 1;
        cfg.sa_iters = 5;
        cfg.seed = 7;
        cfg.tsp_restarts = 1;
        cfg.tsp_ils = 6;
        cfg.tsp_patience = 3;
        cfg.knn_k = 8;
        cfg.threads = 1;
        cfg.mode = mode;
        cfg.verbose_p = false;
        InstanceResult res = run_one_instance(0, cfg, pv);
        if(!res.ok) return false;
        if(!finite_all(res.fvals)) return false;
    }
    return true;
}

int run_self_test(const char* self_prog){
    int failed = 0;
    std::printf("Running self-tests...\n");
    auto run_case = [&](const char* name,bool ok){
        std::printf("  %-18s %s\n", name, ok ? "OK" : "FAILED");
        if(!ok) ++failed;
    };

    run_case("knn", self_test_knn());
    run_case("knn-boundary", self_test_knn_boundary());
    run_case("regime-dispatch", self_test_regime_dispatch());
    run_case("json-output", self_test_json_output_plumbing());
    run_case("oracle-default", self_test_oracle_default_disabled());
    run_case("oracle-path", self_test_oracle_dummy_exec_resolution());
    run_case("oracle-verbose", self_test_oracle_verbose_passthrough(self_prog));
    run_case("cli-bad-input", self_test_cli_rejects_bad_input(self_prog));
    run_case("cli-quick-order", self_test_quick_preset_precedence(self_prog));
    run_case("cli-output", self_test_cli_output_flag(self_prog));
    run_case("overwrite-guard", self_test_output_overwrite_guard(self_prog));
    run_case("exact-small-tsp", self_test_exact_small_tsp());
    run_case("two-opt", self_test_two_opt());
    run_case("swap-post-rem", self_test_swap_post_rem());
    run_case("smoke-modes", self_test_smoke_modes());

    if(failed == 0) std::printf("All self-tests passed.\n");
    else std::printf("%d self-test(s) failed.\n", failed);
    return failed == 0 ? 0 : 1;
}

} // namespace run_cli_detail
