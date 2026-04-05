#pragma once

#include "core.hpp"

namespace run_cli_detail {

enum class SolverMode { BALANCED, SMALLP_REGION, HIGHP_DELETE, HYBRID };
enum class RunRegime { FULL, LOW, MID, HIGH };

struct RunConfig {
    int N = 500;
    int n_inst = 15;
    int restarts = 3;
    int sa_iters = 60000;
    int seed = 2024;
    int tsp_restarts = 5;
    int tsp_ils = 800;
    int tsp_patience = 80;
    int knn_k = 25;
    int verify_knn_checks = 0;
    int threads = 1;
    double grid_cell = 0.0;
    bool verbose_p = false;
    SolverMode mode = SolverMode::BALANCED;
    OracleContext oracle;
};

struct ResultsMetadata {
    int schema_version = 4;
    std::string cli_command;
    std::string mode_name;
    std::string git_commit;
    std::string compiler;
};

struct InstanceResult {
    bool ok = true;
    int index = -1;
    double wall_seconds = 0.0;
    std::vector<double> fvals;
    OracleStats oracle;
    std::array<int, 4> regime_counts{{0,0,0,0}};
    std::array<double, 4> regime_seconds{{0.0,0.0,0.0,0.0}};
    std::string log;
};

struct ChainState {
    std::vector<std::vector<int>> prev_elite;
    int prev_k = -1;
};

const char* solver_mode_name(SolverMode mode);
bool try_parse_solver_mode(const std::string& s,SolverMode& out);
const char* solver_mode_choices();
bool solver_mode_is_experimental(SolverMode mode);
const char* solver_mode_stability_label(SolverMode mode);
const char* external_oracle_mode_choices();
const char* oracle_problem_format_choices();
const char* distance_backend_name();
const char* run_regime_name(RunRegime regime);
int run_regime_index(RunRegime regime);
const char* run_regime_label_from_index(int idx);
RunRegime select_run_regime(SolverMode mode,double p,bool full_tsp);
uint64_t make_stream_seed(uint64_t base_seed,uint64_t instance_idx,uint64_t tag);
uint64_t make_regime_p_seed(uint64_t base_seed,uint64_t instance_idx,RunRegime regime,double p);
std::string banner_title_for_mode(SolverMode mode);
std::string banner_desc_for_mode(SolverMode mode);

void update_chain_state(ChainState& chain,const Tour& result,const std::vector<std::vector<int>>& elite_here,int k);
bool finite_all(const std::vector<double>& v);
bool check_tour_invariants(const Tour& tour);

ResultsMetadata build_results_metadata(int argc,char** argv,SolverMode mode);
bool write_json(const char* fn,int N,const std::vector<double>& pv,const std::vector<std::vector<double>>& af,
                int done,int tgt,int rst,int sa,int seed,int threads,double ws,
                const OracleStats& oracle_stats,const std::string& oracle_status,
                const std::string& backend,const ResultsMetadata& meta,
                const std::array<int,4>& regime_counts,const std::array<double,4>& regime_seconds);

InstanceResult run_one_instance(int ii,const RunConfig& cfg,const std::vector<double>& pv);
int run_cli_execute(const RunConfig& cfg,
                    const std::vector<double>& pv,
                    const std::string& output_path,
                    bool force_output,
                    const ResultsMetadata& results_meta);
int run_self_test(const char* self_prog);
int tsp_main_cli_impl(int argc,char** argv);

} // namespace run_cli_detail
