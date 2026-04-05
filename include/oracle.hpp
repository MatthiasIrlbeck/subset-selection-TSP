#pragma once

#include "problem.hpp"

struct Tour;

enum class ExternalOracleMode { NONE, AUTO, LKH, CONCORDE };
enum class ResolvedOracleMode { NONE, LKH, CONCORDE };
enum class OracleProblemFormat { EUC2D, MATRIX };

struct ExternalOracleConfig {
    ExternalOracleMode mode = ExternalOracleMode::NONE;
    OracleProblemFormat problem_format = OracleProblemFormat::MATRIX;
    std::string lkh_path = "LKH";
    std::string concorde_path = "concorde";
    int time_limit_sec = 0;
    int scale = 1000000;
    int tsp_top = 3;
    int subset_top = 2;
    int min_k = EXACT_SMALL_TSP_K + 1;
    int max_k = 2500;
    int lkh_runs = 6;
    int lkh_max_trials = 0;
    bool use_for_tsp = true;
    bool use_for_subset = true;
    bool inline_feedback = false;
    bool verbose = false;
};

struct OracleContext {
    ExternalOracleConfig cfg;
    ResolvedOracleMode resolved = ResolvedOracleMode::NONE;
    std::string exec_path;
    std::string status = "disabled";
};

static inline const char* oracle_mode_name(ExternalOracleMode m){
    switch(m){
        case ExternalOracleMode::NONE: return "none";
        case ExternalOracleMode::AUTO: return "auto";
        case ExternalOracleMode::LKH: return "lkh";
        case ExternalOracleMode::CONCORDE: return "concorde";
    }
    return "unknown";
}

static inline const char* resolved_mode_name(ResolvedOracleMode m){
    switch(m){
        case ResolvedOracleMode::NONE: return "none";
        case ResolvedOracleMode::LKH: return "lkh";
        case ResolvedOracleMode::CONCORDE: return "concorde";
    }
    return "unknown";
}

static inline bool try_parse_external_oracle_mode(const std::string& s,ExternalOracleMode& out){
    if(s == "none") { out = ExternalOracleMode::NONE; return true; }
    if(s == "auto") { out = ExternalOracleMode::AUTO; return true; }
    if(s == "lkh") { out = ExternalOracleMode::LKH; return true; }
    if(s == "concorde") { out = ExternalOracleMode::CONCORDE; return true; }
    return false;
}

static inline bool try_parse_oracle_problem_format(const std::string& s,OracleProblemFormat& out){
    if(s == "euc2d") { out = OracleProblemFormat::EUC2D; return true; }
    if(s == "matrix") { out = OracleProblemFormat::MATRIX; return true; }
    return false;
}

static inline const char* oracle_problem_format_name(OracleProblemFormat f){
    switch(f){
        case OracleProblemFormat::EUC2D: return "euc2d";
        case OracleProblemFormat::MATRIX: return "matrix";
    }
    return "unknown";
}

static inline OracleProblemFormat resolved_oracle_problem_format(const OracleContext& oracle){
    return oracle.cfg.problem_format;
}

static inline bool external_oracle_applicable(const OracleContext& oracle,int k,bool full_tsp){
    if(oracle.resolved == ResolvedOracleMode::NONE) return false;
    if(k <= EXACT_SMALL_TSP_K) return false;
    if(k < oracle.cfg.min_k || k > oracle.cfg.max_k) return false;
    if(full_tsp && !oracle.cfg.use_for_tsp) return false;
    if(!full_tsp && !oracle.cfg.use_for_subset) return false;
    return true;
}



struct OracleStats {
    uint64_t calls = 0;
    uint64_t solved = 0;
    uint64_t improved = 0;
    uint64_t failed = 0;
    uint64_t tsp_calls = 0;
    uint64_t tsp_solved = 0;
    uint64_t tsp_improved = 0;
    uint64_t subset_calls = 0;
    uint64_t subset_solved = 0;
    uint64_t subset_improved = 0;
    double exact_gain = 0.0;
    double tsp_gain = 0.0;
    double subset_gain = 0.0;

    void add(const OracleStats& o){
        calls += o.calls; solved += o.solved; improved += o.improved; failed += o.failed;
        tsp_calls += o.tsp_calls; tsp_solved += o.tsp_solved; tsp_improved += o.tsp_improved;
        subset_calls += o.subset_calls; subset_solved += o.subset_solved; subset_improved += o.subset_improved;
        exact_gain += o.exact_gain; tsp_gain += o.tsp_gain; subset_gain += o.subset_gain;
    }
};

bool external_oracle_polish_tour(Tour& cand,const Instance& inst,const OracleContext& oracle,bool full_tsp,int post_strength,OracleStats* stats=nullptr);


bool build_oracle_context(const ExternalOracleConfig& cfg,OracleContext& oracle,std::string& err);
