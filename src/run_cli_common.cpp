#include "run_cli_internal.hpp"

namespace run_cli_detail {

static inline uint64_t splitmix64_step(uint64_t z){
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

uint64_t make_stream_seed(uint64_t base_seed,uint64_t instance_idx,uint64_t tag){
    uint64_t z = base_seed ^ (0x9e3779b97f4a7c15ULL * (instance_idx + 1)) ^ tag;
    return splitmix64_step(z);
}


const char* solver_mode_name(SolverMode mode){
    switch(mode){
        case SolverMode::BALANCED: return "balanced";
        case SolverMode::SMALLP_REGION: return "smallp-region";
        case SolverMode::HIGHP_DELETE: return "highp-delete";
        case SolverMode::HYBRID: return "hybrid";
    }
    return "balanced";
}

bool try_parse_solver_mode(const std::string& s,SolverMode& out){
    if(s == "balanced") { out = SolverMode::BALANCED; return true; }
    if(s == "smallp-region") { out = SolverMode::SMALLP_REGION; return true; }
    if(s == "highp-delete") { out = SolverMode::HIGHP_DELETE; return true; }
    if(s == "hybrid") { out = SolverMode::HYBRID; return true; }
    return false;
}

const char* solver_mode_choices(){
    return "balanced, smallp-region, highp-delete, hybrid";
}

bool solver_mode_is_experimental(SolverMode mode){
    return mode != SolverMode::BALANCED;
}

const char* solver_mode_stability_label(SolverMode mode){
    return solver_mode_is_experimental(mode) ? "experimental" : "default";
}

const char* external_oracle_mode_choices(){
    return "none, auto, lkh, concorde";
}

const char* oracle_problem_format_choices(){
    return "matrix, euc2d";
}

static inline bool mode_uses_smallp_region(SolverMode mode){
    return mode == SolverMode::SMALLP_REGION || mode == SolverMode::HYBRID;
}

static inline bool mode_uses_highp_delete(SolverMode mode){
    return mode == SolverMode::HIGHP_DELETE || mode == SolverMode::HYBRID;
}

const char* distance_backend_name(){
    return "coords_exact_grid_knn_edgecache";
}

const char* run_regime_name(RunRegime regime){
    switch(regime){
        case RunRegime::FULL: return "full";
        case RunRegime::LOW: return "low";
        case RunRegime::MID: return "mid";
        case RunRegime::HIGH: return "high";
    }
    return "mid";
}

int run_regime_index(RunRegime regime){
    switch(regime){
        case RunRegime::FULL: return 0;
        case RunRegime::LOW: return 1;
        case RunRegime::MID: return 2;
        case RunRegime::HIGH: return 3;
    }
    return 2;
}

const char* run_regime_label_from_index(int idx){
    switch(idx){
        case 0: return "full";
        case 1: return "low";
        case 2: return "mid";
        case 3: return "high";
    }
    return "mid";
}

RunRegime select_run_regime(SolverMode mode,double p,bool full_tsp){
    if(full_tsp) return RunRegime::FULL;
    if(mode_uses_highp_delete(mode) && p >= REGIME_HIGH_MIN_P) return RunRegime::HIGH;
    if(mode_uses_smallp_region(mode) && p <= REGIME_LOW_MAX_P) return RunRegime::LOW;
    return RunRegime::MID;
}

static inline uint64_t p_key(double p){
    return (uint64_t)std::llround(p * 1000000.0);
}

static inline uint64_t regime_seed_tag(RunRegime regime){
    switch(regime){
        case RunRegime::FULL: return 0xa4093822299f31d0ULL;
        case RunRegime::LOW:  return 0x082efa98ec4e6c89ULL;
        case RunRegime::MID:  return 0x452821e638d01377ULL;
        case RunRegime::HIGH: return 0xbe5466cf34e90c6cULL;
    }
    return 0x452821e638d01377ULL;
}

uint64_t make_regime_p_seed(uint64_t base_seed,uint64_t instance_idx,RunRegime regime,double p){
    uint64_t tag = regime_seed_tag(regime) ^ (p_key(p) * 0x9e3779b97f4a7c15ULL);
    return make_stream_seed(base_seed, instance_idx, tag);
}

std::string banner_title_for_mode(SolverMode mode){
    switch(mode){
        case SolverMode::BALANCED: return "Balanced mode";
        case SolverMode::SMALLP_REGION: return "Small-p region mode (experimental)";
        case SolverMode::HIGHP_DELETE: return "High-p delete mode (experimental)";
        case SolverMode::HYBRID: return "Hybrid mode (experimental)";
    }
    return "Balanced mode";
}

std::string banner_desc_for_mode(SolverMode mode){
    switch(mode){
        case SolverMode::BALANCED:
            return "baseline continuation";
        case SolverMode::SMALLP_REGION:
            return "baseline continuation + low-p region search";
        case SolverMode::HIGHP_DELETE:
            return "baseline continuation + high-p deletion search";
        case SolverMode::HYBRID:
            return "combined regime-specific search";
    }
    return "baseline continuation";
}

void update_chain_state(ChainState& chain,const Tour& result,const std::vector<std::vector<int>>& elite_here,int k){
    chain.prev_elite = elite_here;
    if(chain.prev_elite.empty()) chain.prev_elite.push_back(result.nodes);
    chain.prev_k = k;
}

bool finite_all(const std::vector<double>& v){
    for(double x : v) if(!std::isfinite(x)) return false;
    return true;
}

bool check_tour_invariants(const Tour& tour){
    if(tour.k != (int)tour.nodes.size()) return false;
    if((int)tour.pos.size() != tour.N) return false;
    if((int)tour.in_set.size() != tour.N) return false;
    std::vector<int> seen(tour.N, 0);
    for(int i=0; i<tour.k; ++i){
        int v = tour.nodes[i];
        if(v < 0 || v >= tour.N) return false;
        if(++seen[v] != 1) return false;
        if(tour.pos[v] != i) return false;
        if(!tour.in_set[v]) return false;
    }
    for(int v=0; v<tour.N; ++v){
        if(seen[v]){
            if(tour.pos[v] < 0 || !tour.in_set[v]) return false;
        } else {
            if(tour.pos[v] != -1 || tour.in_set[v]) return false;
        }
    }
    if(tour.edge_valid && tour.k != (int)tour.edge_len.size()) return false;
    return true;
}

} // namespace run_cli_detail
