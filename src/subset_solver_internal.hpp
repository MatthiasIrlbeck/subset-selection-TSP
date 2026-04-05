#pragma once

#include "subset_solver.hpp"
#include "tsp_solver.hpp"

struct TempWorkDir {
    std::string path;
    bool ok = false;
    TempWorkDir(const TempWorkDir&) = delete;
    TempWorkDir& operator=(const TempWorkDir&) = delete;
    TempWorkDir(TempWorkDir&&) = delete;
    TempWorkDir& operator=(TempWorkDir&&) = delete;

    TempWorkDir(const char* prefix){
        std::string templ = std::string("/tmp/") + prefix + "_XXXXXX";
        std::vector<char> buf(templ.begin(), templ.end());
        buf.push_back('\0');
        char* p = mkdtemp(buf.data());
        if(p){
            path = p;
            ok = true;
        }
    }

    ~TempWorkDir(){
        if(ok){
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    }
};

bool subset_swap_descent(Tour& tour,const Instance& inst,int passes);
std::vector<int> resize_seed_cycle(const Instance& inst,const std::vector<int>& seed,int k,Rng& rng,int mode);
std::vector<int> order_cycle_from_set(const Instance& inst,const std::vector<int>& set_nodes,Rng& rng);
std::vector<int> recombine_parent_cycles(const Instance& inst,const std::vector<int>& a,const std::vector<int>& b,int k,Rng& rng,int mode);
std::vector<int> recombine_core_union_cycles(const Instance& inst,const std::vector<int>& a,const std::vector<int>& b,int k,Rng& rng,int mode);
bool exact_small_subset_tour(Tour& cand,const Instance& inst);
void polish_fixed_subset_tour(Tour& cand,const Instance& inst,int strength);
void collect_pr_pred_candidates(const Tour& tour,const Instance& inst,int ri,int add,
                                std::vector<int>& pred_buf,bool force_full);
bool subset_path_relink_bidirectional(const Instance& inst,const std::vector<int>& a,
                                      const std::vector<int>& b,Rng& rng,
                                      std::vector<int>& best_nodes_out,double& best_len_out);
std::vector<std::vector<int>> make_smallp_spatial_seed_pool(const Instance& inst,int k,Rng& rng);
std::vector<std::vector<int>> make_smallp_geometric_master_pool(const Instance& inst,int k,Rng& rng);
bool subset_pair_exchange_descent(Tour& tour,const Instance& inst,Rng& rng,int passes);
bool subset_ruin_recreate_lns(Tour& tour,const Instance& inst,Rng& rng,int rounds);
bool regret_repair_cycle(std::vector<int>& cyc,const Instance& inst,int target_k,
                         const std::vector<int>& pool,const std::vector<uint8_t>& banned);
void polish_subset_candidate(Tour& cand,const Instance& inst,int strength);
