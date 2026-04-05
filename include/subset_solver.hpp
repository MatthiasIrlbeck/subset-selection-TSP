#pragma once

#include "core.hpp"

bool exact_small_tsp_cycle(const Instance& inst,const std::vector<int>& set_nodes,
                           std::vector<int>& best_cycle,double& best_len);
std::vector<std::vector<int>> make_warm_pool_from_elite(const Instance& inst,const std::vector<std::vector<int>>& prev_elite,int k,Rng& rng,
                                                        std::vector<std::vector<int>>* guide_out);
void solve_subset_highp_delete(const Instance& inst,int k,Rng& rng,Tour& best,
                               int sa_iters,int restarts,const std::vector<std::vector<int>>* parent_elite,
                               std::vector<std::vector<int>>* elite_out);
void solve_subset(const Instance& inst,int k,Rng& rng,Tour& best,const OracleContext& oracle,
                  int sa_iters,int restarts,const std::vector<std::vector<int>>* warm_pool,
                  const std::vector<std::vector<int>>* guide_pool,
                  std::vector<std::vector<int>>* elite_out,
                  bool enable_geom_master);
void maybe_posthoc_oracle_select_best(const Instance& inst,const Tour& internal_best,
                                      const std::vector<std::vector<int>>& elite_nodes,
                                      const OracleContext& oracle,bool full_tsp,int top_keep,
                                      Tour& reported_best,OracleStats* stats);
