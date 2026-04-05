#pragma once

#include "core.hpp"

void two_opt_nn(Tour& tour,const Instance& inst,int mp,int cand_cap);
void two_opt_nn_tsp(Tour& tour,const Instance& inst,int mp,int cand_cap);
void or_opt_1(Tour& tour,const Instance& inst,int mp,int nn_cap,int local_window);
void nn_tour(const int* sub,int k,const Instance& inst,int si,std::vector<int>& res);
void farthest_ins(const int* sub,int k,const Instance& inst,std::vector<int>& res);
void solve_tsp(const Instance& inst,Rng& rng,Tour& best,const OracleContext& oracle,
               int restarts,int ils,int patience,
               std::vector<std::vector<int>>* elite_out);
