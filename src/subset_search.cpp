#include "subset_solver_internal.hpp"

void build_rr_candidate_pool(const Tour& base,const Instance& inst,const std::vector<int>& removed_pos,const std::vector<int>& removed_nodes,
                             Rng& rng,int max_pool,std::vector<int>& pool){
    int N = inst.N, kk = inst.knn_k, nt = base.k;
    pool.clear();
    std::vector<uint8_t> mark(N, 0);
    auto try_add = [&](int v){
        if(v < 0 || v >= N) return;
        if(base.in_set[v]) return;
        if(mark[v]) return;
        mark[v] = 1;
        pool.push_back(v);
    };
    auto push_from_seed = [&](int seed,int lim){
        int taken = 0;
        for(int ki=0; ki<kk && taken<lim && (int)pool.size() < max_pool; ++ki){
            int v = inst.knn_at(seed, ki);
            if(base.in_set[v] || mark[v]) continue;
            mark[v] = 1;
            pool.push_back(v);
            ++taken;
        }
    };

    for(int v : removed_nodes){
        if((int)pool.size() >= max_pool) break;
        push_from_seed(v, 18);
    }
    for(int ri : removed_pos){
        if((int)pool.size() >= max_pool) break;
        push_from_seed(base.nodes[(ri - 1 + nt) % nt], 12);
        if((int)pool.size() >= max_pool) break;
        push_from_seed(base.nodes[(ri + 1) % nt], 12);
    }
    for(int tr=0; tr<24 && (int)pool.size() < max_pool; ++tr){
        int anchor = base.nodes[rng.randint(nt)];
        push_from_seed(anchor, 6);
    }
    for(int tr=0; tr<64 && (int)pool.size() < max_pool; ++tr){
        int v = rng.randint(N);
        try_add(v);
    }
}

bool regret_repair_cycle(std::vector<int>& cyc,const Instance& inst,int target_k,
                         const std::vector<int>& pool,const std::vector<uint8_t>& banned){
    int N = inst.N;
    std::vector<uint8_t> in_now(N, 0), used(pool.size(), 0);
    for(int v : cyc) in_now[v] = 1;

    while((int)cyc.size() < target_k){
        int m = (int)cyc.size();
        std::vector<double> edge_base(m, 0.0);
        for(int j=0; j<m; ++j) edge_base[j] = inst.dist(cyc[j], cyc[(j+1)%m]);

        int best_node = -1, best_pos = 0, best_idx = -1;
        double best_score = std::numeric_limits<double>::infinity();
        double best_cost = std::numeric_limits<double>::infinity();

        auto consider_node = [&](int node,int idx_tag){
            if(node < 0 || node >= N) return;
            if(banned[node] || in_now[node]) return;
            SmallNodeDistCache dc(inst, node);
            double c1 = std::numeric_limits<double>::infinity();
            double c2 = std::numeric_limits<double>::infinity();
            int pos1 = 0;
            for(int j=0; j<m; ++j){
                double c = dc.get(cyc[j]) + dc.get(cyc[(j+1)%m]) - edge_base[j];
                if(c < c1){ c2 = c1; c1 = c; pos1 = j + 1; }
                else if(c < c2) c2 = c;
            }
            if(!std::isfinite(c2)) c2 = c1;
            double score = c1 - REGRET_REPAIR_WEIGHT * (c2 - c1);
            if(score < best_score - KNN_VERIFY_EPS || (std::abs(score - best_score) <= KNN_VERIFY_EPS && c1 < best_cost - KNN_VERIFY_EPS)){
                best_score = score;
                best_cost = c1;
                best_node = node;
                best_pos = pos1;
                best_idx = idx_tag;
            }
        };

        for(int i=0; i<(int)pool.size(); ++i){
            if(used[i]) continue;
            consider_node(pool[i], i);
        }
        if(best_node < 0){
            for(int v=0; v<N; ++v) consider_node(v, -1);
        }
        if(best_node < 0) return false;
        cyc.insert(cyc.begin() + best_pos, best_node);
        in_now[best_node] = 1;
        if(best_idx >= 0) used[best_idx] = 1;
    }
    return true;
}



void polish_subset_candidate(Tour& cand,const Instance& inst,int strength){
    if(cand.k <= EXACT_SMALL_TSP_K){
        exact_small_subset_tour(cand, inst);
        for(int rep=0; rep<std::max(1, strength); ++rep){
            subset_swap_descent(cand, inst, 1 + rep);
        }
        exact_small_subset_tour(cand, inst);
        return;
    }
    cand.ensure_edges(inst);
    for(int rep=0; rep<std::max(1, strength); ++rep){
        two_opt_nn(cand, inst, 720 + 140*rep, -1);
        or_opt_1(cand, inst, 8 + 2*rep, TSP_OROPT_NN_CAND + 6 + 2*rep, TSP_OROPT_LOCAL_WINDOW + 2);
        subset_swap_descent(cand, inst, 2 + rep);
        two_opt_nn(cand, inst, 380 + 100*rep, -1);
        or_opt_1(cand, inst, 5, TSP_OROPT_NN_CAND + 8 + 2*rep, TSP_OROPT_LOCAL_WINDOW + 2);
    }
}

static inline double best_insert_cost_pos(const std::vector<int>& cyc,const Instance& inst,int node,int& best_pos){
    int m = (int)cyc.size();
    if(m <= 0){ best_pos = 0; return 0.0; }
    SmallNodeDistCache dc(inst, node);
    double best = std::numeric_limits<double>::infinity();
    best_pos = 0;
    for(int i=0; i<m; ++i){
        int a = cyc[i], b = cyc[(i + 1) % m];
        double c = dc.get(a) + dc.get(b) - inst.dist(a, b);
        if(c < best){
            best = c;
            best_pos = i + 1;
        }
    }
    return best;
}

static inline double try_two_insert_orders(const std::vector<int>& remain,double base_len,
                                           int u,int v,const Instance& inst,
                                           std::vector<int>& best_cyc){
    double best_len = std::numeric_limits<double>::infinity();
    std::vector<int> tmp;
    auto eval_order = [&](int first,int second){
        int pos1 = 0;
        double c1 = best_insert_cost_pos(remain, inst, first, pos1);
        tmp = remain;
        tmp.insert(tmp.begin() + pos1, first);
        int pos2 = 0;
        double c2 = best_insert_cost_pos(tmp, inst, second, pos2);
        double len = base_len + c1 + c2;
        if(len < best_len){
            best_len = len;
            best_cyc = tmp;
            best_cyc.insert(best_cyc.begin() + pos2, second);
        }
    };
    eval_order(u, v);
    eval_order(v, u);
    return best_len;
}

static void rank_pair_pool(const Tour& tour,const Instance& inst,
                           const std::vector<int>& removed_pos,const std::vector<int>& removed_nodes,
                           std::vector<int>& pool){
    if(pool.empty()) return;
    int k = tour.k;
    int ri = removed_pos[0], rj = removed_pos[1];
    auto prev_live = [&](int idx){
        int p = (idx == 0) ? (k - 1) : (idx - 1);
        while(p == ri || p == rj) p = (p == 0) ? (k - 1) : (p - 1);
        return p;
    };
    auto next_live = [&](int idx){
        int n = (idx + 1 == k) ? 0 : (idx + 1);
        while(n == ri || n == rj) n = (n + 1 == k) ? 0 : (n + 1);
        return n;
    };
    int p1 = prev_live(ri), n1 = next_live(ri);
    int p2 = prev_live(rj), n2 = next_live(rj);
    int prv1 = tour.nodes[p1], nxt1 = tour.nodes[n1];
    int prv2 = tour.nodes[p2], nxt2 = tour.nodes[n2];
    double gap1 = inst.dist(prv1, nxt1);
    double gap2 = inst.dist(prv2, nxt2);

    SmallNodeDistCache dr1(inst, removed_nodes[0]), dr2(inst, removed_nodes[1]);
    SmallNodeDistCache dp1(inst, prv1), dn1(inst, nxt1), dp2(inst, prv2), dn2(inst, nxt2);

    std::vector<std::pair<double,int>> scored;
    scored.reserve(pool.size());
    for(int v : pool){
        double srem = std::min(dr1.get(v), dr2.get(v));
        double sg1 = dp1.get(v) + dn1.get(v) - gap1;
        double sg2 = dp2.get(v) + dn2.get(v) - gap2;
        double score = srem + 0.45 * std::min(sg1, sg2);
        scored.push_back({score, v});
    }
    int keep = std::min((int)scored.size(), TWO2_TOP_POOL);
    auto scored_cmp = [](const auto& a,const auto& b){
        if(a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    };
    if(keep < (int)scored.size()) std::partial_sort(scored.begin(), scored.begin() + keep, scored.end(), scored_cmp);
    else std::sort(scored.begin(), scored.end(), scored_cmp);
    pool.resize(keep);
    for(int i=0; i<keep; ++i) pool[i] = scored[i].second;
}

bool subset_pair_exchange_descent(Tour& tour,const Instance& inst,Rng& rng,int passes){
    if(tour.k < 6) return false;
    tour.ensure_edges(inst);
    bool any = false;
    int max_passes = passes;
    if(tour.k > 220) max_passes = std::min(max_passes, 1);
    for(int pass=0; pass<max_passes; ++pass){
        int k = tour.k;
        std::vector<double> score(k, 0.0);
        std::vector<int> ord(k);
        for(int i=0; i<k; ++i){
            int p = (i == 0) ? (k - 1) : (i - 1);
            int n = (i + 1 == k) ? 0 : (i + 1);
            score[i] = tour.edge_len[p] + tour.edge_len[i] - inst.dist(tour.nodes[p], tour.nodes[n]);
            ord[i] = i;
        }
        int top_rm = std::min(k, TWO2_TOP_REMOVE - (k > 160 ? 2 : 0));
        auto rm_cmp = [&](int a,int b){
            if(score[a] != score[b]) return score[a] > score[b];
            return tour.nodes[a] < tour.nodes[b];
        };
        if(top_rm < k) std::partial_sort(ord.begin(), ord.begin() + top_rm, ord.end(), rm_cmp);
        else std::sort(ord.begin(), ord.end(), rm_cmp);
        double best_len = tour.length;
        std::vector<int> best_nodes;
        std::vector<std::pair<double,std::vector<int>>> approx_best;

        auto push_approx = [&](double len,std::vector<int>& cyc){
            if((int)approx_best.size() < TWO2_POLISH_KEEP){
                approx_best.push_back({len, cyc});
            } else {
                auto it = std::max_element(approx_best.begin(), approx_best.end(), [](const auto& x,const auto& y){ return x.first < y.first; });
                if(len < it->first - IMPROVEMENT_EPS) *it = {len, cyc};
            }
        };

        auto base_len_after_remove = [&](int ri,int rj)->double{
            if(ri > rj) std::swap(ri, rj);
            int p1 = (ri == 0) ? (k - 1) : (ri - 1);
            int n1 = (ri + 1 == k) ? 0 : (ri + 1);
            int p2 = (rj == 0) ? (k - 1) : (rj - 1);
            int n2 = (rj + 1 == k) ? 0 : (rj + 1);
            if(n1 == rj){ // adjacent, ri before rj
                double removed = tour.edge_len[p1] + tour.edge_len[ri] + tour.edge_len[rj];
                double added = inst.dist(tour.nodes[p1], tour.nodes[n2]);
                return tour.length - removed + added;
            }
            if(n2 == ri){ // adjacent wrap-around, rj before ri
                double removed = tour.edge_len[p2] + tour.edge_len[rj] + tour.edge_len[ri];
                double added = inst.dist(tour.nodes[p2], tour.nodes[n1]);
                return tour.length - removed + added;
            }
            double removed = tour.edge_len[p1] + tour.edge_len[ri] + tour.edge_len[p2] + tour.edge_len[rj];
            double added = inst.dist(tour.nodes[p1], tour.nodes[n1]) + inst.dist(tour.nodes[p2], tour.nodes[n2]);
            return tour.length - removed + added;
        };

        for(int a=0; a<top_rm; ++a){
            for(int b=a+1; b<top_rm; ++b){
                int ri = ord[a], rj = ord[b];
                if(ri > rj) std::swap(ri, rj);
                std::vector<int> remain;
                remain.reserve(k - 2);
                for(int i=0; i<k; ++i) if(i != ri && i != rj) remain.push_back(tour.nodes[i]);
                double base_len = base_len_after_remove(ri, rj);
                std::vector<int> removed_pos = {ri, rj};
                std::vector<int> removed_nodes = {tour.nodes[ri], tour.nodes[rj]};
                std::vector<int> pool;
                build_rr_candidate_pool(tour, inst, removed_pos, removed_nodes, rng,
                                        std::min(RR_MAX_POOL, RR_POOL_BASE + 2*RR_POOL_PER_RUIN + 24), pool);
                if((int)pool.size() < 2) continue;
                rank_pair_pool(tour, inst, removed_pos, removed_nodes, pool);
                if((int)pool.size() < 2) continue;

                for(int ui=0; ui<(int)pool.size(); ++ui){
                    for(int vi=ui+1; vi<(int)pool.size(); ++vi){
                        std::vector<int> cyc;
                        double approx_len = try_two_insert_orders(remain, base_len, pool[ui], pool[vi], inst, cyc);
                        if(!(approx_len < best_len - IMPROVEMENT_EPS)) continue;
                        push_approx(approx_len, cyc);
                    }
                }
            }
        }

        for(auto& entry : approx_best){
            Tour cand;
            cand.init(inst.N);
            cand.set_tour_only(entry.second.data(), k);
            polish_fixed_subset_tour(cand, inst, 1);
            if(cand.length < best_len - IMPROVEMENT_EPS){
                best_len = cand.length;
                best_nodes = cand.nodes;
            }
        }
        if(best_nodes.empty()) break;
        tour.set_tour(best_nodes.data(), tour.k, inst);
        polish_subset_candidate(tour, inst, 1);
        any = true;
    }
    return any;
}

void choose_ruin_positions_segment(const Tour& tour,const Instance& inst,Rng& rng,int ruin_size,std::vector<int>& pos_out){
    int k = tour.k;
    pos_out.clear();
    if(ruin_size <= 0 || k <= 0) return;
    std::vector<double> score(k, 0.0);
    std::vector<int> ord(k);
    for(int i=0; i<k; ++i){
        int p = (i == 0) ? (k - 1) : (i - 1);
        int n = (i + 1 == k) ? 0 : (i + 1);
        score[i] = tour.edge_len[p] + tour.edge_len[i] - inst.dist(tour.nodes[p], tour.nodes[n]);
        ord[i] = i;
    }
    int top = std::max(1, std::min(k, 8));
    auto ruin_cmp = [&](int a,int b){
        if(score[a] != score[b]) return score[a] > score[b];
        return tour.nodes[a] < tour.nodes[b];
    };
    if(top < k) std::partial_sort(ord.begin(), ord.begin() + top, ord.end(), ruin_cmp);
    else std::sort(ord.begin(), ord.end(), ruin_cmp);
    int anchor = ord[rng.randint(top)];
    int start = anchor - ruin_size/2;
    while(start < 0) start += k;
    for(int t=0; t<ruin_size; ++t) pos_out.push_back((start + t) % k);
    std::sort(pos_out.begin(), pos_out.end());
}

void choose_ruin_positions_spread(const Tour& tour,const Instance& inst,Rng& rng,int ruin_size,std::vector<int>& pos_out){
    int k = tour.k;
    pos_out.clear();
    if(ruin_size <= 0 || k <= 0) return;
    std::vector<double> score(k, 0.0);
    std::vector<int> ord(k);
    for(int i=0; i<k; ++i){
        int p = (i == 0) ? (k - 1) : (i - 1);
        int n = (i + 1 == k) ? 0 : (i + 1);
        score[i] = tour.edge_len[p] + tour.edge_len[i] - inst.dist(tour.nodes[p], tour.nodes[n]);
        ord[i] = i;
    }
    int top = std::min(k, std::max(ruin_size * 5, 14));
    auto ruin_cmp = [&](int a,int b){
        if(score[a] != score[b]) return score[a] > score[b];
        return tour.nodes[a] < tour.nodes[b];
    };
    if(top < k) std::partial_sort(ord.begin(), ord.begin() + top, ord.end(), ruin_cmp);
    else std::sort(ord.begin(), ord.end(), ruin_cmp);
    std::vector<uint8_t> chosen(k, 0);
    int first = ord[rng.randint(std::max(1, std::min(top, 6)))];
    chosen[first] = 1;
    pos_out.push_back(first);
    while((int)pos_out.size() < ruin_size){
        int best_i = -1;
        double best_s = -std::numeric_limits<double>::infinity();
        for(int t=0; t<top; ++t){
            int i = ord[t];
            if(chosen[i]) continue;
            int mincyc = k;
            for(int j : pos_out){
                int d = std::abs(i - j);
                d = std::min(d, k - d);
                if(d < mincyc) mincyc = d;
            }
            double s = score[i] + 0.25 * mincyc;
            if(s > best_s){ best_s = s; best_i = i; }
        }
        if(best_i < 0) break;
        chosen[best_i] = 1;
        pos_out.push_back(best_i);
    }
    std::sort(pos_out.begin(), pos_out.end());
}

void choose_ruin_positions(const Tour& tour,const Instance& inst,Rng& rng,int ruin_size,std::vector<int>& pos_out){
    int k = tour.k;
    pos_out.clear();
    std::vector<double> score(k, 0.0);
    std::vector<int> ord(k);
    for(int i=0; i<k; ++i){
        int p = (i == 0) ? (k - 1) : (i - 1);
        int n = (i + 1 == k) ? 0 : (i + 1);
        score[i] = tour.edge_len[p] + tour.edge_len[i] - inst.dist(tour.nodes[p], tour.nodes[n]);
        ord[i] = i;
    }
    int top = std::min(k, std::max(ruin_size * 4, 10));
    auto ruin_cmp = [&](int a,int b){
        if(score[a] != score[b]) return score[a] > score[b];
        return tour.nodes[a] < tour.nodes[b];
    };
    if(top < k) std::partial_sort(ord.begin(), ord.begin() + top, ord.end(), ruin_cmp);
    else std::sort(ord.begin(), ord.end(), ruin_cmp);
    std::vector<uint8_t> chosen(k, 0);
    int anchor = ord[rng.randint(std::max(1, std::min(top, 6)))];
    chosen[anchor] = 1;
    pos_out.push_back(anchor);
    int anchor_node = tour.nodes[anchor];
    while((int)pos_out.size() < ruin_size){
        int best_i = -1;
        double best_s = -std::numeric_limits<double>::infinity();
        for(int t=0; t<top; ++t){
            int i = ord[t];
            if(chosen[i]) continue;
            double d = std::sqrt(inst.dist2(anchor_node, tour.nodes[i]));
            double s = score[i] + 1.1 / (1.0 + d);
            if(s > best_s){ best_s = s; best_i = i; }
        }
        if(best_i < 0){
            for(int i=0; i<k; ++i) if(!chosen[i]){ best_i = i; break; }
        }
        if(best_i < 0) break;
        chosen[best_i] = 1;
        pos_out.push_back(best_i);
    }
    std::sort(pos_out.begin(), pos_out.end());
}

bool subset_ruin_recreate_lns(Tour& tour,const Instance& inst,Rng& rng,int rounds){
    if(tour.k < 6) return false;
    tour.ensure_edges(inst);
    bool any = false;
    int k = tour.k;
    int max_ruin = std::min(8, std::max(4, k / 18));
    for(int round=0; round<rounds; ++round){
        int ruin_size = 3 + (round % std::max(1, max_ruin - 2));
        ruin_size = std::min(ruin_size, k - 3);
        if(ruin_size < 2) continue;
        std::vector<int> removed_pos;
        if((round % 3) == 0) choose_ruin_positions(tour, inst, rng, ruin_size, removed_pos);
        else if((round % 3) == 1) choose_ruin_positions_segment(tour, inst, rng, ruin_size, removed_pos);
        else choose_ruin_positions_spread(tour, inst, rng, ruin_size, removed_pos);
        if((int)removed_pos.size() != ruin_size) continue;
        std::vector<int> remain;
        remain.reserve(k);
        std::vector<int> removed_nodes;
        removed_nodes.reserve(ruin_size);
        int rp = 0;
        for(int i=0; i<k; ++i){
            if(rp < ruin_size && removed_pos[rp] == i){
                removed_nodes.push_back(tour.nodes[i]);
                ++rp;
            } else remain.push_back(tour.nodes[i]);
        }
        std::vector<int> pool;
        int max_pool = std::min(RR_MAX_POOL, RR_POOL_BASE + ruin_size * RR_POOL_PER_RUIN + 24);
        build_rr_candidate_pool(tour, inst, removed_pos, removed_nodes, rng, max_pool, pool);
        for(int v : removed_nodes){
            bool seen_v = false;
            for(int u : pool) if(u == v){ seen_v = true; break; }
            if(!seen_v) pool.push_back(v);
        }
        if((int)pool.size() < ruin_size) continue;
        std::vector<uint8_t> remain_set(inst.N, 0);
        for(int v : remain) remain_set[v] = 1;
        if(!regret_repair_cycle(remain, inst, k, pool, remain_set)) continue;
        Tour cand;
        cand.init(inst.N);
        cand.set_tour_only(remain.data(), k);
        polish_subset_candidate(cand, inst, ruin_size >= 5 ? 3 : 2);
        if(cand.length < tour.length - IMPROVEMENT_EPS){
            tour = std::move(cand);
            any = true;
        }
    }
    return any;
}


// SA over swap/add/remove moves to select k points minimizing cycle length.
// Multiple restarts (warm-started from the continuation chain when available),
// elite pool with path-relinking between top solutions, geometric seeding for
// low-p. The best subset tour is returned in `best`.
void solve_subset(const Instance& inst,int k,Rng& rng,Tour& best,const OracleContext& oracle,
    int sa_iters,int restarts,const std::vector<std::vector<int>>* warm_pool,
    const std::vector<std::vector<int>>* guide_pool,
    std::vector<std::vector<int>>* elite_out,
    bool enable_geom_master){
    int N = inst.N, kk = inst.knn_k;
    double bg = std::numeric_limits<double>::infinity();
    std::vector<int> bgn;
    ElitePool local_elite(SUBSET_LOCAL_ELITE_KEEP, EliteHashMode::SET);
    int warm_n = warm_pool ? (int)warm_pool->size() : 0;
    std::vector<std::vector<int>> geom_pool;
    if(enable_geom_master) geom_pool = make_smallp_geometric_master_pool(inst, k, rng);
    std::vector<std::vector<int>> spatial_pool = make_smallp_spatial_seed_pool(inst, k, rng);
    {
        std::vector<uint64_t> seen_sets;
        seen_sets.reserve(warm_n + geom_pool.size() + spatial_pool.size());
        if(warm_pool){
            for(const auto& s : *warm_pool){
                if((int)s.size() == k) seen_sets.push_back(subset_set_hash_canon_copy(s));
            }
        }
        auto filter_pool = [&](std::vector<std::vector<int>>& pool){
            if(pool.empty()) return;
            std::vector<std::vector<int>> filtered;
            filtered.reserve(pool.size());
            for(auto& s : pool){
                uint64_t h = subset_set_hash_canon_copy(s);
                bool dup = false;
                for(uint64_t x : seen_sets) if(x == h){ dup = true; break; }
                if(dup) continue;
                seen_sets.push_back(h);
                filtered.push_back(std::move(s));
            }
            pool.swap(filtered);
        };
        filter_pool(geom_pool);
        filter_pool(spatial_pool);
    }
    int geom_n = (int)geom_pool.size();
    int spatial_n = (int)spatial_pool.size();
    int total_restarts = restarts + warm_n + geom_n + spatial_n;

    for(int restart=0; restart<total_restarts; restart++){
        bool use_warm = warm_pool && restart < warm_n;
        bool use_geom = (!use_warm && restart < warm_n + geom_n);
        bool use_spatial = (!use_warm && !use_geom && restart < warm_n + geom_n + spatial_n);
        std::vector<int> init;
        if(use_warm){
            init = (*warm_pool)[restart];
            if((int)init.size() != k) init = resize_seed_cycle(inst, init, k, rng, restart % 3);
        } else if(use_geom){
            init = geom_pool[restart - warm_n];
            if((int)init.size() != k) init = resize_seed_cycle(inst, init, k, rng, (restart - warm_n) % 3);
        } else if(use_spatial){
            init = spatial_pool[restart - warm_n - geom_n];
            if((int)init.size() != k) init = resize_seed_cycle(inst, init, k, rng, (restart - warm_n - geom_n) % 3);
        } else if((restart - warm_n - geom_n - spatial_n) % 2 == 0){
            int si = rng.randint(N);
            init.push_back(si);
            std::vector<uint8_t> in_s(N, 0);
            in_s[si] = 1;
            for(int i=1;i<k;i++){
                int bv = -1;
                int last = init.back();
                for(int ki=0; ki<kk; ki++){
                    int v = inst.knn_at(last, ki);
                    if(!in_s[v]){ bv = v; break; }
                }
                if(bv < 0) for(int v=0; v<N; v++) if(!in_s[v]){ bv = v; break; }
                init.push_back(bv);
                in_s[bv] = 1;
            }
        } else {
            std::vector<int> perm(N);
            std::iota(perm.begin(), perm.end(), 0);
            rng.partial_shuffle(perm.data(), N, k);
            init.assign(perm.begin(), perm.begin() + k);
        }

        std::vector<int> built;
        if(use_warm || use_geom || use_spatial) built = init;
        else if(k > 20){ int si = rng.randint(k); nn_tour(init.data(), k, inst, si, built); }
        else nn_tour(init.data(), k, inst, 0, built);

        Tour tour;
        tour.init(N);
        tour.set_tour_only(built.data(), k);
        two_opt_nn(tour, inst, 400, -1);
        or_opt_1(tour, inst, 4, TSP_OROPT_NN_CAND + 2, TSP_OROPT_LOCAL_WINDOW + 1);
        double cur = tour.length;
        std::vector<int> bn2 = tour.nodes;
        double bl2 = cur;
        std::vector<int> mark(N, 0);
        std::vector<double> node_dist(N, 0.0);
        std::vector<int> pos_seen(std::max(1, k), 0);
        int mark_token = 1, pos_seen_token = 1;
        int isb = 0;
        double cool = std::exp(std::log(SA_T1 / SA_T0) / std::max(sa_iters - 1, 1));
        double T = SA_T0;

        for(int it=0; it<sa_iters; it++){
            T *= cool;
            int nt = tour.k;
            isb++;
            if(rng.uniform() < SA_SWAP_PROB){
                int ri = rng.randint(nt), rem = tour.nodes[ri];
                int prev_idx = (ri == 0) ? (nt - 1) : (ri - 1);
                int next_idx = (ri + 1 == nt) ? 0 : (ri + 1);
                int prv = tour.nodes[prev_idx], nxt = tour.nodes[next_idx];
                double gap_len = inst.dist(prv, nxt);
                double rs = tour.edge_len[prev_idx] + tour.edge_len[ri] - gap_len;

                int add_nodes[SUBSET_MAX_ADD_CAND];
                double add_dp[SUBSET_MAX_ADD_CAND];
                double add_dn[SUBSET_MAX_ADD_CAND];
                double add_gap[SUBSET_MAX_ADD_CAND];
                int ord[SUBSET_MAX_ADD_CAND];
                int add_cnt = 0;
                int add_tok = bump_token(mark, mark_token);

                auto push_add_seed = [&](int seed,int lim){
                    int taken = 0;
                    for(int ki=0; ki<kk && taken<lim && add_cnt < SUBSET_MAX_ADD_CAND; ++ki){
                        int v = inst.knn_at(seed, ki);
                        if(tour.in_set[v] || mark[v] == add_tok) continue;
                        mark[v] = add_tok;
                        add_nodes[add_cnt++] = v;
                        ++taken;
                    }
                };
                auto push_random_outside = [&](){
                    for(int tr=0; tr<48 && add_cnt < SUBSET_MAX_ADD_CAND; ++tr){
                        int v = rng.randint(N);
                        if(tour.in_set[v] || mark[v] == add_tok) continue;
                        mark[v] = add_tok;
                        add_nodes[add_cnt++] = v;
                        break;
                    }
                };

                push_add_seed(rem, SUBSET_ADD_SCAN_REM);
                push_add_seed(prv, SUBSET_ADD_SCAN_GAP);
                push_add_seed(nxt, SUBSET_ADD_SCAN_GAP);
                int anchor1 = tour.nodes[rng.randint(nt)];
                int anchor2 = tour.nodes[(ri + nt/3) % nt];
                int anchor3 = tour.nodes[(ri + (2*nt)/3) % nt];
                push_add_seed(anchor1, SUBSET_GLOBAL_PROBES);
                push_add_seed(anchor2, SUBSET_GLOBAL_PROBES);
                push_add_seed(anchor3, SUBSET_GLOBAL_PROBES);
                for(int g=0; g<SUBSET_GLOBAL_PROBES + 2 && add_cnt < SUBSET_MAX_ADD_CAND; ++g) push_random_outside();
                if(add_cnt == 0){
                    for(int v=0; v<N; ++v){
                        if(!tour.in_set[v]){ add_nodes[add_cnt++] = v; break; }
                    }
                    if(add_cnt == 0) continue;
                }

                dist_many_from(inst, prv, add_nodes, add_cnt, add_dp);
                dist_many_from(inst, nxt, add_nodes, add_cnt, add_dn);
                for(int i=0; i<add_cnt; ++i){
                    add_gap[i] = add_dp[i] + add_dn[i] - gap_len;
                    ord[i] = i;
                }
                int top_eval = (nt <= 120) ? add_cnt : std::min(SUBSET_TOP_ADD, add_cnt);
                auto add_cmp = [&](int lhs,int rhs){
                    if(add_gap[lhs] != add_gap[rhs]) return add_gap[lhs] < add_gap[rhs];
                    return add_nodes[lhs] < add_nodes[rhs];
                };
                if(top_eval < add_cnt) std::partial_sort(ord, ord + top_eval, ord + add_cnt, add_cmp);
                else std::sort(ord, ord + add_cnt, add_cmp);

                double best_delta = std::numeric_limits<double>::infinity();
                int best_add = -1;
                int best_post_pred = (ri > 0) ? (ri - 1) : (nt - 2);

                for(int oi=0; oi<top_eval; ++oi){
                    int idx = ord[oi];
                    int add = add_nodes[idx];
                    int pred_buf[SUBSET_MAX_EDGE_CAND];
                    int pred_cnt = 0;
                    int pred_tok = bump_token(pos_seen, pos_seen_token);
                    auto push_pred = [&](int pred_idx){
                        if(pred_idx < 0 || pred_idx >= nt || pred_idx == ri) return;
                        if(pos_seen[pred_idx] == pred_tok) return;
                        pos_seen[pred_idx] = pred_tok;
                        int succ_idx = next_live_idx(pred_idx, ri, nt);
                        if(succ_idx == pred_idx || succ_idx == ri) return;
                        if(pred_cnt < SUBSET_MAX_EDGE_CAND) pred_buf[pred_cnt++] = pred_idx;
                    };

                    if(nt <= 120){
                        for(int pred_idx=0; pred_idx<nt; ++pred_idx) push_pred(pred_idx);
                    } else {
                        push_pred(prev_idx);
                        for(int off=-SUBSET_LOCAL_EDGE_WINDOW; off<=SUBSET_LOCAL_EDGE_WINDOW+2; ++off){
                            int p = prev_idx + off;
                            while(p < 0) p += nt;
                            while(p >= nt) p -= nt;
                            push_pred(p);
                        }
                        int near_add = 0;
                        for(int ki=0; ki<kk && near_add<SUBSET_EDGE_ADD_NN; ++ki){
                            int v = inst.knn_at(add, ki);
                            int jp = tour.pos[v];
                            if(jp < 0 || jp == ri) continue;
                            push_pred(jp);
                            push_pred(prev_live_idx(jp, ri, nt));
                            ++near_add;
                        }
                    }

                    int node_tok = bump_token(mark, mark_token);
                    node_dist[prv] = add_dp[idx];
                    node_dist[nxt] = add_dn[idx];
                    mark[prv] = node_tok;
                    mark[nxt] = node_tok;

                    int endpoint_nodes[SUBSET_MAX_ENDPOINTS];
                    double endpoint_d[SUBSET_MAX_ENDPOINTS];
                    int endpoint_cnt = 0;
                    for(int t=0; t<pred_cnt; ++t){
                        int pred_idx = pred_buf[t];
                        int succ_idx = next_live_idx(pred_idx, ri, nt);
                        int a = tour.nodes[pred_idx];
                        int b = tour.nodes[succ_idx];
                        if(mark[a] != node_tok){
                            mark[a] = node_tok;
                            if(endpoint_cnt < SUBSET_MAX_ENDPOINTS) endpoint_nodes[endpoint_cnt++] = a;
                        }
                        if(mark[b] != node_tok){
                            mark[b] = node_tok;
                            if(endpoint_cnt < SUBSET_MAX_ENDPOINTS) endpoint_nodes[endpoint_cnt++] = b;
                        }
                    }
                    if(endpoint_cnt > 0){
                        dist_many_from(inst, add, endpoint_nodes, endpoint_cnt, endpoint_d);
                        for(int t=0; t<endpoint_cnt; ++t) node_dist[endpoint_nodes[t]] = endpoint_d[t];
                    }

                    double best_ins = std::numeric_limits<double>::infinity();
                    int best_pred_local = (ri > 0) ? (ri - 1) : (nt - 2);
                    for(int t=0; t<pred_cnt; ++t){
                        int pred_idx = pred_buf[t];
                        int succ_idx = next_live_idx(pred_idx, ri, nt);
                        int a = tour.nodes[pred_idx];
                        int b = tour.nodes[succ_idx];
                        double base = (pred_idx == prev_idx) ? gap_len : tour.edge_len[pred_idx];
                        double c = node_dist[a] + node_dist[b] - base;
                        if(c < best_ins){
                            best_ins = c;
                            best_pred_local = post_idx_from_cur(pred_idx, ri);
                        }
                    }

                    double delta = best_ins - rs;
                    if(delta < best_delta){
                        best_delta = delta;
                        best_add = add;
                        best_post_pred = best_pred_local;
                    }
                }

                if(best_add < 0) continue;
                if(best_delta < 0 || rng.uniform() < std::exp(-best_delta / std::max(T, TEMP_FLOOR_EPS))){
                    tour.apply_swap_post_rem(ri, best_post_pred, best_add, inst, best_delta);
                    cur = tour.length;
                    if(cur < bl2 - IMPROVEMENT_EPS){
                        bl2 = cur;
                        bn2 = tour.nodes;
                        isb = 0;
                    }
                }
            } else {
                if(nt < 4) continue;
                int i = rng.randint(nt), j = rng.randint(nt);
                int d = std::abs(i - j);
                if(d <= 1 || d == nt - 1) continue;
                if(i > j) std::swap(i, j);
                int a = tour.nodes[i], b = tour.nodes[i+1], c = tour.nodes[j], d_node = tour.nodes[(j+1)%nt];
                double delta = inst.dist(a, c) + inst.dist(b, d_node) - tour.edge_len[i] - tour.edge_len[j];
                if(delta < 0 || rng.uniform() < std::exp(-delta / std::max(T, TEMP_FLOOR_EPS))){
                    tour.apply_two_opt(i, j, inst, delta);
                    cur = tour.length;
                    if(cur < bl2 - IMPROVEMENT_EPS){
                        bl2 = cur;
                        bn2 = tour.nodes;
                        isb = 0;
                    }
                }
            }
            // Guard against floating-point drift in accumulated deltas.
            if((it+1) % SA_RECOMPUTE_INTERVAL == 0){
                tour.recompute_length(inst);
                cur = tour.length;
            }
            if((it+1) % SA_DESCENT_INTERVAL == 0){
                subset_swap_descent(tour, inst, 1);
                two_opt_nn(tour, inst, 350, -1);
                or_opt_1(tour, inst, 5, TSP_OROPT_NN_CAND + 2, TSP_OROPT_LOCAL_WINDOW + 1);
                cur = tour.length;
                if(cur < bl2 - IMPROVEMENT_EPS){
                    bl2 = cur;
                    bn2 = tour.nodes;
                    isb = 0;
                }
            }
            if(isb > SA_STAGNATION_LIMIT) break;
        }

        Tour polish;
        polish.init(N);
        polish.set_tour_only(bn2.data(), k);
        polish.recompute_length(inst);
        for(int rep=0; rep<5; ++rep){
            double before = polish.length;
            two_opt_nn(polish, inst, 1100 - 120*rep, -1);
            or_opt_1(polish, inst, 14 - 2*rep, TSP_OROPT_NN_CAND + 2*rep, TSP_OROPT_LOCAL_WINDOW + 1);
            bool subimp = subset_swap_descent(polish, inst, 2);
            two_opt_nn(polish, inst, 650 - 80*rep, -1);
            or_opt_1(polish, inst, 8, TSP_OROPT_NN_CAND + 6, TSP_OROPT_LOCAL_WINDOW + 1);
            if(!subimp && before - polish.length < IMPROVEMENT_EPS) break;
        }
        if(k >= 6){
            subset_pair_exchange_descent(polish, inst, rng, 2);
            subset_ruin_recreate_lns(polish, inst, rng, RR_ROUNDS_BASE + std::min(3, k / 120));
            subset_pair_exchange_descent(polish, inst, rng, 1);
            polish_subset_candidate(polish, inst, 2);
        }
        if(k >= 8){
            std::vector<int> ntp(k), bestp_nodes = polish.nodes;
            double bestp_len = polish.length;
            int noimp2 = 0;
            int fix_ils = std::min(48, std::max(12, k / 35));
            for(int pit=0; pit<fix_ils; ++pit){
                int c[3];
                c[0] = rng.randint(k);
                c[1] = rng.randint(k);
                c[2] = rng.randint(k);
                std::sort(c, c+3);
                if(c[0] == c[1] || c[1] == c[2]) continue;
                int p = 0;
                for(int i=0; i<=c[0]; ++i) ntp[p++] = bestp_nodes[i];
                for(int i=c[1]+1; i<=c[2]; ++i) ntp[p++] = bestp_nodes[i];
                for(int i=c[0]+1; i<=c[1]; ++i) ntp[p++] = bestp_nodes[i];
                for(int i=c[2]+1; i<k; ++i) ntp[p++] = bestp_nodes[i];
                polish.set_tour_only(ntp.data(), k);
                two_opt_nn(polish, inst, 500, -1);
                or_opt_1(polish, inst, 5, TSP_OROPT_NN_CAND + 3, TSP_OROPT_LOCAL_WINDOW + 1);
                subset_swap_descent(polish, inst, 1);
                two_opt_nn(polish, inst, 260, -1);
                if(polish.length < bestp_len - IMPROVEMENT_EPS){
                    bestp_len = polish.length;
                    bestp_nodes = polish.nodes;
                    noimp2 = 0;
                } else {
                    polish.set_tour(bestp_nodes.data(), k, inst);
                    ++noimp2;
                    if(noimp2 > 10) break;
                }
            }
            polish.set_tour(bestp_nodes.data(), k, inst);
        }

        local_elite.try_add(polish.nodes, polish.length);
        if(polish.length < bg){
            bg = polish.length;
            bgn = polish.nodes;
        }
    }

    auto elite_nodes = local_elite.export_nodes();
    int repolish = std::min((int)elite_nodes.size(), SUBSET_FINAL_ELITE_REPOLISH);
    for(int ei=0; ei<repolish; ++ei){
        Tour t;
        t.init(N);
        t.set_tour_only(elite_nodes[ei].data(), k);
        subset_pair_exchange_descent(t, inst, rng, 2);
        subset_ruin_recreate_lns(t, inst, rng, RR_ROUNDS_BASE + 2);
        subset_pair_exchange_descent(t, inst, rng, 1);
        polish_subset_candidate(t, inst, 2);
        local_elite.try_add(t.nodes, t.length);
        if(t.length < bg){
            bg = t.length;
            bgn = t.nodes;
        }
    }

    auto relink_candidate = [&](const std::vector<int>& a,const std::vector<int>& b,int strength){
        if(a.empty() || b.empty() || a.size() != b.size()) return;
        if(subset_set_hash_canon_copy(a) == subset_set_hash_canon_copy(b)) return;
        std::vector<int> rel_nodes;
        double rel_len = std::numeric_limits<double>::infinity();
        if(!subset_path_relink_bidirectional(inst, a, b, rng, rel_nodes, rel_len)) return;
        Tour t;
        t.init(N);
        t.set_tour_only(rel_nodes.data(), k);
        polish_subset_candidate(t, inst, strength);
        if(k >= 6 && strength >= 2){
            subset_pair_exchange_descent(t, inst, rng, 1);
            polish_subset_candidate(t, inst, 1);
        }
        local_elite.try_add(t.nodes, t.length);
        if(t.length < bg){
            bg = t.length;
            bgn = t.nodes;
        }
    };
    auto recombine_candidate = [&](const std::vector<int>& a,const std::vector<int>& b,int mode,int strength){
        if(a.empty() || b.empty() || a.size() != b.size()) return;
        std::vector<int> child = recombine_parent_cycles(inst, a, b, k, rng, mode);
        if((int)child.size() != k) return;
        Tour t;
        t.init(N);
        t.set_tour_only(child.data(), k);
        polish_subset_candidate(t, inst, strength);
        if(k >= 6 && strength >= 2){
            subset_pair_exchange_descent(t, inst, rng, 2);
            polish_subset_candidate(t, inst, 1);
        }
        local_elite.try_add(t.nodes, t.length);
        if(t.length < bg){
            bg = t.length;
            bgn = t.nodes;
        }
    };
    auto core_recombine_candidate = [&](const std::vector<int>& a,const std::vector<int>& b,int mode,int strength){
        if(a.empty() || b.empty() || a.size() != b.size()) return;
        std::vector<int> child = recombine_core_union_cycles(inst, a, b, k, rng, mode);
        if((int)child.size() != k) return;
        Tour t;
        t.init(N);
        t.set_tour_only(child.data(), k);
        polish_subset_candidate(t, inst, strength);
        if(k >= 6){
            subset_pair_exchange_descent(t, inst, rng, strength >= 2 ? 2 : 1);
            polish_subset_candidate(t, inst, 1);
        }
        local_elite.try_add(t.nodes, t.length);
        if(t.length < bg){
            bg = t.length;
            bgn = t.nodes;
        }
    };

    {
        int stages = 1;
        for(int stage=0; stage<stages; ++stage){
            auto elite_now = local_elite.export_nodes();
            int lt = std::min((int)elite_now.size(), SUBSET_LOCAL_RELINK_TOP);
            for(int i=0; i<lt; ++i){
                for(int j=i+1; j<lt; ++j){
                    recombine_candidate(elite_now[i], elite_now[j], 0, 2 - stage);
                    recombine_candidate(elite_now[i], elite_now[j], 2, 2 - stage);
                    core_recombine_candidate(elite_now[i], elite_now[j], 0, 2);
                    core_recombine_candidate(elite_now[i], elite_now[j], 1, 2);
                    relink_candidate(elite_now[i], elite_now[j], 2 - stage);
                }
            }
            if(guide_pool && !guide_pool->empty()){
                int lg = std::min((int)guide_pool->size(), SUBSET_GUIDE_RELINK_GUIDE);
                int ll = std::min((int)elite_now.size(), SUBSET_GUIDE_RELINK_LOCAL);
                int used = 0;
                for(int i=0; i<ll && used < CROSSP_GUIDE_RELINK_LIMIT; ++i){
                    for(int j=0; j<lg && used < CROSSP_GUIDE_RELINK_LIMIT; ++j){
                        recombine_candidate(elite_now[i], (*guide_pool)[j], used & 1, 2);
                        core_recombine_candidate(elite_now[i], (*guide_pool)[j], used & 1, 2);
                        relink_candidate(elite_now[i], (*guide_pool)[j], 2);
                        ++used;
                    }
                }
            }
        }
    }

    if(oracle.cfg.inline_feedback && oracle.cfg.use_for_subset){
        auto elite_now = local_elite.export_nodes();
        int exn = std::min((int)elite_now.size(), std::max(0, oracle.cfg.subset_top));
        for(int ei=0; ei<exn; ++ei){
            Tour t;
            t.init(N);
            t.set_tour(elite_now[ei].data(), k, inst);
            external_oracle_polish_tour(t, inst, oracle, false, 2);
            local_elite.try_add(t.nodes, t.length);
            if(t.length < bg - IMPROVEMENT_EPS){
                bg = t.length;
                bgn = t.nodes;
            }
        }
    }

    best.init(N);
    best.set_tour(bgn.data(), k, inst);
    if(best.k <= EXACT_SMALL_TSP_K) exact_small_subset_tour(best, inst);
    if(elite_out) *elite_out = local_elite.export_nodes();
}


static void collect_posthoc_candidates(const Tour& internal_best,const std::vector<std::vector<int>>& elite_nodes,int top_keep,
                                      bool full_tsp,std::vector<std::vector<int>>& out){
    out.clear();
    std::vector<uint64_t> hashes;
    auto push = [&](const std::vector<int>& nodes){
        if(nodes.empty()) return;
        uint64_t h = elite_hash_nodes(nodes, full_tsp ? EliteHashMode::CYCLE : EliteHashMode::SET);
        for(size_t i=0;i<hashes.size();++i){
            if(hashes[i] == h && out[i].size() == nodes.size()) return;
        }
        hashes.push_back(h);
        out.push_back(nodes);
    };
    push(internal_best.nodes);
    int lim = std::min((int)elite_nodes.size(), std::max(0, top_keep));
    for(int i=0; i<lim; ++i) push(elite_nodes[i]);
}

void maybe_posthoc_oracle_select_best(const Instance& inst,const Tour& internal_best,
                                             const std::vector<std::vector<int>>& elite_nodes,
                                             const OracleContext& oracle,bool full_tsp,int top_keep,
                                             Tour& reported_best,OracleStats* stats){
    reported_best.init(inst.N);
    reported_best.set_tour(internal_best.nodes.data(), internal_best.k, inst);
    if(reported_best.k <= EXACT_SMALL_TSP_K && !full_tsp) exact_small_subset_tour(reported_best, inst);
    if(!external_oracle_applicable(oracle, internal_best.k, full_tsp) || top_keep <= 0) return;

    std::vector<std::vector<int>> cand_nodes;
    collect_posthoc_candidates(internal_best, elite_nodes, top_keep, full_tsp, cand_nodes);
    double best_len = reported_best.length;
    std::vector<int> best_nodes = reported_best.nodes;
    for(const auto& nodes : cand_nodes){
        Tour t;
        t.init(inst.N);
        t.set_tour(nodes.data(), (int)nodes.size(), inst);
        if(t.k <= EXACT_SMALL_TSP_K && !full_tsp) exact_small_subset_tour(t, inst);
        if(t.length < best_len - IMPROVEMENT_EPS){
            best_len = t.length;
            best_nodes = t.nodes;
        }
        external_oracle_polish_tour(t, inst, oracle, full_tsp, 2, stats);
        if(t.length < best_len - IMPROVEMENT_EPS){
            best_len = t.length;
            best_nodes = t.nodes;
        }
    }
    reported_best.set_tour(best_nodes.data(), (int)best_nodes.size(), inst);
}
