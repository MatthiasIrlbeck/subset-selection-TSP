#include "subset_solver_internal.hpp"


bool subset_swap_descent(Tour& tour,const Instance& inst,int passes){
    int N = inst.N, kk = inst.knn_k;
    if(tour.k < 3) return false;
    bool any = false;
    std::vector<int> mark(N, 0);
    std::vector<double> node_dist(N, 0.0);
    std::vector<int> pos_seen(std::max(1, tour.k), 0);
    int mark_token = 1, pos_seen_token = 1;

    for(int pass=0; pass<passes; ++pass){
        int nt = tour.k;
        double global_best = -IMPROVEMENT_EPS;
        int best_ri = -1;
        int best_add = -1;
        int best_post_pred = 0;

        for(int ri=0; ri<nt; ++ri){
            int rem = tour.nodes[ri];
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

            push_add_seed(rem, SUBSET_ADD_SCAN_REM);
            push_add_seed(prv, SUBSET_ADD_SCAN_GAP);
            push_add_seed(nxt, SUBSET_ADD_SCAN_GAP);
            if(nt > 8){
                push_add_seed(tour.nodes[(ri + nt/3) % nt], SUBSET_GLOBAL_PROBES);
                push_add_seed(tour.nodes[(ri + (2*nt)/3) % nt], SUBSET_GLOBAL_PROBES);
            }
            if(add_cnt == 0) continue;

            dist_many_from(inst, prv, add_nodes, add_cnt, add_dp);
            dist_many_from(inst, nxt, add_nodes, add_cnt, add_dn);
            for(int i=0; i<add_cnt; ++i){
                add_gap[i] = add_dp[i] + add_dn[i] - gap_len;
                ord[i] = i;
            }
            int top_eval = (nt <= 80) ? add_cnt : std::min(SUBSET_TOP_ADD, add_cnt);
            auto add_cmp = [&](int lhs,int rhs){
                if(add_gap[lhs] != add_gap[rhs]) return add_gap[lhs] < add_gap[rhs];
                return add_nodes[lhs] < add_nodes[rhs];
            };
            if(top_eval < add_cnt) std::partial_sort(ord, ord + top_eval, ord + add_cnt, add_cmp);
            else std::sort(ord, ord + add_cnt, add_cmp);

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

                if(nt <= 80){
                    for(int pred_idx=0; pred_idx<nt; ++pred_idx) push_pred(pred_idx);
                } else {
                    push_pred(prev_idx);
                    for(int off=-SUBSET_LOCAL_EDGE_WINDOW; off<=SUBSET_LOCAL_EDGE_WINDOW+1; ++off){
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
                if(delta < global_best){
                    global_best = delta;
                    best_ri = ri;
                    best_add = add;
                    best_post_pred = best_pred_local;
                }
            }
        }

        if(best_ri < 0) break;
        tour.apply_swap_post_rem(best_ri, best_post_pred, best_add, inst, global_best);
        any = true;
        two_opt_nn(tour, inst, 250, -1);
        or_opt_1(tour, inst, 4, TSP_OROPT_NN_CAND, TSP_OROPT_LOCAL_WINDOW);
        two_opt_nn(tour, inst, 120, -1);
    }
    return any;
}

static inline std::vector<double> subset_removal_scores(const std::vector<int>& cyc,const Instance& inst){
    int m = (int)cyc.size();
    std::vector<double> score(m, 0.0);
    if(m <= 2) return score;
    for(int i=0; i<m; ++i){
        int p = cyc[(i - 1 + m) % m];
        int u = cyc[i];
        int n = cyc[(i + 1) % m];
        score[i] = inst.dist(p, u) + inst.dist(u, n) - inst.dist(p, n);
    }
    return score;
}

std::vector<int> resize_seed_cycle(const Instance& inst,const std::vector<int>& seed,int k,Rng& rng,int mode){
    int N = inst.N, kk = inst.knn_k;
    std::vector<int> cur = seed;
    if(cur.empty()) return cur;

    while((int)cur.size() > k){
        int m = (int)cur.size();
        auto score = subset_removal_scores(cur, inst);
        std::vector<int> ord(m);
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(), [&](int a,int b){
            if(score[a] != score[b]) return score[a] > score[b];
            return cur[a] < cur[b];
        });
        int wi = ord[0];
        if(mode == 1){
            int top = std::min(m, 6);
            double tot = 0.0;
            for(int t=0; t<top; ++t) tot += (double)(top - t);
            double r = rng.uniform() * tot;
            for(int t=0; t<top; ++t){
                r -= (double)(top - t);
                if(r <= 0.0){ wi = ord[t]; break; }
            }
        } else if(mode == 2){
            int top = std::min(m, 10);
            int anchor = ord[rng.randint(std::max(1, std::min(top, 4)))];
            int anchor_node = cur[anchor];
            double best_s = -std::numeric_limits<double>::infinity();
            for(int t=0; t<top; ++t){
                int i = ord[t];
                double d = std::sqrt(inst.dist2(anchor_node, cur[i]));
                double s = score[i] + 0.9 / (1.0 + d);
                if(s > best_s){ best_s = s; wi = i; }
            }
        }
        cur.erase(cur.begin() + wi);
    }

    while((int)cur.size() < k){
        int m = (int)cur.size();
        if(m == 0){
            cur.push_back(rng.randint(N));
            continue;
        }
        std::vector<uint8_t> in_s(N, 0);
        for(int v : cur) in_s[v] = 1;
        int best_node = -1, best_pos = 0;
        double best_cost = std::numeric_limits<double>::infinity();
        for(int j=0; j<m; ++j){
            int a = cur[j], b = cur[(j+1)%m];
            double base = inst.dist(a, b);
            SmallNodeDistCache da(inst, a), db(inst, b);
            int lim = std::min(kk, 16);
            for(int ki=0; ki<lim; ++ki){
                int va = inst.knn_at(a, ki);
                if(!in_s[va]){
                    da.seed(va, inst.knn_d_at(a, ki));
                    double c = da.get(va) + db.get(va) - base;
                    if(c < best_cost){ best_cost = c; best_node = va; best_pos = j + 1; }
                }
                int vb = inst.knn_at(b, ki);
                if(!in_s[vb]){
                    db.seed(vb, inst.knn_d_at(b, ki));
                    double c = da.get(vb) + db.get(vb) - base;
                    if(c < best_cost){ best_cost = c; best_node = vb; best_pos = j + 1; }
                }
            }
        }
        if(best_node < 0){
            for(int v=0; v<N; ++v){ if(!in_s[v]){ best_node = v; break; } }
            if(best_node < 0) break;
            best_pos = 0;
            best_cost = std::numeric_limits<double>::infinity();
            SmallNodeDistCache dc(inst, best_node);
            for(int j=0; j<m; ++j){
                double c = dc.get(cur[j]) + dc.get(cur[(j+1)%m]) - inst.dist(cur[j], cur[(j+1)%m]);
                if(c < best_cost){ best_cost = c; best_pos = j + 1; }
            }
        }
        cur.insert(cur.begin() + best_pos, best_node);
    }
    return cur;
}


std::vector<int> order_cycle_from_set(const Instance& inst,const std::vector<int>& set_nodes,Rng& rng){
    if((int)set_nodes.size() <= 2) return set_nodes;
    std::vector<int> built;
    int si = rng.randint((int)set_nodes.size());
    nn_tour(set_nodes.data(), (int)set_nodes.size(), inst, si, built);
    return built;
}

std::vector<int> recombine_parent_cycles(const Instance& inst,const std::vector<int>& a,const std::vector<int>& b,int k,Rng& rng,int mode){
    int N = inst.N;
    if(a.empty()) return resize_seed_cycle(inst, b, k, rng, mode % 3);
    if(b.empty()) return resize_seed_cycle(inst, a, k, rng, mode % 3);

    std::vector<uint8_t> in_b(N, 0), used(N, 0);
    for(int v : b) if(v >= 0 && v < N) in_b[v] = 1;

    std::vector<int> merged;
    merged.reserve(std::min(N, (int)a.size() + (int)b.size()));
    for(int v : a){
        if(v >= 0 && v < N && in_b[v] && !used[v]){
            used[v] = 1;
            merged.push_back(v);
        }
    }

    auto push_unique = [&](int v){
        if(v < 0 || v >= N) return;
        if(used[v]) return;
        used[v] = 1;
        merged.push_back(v);
    };

    if(mode == 1){
        for(int v : b) push_unique(v);
        for(int v : a) push_unique(v);
    } else if(mode == 2){
        size_t ia = 0, ib = 0;
        while(ia < a.size() || ib < b.size()){
            if(ia < a.size()) push_unique(a[ia++]);
            if(ib < b.size()) push_unique(b[ib++]);
        }
    } else {
        for(int v : a) push_unique(v);
        for(int v : b) push_unique(v);
    }

    if(merged.empty()) merged = a;
    if((int)merged.size() > 3) merged = order_cycle_from_set(inst, merged, rng);
    if((int)merged.size() != k) merged = resize_seed_cycle(inst, merged, k, rng, mode % 3);
    if((int)merged.size() > 3) merged = order_cycle_from_set(inst, merged, rng);
    return merged;
}

bool exact_small_tsp_cycle(const Instance& inst,const std::vector<int>& set_nodes,
                                  std::vector<int>& best_cycle,double& best_len){
    int k = (int)set_nodes.size();
    if(k <= 0){
        best_cycle.clear();
        best_len = 0.0;
        return true;
    }
    if(k == 1){
        best_cycle = set_nodes;
        best_len = 0.0;
        return true;
    }
    if(k == 2){
        best_cycle = set_nodes;
        best_len = 2.0 * inst.dist(set_nodes[0], set_nodes[1]);
        return true;
    }
    if(k > EXACT_SMALL_TSP_K) return false;

    int m = k - 1;
    if(m >= 31) return false;
    uint32_t total = 1u << m;
    const double INF = std::numeric_limits<double>::infinity();

    std::vector<double> dm((size_t)k * k, 0.0);
    for(int i=0; i<k; ++i){
        for(int j=i+1; j<k; ++j){
            double d = inst.dist(set_nodes[i], set_nodes[j]);
            dm[(size_t)i * k + j] = d;
            dm[(size_t)j * k + i] = d;
        }
    }

    std::vector<double> dp((size_t)total * m, INF);
    std::vector<int16_t> parent((size_t)total * m, -1);

    for(int j=0; j<m; ++j) dp[((size_t)1u << j) * m + j] = dm[j + 1];

    for(uint32_t mask=1; mask<total; ++mask){
        uint32_t bits = mask;
        while(bits){
            int j = __builtin_ctz(bits);
            bits &= bits - 1;
            uint32_t pm = mask ^ (1u << j);
            if(pm == 0) continue;
            double best = INF;
            int bestp = -1;
            uint32_t prev_bits = pm;
            while(prev_bits){
                int i = __builtin_ctz(prev_bits);
                prev_bits &= prev_bits - 1;
                double cand = dp[(size_t)pm * m + i] + dm[(size_t)(i + 1) * k + (j + 1)];
                if(cand < best){
                    best = cand;
                    bestp = i;
                }
            }
            dp[(size_t)mask * m + j] = best;
            parent[(size_t)mask * m + j] = (int16_t)bestp;
        }
    }

    uint32_t full = total - 1u;
    best_len = INF;
    int endj = -1;
    for(int j=0; j<m; ++j){
        double cand = dp[(size_t)full * m + j] + dm[(size_t)(j + 1) * k];
        if(cand < best_len){
            best_len = cand;
            endj = j;
        }
    }
    if(endj < 0 || !std::isfinite(best_len)) return false;

    std::vector<int> ord_idx(k, 0);
    uint32_t mask = full;
    int cur = endj;
    for(int pos = k - 1; pos >= 1; --pos){
        ord_idx[pos] = cur + 1;
        int prev = parent[(size_t)mask * m + cur];
        mask ^= (1u << cur);
        cur = prev;
        if(mask == 0) break;
    }

    best_cycle.resize(k);
    for(int i=0; i<k; ++i) best_cycle[i] = set_nodes[ord_idx[i]];
    return true;
}

bool exact_small_subset_tour(Tour& cand,const Instance& inst){
    if(cand.k > EXACT_SMALL_TSP_K) return false;
    std::vector<int> best_cycle;
    double best_len = 0.0;
    if(!exact_small_tsp_cycle(inst, cand.nodes, best_cycle, best_len)) return false;
    cand.set_tour_only(best_cycle.data(), cand.k);
    cand.recompute_length(inst);
    return true;
}


std::vector<int> recombine_core_union_cycles(const Instance& inst,const std::vector<int>& a,const std::vector<int>& b,int k,Rng& rng,int mode){
    int N = inst.N;
    if(a.empty()) return resize_seed_cycle(inst, b, k, rng, mode % 3);
    if(b.empty()) return resize_seed_cycle(inst, a, k, rng, mode % 3);

    std::vector<uint8_t> in_a(N, 0), in_b(N, 0), used(N, 0), banned(N, 0);
    for(int v : a) if(v >= 0 && v < N) in_a[v] = 1;
    for(int v : b) if(v >= 0 && v < N) in_b[v] = 1;

    std::vector<int> core_a, core_b;
    for(int v : a) if(v >= 0 && v < N && in_b[v]) core_a.push_back(v);
    for(int v : b) if(v >= 0 && v < N && in_a[v]) core_b.push_back(v);
    std::vector<int> cyc = ((mode & 1) == 0) ? core_a : core_b;
    if((int)cyc.size() < 2) return recombine_parent_cycles(inst, a, b, k, rng, mode % 3);

    for(int v : cyc){
        used[v] = 1;
        banned[v] = 0;
    }

    std::vector<int> pool;
    pool.reserve(std::min(N, CORE_RECOMB_UNION_KEEP + (int)cyc.size() * CORE_RECOMB_KNN));
    auto push_pool = [&](int v){
        if(v < 0 || v >= N) return;
        if(used[v]) return;
        used[v] = 1;
        pool.push_back(v);
    };

    for(int v : a) if(!in_b[v]) push_pool(v);
    for(int v : b) if(!in_a[v]) push_pool(v);

    int lim = std::min(inst.knn_k, CORE_RECOMB_KNN);
    for(int seed : cyc){
        for(int ki=0; ki<lim; ++ki) push_pool(inst.knn_at(seed, ki));
        if((int)pool.size() >= CORE_RECOMB_UNION_KEEP) break;
    }
    for(int v : a){
        if(in_b[v]) continue;
        for(int ki=0; ki<lim && (int)pool.size() < CORE_RECOMB_UNION_KEEP + 24; ++ki) push_pool(inst.knn_at(v, ki));
    }
    for(int v : b){
        if(in_a[v]) continue;
        for(int ki=0; ki<lim && (int)pool.size() < CORE_RECOMB_UNION_KEEP + 48; ++ki) push_pool(inst.knn_at(v, ki));
    }

    if((int)cyc.size() > 3) cyc = order_cycle_from_set(inst, cyc, rng);
    while((int)cyc.size() < std::min(3, k) && !pool.empty()){
        int node = pool.back();
        pool.pop_back();
        cyc.push_back(node);
    }
    std::vector<uint8_t> none_banned(N, 0);
    if((int)cyc.size() < k){
        regret_repair_cycle(cyc, inst, k, pool, none_banned);
    }
    if((int)cyc.size() != k) cyc = resize_seed_cycle(inst, cyc, k, rng, mode % 3);
    if((int)cyc.size() > 3) cyc = order_cycle_from_set(inst, cyc, rng);
    return cyc;
}

void polish_fixed_subset_tour(Tour& cand,const Instance& inst,int strength){
    cand.ensure_edges(inst);
    for(int rep=0; rep<std::max(1, strength); ++rep){
        two_opt_nn(cand, inst, 340 + 100*rep, -1);
        or_opt_1(cand, inst, 6 + rep, TSP_OROPT_NN_CAND + 4 + 2*rep, TSP_OROPT_LOCAL_WINDOW + 1);
        two_opt_nn(cand, inst, 180 + 60*rep, -1);
    }
    if(strength >= 3 && cand.k <= EXACT_SMALL_TSP_K) exact_small_subset_tour(cand, inst);
}
