#include "tsp_solver.hpp"

void two_opt_nn(Tour& tour,const Instance& inst,int mp,int cand_cap){
    int n = tour.k;
    tour.ensure_edges(inst);
    if(n < 4) return;
    int kk = (cand_cap > 0) ? std::min(inst.knn_k, cand_cap) : inst.knn_k;
    for(int p=0;p<mp;p++){
        bool imp = false;
        for(int idx=0; idx<n; idx++){
            int a = tour.nodes[idx];
            double old_ab = tour.edge_len[idx];
            for(int ki=0; ki<kk; ki++){
                double ac = inst.knn_d_at(a, ki);
                if(ac >= old_ab) break;
                int c = inst.knn_at(a, ki);
                int j = tour.pos[c];
                if(j < 0) continue;
                int i2 = idx;
                if(i2 == j || std::abs(i2 - j) <= 1) continue;
                if(i2 == 0 && j == n - 1) continue;
                if(j == 0 && i2 == n - 1) continue;
                int ii = std::min(i2, j), jj = std::max(i2, j);
                if(ii == 0 && jj == n - 1) continue;
                int b = tour.nodes[ii+1];
                int d = tour.nodes[(jj+1)%n];
                double delta = ac + inst.dist(b, d) - tour.edge_len[ii] - tour.edge_len[jj];
                if(delta < -IMPROVEMENT_EPS){
                    tour.apply_two_opt(ii, jj, inst, delta);
                    imp = true;
                    break;
                }
            }
            if(imp) break;
        }
        if(!imp) break;
    }
}


// KNN-bounded 2-opt with don't-look bits. Nodes whose outgoing edge wasn't
// improved are marked inactive; on acceptance the reversed segment and
// reverse-KNN parents of the endpoints are woken to cheaply revisit them.
void two_opt_nn_tsp(Tour& tour,const Instance& inst,int mp,int cand_cap){
    int n = tour.k;
    tour.ensure_edges(inst);
    if(n < 4) return;
    const int kk = (cand_cap > 0) ? std::min(inst.knn_k, cand_cap) : inst.knn_k;
    if(kk <= 0) return;
    const int scan_limit = (int)std::min((int64_t)mp * n, (int64_t)std::numeric_limits<int>::max());

    // Don't-look bits on node ids: once a node's current outgoing edge fails to
    // improve, skip it until a later accepted 2-opt changes a nearby candidate
    // relationship. On acceptance we wake the changed segment and reverse-KNN
    // parents of the move endpoints to cheaply revisit likely affected nodes.
    static thread_local std::vector<uint8_t> dlb;
    dlb.assign(tour.N, (uint8_t)0);

    int scans = 0;
    bool any_improved = true;
    while(any_improved && scans < scan_limit){
        any_improved = false;
        for(int idx=0; idx<n && scans < scan_limit; ++idx){
            int a = tour.nodes[idx];
            if(dlb[a]) continue;
            ++scans;

            double old_ab = tour.edge_len[idx];
            bool improved = false;
            for(int ki=0; ki<kk; ++ki){
                double ac = inst.knn_d_at(a, ki);
                if(ac >= old_ab) break;
                int c = inst.knn_at(a, ki);
                int j = tour.pos[c];
                if(j < 0) continue;
                if(idx == j || std::abs(idx - j) <= 1) continue;
                if((idx == 0 && j == n - 1) || (j == 0 && idx == n - 1)) continue;

                int ii = std::min(idx, j), jj = std::max(idx, j);
                if(ii == 0 && jj == n - 1) continue;

                int b = tour.nodes[ii + 1];
                int d = tour.nodes[(jj + 1) % n];
                double delta = ac + inst.dist(b, d) - tour.edge_len[ii] - tour.edge_len[jj];
                if(delta < -IMPROVEMENT_EPS){
                    tour.apply_two_opt(ii, jj, inst, delta);

                    int leftb = tour.nodes[(ii == 0) ? (n - 1) : (ii - 1)];
                    int rightb = tour.nodes[(jj + 1 == n) ? 0 : (jj + 1)];
                    int a2 = tour.nodes[ii];
                    int b2 = tour.nodes[ii + 1];
                    int c2 = tour.nodes[jj];
                    dlb[a2] = 0;
                    dlb[b2] = 0;
                    dlb[c2] = 0;
                    dlb[rightb] = 0;
                    dlb[leftb] = 0;
                    auto wake_rknn = [&](int v){
                        for(int p=inst.rknn_begin[v]; p<inst.rknn_begin[v + 1]; ++p){
                            int u = inst.rknn_nodes[p];
                            if(tour.in_set[u]) dlb[u] = 0;
                        }
                    };
                    wake_rknn(a2);
                    wake_rknn(b2);
                    wake_rknn(c2);
                    wake_rknn(rightb);
                    wake_rknn(leftb);

                    improved = true;
                    any_improved = true;
                    break;
                }
            }
            if(!improved) dlb[a] = 1;
        }
    }
}



void or_opt_1(Tour& tour,const Instance& inst,int mp,int nn_cap,int local_window){
    int n = tour.k;
    tour.ensure_edges(inst);
    if(n < 5) return;

    const int kk = (nn_cap > 0) ? std::min(inst.knn_k, nn_cap) : inst.knn_k;
    std::vector<int> pred_buf(2 * local_window + 2 * kk + 8);
    std::vector<int> seen(tour.N, 0);
    int token = 1;

    for(int p=0; p<mp; ++p){
        double best_delta = -IMPROVEMENT_EPS;
        int br = -1, bi = -1;

        for(int idx=0; idx<n; ++idx){
            int prev = (idx == 0) ? (n - 1) : (idx - 1);
            int next = (idx + 1 == n) ? 0 : (idx + 1);
            int nd = tour.nodes[idx];

            double remove_gain = tour.edge_len[prev] + tour.edge_len[idx] - inst.dist(tour.nodes[prev], tour.nodes[next]);

            bump_token(seen, token);
            int pred_cnt = 0;
            auto push_pred = [&](int j){
                if(j < 0 || j >= n) return;
                if(j == idx || j == prev) return;
                if(seen[j] == token) return;
                if(pred_cnt >= (int)pred_buf.size()) return;
                seen[j] = token;
                pred_buf[pred_cnt++] = j;
            };

            for(int off=-local_window; off<=local_window; ++off){
                int j = idx + off;
                while(j < 0) j += n;
                while(j >= n) j -= n;
                push_pred(j);
            }
            for(int ki=0; ki<kk; ++ki){
                int v = inst.knn_at(nd, ki);
                int j = tour.pos[v];
                if(j < 0) continue;
                push_pred(j);
                push_pred((j == 0) ? (n - 1) : (j - 1));
            }

            SmallNodeDistCache nd_dc(inst, nd);
            for(int ki=0; ki<kk; ++ki){
                int v = inst.knn_at(nd, ki);
                nd_dc.seed(v, inst.knn_d_at(nd, ki));
            }

            for(int t=0; t<pred_cnt; ++t){
                int j = pred_buf[t];
                int succ = (j + 1 == n) ? 0 : (j + 1);
                double insert_cost = nd_dc.get(tour.nodes[j]) + nd_dc.get(tour.nodes[succ]) - tour.edge_len[j];
                double delta = insert_cost - remove_gain;
                if(delta < best_delta){
                    best_delta = delta;
                    br = idx;
                    bi = j;
                }
            }
        }

        if(br < 0) break;
        int post_pred = post_idx_from_cur(bi, br);
        tour.apply_move_post_rem(br, post_pred, inst, best_delta);
    }
}


void nn_tour(const int* sub,int k,const Instance& inst,int si,std::vector<int>& res){
    res.resize(k);
    std::vector<uint8_t> used(k, 0);
    res[0] = sub[si];
    used[si] = 1;
    for(int step=1; step<k; step++){
        int last = res[step-1], bn = -1;
        double bd2 = std::numeric_limits<double>::infinity();
        for(int j=0; j<k; j++) if(!used[j]){
            double d2 = inst.dist2(last, sub[j]);
            if(d2 < bd2){
                bd2 = d2;
                bn = j;
            }
        }
        res[step] = sub[bn];
        used[bn] = 1;
    }
}


void farthest_ins(const int* sub,int k,const Instance& inst,std::vector<int>& res){
    if(k <= 3){
        res.assign(sub, sub + k);
        return;
    }

    double bd2 = -1.0;
    int ai = 0, bi = 1;
    for(int i=0;i<k;i++) for(int j=i+1;j<k;j++){
        double d2 = inst.dist2(sub[i], sub[j]);
        if(d2 > bd2){
            bd2 = d2;
            ai = i;
            bi = j;
        }
    }

    std::vector<int> nxt(k, -1), prv(k, -1);
    std::vector<double> edge_out(k, 0.0);
    std::vector<double> md2(k, std::numeric_limits<double>::infinity());
    std::vector<uint8_t> in_t(k, 0);

    nxt[ai] = bi; prv[bi] = ai;
    nxt[bi] = ai; prv[ai] = bi;
    double dab = inst.dist(sub[ai], sub[bi]);
    edge_out[ai] = dab;
    edge_out[bi] = dab;
    in_t[ai] = in_t[bi] = 1;

    for(int i=0;i<k;i++){
        if(in_t[i]){ md2[i] = 0.0; continue; }
        md2[i] = std::min(inst.dist2(sub[i], sub[ai]), inst.dist2(sub[i], sub[bi]));
    }

    int head = ai;
    for(int iter=0; iter<k-2; ++iter){
        int bf = -1;
        double bfd2 = -1.0;
        for(int i=0;i<k;i++){
            if(in_t[i]) continue;
            if(md2[i] > bfd2){
                bfd2 = md2[i];
                bf = i;
            }
        }

        int node = sub[bf];
        SmallNodeDistCache node_dc(inst, node);
        int best_pred = head;
        double best_cost = std::numeric_limits<double>::infinity();

        int u = head;
        do{
            int v = nxt[u];
            double c = node_dc.get(sub[u]) + node_dc.get(sub[v]) - edge_out[u];
            if(c < best_cost){
                best_cost = c;
                best_pred = u;
            }
            u = nxt[u];
        } while(u != head);

        int best_succ = nxt[best_pred];
        nxt[best_pred] = bf;
        prv[bf] = best_pred;
        nxt[bf] = best_succ;
        prv[best_succ] = bf;
        edge_out[best_pred] = node_dc.get(sub[best_pred]);
        edge_out[bf] = node_dc.get(sub[best_succ]);
        in_t[bf] = 1;

        for(int i=0;i<k;i++){
            if(in_t[i]) continue;
            double d2 = inst.dist2(sub[i], node);
            if(d2 < md2[i]) md2[i] = d2;
        }
    }

    res.resize(k);
    int u = head;
    for(int i=0;i<k;i++){
        res[i] = sub[u];
        u = nxt[u];
    }
}



// Multi-restart ILS for the full Euclidean TSP. Each restart seeds from NN or
// farthest-insertion, then iterates 3-cut segment-shuffle perturbation followed
// by KNN-bounded 2-opt + or-opt-1 local search. Top restarts are re-polished
// with widening candidate sets and collected into an elite pool.
void solve_tsp(const Instance& inst,Rng& rng,Tour& best,const OracleContext& oracle,
               int restarts,int ils,int patience,
               std::vector<std::vector<int>>* elite_out){
    int N = inst.N;
    std::vector<int> all(N);
    std::iota(all.begin(), all.end(), 0);

    struct RestartCand {
        double len = std::numeric_limits<double>::infinity();
        std::vector<int> nodes;
    };
    std::vector<RestartCand> top;
    auto keep_restart = [&](double len,const std::vector<int>& nodes){
        RestartCand c;
        c.len = len;
        c.nodes = nodes;
        auto it = std::lower_bound(top.begin(), top.end(), len,
            [](const RestartCand& a,double v){ return a.len < v; });
        top.insert(it, std::move(c));
        if((int)top.size() > TSP_TOP_RESTART_CAND) top.pop_back();
    };

    std::vector<int> init, nt(N);

    for(int r=0; r<restarts; ++r){
        if(r == 0) farthest_ins(all.data(), N, inst, init);
        else {
            int si = rng.randint(N);
            nn_tour(all.data(), N, inst, si, init);
        }

        Tour t;
        t.init(N);
        t.set_tour_only(init.data(), N);
        two_opt_nn(t, inst, 1200, TSP_INIT_TWOOPT_K);
        or_opt_1(t, inst, 8, TSP_OROPT_NN_CAND, TSP_OROPT_LOCAL_WINDOW);
        two_opt_nn(t, inst, 600, TSP_INIT_TWOOPT_K);

        double len = t.length;
        std::vector<int> best_restart_nodes = t.nodes;
        double best_restart_len = len;
        int no_improve = 0;

        for(int it=0; it<ils; ++it){
            // 3-cut segment shuffle used as the ILS perturbation.
            int c[3];
            c[0] = rng.randint(N);
            c[1] = rng.randint(N);
            c[2] = rng.randint(N);
            std::sort(c, c+3);
            if(c[0] == c[1] || c[1] == c[2]) continue;

            int p = 0;
            for(int i=0; i<=c[0]; ++i) nt[p++] = t.nodes[i];
            for(int i=c[1]+1; i<=c[2]; ++i) nt[p++] = t.nodes[i];
            for(int i=c[0]+1; i<=c[1]; ++i) nt[p++] = t.nodes[i];
            for(int i=c[2]+1; i<N; ++i) nt[p++] = t.nodes[i];

            t.set_tour_only(nt.data(), N);
            two_opt_nn(t, inst, 500, TSP_ILS_TWOOPT_K);
            or_opt_1(t, inst, 3, TSP_OROPT_NN_CAND, TSP_OROPT_LOCAL_WINDOW);
            two_opt_nn(t, inst, 250, TSP_ILS_TWOOPT_K);

            if(t.length < len - IMPROVEMENT_EPS){
                len = t.length;
                no_improve = 0;
                if(len < best_restart_len - IMPROVEMENT_EPS){
                    best_restart_len = len;
                    best_restart_nodes = t.nodes;
                }
            } else {
                t.set_tour_only(best_restart_nodes.data(), N);
                t.length = best_restart_len;
                len = best_restart_len;
                ++no_improve;
                if(no_improve > patience) break;
            }
        }

        keep_restart(best_restart_len, best_restart_nodes);
    }

    ElitePool elite(TSP_TOP_RESTART_CAND, EliteHashMode::CYCLE);
    double best_len = std::numeric_limits<double>::infinity();
    std::vector<int> best_nodes;
    for(const auto& cand : top){
        Tour t;
        t.init(N);
        t.set_tour_only(cand.nodes.data(), N);
        t.recompute_length(inst);

        for(int rep=0; rep<4; ++rep){
            double before = t.length;
            or_opt_1(t, inst, 10 + 2*rep, TSP_OROPT_NN_CAND + 2*rep, TSP_OROPT_LOCAL_WINDOW + (rep > 0));
            two_opt_nn(t, inst, 1200 - 160*rep, TSP_FINAL_TWOOPT_K);
            if(before - t.length < IMPROVEMENT_EPS) break;
        }

        elite.try_add(t.nodes, t.length);
        if(t.length < best_len){
            best_len = t.length;
            best_nodes = t.nodes;
        }
    }

    if(oracle.cfg.inline_feedback && oracle.cfg.use_for_tsp){
        auto elite_nodes = elite.export_nodes();
        int exn = std::min((int)elite_nodes.size(), std::max(0, oracle.cfg.tsp_top));
        for(int ei=0; ei<exn; ++ei){
            Tour t;
            t.init(N);
            t.set_tour(elite_nodes[ei].data(), N, inst);
            external_oracle_polish_tour(t, inst, oracle, true, 2);
            elite.try_add(t.nodes, t.length);
            if(t.length < best_len - IMPROVEMENT_EPS){
                best_len = t.length;
                best_nodes = t.nodes;
            }
        }
    }

    best.init(N);
    best.set_tour(best_nodes.data(), N, inst);
    if(elite_out) *elite_out = elite.export_nodes();
}


