#include "subset_solver_internal.hpp"

void collect_pr_pred_candidates(const Tour& tour,const Instance& inst,int ri,int add,
                                              std::vector<int>& pred_buf,bool force_full){
    int nt = tour.k;
    int prev_idx = (ri == 0) ? (nt - 1) : (ri - 1);
    pred_buf.clear();
    pred_buf.reserve(std::min(nt, 96));

    static thread_local std::vector<int> pos_seen;
    static thread_local int pos_tok = 1;
    if((int)pos_seen.size() < nt) pos_seen.assign(nt, 0);
    int cur_tok = bump_token(pos_seen, pos_tok);

    auto push_pred = [&](int pred_idx){
        if(pred_idx < 0 || pred_idx >= nt || pred_idx == ri) return;
        if(pos_seen[pred_idx] == cur_tok) return;
        pos_seen[pred_idx] = cur_tok;
        int succ_idx = next_live_idx(pred_idx, ri, nt);
        if(succ_idx == pred_idx || succ_idx == ri) return;
        pred_buf.push_back(pred_idx);
    };

    if(force_full || nt <= PR_FULL_EDGE_SCAN_K){
        for(int pred_idx=0; pred_idx<nt; ++pred_idx) push_pred(pred_idx);
        return;
    }

    push_pred(prev_idx);
    for(int off=-PR_LOCAL_EDGE_WINDOW; off<=PR_LOCAL_EDGE_WINDOW+2; ++off){
        int p = prev_idx + off;
        while(p < 0) p += nt;
        while(p >= nt) p -= nt;
        push_pred(p);
    }
    int near_add = 0;
    for(int ki=0; ki<inst.knn_k && near_add<PR_EDGE_ADD_NN; ++ki){
        int v = inst.knn_at(add, ki);
        int jp = tour.pos[v];
        if(jp < 0 || jp == ri) continue;
        push_pred(jp);
        push_pred(prev_live_idx(jp, ri, nt));
        ++near_add;
    }
}

bool subset_path_relink_oneway(const Instance& inst,const std::vector<int>& src_nodes,
                               const std::vector<int>& target_nodes,Rng& rng,
                               std::vector<int>& best_nodes_out,double& best_len_out){
    (void)rng;
    int N = inst.N;
    int k = (int)src_nodes.size();
    if(k < 3 || (int)target_nodes.size() != k) return false;

    Tour cur;
    cur.init(N);
    cur.set_tour_only(src_nodes.data(), k);
    polish_fixed_subset_tour(cur, inst, 1);

    std::vector<uint8_t> target_in(N, 0);
    for(int v : target_nodes) if(v >= 0 && v < N) target_in[v] = 1;

    std::vector<int> best_nodes = cur.nodes;
    double best_len = cur.length;
    bool moved = false;

    std::vector<int> rem_pos, rem_ord, add_nodes, pred_buf;
    std::vector<double> rem_score, add_dp, add_dn, add_gap;

    for(int step=0; step<k; ++step){
        rem_pos.clear();
        rem_score.clear();
        for(int i=0; i<cur.k; ++i){
            if(target_in[cur.nodes[i]]) continue;
            int p = (i == 0) ? (cur.k - 1) : (i - 1);
            int n = (i + 1 == cur.k) ? 0 : (i + 1);
            double score = cur.edge_len[p] + cur.edge_len[i] - inst.dist(cur.nodes[p], cur.nodes[n]);
            rem_pos.push_back(i);
            rem_score.push_back(score);
        }
        if(rem_pos.empty()) break;
        rem_ord.resize(rem_pos.size());
        std::iota(rem_ord.begin(), rem_ord.end(), 0);
        int top_remove = std::min((int)rem_ord.size(), PR_TOP_REMOVE);
        auto rem_cmp = [&](int lhs,int rhs){
            if(rem_score[lhs] != rem_score[rhs]) return rem_score[lhs] > rem_score[rhs];
            return cur.nodes[rem_pos[lhs]] < cur.nodes[rem_pos[rhs]];
        };
        if(top_remove < (int)rem_ord.size()) std::partial_sort(rem_ord.begin(), rem_ord.begin() + top_remove, rem_ord.end(), rem_cmp);
        else std::sort(rem_ord.begin(), rem_ord.end(), rem_cmp);

        add_nodes.clear();
        for(int v : target_nodes) if(!cur.in_set[v]) add_nodes.push_back(v);
        if(add_nodes.empty()) break;

        int best_ri = -1;
        int best_add = -1;
        int best_post_pred = 0;
        double best_delta = std::numeric_limits<double>::infinity();

        for(int rr=0; rr<top_remove; ++rr){
            int ri = rem_pos[rem_ord[rr]];
            int nt = cur.k;
            int prev_idx = (ri == 0) ? (nt - 1) : (ri - 1);
            int next_idx = (ri + 1 == nt) ? 0 : (ri + 1);
            int prv = cur.nodes[prev_idx], nxt = cur.nodes[next_idx];
            double gap_len = inst.dist(prv, nxt);
            double rs = cur.edge_len[prev_idx] + cur.edge_len[ri] - gap_len;

            int ac = (int)add_nodes.size();
            add_dp.resize(ac);
            add_dn.resize(ac);
            add_gap.resize(ac);
            dist_many_from(inst, prv, add_nodes.data(), ac, add_dp.data());
            dist_many_from(inst, nxt, add_nodes.data(), ac, add_dn.data());
            std::vector<int> add_ord(ac);
            for(int i=0; i<ac; ++i){
                add_gap[i] = add_dp[i] + add_dn[i] - gap_len;
                add_ord[i] = i;
            }
            int top_add = std::min(ac, PR_TOP_ADD);
            auto add_cmp = [&](int lhs,int rhs){
                if(add_gap[lhs] != add_gap[rhs]) return add_gap[lhs] < add_gap[rhs];
                return add_nodes[lhs] < add_nodes[rhs];
            };
            if(top_add < ac) std::partial_sort(add_ord.begin(), add_ord.begin() + top_add, add_ord.end(), add_cmp);
            else std::sort(add_ord.begin(), add_ord.end(), add_cmp);

            for(int aa=0; aa<top_add; ++aa){
                int ai = add_ord[aa];
                int add = add_nodes[ai];
                collect_pr_pred_candidates(cur, inst, ri, add, pred_buf, false);
                SmallNodeDistCache dc(inst, add);
                double local_best = std::numeric_limits<double>::infinity();
                int local_post_pred = (ri > 0) ? (ri - 1) : (nt - 2);
                for(int pred_idx : pred_buf){
                    int succ_idx = next_live_idx(pred_idx, ri, nt);
                    int a = cur.nodes[pred_idx];
                    int b = cur.nodes[succ_idx];
                    double base = (pred_idx == prev_idx) ? gap_len : cur.edge_len[pred_idx];
                    double delta = dc.get(a) + dc.get(b) - base - rs;
                    if(delta < local_best){
                        local_best = delta;
                        local_post_pred = post_idx_from_cur(pred_idx, ri);
                    }
                }
                if(local_best < best_delta){
                    best_delta = local_best;
                    best_ri = ri;
                    best_add = add;
                    best_post_pred = local_post_pred;
                }
            }
        }

        if(best_ri < 0 || best_add < 0 || !std::isfinite(best_delta)) break;
        cur.apply_swap_post_rem(best_ri, best_post_pred, best_add, inst, best_delta);
        moved = true;
        if(best_delta < -IMPROVEMENT_EPS || ((step + 1) % PR_POLISH_EVERY) == 0) polish_fixed_subset_tour(cur, inst, 1);
        if(cur.length < best_len - IMPROVEMENT_EPS){
            best_len = cur.length;
            best_nodes = cur.nodes;
        }
    }

    if(!moved) return false;
    Tour fin;
    fin.init(N);
    fin.set_tour_only(best_nodes.data(), k);
    polish_fixed_subset_tour(fin, inst, 1);
    best_nodes_out = fin.nodes;
    best_len_out = fin.length;
    return true;
}

// Bidirectional path-relinking between two subset tours: runs one-way relinking
// a→b and b→a, keeps whichever intermediate solution has the shorter cycle.
bool subset_path_relink_bidirectional(const Instance& inst,const std::vector<int>& a,
                                      const std::vector<int>& b,Rng& rng,
                                      std::vector<int>& best_nodes_out,double& best_len_out){
    if(a.empty() || b.empty() || a.size() != b.size()) return false;
    std::vector<int> n1, n2;
    double l1 = std::numeric_limits<double>::infinity();
    double l2 = std::numeric_limits<double>::infinity();
    bool ok1 = subset_path_relink_oneway(inst, a, b, rng, n1, l1);
    bool ok2 = subset_path_relink_oneway(inst, b, a, rng, n2, l2);
    if(ok1 && (!ok2 || l1 <= l2)){
        best_nodes_out = std::move(n1);
        best_len_out = l1;
        return true;
    }
    if(ok2){
        best_nodes_out = std::move(n2);
        best_len_out = l2;
        return true;
    }
    return false;
}

std::vector<std::vector<int>> make_warm_pool_from_elite(const Instance& inst,const std::vector<std::vector<int>>& prev_elite,int k,Rng& rng,
                                                     std::vector<std::vector<int>>* guide_out){
    std::vector<std::vector<int>> out;
    std::vector<std::vector<int>> guides;
    std::vector<uint64_t> seen_out, seen_guide;
    auto push_unique_hash = [&](std::vector<std::vector<int>>& dst,std::vector<uint64_t>& seen,std::vector<int> s){
        if(s.empty()) return false;
        uint64_t h = subset_set_hash_canon_copy(s);
        for(uint64_t x : seen) if(x == h) return false;
        seen.push_back(h);
        dst.push_back(std::move(s));
        return true;
    };
    auto push_seed = [&](std::vector<int> s){
        if((int)out.size() >= CROSSP_WARM_LIMIT) return;
        push_unique_hash(out, seen_out, std::move(s));
    };

    for(size_t ei=0; ei<prev_elite.size() && (int)guides.size() < CROSSP_ELITE_KEEP; ++ei){
        std::vector<int> g = resize_seed_cycle(inst, prev_elite[ei], k, rng, 0);
        if(push_unique_hash(guides, seen_guide, g)) push_seed(g);
        if((int)out.size() >= CROSSP_WARM_LIMIT) break;
        push_seed(resize_seed_cycle(inst, prev_elite[ei], k, rng, 1));
        if((int)ei < 2 && (int)out.size() < CROSSP_WARM_LIMIT) push_seed(resize_seed_cycle(inst, prev_elite[ei], k, rng, 2));
    }

    int pair_budget = 0;
    for(size_t i=0; i<guides.size() && (int)out.size() < CROSSP_WARM_LIMIT && pair_budget < CROSSP_RECOMB_PAIR_LIMIT; ++i){
        for(size_t j=i+1; j<guides.size() && (int)out.size() < CROSSP_WARM_LIMIT && pair_budget < CROSSP_RECOMB_PAIR_LIMIT; ++j){
            push_seed(recombine_parent_cycles(inst, guides[i], guides[j], k, rng, 0));
            if((int)out.size() < CROSSP_WARM_LIMIT) push_seed(recombine_parent_cycles(inst, guides[i], guides[j], k, rng, 2));
            if((int)out.size() < CROSSP_WARM_LIMIT) push_seed(recombine_core_union_cycles(inst, guides[i], guides[j], k, rng, 0));
            if((int)out.size() < CROSSP_WARM_LIMIT) push_seed(recombine_core_union_cycles(inst, guides[i], guides[j], k, rng, 1));
            std::vector<int> rel_nodes;
            double rel_len = std::numeric_limits<double>::infinity();
            if((int)out.size() < CROSSP_WARM_LIMIT && subset_path_relink_bidirectional(inst, guides[i], guides[j], rng, rel_nodes, rel_len)){
                push_seed(std::move(rel_nodes));
            }
            ++pair_budget;
        }
    }

    if(guide_out) *guide_out = guides;
    return out;
}
