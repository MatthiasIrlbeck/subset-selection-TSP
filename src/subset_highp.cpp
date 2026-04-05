#include "subset_solver_internal.hpp"

struct HighpDeleteSeed {
    std::vector<int> nodes;
    std::vector<int> ref;
};

static std::vector<int> greedy_delete_cycle(const Instance& inst,const std::vector<int>& seed,int k,Rng& rng,int mode){
    std::vector<int> cur = seed;
    while((int)cur.size() > k){
        int m = (int)cur.size();
        std::vector<double> score(m, 0.0);
        std::vector<int> ord(m);
        for(int i=0; i<m; ++i){
            int p = cur[(i - 1 + m) % m];
            int u = cur[i];
            int n = cur[(i + 1) % m];
            double s = inst.dist(p, u) + inst.dist(u, n) - inst.dist(p, n);
            if(mode == 2){
                int ki = std::min(std::max(0, inst.knn_k - 1), 10);
                if(ki >= 0 && inst.knn_k > 0) s += 0.20 * inst.knn_d_at(u, ki);
            }
            score[i] = s;
            ord[i] = i;
        }
        std::sort(ord.begin(), ord.end(), [&](int a,int b){
            if(score[a] != score[b]) return score[a] > score[b];
            return cur[a] < cur[b];
        });
        int wi = ord[0];
        if(mode == 1){
            int top = std::min(m, 8);
            double tot = 0.0;
            for(int t=0; t<top; ++t) tot += (double)(top - t);
            double r = rng.uniform() * tot;
            for(int t=0; t<top; ++t){
                r -= (double)(top - t);
                if(r <= 0.0){ wi = ord[t]; break; }
            }
        }
        cur.erase(cur.begin() + wi);
    }
    return cur;
}

static std::vector<int> segment_delete_cycle(const Instance& inst,const std::vector<int>& seed,int k,Rng& rng,int seg_variant){
    std::vector<int> cur = seed;
    int q = (int)cur.size() - k;
    if(q <= 0) return cur;
    int m = (int)cur.size();
    std::vector<double> score(m, 0.0);
    for(int i=0; i<m; ++i){
        int p = cur[(i - 1 + m) % m];
        int u = cur[i];
        int n = cur[(i + 1) % m];
        score[i] = inst.dist(p, u) + inst.dist(u, n) - inst.dist(p, n);
    }
    std::vector<int> ord(m);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a,int b){
        if(score[a] != score[b]) return score[a] > score[b];
        return cur[a] < cur[b];
    });
    int start = ord[std::min(seg_variant, std::max(0, std::min(2, m - 1)))];
    int seg = std::min(q, std::max(2, q / 2));
    std::vector<uint8_t> del(m, 0);
    for(int t=0; t<seg; ++t) del[(start + t) % m] = 1;
    std::vector<int> next;
    next.reserve(m - seg);
    for(int i=0; i<m; ++i) if(!del[i]) next.push_back(cur[i]);
    if((int)next.size() > k) next = greedy_delete_cycle(inst, next, k, rng, seg_variant % 3);
    return next;
}

static void build_ref_adj(const std::vector<int>& ref,int N,std::vector<int>& ref_prev,std::vector<int>& ref_next){
    ref_prev.assign(N, -1);
    ref_next.assign(N, -1);
    int n = (int)ref.size();
    for(int i=0; i<n; ++i){
        int v = ref[i];
        ref_prev[v] = ref[(i - 1 + n) % n];
        ref_next[v] = ref[(i + 1) % n];
    }
}

static void collect_highp_delete_add_pool(const Tour& tour,const Instance& inst,
                                          const std::vector<int>& ref_prev,const std::vector<int>& ref_next,
                                          int ri,std::vector<int>& pool,std::vector<int>& mark,int& mark_token){
    int nt = tour.k;
    int prev_idx = (ri == 0) ? (nt - 1) : (ri - 1);
    int next_idx = (ri + 1 == nt) ? 0 : (ri + 1);
    int rem = tour.nodes[ri], prv = tour.nodes[prev_idx], nxt = tour.nodes[next_idx];
    int tok = bump_token(mark, mark_token);
    pool.clear();
    auto try_add = [&](int v){
        if(v < 0 || v >= tour.N) return;
        if(tour.in_set[v]) return;
        if(mark[v] == tok) return;
        mark[v] = tok;
        pool.push_back(v);
    };
    auto walk_ref = [&](int seed,bool forward){
        int v = seed;
        for(int hop=0; hop<HIGHP_DELETE_REF_HOPS; ++hop){
            v = forward ? ref_next[v] : ref_prev[v];
            if(v < 0) break;
            try_add(v);
            if((int)pool.size() >= HIGHP_DELETE_REF_CAND) break;
        }
    };
    walk_ref(rem, true); walk_ref(rem, false);
    walk_ref(prv, true); walk_ref(prv, false);
    walk_ref(nxt, true); walk_ref(nxt, false);
    for(int seed : {prv, nxt, rem}){
        for(int ki=0; ki<inst.knn_k && (int)pool.size() < HIGHP_DELETE_REF_CAND; ++ki){
            int v = inst.knn_at(seed, ki);
            try_add(v);
        }
    }
}

static bool highp_delete_exchange_descent(Tour& tour,const Instance& inst,const std::vector<int>& ref,int passes=HIGHP_DELETE_PASSES){
    if(tour.k < 3 || ref.empty()) return false;
    std::vector<int> ref_prev, ref_next;
    build_ref_adj(ref, tour.N, ref_prev, ref_next);
    std::vector<int> mark(tour.N, 0), pred_buf, add_pool;
    pred_buf.reserve(SUBSET_MAX_EDGE_CAND);
    add_pool.reserve(HIGHP_DELETE_REF_CAND);
    int mark_token = 1;
    bool any = false;
    for(int pass=0; pass<passes; ++pass){
        double global_best = -IMPROVEMENT_EPS;
        int best_ri = -1, best_add = -1, best_post_pred = 0;
        for(int ri=0; ri<tour.k; ++ri){
            int nt = tour.k;
            int prev_idx = (ri == 0) ? (nt - 1) : (ri - 1);
            int next_idx = (ri + 1 == nt) ? 0 : (ri + 1);
            int prv = tour.nodes[prev_idx], nxt = tour.nodes[next_idx];
            double gap_len = inst.dist(prv, nxt);
            double rs = tour.edge_len[prev_idx] + tour.edge_len[ri] - gap_len;

            collect_highp_delete_add_pool(tour, inst, ref_prev, ref_next, ri, add_pool, mark, mark_token);
            if(add_pool.empty()) continue;
            for(int add : add_pool){
                collect_pr_pred_candidates(tour, inst, ri, add, pred_buf, nt <= PR_FULL_EDGE_SCAN_K);
                SmallNodeDistCache dc(inst, add);
                double best_ins = std::numeric_limits<double>::infinity();
                int best_post = (ri > 0) ? (ri - 1) : (nt - 2);
                for(int pred_idx : pred_buf){
                    int succ_idx = next_live_idx(pred_idx, ri, nt);
                    int a = tour.nodes[pred_idx];
                    int b = tour.nodes[succ_idx];
                    double base = (pred_idx == prev_idx) ? gap_len : tour.edge_len[pred_idx];
                    double c = dc.get(a) + dc.get(b) - base;
                    if(c < best_ins){ best_ins = c; best_post = post_idx_from_cur(pred_idx, ri); }
                }
                double delta = best_ins - rs;
                if(delta < global_best){
                    global_best = delta;
                    best_ri = ri;
                    best_add = add;
                    best_post_pred = best_post;
                }
            }
        }
        if(best_ri < 0) break;
        tour.apply_swap_post_rem(best_ri, best_post_pred, best_add, inst, global_best);
        two_opt_nn(tour, inst, 260, -1);
        or_opt_1(tour, inst, 4, TSP_OROPT_NN_CAND + 4, TSP_OROPT_LOCAL_WINDOW + 1);
        two_opt_nn(tour, inst, 120, -1);
        any = true;
    }
    return any;
}

static std::vector<HighpDeleteSeed> make_highp_delete_seed_pool(const Instance& inst,int k,const std::vector<std::vector<int>>& parent_elite,Rng& rng){
    std::vector<HighpDeleteSeed> out;
    std::vector<uint64_t> seen;
    auto push_seed = [&](std::vector<int> s,const std::vector<int>& ref){
        if((int)s.size() != k) return;
        uint64_t h = subset_set_hash_canon_copy(s);
        for(uint64_t x : seen) if(x == h) return;
        seen.push_back(h);
        HighpDeleteSeed hs;
        hs.nodes = std::move(s);
        hs.ref = ref;
        out.push_back(std::move(hs));
    };
    int use = std::min((int)parent_elite.size(), HIGHP_DELETE_PARENT_USE);
    for(int i=0; i<use && (int)out.size() < HIGHP_DELETE_SEED_BUDGET; ++i){
        const auto& ref = parent_elite[i];
        if((int)ref.size() < k) continue;
        push_seed(greedy_delete_cycle(inst, ref, k, rng, 0), ref);
        if((int)out.size() < HIGHP_DELETE_SEED_BUDGET) push_seed(greedy_delete_cycle(inst, ref, k, rng, 1), ref);
        if((int)out.size() < HIGHP_DELETE_SEED_BUDGET) push_seed(greedy_delete_cycle(inst, ref, k, rng, 2), ref);
        for(int sv=0; sv<HIGHP_DELETE_SEG_SEEDS && (int)out.size() < HIGHP_DELETE_SEED_BUDGET; ++sv){
            push_seed(segment_delete_cycle(inst, ref, k, rng, sv), ref);
        }
        if((int)out.size() < HIGHP_DELETE_SEED_BUDGET) push_seed(resize_seed_cycle(inst, ref, k, rng, i % 3), ref);
    }
    return out;
}

// Deletion-based high-p solver: starts from a full-TSP parent tour and
// iteratively removes the cheapest-to-excise nodes down to k, then refines
// via LNS ruin-recreate and path-relinking against the elite pool.
void solve_subset_highp_delete(const Instance& inst,int k,Rng& rng,Tour& best,
    int sa_iters,int restarts,const std::vector<std::vector<int>>* parent_elite,
    std::vector<std::vector<int>>* elite_out){
    int N = inst.N;
    int effort = std::max(1, sa_iters / 25000);
    int delete_passes = HIGHP_DELETE_PASSES + std::min(3, effort - 1);
    int delete_tail_passes = 2 + std::min(2, std::max(0, effort - 2));
    int rr_rounds = RR_ROUNDS_BASE + 1 + std::min(4, std::max(0, effort - 1) / 2);
    int final_polish_strength = 2 + std::min(1, std::max(0, effort - 3));
    ElitePool local_elite(SUBSET_LOCAL_ELITE_KEEP, EliteHashMode::SET);
    double bg = std::numeric_limits<double>::infinity();
    std::vector<int> bgn;
    std::vector<HighpDeleteSeed> seeds;
    if(parent_elite && !parent_elite->empty()) seeds = make_highp_delete_seed_pool(inst, k, *parent_elite, rng);
    if(seeds.empty() && parent_elite){
        for(const auto& ref : *parent_elite){
            if((int)ref.size() >= k){
                HighpDeleteSeed hs;
                hs.nodes = resize_seed_cycle(inst, ref, k, rng, 0);
                hs.ref = ref;
                seeds.push_back(std::move(hs));
                if((int)seeds.size() >= 4) break;
            }
        }
    }
    int total = std::max(restarts, (int)seeds.size());
    total = std::max(total, std::min(HIGHP_DELETE_SEED_BUDGET, restarts + std::max(0, effort - 1)));
    for(int it=0; it<total; ++it){
        std::vector<int> init, ref;
        if(it < (int)seeds.size()){
            init = seeds[it].nodes;
            ref = seeds[it].ref;
        } else if(parent_elite && !parent_elite->empty()){
            ref = (*parent_elite)[rng.randint((int)parent_elite->size())];
            init = resize_seed_cycle(inst, ref, k, rng, it % 3);
        } else {
            std::vector<int> perm(N);
            std::iota(perm.begin(), perm.end(), 0);
            rng.partial_shuffle(perm.data(), N, k);
            init.assign(perm.begin(), perm.begin() + k);
            ref = init;
        }
        Tour tour;
        tour.init(N);
        tour.set_tour_only(init.data(), k);
        polish_fixed_subset_tour(tour, inst, 1);
        highp_delete_exchange_descent(tour, inst, ref, delete_passes);
        subset_pair_exchange_descent(tour, inst, rng, 1 + std::min(1, std::max(0, effort - 2)));
        subset_ruin_recreate_lns(tour, inst, rng, rr_rounds);
        highp_delete_exchange_descent(tour, inst, ref, delete_tail_passes);
        polish_subset_candidate(tour, inst, final_polish_strength);
        local_elite.try_add(tour.nodes, tour.length);
        if(tour.length < bg){ bg = tour.length; bgn = tour.nodes; }
    }
    auto elite_nodes = local_elite.export_nodes();
    int rep = std::min((int)elite_nodes.size(), 2);
    for(int i=0; i<rep; ++i){
        Tour t;
        t.init(N);
        t.set_tour_only(elite_nodes[i].data(), k);
        polish_subset_candidate(t, inst, final_polish_strength);
        local_elite.try_add(t.nodes, t.length);
        if(t.length < bg){ bg = t.length; bgn = t.nodes; }
    }
    best.init(N);
    best.set_tour_only(bgn.data(), k);
    best.recompute_length(inst);
    if(elite_out) *elite_out = local_elite.export_nodes();
}
